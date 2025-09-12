import os
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# -----------------------------
# Dataset
# -----------------------------
class PairedDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.source_files = sorted(glob(os.path.join(source_dir, "*.png")))
        self.target_files = sorted(glob(os.path.join(target_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        src = Image.open(self.source_files[idx]).convert("L")  # grayscale
        tgt = Image.open(self.target_files[idx]).convert("L")

        if self.transform:
            src = self.transform(src)
            tgt = self.transform(tgt)

        return src, tgt


# -----------------------------
# U-Net Model
# -----------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.enc1 = UNetBlock(in_ch, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = UNetBlock(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = UNetBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return torch.sigmoid(out)  # grayscale MRI


# -----------------------------
# Loss: L1 + SSIM
# -----------------------------
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        # Simplified SSIM for training
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        sigma1 = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1 ** 2
        sigma2 = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2 ** 2
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
        return 1 - ssim_map.mean()


# -----------------------------
# Training Loop
# -----------------------------
def train_pipeline(source_dir, target_dir, save_dir, epochs=100, batch_size=8, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pix2pix model on {device}")

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    dataset = PairedDataset(source_dir, target_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()
    ssim_loss = SSIMLoss()

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)

            pred = model(src)
            loss = l1_loss(pred, tgt) + 0.5 * ssim_loss(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{save_dir}/unet_epoch{epoch}.pth")


# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    train_pipeline(
        source_dir="/mnt/x-bigdata/MRI-Proc/fraser/WORKING/FDQ2CINE/pix2pixT_Prod/A/train",
        target_dir="/mnt/x-bigdata/MRI-Proc/fraser/WORKING/FDQ2CINE/pix2pixT_Prod/B/train",
        save_dir="/mnt/x-bigdata/MRI-Proc/fraser/WORKING/FDQ2CINE/pix2pixT_Prod/checkpoints",
        epochs=100,
        batch_size=4,
        lr=1e-4
    )
