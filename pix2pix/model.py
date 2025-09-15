import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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
# Validation Functions
# -----------------------------
def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
    mse = mean_squared_error(img1.flatten(), img2.flatten())
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (SSIM)"""
    # Convert to numpy arrays if they're tensors
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure images are in the correct format
    if img1.ndim == 3:
        img1 = img1.squeeze()
    if img2.ndim == 3:
        img2 = img2.squeeze()
    
    # Ensure images are in the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")
    
    # Calculate SSIM using the same method as in SSIMLoss
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    
    return ssim

def validate_model(model, val_loader, device, save_visualizations=True, save_dir=None):
    """
    Validate the model on validation dataset
    
    Args:
        model: Trained UNet model
        val_loader: DataLoader for validation data
        device: Device to run validation on
        save_visualizations: Whether to save visualization images
        save_dir: Directory to save visualizations (if save_visualizations=True)
    
    Returns:
        dict: Dictionary containing validation metrics
    """
    model.eval()
    total_l1_loss = 0
    total_ssim_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_samples = 0
    
    l1_loss = nn.L1Loss()
    ssim_loss = SSIMLoss()
    
    if save_visualizations and save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(val_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # Forward pass
            pred = model(src)
            
            # Calculate losses
            l1 = l1_loss(pred, tgt)
            ssim_l = ssim_loss(pred, tgt)
            
            # Calculate metrics
            for i in range(pred.shape[0]):
                pred_np = pred[i].cpu().numpy()
                tgt_np = tgt[i].cpu().numpy()
                
                psnr = calculate_psnr(pred_np, tgt_np)
                ssim = calculate_ssim(pred[i], tgt[i])
                
                total_psnr += psnr
                total_ssim += ssim
                num_samples += 1
            
            total_l1_loss += l1.item()
            total_ssim_loss += ssim_l.item()
            
            # Save visualizations for first few batches
            if save_visualizations and save_dir and batch_idx < 30:
                for i in range(min(4, pred.shape[0])):  # Save up to 4 samples per batch
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Source image
                    axes[0].imshow(src[i].cpu().squeeze(), cmap='gray')
                    axes[0].set_title('Source')
                    axes[0].axis('off')
                    
                    # Target image
                    axes[1].imshow(tgt[i].cpu().squeeze(), cmap='gray')
                    axes[1].set_title('Target')
                    axes[1].axis('off')
                    
                    # Predicted image
                    axes[2].imshow(pred[i].cpu().squeeze(), cmap='gray')
                    axes[2].set_title('Predicted')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f"{save_dir}/validation_batch{batch_idx}_sample{i}.png", 
                              dpi=150, bbox_inches='tight')
                    plt.close()
    
    # Calculate average metrics
    avg_l1_loss = total_l1_loss / len(val_loader)
    avg_ssim_loss = total_ssim_loss / len(val_loader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    metrics = {
        'l1_loss': avg_l1_loss,
        'ssim_loss': avg_ssim_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }
    
    print(f"Validation Metrics:")
    print(f"  L1 Loss: {avg_l1_loss:.4f}")
    print(f"  SSIM Loss: {avg_ssim_loss:.4f}")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    
    return metrics

# -----------------------------
# Inference Functions
# -----------------------------
def load_model(model_path, device=None):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load model on (if None, auto-detect)
    
    Returns:
        model: Loaded UNet model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Model loaded from {model_path} on {device}")
    return model

def preprocess_image(image_path, size=(256, 256)):
    """
    Preprocess a single image for inference
    
    Args:
        image_path: Path to the input image
        size: Target size for the image (height, width)
    
    Returns:
        tensor: Preprocessed image tensor
    """
    try:
        # Open image
        image = Image.open(image_path)
        print(f"  Original image mode: {image.mode}, size: {image.size}")
        
        # Convert to grayscale if not already
        if image.mode != 'L':
            print(f"  Converting from {image.mode} to grayscale")
            # Handle different image modes more robustly
            if image.mode in ('RGBA', 'LA'):
                # Remove alpha channel first
                image = image.convert('RGB')
            elif image.mode == 'P':
                # Convert palette mode to RGB first
                image = image.convert('RGB')
            
            # Now convert to grayscale
            image = image.convert("L")
        
        print(f"  After conversion - mode: {image.mode}, size: {image.size}")
        
        # Apply transforms
        transform = T.Compose([
            T.Resize(size),
            T.ToTensor()
        ])
        
        image_tensor = transform(image)
        print(f"  Tensor shape after transform: {image_tensor.shape}")
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        print(f"  Final tensor shape: {image_tensor.shape}")
        
        return image_tensor
        
    except Exception as e:
        print(f"  ERROR in preprocess_image: {str(e)}")
        raise e

def postprocess_image(tensor, save_path=None):
    """
    Postprocess model output tensor to image
    
    Args:
        tensor: Model output tensor
        save_path: Optional path to save the image
    
    Returns:
        PIL.Image: Postprocessed image
    """
    # Convert tensor to numpy and remove batch dimension
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy and scale to 0-255
    image_np = tensor.detach().cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    # Convert to PIL Image
    image = Image.fromarray(image_np.squeeze(), mode='L')
    
    if save_path:
        image.save(save_path)
        print(f"Image saved to {save_path}")
    
    return image

def inference_single(model, image_path, output_path=None, device=None):
    """
    Run inference on a single image
    
    Args:
        model: Trained UNet model
        image_path: Path to input image
        output_path: Optional path to save output image
        device: Device to run inference on
    
    Returns:
        PIL.Image: Generated image
    """
    if device is None:
        device = next(model.parameters()).device
    
    print(f"  Starting inference for {os.path.basename(image_path)}")
    
    try:
        # Preprocess input
        print(f"  Step 1: Preprocessing...")
        input_tensor = preprocess_image(image_path).to(device)
        print(f"  ✓ Preprocessing complete. Input tensor shape: {input_tensor.shape}")
        print(f"  ✓ Input tensor device: {input_tensor.device}")
        print(f"  ✓ Input tensor dtype: {input_tensor.dtype}")
        print(f"  ✓ Input tensor range: [{input_tensor.min().item():.4f}, {input_tensor.max().item():.4f}]")
        
        # Check if tensor has NaN or Inf values
        if torch.isnan(input_tensor).any():
            print(f"  ✗ ERROR: Input tensor contains NaN values!")
            raise ValueError("Input tensor contains NaN values")
        if torch.isinf(input_tensor).any():
            print(f"  ✗ ERROR: Input tensor contains Inf values!")
            raise ValueError("Input tensor contains Inf values")
        
        # Run inference
        print(f"  Step 2: Running model inference...")
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        print(f"  ✓ Model inference complete. Output tensor shape: {output_tensor.shape}")
        print(f"  ✓ Output tensor range: [{output_tensor.min().item():.4f}, {output_tensor.max().item():.4f}]")
        
        # Check output tensor for issues
        if torch.isnan(output_tensor).any():
            print(f"  ✗ ERROR: Output tensor contains NaN values!")
            raise ValueError("Output tensor contains NaN values")
        if torch.isinf(output_tensor).any():
            print(f"  ✗ ERROR: Output tensor contains Inf values!")
            raise ValueError("Output tensor contains Inf values")
        
        # Postprocess output
        print(f"  Step 3: Postprocessing...")
        output_image = postprocess_image(output_tensor, output_path)
        print(f"  ✓ Postprocessing complete")
        
        return output_image
        
    except Exception as e:
        print(f"  ✗ ERROR in inference_single: {str(e)}")
        print(f"  Image path: {image_path}")
        print(f"  Device: {device}")
        import traceback
        print(f"  Full traceback: {traceback.format_exc()}")
        raise e

def inference_batch(model, input_dir, output_dir, device=None):
    """
    Run inference on a batch of images
    
    Args:
        model: Trained UNet model
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        device: Device to run inference on
    
    Returns:
        list: List of output image paths
    """
    if device is None:
        device = next(model.parameters()).device
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG files in input directory
    input_files = glob(os.path.join(input_dir, "*.png"))
    input_files.sort()  # Sort files for consistent processing order
    output_paths = []
    
    print(f"Processing {len(input_files)} images...")
    print(f"Found files: {[os.path.basename(f) for f in input_files[:10]]}...")  # Show first 10 files
    
    for i, input_path in enumerate(input_files):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, f"generated_{filename}")
        
        print(f"Processing {i+1}/{len(input_files)}: {filename}")
        
        try:
            # Check if file exists and is readable
            if not os.path.exists(input_path):
                print(f"  ERROR: File does not exist: {input_path}")
                continue
                
            # Check file size
            file_size = os.path.getsize(input_path)
            print(f"  File size: {file_size} bytes")
            
            # Try to open and check image properties
            try:
                from PIL import Image
                with Image.open(input_path) as img:
                    print(f"  Image format: {img.format}")
                    print(f"  Image mode: {img.mode}")
                    print(f"  Image size: {img.size}")
            except Exception as img_error:
                print(f"  ERROR: Cannot open image: {str(img_error)}")
                continue
            
            # Run inference
            inference_single(model, input_path, output_path, device)
            output_paths.append(output_path)
            print(f"  SUCCESS: Generated {output_path}")
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {str(e)}")
            import traceback
            print(f"  Full traceback: {traceback.format_exc()}")
    
    print(f"Inference complete. {len(output_paths)} images saved to {output_dir}")
    return output_paths

def diagnose_image(image_path):
    """
    Diagnose potential issues with a specific image file
    
    Args:
        image_path: Path to the image file to diagnose
    
    Returns:
        dict: Dictionary containing diagnostic information
    """
    diagnosis = {
        'file_exists': False,
        'file_size': 0,
        'is_readable': False,
        'image_format': None,
        'image_mode': None,
        'image_size': None,
        'can_convert_to_tensor': False,
        'error_messages': []
    }
    
    try:
        # Check if file exists
        if os.path.exists(image_path):
            diagnosis['file_exists'] = True
            diagnosis['file_size'] = os.path.getsize(image_path)
        else:
            diagnosis['error_messages'].append("File does not exist")
            return diagnosis
        
        # Try to open with PIL
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                diagnosis['is_readable'] = True
                diagnosis['image_format'] = img.format
                diagnosis['image_mode'] = img.mode
                diagnosis['image_size'] = img.size
                
                # Try to convert to grayscale
                try:
                    gray_img = img.convert("L")
                    diagnosis['can_convert_to_grayscale'] = True
                except Exception as e:
                    diagnosis['error_messages'].append(f"Cannot convert to grayscale: {str(e)}")
                    
        except Exception as e:
            diagnosis['error_messages'].append(f"Cannot open with PIL: {str(e)}")
            return diagnosis
        
        # Try to preprocess (convert to tensor)
        try:
            tensor = preprocess_image(image_path)
            diagnosis['can_convert_to_tensor'] = True
            diagnosis['tensor_shape'] = tensor.shape
        except Exception as e:
            diagnosis['error_messages'].append(f"Cannot preprocess: {str(e)}")
            
    except Exception as e:
        diagnosis['error_messages'].append(f"Unexpected error: {str(e)}")
    
    return diagnosis

def compare_images(image1_path, image2_path):
    """
    Compare two images and show their differences
    
    Args:
        image1_path: Path to first image (e.g., working "00*" image)
        image2_path: Path to second image (e.g., failing "1*" image)
    """
    print(f"Comparing images:")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")
    print()
    
    # Diagnose both images
    diag1 = diagnose_image(image1_path)
    diag2 = diagnose_image(image2_path)
    
    print("Image 1 diagnosis:")
    for key, value in diag1.items():
        print(f"  {key}: {value}")
    print()
    
    print("Image 2 diagnosis:")
    for key, value in diag2.items():
        print(f"  {key}: {value}")
    print()
    
    # Compare key properties
    print("Comparison:")
    for key in ['file_size', 'image_format', 'image_mode', 'image_size']:
        if key in diag1 and key in diag2:
            val1, val2 = diag1[key], diag2[key]
            match = "✓" if val1 == val2 else "✗"
            print(f"  {key}: {val1} vs {val2} {match}")

# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    pass
