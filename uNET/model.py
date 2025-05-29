import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Dict, Optional, Union
from pathlib import Path

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 1,
                 features: list = [64, 128, 256, 512],
                 dropout_rate: float = 0.1):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Downsampling path
        in_features = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_features, feature))
            in_features = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Upsampling path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            x = self.dropout(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

    @classmethod
    def from_config(cls, config: Union[Dict, str, Path]) -> 'UNet':
        """
        Create a UNet model from a configuration dictionary or JSON file.
        
        Args:
            config: Either a dictionary of parameters or a path to a JSON file
            
        Returns:
            UNet model instance
        """
        if isinstance(config, (str, Path)):
            with open(config, 'r') as f:
                config = json.load(f)
        
        return cls(
            in_channels=config.get('in_channels', 3),
            out_channels=config.get('out_channels', 1),
            features=config.get('features', [64, 128, 256, 512]),
            dropout_rate=config.get('dropout_rate', 0.1)
        )

def train_step(model: UNet,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               device: torch.device) -> float:
    """
    Perform one training epoch.
    
    Args:
        model: UNet model
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model: UNet,
            dataloader: torch.utils.data.DataLoader,
            criterion: nn.Module,
            device: torch.device) -> float:
    """
    Evaluate the model on the validation/test set.
    
    Args:
        model: UNet model
        dataloader: Validation/test dataloader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Average loss on the validation/test set
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def save_checkpoint(model: UNet,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   path: Union[str, Path]):
    """
    Save model checkpoint.
    
    Args:
        model: UNet model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model: UNet,
                   optimizer: torch.optim.Optimizer,
                   path: Union[str, Path]) -> tuple:
    """
    Load model checkpoint.
    
    Args:
        model: UNet model
        optimizer: Optimizer
        path: Path to checkpoint
        
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
