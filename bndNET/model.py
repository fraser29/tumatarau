import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class BoundaryPointNet(nn.Module):
    def __init__(self, num_points: int = 1024):
        """
        Initialize the Boundary Point Detection Network for 3D volumes
        
        Args:
            num_points: Maximum number of points to process (default: 1024)
        """
        super(BoundaryPointNet, self).__init__()
        
        # 3D Volume feature extraction
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)  # Single channel input
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        
        # Point feature processing
        self.point_conv1 = nn.Conv1d(3, 64, 1)  # 3 for xyz coordinates
        self.point_conv2 = nn.Conv1d(64, 128, 1)
        self.point_conv3 = nn.Conv1d(128, 256, 1)
        
        # Normal feature processing
        self.normal_conv1 = nn.Conv1d(3, 64, 1)  # 3 for normal vectors
        self.normal_conv2 = nn.Conv1d(64, 128, 1)
        self.normal_conv3 = nn.Conv1d(128, 256, 1)
        
        # Fusion layers
        self.fusion_conv1 = nn.Conv1d(256 + 256 + 256, 512, 1)
        self.fusion_conv2 = nn.Conv1d(512, 256, 1)
        self.fusion_conv3 = nn.Conv1d(256, 128, 1)
        
        # Output layers
        self.boundary_classifier = nn.Conv1d(128, 1, 1)  # Binary classification for boundary points
        self.normal_regressor = nn.Conv1d(128, 3, 1)     # Normal vector regression
        
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm3d(32)
        self.batch_norm2 = nn.BatchNorm3d(64)
        self.batch_norm3 = nn.BatchNorm3d(128)
        self.batch_norm4 = nn.BatchNorm3d(256)
        
        self.point_bn1 = nn.BatchNorm1d(64)
        self.point_bn2 = nn.BatchNorm1d(128)
        self.point_bn3 = nn.BatchNorm1d(256)
        
        self.normal_bn1 = nn.BatchNorm1d(64)
        self.normal_bn2 = nn.BatchNorm1d(128)
        self.normal_bn3 = nn.BatchNorm1d(256)
        
        self.fusion_bn1 = nn.BatchNorm1d(512)
        self.fusion_bn2 = nn.BatchNorm1d(256)
        self.fusion_bn3 = nn.BatchNorm1d(128)
        
        self.max_pool = nn.MaxPool3d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def forward(self, volume: torch.Tensor, points: torch.Tensor, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network
        
        Args:
            volume: Input 3D volume tensor of shape (B, 1, D, H, W)
            points: Point coordinates tensor of shape (B, N, 3)
            normals: Normal vectors tensor of shape (B, N, 3)
            
        Returns:
            Tuple of (boundary_scores, predicted_normals)
        """
        # Validate input volume channels
        if volume.size(1) != 1:
            raise ValueError(f"Expected volume with 1 channel, but got {volume.size(1)} channels")
        
        batch_size = volume.size(0)
        
        # Process volume features
        x = F.relu(self.batch_norm1(self.conv1(volume)))
        x = self.max_pool(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.max_pool(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.max_pool(x)
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = x.view(batch_size, 256, 1)
        
        # Process point features
        points = points.transpose(1, 2)  # (B, 3, N)
        p = F.relu(self.point_bn1(self.point_conv1(points)))
        p = F.relu(self.point_bn2(self.point_conv2(p)))
        p = F.relu(self.point_bn3(self.point_conv3(p)))
        
        # Process normal features
        normals = normals.transpose(1, 2)  # (B, 3, N)
        n = F.relu(self.normal_bn1(self.normal_conv1(normals)))
        n = F.relu(self.normal_bn2(self.normal_conv2(n)))
        n = F.relu(self.normal_bn3(self.normal_conv3(n)))
        
        # Concatenate and fuse features
        x = x.expand(-1, -1, points.size(2))  # Expand volume features to match point count
        combined = torch.cat([x, p, n], dim=1)
        
        # Fusion layers
        f = F.relu(self.fusion_bn1(self.fusion_conv1(combined)))
        f = F.relu(self.fusion_bn2(self.fusion_conv2(f)))
        f = F.relu(self.fusion_bn3(self.fusion_conv3(f)))
        
        # Output predictions
        boundary_scores = torch.sigmoid(self.boundary_classifier(f))
        predicted_normals = self.normal_regressor(f)
        
        return boundary_scores.squeeze(1), predicted_normals.transpose(1, 2)

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[List[float], List[float]]:
    """
    Train the boundary point detection model
    
    Args:
        model: The BoundaryPointNet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (training_losses, validation_losses)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Loss functions
    boundary_criterion = nn.BCELoss()
    normal_criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        
        for batch_idx, (volumes, points, normals, boundary_labels, target_normals) in enumerate(train_loader):
            volumes = volumes.to(device)
            points = points.to(device)
            normals = normals.to(device)
            boundary_labels = boundary_labels.to(device)
            target_normals = target_normals.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            boundary_scores, predicted_normals = model(volumes, points, normals)
            
            # Calculate losses
            boundary_loss = boundary_criterion(boundary_scores, boundary_labels)
            normal_loss = normal_criterion(predicted_normals, target_normals)
            total_loss = boundary_loss + normal_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for volumes, points, normals, boundary_labels, target_normals in val_loader:
                volumes = volumes.to(device)
                points = points.to(device)
                normals = normals.to(device)
                boundary_labels = boundary_labels.to(device)
                target_normals = target_normals.to(device)
                
                boundary_scores, predicted_normals = model(volumes, points, normals)
                
                boundary_loss = boundary_criterion(boundary_scores, boundary_labels)
                normal_loss = normal_criterion(predicted_normals, target_normals)
                total_loss = boundary_loss + normal_loss
                
                epoch_val_loss += total_loss.item()
        
        # Calculate average losses
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def predict_boundaries(
    model: nn.Module,
    volume: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict boundary points and their normals for a given 3D volume and point set
    
    Args:
        model: Trained BoundaryPointNet model
        volume: Input 3D volume tensor of shape (1, 1, D, H, W)
        points: Point coordinates tensor of shape (1, N, 3)
        normals: Normal vectors tensor of shape (1, N, 3)
        device: Device to run inference on
        threshold: Classification threshold for boundary points
        
    Returns:
        Tuple of (boundary_points, boundary_normals) as numpy arrays
    """
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        volume = volume.to(device)
        points = points.to(device)
        normals = normals.to(device)
        
        boundary_scores, predicted_normals = model(volume, points, normals)
        
        # Convert to numpy arrays
        boundary_scores = boundary_scores.cpu().numpy()
        predicted_normals = predicted_normals.cpu().numpy()
        points = points.cpu().numpy()
        
        # Get boundary points
        boundary_mask = boundary_scores > threshold
        boundary_points = points[0][boundary_mask[0]]
        boundary_normals = predicted_normals[0][boundary_mask[0]]
        
        return boundary_points, boundary_normals
