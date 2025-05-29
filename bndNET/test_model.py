import torch
import numpy as np
from model import BoundaryPointNet, train_model, predict_boundaries
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random
from mpl_toolkits.mplot3d import Axes3D

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=100, volume_size=64, num_points=1024):
        """
        Create a synthetic dataset for testing the boundary point detection model
        
        Args:
            num_samples: Number of samples to generate
            volume_size: Size of the cubic volume (D=H=W)
            num_points: Number of points per sample
        """
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.num_points = num_points
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate a synthetic 3D volume with some geometric shapes
        volume = self._generate_volume()
        
        # Generate random points in the volume space
        points = self._generate_points()
        
        # Generate normals for the points
        normals = self._generate_normals(points)
        
        # Generate boundary labels (1 for boundary points, 0 for interior points)
        boundary_labels = self._generate_boundary_labels(points)
        
        # Convert to tensors
        volume = torch.FloatTensor(volume)
        points = torch.FloatTensor(points)
        normals = torch.FloatTensor(normals)
        boundary_labels = torch.FloatTensor(boundary_labels)
        
        return volume, points, normals, boundary_labels, normals  # Using normals as target normals for simplicity
    
    def _generate_volume(self):
        """Generate a synthetic 3D volume with geometric shapes"""
        volume = np.zeros((1, self.volume_size, self.volume_size, self.volume_size))
        
        # Generate some random shapes
        num_shapes = random.randint(3, 7)
        for _ in range(num_shapes):
            shape_type = random.choice(['sphere', 'cube'])
            if shape_type == 'sphere':
                center_x = random.randint(20, self.volume_size-20)
                center_y = random.randint(20, self.volume_size-20)
                center_z = random.randint(20, self.volume_size-20)
                radius = random.randint(10, 20)
                intensity = random.random()
                
                x, y, z = np.ogrid[:self.volume_size, :self.volume_size, :self.volume_size]
                mask = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= radius**2
                volume[0][mask] = intensity
                    
            else:  # cube
                x1 = random.randint(0, self.volume_size-30)
                y1 = random.randint(0, self.volume_size-30)
                z1 = random.randint(0, self.volume_size-30)
                size = random.randint(15, 30)
                intensity = random.random()
                
                volume[0, x1:x1+size, y1:y1+size, z1:z1+size] = intensity
        
        return volume
    
    def _generate_points(self):
        """Generate random points in the volume space"""
        points = np.random.rand(self.num_points, 3)
        points[:, 0] *= self.volume_size  # x coordinate
        points[:, 1] *= self.volume_size  # y coordinate
        points[:, 2] *= self.volume_size  # z coordinate
        return points
    
    def _generate_normals(self, points):
        """Generate random normal vectors for the points"""
        normals = np.random.randn(self.num_points, 3)
        # Normalize the normals
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        return normals
    
    def _generate_boundary_labels(self, points):
        """Generate synthetic boundary labels"""
        # In this example, we'll mark points near the edges of the volume as boundary points
        boundary_labels = np.zeros(self.num_points)
        margin = 10
        boundary_mask = (
            (points[:, 0] < margin) | 
            (points[:, 0] > self.volume_size - margin) |
            (points[:, 1] < margin) | 
            (points[:, 1] > self.volume_size - margin) |
            (points[:, 2] < margin) | 
            (points[:, 2] > self.volume_size - margin)
        )
        boundary_labels[boundary_mask] = 1
        return boundary_labels

def visualize_results(volume, points, boundary_points, boundary_normals):
    """Visualize the input volume and detected boundary points"""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot volume slices
    ax1 = fig.add_subplot(121, projection='3d')
    # Plot a few points from the volume to show its structure
    x, y, z = np.where(volume[0] > 0.5)
    ax1.scatter(x, y, z, c='gray', alpha=0.1, s=1)
    ax1.set_title('Input Volume')
    
    # Plot points and boundary points
    ax2 = fig.add_subplot(122, projection='3d')
    # Plot a few points from the volume to show its structure
    ax2.scatter(x, y, z, c='gray', alpha=0.1, s=1)
    # Plot all points
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c='blue', s=1, alpha=0.5, label='All Points')
    # Plot boundary points
    ax2.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2], 
               c='red', s=5, label='Boundary Points')
    
    # Plot normal vectors for boundary points
    for point, normal in zip(boundary_points, boundary_normals):
        ax2.quiver(point[0], point[1], point[2],
                  normal[0] * 5, normal[1] * 5, normal[2] * 5,
                  color='red', arrow_length_ratio=0.2)
    
    ax2.set_title('Detected Boundary Points')
    ax2.legend()
    plt.show()

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create synthetic datasets
    train_dataset = SyntheticDataset(num_samples=50, volume_size=64)
    val_dataset = SyntheticDataset(num_samples=10, volume_size=64)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Initialize model
    model = BoundaryPointNet()
    
    # Train model
    print("Training model...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,  # Small number for testing
        learning_rate=0.001
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()
    
    # Test prediction on a single sample
    print("\nTesting prediction on a single sample...")
    test_dataset = SyntheticDataset(num_samples=1, volume_size=64)
    volume, points, normals, _, _ = test_dataset[0]
    
    # Add batch dimension
    volume = volume.unsqueeze(0)
    points = points.unsqueeze(0)
    normals = normals.unsqueeze(0)
    
    # Make prediction
    boundary_points, boundary_normals = predict_boundaries(
        model=model,
        volume=volume,
        points=points,
        normals=normals,
        threshold=0.5
    )
    
    # Visualize results
    visualize_results(
        volume=volume[0].numpy(),
        points=points[0].numpy(),
        boundary_points=boundary_points,
        boundary_normals=boundary_normals
    )

if __name__ == "__main__":
    main() 