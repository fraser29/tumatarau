import torch
from bndNET.model import BoundaryPointNet
from bndNET.training import train_model
from bndNET.prediction import predict_boundaries

model = BoundaryPointNet()
train_losses, val_losses = train_model(model, train_loader, val_loader)
boundary_points, boundary_normals = predict_boundaries(model, image, points, normals)



# For RGB images (default)
model_rgb = BoundaryPointNet(input_channels=3)

# For grayscale images
model_grayscale = BoundaryPointNet(input_channels=1)


# RGB image (3 channels)
rgb_image = torch.randn(1, 3, 256, 256)  # (batch_size, channels, height, width)
model_rgb(rgb_image, points, normals)  # Works fine

# Grayscale image (1 channel)
gray_image = torch.randn(1, 1, 256, 256)
model_grayscale(gray_image, points, normals)  # Works fine

# Wrong number of channels
model_rgb(gray_image, points, normals)  # Raises ValueError
model_grayscale(rgb_image, points, normals)  # Raises ValueError