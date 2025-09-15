# Pix2Pix Model - Validation and Inference

This document explains how to use the validation and inference functionality for the pix2pix model.

## Features

### Validation
- **Metrics**: PSNR, SSIM, L1 Loss, SSIM Loss
- **Visualization**: Side-by-side comparison of source, target, and predicted images
- **Batch Processing**: Efficient validation on large datasets

### Inference
- **Single Image**: Process individual images
- **Batch Processing**: Process entire directories of images
- **Flexible I/O**: Support for various image formats and output options

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Load a Trained Model

```python
from model import load_model

# Load your trained model
model = load_model("path/to/your/model.pth")
```

### 2. Validate the Model

```python
from model import PairedDataset, validate_model
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Create validation dataset
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

val_dataset = PairedDataset("path/to/val/source", "path/to/val/target", transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Run validation
metrics = validate_model(
    model=model,
    val_loader=val_loader,
    device="cuda",  # or "cpu"
    save_visualizations=True,
    save_dir="validation_results"
)

print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.4f}")
```

### 3. Single Image Inference

```python
from model import inference_single

# Process a single image
output_image = inference_single(
    model=model,
    image_path="input_image.png",
    output_path="output_image.png"
)
```

### 4. Batch Inference

```python
from model import inference_batch

# Process all images in a directory
output_paths = inference_batch(
    model=model,
    input_dir="input_directory",
    output_dir="output_directory"
)
```

## Detailed Usage

### Validation Functions

#### `validate_model(model, val_loader, device, save_visualizations=True, save_dir=None)`

Validates the model on a validation dataset and returns comprehensive metrics.

**Parameters:**
- `model`: Trained UNet model
- `val_loader`: DataLoader for validation data
- `device`: Device to run validation on ("cuda" or "cpu")
- `save_visualizations`: Whether to save comparison images
- `save_dir`: Directory to save visualizations

**Returns:**
- Dictionary containing:
  - `l1_loss`: Average L1 loss
  - `ssim_loss`: Average SSIM loss
  - `psnr`: Average Peak Signal-to-Noise Ratio
  - `ssim`: Average Structural Similarity Index

#### `calculate_psnr(img1, img2, max_val=1.0)`

Calculate Peak Signal-to-Noise Ratio between two images.

#### `calculate_ssim(img1, img2)`

Calculate Structural Similarity Index between two images.

### Inference Functions

#### `load_model(model_path, device=None)`

Load a trained model from checkpoint.

**Parameters:**
- `model_path`: Path to the model checkpoint (.pth file)
- `device`: Device to load model on (auto-detects if None)

**Returns:**
- Loaded UNet model in evaluation mode

#### `inference_single(model, image_path, output_path=None, device=None)`

Run inference on a single image.

**Parameters:**
- `model`: Trained UNet model
- `image_path`: Path to input image
- `output_path`: Optional path to save output image
- `device`: Device to run inference on

**Returns:**
- PIL Image object of the generated image

#### `inference_batch(model, input_dir, output_dir, device=None)`

Run inference on all images in a directory.

**Parameters:**
- `model`: Trained UNet model
- `input_dir`: Directory containing input images
- `output_dir`: Directory to save output images
- `device`: Device to run inference on

**Returns:**
- List of output image paths

#### `preprocess_image(image_path, size=(256, 256))`

Preprocess a single image for inference.

#### `postprocess_image(tensor, save_path=None)`

Convert model output tensor to PIL Image.

## Advanced Usage

### Custom Preprocessing

```python
import torchvision.transforms as T

def custom_preprocess(image_path, size=(512, 512)):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    from PIL import Image
    image = Image.open(image_path).convert("L")
    return transform(image).unsqueeze(0)

# Use custom preprocessing
input_tensor = custom_preprocess("image.png")
```

### Model Ensemble

```python
def ensemble_inference(model_paths, image_path, output_path=None):
    """Run inference with multiple models and average results"""
    models = [load_model(path) for path in model_paths]
    
    input_tensor = preprocess_image(image_path)
    
    predictions = []
    with torch.no_grad():
        for model in models:
            pred = model(input_tensor)
            predictions.append(pred)
    
    # Average predictions
    ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
    
    return postprocess_image(ensemble_pred, output_path)
```

### Real-time Inference

```python
def real_time_inference(model, image_tensor):
    """Run inference on a preprocessed tensor"""
    with torch.no_grad():
        output_tensor = model(image_tensor)
    return output_tensor
```

## Example Script

Run the example script to see all functionality in action:

```bash
python example_validation_inference.py
```

Make sure to update the paths in the script to match your data structure.

## Data Format

### Input Images
- Supported formats: PNG, JPG, JPEG
- Will be automatically converted to grayscale
- Automatically resized to 256x256 (configurable)

### Output Images
- Saved as PNG files
- Grayscale format
- Same size as input (after preprocessing)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model not found**: Check the model path and ensure the file exists
3. **Image loading errors**: Ensure images are in supported formats
4. **Import errors**: Install all requirements with `pip install -r requirements.txt`

### Performance Tips

1. Use GPU for faster inference: `device="cuda"`
2. Process images in batches for better efficiency
3. Use appropriate image sizes (256x256 is recommended)
4. Consider model quantization for deployment

## Metrics Explanation

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better (typically 20-40 dB)
- **SSIM (Structural Similarity Index)**: Range 0-1, higher is better
- **L1 Loss**: Mean Absolute Error, lower is better
- **SSIM Loss**: 1 - SSIM, lower is better

## File Structure

```
pix2pix/
├── model.py                           # Main model with validation/inference
├── example_validation_inference.py    # Example usage script
├── requirements.txt                   # Dependencies
├── README_validation_inference.md     # This documentation
└── __init__.py
```
