#!/usr/bin/env python3
"""
Example script demonstrating validation and inference for the pix2pix model.

This script shows how to:
1. Load a trained model
2. Validate the model on a validation dataset
3. Run inference on single images
4. Run batch inference on multiple images
"""

import os
import torch
from model import (
    UNet, PairedDataset, load_model, validate_model, 
    inference_single, inference_batch, preprocess_image, postprocess_image
)
from torch.utils.data import DataLoader
import torchvision.transforms as T


# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def run_validation(checkpoint_path, val_source_dir, val_target_dir, validation_output_dir):
    
    # Paths - Update these to match your data structure
    model_path = checkpoint_path  # Update this path
    
    # Create output directories
    os.makedirs(validation_output_dir, exist_ok=True)
    
    # 1. Load trained model
    print("\n" + "="*50)
    print("1. Loading trained model...")
    print("="*50)
    
    if os.path.exists(model_path):
        model = load_model(model_path, device)
    else:
        print(f"Model not found at {model_path}")
        print("Please update the model_path variable with the correct path to your trained model.")
        return
    
    # 2. Validate model
    print("\n" + "="*50)
    print("2. Validating model...")
    print("="*50)
    
    if os.path.exists(val_source_dir) and os.path.exists(val_target_dir):
        # Create validation dataset
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        
        val_dataset = PairedDataset(val_source_dir, val_target_dir, transform)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # Run validation
        metrics = validate_model(
            model=model,
            val_loader=val_loader,
            device=device,
            save_visualizations=True,
            save_dir=validation_output_dir
        )
        
        print(f"\nValidation completed. Results saved to {validation_output_dir}")
    else:
        print(f"Validation directories not found:")
        print(f"  Source: {val_source_dir}")
        print(f"  Target: {val_target_dir}")
        print("Please update the paths and try again.")
    

def run_inference(model_path, single_image_path, single_output_path):
    # 3. Single image inference
    print("\n" + "="*50)
    print("3. Single image inference...")
    print("="*50)
    
    if os.path.exists(model_path):
        model = load_model(model_path, device)
        
        if os.path.exists(single_image_path):
            try:
                output_image = inference_single(
                    model=model,
                    image_path=single_image_path,
                    output_path=single_output_path,
                    device=device
                )
                print(f"Single image inference completed. Output saved to {single_output_path}")
            except Exception as e:
                print(f"Error in single image inference: {str(e)}")
        else:
            print(f"Single image not found at {single_image_path}")
            print("Please update the single_image_path variable with the correct path.")
    else:
        print(f"Model not found at {model_path}")
        print("Please update the model_path variable with the correct path.")
    

def run_batch_inference(model_path, inference_input_dir, inference_output_dir):
    # 4. Batch inference
    print("\n" + "="*50)
    print("4. Batch inference...")
    print("="*50)
    
    if os.path.exists(model_path):
        model = load_model(model_path, device)
        
        if os.path.exists(inference_input_dir):
            try:
                output_paths = inference_batch(
                    model=model,
                    input_dir=inference_input_dir,
                    output_dir=inference_output_dir,
                    device=device
                )
                print(f"Batch inference completed. {len(output_paths)} images processed.")
            except Exception as e:
                print(f"Error in batch inference: {str(e)}")
        else:
            print(f"Input directory not found at {inference_input_dir}")
            print("Please update the inference_input_dir variable with the correct path.")
    else:
        print(f"Model not found at {model_path}")
        print("Please update the model_path variable with the correct path.")
    
    print("\n" + "="*50)
    print("Example completed!")
    print("="*50)


def main():
    # Example usage - update these paths as needed
    checkpoint_path = "/path/to/your/model.pth"
    val_source_dir = "/path/to/val/source"
    val_target_dir = "/path/to/val/target"
    validation_output_dir = "/path/to/validation/results"
    single_image_path = "/path/to/single/image.png"
    single_output_path = "/path/to/single/output.png"
    inference_input_dir = "/path/to/inference/input"
    inference_output_dir = "/path/to/inference/output"
    
    run_validation(checkpoint_path, val_source_dir, val_target_dir, validation_output_dir)
    run_inference(checkpoint_path, single_image_path, single_output_path)
    run_batch_inference(checkpoint_path, inference_input_dir, inference_output_dir)


if __name__ == "__main__":
    # Run basic example
    main()
    
