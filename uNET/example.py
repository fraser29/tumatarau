import json
import torch
import torch.nn as nn
from train import UNetTrainer

# Create model from JSON config
config = {
    "in_channels": 3,
    "out_channels": 1,
    "features": [64, 128, 256, 512],
    "dropout_rate": 0.1,
    "num_epochs": 100,
    "work_dir": "work_dir"
}


trainer = UNetTrainer.from_config(config)
trainer.train()
