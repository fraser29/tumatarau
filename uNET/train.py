import os
import torch
import torch.nn as nn
from model import UNet, train_step, evaluate, save_checkpoint
from typing import Dict, Union, Path



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UNetTrainer:
    def __init__(self, model: UNet, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, num_epochs: int, work_dir: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.work_dir = work_dir
        self.optimizer = torch.optim.Adam(model.parameters())
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = train_step(self.model, self.train_loader, self.optimizer, self.criterion, self.device)
            val_loss = evaluate(self.model, self.val_loader, self.criterion, self.device)
            
            self.save_checkpoint(epoch, train_loss)

    def save_checkpoint(self, epoch: int, train_loss: float):
        save_checkpoint(self.model, self.optimizer, epoch, train_loss, os.path.join(self.work_dir, f"checkpoint_epoch_{epoch}.pt"))

    @classmethod
    def from_config(cls, config: Union[Dict, str, Path]) -> 'UNetTrainer':
        model = UNet.from_config(config)
        num_epochs = config.get("num_epochs", 100)
        work_dir = config.get("work_dir")
        model = model.to(device)
        return cls(model, num_epochs=num_epochs, work_dir=work_dir)

