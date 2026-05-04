import torch
import copy

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm 
from typing import Any 

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def run_epoch(model: nn.Module, ema: EMA, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, cfg: Any) -> float:
    
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in progress_bar:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        
        ema.update(model)
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


def evaluate_model(model: nn.Module, loader: DataLoader, cfg: Any) -> float:

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    return correct / total
