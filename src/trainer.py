%%writefile trainer.py
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def run_epoch(model, ema, loader, loss_fn, optimizer, cfg):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        mixed_images, labels_a, labels_b, lam = mixup_data(images, labels)
        
        optimizer.zero_grad()
        predictions = model(mixed_images)
        loss = mixup_criterion(loss_fn, predictions, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()
        ema.update(model)
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(loader)

def evaluate_model(model, loader, cfg):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total
