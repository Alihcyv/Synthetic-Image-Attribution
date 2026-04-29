import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm 

def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, cfg: CFG) -> float:
    """
    Trains the model for one epoch.
    Returns the average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    
    # Use tqdm for a professional progress bar instead of manual print
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in progress_bar:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar with current batch loss
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


def evaluate_model(model: nn.Module, loader: DataLoader, cfg: CFG) -> float:
    """
    Evaluates the model accuracy on a validation set.
    Returns the accuracy score.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        # tqdm added here for long validation sets
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    return correct / total


def train_one_fold(fold: int, train_df, val_df, cfg: CFG) -> float:
    """
    Logic for training a single fold of cross-validation.
    """
    print(f"\n{'#'*20} FOLD -> {fold+1} {'#'*20}")
    
    # 1. Data Preparation
    train_loader, val_loader = prepare_dataloaders(train_df, val_df, cfg)
    
    # 2. Model Initialization
    model = load_model(cfg)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=cfg.NUM_EPOCHS, 
        eta_min=1e-6
    )

    best_val_acc = 0.0
    
    for epoch in range(cfg.NUM_EPOCHS):
        # Training phase
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, cfg)
        
        # Step the scheduler after each epoch
        scheduler.step() 
        
        # Validation phase
        val_acc = evaluate_model(model, val_loader, cfg)
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_modelfold{fold+1}.pth')
            print(f"Epoch {epoch+1}: New best model saved with Acc: {val_acc:.4f}")
        
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
    return best_val_acc
