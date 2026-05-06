%%writefile main.py
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from src.config import CFG
from src.dataset import prepare_dataloaders
from src.model import SpectralHELIX_V2, EMA
from src.trainer import run_epoch, evaluate_model
from src.inference_tta import run_inference

def train_one_fold(fold, train_df, val_df, cfg):
    print(f"\n{'#'*20} FOLD -> {fold+1} {'#'*20}")
    train_loader, val_loader = prepare_dataloaders(train_df, val_df, cfg)
    
    model = SpectralHELIX_V2(cfg).to(cfg.DEVICE)
    ema = EMA(model, cfg.EMA_DECAY)
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg.LR_BACKBONE},
        {'params': model.spectral_proj.parameters(), 'lr': cfg.LR_SPECTRAL},
        {'params': model.spectral_blocks.parameters(), 'lr': cfg.LR_SPECTRAL},
        {'params': model.spectral_head.parameters(), 'lr': cfg.LR_SPECTRAL},
        {'params': model.spatial_head.parameters(), 'lr': cfg.LR_SPECTRAL},
        {'params': model.gate_net.parameters(), 'lr': cfg.LR_SPECTRAL}
    ], weight_decay=cfg.WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
    best_val_acc = 0.0
    
    for epoch in range(cfg.NUM_EPOCHS):
        train_loss = run_epoch(model, ema, train_loader, loss_fn, optimizer, cfg)
        scheduler.step() 
        val_acc = evaluate_model(ema.shadow, val_loader, cfg)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ema.shadow.state_dict(), f'best_modelfold{fold+1}.pth')
            print(f"Epoch {epoch+1}: New best model saved with Acc: {val_acc:.4f}")
        
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
    return best_val_acc

def main():
    train_df = pd.read_csv(CFG.TRAIN_FILE)
    kf = StratifiedKFold(n_splits=CFG.NUM_SPLIT, shuffle=True, random_state=42)
    fold_results = []

    for fold, (idx_tr, idx_val) in enumerate(kf.split(train_df, train_df['y'])):
        acc = train_one_fold(fold, train_df.iloc[idx_tr], train_df.iloc[idx_val], CFG)
        fold_results.append(acc)

    print(f"\nOverall CV Accuracy: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    run_inference(CFG)

if __name__ == "__main__":
    main()
