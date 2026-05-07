%%writefile train_full.py
import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG
from dataset import ImageDataset, get_transforms
from model import SpectralHELIX_V2, EMA
from trainer import run_epoch
from inference_tta import run_inference

def train_full_data():
    print(f"\n{'='*20} TRAINING ON FULL DATASET {'='*20}")
    
    train_df = pd.read_csv(CFG.TRAIN_FILE)
    
    train_dataset = ImageDataset(train_df, CFG.IMAGE_DIR, get_transforms(CFG, is_train=True))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CFG.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    model = SpectralHELIX_V2(CFG).to(CFG.DEVICE)
    ema = EMA(model, CFG.EMA_DECAY)
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': CFG.LR_BACKBONE},
        {'params': model.spectral_proj.parameters(), 'lr': CFG.LR_SPECTRAL},
        {'params': model.spectral_blocks.parameters(), 'lr': CFG.LR_SPECTRAL},
        {'params': model.spectral_head.parameters(), 'lr': CFG.LR_SPECTRAL},
        {'params': model.spatial_head.parameters(), 'lr': CFG.LR_SPECTRAL},
        {'params': model.gate_net.parameters(), 'lr': CFG.LR_SPECTRAL}
    ], weight_decay=CFG.WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.NUM_EPOCHS)
    
    for epoch in range(CFG.NUM_EPOCHS):
        train_loss = run_epoch(model, ema, train_loader, loss_fn, optimizer, CFG)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{CFG.NUM_EPOCHS} | Loss: {train_loss:.4f}")
        
        if (epoch + 1) % 5 == 0 or (epoch + 1) == CFG.NUM_EPOCHS:
            save_path = f'full_model_epoch_{epoch+1}.pth'
            torch.save(ema.shadow.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    final_path = 'best_modelfold_full.pth'
    torch.save(ema.shadow.state_dict(), final_path)
    print(f"\nFinal full model saved as {final_path}")

    run_inference(CFG)

if __name__ == "__main__":
    train_full_data()
