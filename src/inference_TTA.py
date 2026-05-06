%%writefile inference_tta.py
import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader
from config import CFG
from dataset import ImageDataset, get_transforms
from model import SpectralHELIX_V2

def run_inference(cfg):
    print("\nStarting Inference...")
    test_df = pd.read_csv(cfg.TEST_FILE)
    inference_dataset = ImageDataset(test_df, cfg.IMAGE_DIR, get_transforms(cfg, False))
    inference_loader = DataLoader(inference_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)
    
    num_samples, num_class = len(inference_dataset), cfg.NUM_CLASS
    total_probs = np.zeros((num_samples, num_class))

    for fold in range(cfg.NUM_SPLIT):
        path = f'best_modelfold{fold+1}.pth'
        if not os.path.exists(path): continue
        
        model = SpectralHELIX_V2(cfg).to(cfg.DEVICE)
        model.load_state_dict(torch.load(path, map_location=cfg.DEVICE))
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for batch in inference_loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(cfg.DEVICE)
                batch_probs = torch.zeros((images.size(0), num_class)).to(cfg.DEVICE)
                
                for tta_step in range(cfg.TTA_STEPS):
                    img_variant = images.clone()
                    if tta_step == 1: img_variant = TF.hflip(img_variant)
                    if tta_step == 2: img_variant = TF.rotate(img_variant, 10)
                    if tta_step == 3: img_variant = TF.rotate(img_variant, -10)
                    if tta_step == 4: img_variant = TF.adjust_brightness(img_variant, 1.1)
                    
                    outputs = model(img_variant)
                    batch_probs += torch.softmax(outputs, dim=1)
                
                fold_probs.append((batch_probs / cfg.TTA_STEPS).cpu().numpy())
        
        total_probs += np.concatenate(fold_probs, axis=0)
        del model
        torch.cuda.empty_cache()

    final_predictions = np.argmax(total_probs, axis=1)
    submission = pd.DataFrame({"ID": test_df["ID"], "TARGET": final_predictions})
    submission.to_csv("submission.csv", index=False)
    print("Submission saved successfully!")
