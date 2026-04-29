import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from src.dataset import ImageDataset, get_transforms
from src.model import load_model

def run_inference(cfg: CFG):
    """Predicts classes for the test set using ensemble of folds and TTA."""
    print("\n" + "="*30 + "\nStarting Inference...\n" + "="*30)
    
    test_df = pd.read_csv(cfg.TEST_FILE)
    inference_dataset = ImageDataset(
        dataframe=test_df,
        img_dir=cfg.IMAGE_DIR,
        transform=get_transforms(cfg, is_train=False)
    )
    inference_loader = DataLoader(
        inference_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4
    )

    num_samples = len(inference_dataset)
    num_class = cfg.NUM_CLASS
    total_probs = np.zeros((num_samples, num_class))

    for fold in range(cfg.NUM_SPLIT):
        path = f'best_modelfold{fold+1}.pth'
        print(f"Loading fold model: {path}")
        
        model = load_model(cfg)
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
                    if tta_step % 2 == 0:
                        img_variant = TF.hflip(img_variant)
                    
                    outputs = model(img_variant)
                    batch_probs += torch.softmax(outputs, dim=1)
                
                fold_probs.append((batch_probs / cfg.TTA_STEPS).cpu().numpy())

        total_probs += np.concatenate(fold_probs, axis=0)
        del model
        torch.cuda.empty_cache()

    final_predictions = np.argmax(total_probs, axis=1)
    submission = pd.DataFrame({"ID": test_df["ID"], "TARGET": final_predictions})
    submission.to_csv("submission.csv", index=False)
    print("Submission file saved successfully!")
