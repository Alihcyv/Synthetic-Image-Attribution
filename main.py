import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.config import CFG
from src.trainer import train_one_fold
from src.inference import run_inference

def main():
    # Load data
    train_df = pd.read_csv(CFG.TRAIN_FILE)
    
    # Cross-Validation Setup
    kf = StratifiedKFold(n_splits=CFG.NUM_SPLIT, shuffle=True, random_state=42)
    fold_results = []

    for fold, (idx_tr, idx_val) in enumerate(kf.split(train_df, train_df['y'])):
        train_subset = train_df.iloc[idx_tr]
        val_subset = train_df.iloc[idx_val]
        
        acc = train_one_fold(fold, train_subset, val_subset, CFG)
        fold_results.append(acc)

    print(f"\nOverall CV Accuracy: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    
    # Run Inference
    run_inference(CFG)

if __name__ == "__main__":
    main()
