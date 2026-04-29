import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Tuple, Optional, Any

class ImageDataset(Dataset):
    """
    Custom Dataset for loading images from a dataframe.
    Supports both training (image, label) and inference (image only) modes.
    """
    def __init__(self, dataframe: pd.DataFrame, img_dir: str, transform: Optional[transforms.Compose] = None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Any:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['path'])
        
        # Convert to RGB to ensure 3 channels (handles grayscale or RGBA images)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Return image and label if target 'y' exists in the dataframe
        if 'y' in row:
            return image, row['y']
        
        return image


def get_transforms(cfg: Any, is_train: bool = True) -> transforms.Compose:
    """
    Creates transformation pipeline based on the mode (train or validation/test).
    """
    # Standard ImageNet normalization constants
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=1/3),
            transforms.RandomAutocontrast(p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])


def prepare_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Any) -> Tuple[DataLoader, DataLoader]:
    """
    Prepares train and validation DataLoaders.
    """
    train_transform = get_transforms(cfg, is_train=True)
    val_transform = get_transforms(cfg, is_train=False)

    train_dataset = ImageDataset(
        dataframe=train_df,
        img_dir=cfg.IMAGE_DIR,
        transform=train_transform
    )
    
    val_dataset = ImageDataset(
        dataframe=val_df,
        img_dir=cfg.IMAGE_DIR,
        transform=val_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS if hasattr(cfg, 'NUM_WORKERS') else 4,
        pin_memory=True # Added for faster GPU data transfer
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False, 
        num_workers=cfg.NUM_WORKERS if hasattr(cfg, 'NUM_WORKERS') else 4,
        pin_memory=True # Added for faster GPU data transfer
    )

    return train_loader, val_loader
