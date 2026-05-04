import os
import torch
import pandas as pd
import torchvision.transforms.v2 as v2

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional, Any

class ImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_dir: str, transform: Optional[transforms.Compose] = None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Any:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['path'])
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if 'y' in row:
            return image, row['y']
        
        return image


def get_transforms(cfg: Any, is_train: bool = True) -> v2.Compose:
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    if is_train:
        return v2.Compose([
            v2.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE), antialias=True),
            v2.RandomResizedCrop(size=(cfg.CROP_SIZE, cfg.CROP_SIZE), scale=(0.8, 1.0)),
            v2.RandomRotation(degrees=15),
            v2.RandomApply([v2.RandomChoice([
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                v2.RandomGrayscale(p=1.0),
                v2.RandomAdjustSharpness(sharpness_factor=2, p=1.0) 
            ])], p=0.8),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=norm_mean, std=norm_std),
        ])
    return v2.Compose([
        v2.Resize((cfg.CROP_SIZE, cfg.CROP_SIZE), antialias=True),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=norm_mean, std=norm_std),
    ])
    
def prepare_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Any) -> Tuple[DataLoader, DataLoader]:
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
        num_workers=4,
        pin_memory=True 
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False, 
        num_workers=4,
        pin_memory=True 
    )

    return train_loader, val_loader
