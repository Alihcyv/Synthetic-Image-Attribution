import torch
import os

class CFG:
    MODEL_NAME = "convnext_tiny" 
    NUM_CLASS = 10
    IMG_SIZE = 256
    CROP_SIZE = 224
    
    LR_BACKBONE = 4e-5 
    LR_SPECTRAL = 5e-4
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    NUM_SPLIT = 3
    LABEL_SMOOTHING = 0.05
    TTA_STEPS = 5
    SPECTRAL_DROPOUT = 0.1
    EMA_DECAY = 0.999
    
    DATA_ROOT = '/kaggle/input/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge/Data/Data'
    
    TRAIN_FILE = os.path.join(DATA_ROOT, 'training.csv')
    TEST_FILE = os.path.join(DATA_ROOT, 'test.csv')
    IMAGE_DIR = os.path.dirname(DATA_ROOT) 
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
