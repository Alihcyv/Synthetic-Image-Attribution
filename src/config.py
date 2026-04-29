import torch
import os

class CFG:
    MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"
    NUM_CLASS = 10
    IMG_SIZE = 224
    
    LR = 3e-4
    BATCH_SIZE = 64
    NUM_EPOCHS = 5
    NUM_SPLIT = 6
    LABEL_SMOOTHING = 0.1
    TTA_STEPS = 5
    
    DATA_ROOT = '/kaggle/input/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge/Data/Data'
    TRAIN_FILE = os.path.join(DATA_ROOT, 'training.csv')
    TEST_FILE = os.path.join(DATA_ROOT, 'test.csv')
    
    IMAGE_DIR = os.path.dirname(DATA_ROOT) 
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
