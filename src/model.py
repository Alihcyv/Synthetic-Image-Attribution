import timm
from torch import nn

def load_model(cfg) -> nn.Module:
    """
    Creates a model using the timm library, sets the number of classes,
    and moves the model to the specified device (CPU/GPU).
    """
    print(f"Loading model: {cfg.MODEL_NAME}...")

    try:
        model = timm.create_model(
            cfg.MODEL_NAME, 
            pretrained=True, 
            num_classes=cfg.NUM_CLASS
        )
    except Exception as e:
        print(f"Error loading model {cfg.MODEL_NAME}: {e}")
        raise e

    model = model.to(cfg.DEVICE)
    return model

