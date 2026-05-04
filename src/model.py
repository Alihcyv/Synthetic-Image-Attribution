import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class GlobalFilterBlock(nn.Module):
    def __init__(self, dim, h, w, drop_path=0.0):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w // 2 + 1, 2) * 0.02)
        self.drop_path = nn.Dropout2d(p=drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_filtered = x_fft * weight
        x_ifft = torch.fft.irfft2(x_filtered, s=(H, W), norm='ortho')
        return x + self.drop_path(x_ifft)

class SpectralHELIX_V2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 1. Backbone
        base_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.backbone = base_model.features 
        
        self.feature_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 2. Spectral Branch
        self.spectral_proj = nn.Conv2d(768, 256, kernel_size=1)
        self.spectral_blocks = nn.Sequential(
            GlobalFilterBlock(256, 7, 7, drop_path=cfg.SPECTRAL_DROPOUT),
            nn.GELU(),
            GlobalFilterBlock(256, 7, 7, drop_path=cfg.SPECTRAL_DROPOUT),
            nn.GELU(),
            GlobalFilterBlock(256, 7, 7, drop_path=cfg.SPECTRAL_DROPOUT)
        )
        self.spectral_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, cfg.NUM_CLASS)
        )
        
        # 3. Spatial Head
        self.spatial_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(768, cfg.NUM_CLASS)
        )
        
        # 4. Dynamic Gating Network
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(768, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x) 
        spatial_logits = self.spatial_head(features)
        
        spec_feat = self.feature_pool(features) 
        spec_feat = self.spectral_proj(spec_feat)
        spec_feat = self.spectral_blocks(spec_feat)
        spectral_logits = self.spectral_head(spec_feat)
        
        g = self.gate_net(features)
        return g * spectral_logits + (1.0 - g) * spatial_logits
