%%writefile model.py
import torch
import torch.nn as nn
import copy
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

class GlobalFilterBlock(nn.Module):
    def __init__(self, dim, h, w, drop_path=0.0):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w // 2 + 1, 2) * 0.02)
        self.drop_path = nn.Dropout2d(p=drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_ifft = torch.fft.irfft2(x_fft * weight, s=(H, W), norm='ortho')
        return x + self.drop_path(x_ifft)

class SpectralHELIX_V2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        base_model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        self.backbone = base_model.features 
        self.feature_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.spectral_proj = nn.Conv2d(768, 256, kernel_size=1)
        self.spectral_blocks = nn.Sequential(
            GlobalFilterBlock(256, 7, 7, drop_path=cfg.SPECTRAL_DROPOUT), nn.GELU(),
            GlobalFilterBlock(256, 7, 7, drop_path=cfg.SPECTRAL_DROPOUT), nn.GELU(),
            GlobalFilterBlock(256, 7, 7, drop_path=cfg.SPECTRAL_DROPOUT), nn.GELU(),
            GlobalFilterBlock(256, 7, 7, drop_path=cfg.SPECTRAL_DROPOUT),
        )
        
        self.spectral_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, cfg.NUM_CLASS)
        )
        
        self.spatial_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(768, cfg.NUM_CLASS)
        )
        
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(768, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        spatial_logits = self.spatial_head(features)
        spec_feat = self.spectral_proj(self.feature_pool(features))
        spectral_logits = self.spectral_head(self.spectral_blocks(spec_feat))
        g = self.gate_net(features)
        return g * spectral_logits + (1.0 - g) * spatial_logits
