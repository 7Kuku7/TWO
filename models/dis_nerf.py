import torch
import torch.nn as nn
from .backbone import get_content_encoder, get_distortion_encoder
from .mi_estimator import MIEstimator

class DisNeRFQA(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Content Branch (F) - ViT-Base
        self.content_encoder = get_content_encoder(pretrained=True)
        self.content_dim = 768 # ViT-Base feature dim
        
        # 2. Distortion Branch (G) - Swin-Tiny
        self.distortion_encoder = get_distortion_encoder(pretrained=True)
        self.distortion_dim = 768 # Swin-Tiny feature dim (usually 768 at final stage if num_classes=0 and global pool)
        # Wait, Swin-Tiny output dim might be different. 
        # timm swin_tiny_patch4_window7_224 usually outputs 768.
        
        # 3. MI Estimator (M)
        # Input dim is sum of feature dims
        self.mi_estimator = MIEstimator(feature_dim=self.content_dim + self.distortion_dim)
        
        # 4. Regression Head (H)
        self.regressor = nn.Sequential(
            nn.Linear(self.content_dim + self.distortion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x_content, x_distortion):
        """
        Args:
            x_content: [B, T, C, H, W]
            x_distortion: [B, T, C, H, W]
        Returns:
            score: [B, 1]
            feat_c: [B, D_c] (aggregated)
            feat_d: [B, D_d] (aggregated)
        """
        B, T, C, H, W = x_content.shape
        
        # Flatten time for encoder
        # [B*T, C, H, W]
        x_c_flat = x_content.view(B * T, C, H, W)
        x_d_flat = x_distortion.view(B * T, C, H, W)
        
        # Extract features
        # [B*T, D]
        feat_c_all = self.content_encoder(x_c_flat)
        feat_d_all = self.distortion_encoder(x_d_flat)
        
        # Reshape back to [B, T, D]
        feat_c_seq = feat_c_all.view(B, T, -1)
        feat_d_seq = feat_d_all.view(B, T, -1)
        
        # Temporal Pooling (Mean)
        # [B, D]
        feat_c = feat_c_seq.mean(dim=1)
        feat_d = feat_d_seq.mean(dim=1)
        
        # Regression
        combined = torch.cat([feat_c, feat_d], dim=1)
        score = self.regressor(combined)
        
        return score, feat_c, feat_d
