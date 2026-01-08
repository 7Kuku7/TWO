import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_content_encoder, get_distortion_encoder
from .mi_estimator import MIEstimator

class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion using Gated Attention.
    Learns to dynamically weight Content and Distortion features.
    """
    def __init__(self, feature_dim=768):
        super().__init__()
        # Attention weights
        self.attn_fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 2), # Output 2 weights (alpha, beta)
            nn.Softmax(dim=1)
        )
        
    def forward(self, feat_c, feat_d):
        # feat_c, feat_d: [B, D]
        combined = torch.cat([feat_c, feat_d], dim=1)
        weights = self.attn_fc(combined) # [B, 2]
        
        alpha = weights[:, 0].unsqueeze(1) # Weight for content
        beta = weights[:, 1].unsqueeze(1)  # Weight for distortion
        
        # Weighted fusion
        # We still concat them, but weighted, or sum them?
        # Concatenation preserves more info. Let's weight them before concat.
        # Or better: Fused = alpha * C + beta * D (if dimensions match)
        # But C and D represent different things.
        # Let's use the weights to scale the features before concatenation.
        
        feat_c_weighted = feat_c * (1 + alpha) # Residual-like scaling
        feat_d_weighted = feat_d * (1 + beta)
        
        return torch.cat([feat_c_weighted, feat_d_weighted], dim=1)

class DisNeRFQA_Advanced(nn.Module):
    def __init__(self, num_subscores=4, use_fusion=True):
        super().__init__()
        self.use_fusion = use_fusion
        
        # 1. Backbones
        self.content_encoder = get_content_encoder(pretrained=True)
        self.distortion_encoder = get_distortion_encoder(pretrained=True)
        
        # Feature dimensions
        self.feat_dim = 768 # ViT-Base and Swin-Tiny (projected)
        
        # 2. MI Estimator
        self.mi_estimator = MIEstimator(self.feat_dim)
        
        # 3. Adaptive Fusion (Innovation)
        if self.use_fusion:
            self.fusion = AdaptiveFeatureFusion(self.feat_dim)
        
        # 4. Heads
        # Main Quality Regressor (0-1)
        # If fusion, input is 2*D (weighted). If no fusion, input is 2*D (concat).
        # Dimensions are same, just logic differs.
        self.regressor = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
        
        # Auxiliary Sub-score Regressor (Multi-task Innovation)
        # Predicts: [Discomfort, Blur, Lighting, Artifacts] (normalized 0-1)
        self.num_subscores = num_subscores
        self.subscore_head = nn.Sequential(
            nn.Linear(self.feat_dim, 256), # Uses distortion features
            nn.ReLU(),
            nn.Linear(256, num_subscores),
            nn.Sigmoid() # Assuming subscores are also normalized to 0-1
        )
        
        # Contrastive Projectors
        self.proj_c = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.proj_d = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x_content, x_distortion):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x_content.shape
        
        # --- Feature Extraction ---
        x_c_flat = x_content.view(b * t, c, h, w)
        x_d_flat = x_distortion.view(b * t, c, h, w)
        
        # Content Branch
        feat_c_raw = self.content_encoder.forward_features(x_c_flat)
        if hasattr(self.content_encoder, 'global_pool'):
             feat_c_seq = self.content_encoder.forward_head(feat_c_raw, pre_logits=True)
        else:
             feat_c_seq = feat_c_raw[:, 0]
             
        # Distortion Branch
        feat_d_raw = self.distortion_encoder.forward_features(x_d_flat)
        feat_d_seq = feat_d_raw.mean(dim=[1, 2])
        
        # Reshape back
        feat_c_seq = feat_c_seq.view(b, t, -1)
        feat_d_seq = feat_d_seq.view(b, t, -1)
        
        # Temporal Pooling
        feat_c = feat_c_seq.mean(dim=1)
        feat_d = feat_d_seq.mean(dim=1)
        
        # --- Adaptive Fusion ---
        if self.use_fusion:
            feat_fused = self.fusion(feat_c, feat_d)
        else:
            # Simple Concatenation (Ablation)
            feat_fused = torch.cat([feat_c, feat_d], dim=1)
        
        # --- Predictions ---
        # 1. Quality Score
        score = self.regressor(feat_fused)
        
        # 2. Sub-scores (Auxiliary)
        sub_scores = self.subscore_head(feat_d)
        
        # 3. Projections
        proj_c = self.proj_c(feat_c)
        proj_d = self.proj_d(feat_d)
        
        return score, sub_scores, proj_c, proj_d, feat_c, feat_d

class MultiTaskLoss(nn.Module):
    # 基于同方差不确定性(Homoscedastic Uncertainty)的自适应多任务Loss权重
    # Reference: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018.
    def __init__(self, num_tasks=4):
        super(MultiTaskLoss, self).__init__()
        # log_vars 是可学习的参数 (log(sigma^2))
        # 初始化为0，即初始权重为 1.0
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, input_losses):
        # input_losses: list of losses [L_reg, L_rank, L_mi, L_sub]
        # 确保输入是列表
        loss_sum = 0
        for i, loss in enumerate(input_losses):
            # 核心公式: L = (1 / 2*sigma^2) * L_i + log(sigma)
            # log_vars[i] = log(sigma^2)
            precision = torch.exp(-self.log_vars[i])
            loss_sum += 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return loss_sum