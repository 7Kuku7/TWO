import torch
import torch.nn as nn
import math

class MIEstimator(nn.Module):
    """
    Variational Mutual Information Estimator (Modified for Decoupling).
    Uses a Gaussian approximation q(y|x) to estimate the upper bound of MI.
    """
    def __init__(self, feature_dim):
        super(MIEstimator, self).__init__()
        
        # [修改 1] 输入维度改为 feature_dim (原本是 *2)
        # 也就是只接收 content_feature，尝试预测 distortion_feature
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), # 增加一层以增强表达能力
            nn.ReLU()
        )
        
        # 输出均值 mu
        self.mu_layer = nn.Linear(256, feature_dim)
        # 输出对数方差 logvar
        self.logvar_layer = nn.Linear(256, feature_dim)
        
    def get_params(self, x):
        """
        输入 x (Content)，预测 y (Distortion) 的分布参数
        """
        h = self.net(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        # 限制 logvar 范围防止数值不稳定
        logvar = torch.clamp(logvar, min=-10, max=10) 
        return mu, logvar

    def log_likelihood(self, y, mu, logvar):
        """
        计算高斯分布的 Log Likelihood
        """
        # log N(y|mu, var) = -0.5 * (logvar + log(2pi) + (y-mu)^2/exp(logvar))
        return -0.5 * (logvar + math.log(2 * math.pi) + (y - mu).pow(2) / torch.exp(logvar))

    def forward(self, x, y):
        """
        Args:
            x: Content features [B, D]
            y: Distortion features [B, D]
        Returns:
            loss: scalar
        """
        # [修改 2] 变分估计的核心逻辑
        
        # 步骤 A: 训练 Estimator (让它尽可能预测准确)
        # 注意：这里必须 detach x，否则 Encoder 会为了迎合 Estimator 而改变，这不是我们想要的
        mu, logvar = self.get_params(x.detach())
        
        # 估计器的 Loss 是 Negative Log Likelihood (NLL)
        # 我们希望最大化 Likelihood，即最小化 NLL
        loss_lld = -self.log_likelihood(y, mu, logvar).mean()
        
        # 步骤 B: 训练 Encoder (解耦核心)
        # 这次不 detach x。我们希望 x 更新，使得 Estimator 预测变得**不准** (不确定性增加)
        mu_enc, logvar_enc = self.get_params(x)
        
        # 计算 Variational Upper Bound (VUB)
        # 简单的变分策略：最小化 x 和 y 的互信息 => 最小化 E[log q(y|x)]
        # 也就是让 Encoder 产生的 x 使得 y 的预测概率变低 (熵变大)
        # 这里直接使用 Log Likelihood 作为互信息的近似 (忽略边缘分布 H(Y) 常数项)
        loss_decouple = self.log_likelihood(y, mu_enc, logvar_enc).mean()
        
        # 总 Loss: 
        # loss_lld 用于优化 self.net
        # loss_decouple 用于优化 Encoder (DisNeRF 主干)
        return loss_lld + 0.1 * loss_decouple
