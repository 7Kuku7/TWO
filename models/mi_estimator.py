import torch
import torch.nn as nn
import torch.nn.functional as F

class MIEstimator(nn.Module):
    """
    MINE: Mutual Information Neural Estimation
    Reference: Belghazi et al., ICML 2018
    作用：通过统计网络 T_theta 区分联合分布 (Joint) 和边缘分布 (Marginal)
    """
    def __init__(self, feature_dim, hidden_dim=128):
        super(MIEstimator, self).__init__()
        # 这里的 feature_dim * 2 是因为输入是 Z_geo 和 Z_pho 的拼接
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # MINE 的输出是一个标量分数 T
        )

    def forward(self, x, y):
        """
        x: Geometry features [Batch, Dim]
        y: Photometry features [Batch, Dim]
        """
        # 1. 联合分布 (Joint Distribution) - 对应图中的 Top Path
        # 直接拼接：真实的 (x, y) 对
        joint = torch.cat([x, y], dim=1)
        
        # 2. 边缘分布 (Marginal Distribution) - 对应图中的 Bottom Path
        # 关键步骤：Shuffle (乱序)！对应图中的“交叉线”
        # 保持 x 不变，打乱 y 的顺序，制造假的 (x, y_tilde) 对
        y_shuffled = y[torch.randperm(y.size(0))] 
        marginal = torch.cat([x, y_shuffled], dim=1)
        
        # 3. 统计网络打分
        t_joint = self.net(joint)       # T(x, y)
        t_marginal = self.net(marginal) # T(x, y_tilde)
        
        # 4. 计算 MINE Loss (逼近 KL 散度)
        # 理论公式: Loss = - ( E[T_joint] - log(E[exp(T_marginal)]) )
        # 加上 1e-6 是为了防止 log(0)
        loss = -(torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)) + 1e-6))
        
        return loss