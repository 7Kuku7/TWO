import torch
import torch.nn as nn

class MIEstimator(nn.Module):
    """
    Variational Mutual Information Estimator.
    Estimates the conditional distribution q(y|x) or joint distribution parameters
    to calculate the variational upper bound of MI.
    Based on DisPA design.
    """
    def __init__(self, feature_dim):
        super().__init__()
        # Input is concatenation of content and distortion features
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # Output mean and log_var
        )
        
    def forward(self, x, y):
        """
        Args:
            x: Content features [B, D]
            y: Distortion features [B, D]
        Returns:
            mu: Mean of the variational distribution
            logvar: Log variance
        """
        # Concatenate features
        combined = torch.cat([x, y], dim=1)
        out = self.net(combined)
        mu, logvar = out.chunk(2, dim=1)
        return mu, logvar
