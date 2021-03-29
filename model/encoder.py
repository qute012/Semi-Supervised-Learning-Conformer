import torch.nn as nn

from activation import Swish

class FeedForwardModule(nn.Module):
    def __init__(
            self,
            in_dim,
            expansion_factor=4,
            dropout_p=0.1,
    ):
        super().__init__()

        expansion_dim = in_dim * expansion_factor

        self.modules = nn.Sequential(
            nn.Linear(in_dim, expansion_dim),
            nn.LayerNorm(expansion_dim),
            Swish(),
            nn.Dropout(dropout_p),
            nn.Linear(expansion_dim, in_dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        return x + self.modules(x)