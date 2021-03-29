import torch.nn as nn

from activation import Swish
from attention import RelPositionMultiHeadAttention
from convolution import ConvolutionModule

class FeedForwardModule(nn.Module):
    def __init__(
            self,
            in_dim,
            expansion_factor=4,
            dropout_p=0.1,
    ):
        super().__init__()

        expansion_dim = in_dim * expansion_factor

        self.layers = nn.Sequential(
            nn.Linear(in_dim, expansion_dim),
            nn.LayerNorm(expansion_dim),
            Swish(),
            nn.Dropout(dropout_p),
            nn.Linear(expansion_dim, in_dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        return x + self.layers(x)

class Residual(nn.Module):
    def __init__(
            self,
            *layers
    ):
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return x + self.layers(x)

class Scale(nn.Module):
    def __init__(
            self,
            *layers,
            scale
    ):
        self.scale = scale
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return self.layers(x) * self.scale

class ConformerBlock(nn.Module):
    def __init__(
            self,
            hidden_dim,
            dropout_p=0.1
    ):
        super(ConformerBlock, self).__init__()

        self.layers = nn.Sequential(
            Residual(
                Scale(FeedForwardModule(hidden_dim), 0.5)
            ),
            Residual(
                nn.LayerNorm(hidden_dim),
                RelPositionMultiHeadAttention,
                nn.Dropout(dropout_p)
            ),
            Residual(
                ConvolutionModule(hidden_dim)
            ),
            Residual(
                Scale(FeedForwardModule(hidden_dim), 0.5)
            ),
            nn.LayerNorm(hidden_dim)
        )


