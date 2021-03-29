import torch.nn as nn

from activation import GLU, Swish

class PointwiseConv(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class DepthwiseConv1d(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            kernel_size=32,
            padding=0
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x

class Transpose(nn.Module):
    def __init__(
            self,
            dim0,
            dim1,
    ):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1).contiguous()

class ConvolutionModule(nn.Module):
    def __init__(
            self,
            in_dim,
            kernel_size=32,
            expansion_factor=2,
            dropout_p=0.1
    ):
        super().__init__()
        expansion_dim = in_dim * expansion_factor

        self.modules = nn.Sequential(
            nn.LayerNorm(in_dim),
            Transpose(1,0),
            PointwiseConv(in_dim, expansion_dim),
            GLU(dim=-1),
            DepthwiseConv1d(in_dim, in_dim, kernel_size),
            nn.BatchNorm1d(in_dim),
            Swish(),
            PointwiseConv(in_dim, in_dim),
            Transpose(1,0),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        return x + self.modules(x)