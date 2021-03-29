import torch
import torch.nn as nn

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

class ConvolutionModule(nn.Module):
    def __init__(
            self
    ):
        super().__init__()

    def forward(self):