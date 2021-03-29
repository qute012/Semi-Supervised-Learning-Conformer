import torch.nn as nn

from sub_sample import SubSampling
from block import ConformerBlock


class ConformerEncoder(nn.Module):
    def __init__(
            self,
            in_dim,
            n_layers,
            hidden_dim,
            num_heads,
            kernel_size=32,
            conv_expansion_factor=2,
            ffn_expansion_factor=4,
            dropout_p=0.1
    ):
        super().__init__()

        self.subsampling = SubSampling(in_dim, hidden_dim, dropout_p)
        self.blocks = []
        for _ in range(n_layers):
            self.layers.append(
                ConformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    conv_expansion_factor=conv_expansion_factor,
                    ffn_expansion_factor=ffn_expansion_factor,
                    dropout_p=dropout_p
                )
            )
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = self.subsampling(x)
        x = self.layers(x)
        return x
