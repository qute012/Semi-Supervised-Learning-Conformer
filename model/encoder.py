import torch
import torch.nn as nn

from sub_sample import SubSampling
from block import ConformerBlock


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_length=2000, dropout_p=0.1):
        super().__init__()
        pe = torch.zeros(max_length, dim_model, requires_grad=False)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        exp_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * exp_term)
        pe[:, 1::2] = torch.cos(position * exp_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

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

    def forward(self, x, input_length):
        x, input_length = self.subsampling(x, input_length)
        x = self.layers(x)
        return x, input_length
