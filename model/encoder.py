import torch
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

    @staticmethod
    def make_pad_mask(self, lengths, length_dim=-1):
        if length_dim == 0:
            raise ValueError("length_dim cannot be 0: {}".format(length_dim))

        if not isinstance(lengths, list):
            lengths = lengths.tolist()
        btz = int(len(lengths))

        max_len = int(max(lengths))

        seq_range = torch.arange(0, max_len, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(btz, max_len)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand

        mask = ~mask.unsqueeze(1)
        mask = mask.eq(0)
        return mask

    def forward(self, x, input_length, pad_mask=True):
        x, input_length = self.subsampling(x, input_length)

        if pad_mask:
            mask = self.make_pad_mask(input_length, len(input_length.size()))
        else:
            mask = None

        x, mask = self.layers(x, mask)
        return x, input_length
