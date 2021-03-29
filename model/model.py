import torch.nn as nn

class Conformer(nn.Module):
    def __init__(
            self,
            in_dim,
            n_blocks,
            hidden_dim,
            num_heads,
            kernel_size=32,
            conv_expansion_factor=2,
            ffn_expansion_factor=4,
            n_layers=1,
            dropout_p=0.1
    ):
        super().__init__()

    def forward(self):
