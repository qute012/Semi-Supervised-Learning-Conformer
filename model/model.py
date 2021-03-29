import torch.nn as nn

class Conformer(nn.Module):
    def __init__(
            self,
            n_classes,
            in_dim,
            encoder_dim=256,
            enc_layers=16,
            num_heads=4,
            kernel_size=32,
            conv_expansion_factor=2,
            ffn_expansion_factor=4,
            decoder_dim=640,
            dec_layers=1,
            dropout_p=0.1
    ):
        super().__init__()

    def forward(self):
