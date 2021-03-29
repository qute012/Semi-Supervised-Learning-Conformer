import torch.nn as nn

from encoder import ConformerEncoder


class ConformerCTC(nn.Module):
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
            dropout_p=0.1,
            pad_id=0,
            eos_id=1,
    ):
        super().__init__()
        self.encoder = ConformerEncoder(
            in_dim=in_dim,
            n_layers=enc_layers,
            hidden_dim=encoder_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            conv_expansion_factor=conv_expansion_factor,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout_p=dropout_p
        )

        self.pad_id = pad_id
        self.eos_id = eos_id

        self.criterion = nn.CTCLoss(
            blank=pad_id,
            eduction='mean',
            zero_infinity=True
        )


    def forward(self, inputs, input_length, target):
        enc_state, input_length = self.encoder(inputs, input_length)

        return enc_state, input_length


class ConformerTransducer(nn.Module):
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
            dropout_p=0.1,
            pad_id=0,
            eos_id=1,
            sos_id=2
    ):
        super().__init__()
        self.encoder = ConformerEncoder(
            in_dim=in_dim,
            n_layers=enc_layers,
            hidden_dim=encoder_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            conv_expansion_factor=conv_expansion_factor,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout_p=dropout_p
        )
        self.decoder = nn.LSTM(
            decoder_dim,
            decoder_dim,
            num_layers=dec_layers
        )

        from warprnnt_pytorch import RNNTLoss

        self.criterion = RNNTLoss(
            size_average=True,
            blank_label=pad_id
        )

        self.pad_id = pad_id
        self.eos_id = eos_id
        self.sos_id = sos_id

    def forward(self, inputs, input_length, target):
        enc_state, input_length = self.encoder(inputs, input_length)

