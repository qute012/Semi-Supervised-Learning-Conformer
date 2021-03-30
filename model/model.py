import torch
import torch.nn as nn

from encoder import ConformerEncoder


class ConformerForPreTraining(ConformerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def state_dict(self):
        return self.encoder.state_dict()

    @property
    def load_state_dict(self, state_dict, strict=True):
        self.encoder.load_state_dict(state_dict, strict)

    def negative_sampling(self, y, n, count_pad=None):
        raise NotImplementedError

    def forward(self, x, input_length):
        enc_state = super().forward(x, input_length)


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
            pad_id=0
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

        self.fc = nn.Linear(encoder_dim, n_classes)

        self.criterion = nn.CTCLoss(
            blank=pad_id,
            reduction='mean',
            zero_infinity=True
        )

        self.pad_id = pad_id

    def from_pretrained(self, path):
        state_dict = torch.load(path)
        self.encoder.load_state_dict(state_dict['model'])

    def forward(self, inputs, input_length, target, target_length):
        enc_state, input_length = self.encoder(inputs, input_length)
        loss = self.criterion(enc_state.log_softmax(2), target, input_length, target_length)
        return loss, enc_state, input_length

    @torch.no_grad()
    def recognize(self):
        raise NotImplementedError


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
        self.fc = nn.Linear(decoder_dim, n_classes)

        from warprnnt_pytorch import RNNTLoss

        self.criterion = RNNTLoss(
            size_average=True,
            blank_label=pad_id
        )

        self.pad_id = pad_id
        self.eos_id = eos_id
        self.sos_id = sos_id

    def from_pretrained(self, path):
        state_dict = torch.load(path)
        self.encoder.load_state_dict(state_dict['model'])

    def forward(self, inputs, input_length, target):
        enc_state, input_length = self.encoder(inputs, input_length)
        raise NotImplementedError

    @torch.no_grad()
    def recognize(self):
        raise NotImplementedError

    def add_eos(self, inputs):
        padded_eos = torch.zeros(inputs.size(0), 1).fill_(self.eos_id)
        inputs = torch.cat((inputs, padded_eos), dim=1)
        return inputs
