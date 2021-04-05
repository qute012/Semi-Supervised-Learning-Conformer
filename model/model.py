import random
import torch
import torch.nn as nn

from encoder import ConformerEncoder
from sub_sampling import SubSampling
from ..criterion.losses import ContrastiveLoss


class ConformerForPreTraining(ConformerEncoder):
    def __init__(
            self,
            in_dim,
            encoder_dim=256,
            enc_layers=16,
            num_heads=4,
            kernel_size=32,
            conv_expansion_factor=2,
            ffn_expansion_factor=4,
            dropout_p=0.1,
    ):
        super(ConformerForPreTraining, self).__init__(
            in_dim=in_dim,
            n_layers=enc_layers,
            hidden_dim=encoder_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            conv_expansion_factor=conv_expansion_factor,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout_p=dropout_p
        )

        self.subsampling = SubSampling(in_dim, encoder_dim, dropout_p)
        self.out_proj = nn.Linear(encoder_dim, encoder_dim)
        self.quantization = nn.Linear(encoder_dim, encoder_dim)
        self.criterion = ContrastiveLoss(reduce='sum')

    @property
    def state_dict(self):
        return self.state_dict()

    @property
    def load_state_dict(self, state_dict, strict=True):
        self.load_state_dict(state_dict, strict)

    @classmethod
    def buffered_arange(cls, max):
        if not hasattr(cls.buffered_arange, "buf"):
            cls.buffered_arange.buf = torch.LongTensor()
        if max > cls.buffered_arange.buf.numel():
            cls.buffered_arange.buf.resize_(max)
            torch.arange(max, out=cls.buffered_arange.buf)
        return cls.buffered_arange.buf[:max]

    @torch.no_grad
    def negative_sampling(self, y, n=10, input_length=None):
        bsz, tsz, fsz = y.shape

        y = y.view(-1, fsz)

        high = tsz

        if input_length is None:
            max_seq_len = tsz

            ctx_idxs = self.buffered_arange(tsz) \
                .unsqueeze(-1) \
                .expand(-1, n) \
                .flatten()

            neg_idxs = torch.randint(
                low=0,
                high=max_seq_len,
                size=(bsz, n * tsz)
            )

            neg_idxs[neg_idxs >= ctx_idxs] += 1
        else:
            ctx_idxs = self.buffered_arange(tsz) \
                .unsqueeze(-1) \
                .expand(-1, n) \
                .flatten()

            neg_idxs = []

            for max_seq_len in input_length:
                neg_idx = torch.randint(
                    low=0,
                    high=max_seq_len,
                    size=(1, n * tsz)
                )

                neg_idx[neg_idx >= ctx_idxs] += 1

                neg_idxs.append(neg_idx)

            neg_idxs = torch.stack(neg_idxs)

        if n > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high

        negatives = y[neg_idxs.view(-1)]
        negatives = negatives \
            .view(bsz, tsz, n, fsz) \
            .permute(2, 0, 1, 3)
        return negatives

    def masking(self, x, F=27, T_ratio=0.05):
        bsz, tsz, fsz = x.shape

        f0 = random.randint(0, fsz - F)
        x[:, :, f0:f0 + F] = 0

        T = random.randint(0, int(tsz * T_ratio))
        t0 = random.randint(0, tsz - T)
        x[:, t0:t0 + T, :] = 0
        return x

    def forward(self, x, input_length):
        encoded_features, input_length = self.subsampling(x, input_length)

        y = encoded_features

        x = self.masking(encoded_features)
        context_vector = super().forward(encoded_features, input_length)
        raise NotImplementedError


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
        super(ConformerCTC, self).__init__()
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
        super(ConformerTransducer, self).__init__()
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
