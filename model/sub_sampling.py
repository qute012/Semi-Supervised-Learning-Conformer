import torch.nn as nn


class SubSampling(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            dropout_p=0.1,
    ):
        super(SubSampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_dim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, 2),
            nn.ReLU()
        )

        # 0 padding, (I-K+S)//S
        subsample_size = out_dim * self._subsampling_size(in_dim)
        self.w = nn.Sequential(
            nn.Linear(subsample_size, out_dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, x, input_length=None):
        x = x.unsqueeze(1)
        x = self.conv(x)
        bsz, csz, tsz, fsz = x.shape
        x = self.w(x.transpose(1, 2).view(bsz, tsz, csz * fsz))

        input_length = self._subsampling_size(input_length)
        return x, input_length

    def _subsampling_size(self, size):
        return ((size - 1) // 2 - 1) // 2