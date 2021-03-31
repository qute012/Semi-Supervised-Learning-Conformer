import math
import torch
import torch.nn as nn

from activation import Swish
from attention import RelPositionMultiHeadAttention
from convolution import ConvolutionModule


class FeedForwardModule(nn.Module):
    def __init__(
            self,
            in_dim,
            expansion_factor=4,
            dropout_p=0.1,
    ):
        super(FeedForwardModule, self).__init__()

        expansion_dim = in_dim * expansion_factor

        self.layers = nn.Sequential(
            nn.Linear(in_dim, expansion_dim),
            nn.LayerNorm(expansion_dim),
            Swish(),
            nn.Dropout(dropout_p),
            nn.Linear(expansion_dim, in_dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        return self.layers(x)


class Residual(nn.Module):
    def __init__(
            self,
            *layers
    ):
        super(Residual, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return x + self.layers(x)


class Scale(nn.Module):
    def __init__(
            self,
            *layers,
            scale
    ):
        super(Scale, self).__init__()
        self.scale = scale
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return self.layers(x) * self.scale


class OptionalSequential(nn.Module):
    def __init__(
            self,
            *layers
    ):
        super(OptionalSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *args, **kwargs):
        opt_params = list(kwargs.keys())
        opt_inputs = list(kwargs.values())

        for layer in self.layers:
            params = set(layer.forward.__code__.co_varnames)
            opt_lparams = list(set(opt_params) & params)
            opt_linputs = [opt_inputs[opt_params.index(k)] for k in opt_lparams]
            opt = self.make_dict(opt_lparams, opt_linputs)

            if len(opt) == 0:
                inputs = layer(*inputs)
            else:
                inputs = layer(*inputs, **opt)
        return inputs

    @staticmethod
    def make_dict(keys, values):
        temp_dict = dict()
        for k, v in zip(keys, values):
            temp_dict[k] = v
        return temp_dict


class MHSAParallelInput(nn.Module):
    def __init__(
            self,
            *fn
    ):
        super(MHSAParallelInput, self).__init__()
        self.fn = nn.ModuleList(fn)

    def forward(self, x, mask):
        return x, x, x, self.fn(x), mask


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_length=2000, dropout_p=0.1):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, dim_model, requires_grad=False)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        exp_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * exp_term)
        pe[:, 1::2] = torch.cos(position * exp_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class ConformerBlock(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_heads,
            kernel_size=32,
            conv_expansion_factor=2,
            ffn_expansion_factor=4,
            dropout_p=0.1
    ):
        super(ConformerBlock, self).__init__()

        #TODO: How about separating between MHA from feedforward?
        self.layers = OptionalSequential(
            Residual(
                Scale(FeedForwardModule(hidden_dim, ffn_expansion_factor), 0.5)
            ),
            Residual(
                nn.LayerNorm(hidden_dim),
                MHSAParallelInput(PositionalEncoding(hidden_dim)),
                RelPositionMultiHeadAttention(hidden_dim, num_heads),
                nn.Dropout(dropout_p)
            ),
            Residual(
                ConvolutionModule(hidden_dim, kernel_size, conv_expansion_factor)
            ),
            Residual(
                Scale(FeedForwardModule(hidden_dim, ffn_expansion_factor), 0.5)
            ),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x, mask):
        return self.layers(x, mask=mask)
