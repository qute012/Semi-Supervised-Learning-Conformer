import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_heads,
            dropout_p=0,
    ):
        super(MultiHeadAttention, self).__init__()
        self.d_k = hidden_dim//num_heads
        self.num_heads = num_heads
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward_qkv(self, q, k, v):
        btz = q.size(0)
        q = q.view(btz, -1, self.num_heads, self.d_k)
        k = k.view(btz, -1, self.num_heads, self.d_k)
        v = v.view(btz, -1, self.num_heads, self.d_k)

        q = self.w_q(q).transpose(1, 2)
        k = self.w_k(k).transpose(1, 2)
        v = self.w_v(v).transpose(1, 2)
        return q, k, v

    def forward_attention(self, v, scores, mask):
        btz = v.size(0)

        attn_dist = torch.softmax(scores, dim=-1)
        if mask is not None:
            attn_dist = attn_dist.masked_fill(mask, 1e-9)

        attn_dist = self.dropout(attn_dist)
        attn_v = torch.matul(attn_dist, v)
        attn_v = attn_v.tranpose(1,2).contiguous().view(btz, -1, self.num_heads*self.d_k)

        return self.out_proj(attn_v)

    def forward(self, q, k, v, mask=None):
        q,k,v = self.forward_qkv(q, k, v)
        scores = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    def __init__(
            self,
            hidden_dim,
            num_heads,
            dropout_p=0
    ):
        super(RelPositionMultiHeadAttention, self).__init__(hidden_dim, num_heads, dropout_p)
        self.w_pos = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.d_k))
        torch.nn.init.xavier_uniform(self.pos_bias_u)
        torch.nn.init.xavier_uniform(self.pos_bias_v)

    def rel_shift(self, p, zero_triu=False):
        zero_pad = torch.zeros_like(p[:,:,:,0])
        x_padded = torch.cat([zero_pad,p], dim=-1)

        x_padded = x_padded.view(*p.shape[:2], p.size(3)+1, p.size(2))
        p = x_padded[:, :, 1:].view_as(p)

        if zero_triu:
            ones = torch.ones_like(p[0,0,:,:])
            p = p * torch.tril(ones, p.size(3) - p.size(2))[None,None,:,:]
        return p

    def forward(self, q, k, v, pos, mask=None):
        btz = q.size(0)

        q, k, v = self.forward_qkv(q, k, v)
        q = q.transpose(1,2)

        p = self.w_pos(pos).view(btz, -1, self.num_heads, self.d_k)
        p = p.transpose(1,2)

        q_with_bias_u = (q+self.pos_bias_u).transpose(1,2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2,-1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2,-1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)