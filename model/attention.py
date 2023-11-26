# Based on https://arxiv.org/pdf/2001.04451.pdf
# Implementation from https://github.com/lucidrains/reformer-pytorch
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, x: torch.Tensor):
        x, _ = self.attention(x, x, x, need_weights=False)
        return x


class ReversibleAttention(nn.Module):
    def __init__(self, seq_len: int, num_heads=2):
        super().__init__()

        self.attention = Attention(seq_len, num_heads) # F
        self.feedforward = nn.Linear(seq_len, seq_len) # G

    def forward(self, x: torch.Tensor):
        x1, x2 = x.chunk(2, 1)

        y1 = x1 + self.attention(x2)
        y2 = x2 + self.feedforward(y1)

        return torch.cat((y1, y2), 1)

    def reverse(self, y: torch.Tensor):
        y1, y2 = y.chunk(2, 1)

        x2 = y2 - self.feedforward(y1)
        x1 = y1 - self.attention(x2)

        return torch.cat((x1, x2), 1)
