import torch
from torch import nn
from .flow import Flow
from .attention import ReversibleAttention
from util import ZeroConv


class Patchification(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, permute: bool = False):
        super().__init__()

        self.permute = permute
        self.proj = nn.Conv2d(in_channels, out_channels, 
                              patch_size, patch_size)
        self.inv_proj = nn.ConvTranspose2d(out_channels, in_channels, 
                                           patch_size, patch_size)
        self.patch_size = patch_size
    
    def get_seq_len(self, img_size: int):
        return int((img_size // self.patch_size)**2)
    
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape

        x = self.proj(x)

        x = x.reshape(b, self.proj.out_channels, self.get_seq_len(h))

        if self.permute:
            x = x.permute(0, 2, 1)
        
        return x
    
    def reverse(self, x: torch.Tensor):
        b, c, k = x.shape

        if self.permute:
            x = x.permute(0, 2, 1)
        hw = int(k**0.5) # assume that the image is square
        x = x.reshape(b, c, hw, hw)

        x = self.inv_proj(x)

        return x

from math import log, pi
def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


class Block(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, attention_heads: int) -> None:
        super().__init__()

        self.attention = ReversibleAttention(seq_len, attention_heads)
        self.flow = Flow(in_channels, seq_len)
        self.prior = ZeroConv(in_channels, in_channels * 2)
    
    def forward(self, x: torch.Tensor):
        x = self.attention(x)
        x, log_det = self.flow(x)

        zero = torch.zeros_like(x)
        mean, log_sd = self.prior(zero).chunk(2, 1)
        log_p = gaussian_log_p(x, mean, log_sd)
        log_p = log_p.reshape(x.shape[0], -1).sum(1)

        return x, log_det, log_p

    def reverse(self, out: torch.Tensor):
        x = self.flow.reverse(out)
        x = self.attention.reverse(x)

        return x


class FlowFormer(nn.Module):
    def __init__(self, in_channels: int, img_size: int, num_blocks: int = 2, patch_size: int = 2, attention_heads: int = 4):
        super().__init__()

        self.patch = Patchification(in_channels, 
                                    out_channels=in_channels, 
                                    patch_size=patch_size)
        seq_len = self.patch.get_seq_len(img_size)
        self.pos_embedding = nn.Parameter(torch.empty(1, in_channels, seq_len).normal_(std=0.02)) # From ViT

        seq_len //= 4
        in_channels *= 4
        self.blocks = nn.ModuleList([Block(in_channels, seq_len, attention_heads) for _ in range(num_blocks)])
        
    
    def forward(self, x: torch.Tensor):
        x = self.patch(x)
        x += self.pos_embedding
        b, c, k = x.shape
        x = x.reshape(b, c*4, k//4)
        log_det = 0
        log_p = 0
        for block in self.blocks:
            x, ld, lp = block(x)
            log_det += ld
            log_p += lp

        return log_p, log_det

    def reverse(self, z: torch.Tensor):
        x = z
        for block in reversed(self.blocks):
            x = block.reverse(x)
        b, c4, k4 = x.shape
        x = x.reshape(b, c4//4, k4*4)
        x -= self.pos_embedding
        x = self.patch.reverse(x)
        return x
