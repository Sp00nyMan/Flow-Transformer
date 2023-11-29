from math import log, pi
import torch
import torch.nn as nn
from torch.nn import functional as F


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)
def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class ZeroConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, 3)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class Patchification(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, permute: bool = False):
        super().__init__()

        self.permute = permute
        out_channels = in_channels * patch_size**2 # ensures no information loss
        self.proj = nn.Conv2d(in_channels, out_channels, 
                              patch_size, patch_size)
        self.inv_proj = nn.ConvTranspose2d(out_channels, in_channels, 
                                           patch_size, patch_size)
        self.patch_size = patch_size
        
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape

        x = self.proj(x)
        x = x.reshape(b, c, h*w)
        
        return x
    
    def reverse(self, x: torch.Tensor):
        b, c, k = x.shape

        channel_scale = self.patch_size ** 2
        hw = int((k // channel_scale)**0.5)

        x = x.reshape(b, c * channel_scale, hw, hw)

        x = self.inv_proj(x)

        return x