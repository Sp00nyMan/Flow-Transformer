import torch
from torch import nn
from .flow import Flow
from .attention import ReversibleAttention


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
        self.blocks = [(ReversibleAttention(seq_len, attention_heads), Flow(in_channels, seq_len)) for _ in range(num_blocks)]
        
    
    def forward(self, x: torch.Tensor):
        x = self.patch(x)
        x += self.pos_embedding
        b, c, k = x.shape
        x = x.reshape(b, c*4, k//4)
        log_det = 0
        for attn, flow in self.blocks:
            x = attn(x)
            x, ld = flow(x)
            log_det += ld

        return x, log_det

    def reverse(self, z: torch.Tensor):
        x = z
        for attn, flow in reversed(self.blocks):
            x = flow.reverse(x)
            x = attn.reverse(x)
        b, c4, k4 = x.shape
        x = x.reshape(b, c4//4, k4*4)
        x -= self.pos_embedding
        x = self.patch.reverse(x)
        return x
