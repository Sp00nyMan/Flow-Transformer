import torch
from torch import nn
from flow import Flow


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
    def __init__(self, in_channels: int, img_size: int, patch_size: int = 2):
        super().__init__()

        self.patch = Patchification(in_channels, 
                                    out_channels=in_channels, 
                                    patch_size=patch_size)
        self.attention = None

        self.flow = Flow(in_channels*4, self.patch.get_seq_len(img_size)//4)
    
    def forward(self, x: torch.Tensor):
        x = self.patch(x)
        b, c, k = x.shape
        x = x.reshape(b, c*4, k//4)
        # x = self.attention(x)
        x, log_det = self.flow(x)

        return x, log_det

    def reverse(self, z: torch.Tensor):
        x = self.flow.reverse(z)
        # x = self.attention.reverse(x)
        b, c4, k4 = x.shape
        x = x.reshape(b, c4//4, k4*4)
        x = self.patch.reverse(x)
        return x
