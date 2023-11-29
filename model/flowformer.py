import torch
from torch import nn
from .flow import Flow
from .attention import ReversibleAttention
from .util import ZeroConv, Patchification, gaussian_log_p, gaussian_sample


class AttentionFlowBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = ReversibleAttention(hidden_dim, num_heads)
        self.flow = Flow(in_channels, hidden_dim)
    
    def forward(self, x: torch.Tensor):
        x = self.attention(x)
        x, log_det = self.flow(x)
        return x, log_det
    
    def reverse(self, out: torch.Tensor):
        x = self.flow.reverse(out)
        x = self.attention.reverse(x)

        return x


class Block(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, attention_heads: int, 
                 num_attnflow: int, split: bool = True) -> None:
        super().__init__()
        in_channels *= 4
        hidden_dim //= 4
        self.blocks = nn.ModuleList(
            [AttentionFlowBlock(in_channels, hidden_dim, attention_heads) for _ in range(num_attnflow)]
        )
        self.split = split
        if split:
            in_channels //= 2
        self.prior = ZeroConv(in_channels, in_channels * 2)

        self.out_shape = torch.Size((in_channels, hidden_dim))
    
    def forward(self, x: torch.Tensor):
        b, c, k = x.shape
        x = x.reshape(b, c*4, k//4)

        log_det = 0
        for block in self.blocks:
            x, ld = block(x)
            log_det += ld

        if self.split:
            out, z_new = x.chunk(2, 1)
            mean, log_std = self.prior(out).chunk(2, 1)
        else:
            mean, log_std = self.prior(torch.zeros_like(x)).chunk(2, 1)
            out = z_new = x
        log_p = gaussian_log_p(z_new, mean, log_std)
        log_p = log_p.reshape(x.shape[0], -1).sum(1)

        return out, log_det, log_p

    def reverse(self, out: torch.Tensor, eps: torch.Tensor):
        if self.split:
            mean, log_std = self.prior(out).chunk(2, 1)
            z = gaussian_sample(eps, mean, log_std)
            out = torch.cat((out, z), 1)
        else:
            mean, log_std = self.prior(torch.zeros_like(out)).chunk(2, 1)
            z = gaussian_sample(eps, mean, log_std)
            out = z
                    
        x = out
        for block in reversed(self.blocks):
            x = block.reverse(x)
        
        b, c, k = x.shape
        x = x.reshape(b, c//4, k*4)
        return x


class FlowFormer(nn.Module):
    def __init__(self, in_channels: int, img_size: int, num_blocks: int = 2, patch_size: int = 2, attention_heads: int = 4, num_attnflow: int = 2):
        super().__init__()

        self.patch = Patchification(in_channels, 
                                    patch_size=patch_size)
        hidden_dim = img_size**2
        self.pos_embedding = nn.Parameter(torch.empty(1, in_channels, hidden_dim).normal_(std=0.02)) # From ViT

        self.blocks = []
        for _ in range(num_blocks - 1):
            self.blocks.append(Block(in_channels, hidden_dim, attention_heads, num_attnflow=num_attnflow))
            in_channels, hidden_dim = self.blocks[-1].out_shape
        self.blocks.append(Block(in_channels, hidden_dim, attention_heads, num_attnflow=num_attnflow, split=False))
        self.blocks = nn.ModuleList(self.blocks)
        
    
    def forward(self, x: torch.Tensor):
        x = self.patch(x)
        x += self.pos_embedding

        log_det = 0
        log_p = 0
        for block in self.blocks:
            x, ld, lp = block(x)
            log_det += ld
            log_p += lp

        return log_p, log_det

    def reverse(self, z_list: list[torch.Tensor]):
        x = z_list[-1]
        for block, z in reversed(list(zip(self.blocks, z_list))):
            x = block.reverse(x, z)
        x -= self.pos_embedding
        x = self.patch.reverse(x)
        return x

    def sample(self, num_samples: int, temp: float):
        z_list = [torch.randn(num_samples, *block.out_shape).to(self.pos_embedding.device) * temp for block in self.blocks]
        samples = self.reverse(z_list)
        return samples