import numpy as np
from scipy import linalg as la
import torch
from torch import nn
from torch.nn import functional as F

logabs = lambda x: torch.log(torch.abs(x))

class ActNorm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, seq_len = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = seq_len * torch.sum(log_abs)

        return self.scale * (input + self.loc), logdet

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConvLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        weight = np.random.randn(in_channels, in_channels)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, seq_len = input.shape

        weight = self.calc_weight()

        out = F.conv1d(input, weight)
        logdet = seq_len * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2)

    def reverse(self, output):
        weight = self.calc_weight()

        weight = weight.squeeze()
        weight = weight.inverse().unsqueeze(2)

        return F.conv1d(output, weight)


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


class NetBlock(nn.Module):
    def __init__(self, in_channels, in_seq_len, hidden_sizes: list[int]=[1, 2]) -> None:
        super().__init__()

        self.in_channels = in_channels
        layers = [nn.Linear(in_channels * in_seq_len, hidden_sizes[0]),
                  nn.ReLU()]
        for inp, out in zip(hidden_sizes, hidden_sizes[1:]):
            layers.append(nn.Linear(inp, out))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(out, in_channels*in_seq_len))
        layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        B, C, S = x.shape
        assert C == self.in_channels
        x = x.reshape((B, -1))

        x = self.layers(x)

        x = x.reshape((B, C, S))

        return x


class AffineCoupling(nn.Module):
    def __init__(self, in_channels, in_seq, affine=True):
        super().__init__()

        # self.affine = affine
        in_channels //= 2
        self.net = NetBlock(in_channels, in_seq)

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        # if self.affine:
        #     log_s, t = self.net(in_a).chunk(2, 1)
        #     s = F.sigmoid(log_s + 2)
        #     out_b = (in_b + t) * s

        #     logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
        # else:
        net_out = self.net(in_a)
        out_b = in_b + net_out
        logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        # if self.affine:
        #     log_s, t = self.net(out_a).chunk(2, 1)
        #     s = F.sigmoid(log_s + 2)
        #     in_b = out_b / s - t
        # else:
        net_out = self.net(out_a)
        in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.norm = ActNorm(in_channels)
        self.inv_conv = InvConvLU(in_channels)
        self.coupling = AffineCoupling(in_channels)

    def forward(self, x: torch.Tensor):
        z, log_det_norm = self.norm(x)
        z, log_det_conv = self.inv_conv(z)
        z, log_det_coup = self.coupling(z)

        log_det = log_det_norm + log_det_conv + \
            log_det_coup if log_det_coup else 0

        return z, log_det

    def reverse(self, z: torch.Tensor):
        x = self.coupling.reverse(z)
        x = self.inv_conv.reverse(x)
        x = self.norm.reverse(x)

        return x


class Patchification(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, permute: bool = False):
        super().__init__()

        self.permute = permute
        self.proj = nn.Conv2d(in_channels, out_channels, 
                              patch_size, patch_size)
        self.inv_proj = nn.ConvTranspose2d(out_channels, in_channels, 
                                           patch_size, patch_size)
        self.patch_size = patch_size
    
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape

        x = self.proj(x)

        x = x.reshape(b, self.proj.out_channels, -1)#h*w // self.patch_size**2)

        if self.permute:
            x = x.permute(0, 2, 1)
        
        return x
    
    def reverse(self, x: torch.Tensor):
        b, c, k = x.shape

        if self.permute:
            x = x.permute(0, 2, 1)
        
        x = x.reshape(b, c, (k * self.patch_size ** 2)**0.5) # assume that the image is square

        x = self.inv_proj(x)

        return x


class FlowFormer(nn.Module):
    def __init__(self, in_channels: int, patch_size: int = 2):
        super().__init__()

        self.patch = Patchification(in_channels, 
                                    out_channels=in_channels, 
                                    patch_size=patch_size)
        self.attention = None

        self.flow = Flow(in_channels)
    
    def forward(self, x: torch.Tensor):
        x = self.patch(x)
        b, c, k = x.shape
        x = x.reshape(b, c*4, k//4)
        x = self.attention(x)
        x = self.flow(x)

        return x

    def reverse(self, z: torch.Tensor):
        x = self.flow.reverse(z)
        x = self.attention.reverse(x)
        b, c4, k4 = x.shape
        x = x.reshape(b, c4//4, k4*4)
        x = self.patch.reverse(x)
        return x