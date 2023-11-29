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


class NetBlock(nn.Module):
    def __init__(self, in_channels, in_seq_len, hidden_sizes: list[int]=[256, 256]) -> None:
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
    def __init__(self, in_channels, in_seq, affine=False):
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
    def __init__(self, in_channels: int, seq_len: int) -> None:
        super().__init__()

        self.norm = ActNorm(in_channels)
        self.inv_conv = InvConvLU(in_channels)
        self.coupling = AffineCoupling(in_channels, seq_len)
        # TODO potentially add another ActNorm

    def forward(self, x: torch.Tensor):
        z, log_det_norm = self.norm(x)
        z, log_det_conv = self.inv_conv(z)
        z, log_det_coup = self.coupling(z)

        log_det = log_det_norm + log_det_conv 
        if log_det_coup:
            log_det += log_det_coup

        return z, log_det

    def reverse(self, z: torch.Tensor):
        x = self.coupling.reverse(z)
        x = self.inv_conv.reverse(x)
        x = self.norm.reverse(x)

        return x