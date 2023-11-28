from math import log
import torch
from tqdm import tqdm

from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F


from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

class MetricsCalculator:
    def __init__(self, device) -> None:
        self.device = device
        self.fid = FrechetInceptionDistance(reset_real_features=False, normalize=True).to(device)
        self.kid = KernelInceptionDistance(reset_real_features=False, subset_size=4, normalize=True).to(device)

    def initialize(self, real_data: DataLoader, max_samples=1000):
        for batch, _ in tqdm(real_data, total=max_samples//real_data.batch_size):
            batch = batch.to(self.device)
            if max_samples <= 0:
                break
            batch = batch.detach()
            self.fid.update(batch, True)
            self.kid.update(batch, True)
            max_samples -= batch.shape[0]

    def compute(self, generated_samples: Tensor):
        self.fid.reset()
        self.kid.reset()

        generated_samples = generated_samples.to(self.device)

        self.fid.update(generated_samples, False)
        self.kid.update(generated_samples, False)

        fid = self.fid.compute()
        kid_m, kid_s = self.kid.compute()

        metrics_dict = {
            "FID": fid.item(),
            "KID_mean": kid_m.item(),
            "KID_std": kid_s.item(),

        }
        return metrics_dict


class GlowLoss(nn.Module):
    def __init__(self, input_channels, input_size, n_bins) -> None:
        super(GlowLoss, self).__init__()
        self.n_pixels = input_size ** 2 * input_channels
        self.base_loss = -log(n_bins) * self.n_pixels
    
    def forward(self, log_p, log_det):
        log_det = log_det.mean()
        loss = self.base_loss + log_p + log_det
        return (
            (-loss / (log(2) * self.n_pixels)).mean(), 
            (log_p / (log(2) * self.n_pixels)).mean(), 
            (log_det / (log(2) * self.n_pixels)).mean(),
        )


def preprocess_images(images: Tensor, device, n_bits):
    images = images.to(device)

    images *= 255
    if n_bits < 8:
        images = torch.floor(images / 2 ** (8 - n_bits))
    n_bins = 2**n_bits
    images /= n_bins - 0.5
    images += torch.rand_like(images) / n_bins

    return images