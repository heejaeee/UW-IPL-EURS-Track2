"""RGB-only statistical normalization plus learned color/spatial restoration.

No scene IDs, degradation metadata, or reference image enter forward. Paired GT
is used only by the trainer. This is an independent experiment, not a claim of
NAFNet/HDRNet reproduction or measured Jetson performance.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ColorResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels * 2, 1), nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
            nn.SiLU(), nn.Conv2d(channels * 2, channels, 1))

    def forward(self, x):
        return x + self.net(x)


class StatisticalRestorer(nn.Module):
    def __init__(self, conditioning_size=(192, 256)):
        super().__init__()
        self.conditioning_size = tuple(conditioning_size)
        self.enc0 = nn.Sequential(nn.Conv2d(6, 24, 3, padding=1), nn.SiLU(), ColorResidual(24))
        self.enc1 = nn.Sequential(nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.SiLU(), ColorResidual(32))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.SiLU(), ColorResidual(48))
        self.enc3 = nn.Sequential(nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.SiLU(), ColorResidual(64))
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 48, 48, 1), nn.SiLU(), ColorResidual(48))
        self.dec1 = nn.Sequential(nn.Conv2d(48 + 32, 32, 1), nn.SiLU(), ColorResidual(32))
        self.dec0 = nn.Sequential(nn.Conv2d(32 + 24, 24, 1), nn.SiLU(), ColorResidual(24))
        self.spatial = nn.Conv2d(24, 3, 3, padding=1)
        self.color = nn.Sequential(nn.Conv2d(64 + 6, 64, 1), nn.SiLU(), nn.Conv2d(64, 12, 1))
        nn.init.zeros_(self.spatial.weight); nn.init.zeros_(self.spatial.bias)
        nn.init.zeros_(self.color[-1].weight); nn.init.zeros_(self.color[-1].bias)

    def forward_details(self, raw):
        mean = raw.mean((-2, -1), keepdim=True)
        std = ((raw - mean).square().mean((-2, -1), keepdim=True) + (1 / 255.) ** 2).sqrt()
        standardized = (raw - mean) / std
        small = F.interpolate(torch.cat((raw, standardized), 1), self.conditioning_size,
                              mode='bilinear', align_corners=False)
        e0 = self.enc0(small); e1 = self.enc1(e0); e2 = self.enc2(e1); e3 = self.enc3(e2)
        coefficients = self.color(torch.cat((e3.mean((-2, -1), keepdim=True), mean, std), 1))
        offset = .5 * coefficients[:, :3].tanh()
        matrix = .5 * coefficients[:, 3:].tanh()
        # Explicit nine products avoid a variable-shape batched MatMul in ONNX.
        color_delta = torch.cat([sum(matrix[:, 3 * c + j:3 * c + j + 1] * standardized[:, j:j + 1]
                                     for j in range(3)) for c in range(3)], 1)
        global_rgb = raw + offset + color_delta
        up = lambda x, ref: F.interpolate(x, ref.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat((up(e3, e2), e2), 1))
        d1 = self.dec1(torch.cat((up(d2, e1), e1), 1))
        d0 = self.dec0(torch.cat((up(d1, e0), e0), 1))
        residual = .25 * up(self.spatial(d0).tanh(), raw)
        unclipped = global_rgb + residual
        return unclipped.clamp(0, 1), {'unclipped': unclipped, 'global_rgb': global_rgb,
                                     'offset': offset, 'normalized_color_matrix': matrix}

    def forward(self, raw):
        return self.forward_details(raw)[0]
