"""C2-small: deployable variant of StatisticalRestorerV3.

Two changes forced by the Track-2 rules check (export_official04_v3.py):
1. The percentile statistics are computed with a fixed 256-level CDF instead of
   torch.sort (ONNX TopK with K=16384 is not supported by TensorRT, limit 3840).
2. The condition MLP is shrunk (hidden 96) so that the restorer stays ~158k
   parameters and the full candidate fits the 1,000,000-initializer cap with
   the preserved semantic path (804k).
Everything else (grid, inverse blend, residual path) is identical to v3.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .statistical_restorer_v3 import StatisticalRestorerV3, condition_grid, PCT


class StatisticalRestorerV3S(StatisticalRestorerV3):
    def __init__(self, conditioning_size=(192, 256), stat_size=(128, 128), initial_alpha_logit=-2., grid=None, hidden=96, levels=256):
        super().__init__(conditioning_size, stat_size, initial_alpha_logit, grid)
        features = 3 * len(PCT) + 6 + 3 + 3 + 64 + 6
        self.condition = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                       nn.Linear(hidden, len(self.keys) + 3))
        nn.init.zeros_(self.condition[-1].weight)
        with torch.no_grad():
            self.condition[-1].bias.zero_(); self.condition[-1].bias[len(self.keys):] = initial_alpha_logit
        self.register_buffer('levels', torch.linspace(0, 1, levels))
        self.register_buffer('pct_fraction', torch.tensor([p / 100 for p in PCT]))

    def statistics(self, raw):
        small = F.interpolate(raw, self.stat_size, mode='bilinear', align_corners=False).flatten(2)  # B,3,N
        cdf = (small[:, :, :, None] <= self.levels).float().mean(2)  # B,3,L
        # percentile = fraction of levels whose CDF is below p (8-bit quantised quantile)
        pct = (cdf[:, :, None, :] < self.pct_fraction[None, None, :, None]).float().sum(-1) / (self.levels.numel() - 1)  # B,3,9
        mean = small.mean(-1); std = small.std(-1)
        logm = (mean + 1e-3).log()
        ratios = torch.stack((logm[:, 0] - logm[:, 1], logm[:, 1] - logm[:, 2], logm[:, 0] - logm[:, 2]), 1)
        rng = (pct[:, :, -1] - pct[:, :, 0] + 1e-3).log()
        return torch.cat((pct.flatten(1), mean, std, ratios, rng), 1)
