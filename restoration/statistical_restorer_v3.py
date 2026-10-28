"""Experiment C2: restorer with a condition classifier over the TRAIN-fitted degradation grid.

Diagnostic (condition_classifier_diagnostic.json): a small MLP on global colour
percentiles identifies the (water, depth) condition well enough that inverting
with the probability-weighted TRAIN-mean parameters reaches ~31dB on water1-3
(the v1/v2 learned path: ~29dB) while heavy waters still favour the learned
prior. V3 therefore predicts a distribution over the 192 TRAIN-fitted
conditions from percentile statistics + the pooled CNN features, takes the
expected (T, O), inverts, and blends with the learned colour path through a
per-channel alpha. The grid parameters are fixed buffers derived only from the
16 TRAIN scenes. Inference receives raw RGB only.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .statistical_restorer import StatisticalRestorer

RUN = Path(__file__).resolve().parents[1] / 'test/runs/official04_20261024'
PCT = [0.5, 2, 10, 30, 50, 70, 90, 98, 99.5]


def condition_grid(fits_path=RUN / 'train_degradation_bank/all_train_fits.jsonl', minimum_psnr=28.):
    groups = {}
    for line in Path(fits_path).read_text().splitlines():
        r = json.loads(line)
        if r['train_forward_fit_psnr'] >= minimum_psnr:
            groups.setdefault((r['water'], r['depth']), []).append((r['transmission'], r['offset']))
    keys = sorted(groups)
    t = torch.tensor([[sum(v[0][c] for v in groups[k]) / len(groups[k]) for c in range(3)] for k in keys])
    o = torch.tensor([[sum(v[1][c] for v in groups[k]) / len(groups[k]) for c in range(3)] for k in keys])
    return keys, t, o


class StatisticalRestorerV3(StatisticalRestorer):
    def __init__(self, conditioning_size=(192, 256), stat_size=(128, 128), initial_alpha_logit=-2., grid=None):
        super().__init__(conditioning_size)
        keys, t, o = grid if grid is not None else condition_grid()
        self.keys = keys; self.stat_size = tuple(stat_size)
        self.register_buffer('grid_t', t.clamp_min(1e-3)); self.register_buffer('grid_o', o)
        n = self.stat_size[0] * self.stat_size[1]
        self.register_buffer('pct_index', torch.tensor([min(n - 1, max(0, round(p / 100 * (n - 1)))) for p in PCT]))
        features = 3 * len(PCT) + 6 + 3 + 3 + 64 + 6
        self.condition = nn.Sequential(nn.Linear(features, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                       nn.Linear(256, len(keys) + 3))
        nn.init.zeros_(self.condition[-1].weight)
        with torch.no_grad():
            self.condition[-1].bias.zero_(); self.condition[-1].bias[len(keys):] = initial_alpha_logit
        # Feature normalisation is fixed from the first batch statistics the trainer supplies (see set_normalization).
        self.register_buffer('feature_mean', torch.zeros(features - 64 - 6)); self.register_buffer('feature_std', torch.ones(features - 64 - 6))

    def statistics(self, raw):
        small = F.interpolate(raw, self.stat_size, mode='bilinear', align_corners=False).flatten(2)  # B,3,N
        sorted_values, _ = small.sort(-1)
        pct = sorted_values[:, :, self.pct_index]  # B,3,9
        mean = small.mean(-1); std = small.std(-1)
        logm = (mean + 1e-3).log()
        ratios = torch.stack((logm[:, 0] - logm[:, 1], logm[:, 1] - logm[:, 2], logm[:, 0] - logm[:, 2]), 1)
        rng = (pct[:, :, -1] - pct[:, :, 0] + 1e-3).log()
        return torch.cat((pct.flatten(1), mean, std, ratios, rng), 1)

    @torch.no_grad()
    def set_normalization(self, raw_batch):
        stats = self.statistics(raw_batch)
        self.feature_mean.copy_(stats.mean(0)); self.feature_std.copy_(stats.std(0) + 1e-3)

    def forward_details(self, raw):
        mean = raw.mean((-2, -1), keepdim=True)
        std = ((raw - mean).square().mean((-2, -1), keepdim=True) + (1 / 255.) ** 2).sqrt()
        standardized = (raw - mean) / std
        small = F.interpolate(torch.cat((raw, standardized), 1), self.conditioning_size, mode='bilinear', align_corners=False)
        e0 = self.enc0(small); e1 = self.enc1(e0); e2 = self.enc2(e1); e3 = self.enc3(e2)
        pooled = torch.cat((e3.mean((-2, -1), keepdim=True), mean, std), 1)
        coefficients = self.color(pooled)
        offset = .5 * coefficients[:, :3].tanh()
        matrix = .5 * coefficients[:, 3:].tanh()
        color_delta = torch.cat([sum(matrix[:, 3 * c + j:3 * c + j + 1] * standardized[:, j:j + 1]
                                     for j in range(3)) for c in range(3)], 1)
        learned_rgb = raw + offset + color_delta
        stats = (self.statistics(raw) - self.feature_mean) / self.feature_std
        head = self.condition(torch.cat((stats, pooled.flatten(1)), 1))
        logits = head[:, :len(self.keys)]; alpha = head[:, len(self.keys):].sigmoid()
        prob = logits.softmax(1)
        t = prob @ self.grid_t; o = prob @ self.grid_o  # B,3 expected parameters
        inverse_rgb = (raw - o[:, :, None, None]) / t[:, :, None, None]
        a = alpha[:, :, None, None]
        global_rgb = a * inverse_rgb + (1 - a) * learned_rgb
        up = lambda x, ref: F.interpolate(x, ref.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat((up(e3, e2), e2), 1))
        d1 = self.dec1(torch.cat((up(d2, e1), e1), 1))
        d0 = self.dec0(torch.cat((up(d1, e0), e0), 1))
        residual = .25 * up(self.spatial(d0).tanh(), raw)
        unclipped = global_rgb + residual
        return unclipped.clamp(0, 1), {'unclipped': unclipped, 'global_rgb': global_rgb, 'learned_rgb': learned_rgb,
                                     'inverse_rgb': inverse_rgb, 'condition_logits': logits, 'expected_t': t, 'expected_o': o,
                                     'alpha': alpha, 'offset': offset, 'normalized_color_matrix': matrix}
