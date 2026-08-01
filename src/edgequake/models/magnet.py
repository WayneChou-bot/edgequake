"""MagNet-style early-magnitude estimator (Phase 5).

Input:  4 s of P-onset waveform (3 ch x 400 samples @100 Hz, peak-normalized)
        + aux scalars [log10(peak counts), log10(std counts)]
Output: (mu, log_var) of the FINAL catalog magnitude — a Gaussian head, so
        every estimate carries its own uncertainty.

KEEP IN SYNC with the copy in kaggle/edgequake_magnet.ipynb (the notebook is
self-contained for Kaggle; this module is what the live engine imports).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MagNet(nn.Module):
    N_IN = 400  # samples (4 s @ 100 Hz)

    def __init__(self, n_aux: int = 2, dropout: float = 0.2):
        super().__init__()
        def blk(ci, co, k, s):
            return [nn.Conv1d(ci, co, k, stride=s, padding=k // 2),
                    nn.BatchNorm1d(co), nn.ReLU()]
        self.conv = nn.Sequential(
            *blk(3, 32, 7, 2), *blk(32, 64, 5, 2),
            *blk(64, 64, 3, 2), *blk(64, 128, 3, 2))   # 400 -> 25
        self.head = nn.Sequential(
            nn.Linear(256 + n_aux, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor, aux: torch.Tensor):
        h = self.conv(x)
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)   # 256
        out = self.head(torch.cat([h, aux], dim=1))
        mu = out[:, 0]
        log_var = out[:, 1].clamp(-6.0, 3.0)
        return mu, log_var

    @torch.no_grad()
    def estimate(self, x: torch.Tensor, aux: torch.Tensor):
        """Returns (mag, sigma) tensors."""
        self.eval()
        mu, log_var = self(x, aux)
        return mu, torch.exp(0.5 * log_var)


def gaussian_nll(mu, log_var, y, weight=None):
    nll = 0.5 * (log_var + (y - mu) ** 2 / torch.exp(log_var))
    if weight is not None:
        nll = nll * weight
    return nll.mean()
