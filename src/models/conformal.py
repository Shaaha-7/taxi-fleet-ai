"""Split conformal prediction intervals, calibrated separately per horizon, zone and channel.

Exchangeability is only approximate for spatio-temporal series, so coverage is always
reported on the test period rather than assumed.
"""
from __future__ import annotations

import math

import torch


class SplitConformal:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.q: torch.Tensor | None = None       # [H, N, C]

    def calibrate(self, pred: torch.Tensor, y: torch.Tensor) -> "SplitConformal":
        """pred, y: [S, H, N, C] counts from a held-out calibration (validation) period."""
        scores = (y - pred).abs()
        n = scores.shape[0]
        level = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q = torch.quantile(scores, level, dim=0)
        return self

    def interval(self, pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q.to(pred.device)
        return (pred - q).clamp_min(0.0), pred + q
