"""Shared evaluation metrics."""
from __future__ import annotations

import numpy as np
import torch


def psnr_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.detach().float().clamp(0, 1)
    gt = gt.detach().float().clamp(0, 1)
    mse = torch.mean((pred - gt) ** 2).item()
    if mse < 1e-12:
        return 99.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def ssim_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Simple channel-mean SSIM with 11x11 box filter."""
    from numpy.lib.stride_tricks import sliding_window_view

    def _one(x: np.ndarray, y: np.ndarray) -> float:
        c1, c2 = 0.01**2, 0.03**2
        k = 11
        xp = np.pad(x, k // 2, mode="reflect")
        yp = np.pad(y, k // 2, mode="reflect")

        def mf(a: np.ndarray) -> np.ndarray:
            return sliding_window_view(a, (k, k)).mean(axis=(-2, -1))

        mx, my = mf(xp), mf(yp)
        sx = mf(xp * xp) - mx * mx
        sy = mf(yp * yp) - my * my
        sxy = mf(xp * yp) - mx * my
        num = (2 * mx * my + c1) * (2 * sxy + c2)
        den = (mx * mx + my * my + c1) * (sx + sy + c2)
        return float((num / (den + 1e-12)).mean())

    p = pred.detach().float().cpu().clamp(0, 1)
    g = gt.detach().float().cpu().clamp(0, 1)
    if p.dim() == 4:
        p, g = p[0], g[0]
    p = p.permute(1, 2, 0).numpy()
    g = g.permute(1, 2, 0).numpy()
    return float(np.mean([_one(p[:, :, c], g[:, :, c]) for c in range(3)]))
