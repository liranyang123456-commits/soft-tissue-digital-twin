"""Frozen common holdout protocol for EndoNeRF reconstruction."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from mvbrdf_shr.metrics import ssim_map_torch, ssim_torch


@dataclass(frozen=True)
class EndoNeRFHoldout:
    n_frames: int
    test_every: int
    test_offset: int
    train_ids: tuple[int, ...]
    holdout_ids: tuple[int, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["train_ids"] = list(self.train_ids)
        result["holdout_ids"] = list(self.holdout_ids)
        return result


def endonerf_holdout(
    n_frames: int,
    *,
    test_every: int = 8,
    test_offset: int = 1,
) -> EndoNeRFHoldout:
    """Match the official EndoGaussian EndoNeRF split.

    The official loader uses ``(frame_id - 1) % 8 == 0`` for test frames.
    """
    if n_frames < 2:
        raise ValueError("at least two frames are required")
    if test_every < 2:
        raise ValueError("test_every must be at least two")
    holdout = tuple(
        frame_id
        for frame_id in range(n_frames)
        if (frame_id - test_offset) % test_every == 0
    )
    train = tuple(frame_id for frame_id in range(n_frames) if frame_id not in holdout)
    return EndoNeRFHoldout(n_frames, test_every, test_offset, train, holdout)


def masked_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """PSNR on pixels selected by a two-dimensional tissue mask."""
    prediction = prediction.detach().float().clamp(0, 1)
    target = target.detach().float().clamp(0, 1)
    mask = mask.detach().float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.expand_as(prediction)
    denominator = mask.sum().clamp_min(1.0)
    mse = (((prediction - target) ** 2) * mask).sum() / denominator
    if float(mse) < 1e-12:
        return 99.0
    return float(-10.0 * torch.log10(mse))


def image_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tissue_mask: torch.Tensor,
) -> dict[str, float]:
    """Return full-image and tissue-restricted PSNR/SSIM."""
    prediction = prediction.detach().float().clamp(0, 1)
    target = target.detach().float().clamp(0, 1)
    mask = tissue_mask.detach().float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    full_mse = torch.mean((prediction - target) ** 2)
    full_psnr = (
        99.0 if float(full_mse) < 1e-12 else float(-10.0 * torch.log10(full_mse))
    )
    ssim_map = ssim_map_torch(prediction, target)
    mask_2d = mask[0].detach().cpu().numpy()
    tissue_ssim = float(
        (ssim_map * mask_2d).sum() / max(float(mask_2d.sum()), 1.0)
    )
    return {
        "full_psnr": full_psnr,
        "full_ssim": ssim_torch(prediction, target),
        "tissue_psnr": masked_psnr(prediction, target, mask),
        "tissue_ssim": tissue_ssim,
        "tissue_fraction": float(mask.mean()),
    }


def aggregate_metrics(rows: list[dict]) -> dict:
    keys = ("full_psnr", "full_ssim", "tissue_psnr", "tissue_ssim", "tissue_fraction")
    return {
        "n": len(rows),
        **{
            key: float(np.mean([row[key] for row in rows]))
            for key in keys
            if rows
        },
    }
