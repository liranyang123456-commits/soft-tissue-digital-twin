"""Incident light field: SH environment + dominant directional light.

Supports per-view light intensity modulation (multi-illumination sequences
as in SHIQ / NSH capture systems).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..types import LightParams
from .envlight import default_sh


class LightField(nn.Module):
    """Predict global lighting from one or more views."""

    def __init__(self, in_ch: int = 3, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_sh = nn.Linear(hidden * 2, 9 * 3)
        self.fc_dir = nn.Linear(hidden * 2, 3)
        self.fc_int = nn.Linear(hidden * 2, 1)

    def forward(
        self,
        image: torch.Tensor,
        light_int_scale: torch.Tensor | None = None,
    ) -> LightParams:
        """
        Args:
            image: (B,3,H,W)
            light_int_scale: optional (B,) or (B,1) multiplier for multi-light seqs
        """
        B = image.shape[0]
        feat = self.backbone(image).flatten(1)
        sh = self.fc_sh(feat).view(B, 9, 3)
        # Residual around a neutral ambient so early training is stable.
        sh = sh + default_sh(B, image.device, image.dtype)
        # Keep unit-vector normalization in FP32 for stable AMP gradients.
        direction = F.normalize(self.fc_dir(feat).float(), dim=-1, eps=1e-4)
        # Bias upward-facing light
        direction = F.normalize(
            direction + direction.new_tensor([0.3, 0.5, 0.7]),
            dim=-1,
            eps=1e-4,
        )
        intensity = F.softplus(self.fc_int(feat)) + 0.5
        if light_int_scale is not None:
            scale = light_int_scale.view(B, 1).to(dtype=intensity.dtype)
            intensity = intensity * scale
        return LightParams(env_sh=sh, light_dir=direction, light_int=intensity)
