"""Geometry field: screen-space normal/depth predictor (hash-ready stub).

Full 3D hash-grid NeRF geometry can replace ScreenGeometryField later;
the interface (image → GeometryParams) stays the same.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..types import GeometryParams


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScreenGeometryField(nn.Module):
    """Predict per-pixel normal (+ optional depth) from RGB.

    Multi-view consistency is enforced via photometric + normal smoothness
    losses in the pipeline, not hard 3D lifting (Phase A).
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(3, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.dec2 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = ConvBlock(base_ch * 2 + base_ch, base_ch)
        self.head_n = nn.Conv2d(base_ch, 3, 1)
        self.head_d = nn.Conv2d(base_ch, 1, 1)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor) -> GeometryParams:
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # Align upsampled feature to the skip target size (odd H/W after pooling
        # would otherwise cause a cat mismatch on non-square inputs like DTU).
        u3 = F.interpolate(self.up(e3), size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u3, e2], dim=1))
        u2 = F.interpolate(self.up(d2), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u2, e1], dim=1))
        n = F.normalize(self.head_n(d1), dim=1, eps=1e-6)
        # Bias toward camera-facing so specular lobe is meaningful out of the box.
        n = F.normalize(n + torch.tensor([0, 0, 1.0], device=n.device).view(1, 3, 1, 1), dim=1)
        depth = torch.sigmoid(self.head_d(d1))
        return GeometryParams(normal=n, depth=depth)
