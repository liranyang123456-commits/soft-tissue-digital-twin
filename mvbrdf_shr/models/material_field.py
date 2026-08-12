"""Material field: per-pixel absorption / reflectance coefficients.

Each pixel (≈ surface point under orthographic / known-depth assumption)
stores independent material parameters that govern how light is absorbed
and reflected — the heart of the MVBRDF-SHR idea.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..types import MaterialParams


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


class MaterialField(nn.Module):
    """Image-conditioned material parameter field.

    Inputs may be a single view or a view stacked with optional geometry cues
    (normal) for better disentanglement.
    """

    def __init__(self, in_ch: int = 6, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.dec2 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = ConvBlock(base_ch * 2 + base_ch, base_ch)
        self.head = nn.Conv2d(base_ch, 7, 1)  # albedo3 + spec1 + rough1 + metal1 + abs1
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor, normal: torch.Tensor | None = None) -> MaterialParams:
        if normal is None:
            # Fallback: duplicate image channels to keep in_ch=6 path simple via 3ch path
            x = torch.cat([image, image], dim=1)[:, :6]
        else:
            x = torch.cat([image, normal], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # Align upsampled feature to skip target (odd H/W safe on non-square inputs).
        u3 = F.interpolate(self.up(e3), size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u3, e2], dim=1))
        u2 = F.interpolate(self.up(d2), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u2, e1], dim=1))
        raw = self.head(d1)
        albedo = torch.sigmoid(raw[:, 0:3])
        specular = torch.sigmoid(raw[:, 3:4])
        roughness = 0.05 + 0.95 * torch.sigmoid(raw[:, 4:5])
        metalness = torch.sigmoid(raw[:, 5:6])
        absorption = F.softplus(raw[:, 6:7])
        return MaterialParams(
            albedo=albedo,
            specular=specular,
            roughness=roughness,
            metalness=metalness,
            absorption=absorption,
        )


class MultiViewMaterialAggregator(nn.Module):
    """Fuse per-view material predictions into a view-consistent estimate.

    Softmax-weighted average by inverse specular energy — views where a pixel
    is less specular are more trustworthy for albedo (classic multi-light idea,
    related to SHIQ / NSH capture).
    """

    def forward(self, mats: list[MaterialParams]) -> MaterialParams:
        assert mats, "empty material list"
        if len(mats) == 1:
            return mats[0]
        specs = torch.stack([m.specular for m in mats], dim=0)  # (V,B,1,H,W)
        weights = torch.softmax(-specs * 5.0, dim=0)
        def agg(attr: str) -> torch.Tensor:
            stacked = torch.stack([getattr(m, attr) for m in mats], dim=0)
            # broadcast weights to channel dim
            w = weights
            if stacked.shape[2] != 1:
                w = weights.expand_as(stacked)
            return (stacked * w).sum(dim=0)
        return MaterialParams(
            albedo=agg("albedo"),
            specular=agg("specular"),
            roughness=agg("roughness"),
            metalness=agg("metalness"),
            absorption=agg("absorption"),
        )
