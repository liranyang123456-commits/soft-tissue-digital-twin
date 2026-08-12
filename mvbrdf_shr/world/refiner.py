"""Final generative refinement after world-space physical rendering."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..models.restorer import Down, ResBlock, SEBlock, Up


class PhysicsGuidedWorldRefiner(nn.Module):
    """Repair only regions where physical diffuse rendering is uncertain.

    Conditions: input RGB, physical diffuse, specular mask, normal, depth,
    cross-view projected texture, and rendered albedo.
    """

    def __init__(self, base: int = 96, n_res: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(17, base, 3, padding=1),
            nn.SiLU(inplace=True),
            ResBlock(base),
        )
        self.d1 = Down(base, base * 2, n_res)
        self.d2 = Down(base * 2, base * 4, n_res)
        self.d3 = Down(base * 4, base * 8, n_res)
        self.mid = nn.Sequential(
            ResBlock(base * 8), ResBlock(base * 8), SEBlock(base * 8)
        )
        self.u2 = Up(base * 8, base * 4, base * 4, n_res)
        self.u1 = Up(base * 4, base * 2, base * 2, n_res)
        self.u0 = Up(base * 2, base, base, n_res)
        self.out_rgb = nn.Conv2d(base, 3, 1)

    @staticmethod
    def normalize_depth(depth: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        valid = alpha > 1e-3
        if not bool(valid.any()):
            return torch.zeros_like(depth)
        values = depth[valid]
        lo = torch.quantile(values.detach(), 0.02)
        hi = torch.quantile(values.detach(), 0.98)
        return ((depth - lo) / (hi - lo + 1e-6)).clamp(0, 1) * valid

    def forward(
        self,
        image: torch.Tensor,
        diffuse: torch.Tensor,
        mask: torch.Tensor,
        normal: torch.Tensor,
        depth: torch.Tensor,
        projected_texture: torch.Tensor,
        albedo: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        depth = self.normalize_depth(depth, alpha)
        conditions = torch.cat(
            [
                image,
                diffuse,
                mask,
                normal,
                depth,
                projected_texture,
                albedo,
            ],
            dim=1,
        )
        s0 = self.stem(conditions)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        x = self.mid(s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        x = self.u0(x, s0)
        residual = self.out_rgb(x)
        # The physical diffuse render is the globally consistent estimate.
        # Generation is restricted to the predicted specular region so it
        # cannot re-introduce highlights by copying the observation elsewhere.
        return (diffuse + residual * mask).clamp(0, 1)

    def load_shiq_restorer(self, checkpoint: str | Path) -> int:
        """Transfer the trained SHIQ Restorer into the world-space refiner."""
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = state.get("model", state)
        source = {
            key.removeprefix("restorer."): value
            for key, value in source.items()
            if key.startswith("restorer.")
        }
        own = self.state_dict()
        copied = 0
        for key, value in source.items():
            if key == "stem.0.weight" and value.shape[0] == own[key].shape[0]:
                target = own[key].clone()
                # A,diffuse,mask,normal are channels 0:10 in both models.
                target[:, :10] = value[:, :10]
                # SHIQ albedo channels 10:13 map to world channels 14:17.
                target[:, 14:17] = value[:, 10:13]
                own[key] = target
                copied += 1
            elif key in own and own[key].shape == value.shape:
                own[key] = value
                copied += 1
        self.load_state_dict(own)
        return copied

