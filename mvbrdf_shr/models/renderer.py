"""Differentiable BRDF renderer: full / diffuse / specular / mask.

Core of highlight removal: I_diff = render with specular lobe disabled.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..types import GeometryParams, LightParams, MaterialParams, RenderOutputs
from .brdf import diffuse_term, energy_violation, f0_from_metalness, specular_term
from .envlight import sh_to_irradiance


class BRDFRenderer(nn.Module):
    """Per-pixel physically based renderer."""

    def __init__(self, tau_init: float = 0.3):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        # Default camera looks along +Z (OpenCV-ish screen space).
        self.register_buffer(
            "view_dir",
            torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self,
        geom: GeometryParams,
        mat: MaterialParams,
        light: LightParams,
        input_image: torch.Tensor | None = None,
    ) -> RenderOutputs:
        B, _, H, W = mat.albedo.shape
        n = geom.normal
        irr = sh_to_irradiance(n, light.env_sh)

        # Diffuse (view-independent absorption/reflection path)
        diff = diffuse_term(mat.albedo, irr)
        # Scale diffuse by (1 - metalness) as in Disney/principled BRDF
        diff = diff * (1.0 - mat.metalness)

        # Specular (view-dependent → highlights)
        L = light.light_dir.view(B, 3, 1, 1).expand(-1, -1, H, W)
        V = self.view_dir.expand(B, -1, H, W)
        f0 = f0_from_metalness(mat.albedo, mat.metalness)
        # Blend base F0 with explicit specular reflectance weight
        f0 = f0 * (0.5 + 0.5 * mat.specular)
        light_mag = light.light_int.view(B, 1, 1, 1) * (0.5 + mat.specular)
        alpha = mat.roughness
        spec = specular_term(n, L, V, alpha, f0, light_mag)

        full = (diff + spec).clamp(0.0, 1.0)
        diffuse_only = diff.clamp(0.0, 1.0)
        specular_only = spec.clamp(0.0, 1.0)

        ref = input_image if input_image is not None else full
        delta = (ref - diffuse_only).abs().mean(dim=1, keepdim=True)
        mask = (delta / (self.tau.abs() + 1e-4)).clamp(0.0, 1.0)

        return RenderOutputs(
            full=full, diffuse=diffuse_only, specular=specular_only, mask=mask
        )

    @staticmethod
    def energy_loss(mat: MaterialParams) -> torch.Tensor:
        return energy_violation(mat.albedo, mat.specular, mat.absorption).mean()
