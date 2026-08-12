"""Shared dataclasses for MVBRDF-SHR."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class MaterialParams:
    """Per-pixel / per-point BRDF parameters.

    albedo:      (B,3,H,W) diffuse reflectance ρ_d in [0,1]
    specular:    (B,1,H,W) specular reflectance weight ρ_s in [0,1]
    roughness:   (B,1,H,W) GGX α in [0.05,1]
    metalness:   (B,1,H,W) in [0,1]
    absorption:  (B,1,H,W) light absorption ≥ 0
    """

    albedo: torch.Tensor
    specular: torch.Tensor
    roughness: torch.Tensor
    metalness: torch.Tensor
    absorption: torch.Tensor


@dataclass(slots=True)
class GeometryParams:
    """Screen-space or field geometry.

    normal: (B,3,H,W) unit normals
    depth:  (B,1,H,W) optional depth / disparity
    """

    normal: torch.Tensor
    depth: torch.Tensor | None = None


@dataclass(slots=True)
class LightParams:
    """Incident lighting.

    env_sh:     (B,9,3) 3rd-order SH RGB coefficients
    light_dir:  (B,3) dominant light direction (unit)
    light_int:  (B,1) dominant light intensity
    """

    env_sh: torch.Tensor
    light_dir: torch.Tensor
    light_int: torch.Tensor


@dataclass(slots=True)
class RenderOutputs:
    """Differentiable BRDF render outputs."""

    full: torch.Tensor       # (B,3,H,W) diffuse + specular
    diffuse: torch.Tensor    # (B,3,H,W) diffuse only → highlight-free prior
    specular: torch.Tensor   # (B,3,H,W) specular lobe only
    mask: torch.Tensor       # (B,1,H,W) physics specular mask


@dataclass(slots=True)
class ViewSample:
    """One camera view in a multi-view batch."""

    image: torch.Tensor              # (3,H,W) or will be batched
    pose: torch.Tensor | None = None  # (4,4) c2w
    light_int: float = 1.0


@dataclass(slots=True)
class Batch:
    """Training / eval batch.

    For multi-view: image is (B,V,3,H,W); for single-view (B,3,H,W).
    """

    image: torch.Tensor
    image_clean: torch.Tensor | None = None
    specular_gt: torch.Tensor | None = None  # (B,3,H,W), SHIQ S = A - D
    mask_gt: torch.Tensor | None = None
    poses: torch.Tensor | None = None       # (B,V,4,4)
    light_ints: torch.Tensor | None = None  # (B,V) or (B,)
    domain_ids: torch.Tensor | None = None  # (B,), public-dataset domain index
    meta: dict = field(default_factory=dict)
