"""Cook-Torrance / GGX BRDF primitives + energy helpers.

Formalizes per-point absorption / diffuse / specular reflectance:
  a + ρ_d + ρ_s  ≈  1   (soft constraint via losses)
"""
from __future__ import annotations

import torch

_EPS = 1e-6
_PI = 3.141592653589793


def normalize(t: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return t / (t.norm(dim=dim, keepdim=True) + _EPS)


def ggx_d(n_dot_h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a2 = alpha * alpha
    denom = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0
    return a2 / (denom * denom + _EPS)


def smith_g1(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    k = (alpha + 1.0) ** 2 / 8.0
    return x / (x * (1.0 - k) + k + _EPS)


def schlick_fresnel(cos_theta: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    return f0 + (1.0 - f0) * (1.0 - cos_theta).clamp(min=0.0) ** 5


def diffuse_term(albedo: torch.Tensor, irradiance: torch.Tensor) -> torch.Tensor:
    """Lambertian: ρ_d / π * E."""
    return (albedo / _PI) * irradiance


def specular_term(
    normal: torch.Tensor,
    light_dir: torch.Tensor,
    view_dir: torch.Tensor,
    alpha: torch.Tensor,
    f0: torch.Tensor,
    light_mag: torch.Tensor,
) -> torch.Tensor:
    """Microfacet specular lobe (GGX + Smith + Schlick).

    GGX can exceed the FP16 finite range around sharp normal-incidence lobes.
    Keep this analytic branch in FP32 even when the learned networks use AMP.
    """
    n = normalize(normal.float(), dim=1)
    L = normalize(light_dir.float(), dim=1)
    V = normalize(view_dir.float(), dim=1)
    alpha = alpha.float()
    f0 = f0.float()
    light_mag = light_mag.float()
    H = normalize(L + V, dim=1)
    n_dot_l = (n * L).sum(dim=1, keepdim=True).clamp(min=0.0)
    n_dot_v = (n * V).sum(dim=1, keepdim=True).clamp(min=1e-3)
    n_dot_h = (n * H).sum(dim=1, keepdim=True).clamp(min=0.0)
    h_dot_v = (H * V).sum(dim=1, keepdim=True).clamp(min=0.0)
    D = ggx_d(n_dot_h, alpha)
    G = smith_g1(n_dot_l, alpha) * smith_g1(n_dot_v, alpha)
    F = schlick_fresnel(h_dot_v, f0)
    spec = D * G * F / (4.0 * n_dot_l * n_dot_v + _EPS)
    return spec * n_dot_l * light_mag


def f0_from_metalness(albedo: torch.Tensor, metalness: torch.Tensor) -> torch.Tensor:
    """Dielectric F0=0.04 blended with albedo for metals."""
    return 0.04 * (1.0 - metalness) + albedo * metalness


def energy_violation(
    albedo: torch.Tensor, specular: torch.Tensor, absorption: torch.Tensor
) -> torch.Tensor:
    """Soft energy: mean ReLU(ρ_d_mean + ρ_s + a - 1)."""
    rho_d = albedo.mean(dim=1, keepdim=True)
    return torch.relu(rho_d + specular + absorption - 1.0)
