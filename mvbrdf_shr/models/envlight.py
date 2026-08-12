"""3rd-order spherical harmonic environment lighting."""
from __future__ import annotations

import torch

# Ramamoorthi & Hanrahan style irradiance convolution for order-2 SH
# (we store order-3 coeffs but use first 9 = L0..L2 for diffuse).
_C1 = 0.429043
_C2 = 0.511664
_C3 = 0.743125
_C4 = 0.886227
_C5 = 0.247708


def sh_to_irradiance(normal: torch.Tensor, sh: torch.Tensor) -> torch.Tensor:
    """Compute diffuse irradiance from SH env light.

    Args:
        normal: (B,3,H,W) unit normals
        sh:     (B,9,3) SH coefficients (bands 0..2)
    Returns:
        (B,3,H,W) irradiance
    """
    n = normal
    nx, ny, nz = n[:, 0:1], n[:, 1:2], n[:, 2:3]
    # sh[:, k] is (B,3) → broadcast over HxW
    def c(k: int) -> torch.Tensor:
        return sh[:, k, :].view(sh.shape[0], 3, 1, 1)

    irr = (
        _C1 * c(8) * (nx * nx - ny * ny)
        + _C3 * c(6) * nz * nz
        + _C4 * c(0)
        - _C5 * c(6)
        + 2.0 * _C1 * (c(4) * nx * ny + c(7) * nx * nz + c(5) * ny * nz)
        + 2.0 * _C2 * (c(3) * nx + c(1) * ny + c(2) * nz)
    )
    return irr.clamp(min=0.0)


def default_sh(batch: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Neutral gray ambient SH (mostly L0)."""
    sh = torch.zeros(batch, 9, 3, device=device, dtype=dtype)
    sh[:, 0, :] = 1.0  # DC
    return sh
