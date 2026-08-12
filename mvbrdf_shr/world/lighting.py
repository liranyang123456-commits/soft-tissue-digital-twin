"""Per-frame SH/directional lighting and a NeILF-style incident light field."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def latlong_to_sh(envmap: torch.Tensor) -> torch.Tensor:
    """Project Stanford-ORB's +Z-left, +Y-up lat-long map to real SH."""
    if envmap.ndim != 3 or envmap.shape[0] != 3:
        raise ValueError("envmap must have shape 3xHxW")
    height, width = envmap.shape[-2:]
    phi = (torch.arange(height, device=envmap.device, dtype=envmap.dtype) + 0.5)
    phi = phi * torch.pi / height
    theta = (
        torch.arange(width, device=envmap.device, dtype=envmap.dtype) + 0.5
    ) * (2 * torch.pi / width) - 0.5 * torch.pi
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")
    sin_phi = torch.sin(phi)
    x = -torch.cos(theta) * sin_phi
    y = torch.cos(phi)
    z = -torch.sin(theta) * sin_phi
    basis = torch.stack(
        [
            torch.full_like(x, 0.282095),
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3 * z.square() - 1),
            1.092548 * x * z,
            0.546274 * (x.square() - y.square()),
        ],
        dim=-1,
    )
    solid_angle = sin_phi * (torch.pi / height) * (2 * torch.pi / width)
    return torch.einsum("chw,hwk,hw->kc", envmap, basis, solid_angle)


def latlong_residual_to_sh(
    envmap: torch.Tensor, dominant_quantile: float = 0.995
) -> torch.Tensor:
    """Project the non-dominant environment after extracting a delta lobe.

    ``latlong_dominant_light`` collapses the brightest environment support to
    one directional light for GGX highlights.  Projecting the *full* map to SH
    and then adding that directional light double-counts the same energy.
    This function removes that support before the low-order SH projection.
    """
    if not 0.0 < dominant_quantile < 1.0:
        raise ValueError("dominant_quantile must be in (0, 1)")
    luminance = (
        0.2126 * envmap[0] + 0.7152 * envmap[1] + 0.0722 * envmap[2]
    )
    support = luminance >= torch.quantile(luminance, dominant_quantile)
    residual = envmap.masked_fill(support.unsqueeze(0), 0.0)
    return latlong_to_sh(residual)


def latlong_dominant_light(
    envmap: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Extract a dominant directional lobe for GGX from a Stanford envmap."""
    if envmap.ndim != 3 or envmap.shape[0] != 3:
        raise ValueError("envmap must have shape 3xHxW")
    height, width = envmap.shape[-2:]
    luminance = (
        0.2126 * envmap[0] + 0.7152 * envmap[1] + 0.0722 * envmap[2]
    )
    phi = (torch.arange(height, device=envmap.device, dtype=envmap.dtype) + 0.5)
    phi = phi * torch.pi / height
    theta = (
        torch.arange(width, device=envmap.device, dtype=envmap.dtype) + 0.5
    ) * (2 * torch.pi / width) - 0.5 * torch.pi
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")
    sin_phi = torch.sin(phi)
    directions = torch.stack(
        (
            -torch.cos(theta) * sin_phi,
            torch.cos(phi),
            -torch.sin(theta) * sin_phi,
        ),
        dim=-1,
    )
    support = luminance >= torch.quantile(luminance, 0.995)
    solid_angle = sin_phi * (torch.pi / height) * (2 * torch.pi / width)
    weights = luminance * support * solid_angle
    direction = (directions * weights[..., None]).sum(dim=(0, 1))
    power = (envmap * support[None] * solid_angle[None]).sum(dim=(1, 2))
    intensity = max(float(power.max()), 1e-6)
    return F.normalize(direction, dim=0), power / intensity, intensity


@dataclass(slots=True)
class WorldLight:
    env_sh: torch.Tensor      # (9,3)
    direction: torch.Tensor   # (3,), surface-to-light
    intensity: torch.Tensor   # (1,)
    color: torch.Tensor       # (3,)
    position: torch.Tensor    # (3,)
    point_weight: torch.Tensor  # (1,), 0=directional, 1=point
    exposure: torch.Tensor    # (1,), camera exposure multiplier
    white_balance: torch.Tensor  # (3,), camera RGB gains
    # Optional quadrature samples for direct environment integration.
    env_directions: torch.Tensor | None = None  # (S,3)
    env_radiance: torch.Tensor | None = None  # (S,3)
    env_solid_angle: torch.Tensor | None = None  # (S,)


class PerFrameLighting(nn.Module):
    """First-level lighting: SH environment plus one dominant light per frame."""

    def __init__(
        self,
        n_frames: int,
        shared: bool = False,
        light_ids: list[int | None] | None = None,
    ):
        super().__init__()
        self.shared = shared
        if light_ids is None:
            light_ids = list(range(n_frames))
        if len(light_ids) != n_frames:
            raise ValueError("light_ids must contain one entry per frame")
        normalized = [
            int(light_id) if light_id is not None else index
            for index, light_id in enumerate(light_ids)
        ]
        unique = sorted(set(normalized))
        lookup = {light_id: index for index, light_id in enumerate(unique)}
        mapping = torch.tensor([lookup[light_id] for light_id in normalized])
        if shared:
            mapping.zero_()
        self.register_buffer("frame_to_light", mapping, persistent=True)
        n_lights = 1 if shared else len(unique)
        sh = torch.zeros(n_lights, 9, 3)
        sh[:, 0] = 1.0
        self.env_sh = nn.Parameter(sh)
        direction = torch.tensor([0.3, -0.4, 0.85]).view(1, 3).repeat(n_lights, 1)
        self.direction_raw = nn.Parameter(direction)
        self.log_intensity = nn.Parameter(torch.zeros(n_lights, 1))
        self.color_logits = nn.Parameter(torch.full((n_lights, 3), 2.2))
        self.position = nn.Parameter(
            torch.tensor([0.0, -2.0, 2.0]).view(1, 3).repeat(n_lights, 1)
        )
        self.point_light_logits = nn.Parameter(torch.full((n_lights, 1), -8.0))
        self.log_exposure = nn.Parameter(torch.zeros(n_frames, 1))
        self.log_white_balance = nn.Parameter(torch.zeros(n_frames, 3))

    def light_index(self, frame_id: int | torch.Tensor) -> int | torch.Tensor:
        return self.frame_to_light[frame_id]

    def sensor(self, frame_id: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positive exposure and chromatic gains for a capture."""
        exposure = self.log_exposure[frame_id].clamp(-8.0, 8.0).exp()
        gains = self.log_white_balance[frame_id].clamp(-4.0, 4.0).exp()
        # Remove the luminance scale ambiguity; exposure owns the global gain.
        gains = gains / gains.prod(dim=-1, keepdim=True).pow(1.0 / 3.0)
        return exposure, gains

    def forward(self, frame_id: int | torch.Tensor) -> WorldLight:
        light_index = self.light_index(frame_id)
        exposure, white_balance = self.sensor(frame_id)
        return WorldLight(
            env_sh=self.env_sh[light_index],
            direction=F.normalize(self.direction_raw[light_index], dim=-1),
            intensity=self.log_intensity[light_index].exp(),
            color=torch.sigmoid(self.color_logits[light_index]),
            position=self.position[light_index],
            point_weight=torch.sigmoid(self.point_light_logits[light_index]),
            exposure=exposure,
            white_balance=white_balance,
        )


def positional_encoding(x: torch.Tensor, n_freqs: int = 4) -> torch.Tensor:
    values = [x]
    for k in range(n_freqs):
        f = (2.0**k) * torch.pi
        values.extend([torch.sin(f * x), torch.cos(f * x)])
    return torch.cat(values, dim=-1)


class NeuralIncidentLightField(nn.Module):
    """NeILF-style Li(x, wi): 3D position + 2D direction manifold -> RGB.

    The implementation accepts a 3D unit direction (rather than explicitly
    storing two angles); the direction still has two degrees of freedom.
    """

    def __init__(self, hidden: int = 128, n_freqs: int = 4):
        super().__init__()
        self.n_freqs = n_freqs
        encoded_dim = 6 * (1 + 2 * n_freqs)
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(
        self, positions: torch.Tensor, incident_directions: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast positions/directions and return non-negative incident RGB."""
        x, d = torch.broadcast_tensors(positions, incident_directions)
        features = torch.cat(
            [
                positional_encoding(x, self.n_freqs),
                positional_encoding(F.normalize(d, dim=-1), self.n_freqs),
            ],
            dim=-1,
        )
        return F.softplus(self.net(features))


def fibonacci_sphere(
    n_samples: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Deterministic, approximately uniform directions on S²."""
    i = torch.arange(n_samples, device=device, dtype=dtype) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    radius = torch.sqrt((1.0 - z * z).clamp(min=0))
    golden = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0, device=device, dtype=dtype)))
    theta = golden * i
    return torch.stack([radius * torch.cos(theta), radius * torch.sin(theta), z], dim=-1)

