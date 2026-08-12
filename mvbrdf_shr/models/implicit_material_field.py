"""Bounded implicit residual for world-space material image formation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fourier_features(x: torch.Tensor, levels: int) -> torch.Tensor:
    features = [x]
    for level in range(levels):
        frequency = (2.0**level) * torch.pi
        features.extend((torch.sin(frequency * x), torch.cos(frequency * x)))
    return torch.cat(features, dim=-1)


class ImplicitMaterialImagingField(nn.Module):
    """Learn a spatially continuous, view/light-conditioned BRDF residual.

    The field never receives frame or view identifiers.  Its zero-initialized
    output preserves the explicit renderer exactly at construction, while a
    bounded log-response prevents it from replacing the physical BRDF.
    """

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 3,
        position_levels: int = 4,
        direction_levels: int = 2,
        residual_limit: float = 0.10,
    ):
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least 2")
        self.position_levels = int(position_levels)
        self.direction_levels = int(direction_levels)
        self.residual_limit = float(residual_limit)
        position_dim = 3 * (1 + 2 * self.position_levels)
        # xyz, normal, four rotation-invariant BRDF angles, and explicit
        # material (albedo3 + diffuse/spec/rough/metal/absorb).  Absolute
        # camera/light directions and intensities are deliberately excluded:
        # they invite capture-specific memorization and fail under novel light.
        input_dim = position_dim + 3 + 4 + 8
        modules: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(layers - 2):
            modules.extend((nn.Linear(hidden, hidden), nn.SiLU()))
        output = nn.Linear(hidden, 6)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        modules.append(output)
        self.network = nn.Sequential(*modules)

    def forward(
        self,
        position: torch.Tensor,
        normal: torch.Tensor,
        view_direction: torch.Tensor,
        light_direction: torch.Tensor,
        material_features: torch.Tensor,
        environment_rgb: torch.Tensor,
        direct_intensity: torch.Tensor,
    ) -> torch.Tensor:
        center = position.detach().mean(dim=0, keepdim=True)
        radius = (position.detach() - center).norm(dim=-1).amax().clamp_min(1e-4)
        position = (position - center) / radius
        count = position.shape[0]

        def expand(value: torch.Tensor, width: int) -> torch.Tensor:
            value = value.reshape(-1, width)
            return value.expand(count, -1) if value.shape[0] == 1 else value

        normal = F.normalize(normal, dim=-1, eps=1e-6)
        view_direction = F.normalize(view_direction, dim=-1, eps=1e-6)
        light_direction = F.normalize(
            expand(light_direction, 3), dim=-1, eps=1e-6
        )
        half_direction = F.normalize(
            view_direction + light_direction, dim=-1, eps=1e-6
        )
        angular = torch.cat(
            (
                (normal * view_direction).sum(-1, keepdim=True),
                (normal * light_direction).sum(-1, keepdim=True),
                (view_direction * light_direction).sum(-1, keepdim=True),
                (normal * half_direction).sum(-1, keepdim=True),
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                _fourier_features(position, self.position_levels),
                normal,
                angular,
                material_features,
            ),
            dim=-1,
        )
        # Six bounded log-scales: RGB diffuse and RGB specular response.
        return torch.tanh(self.network(features)) * self.residual_limit

