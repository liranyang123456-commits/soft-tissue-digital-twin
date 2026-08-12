"""Factorized surface-material and illumination latent neural BRDF."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fourier(x: torch.Tensor, levels: int) -> torch.Tensor:
    values = [x]
    for level in range(levels):
        frequency = torch.pi * (2.0**level)
        values.extend((torch.sin(frequency * x), torch.cos(frequency * x)))
    return torch.cat(values, dim=-1)


class DualLatentNeuralBRDF(nn.Module):
    """Decode diffuse/specular BRDF from disentangled physical latents.

    Material latents are shared by a world-space surface point across every
    view. Illumination latents are computed from physical light quantities;
    neither branch accepts frame/view identifiers.
    """

    def __init__(
        self,
        material_latent: int = 32,
        light_latent: int = 16,
        hidden: int = 96,
        position_levels: int = 3,
        mode: str = "replacement",
        residual_limit: float = 0.35,
        highlight_gate: bool = False,
    ):
        super().__init__()
        if mode not in {"replacement", "log_residual"}:
            raise ValueError("mode must be 'replacement' or 'log_residual'")
        self.position_levels = int(position_levels)
        self.mode = mode
        self.residual_limit = float(residual_limit)
        self.highlight_gate = bool(highlight_gate)
        position_dim = 3 * (1 + 2 * self.position_levels)
        self.material_encoder = nn.Sequential(
            nn.Linear(position_dim + 8, hidden),
            nn.SiLU(),
            nn.Linear(hidden, material_latent),
        )
        # direction3, log-intensity1, RGB color3, low-order environment RGB3
        self.light_encoder = nn.Sequential(
            nn.Linear(10, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, light_latent),
        )
        # Four invariant BRDF angles plus the two factorized latents.
        self.decoder = nn.Sequential(
            nn.Linear(material_latent + light_latent + 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6),
        )
        if self.mode == "log_residual":
            # The physical GGX renderer is reproduced exactly at initialization.
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def _expand(value: torch.Tensor, leading: torch.Size) -> torch.Tensor:
        return value.reshape(*((1,) * len(leading)), value.shape[-1]).expand(
            *leading, value.shape[-1]
        )

    def encode_material(
        self, position: torch.Tensor, material_features: torch.Tensor
    ) -> torch.Tensor:
        center = position.detach().reshape(-1, 3).mean(0)
        radius = (
            (position.detach().reshape(-1, 3) - center)
            .norm(dim=-1)
            .amax()
            .clamp_min(1e-4)
        )
        normalized = (position - center) / radius
        return self.material_encoder(
            torch.cat((_fourier(normalized, self.position_levels), material_features), -1)
        )

    def encode_light(
        self,
        direction: torch.Tensor,
        intensity: torch.Tensor,
        color: torch.Tensor,
        environment_rgb: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                F.normalize(direction, dim=-1, eps=1e-6),
                torch.log1p(intensity.clamp_min(0)),
                color,
                environment_rgb,
            ),
            dim=-1,
        )
        return self.light_encoder(features)

    def forward(
        self,
        position: torch.Tensor,
        normal: torch.Tensor,
        view_direction: torch.Tensor,
        light_direction: torch.Tensor,
        material_features: torch.Tensor,
        light_intensity: torch.Tensor,
        light_color: torch.Tensor,
        environment_rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        material_latent = self.encode_material(position, material_features)
        leading = position.shape[:-1]
        light_latent = self.encode_light(
            light_direction.reshape(-1, 3)[:1],
            light_intensity.reshape(-1, 1)[:1],
            light_color.reshape(-1, 3)[:1],
            environment_rgb.reshape(-1, 3)[:1],
        )
        light_latent = self._expand(light_latent[0], leading)
        diffuse, specular = self.decode(
            material_latent,
            light_latent,
            normal,
            view_direction,
            light_direction,
        )
        return diffuse, specular, {
            "material": material_latent,
            "light": light_latent,
        }

    def decode(
        self,
        material_latent: torch.Tensor,
        light_latent: torch.Tensor,
        normal: torch.Tensor,
        view_direction: torch.Tensor,
        light_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normal = F.normalize(normal, dim=-1, eps=1e-6)
        view = F.normalize(view_direction, dim=-1, eps=1e-6)
        light = F.normalize(light_direction, dim=-1, eps=1e-6)
        half = F.normalize(view + light, dim=-1, eps=1e-6)
        angular = torch.cat(
            (
                (normal * view).sum(-1, keepdim=True),
                (normal * light).sum(-1, keepdim=True),
                (view * light).sum(-1, keepdim=True),
                (normal * half).sum(-1, keepdim=True),
            ),
            -1,
        )
        decoded = F.softplus(
            self.decoder(torch.cat((material_latent, light_latent, angular), -1))
        )
        return decoded[..., :3], decoded[..., 3:]

    def decode_log_specular_residual(
        self,
        material_latent: torch.Tensor,
        light_latent: torch.Tensor,
        normal: torch.Tensor,
        view_direction: torch.Tensor,
        light_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Return a bounded log-domain multiplier for the explicit GGX lobe."""

        normal = F.normalize(normal, dim=-1, eps=1e-6)
        view = F.normalize(view_direction, dim=-1, eps=1e-6)
        light = F.normalize(light_direction, dim=-1, eps=1e-6)
        half = F.normalize(view + light, dim=-1, eps=1e-6)
        angular = torch.cat(
            (
                (normal * view).sum(-1, keepdim=True),
                (normal * light).sum(-1, keepdim=True),
                (view * light).sum(-1, keepdim=True),
                (normal * half).sum(-1, keepdim=True),
            ),
            -1,
        )
        raw = self.decoder(
            torch.cat((material_latent, light_latent, angular), -1)
        )[..., 3:]
        return torch.tanh(raw) * self.residual_limit

