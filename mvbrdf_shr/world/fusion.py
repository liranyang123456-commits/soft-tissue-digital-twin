"""Confidence-gated fusion of MVBRDF-SHR Pro and world-space rendering."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..models.pipeline_pro import MVBRDFSHRPro
from .renderer import WorldRenderOutputs


def _batched(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor


class HybridMVBRDFSHR(nn.Module):
    """Fuse single-image restoration with persistent world-space physics.

    The world branch is trusted only where it has visible geometry and can
    reconstruct the observed full radiance. Everywhere else the model falls
    back to the single-image prediction. A zero-initialized bounded correction
    is active only near uncertain gate values, preventing an untrained fusion
    head from degrading either source.
    """

    def __init__(
        self,
        single: MVBRDFSHRPro | None = None,
        resolution: int = 200,
        base_ch: int = 48,
        restorer_base: int = 96,
        gate_hidden: int = 48,
        reconstruction_temperature: float = 0.08,
        correction_limit: float = 0.05,
    ):
        super().__init__()
        self.single = single or MVBRDFSHRPro(
            resolution, base_ch, restorer_base
        )
        self.reconstruction_temperature = reconstruction_temperature
        self.correction_limit = correction_limit
        # image, world D/S/mask/N/albedo/alpha, Pro pred/S/mask = 24 channels.
        self.gate = nn.Sequential(
            nn.Conv2d(24, gate_hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, gate_hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1),
        )
        self.correction = nn.Sequential(
            nn.Conv2d(9, gate_hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 3, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.zeros_(self.correction[-2].weight)
        nn.init.zeros_(self.correction[-2].bias)

    def load_single_checkpoint(self, checkpoint: str | Path) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.single.load_state_dict(state.get("model", state), strict=False)

    @staticmethod
    def _world_buffers(
        world: WorldRenderOutputs | dict[str, Any],
    ) -> WorldRenderOutputs:
        if isinstance(world, WorldRenderOutputs):
            return world
        render = world.get("render")
        if not isinstance(render, WorldRenderOutputs):
            raise TypeError("world must be WorldRenderOutputs or pipeline output")
        return render

    def forward(
        self,
        image: torch.Tensor,
        world: WorldRenderOutputs | dict[str, Any] | None = None,
        detach_backbones: bool = True,
    ) -> dict[str, Any]:
        image = _batched(image)
        pro = self.single(
            image, use_refine=True, detach_physics=detach_backbones
        )
        if world is None:
            return {
                "pred": pro["pred"],
                "single_pred": pro["pred"],
                "single": pro,
                "world": None,
                "gate": torch.zeros_like(pro["mask_pred"]),
                "mode": "single",
            }

        render = self._world_buffers(world)
        world_diffuse = _batched(render.diffuse)
        world_specular = _batched(render.specular)
        world_mask = _batched(render.mask)
        world_normal = _batched(render.normal)
        world_albedo = _batched(render.albedo)
        world_alpha = _batched(render.alpha)
        single_pred = pro["pred"]
        single_specular = pro["spec_pred"]
        single_mask = pro["mask_pred"]

        if detach_backbones:
            world_diffuse = world_diffuse.detach()
            world_specular = world_specular.detach()
            world_mask = world_mask.detach()
            world_normal = world_normal.detach()
            world_albedo = world_albedo.detach()
            world_alpha = world_alpha.detach()
            single_pred = single_pred.detach()
            single_specular = single_specular.detach()
            single_mask = single_mask.detach()

        reconstructed = (world_diffuse + world_specular).clamp(0, 1)
        error = (image - reconstructed).abs().mean(dim=1, keepdim=True)
        analytic_confidence = world_alpha * torch.exp(
            -error / self.reconstruction_temperature
        )
        analytic_confidence = analytic_confidence.clamp(1e-4, 1 - 1e-4)
        analytic_logit = torch.logit(analytic_confidence)

        features = torch.cat(
            [
                image,
                world_diffuse,
                world_specular,
                world_mask,
                world_normal,
                world_albedo,
                world_alpha,
                single_pred,
                single_specular,
                single_mask,
            ],
            dim=1,
        )
        gate = torch.sigmoid(analytic_logit + self.gate(features))
        fused = gate * world_diffuse + (1 - gate) * single_pred
        uncertainty = 4 * gate * (1 - gate)
        correction_input = torch.cat(
            [image, fused, world_mask, single_mask, uncertainty], dim=1
        )
        correction = (
            self.correction(correction_input)
            * uncertainty
            * self.correction_limit
        )
        prediction = (fused + correction).clamp(0, 1)
        return {
            "pred": prediction,
            "fused": fused,
            "single_pred": single_pred,
            "world_diffuse": world_diffuse,
            "world_reconstruction_error": error,
            "gate": gate,
            "single": pro,
            "world": world,
            "mode": "hybrid",
        }

    def fusion_parameters(self):
        yield from self.gate.parameters()
        yield from self.correction.parameters()
