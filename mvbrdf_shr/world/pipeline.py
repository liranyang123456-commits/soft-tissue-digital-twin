"""End-to-end world-space 3D Gaussian BRDF highlight-removal pipeline."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.dual_latent_brdf import DualLatentNeuralBRDF
from ..models.implicit_material_field import ImplicitMaterialImagingField
from .camera import PerspectiveCamera
from .lighting import NeuralIncidentLightField, PerFrameLighting, WorldLight
from .refiner import PhysicsGuidedWorldRefiner
from .renderer import GaussianBRDFRenderer
from .scene import GaussianBRDFField


class WorldGaussianBRDFPipeline(nn.Module):
    """Persistent 3D scene -> physical decomposition -> optional refinement."""

    def __init__(
        self,
        scene: GaussianBRDFField,
        n_frames: int,
        lighting_mode: str = "sh",
        use_refiner: bool = True,
        refiner_base: int = 96,
        chunk_size: int = 64,
        raster_backend: str = "auto",
        shared_lighting: bool = False,
        background_color: tuple[float, float, float] | list[float] | None = None,
        linear_hdr: bool = False,
        shadow_mode: str = "none",
        shadow_strength: float = 1.0,
        light_ids: list[int | None] | None = None,
        use_implicit_material: bool = False,
        implicit_material_hidden: int = 64,
        implicit_material_layers: int = 3,
        implicit_material_residual_limit: float = 0.10,
        use_neural_brdf: bool = False,
        neural_brdf_material_latent: int = 32,
        neural_brdf_light_latent: int = 16,
        neural_brdf_hidden: int = 96,
        neural_brdf_strength: float = 1.0,
        neural_brdf_mode: str = "replacement",
        neural_brdf_residual_limit: float = 0.35,
        neural_brdf_highlight_gate: bool = False,
        neural_brdf_boundary_fallback: bool = False,
    ):
        super().__init__()
        if lighting_mode not in {"sh", "neilf"}:
            raise ValueError("lighting_mode must be 'sh' or 'neilf'")
        if use_implicit_material and use_neural_brdf:
            raise ValueError(
                "implicit material residual and dual-latent neural BRDF are "
                "mutually exclusive"
            )
        self.scene = scene
        self.lighting_mode = lighting_mode
        self.linear_hdr = linear_hdr
        self.register_buffer(
            "background_color",
            None
            if background_color is None
            else torch.tensor(background_color, dtype=torch.float32).view(3, 1, 1),
            persistent=False,
        )
        self.frame_lighting = PerFrameLighting(
            n_frames, shared=shared_lighting, light_ids=light_ids
        )
        self.neural_light = (
            NeuralIncidentLightField() if lighting_mode == "neilf" else None
        )
        self.implicit_material = (
            ImplicitMaterialImagingField(
                hidden=implicit_material_hidden,
                layers=implicit_material_layers,
                residual_limit=implicit_material_residual_limit,
            )
            if use_implicit_material
            else None
        )
        self.neural_brdf = (
            DualLatentNeuralBRDF(
                material_latent=neural_brdf_material_latent,
                light_latent=neural_brdf_light_latent,
                hidden=neural_brdf_hidden,
                mode=neural_brdf_mode,
                residual_limit=neural_brdf_residual_limit,
                highlight_gate=neural_brdf_highlight_gate,
            )
            if use_neural_brdf
            else None
        )
        self.neural_brdf_strength = float(neural_brdf_strength)
        self.neural_brdf_boundary_fallback = bool(
            neural_brdf_boundary_fallback
        )
        self.renderer = GaussianBRDFRenderer(
            chunk_size=chunk_size,
            backend=raster_backend,
            output_clamp=not linear_hdr,
            shadow_mode=shadow_mode,
            shadow_strength=shadow_strength,
        )
        self.refiner = (
            PhysicsGuidedWorldRefiner(base=refiner_base) if use_refiner else None
        )

    def forward(
        self,
        camera: PerspectiveCamera,
        frame_id: int,
        image: torch.Tensor | None = None,
        refine: bool = True,
        apply_sensor: bool = True,
        light_override: WorldLight | None = None,
    ) -> dict:
        if self.lighting_mode == "neilf":
            render = self.renderer(
                self.scene,
                camera,
                neural_light=self.neural_light,
                implicit_material=self.implicit_material,
                neural_brdf=self.neural_brdf,
                neural_brdf_strength=self.neural_brdf_strength,
            )
        else:
            selected_light = (
                light_override
                if light_override is not None
                else self.frame_lighting(frame_id)
            )
            render = self.renderer(
                self.scene,
                camera,
                light=selected_light,
                implicit_material=self.implicit_material,
                neural_brdf=self.neural_brdf,
                neural_brdf_strength=self.neural_brdf_strength,
            )
            if (
                self.neural_brdf is not None
                and self.neural_brdf_boundary_fallback
            ):
                explicit = self.renderer(
                    self.scene,
                    camera,
                    light=selected_light,
                    implicit_material=None,
                    neural_brdf=None,
                )
                alpha = render.alpha.unsqueeze(0)
                eroded = -F.max_pool2d(-alpha, 5, 1, 2)[0]
                interior = (eroded / render.alpha.clamp_min(1e-6)).clamp(0, 1)
                highlight = torch.sigmoid(
                    60 * (explicit.specular.mean(dim=0, keepdim=True) - 0.05)
                )
                interior = interior * highlight
                render.full = (
                    explicit.full * (1 - interior) + render.full * interior
                )
                render.specular = (
                    explicit.specular * (1 - interior)
                    + render.specular * interior
                )
        exposure, white_balance = self.frame_lighting.sensor(frame_id)
        if apply_sensor:
            sensor_gain = exposure.view(1, 1, 1) * white_balance.view(3, 1, 1)
            render.full = render.full * sensor_gain
            render.diffuse = render.diffuse * sensor_gain
            render.specular = render.specular * sensor_gain
        if self.background_color is not None:
            background = self.background_color * (1 - render.alpha)
            render.full = render.full + background
            render.diffuse = render.diffuse + background
        pred = render.diffuse
        if refine and self.refiner is not None and image is not None:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            pred = self.refiner(
                image,
                render.diffuse.unsqueeze(0),
                render.mask.unsqueeze(0),
                render.normal.unsqueeze(0),
                render.depth.unsqueeze(0),
                render.projected_texture.unsqueeze(0),
                render.albedo.unsqueeze(0),
                render.alpha.unsqueeze(0),
            )[0]
        return {
            "render": render,
            "pred": pred,
            "image": image[0] if image is not None and image.ndim == 4 else image,
            "materials": self.scene.materials(),
            "implicit_residual": render.implicit_residual,
            "sensor": {
                "exposure": exposure,
                "white_balance": white_balance,
            },
        }

    def load_shiq_refiner(self, checkpoint: str) -> int:
        if self.refiner is None:
            raise RuntimeError("pipeline has no refiner")
        return self.refiner.load_shiq_restorer(checkpoint)

