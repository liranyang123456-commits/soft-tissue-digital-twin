"""End-to-end MVBRDF-SHR pipeline."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..types import Batch, RenderOutputs
from .generative_refine import GenerativeRefiner
from .geometry_field import ScreenGeometryField
from .light_field import LightField
from .material_field import MaterialField, MultiViewMaterialAggregator
from .renderer import BRDFRenderer


class MVBRDFSHR(nn.Module):
    """Multi-View BRDF Field Specular Highlight Removal.

    Stages:
      1 GeometryField  → normal / depth
      2 MaterialField  → ρ_d, ρ_s, α, m, absorption
      3 LightField     → SH + dominant light
      4 BRDFRenderer   → I_full / I_diff / M
      5 GenerativeRefiner (optional) → Î
    """

    def __init__(self, use_generative: bool = True, resolution: int = 256):
        super().__init__()
        self.resolution = resolution
        self.geometry = ScreenGeometryField()
        self.material = MaterialField(in_ch=6)
        self.light = LightField()
        self.aggregator = MultiViewMaterialAggregator()
        self.renderer = BRDFRenderer()
        self.refiner = GenerativeRefiner() if use_generative else None

    def _prepare_image(self, img: torch.Tensor) -> torch.Tensor:
        """Accept (B,3,H,W) or (B,V,3,H,W); return primary view + optional views."""
        if img.dim() == 4:
            return img, None
        if img.dim() == 5:
            return img[:, 0], img
        raise ValueError(f"expected 4D or 5D image, got {img.shape}")

    def forward_physics(
        self,
        image: torch.Tensor,
        light_int: torch.Tensor | None = None,
        multi: torch.Tensor | None = None,
    ) -> dict:
        """Run geometry → material → light → render.

        If `multi` is (B,V,3,H,W), aggregate materials across views.
        """
        geom = self.geometry(image)
        if multi is not None:
            mats = []
            V = multi.shape[1]
            for v in range(V):
                g_v = self.geometry(multi[:, v])
                mats.append(self.material(multi[:, v], g_v.normal))
            mat = self.aggregator(mats)
        else:
            mat = self.material(image, geom.normal)
        light = self.light(image, light_int_scale=light_int)
        render: RenderOutputs = self.renderer(geom, mat, light, input_image=image)
        return {
            "geom": geom,
            "mat": mat,
            "light": light,
            "render": render,
        }

    def forward(
        self,
        image: torch.Tensor,
        light_int: torch.Tensor | None = None,
        use_refine: bool = True,
    ) -> dict:
        primary, multi = self._prepare_image(image)
        out = self.forward_physics(primary, light_int=light_int, multi=multi)
        render = out["render"]
        pred = render.diffuse
        if use_refine and self.refiner is not None:
            pred = self.refiner(
                primary, render.diffuse, render.mask, out["geom"].normal
            )
        out["pred"] = pred
        out["image"] = primary
        return out

    def forward_batch(self, batch: Batch, use_refine: bool = True) -> dict:
        light = None
        if batch.light_ints is not None:
            li = batch.light_ints
            light = li[:, 0] if li.dim() == 2 else li
        return self.forward(batch.image, light_int=light, use_refine=use_refine)
