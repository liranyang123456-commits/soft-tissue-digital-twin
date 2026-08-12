"""Upgraded end-to-end pipeline targeting SOTA numbers."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..types import RenderOutputs
from .generative_refine import GenerativeRefiner
from .geometry_field import ScreenGeometryField
from .light_field import LightField
from .material_field import MaterialField, MultiViewMaterialAggregator
from .renderer import BRDFRenderer
from .restorer import DHANRestorer, HighlightRestorer, SpecularDecomposer


class MVBRDFSHRPro(nn.Module):
    """Physics BRDF fields + strong HighlightRestorer (SOTA push).

    restorer_kind selects the refinement backbone (path A):
    - "unet"  : the original multi-scale residual UNet with SE attention.
    - "dhan"  : physics-conditioned DHAN Processor (SOTA dual-domain Transformer
               + FFT frequency + dual attention), with the physics prior
               (albedo/normal/mask) injected via a zero-init side branch.
    """

    def __init__(
        self,
        resolution: int = 200,
        base_ch: int = 48,
        restorer_base: int = 64,
        use_strong_restorer: bool = True,
        restorer_kind: str = "unet",
    ):
        super().__init__()
        self.resolution = resolution
        self.geometry = ScreenGeometryField(base_ch=base_ch)
        self.material = MaterialField(in_ch=6, base_ch=base_ch)
        self.light = LightField(hidden=96)
        self.aggregator = MultiViewMaterialAggregator()
        self.renderer = BRDFRenderer()
        self.decomposer = SpecularDecomposer(base=restorer_base, n_res=2)
        self.use_strong_restorer = use_strong_restorer
        self.restorer_kind = restorer_kind
        if not use_strong_restorer:
            self.restorer = GenerativeRefiner(backend="tiny")
        elif restorer_kind == "dhan":
            self.restorer = DHANRestorer(dim=36)
        else:
            self.restorer = HighlightRestorer(in_ch=13, base=restorer_base, n_res=2)

    def _prepare_image(self, img: torch.Tensor):
        if img.dim() == 4:
            return img, None
        if img.dim() == 5:
            return img[:, 0], img
        raise ValueError(f"bad shape {img.shape}")

    def forward_physics(self, image, light_int=None, multi=None):
        geom = self.geometry(image)
        if multi is not None:
            mats = []
            for v in range(multi.shape[1]):
                g_v = self.geometry(multi[:, v])
                mats.append(self.material(multi[:, v], g_v.normal))
            mat = self.aggregator(mats)
        else:
            mat = self.material(image, geom.normal)
        light = self.light(image, light_int_scale=light_int)
        render = self.renderer(geom, mat, light, input_image=image)
        return {"geom": geom, "mat": mat, "light": light, "render": render}

    def forward(self, image, light_int=None, use_refine: bool = True, detach_physics: bool = False):
        primary, multi = self._prepare_image(image)
        out = self.forward_physics(primary, light_int=light_int, multi=multi)
        render: RenderOutputs = out["render"]
        phys_spec, phys_mask = render.specular, render.mask
        if detach_physics:
            phys_spec, phys_mask = phys_spec.detach(), phys_mask.detach()
        spec_pred, mask_pred, mask_logit, coarse = self.decomposer(
            primary, phys_spec, phys_mask
        )
        out["spec_pred"] = spec_pred
        out["mask_pred"] = mask_pred
        out["mask_logit"] = mask_logit
        out["coarse"] = coarse
        if use_refine:
            if self.use_strong_restorer:
                # The analytically reconstructed A-S is the strongest SHIQ prior.
                diff, msk, nrm, alb = coarse, mask_pred, out["geom"].normal, out["mat"].albedo
                if detach_physics:
                    nrm, alb = nrm.detach(), alb.detach()
                pred = self.restorer(primary, diff, msk, nrm, alb)
            else:
                pred = self.restorer(primary, render.diffuse, render.mask, out["geom"].normal)
        else:
            pred = render.diffuse
        out["pred"] = pred
        out["image"] = primary
        return out
