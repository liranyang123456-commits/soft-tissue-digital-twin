"""Losses for MVBRDF-SHR."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .models.renderer import BRDFRenderer
from .types import MaterialParams, RenderOutputs


@dataclass
class LossWeights:
    photo: float = 1.0
    diffuse: float = 1.0
    cons: float = 0.1
    spec_ndf: float = 0.05
    mask: float = 0.2
    refine: float = 1.0
    smooth: float = 0.01


def total_loss(
    out: dict,
    image_clean: torch.Tensor | None = None,
    mask_gt: torch.Tensor | None = None,
    weights: LossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute training loss and a detached metrics dict."""
    w = weights or LossWeights()
    render: RenderOutputs = out["render"]
    mat: MaterialParams = out["mat"]
    image = out["image"]
    pred = out["pred"]

    logs: dict[str, float] = {}
    loss = image.new_zeros(())

    # Photometric: full render should match input (with highlights)
    l_photo = F.l1_loss(render.full, image)
    loss = loss + w.photo * l_photo
    logs["photo"] = float(l_photo.detach())

    # Diffuse supervision when GT clean available
    if image_clean is not None:
        l_diff = F.l1_loss(render.diffuse, image_clean)
        loss = loss + w.diffuse * l_diff
        logs["diffuse"] = float(l_diff.detach())
        l_ref = F.l1_loss(pred, image_clean)
        loss = loss + w.refine * l_ref
        logs["refine"] = float(l_ref.detach())

    # Energy conservation on material coeffs
    l_cons = BRDFRenderer.energy_loss(mat)
    loss = loss + w.cons * l_cons
    logs["cons"] = float(l_cons.detach())

    # NDF-inspired: where specular is strong, albedo should not absorb the highlight
    # → penalize high albedo in high specular-render regions
    spec_energy = render.specular.mean(dim=1, keepdim=True)
    l_ndf = (mat.albedo.mean(dim=1, keepdim=True) * spec_energy.detach()).mean()
    loss = loss + w.spec_ndf * l_ndf
    logs["spec_ndf"] = float(l_ndf.detach())

    if mask_gt is not None:
        l_mask = F.l1_loss(render.mask, mask_gt)
        loss = loss + w.mask * l_mask
        logs["mask"] = float(l_mask.detach())

    # Mild spatial smoothness on albedo / roughness
    def tv(x: torch.Tensor) -> torch.Tensor:
        return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (
            x[:, :, 1:, :] - x[:, :, :-1, :]
        ).abs().mean()

    l_sm = tv(mat.albedo) + tv(mat.roughness)
    loss = loss + w.smooth * l_sm
    logs["smooth"] = float(l_sm.detach())
    logs["total"] = float(loss.detach())
    return loss, logs
