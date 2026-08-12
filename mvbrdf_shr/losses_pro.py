"""Enhanced losses for SOTA push: L1 + SSIM + gradient + highlight-weighted."""
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
    cons: float = 0.05
    spec_ndf: float = 0.05
    mask: float = 0.3
    refine: float = 1.5
    refine_ssim: float = 0.5
    refine_grad: float = 0.2
    hl_weight: float = 2.0
    smooth: float = 0.005
    specular: float = 2.0
    coarse: float = 1.0
    mask_pred: float = 0.5
    refine_mse: float = 0.0
    mask_dice: float = 0.5
    spec_highlight_weight: float = 4.0
    # Perceptual + frequency losses for the SOTA push (paths B). Both default
    # off (0.0) so existing runs are unchanged; enable via config/CLI. LPIPS
    # sharpens high-frequency texture that L1/SSIM under-weight (the gap to
    # NeuralDRM is largely in highlight-edge detail); FFT loss directly targets
    # the frequency content that DHAN's FrequencyProcessor is built around.
    refine_lpips: float = 0.0
    refine_fft: float = 0.0


def _ssim_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    C1, C2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(pred, 7, 1, 3)
    mu_y = F.avg_pool2d(gt, 7, 1, 3)
    sigma_x = F.avg_pool2d(pred * pred, 7, 1, 3) - mu_x * mu_x
    sigma_y = F.avg_pool2d(gt * gt, 7, 1, 3) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * gt, 7, 1, 3) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + 1e-8
    )
    return 1.0 - ssim.mean()


def _grad_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    def g(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    px, py = g(pred)
    gx, gy = g(gt)
    return (px - gx).abs().mean() + (py - gy).abs().mean()


def _fft_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Frequency-domain L1 on the magnitude spectrum.

    Complements spatial L1/SSIM: highlight removal errors are concentrated at
    high frequencies (sharp specular edges, fine texture under the highlight),
    which spatial losses under-weight. Aligning the FFT magnitude spectrum
    directly penalizes these errors and mirrors the inductive bias of DHAN's
    FrequencyProcessor.
    """
    pred_mag = torch.fft.fft2(pred, norm="ortho").abs()
    gt_mag = torch.fft.fft2(gt, norm="ortho").abs()
    return (pred_mag - gt_mag).abs().mean()


def total_loss(
    out: dict,
    image_clean: torch.Tensor | None = None,
    mask_gt: torch.Tensor | None = None,
    weights: LossWeights | None = None,
    specular_gt: torch.Tensor | None = None,
    lpips_fn=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    w = weights or LossWeights()
    render: RenderOutputs = out["render"]
    mat: MaterialParams = out["mat"]
    image = out["image"]
    pred = out["pred"]

    logs: dict[str, float] = {}
    loss = image.new_zeros(())

    l_photo = F.l1_loss(render.full, image)
    loss = loss + w.photo * l_photo
    logs["photo"] = float(l_photo.detach())

    if image_clean is not None:
        l_diff = F.l1_loss(render.diffuse, image_clean)
        loss = loss + w.diffuse * l_diff
        logs["diffuse"] = float(l_diff.detach())

        # Highlight-weighted refine L1
        if mask_gt is not None:
            weight = 1.0 + w.hl_weight * mask_gt
            l_ref = ((pred - image_clean).abs() * weight).mean()
        else:
            l_ref = F.l1_loss(pred, image_clean)
        loss = loss + w.refine * l_ref
        logs["refine"] = float(l_ref.detach())

        if w.refine_mse > 0:
            l_mse = F.mse_loss(pred, image_clean)
            loss = loss + w.refine_mse * l_mse
            logs["mse"] = float(l_mse.detach())

        if w.refine_ssim > 0:
            l_ssim = _ssim_loss(pred, image_clean)
            loss = loss + w.refine_ssim * l_ssim
            logs["ssim"] = float(l_ssim.detach())
        if w.refine_grad > 0:
            l_g = _grad_loss(pred, image_clean)
            loss = loss + w.refine_grad * l_g
            logs["grad"] = float(l_g.detach())

        # Perceptual (LPIPS) loss — sharpens highlight-edge texture that L1/SSIM
        # under-weight. lpips_fn is a cached criterion held by the caller (VGG
        # weights are expensive to reload). Input must be in [-1, 1].
        if w.refine_lpips > 0 and lpips_fn is not None:
            l_lpips = lpips_fn(pred.clamp(0, 1) * 2 - 1, image_clean * 2 - 1)
            loss = loss + w.refine_lpips * l_lpips
            logs["lpips"] = float(l_lpips.detach())

        # Frequency-domain L1 — directly aligns the FFT magnitude spectrum,
        # targeting high-frequency highlight-removal errors.
        if w.refine_fft > 0:
            l_fft = _fft_l1(pred, image_clean)
            loss = loss + w.refine_fft * l_fft
            logs["fft"] = float(l_fft.detach())

        if "coarse" in out:
            l_coarse = F.l1_loss(out["coarse"], image_clean)
            loss = loss + w.coarse * l_coarse
            logs["coarse"] = float(l_coarse.detach())

    if specular_gt is not None and "spec_pred" in out:
        spec_weight = 1.0
        if mask_gt is not None:
            spec_weight = spec_weight + w.spec_highlight_weight * mask_gt
        l_spec_rgb = ((out["spec_pred"] - specular_gt).abs() * spec_weight).mean()
        l_spec_grad = _grad_loss(out["spec_pred"], specular_gt)
        l_spec = l_spec_rgb + 0.2 * l_spec_grad
        loss = loss + w.specular * l_spec
        logs["specular_rgb"] = float(l_spec_rgb.detach())

    if mask_gt is not None and "mask_logit" in out:
        positives = mask_gt.sum().clamp(min=1)
        negatives = mask_gt.numel() - positives
        positive_weight = (negatives / positives).clamp(1, 20).detach()
        l_mask_pred = F.binary_cross_entropy_with_logits(
            out["mask_logit"], mask_gt, pos_weight=positive_weight
        )
        probability = torch.sigmoid(out["mask_logit"])
        intersection = (probability * mask_gt).sum()
        l_dice = 1 - (2 * intersection + 1) / (
            probability.sum() + mask_gt.sum() + 1
        )
        loss = loss + w.mask_pred * l_mask_pred + w.mask_dice * l_dice
        logs["mask_pred"] = float(l_mask_pred.detach())
        logs["mask_dice"] = float(l_dice.detach())

    l_cons = BRDFRenderer.energy_loss(mat)
    loss = loss + w.cons * l_cons
    logs["cons"] = float(l_cons.detach())

    spec_energy = render.specular.mean(dim=1, keepdim=True)
    l_ndf = (mat.albedo.mean(dim=1, keepdim=True) * spec_energy.detach()).mean()
    loss = loss + w.spec_ndf * l_ndf
    logs["spec_ndf"] = float(l_ndf.detach())

    if mask_gt is not None:
        l_mask = F.l1_loss(render.mask, mask_gt)
        loss = loss + w.mask * l_mask
        logs["mask"] = float(l_mask.detach())

    def tv(x):
        return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (
            x[:, :, 1:, :] - x[:, :, :-1, :]
        ).abs().mean()

    l_sm = tv(mat.albedo) + tv(mat.roughness)
    loss = loss + w.smooth * l_sm
    logs["smooth"] = float(l_sm.detach())
    logs["total"] = float(loss.detach())
    return loss, logs
