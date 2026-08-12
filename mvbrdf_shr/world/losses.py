"""Losses for joint world-space geometry, material, light, and refinement."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .scene import GaussianBRDFField


@dataclass(slots=True)
class WorldLossWeights:
    photo_l1: float = 1.0
    photo_ssim: float = 0.2
    photo_log: float = 0.0
    photo_highlight: float = 0.0
    diffuse_gt: float = 1.0
    refine_gt: float = 1.0
    specular_gt: float = 0.5
    albedo_gt: float = 1.0
    normal_gt: float = 0.5
    depth_gt: float = 0.2
    # Soft visual-hull depth (NoGT); distinct from evaluation GT depth.
    depth_prior: float = 0.0
    energy: float = 0.1
    material_smooth: float = 0.02
    normal_axis: float = 0.02
    scale: float = 0.005
    opacity_sparse: float = 0.001
    texture_consistency: float = 0.02
    exposure_prior: float = 0.001
    white_balance_prior: float = 0.001
    silhouette_bce: float = 0.0
    silhouette_dice: float = 0.0
    background_leak: float = 0.0
    depth_smooth: float = 0.0
    normal_smooth: float = 0.0
    normal_edge_aware: float = 0.0
    normal_planar: float = 0.0
    normal_prior: float = 0.0
    normal_depth_consistency: float = 0.0
    depth_from_normal: float = 0.0
    normal_depth_scales: int = 1
    implicit_residual: float = 0.0


def _ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred.ndim == 3:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    c1, c2 = 0.01**2, 0.03**2
    mean_p = F.avg_pool2d(pred, 7, 1, 3)
    mean_t = F.avg_pool2d(target, 7, 1, 3)
    var_p = F.avg_pool2d(pred.square(), 7, 1, 3) - mean_p.square()
    var_t = F.avg_pool2d(target.square(), 7, 1, 3) - mean_t.square()
    covariance = F.avg_pool2d(pred * target, 7, 1, 3) - mean_p * mean_t
    ssim = ((2 * mean_p * mean_t + c1) * (2 * covariance + c2)) / (
        (mean_p.square() + mean_t.square() + c1) * (var_p + var_t + c2) + 1e-8
    )
    if mask is not None:
        valid = mask.to(ssim) > 0.5
        if valid.ndim == 3:
            valid = valid.unsqueeze(0)
        valid = valid.expand_as(ssim)
        if valid.any():
            return 1 - ssim[valid].mean()
    return 1 - ssim.mean()


def _material_smoothness(scene: GaussianBRDFField, max_points: int = 512) -> torch.Tensor:
    count = min(scene.n_gaussians, max_points)
    ids = torch.randperm(scene.n_gaussians, device=scene.means.device)[:count]
    points = scene.means[ids]
    distance = torch.cdist(points.detach(), points.detach())
    distance.fill_diagonal_(float("inf"))
    neighbor = distance.argmin(dim=1)
    material = scene.materials()
    features = torch.cat(
        [
            material.albedo,
            material.specular,
            material.roughness,
            material.metalness,
            material.absorption,
        ],
        dim=-1,
    )[ids]
    spatial_weight = torch.exp(-distance[torch.arange(count), neighbor].detach() * 20)
    return (
        (features - features[neighbor]).abs().mean(-1) * spatial_weight
    ).mean()


def _normal_planar_consistency(
    scene: GaussianBRDFField, max_points: int = 512, neighbors: int = 4
) -> torch.Tensor:
    """Encourage locally adjacent surface splats to share a tangent plane."""
    count = min(scene.n_gaussians, max_points)
    if count < 2:
        return scene.means.new_zeros(())
    ids = torch.randperm(scene.n_gaussians, device=scene.means.device)[:count]
    points = scene.means[ids]
    distance = torch.cdist(points.detach(), points.detach())
    distance.fill_diagonal_(float("inf"))
    k = min(neighbors, count - 1)
    values, indices = distance.topk(k, dim=1, largest=False)
    normals = scene.normals[ids]
    neighbor_normals = normals[indices]
    cosine = (
        normals[:, None] * neighbor_normals
    ).sum(dim=-1).abs().clamp(0, 1)
    local_scale = scene.scales[ids].topk(2, dim=-1).values.mean(
        dim=-1, keepdim=True
    ).detach()
    spatial = torch.exp(-values / local_scale.clamp_min(1e-4))
    color = scene.texture[ids].detach()
    color_edge = torch.exp(
        -8.0 * (color[:, None] - color[indices]).abs().mean(dim=-1)
    )
    weight = spatial * color_edge
    return ((1 - cosine) * weight).sum() / weight.sum().clamp_min(1e-6)


def _surface_terms(
    depth: torch.Tensor,
    alpha: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    valid_mask: torch.Tensor | None,
):
    """Unproject a depth map and build pseudo-normals, view dirs, and mask.

    Returns the world-space pseudo-normal (gradients preserved), the detached
    per-pixel view direction, and the valid-pixel mask over the image interior.
    """
    h, w = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
    directions = pix @ torch.linalg.inv(K).T
    points_cam = directions * depth[..., None]

    span_x = points_cam[1:-1, 2:] - points_cam[1:-1, :-2]
    span_y = points_cam[2:, 1:-1] - points_cam[:-2, 1:-1]
    pseudo = F.normalize(torch.cross(span_y, span_x, dim=-1), dim=-1, eps=1e-6)
    rotation = c2w[:3, :3]
    pseudo_world = F.normalize(pseudo @ rotation.T, dim=-1, eps=1e-6)

    points_world = points_cam[1:-1, 1:-1] @ rotation.T + c2w[:3, 3]
    view_dir = F.normalize(c2w[:3, 3] - points_world, dim=-1, eps=1e-6).detach()

    center_depth = depth[1:-1, 1:-1]
    continuity = (
        (depth[1:-1, 2:] - depth[1:-1, :-2]).abs()
        <= 0.10 * center_depth.clamp_min(1e-4)
    ) & (
        (depth[2:, 1:-1] - depth[:-2, 1:-1]).abs()
        <= 0.10 * center_depth.clamp_min(1e-4)
    )
    foreground = alpha[1:-1, 1:-1] > 0.5
    if valid_mask is not None:
        foreground = foreground & (
            valid_mask.to(depth).squeeze(0)[1:-1, 1:-1] > 0.5
        )
    mask = foreground & continuity & (center_depth > 1e-4)
    return pseudo_world, view_dir, mask


def _orient(vectors: torch.Tensor, view_dir: torch.Tensor) -> torch.Tensor:
    """Flip vectors toward the camera using a detached sign decision."""
    sign = torch.where(
        (vectors.detach() * view_dir).sum(dim=-1, keepdim=True) < 0,
        -1.0,
        1.0,
    )
    return vectors * sign


def _scale_intrinsics(K: torch.Tensor, factor: float) -> torch.Tensor:
    scaled = K.clone()
    scaled[0, 0] = scaled[0, 0] / factor
    scaled[1, 1] = scaled[1, 1] / factor
    scaled[0, 2] = scaled[0, 2] / factor
    scaled[1, 2] = scaled[1, 2] / factor
    return scaled


def _downsample(tensor: torch.Tensor, factor: float) -> torch.Tensor:
    """Area-average a (H,W) or (C,H,W) map by an integer factor."""
    if factor == 1:
        return tensor
    squeeze = tensor.ndim == 2
    if squeeze:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    out = F.avg_pool2d(tensor, kernel_size=int(factor))
    if squeeze:
        return out.squeeze(0).squeeze(0)
    return out.squeeze(0)


def _depth_normal_consistency(
    output: dict,
    camera,
    valid_mask: torch.Tensor | None,
    scales: int = 1,
) -> torch.Tensor:
    """Align rendered normals with pseudo-normals derived from rendered depth.

    The pseudo-normal is detached so gradients reshape the shading normals
    rather than the geometry.  When ``scales`` > 1 the loss is also averaged
    over coarser, area-downsampled maps so both fine detail and broader
    surface structure constrain the normal field.
    """
    render = output["render"]
    depth0 = render.depth.squeeze(0)
    alpha0 = render.alpha.squeeze(0)
    normal0 = render.normal  # (3,H,W)
    total = depth0.new_zeros(())
    count = 0
    for level in range(max(int(scales), 1)):
        factor = 2 ** level
        depth = _downsample(depth0, factor)
        alpha = _downsample(alpha0, factor)
        normal = _downsample(normal0, factor)
        if depth.shape[0] < 3 or depth.shape[1] < 3:
            break
        K = _scale_intrinsics(camera.K, float(factor))
        mask_full = None
        if valid_mask is not None:
            mask_full = _downsample(
                valid_mask.to(depth0).squeeze(0), factor
            ).unsqueeze(0)
        pseudo_world, view_dir, mask = _surface_terms(
            depth, alpha, K, camera.c2w, mask_full
        )
        pseudo_world = _orient(pseudo_world, view_dir).detach()
        rendered = normal[:, 1:-1, 1:-1].permute(1, 2, 0)
        rendered = _orient(rendered, view_dir)
        if not bool(mask.any()):
            continue
        cosine = (rendered * pseudo_world).sum(dim=-1).clamp(-1.0, 1.0)
        total = total + (1.0 - cosine)[mask].mean()
        count += 1
    if count == 0:
        return depth0.new_zeros(())
    return total / count


def _normal_to_depth_consistency(
    output: dict,
    camera,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Align rendered depth with shading normals (the reverse coupling).

    Symmetric to :func:`_depth_normal_consistency`, but here the depth-derived
    pseudo-normal keeps its gradients while the shading normal is detached, so
    the surface geometry is pulled toward the orientation predicted by
    shading.  This lets photometric normals refine geometry in regions where
    the multi-view depth is poorly constrained.
    """
    render = output["render"]
    depth = render.depth.squeeze(0)
    alpha = render.alpha.squeeze(0)
    pseudo_world, view_dir, mask = _surface_terms(
        depth, alpha, camera.K, camera.c2w, valid_mask
    )
    if not bool(mask.any()):
        return depth.new_zeros(())
    pseudo_world = _orient(pseudo_world, view_dir)
    rendered = render.normal[:, 1:-1, 1:-1].permute(1, 2, 0)
    rendered = _orient(rendered, view_dir).detach()
    cosine = (pseudo_world * rendered).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cosine)[mask].mean()


def world_total_loss(
    output: dict,
    target: torch.Tensor,
    scene: GaussianBRDFField,
    diffuse_gt: torch.Tensor | None = None,
    specular_gt: torch.Tensor | None = None,
    albedo_gt: torch.Tensor | None = None,
    normal_gt: torch.Tensor | None = None,
    depth_gt: torch.Tensor | None = None,
    depth_prior: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    weights: WorldLossWeights | None = None,
    camera=None,
    step: int = 0,
    normal_depth_start: int = 0,
    depth_from_normal_start: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or WorldLossWeights()
    render = output["render"]
    target = target.to(render.full)
    loss = target.new_zeros(())
    logs: dict[str, float] = {}

    photo_mask = None
    if valid_mask is not None:
        photo_mask = (valid_mask.to(render.full) > 0.5).expand_as(render.full)
    photo = (
        (render.full - target).abs()[photo_mask].mean()
        if photo_mask is not None and photo_mask.any()
        else F.l1_loss(render.full, target)
    )
    photo_ssim = _ssim_loss(render.full, target, valid_mask)
    log_photo = F.l1_loss(
        torch.log1p(render.full.clamp_min(0)),
        torch.log1p(target.clamp_min(0)),
    )
    luminance = target.mean(dim=0, keepdim=True)
    foreground = (
        valid_mask.to(luminance) > 0.5
        if valid_mask is not None
        else torch.ones_like(luminance, dtype=torch.bool)
    )
    if bool(foreground.any()):
        highlight_threshold = torch.quantile(
            luminance[foreground].detach(), 0.9
        )
        highlight_mask = foreground & (luminance >= highlight_threshold)
        highlight_photo = (
            (render.full - target)
            .abs()[highlight_mask.expand_as(render.full)]
            .mean()
        )
    else:
        highlight_photo = target.new_zeros(())
    loss = (
        loss
        + weights.photo_l1 * photo
        + weights.photo_ssim * photo_ssim
        + weights.photo_log * log_photo
        + weights.photo_highlight * highlight_photo
    )
    logs["photo"] = float(photo.detach())
    logs["photo_ssim"] = float(photo_ssim.detach())
    logs["photo_log"] = float(log_photo.detach())
    logs["photo_highlight"] = float(highlight_photo.detach())
    implicit_residual = output.get("implicit_residual")
    if isinstance(implicit_residual, torch.Tensor):
        implicit_penalty = implicit_residual.square().mean()
        loss = loss + weights.implicit_residual * implicit_penalty
        logs["implicit_residual"] = float(implicit_penalty.detach())

    if valid_mask is not None:
        silhouette = (valid_mask.to(render.alpha) > 0.5).float()
        alpha = render.alpha.clamp(1e-5, 1 - 1e-5)
        silhouette_bce = F.binary_cross_entropy(alpha, silhouette)
        intersection = (alpha * silhouette).sum()
        silhouette_dice = 1 - (2 * intersection + 1) / (
            alpha.sum() + silhouette.sum() + 1
        )
        background = silhouette < 0.5
        background_leak = (
            alpha[background].mean() if background.any() else alpha.new_zeros(())
        )
        foreground_x = silhouette[:, :, 1:] * silhouette[:, :, :-1]
        foreground_y = silhouette[:, 1:, :] * silhouette[:, :-1, :]
        depth_dx = (render.depth[:, :, 1:] - render.depth[:, :, :-1]).abs()
        depth_dy = (render.depth[:, 1:, :] - render.depth[:, :-1, :]).abs()
        depth_smooth = (
            (depth_dx * foreground_x).sum() / foreground_x.sum().clamp_min(1)
            + (depth_dy * foreground_y).sum() / foreground_y.sum().clamp_min(1)
        )
        normal_dx = (render.normal[:, :, 1:] - render.normal[:, :, :-1]).abs().mean(0, keepdim=True)
        normal_dy = (render.normal[:, 1:, :] - render.normal[:, :-1, :]).abs().mean(0, keepdim=True)
        normal_smooth = (
            (normal_dx * foreground_x).sum() / foreground_x.sum().clamp_min(1)
            + (normal_dy * foreground_y).sum() / foreground_y.sum().clamp_min(1)
        )
        image_dx = (target[:, :, 1:] - target[:, :, :-1]).abs().mean(
            0, keepdim=True
        )
        image_dy = (target[:, 1:, :] - target[:, :-1, :]).abs().mean(
            0, keepdim=True
        )
        edge_x = torch.exp(-10.0 * image_dx)
        edge_y = torch.exp(-10.0 * image_dy)
        normal_edge_aware = (
            (normal_dx * foreground_x * edge_x).sum()
            / (foreground_x * edge_x).sum().clamp_min(1)
            + (normal_dy * foreground_y * edge_y).sum()
            / (foreground_y * edge_y).sum().clamp_min(1)
        )
        loss = (
            loss
            + weights.silhouette_bce * silhouette_bce
            + weights.silhouette_dice * silhouette_dice
            + weights.background_leak * background_leak
            + weights.depth_smooth * depth_smooth
            + weights.normal_smooth * normal_smooth
            + weights.normal_edge_aware * normal_edge_aware
        )
        logs.update(
            silhouette_bce=float(silhouette_bce.detach()),
            silhouette_dice=float(silhouette_dice.detach()),
            background_leak=float(background_leak.detach()),
            depth_smooth=float(depth_smooth.detach()),
            normal_smooth=float(normal_smooth.detach()),
            normal_edge_aware=float(normal_edge_aware.detach()),
        )

    if (
        weights.normal_depth_consistency > 0
        and camera is not None
        and step >= normal_depth_start
    ):
        normal_depth = _depth_normal_consistency(
            output, camera, valid_mask, scales=weights.normal_depth_scales
        )
        loss = loss + weights.normal_depth_consistency * normal_depth
        logs["normal_depth_consistency"] = float(normal_depth.detach())
    if (
        weights.depth_from_normal > 0
        and camera is not None
        and step >= depth_from_normal_start
    ):
        depth_from_normal = _normal_to_depth_consistency(
            output, camera, valid_mask
        )
        loss = loss + weights.depth_from_normal * depth_from_normal
        logs["depth_from_normal"] = float(depth_from_normal.detach())

    if diffuse_gt is not None:
        diffuse_gt = diffuse_gt.to(render.diffuse)
        diffuse_loss = F.l1_loss(render.diffuse, diffuse_gt)
        refine_loss = F.l1_loss(output["pred"], diffuse_gt)
        loss = (
            loss
            + weights.diffuse_gt * diffuse_loss
            + weights.refine_gt * refine_loss
        )
        logs["diffuse"] = float(diffuse_loss.detach())
        logs["refine"] = float(refine_loss.detach())

    if specular_gt is not None:
        spec_loss = F.l1_loss(render.specular, specular_gt.to(render.specular))
        loss = loss + weights.specular_gt * spec_loss
        logs["specular"] = float(spec_loss.detach())

    mask = (
        valid_mask.to(render.full).expand_as(render.full) > 0.5
        if valid_mask is not None
        else None
    )
    if albedo_gt is not None:
        target_albedo = albedo_gt.to(render.albedo)
        albedo_loss = (
            (render.albedo - target_albedo).abs()[mask].mean()
            if mask is not None and mask.any()
            else F.l1_loss(render.albedo, target_albedo)
        )
        loss = loss + weights.albedo_gt * albedo_loss
        logs["albedo"] = float(albedo_loss.detach())
    if normal_gt is not None:
        target_normal = F.normalize(normal_gt.to(render.normal), dim=0, eps=1e-6)
        cosine = (render.normal * target_normal).sum(dim=0, keepdim=True).clamp(-1, 1)
        normal_mask = mask[:1] if mask is not None else torch.ones_like(cosine, dtype=torch.bool)
        normal_loss = (1 - cosine)[normal_mask].mean()
        loss = loss + weights.normal_gt * normal_loss
        logs["normal"] = float(normal_loss.detach())
    if depth_gt is not None and weights.depth_gt > 0:
        target_depth = depth_gt.to(render.depth)
        depth_mask = (
            (valid_mask.to(render.depth) > 0.5)
            if valid_mask is not None
            else target_depth > 0
        )
        depth_mask = depth_mask & (target_depth > 0)
        if depth_mask.any():
            pred_log = render.depth[depth_mask].clamp_min(1e-6).log()
            target_log = target_depth[depth_mask].clamp_min(1e-6).log()
            residual = pred_log - target_log
            depth_loss = (residual - residual.mean()).abs().mean()
            loss = loss + weights.depth_gt * depth_loss
            logs["depth"] = float(depth_loss.detach())
    if depth_prior is not None and weights.depth_prior > 0:
        prior_depth = depth_prior.to(render.depth)
        prior_mask = (
            (valid_mask.to(render.depth) > 0.5)
            if valid_mask is not None
            else prior_depth > 0
        )
        prior_mask = prior_mask & (prior_depth > 0)
        if prior_mask.any():
            pred_log = render.depth[prior_mask].clamp_min(1e-6).log()
            prior_log = prior_depth[prior_mask].clamp_min(1e-6).log()
            residual = pred_log - prior_log
            prior_loss = (residual - residual.mean()).abs().mean()
            loss = loss + weights.depth_prior * prior_loss
            logs["depth_prior"] = float(prior_loss.detach())

    energy = (scene.energy_sum() - 1).abs().mean()
    smooth = _material_smoothness(scene)
    normal_planar = (
        _normal_planar_consistency(scene)
        if weights.normal_planar > 0
        else scene.means.new_zeros(())
    )
    axis = scene.rotations[:, :, 2]
    normal_axis = (1 - (axis * scene.normals).sum(-1).abs()).mean()
    scale_reg = scene.scales.mean()
    opacity_sparse = scene.opacity.mean()
    texture_consistency = F.l1_loss(
        scene.texture, scene.materials().albedo.detach()
    )
    sensor = output.get("sensor", {})
    exposure_prior = sensor.get("exposure", loss.new_ones(1)).log().square().mean()
    white_balance_prior = (
        sensor.get("white_balance", loss.new_ones(3)).log().square().mean()
    )
    loss = (
        loss
        + weights.energy * energy
        + weights.material_smooth * smooth
        + weights.normal_planar * normal_planar
        + weights.normal_axis * normal_axis
        + weights.scale * scale_reg
        + weights.opacity_sparse * opacity_sparse
        + weights.texture_consistency * texture_consistency
        + weights.exposure_prior * exposure_prior
        + weights.white_balance_prior * white_balance_prior
    )
    logs.update(
        energy=float(energy.detach()),
        material_smooth=float(smooth.detach()),
        normal_planar=float(normal_planar.detach()),
        normal_axis=float(normal_axis.detach()),
        exposure_prior=float(exposure_prior.detach()),
        white_balance_prior=float(white_balance_prior.detach()),
    )
    prior = getattr(scene, "normal_prior", None)
    if weights.normal_prior > 0 and prior is not None:
        target = F.normalize(prior.to(scene.normals), dim=-1, eps=1e-6)
        cosine = (scene.normals * target).sum(dim=-1).clamp(-1.0, 1.0)
        normal_prior = (1.0 - cosine).mean()
        loss = loss + weights.normal_prior * normal_prior
        logs["normal_prior"] = float(normal_prior.detach())
    logs["total"] = float(loss.detach())
    return loss, logs

