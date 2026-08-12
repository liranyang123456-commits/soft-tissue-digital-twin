"""Metrics for intrinsic reflectance and decomposition evaluation.

All metrics operate on :class:`torch.Tensor` inputs and return Python scalars
or dictionaries suitable for JSON serialization.  Masks are broadcast to the
input shape; an empty mask produces ``None`` instead of an invalid number.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

MetricValue = float | None
_LPIPS_MODELS: dict[tuple[str, str], torch.nn.Module] = {}


def _masked_values(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Flatten values selected by a broadcast-compatible validity mask."""
    value = value.detach().float()
    if mask is None:
        return value.reshape(-1)
    valid = mask.detach().to(device=value.device, dtype=torch.bool)
    try:
        valid = torch.broadcast_to(valid, value.shape)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot broadcast to {tuple(value.shape)}"
        ) from error
    return value[valid]


def _finite_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> MetricValue:
    """Return the mean over masked finite entries, or ``None`` when empty."""
    selected = _masked_values(value, mask)
    selected = selected[torch.isfinite(selected)]
    if selected.numel() == 0:
        return None
    return float(selected.mean().item())


def masked_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    data_range: float = 1.0,
) -> MetricValue:
    """Compute PSNR over valid elements.

    Args:
        pred: Predicted values of arbitrary shape.
        target: Ground-truth tensor with the same shape as ``pred``.
        mask: Optional broadcast-compatible validity mask.
        data_range: Difference between the maximum and minimum valid values.

    Returns:
        PSNR in decibels, ``inf`` for identical inputs, or ``None`` if no
        finite valid entries exist.
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    mse = _finite_mean((pred.detach().float() - target.detach().float()).square(), mask)
    if mse is None:
        return None
    if mse == 0.0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(data_range**2 / mse)).item())


def masked_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    data_range: float = 1.0,
    window_size: int = 11,
) -> MetricValue:
    """Compute channel-wise local SSIM and average it over valid pixels.

    Inputs may be ``C x H x W`` or ``N x C x H x W``. The implementation uses
    a box window and crops no border pixels.
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    if pred.ndim not in (3, 4):
        raise ValueError("SSIM inputs must have shape CxHxW or NxCxHxW")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    x = pred.detach().float()
    y = target.detach().float().to(x.device)
    if x.ndim == 3:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
    channels = x.shape[1]
    kernel = torch.full(
        (channels, 1, window_size, window_size),
        1.0 / window_size**2,
        device=x.device,
        dtype=x.dtype,
    )
    padding = window_size // 2

    def average(value: torch.Tensor) -> torch.Tensor:
        return F.conv2d(value, kernel, padding=padding, groups=channels)

    mean_x, mean_y = average(x), average(y)
    var_x = (average(x.square()) - mean_x.square()).clamp_min(0)
    var_y = (average(y.square()) - mean_y.square()).clamp_min(0)
    covariance = average(x * y) - mean_x * mean_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)
    ).clamp_min(torch.finfo(x.dtype).tiny)
    score_mask = mask
    if score_mask is not None and score_mask.ndim == score.ndim - 1:
        score_mask = score_mask.unsqueeze(1)
    return _finite_mean(score, score_mask)


def aligned_albedo_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    affine: bool = False,
    per_channel: bool = False,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Evaluate albedo after least-squares scale or affine alignment.

    Alignment solves ``aligned = scale * pred + bias`` globally or separately
    per channel. ``bias`` is zero unless ``affine=True``.
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    if pred.ndim < 2:
        raise ValueError("albedo inputs must include a channel dimension")
    x = pred.detach().float()
    y = target.detach().float().to(x.device)
    channel_dim = 0 if x.ndim == 3 else 1
    channel_count = x.shape[channel_dim]
    valid = torch.ones_like(x, dtype=torch.bool)
    if mask is not None:
        candidate = mask.detach().to(device=x.device, dtype=torch.bool)
        if candidate.ndim == x.ndim - 1:
            candidate = candidate.unsqueeze(channel_dim)
        try:
            valid = torch.broadcast_to(candidate, x.shape)
        except RuntimeError as error:
            raise ValueError("mask cannot broadcast to albedo shape") from error
    valid = valid & torch.isfinite(x) & torch.isfinite(y)
    groups = range(channel_count) if per_channel else (None,)
    scales: list[float] = []
    biases: list[float] = []
    aligned = torch.full_like(x, float("nan"))
    for channel in groups:
        selector = valid if channel is None else valid.select(channel_dim, channel)
        x_group = x if channel is None else x.select(channel_dim, channel)
        y_group = y if channel is None else y.select(channel_dim, channel)
        xv, yv = x_group[selector], y_group[selector]
        if xv.numel() == 0:
            return {"mae": None, "rmse": None, "scale": None, "bias": None}
        if affine:
            x_mean, y_mean = xv.mean(), yv.mean()
            denominator = ((xv - x_mean) ** 2).sum()
            scale = (
                ((xv - x_mean) * (yv - y_mean)).sum() / denominator
                if denominator > eps
                else torch.zeros((), device=x.device)
            )
            bias = y_mean - scale * x_mean
        else:
            denominator = xv.square().sum()
            scale = (
                (xv * yv).sum() / denominator
                if denominator > eps
                else torch.zeros((), device=x.device)
            )
            bias = torch.zeros((), device=x.device)
        if channel is None:
            aligned = scale * x + bias
        else:
            aligned.select(channel_dim, channel).copy_(scale * x_group + bias)
        scales.append(float(scale.item()))
        biases.append(float(bias.item()))
    error = aligned - y
    mae = _finite_mean(error.abs(), valid)
    mse = _finite_mean(error.square(), valid)
    return {
        "mae": mae,
        "rmse": None if mse is None else float(mse**0.5),
        "scale": scales if per_channel else scales[0],
        "bias": biases if per_channel else biases[0],
    }


def scale_aligned_albedo_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    per_channel: bool = False,
) -> dict[str, Any]:
    """Evaluate albedo after multiplicative least-squares alignment."""
    return aligned_albedo_metrics(
        pred, target, mask, affine=False, per_channel=per_channel
    )


def linear_aligned_albedo_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    per_channel: bool = False,
) -> dict[str, Any]:
    """Evaluate albedo after affine (scale and bias) alignment."""
    return aligned_albedo_metrics(
        pred, target, mask, affine=True, per_channel=per_channel
    )


def optional_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    net: str = "alex",
) -> MetricValue:
    """Compute LPIPS when its optional dependency is available.

    ``None`` is returned when ``lpips`` cannot be imported or initialized.
    Inputs are expected in ``[0, 1]`` and are converted to ``[-1, 1]``.
    """
    try:
        import lpips  # type: ignore[import-not-found]

        key = (net, str(pred.device))
        model = _LPIPS_MODELS.get(key)
        if model is None:
            model = lpips.LPIPS(net=net).to(pred.device).eval()
            _LPIPS_MODELS[key] = model
    except (ImportError, ModuleNotFoundError, RuntimeError, OSError):
        return None
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    x, y = pred.detach().float(), target.detach().float().to(pred.device)
    if x.ndim == 3:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError("LPIPS inputs must have shape CxHxW or NxCxHxW")
    if mask is not None:
        valid = mask.detach().to(device=x.device, dtype=x.dtype)
        if valid.ndim == x.ndim - 1:
            valid = valid.unsqueeze(1)
        try:
            valid = torch.broadcast_to(valid, x.shape)
        except RuntimeError as error:
            raise ValueError("mask cannot broadcast to LPIPS input shape") from error
        x, y = x * valid, y * valid
    with torch.no_grad():
        value = model(x * 2 - 1, y * 2 - 1)
    return _finite_mean(value)


def normal_angular_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    thresholds: Iterable[float] = (11.25, 22.5, 30.0),
    eps: float = 1e-8,
) -> dict[str, MetricValue]:
    """Return normal mean/median angular error and threshold accuracies."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    channel_dim = 0 if pred.ndim == 3 else 1
    if pred.shape[channel_dim] != 3:
        raise ValueError("normal tensors must have three channels")
    x = F.normalize(pred.detach().float(), dim=channel_dim, eps=eps)
    y = F.normalize(target.detach().float().to(x.device), dim=channel_dim, eps=eps)
    cosine = (x * y).sum(dim=channel_dim).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    angle_mask = mask
    if angle_mask is not None and angle_mask.ndim == pred.ndim:
        if angle_mask.shape[channel_dim] != 1:
            raise ValueError("normal mask channel dimension must be one")
        angle_mask = angle_mask.squeeze(channel_dim)
    selected = _masked_values(angle, angle_mask)
    selected = selected[torch.isfinite(selected)]
    result: dict[str, MetricValue] = {"mean": None, "median": None}
    threshold_values = tuple(float(threshold) for threshold in thresholds)
    if selected.numel() == 0:
        result.update({f"acc_{threshold:g}": None for threshold in threshold_values})
        return result
    result["mean"] = float(selected.mean().item())
    result["median"] = float(selected.median().item())
    result.update(
        {
            f"acc_{threshold:g}": float((selected <= threshold).float().mean().item())
            for threshold in threshold_values
        }
    )
    return result


def erode_binary_mask(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Erode a binary mask with a square kernel."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    value = mask.detach().float()
    original_ndim = value.ndim
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[None]
    elif value.ndim != 4:
        raise ValueError("mask must have shape HW, CHW, or NCHW")
    eroded = 1 - F.max_pool2d(
        1 - (value > 0.5).float(),
        kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    if original_ndim == 2:
        return eroded[0, 0] > 0.5
    if original_ndim == 3:
        return eroded[0] > 0.5
    return eroded > 0.5


def linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    """Stanford-ORB's standard sRGB tone mapping for linear RGB."""
    value = value.detach().float()
    return torch.where(
        value > 0.0031308,
        1.055 * value.clamp_min(0.0031308).pow(1 / 2.4) - 0.055,
        12.92 * value,
    )


def _orb_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_value: float,
    tonemap: bool,
    scale_invariant: bool,
    use_gt_median: bool,
) -> tuple[MetricValue, torch.Tensor, torch.Tensor]:
    """Numerically match the official Stanford-ORB ``calc_PSNR`` routine."""
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[0] != 3:
        raise ValueError("ORB image inputs must have identical 3xHxW shapes")
    valid = erode_binary_mask(mask).to(device=pred.device)
    if valid.ndim == 3:
        valid = valid.squeeze(0)
    if not valid.any():
        return None, pred.detach().float(), target.detach().float()
    valid_rgb = valid.unsqueeze(0)
    x = pred.detach().float().clamp_min(0) * valid_rgb
    y = target.detach().float().to(x.device).clamp_min(0) * valid_rgb
    if use_gt_median:
        clipped = y.clamp(0, 1)
        denominator = clipped.mean()
        if denominator > 1e-8:
            median = linear_to_srgb(clipped).mean() / denominator
            x, y = x * median, y * median
    if scale_invariant:
        for channel in range(3):
            xv, yv = x[channel][valid], y[channel][valid]
            denominator = xv.square().sum()
            scale = (
                (xv * yv).sum() / denominator
                if denominator > 1e-6
                else torch.zeros((), device=x.device)
            )
            x[channel] *= scale
    x, y = x.clamp(0, max_value), y.clamp(0, max_value)
    if tonemap:
        x, y = linear_to_srgb(x), linear_to_srgb(y)
    mse = (x - y).square().mean()
    baseline = (0.5 * valid_rgb - y).square().mean()
    best_mse = torch.minimum(mse, baseline)
    score = (
        float("inf")
        if best_mse <= 0
        else float((-10.0 * torch.log10(best_mse)).item())
    )
    return score, x, y


def _orb_ssim(pred: torch.Tensor, target: torch.Tensor) -> MetricValue:
    """Official 3x3 SSIM, with a dependency-free compatible fallback."""
    x, y = pred.unsqueeze(0), target.unsqueeze(0)
    try:
        from kornia.losses import ssim_loss  # type: ignore[import-not-found]

        return float(1.0 - 2.0 * ssim_loss(x, y, 3).item())
    except (ImportError, ModuleNotFoundError, RuntimeError, OSError):
        weights = torch.tensor([0.3078013, 0.3843974, 0.3078013], device=x.device)
        kernel_2d = torch.outer(weights, weights)
        kernel = kernel_2d.expand(3, 1, 3, 3)

        def average(value: torch.Tensor) -> torch.Tensor:
            return F.conv2d(value, kernel, padding=1, groups=3)

        mean_x, mean_y = average(x), average(y)
        var_x = (average(x.square()) - mean_x.square()).clamp_min(0)
        var_y = (average(y.square()) - mean_y.square()).clamp_min(0)
        covariance = average(x * y) - mean_x * mean_y
        score = ((2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)) / (
            (mean_x.square() + mean_y.square() + 0.01**2)
            * (var_x + var_y + 0.03**2)
        ).clamp_min(torch.finfo(x.dtype).tiny)
        return float(score.mean())


def orb_image_similarity(
    pred_linear: torch.Tensor,
    target_linear: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale_invariant: bool,
    hdr_available: bool,
    compute_lpips: bool = False,
) -> dict[str, MetricValue]:
    """Official Stanford-ORB PSNR-H/PSNR-L, SSIM, and optional LPIPS.

    PSNR-H is deliberately ``None`` for an LDR-only reference. Converting an
    8-bit image back to linear RGB does not recover an HDR reference.
    """
    psnr_h: MetricValue = None
    if hdr_available:
        psnr_h, _, _ = _orb_psnr(
            pred_linear,
            target_linear,
            mask,
            max_value=4.0,
            tonemap=False,
            scale_invariant=scale_invariant,
            use_gt_median=True,
        )
    psnr_l, pred_srgb, target_srgb = _orb_psnr(
        pred_linear,
        target_linear,
        mask,
        max_value=1.0,
        tonemap=True,
        scale_invariant=scale_invariant,
        use_gt_median=False,
    )
    result: dict[str, MetricValue] = {
        "psnr_h": psnr_h,
        "psnr_l": psnr_l,
        "ssim": _orb_ssim(pred_srgb, target_srgb),
        "lpips": None,
    }
    if compute_lpips:
        # The official code applies sRGB mapping once more before VGG LPIPS.
        result["lpips"] = optional_lpips(
            linear_to_srgb(pred_srgb),
            linear_to_srgb(target_srgb),
            net="vgg",
        )
    return result


def world_normal_to_orb_camera(
    normal_world: torch.Tensor, camera_c2w: torch.Tensor
) -> torch.Tensor:
    """Convert world normals to Stanford-ORB's OpenGL camera-space output."""
    if normal_world.ndim != 3 or normal_world.shape[0] != 3:
        raise ValueError("normal_world must have shape 3xHxW")
    rotation_world_to_opencv = camera_c2w[:3, :3].T.to(normal_world)
    opencv_to_opengl = torch.diag(
        torch.tensor([1.0, -1.0, -1.0], device=normal_world.device)
    ).to(normal_world)
    rotation = opencv_to_opengl @ rotation_world_to_opencv
    return F.normalize(
        torch.einsum("ij,jhw->ihw", rotation, normal_world),
        dim=0,
        eps=1e-6,
    )


def orb_normal_cosine_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> MetricValue:
    """Stanford-ORB geometry normal score (lower is better)."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    x = F.normalize(pred.detach().float(), dim=0, eps=1e-6)
    y = F.normalize(target.detach().float().to(x.device), dim=0, eps=1e-6)
    valid = erode_binary_mask(mask).to(x.device)
    if valid.ndim == 3:
        valid = valid.squeeze(0)
    if not valid.any():
        return None
    return float((1 - (x * y).sum(dim=0))[valid].mean())


def orb_scale_invariant_albedo_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> MetricValue:
    """Stanford-ORB material PSNR with per-channel scale alignment."""
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[0] != 3:
        raise ValueError("albedo inputs must have identical 3xHxW shapes")
    x = pred.detach().float()
    y = target.detach().float().to(x.device).clamp_min(0)
    valid = erode_binary_mask(mask).to(x.device)
    if valid.ndim == 3:
        valid = valid.squeeze(0)
    if not valid.any():
        return None
    aligned = x.clone()
    for channel in range(3):
        xv, yv = x[channel][valid], y[channel][valid]
        denominator = xv.square().sum()
        scale = (xv * yv).sum() / denominator.clamp_min(1e-6)
        aligned[channel] *= scale
    valid_rgb = valid.unsqueeze(0)
    aligned = aligned.clamp(0, 1) * valid_rgb
    y = y.clamp(0, 1) * valid_rgb
    mse = (aligned - y).square().mean()
    baseline = (0.5 * valid_rgb - y).square().mean()
    best_mse = torch.minimum(mse, baseline)
    if best_mse <= 0:
        return float("inf")
    return float(-10 * torch.log10(best_mse))


def orb_scene_depth_mse(
    predictions: Iterable[torch.Tensor],
    targets: Iterable[torch.Tensor],
    masks: Iterable[torch.Tensor],
    *,
    table_scale: bool = False,
) -> MetricValue:
    """Stanford-ORB scene-global scale-aligned depth SI-MSE.

    Matches the official protocol: one scale for all test views in a scene,
    then mean squared error over eroded valid pixels. When ``table_scale`` is
    True, returns SI-MSE ``× 10^3`` as published in the Stanford-ORB tables
    (NeRF baseline = 2.19).
    """
    triples = list(zip(predictions, targets, masks))
    if not triples:
        return None
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    prepared: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for pred, target, mask in triples:
        # Clone so callers that reuse render buffers cannot corrupt the metric.
        x = pred.detach().float().cpu().clone().squeeze()
        y = target.detach().float().cpu().clone().squeeze()
        valid = erode_binary_mask(mask.detach().cpu()).squeeze()
        if valid.ndim == 3:
            valid = valid[0]
        valid = valid.bool() & torch.isfinite(x) & torch.isfinite(y) & (y > 1e-6)
        if not valid.any():
            continue
        xv = x[valid].to(torch.float64)
        yv = y[valid].to(torch.float64)
        if float(xv.square().sum()) <= 1e-12:
            x = torch.ones_like(x)
            xv = x[valid].to(torch.float64)
        numerator = numerator + (xv * yv).sum()
        denominator = denominator + xv.square().sum()
        prepared.append((x, y, valid))
    if not prepared:
        return None
    scale = numerator / denominator.clamp_min(1e-12)
    scores = [
        ((scale.float() * pred - target).square()[valid]).mean()
        for pred, target, valid in prepared
    ]
    value = float(torch.stack(scores).mean())
    return value * 1e3 if table_scale else value


def bidirectional_chamfer(
    predicted_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    chunk_size: int = 4096,
) -> MetricValue:
    """Official shape score: mean of two mean squared nearest distances."""
    if predicted_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError("point clouds must have shape Nx3")
    if predicted_points.shape[1] != 3 or target_points.shape[1] != 3:
        raise ValueError("point clouds must have shape Nx3")
    if not len(predicted_points) or not len(target_points):
        return None
    x = predicted_points.detach().float()
    y = target_points.detach().float().to(x.device)
    try:
        from scipy.spatial import cKDTree

        x_numpy = x.cpu().numpy()
        y_numpy = y.cpu().numpy()
        x_to_y = cKDTree(y_numpy).query(x_numpy, k=1, workers=-1)[0]
        y_to_x = cKDTree(x_numpy).query(y_numpy, k=1, workers=-1)[0]
        return float(0.5 * (np.square(x_to_y).mean() + np.square(y_to_x).mean()))
    except (ImportError, ModuleNotFoundError, ValueError, OSError):
        pass

    def one_way(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        scores = []
        for start in range(0, len(source), chunk_size):
            distance = torch.cdist(source[start : start + chunk_size], destination)
            scores.append(distance.square().min(dim=1).values)
        return torch.cat(scores).mean()

    return float(0.5 * (one_way(x, y) + one_way(y, x)))


def orb_shape_chamfer(
    output_mesh: str,
    target_mesh: str,
    *,
    num_samples: int = 30_000,
    seed: int | None = None,
) -> MetricValue:
    """Stanford-ORB table-scale Chamfer (raw bidirectional MSE times 2000)."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    try:
        import trimesh  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        return None
    if seed is not None:
        old_numpy_state = np.random.get_state()
        np.random.seed(seed)
    try:
        predicted = trimesh.load(output_mesh, process=False)
        target = trimesh.load_mesh(target_mesh)
        pred_points = (
            trimesh.sample.sample_surface(predicted, num_samples)[0]
            if hasattr(predicted, "faces") and len(predicted.faces) > 0
            else np.asarray(predicted.vertices)
        )
        target_points = np.asarray(target.vertices)
        raw_score = bidirectional_chamfer(
            torch.from_numpy(pred_points), torch.from_numpy(target_points)
        )
        # The official repository documents that paper/leaderboard values are
        # 2,000 times the raw JSON Chamfer value (despite the paper saying 1e3).
        return None if raw_score is None else 2_000.0 * raw_score
    except (OSError, ValueError, TypeError):
        return None
    finally:
        if seed is not None:
            np.random.set_state(old_numpy_state)


def decomposition_metrics(
    pred_diffuse: torch.Tensor,
    target_diffuse: torch.Tensor,
    pred_specular: torch.Tensor,
    target_specular: torch.Tensor,
    mask: torch.Tensor | None = None,
    data_range: float = 1.0,
) -> dict[str, MetricValue]:
    """Compute MAE, RMSE, PSNR, and SSIM for diffuse/specular components."""
    result: dict[str, MetricValue] = {}
    for name, pred, target in (
        ("diffuse", pred_diffuse, target_diffuse),
        ("specular", pred_specular, target_specular),
    ):
        error = pred.detach().float() - target.detach().float()
        mse = _finite_mean(error.square(), mask)
        result[f"{name}_mae"] = _finite_mean(error.abs(), mask)
        result[f"{name}_rmse"] = None if mse is None else float(mse**0.5)
        result[f"{name}_psnr"] = masked_psnr(pred, target, mask, data_range)
        result[f"{name}_ssim"] = masked_ssim(pred, target, mask, data_range)
    return result


def reconstruction_residual(
    observation: torch.Tensor,
    diffuse: torch.Tensor,
    specular: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, MetricValue]:
    """Measure residual of ``observation = diffuse + specular``."""
    if observation.shape != diffuse.shape or observation.shape != specular.shape:
        raise ValueError("observation, diffuse, and specular must share a shape")
    residual = observation.detach().float() - diffuse.detach().float() - specular.detach().float()
    mse = _finite_mean(residual.square(), mask)
    return {
        "mae": _finite_mean(residual.abs(), mask),
        "rmse": None if mse is None else float(mse**0.5),
        "bias": _finite_mean(residual, mask),
    }


def multiview_albedo_consistency(
    samples: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    view_dim: int = 1,
) -> dict[str, MetricValue]:
    """Compute robust cross-view variance and MAD for corresponding samples.

    Args:
        samples: Already-corresponded albedo samples. By default shape is
            ``N x V x C`` (additional trailing dimensions are supported).
        valid_mask: Boolean mask broadcast-compatible with ``samples``.
        view_dim: Dimension containing observations of the same surface point.

    Returns:
        Means of per-element variance and median absolute deviation. Elements
        with fewer than two valid views are omitted.
    """
    x = samples.detach().float()
    valid = valid_mask.detach().to(device=x.device, dtype=torch.bool)
    while valid.ndim < x.ndim:
        valid = valid.unsqueeze(-1)
    try:
        valid = torch.broadcast_to(valid, x.shape)
    except RuntimeError as error:
        raise ValueError("valid_mask cannot broadcast to samples") from error
    valid = valid & torch.isfinite(x)
    count = valid.sum(dim=view_dim)
    safe_count = count.clamp_min(1)
    mean = torch.where(valid, x, 0).sum(dim=view_dim) / safe_count
    variance = (
        torch.where(valid, (x - mean.unsqueeze(view_dim)).square(), 0).sum(dim=view_dim)
        / (safe_count - 1).clamp_min(1)
    )
    nan_x = torch.where(valid, x, torch.full_like(x, float("nan")))
    median = torch.nanquantile(nan_x, 0.5, dim=view_dim)
    absolute_deviation = (x - median.unsqueeze(view_dim)).abs()
    mad = torch.nanquantile(
        torch.where(valid, absolute_deviation, torch.full_like(x, float("nan"))),
        0.5,
        dim=view_dim,
    )
    enough = count >= 2
    return {
        "robust_variance": _finite_mean(variance, enough),
        "mad": _finite_mean(mad, enough),
    }


def aggregate_metrics(
    records: Iterable[Mapping[str, float | int | None]],
) -> dict[str, MetricValue]:
    """Aggregate metric records by finite arithmetic mean, ignoring ``None``."""
    values: dict[str, list[float]] = {}
    keys: set[str] = set()
    for record in records:
        keys.update(record)
        for key, value in record.items():
            if value is None:
                continue
            scalar = float(value)
            if torch.isfinite(torch.tensor(scalar)):
                values.setdefault(key, []).append(scalar)
    return {
        key: (float(sum(values[key]) / len(values[key])) if values.get(key) else None)
        for key in sorted(keys)
    }
