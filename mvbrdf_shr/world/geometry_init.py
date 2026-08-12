"""Geometry and normal initialization without evaluation ground truth."""
from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn.functional as F

from .data import MultiViewScene


def _sample_image(image: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[-2:]
    grid = torch.stack(
        [
            2 * uv[:, 0] / max(width - 1, 1) - 1,
            2 * uv[:, 1] / max(height - 1, 1) - 1,
        ],
        dim=-1,
    ).view(1, 1, -1, 2)
    return F.grid_sample(
        image.unsqueeze(0), grid, mode="bilinear", align_corners=True
    )[0, :, 0].T


def _silhouette_center(scene: MultiViewScene, frame_ids: list[int]) -> torch.Tensor:
    matrices, vectors = [], []
    for frame_id in frame_ids:
        frame = scene[frame_id]
        if frame.mask_gt is None:
            continue
        locations = torch.nonzero(frame.mask_gt[0] > 0.5)
        if locations.numel() == 0:
            continue
        yx = locations.float().mean(dim=0)
        pixel = torch.tensor([yx[1], yx[0], 1.0], dtype=frame.camera.K.dtype)
        direction_camera = torch.linalg.solve(frame.camera.K, pixel)
        direction = F.normalize(
            frame.camera.c2w[:3, :3] @ direction_camera, dim=0
        )
        projection = torch.eye(3) - direction[:, None] @ direction[None]
        matrices.append(projection)
        vectors.append(projection @ frame.camera.center)
    if len(matrices) < 2:
        raise ValueError("visual hull requires masks from at least two training views")
    return torch.linalg.solve(
        torch.stack(matrices).sum(dim=0) + 1e-5 * torch.eye(3),
        torch.stack(vectors).sum(dim=0),
    )


@torch.inference_mode()
def initialize_multiview_mask_points(
    scene: MultiViewScene,
    frame_ids: list[int],
    max_points: int = 50000,
    views: int = 8,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Unproject mask pixels toward the silhouette center from several views."""
    device = torch.device(device)
    center = _silhouette_center(scene, frame_ids).to(device)
    selected = frame_ids[:: max(1, len(frame_ids) // max(views, 1))][:views]
    if not selected:
        selected = frame_ids[:1]
    per_view = max(1, max_points // max(len(selected), 1))
    points: list[torch.Tensor] = []
    colors: list[torch.Tensor] = []
    for frame_id in selected:
        frame = scene[frame_id]
        camera = frame.camera.to(device)
        origins, directions = camera.pixel_rays()
        flat_origins = origins.reshape(-1, 3)
        flat_directions = F.normalize(directions.reshape(-1, 3), dim=-1)
        image = frame.image.to(device).permute(1, 2, 0).reshape(-1, 3)
        if frame.mask_gt is None:
            candidates = torch.arange(flat_directions.shape[0], device=device)
        else:
            candidates = torch.where(frame.mask_gt.to(device).reshape(-1) > 0.5)[0]
        if candidates.numel() == 0:
            continue
        count = min(per_view, int(candidates.numel()))
        ids = candidates[
            torch.linspace(0, candidates.numel() - 1, count, device=device).long()
        ]
        # Place samples near the plane through the silhouette center.
        depth = ((center - flat_origins[ids]) * flat_directions[ids]).sum(dim=-1)
        depth = depth.clamp_min(1e-3)
        sample = flat_origins[ids] + flat_directions[ids] * depth[:, None]
        points.append(sample.cpu())
        colors.append(image[ids].cpu())
    if not points:
        raise ValueError("multiview mask unprojection produced no points")
    merged = torch.cat(points, dim=0)
    merged_colors = torch.cat(colors, dim=0)
    if len(merged) > max_points:
        ids = torch.linspace(0, len(merged) - 1, max_points).long()
        merged = merged[ids]
        merged_colors = merged_colors[ids]
    return merged, merged_colors, None


def _estimate_object_radius(
    scene: MultiViewScene, frame_ids: list[int], center: torch.Tensor
) -> float:
    """Estimate object radius from silhouette extent at the object center."""
    radii = []
    for frame_id in frame_ids:
        frame = scene[frame_id]
        if frame.mask_gt is None:
            continue
        ys, xs = torch.where(frame.mask_gt[0] > 0.5)
        if len(ys) == 0:
            continue
        camera = frame.camera
        dist = torch.linalg.norm(camera.center - center).item()
        # Corner pixels of the silhouette AABB → angular half-extent.
        corners = torch.tensor(
            [
                [float(xs.min()), float(ys.min()), 1.0],
                [float(xs.max()), float(ys.min()), 1.0],
                [float(xs.min()), float(ys.max()), 1.0],
                [float(xs.max()), float(ys.max()), 1.0],
            ],
            dtype=camera.K.dtype,
        )
        dirs_cam = torch.linalg.solve(camera.K, corners.transpose(0, 1)).transpose(0, 1)
        dirs = F.normalize(dirs_cam @ camera.c2w[:3, :3].transpose(0, 1), dim=-1)
        axis = F.normalize(center - camera.center, dim=0)
        cos = (dirs * axis[None]).sum(dim=-1).clamp(-1, 1)
        # Distance from center to the silhouette cone at the center plane.
        half = dist * torch.tan(torch.acos(cos)).max().item()
        radii.append(half)
    if not radii:
        return 1.0
    # Slight padding so carving does not clip the surface.
    return float(max(radii) * 1.25)


@torch.inference_mode()
def initialize_visual_hull(
    scene: MultiViewScene,
    frame_ids: list[int],
    max_points: int = 50000,
    resolution: int = 64,
    min_view_fraction: float = 0.8,
    normal_smoothing_kernel: int = 1,
    coarse_resolution: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxel-carve a mask-only surface from training cameras."""
    device = torch.device(device)
    center = _silhouette_center(scene, frame_ids)
    radius = _estimate_object_radius(scene, frame_ids, center)
    carve_ids = frame_ids[:: max(1, len(frame_ids) // 24)][:24]

    def vote(points: torch.Tensor) -> torch.Tensor:
        inside_votes = torch.zeros(
            len(points), dtype=torch.float32, device=device
        )
        valid_votes = torch.zeros_like(inside_votes)
        used = 0
        for frame_id in carve_ids:
            frame = scene[frame_id]
            if frame.mask_gt is None:
                continue
            camera = frame.camera.to(device)
            uv, _, projected = camera.project(points)
            sample = _sample_image(frame.mask_gt.to(device), uv)[:, 0] > 0.5
            valid_votes += projected.float()
            inside_votes += (projected & sample).float()
            used += 1
        if used == 0:
            raise ValueError("visual hull requires at least one masked training view")
        return (valid_votes > 0) & (
            inside_votes / valid_votes.clamp_min(1) >= float(min_view_fraction)
        )

    coarse = int(coarse_resolution)
    if 8 <= coarse < resolution:
        coarse_axis = torch.linspace(-radius, radius, coarse, device=device)
        coarse_z, coarse_y, coarse_x = torch.meshgrid(
            coarse_axis, coarse_axis, coarse_axis, indexing="ij"
        )
        coarse_grid = (
            torch.stack([coarse_x, coarse_y, coarse_z], dim=-1).reshape(-1, 3)
            + center.to(device)
        )
        coarse_inside = vote(coarse_grid)
        if bool(coarse_inside.any()):
            occupied = coarse_grid[coarse_inside]
            lower, upper = occupied.amin(dim=0), occupied.amax(dim=0)
            center = (lower + upper) * 0.5
            coarse_spacing = (2 * radius) / max(coarse - 1, 1)
            radius = float(
                ((upper - lower) * 0.5).max().add(2 * coarse_spacing)
            )
    axis = torch.linspace(-radius, radius, resolution, device=device)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3) + center.to(device)
    occupancy = vote(grid).view(resolution, resolution, resolution)
    # Keep only the shell so Gaussians concentrate near the object surface.
    eroded = (
        F.max_pool3d(
            (~occupancy).float()[None, None],
            kernel_size=3,
            stride=1,
            padding=1,
        )[0, 0]
        < 0.5
    )
    surface = occupancy & ~eroded
    surface_points = grid.view(resolution, resolution, resolution, 3)[surface]
    if len(surface_points) == 0:
        surface_points = grid.view(-1, 3)[occupancy.view(-1)]
    # Smooth the binary hull before finite differences. Raw voxel gradients are
    # axis-quantized and unstable on planar faces viewed obliquely.
    kernel = max(1, int(normal_smoothing_kernel))
    if kernel % 2 == 0:
        raise ValueError("normal_smoothing_kernel must be odd")
    occ = (
        F.avg_pool3d(
            occupancy.float()[None, None],
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )[0, 0]
        if kernel > 1
        else occupancy.float()
    )
    grad_x = F.pad(occ[:, :, 2:] - occ[:, :, :-2], (1, 1, 0, 0, 0, 0))
    grad_y = F.pad(occ[:, 2:, :] - occ[:, :-2, :], (0, 0, 1, 1, 0, 0))
    grad_z = F.pad(occ[2:, :, :] - occ[:-2, :, :], (0, 0, 0, 0, 1, 1))
    gradient = torch.stack([grad_x, grad_y, grad_z], dim=-1)
    surface_normals = -gradient[surface]
    if len(surface_points) == 0:
        raise ValueError("visual-hull carving produced no surface points")
    if len(surface_points) > max_points:
        ids = torch.linspace(0, len(surface_points) - 1, max_points, device=device).long()
        surface_points = surface_points[ids]
        surface_normals = surface_normals[ids]
    radial = F.normalize(surface_points - center.to(device), dim=-1, eps=1e-6)
    normals = torch.where(
        surface_normals.norm(dim=-1, keepdim=True) > 1e-6,
        F.normalize(surface_normals, dim=-1, eps=1e-6),
        radial,
    )
    spacing = (2 * radius) / max(resolution - 1, 1)
    scales = surface_points.new_tensor([1.2 * spacing, 1.2 * spacing, 0.2 * spacing])
    return surface_points.cpu(), normals.cpu(), scales.expand(len(surface_points), -1).cpu()


@torch.no_grad()
def initialize_monocular_fused_points(
    scene: MultiViewScene,
    frame_ids: list[int],
    max_points: int = 50000,
    resolution: int = 96,
    min_view_fraction: float = 0.8,
    normal_smoothing_kernel: int = 3,
    device: torch.device | str = "cpu",
    mono_input_size: int = 384,
    depth_points_per_view: int = 6000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse visual-hull carving with monocular-depth surface points (NoGT).

    Visual-hull carving yields a watertight but coarse shell that overestimates
    concavities.  Monocular depth contributes viewpoint-specific surface
    detail.  Each training view's MiDaS relative depth is scale/shift-aligned
    to the visual-hull reference depth (first-hit along each ray) and
    unprojected to world points on the object mask; the union with the hull
    shell initializes the Gaussians.  Geometry is only *initialized* here — no
    persistent depth supervision is applied, avoiding the train-time conflict
    observed with a persistent monocular depth prior.  Returns
    ``(points, normals, scales)`` like :func:`initialize_visual_hull`.
    """
    device = torch.device(device)
    center = _silhouette_center(scene, frame_ids)
    radius = _estimate_object_radius(scene, frame_ids, center)

    # Coarse visual-hull occupancy for reference depth and normals.
    res = int(resolution)
    axis = torch.linspace(-radius, radius, res, device=device)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3) + center.to(device)
    inside_votes = torch.zeros(len(grid), device=device)
    valid_votes = torch.zeros(len(grid), device=device)
    carve_ids = frame_ids[:: max(1, len(frame_ids) // 24)][:24]
    for fid in carve_ids:
        frame = scene[fid]
        if frame.mask_gt is None:
            continue
        camera = frame.camera.to(device)
        uv, _, projected = camera.project(grid)
        sample = _sample_image(frame.mask_gt.to(device), uv)[:, 0] > 0.5
        valid_votes += projected.float()
        inside_votes += (projected & sample).float()
    occupancy = (
        (valid_votes > 0)
        & (inside_votes / valid_votes.clamp_min(1) >= float(min_view_fraction))
    ).view(res, res, res)

    # Per-view monocular depth aligned to the hull, unprojected to world.
    model = _load_monocular_depth_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    mono_points: list[torch.Tensor] = []
    for fid in frame_ids:
        frame = scene[fid]
        if frame.mask_gt is None:
            continue
        camera = frame.camera.to(device)
        image = frame.image.to(device).float()
        h, w = image.shape[-2:]
        img = image / (1.0 + image) if float(image.max()) > 1.5 else image
        img = img.clamp(0.0, 1.0)
        inp = F.interpolate(
            img.unsqueeze(0), size=(mono_input_size, mono_input_size),
            mode="bicubic", align_corners=False,
        )
        inp = (inp - mean) / std
        disp = model(inp)
        disp = F.interpolate(
            disp.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        )[0, 0]
        mask = frame.mask_gt[0].to(device) > 0.5
        if not bool(mask.any()):
            continue
        ys, xs = torch.nonzero(mask, as_tuple=True)
        ref_depth = _ray_visual_hull_depth(
            camera, occupancy, center.to(device), radius, xs, ys
        )
        # MiDaS outputs disparity (inverse depth up to scale/shift), so align
        # in disparity space: 1/ref_depth ~= a*disp + b, then invert.
        disp_flat = disp[ys, xs]
        amat = torch.stack([disp_flat, torch.ones_like(disp_flat)], dim=-1)
        sol = torch.linalg.lstsq(amat, 1.0 / ref_depth).solution
        metric_inv = disp_flat * sol[0] + sol[1]
        aligned = (1.0 / metric_inv.clamp_min(1e-3)).clamp_min(1e-3)
        # Reject points whose aligned depth departs strongly from the hull
        # reference (monocular depth is unreliable at grazing/occluding edges).
        keep = (aligned > 0.5 * ref_depth) & (aligned < 2.0 * ref_depth)
        xs, ys, aligned = xs[keep], ys[keep], aligned[keep]
        if len(xs) > depth_points_per_view:
            sel = torch.randperm(len(xs), device=device)[:depth_points_per_view]
            xs, ys, aligned = xs[sel], ys[sel], aligned[sel]
        pix = torch.stack(
            [xs.float(), ys.float(), torch.ones(len(xs), device=device)], dim=-1
        )
        dirs = pix @ torch.linalg.inv(camera.K).T
        pts_cam = dirs * aligned[..., None]
        pts_world = pts_cam @ camera.c2w[:3, :3].T + camera.c2w[:3, 3]
        mono_points.append(pts_world)
    mono = (
        torch.cat(mono_points, dim=0)
        if mono_points
        else torch.zeros(0, 3, device=device)
    )

    # Union with the hull shell.
    vh_points, _, _ = initialize_visual_hull(
        scene,
        frame_ids,
        max_points=max_points,
        resolution=resolution,
        min_view_fraction=min_view_fraction,
        normal_smoothing_kernel=normal_smoothing_kernel,
        device=device,
    )
    all_points = torch.cat([vh_points.to(device), mono], dim=0)

    # Normals from the smoothed occupancy gradient via nearest-voxel lookup.
    kernel = max(1, int(normal_smoothing_kernel))
    if kernel % 2 == 0:
        kernel += 1
    occ = (
        F.avg_pool3d(
            occupancy.float()[None, None], kernel_size=kernel, stride=1,
            padding=kernel // 2,
        )[0, 0]
        if kernel > 1
        else occupancy.float()
    )
    gx = F.pad(occ[:, :, 2:] - occ[:, :, :-2], (1, 1, 0, 0, 0, 0))
    gy = F.pad(occ[:, 2:, :] - occ[:, :-2, :], (0, 0, 1, 1, 0, 0))
    gz = F.pad(occ[2:, :, :] - occ[:-2, :, :], (0, 0, 0, 0, 1, 1))
    gradient = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    local = (all_points - center.to(device)) / max(radius, 1e-6)
    coords = ((local + 1) * 0.5 * (res - 1)).round().long()
    inside = ((coords >= 0) & (coords < res)).all(dim=-1)
    flat = (
        coords[:, 2].clamp(0, res - 1) * res * res
        + coords[:, 1].clamp(0, res - 1) * res
        + coords[:, 0].clamp(0, res - 1)
    )
    surf_n = -gradient[flat]
    radial = F.normalize(all_points - center.to(device), dim=-1, eps=1e-6)
    normals = torch.where(
        (surf_n.norm(dim=-1, keepdim=True) > 1e-6) & inside[:, None],
        F.normalize(surf_n, dim=-1, eps=1e-6),
        radial,
    )

    if len(all_points) > max_points:
        ids = torch.linspace(0, len(all_points) - 1, max_points, device=device).long()
        all_points = all_points[ids]
        normals = normals[ids]
    spacing = (2 * radius) / max(res - 1, 1)
    scales = all_points.new_tensor([1.2 * spacing, 1.2 * spacing, 0.2 * spacing])
    return (
        all_points.cpu(),
        normals.cpu(),
        scales.expand(len(all_points), -1).cpu(),
    )


def _image_photometric_normal_map(
    scene: MultiViewScene,
    frame_ids: list[int],
    device: torch.device,
    face_camera: bool,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.55,
    irls_iterations: int = 0,
    huber_delta: float = 1.5,
    shadow_threshold: float = 0.01,
    saturation_threshold: float = 0.98,
) -> tuple[torch.Tensor, torch.Tensor, object]:
    """Solve image-space PS for one multi-light view group.

    Returns ``(normal_world CHW, confidence HW, camera)``.
    """
    frame = scene[frame_ids[0]]
    camera = frame.camera.to(device)
    rotation = camera.c2w[:3, :3]
    mask = frame.mask_gt.to(device)[0] > 0.5 if frame.mask_gt is not None else None
    images = []
    lights = []
    for frame_id in frame_ids:
        observation = scene[frame_id]
        if observation.light_direction is None:
            continue
        images.append(observation.image.to(device))
        light_camera = rotation.transpose(0, 1) @ observation.light_direction.to(device)
        lights.append(F.normalize(light_camera, dim=0))
    if len(images) < 6:
        raise ValueError("image photometric stereo requires at least 6 lights")
    rgb = torch.stack(images)
    light = torch.stack(lights)
    height, width = rgb.shape[-2:]
    if mask is not None:
        ys, xs = torch.where(mask)
    else:
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        ys, xs = ys.reshape(-1), xs.reshape(-1)
    if len(ys) == 0:
        raise ValueError("image photometric stereo mask is empty")
    # Tighter highlight rejection; fuse RGB channel solves (~18–19° on DiLiGenT).
    lower_q, upper_q = float(lower_quantile), float(upper_quantile)
    normal_acc = torch.zeros(len(ys), 3, device=device)
    conf_acc = torch.zeros(len(ys), device=device)
    channel_stacks = [rgb[:, c] for c in range(3)] + [rgb.mean(dim=1)]
    legacy_quantile = int(irls_iterations) <= 0
    for samples_full in channel_stacks:
        samples = samples_full[:, ys, xs]
        lower = torch.quantile(samples, lower_q, dim=0)
        upper = torch.quantile(samples, upper_q, dim=0)
        quantile_weight = (samples >= lower) & (samples <= upper)
        base_weight = (
            quantile_weight
            if legacy_quantile
            else (
                quantile_weight
                & (samples >= float(shadow_threshold))
                & (samples <= float(saturation_threshold))
            )
        ).float()
        weight = base_weight
        eye = torch.eye(3, device=device)[None]
        for iteration in range(max(0, int(irls_iterations)) + 1):
            gram = torch.einsum("ln,li,lj->nij", weight, light, light)
            rhs = torch.einsum("ln,li,ln->ni", weight, light, samples)
            gram = gram + 1e-4 * eye
            gradient = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
            if iteration >= int(irls_iterations):
                break
            prediction = torch.einsum("li,ni->ln", light, gradient)
            residual = (samples - prediction).abs()
            scale = residual.median(dim=0).values.clamp_min(1e-4)
            cutoff = float(huber_delta) * scale
            robust = torch.minimum(
                torch.ones_like(residual),
                cutoff[None] / residual.clamp_min(1e-6),
            )
            weight = base_weight * robust
        normal_camera = F.normalize(gradient, dim=-1, eps=1e-6)
        facing = normal_camera[:, 2:3]
        flip = facing > 0 if face_camera else facing < 0
        normal_camera = torch.where(flip, -normal_camera, normal_camera)
        normal_world = F.normalize(normal_camera @ rotation.transpose(0, 1), dim=-1)
        enough = weight.sum(dim=0) >= max(6, math.ceil(len(light) * 0.3))
        if legacy_quantile:
            conf = enough.float()
        else:
            prediction = torch.einsum("li,ni->ln", light, gradient)
            residual = (samples - prediction).abs()
            expected_support = max(upper_q - lower_q, 0.1)
            support = (
                weight.sum(dim=0) / max(len(light), 1) / expected_support
            ).clamp(max=1)
            relative_residual = residual.median(dim=0).values / (
                samples.abs().median(dim=0).values + 0.05
            )
            eigenvalues = torch.linalg.eigvalsh(gram)
            conditioning = (
                eigenvalues[:, 0] / eigenvalues[:, -1].clamp_min(1e-6)
            ).clamp(0, 1)
            conf = (
                support
                * torch.exp(-relative_residual)
                * (conditioning / 0.05).clamp(max=1)
            )
            conf = torch.where(enough, conf, torch.zeros_like(conf))
        normal_acc = normal_acc + normal_world * conf[:, None]
        conf_acc = conf_acc + conf
    normal_world = F.normalize(normal_acc, dim=-1, eps=1e-6)
    confidence = (
        (conf_acc > 0).float()
        if legacy_quantile
        else (conf_acc / len(channel_stacks)).clamp(0, 1)
    )
    normal_map = torch.zeros(3, height, width, device=device)
    confidence_map = torch.zeros(height, width, device=device)
    normal_map[:, ys, xs] = normal_world.transpose(0, 1)
    confidence_map[ys, xs] = confidence
    packed = F.pad(normal_map[None], (1, 1, 1, 1), mode="replicate")
    smoothed = F.normalize(F.avg_pool2d(packed, kernel_size=3, stride=1)[0], dim=0, eps=1e-6)
    normal_map = torch.where(confidence_map[None] > 0, smoothed, normal_map)
    return normal_map, confidence_map, camera


def _build_occupancy_volume(
    scene: MultiViewScene,
    frame_ids: list[int],
    resolution: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Return ``(occupancy[R,R,R], center, radius, grid_points[N,3])``."""
    center = _silhouette_center(scene, frame_ids)
    radius = _estimate_object_radius(scene, frame_ids, center)
    axis = torch.linspace(-radius, radius, resolution, device=device)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3) + center.to(device)
    occupancy = torch.ones(len(grid), dtype=torch.bool, device=device)
    # Subsample carving views for speed on dense multi-light sets.
    carve_ids = frame_ids[:: max(1, len(frame_ids) // 24)][:24]
    used = 0
    for frame_id in carve_ids:
        frame = scene[frame_id]
        if frame.mask_gt is None:
            continue
        camera = frame.camera.to(device)
        uv, _, projected = camera.project(grid)
        sample = _sample_image(frame.mask_gt.to(device), uv)[:, 0] > 0.5
        occupancy &= (~projected) | sample
        used += 1
    if used == 0:
        raise ValueError("visual hull requires at least one masked training view")
    occupancy = occupancy.view(resolution, resolution, resolution)
    return occupancy, center.to(device), radius, grid


def _ray_visual_hull_depth(
    camera,
    occupancy: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    xs: torch.Tensor,
    ys: torch.Tensor,
    samples: int = 64,
) -> torch.Tensor:
    """First-hit depth along rays through ``(xs, ys)`` against a binary volume."""
    device = occupancy.device
    resolution = occupancy.shape[0]
    pixel = torch.stack(
        [xs.float(), ys.float(), torch.ones(len(xs), device=device)], dim=-1
    )
    direction_camera = torch.linalg.solve(camera.K, pixel.transpose(0, 1)).transpose(0, 1)
    direction = F.normalize(direction_camera @ camera.c2w[:3, :3].transpose(0, 1), dim=-1)
    origin = camera.center[None].expand(len(direction), -1)
    # Sample along the segment that intersects the carving AABB.
    to_center = center[None] - origin
    mid = (to_center * direction).sum(dim=-1).clamp_min(1e-3)
    half = radius * 1.05
    t0 = (mid - half).clamp_min(1e-3)
    t1 = (mid + half).clamp_min(t0 + 1e-3)
    ts = torch.linspace(0, 1, samples, device=device)
    depths = t0[:, None] + (t1 - t0)[:, None] * ts[None]
    points = origin[:, None, :] + direction[:, None, :] * depths[..., None]
    # Map world points → voxel indices in [-radius, radius]^3.
    local = (points - center[None, None, :]) / max(radius, 1e-6)
    coords = ((local + 1) * 0.5 * (resolution - 1)).round().long()
    inside = ((coords >= 0) & (coords < resolution)).all(dim=-1)
    flat = (
        coords[..., 2].clamp(0, resolution - 1) * resolution * resolution
        + coords[..., 1].clamp(0, resolution - 1) * resolution
        + coords[..., 0].clamp(0, resolution - 1)
    )
    occ_flat = occupancy.reshape(-1)
    hit = inside & occ_flat[flat]
    # First hit along each ray; fall back to mid-plane depth.
    has_hit = hit.any(dim=-1)
    first = hit.float().argmax(dim=-1)
    depth = depths.gather(1, first[:, None]).squeeze(1)
    depth = torch.where(has_hit, depth, mid)
    return depth.clamp_min(1e-3)


@torch.inference_mode()
def initialize_image_photometric_stereo(
    scene: MultiViewScene,
    frame_ids: list[int],
    max_points: int = 50000,
    max_views: int = 12,
    device: torch.device | str = "cpu",
    face_camera: bool = False,
    hull_resolution: int = 48,
    fuse_world_surface: bool = False,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.55,
    irls_iterations: int = 0,
    huber_delta: float = 1.5,
    shadow_threshold: float = 0.01,
    saturation_threshold: float = 0.98,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unproject image-space PS with visual-hull depths per source view.

    Returns points, normals, colors, scales, and ``source_view_id`` (camera_id of
    the view that created each Gaussian). Multi-view samples must not be alpha-
    blended together; the renderer view-locks opacity by this id.
    """
    device = torch.device(device)
    occupancy, center, radius, _ = _build_occupancy_volume(
        scene, frame_ids, resolution=hull_resolution, device=device
    )
    allowed = set(frame_ids)
    groups = [
        [index for index in values if index in allowed]
        for values in scene.group_by_view().values()
    ]
    groups = [values for values in groups if len(values) >= 6][:max_views]
    if not groups:
        raise ValueError("image photometric stereo requires multi-light views")
    per_view = max(1, max_points // max(len(groups), 1))
    all_points: list[torch.Tensor] = []
    all_normals: list[torch.Tensor] = []
    all_colors: list[torch.Tensor] = []
    all_view_ids: list[torch.Tensor] = []
    for values in groups:
        try:
            normal_map, confidence_map, camera = _image_photometric_normal_map(
                scene,
                values,
                device,
                face_camera=face_camera,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                irls_iterations=irls_iterations,
                huber_delta=huber_delta,
                shadow_threshold=shadow_threshold,
                saturation_threshold=saturation_threshold,
            )
        except ValueError:
            continue
        mask = confidence_map > 0.5
        ys, xs = torch.where(mask)
        if len(ys) == 0:
            continue
        if len(ys) > per_view:
            ids = torch.linspace(0, len(ys) - 1, per_view, device=device).long()
            ys, xs = ys[ids], xs[ids]
        depth = _ray_visual_hull_depth(camera, occupancy, center, radius, xs, ys)
        rotation = camera.c2w[:3, :3]
        pixel = torch.stack(
            [xs.float(), ys.float(), torch.ones(len(xs), device=device)], dim=-1
        )
        direction_camera = torch.linalg.solve(camera.K, pixel.transpose(0, 1)).transpose(
            0, 1
        )
        direction = F.normalize(direction_camera @ rotation.transpose(0, 1), dim=-1)
        origin = camera.center[None].expand(len(direction), -1)
        points = origin + direction * depth[:, None]
        normals = normal_map[:, ys, xs].transpose(0, 1)
        colors = scene[values[0]].image.to(device)[:, ys, xs].transpose(0, 1)
        frame0 = scene[values[0]]
        view_id = int(
            frame0.view_id
            if getattr(frame0, "view_id", None) is not None
            else getattr(camera, "view_id", values[0])
        )
        all_points.append(points)
        all_normals.append(normals)
        all_colors.append(colors)
        all_view_ids.append(
            torch.full((len(points),), view_id, device=device, dtype=torch.long)
        )
    if not all_points:
        raise ValueError("image photometric stereo produced no samples")
    points = torch.cat(all_points, dim=0)
    normals = F.normalize(torch.cat(all_normals, dim=0), dim=-1, eps=1e-6)
    colors = torch.cat(all_colors, dim=0).clamp(0, 1)
    source_view_id = torch.cat(all_view_ids, dim=0)
    spacing = (2 * radius) / max(hull_resolution - 1, 1)
    if fuse_world_surface:
        # Per-view first-hit shells overlap around the same physical surface.
        # Collapse those observations before rendering held-out views; otherwise
        # view locking either leaves holes or alpha-blends incompatible shells.
        voxel_size = max(float(spacing), 1e-6)
        voxel = torch.floor((points - center[None]) / voxel_size).long()
        _, inverse = torch.unique(voxel, dim=0, return_inverse=True)
        count = torch.bincount(inverse, minlength=int(inverse.max()) + 1).to(points)
        fused_points = torch.zeros(len(count), 3, device=device, dtype=points.dtype)
        fused_colors = torch.zeros_like(fused_points)
        fused_points.index_add_(0, inverse, points)
        fused_colors.index_add_(0, inverse, colors)
        fused_points /= count[:, None].clamp_min(1)
        fused_colors /= count[:, None].clamp_min(1)

        # Enforce a common outward orientation before averaging observations.
        radial = F.normalize(fused_points[inverse] - center[None], dim=-1, eps=1e-6)
        aligned = torch.where(
            (normals * radial).sum(dim=-1, keepdim=True) < 0,
            -normals,
            normals,
        )
        fused_normals = torch.zeros_like(fused_points)
        fused_normals.index_add_(0, inverse, aligned)
        points = fused_points
        fused_radial = F.normalize(points - center[None], dim=-1, eps=1e-6)
        normals = torch.where(
            fused_normals.norm(dim=-1, keepdim=True) > 1e-6,
            F.normalize(fused_normals, dim=-1, eps=1e-6),
            fused_radial,
        )
        colors = fused_colors.clamp(0, 1)
        # -1 marks a unified world surface rather than source-owned shells.
        source_view_id = torch.full(
            (len(points),), -1, device=device, dtype=torch.long
        )
    if len(points) > max_points:
        ids = torch.linspace(0, len(points) - 1, max_points, device=device).long()
        points = points[ids]
        normals = normals[ids]
        colors = colors[ids]
        source_view_id = source_view_id[ids]
    # Compact anisotropic kernels so same-view splats cover the silhouette
    # without bleeding into neighboring views' depth layers.
    scales = points.new_tensor(
        [0.55 * spacing, 0.55 * spacing, 0.12 * spacing]
    ).expand(len(points), -1)
    return (
        points.cpu(),
        normals.cpu(),
        colors.cpu(),
        scales.cpu(),
        source_view_id.cpu(),
    )


@torch.inference_mode()
def initialize_photometric_normals(
    scene: MultiViewScene,
    points: torch.Tensor,
    frame_ids: list[int],
    device: torch.device | str,
    max_views: int = 16,
    face_camera: bool = True,
) -> torch.Tensor:
    """Robust calibrated photometric-stereo initialization at Gaussian points.

    ``face_camera=True`` orients normals toward each camera (Stanford-style).
    DiLiGenT-MV ground-truth normals use the opposite outward convention, so
    callers should pass ``face_camera=False`` there.
    """
    device = torch.device(device)
    points_device = points.to(device)
    normal_sum = torch.zeros_like(points_device)
    confidence_sum = torch.zeros(len(points_device), 1, device=device)
    allowed = set(frame_ids)
    groups = [
        [index for index in values if index in allowed]
        for values in scene.group_by_view().values()
    ]
    groups = [values for values in groups if len(values) >= 6][:max_views]
    for values in groups:
        frame = scene[values[0]]
        camera = frame.camera.to(device)
        uv, depth, projected = camera.project(points_device)
        pixel_x = uv[:, 0].round().long().clamp(0, camera.width - 1)
        pixel_y = uv[:, 1].round().long().clamp(0, camera.height - 1)
        pixel_index = pixel_y * camera.width + pixel_x
        nearest_depth = torch.full(
            (camera.height * camera.width,),
            torch.inf,
            device=device,
            dtype=depth.dtype,
        )
        if bool(projected.any()):
            nearest_depth.scatter_reduce_(
                0,
                pixel_index[projected],
                depth[projected],
                reduce="amin",
                include_self=True,
            )
        visible = projected & (
            depth <= nearest_depth[pixel_index] * 1.01 + 1e-4
        )
        mask = (
            _sample_image(frame.mask_gt.to(device), uv)[:, 0] > 0.5
            if frame.mask_gt is not None
            else visible
        )
        mask &= visible
        observations, lights = [], []
        for frame_id in values:
            observation = scene[frame_id]
            if observation.light_direction is None:
                continue
            rgb = _sample_image(observation.image.to(device), uv)
            observations.append(rgb.mean(dim=-1))
            lights.append(observation.light_direction.to(device))
        if len(observations) < 6:
            continue
        intensity = torch.stack(observations)
        light = F.normalize(torch.stack(lights), dim=-1)
        lower = torch.quantile(intensity, 0.15, dim=0)
        upper = torch.quantile(intensity, 0.70, dim=0)
        weight = (
            (intensity >= lower)
            & (intensity <= upper)
            & mask[None]
            & visible[None]
        ).float()
        a = torch.einsum("ln,li,lj->nij", weight, light, light)
        b = torch.einsum("ln,li,ln->ni", weight, light, intensity)
        a = a + 1e-4 * torch.eye(3, device=device)[None]
        gradient = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
        normal = F.normalize(gradient, dim=-1, eps=1e-6)
        if face_camera:
            reference = F.normalize(
                camera.center[None] - points_device, dim=-1, eps=1e-6
            )
        else:
            # A camera-relative sign is inconsistent when several views vote
            # for one world-space point. DiLiGenT uses the inward-oriented
            # convention exercised by the image-PS path, so align every view
            # to one shared radial reference before accumulation.
            reference = F.normalize(
                points_device.mean(dim=0, keepdim=True) - points_device,
                dim=-1,
                eps=1e-6,
            )
        facing = (normal * reference).sum(dim=-1, keepdim=True)
        flip = facing < 0
        normal = torch.where(flip, -normal, normal)
        confidence = weight.sum(dim=0) >= max(6, math.ceil(len(observations) * 0.3))
        confidence = confidence[:, None] & torch.isfinite(normal).all(dim=-1, keepdim=True)
        normal_sum += normal * confidence
        confidence_sum += confidence
    fallback = F.normalize(
        points_device - points_device.mean(dim=0, keepdim=True), dim=-1, eps=1e-6
    )
    if not face_camera:
        fallback = -fallback
    normals = torch.where(
        confidence_sum > 0,
        F.normalize(normal_sum, dim=-1, eps=1e-6),
        fallback,
    )
    return normals.cpu()


@torch.inference_mode()
def initialize_fused_image_photometric_normals(
    scene: MultiViewScene,
    points: torch.Tensor,
    frame_ids: list[int],
    device: torch.device | str,
    max_views: int = 20,
    face_camera: bool = False,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.55,
    irls_iterations: int = 0,
    huber_delta: float = 1.5,
    shadow_threshold: float = 0.01,
    saturation_threshold: float = 0.98,
) -> torch.Tensor:
    """Transfer robust per-view PS maps onto one visibility-filtered world surface."""
    device = torch.device(device)
    points_device = points.to(device)
    normal_sum = torch.zeros_like(points_device)
    confidence_sum = torch.zeros(len(points_device), 1, device=device)
    allowed = set(frame_ids)
    groups = [
        [index for index in values if index in allowed]
        for values in scene.group_by_view().values()
    ]
    groups = [values for values in groups if len(values) >= 6][:max_views]
    for values in groups:
        try:
            normal_map, confidence_map, camera = _image_photometric_normal_map(
                scene,
                values,
                device,
                face_camera=face_camera,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                irls_iterations=irls_iterations,
                huber_delta=huber_delta,
                shadow_threshold=shadow_threshold,
                saturation_threshold=saturation_threshold,
            )
        except ValueError:
            continue
        uv, depth, projected = camera.project(points_device)
        pixel_x = uv[:, 0].round().long().clamp(0, camera.width - 1)
        pixel_y = uv[:, 1].round().long().clamp(0, camera.height - 1)
        pixel_index = pixel_y * camera.width + pixel_x
        nearest_depth = torch.full(
            (camera.height * camera.width,),
            torch.inf,
            device=device,
            dtype=depth.dtype,
        )
        if bool(projected.any()):
            nearest_depth.scatter_reduce_(
                0,
                pixel_index[projected],
                depth[projected],
                reduce="amin",
                include_self=True,
            )
        visible = projected & (
            depth <= nearest_depth[pixel_index] * 1.01 + 1e-4
        )
        sampled_normal = _sample_image(normal_map, uv)
        reference = F.normalize(
            points_device.mean(dim=0, keepdim=True) - points_device,
            dim=-1,
            eps=1e-6,
        )
        sampled_normal = torch.where(
            (sampled_normal * reference).sum(dim=-1, keepdim=True) < 0,
            -sampled_normal,
            sampled_normal,
        )
        sampled_confidence = _sample_image(confidence_map[None], uv)[:, :1]
        confidence = (
            visible[:, None]
            & (sampled_confidence > 0.5)
            & torch.isfinite(sampled_normal).all(dim=-1, keepdim=True)
        )
        normal_sum += sampled_normal * confidence
        confidence_sum += confidence
    fallback = F.normalize(
        points_device.mean(dim=0, keepdim=True) - points_device,
        dim=-1,
        eps=1e-6,
    )
    normals = torch.where(
        confidence_sum > 0,
        F.normalize(normal_sum, dim=-1, eps=1e-6),
        fallback,
    )
    return normals.cpu()


@torch.inference_mode()
def attach_image_photometric_normal_priors(
    scene: MultiViewScene,
    frame_ids: list[int],
    *,
    device: torch.device | str,
    max_views: int = 20,
    face_camera: bool = False,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.55,
    irls_iterations: int = 0,
    huber_delta: float = 1.5,
    shadow_threshold: float = 0.01,
    saturation_threshold: float = 0.98,
) -> int:
    """Attach train-view PS maps for image-space normal supervision."""
    device = torch.device(device)
    allowed = set(frame_ids)
    groups = [
        [index for index in values if index in allowed]
        for values in scene.group_by_view().values()
    ]
    groups = [values for values in groups if len(values) >= 6][:max_views]
    attached = 0
    for values in groups:
        try:
            normal_map, confidence_map, _ = _image_photometric_normal_map(
                scene,
                values,
                device,
                face_camera=face_camera,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                irls_iterations=irls_iterations,
                huber_delta=huber_delta,
                shadow_threshold=shadow_threshold,
                saturation_threshold=saturation_threshold,
            )
        except ValueError:
            continue
        normal_cpu = normal_map.detach().cpu()
        confidence_cpu = confidence_map[None].detach().cpu()
        for frame_id in values:
            scene.frames[frame_id] = replace(
                scene.frames[frame_id],
                normal_prior_image=normal_cpu,
                normal_prior_confidence=confidence_cpu,
            )
            attached += 1
    return attached


@torch.inference_mode()
def attach_visual_hull_depth_priors(
    scene: MultiViewScene,
    frame_ids: list[int],
    *,
    resolution: int = 96,
    device: torch.device | str = "cpu",
) -> int:
    """Attach soft per-pixel depth priors from silhouette visual hull (NoGT).

    Writes ``frame.depth_prior`` for masked pixels. Does not touch ``depth_gt``.
    Returns the number of frames that received a prior.
    """
    device = torch.device(device)
    occupancy, center, radius, _ = _build_occupancy_volume(
        scene, frame_ids, resolution=resolution, device=device
    )
    attached = 0
    for frame_id in frame_ids:
        frame = scene[frame_id]
        if frame.mask_gt is None:
            continue
        mask = frame.mask_gt[0] > 0.5
        ys, xs = torch.where(mask)
        if len(ys) == 0:
            continue
        camera = frame.camera.to(device)
        depth = _ray_visual_hull_depth(
            camera, occupancy, center, radius, xs.to(device), ys.to(device)
        )
        prior = torch.zeros(1, camera.height, camera.width, dtype=torch.float32)
        prior[0, ys.cpu(), xs.cpu()] = depth.detach().cpu().float()
        scene.frames[frame_id] = replace(frame, depth_prior=prior)
        attached += 1
    return attached


_MONO_DEPTH_CACHE: dict[str, torch.nn.Module] = {}


def _load_monocular_depth_model(device: torch.device) -> torch.nn.Module:
    """Load and cache the MiDaS monocular relative-depth network."""
    if "midas" not in _MONO_DEPTH_CACHE:
        model = torch.hub.load(
            "isl-org/MiDaS", "DPT_Hybrid", pretrained=True, trust_repo=True
        )
        model.eval().to(device)
        _MONO_DEPTH_CACHE["midas"] = model
    return _MONO_DEPTH_CACHE["midas"]


@torch.no_grad()
def attach_monocular_depth_priors(
    scene: MultiViewScene,
    frame_ids: list[int],
    *,
    device: torch.device | str = "cpu",
    input_size: int = 384,
) -> int:
    """Attach per-pixel monocular relative-depth priors (MiDaS, NoGT).

    Writes ``frame.depth_prior`` for masked pixels.  MiDaS predicts inverse
    depth (larger = closer); it is inverted to a depth ordering and normalized
    by the in-mask median.  The scale-invariant log-depth loss consumes the
    prior without requiring metric scale, and ``depth_gt`` is never touched.
    Returns the number of frames that received a prior.
    """
    device = torch.device(device)
    model = _load_monocular_depth_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    attached = 0
    for frame_id in frame_ids:
        frame = scene[frame_id]
        if frame.mask_gt is None:
            continue
        image = frame.image.to(device).float()
        h, w = image.shape[-2:]
        inp = F.interpolate(
            image.unsqueeze(0),
            size=(input_size, input_size),
            mode="bicubic",
            align_corners=False,
        )
        inp = (inp - mean) / std
        disparity = model(inp)
        disparity = F.interpolate(
            disparity.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        depth = 1.0 / disparity.clamp_min(1e-3)
        mask = frame.mask_gt[0].to(device) > 0.5
        if not bool(mask.any()):
            continue
        depth = depth / depth[mask].median().clamp_min(1e-6)
        prior = torch.zeros(1, h, w, dtype=torch.float32)
        prior[0] = (depth * mask.float()).cpu()
        scene.frames[frame_id] = replace(frame, depth_prior=prior)
        attached += 1
    return attached


@torch.no_grad()
def attach_monocular_normal_priors(
    scene: MultiViewScene,
    frame_ids: list[int],
    *,
    device: torch.device | str = "cpu",
    input_size: int = 384,
) -> int:
    """Attach world-space monocular normal priors from MiDaS depth (NoGT).

    Runs the MiDaS network on each training view, unprojects the relative
    depth with the known intrinsics, and converts neighbor-span cross products
    into world-space normals written to ``frame.normal_prior_image`` with a
    continuity-based ``frame.normal_prior_confidence``.  Unlike the depth
    prior, this supervises only the shading-normal field through
    ``image_normal_prior_weight``; scene geometry is never touched, so the
    prior cannot destabilize the reconstructed surface.  Returns the number of
    frames that received a prior.
    """
    device = torch.device(device)
    model = _load_monocular_depth_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    attached = 0
    for frame_id in frame_ids:
        frame = scene[frame_id]
        if frame.mask_gt is None:
            continue
        image = frame.image.to(device).float()
        h, w = image.shape[-2:]
        inp = F.interpolate(
            image.unsqueeze(0),
            size=(input_size, input_size),
            mode="bicubic",
            align_corners=False,
        )
        inp = (inp - mean) / std
        disparity = model(inp)
        disparity = F.interpolate(
            disparity.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        depth = 1.0 / disparity.clamp_min(1e-3)
        mask = frame.mask_gt[0].to(device) > 0.5
        if not bool(mask.any()):
            continue

        camera = frame.camera.to(device)
        K = camera.K
        c2w = camera.c2w
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=depth.dtype),
            torch.arange(w, device=device, dtype=depth.dtype),
            indexing="ij",
        )
        pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
        directions = pix @ torch.linalg.inv(K).T
        points_cam = directions * depth[..., None]

        span_x = points_cam[1:-1, 2:] - points_cam[1:-1, :-2]
        span_y = points_cam[2:, 1:-1] - points_cam[:-2, 1:-1]
        normal_cam = F.normalize(
            torch.cross(span_y, span_x, dim=-1), dim=-1, eps=1e-6
        )
        rotation = c2w[:3, :3]
        normal_world = F.normalize(normal_cam @ rotation.T, dim=-1, eps=1e-6)
        points_world = points_cam[1:-1, 1:-1] @ rotation.T + c2w[:3, 3]
        view_dir = F.normalize(c2w[:3, 3] - points_world, dim=-1, eps=1e-6)
        sign = torch.where(
            (normal_world * view_dir).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0
        )
        normal_world = normal_world * sign

        center_depth = depth[1:-1, 1:-1]
        continuity = (
            (depth[1:-1, 2:] - depth[1:-1, :-2]).abs()
            <= 0.10 * center_depth.clamp_min(1e-4)
        ) & (
            (depth[2:, 1:-1] - depth[:-2, 1:-1]).abs()
            <= 0.10 * center_depth.clamp_min(1e-4)
        )
        interior = mask[1:-1, 1:-1] & continuity

        normal_full = torch.zeros(3, h, w, dtype=torch.float32)
        conf_full = torch.zeros(1, h, w, dtype=torch.float32)
        normal_full[:, 1:-1, 1:-1] = normal_world.permute(2, 0, 1).cpu()
        conf_full[:, 1:-1, 1:-1] = interior.float()[None].cpu()
        scene.frames[frame_id] = replace(
            frame,
            normal_prior_image=normal_full,
            normal_prior_confidence=conf_full,
        )
        attached += 1
    return attached


def _load_omnidata_normal_model(device: torch.device) -> torch.nn.Module:
    """Load and cache the Omnidata DPT-Hybrid surface-normal network."""
    if "omnidata_normal" not in _MONO_DEPTH_CACHE:
        # torch>=2.11 raises a spurious KeyError in the fork-repo validation
        # for non-allowlisted hubs; the repo is trusted, so bypass the check.
        torch.hub._validate_not_a_forked_repo = lambda *a, **k: None
        model = torch.hub.load(
            "alexsax/omnidata_models:main",
            "surface_normal_dpt_hybrid_384",
            pretrained=True,
            trust_repo=True,
            verbose=False,
        )
        model.eval().to(device)
        _MONO_DEPTH_CACHE["omnidata_normal"] = model
    return _MONO_DEPTH_CACHE["omnidata_normal"]


@torch.no_grad()
def attach_omnidata_normal_priors(
    scene: MultiViewScene,
    frame_ids: list[int],
    *,
    device: torch.device | str = "cpu",
    input_size: int = 384,
    edge_erode: int = 0,
) -> int:
    """Attach world-space monocular normal priors from Omnidata (NoGT).

    Unlike the MiDaS-derived prior, Omnidata regresses the surface normal
    directly, yielding smooth, full-coverage maps rather than sparse
    depth-gradient estimates.  The camera-space prediction is mapped to the
    OpenCV-style frame used throughout (its +z axis points toward the camera,
    so the z component is negated), rotated into world space, and oriented
    toward the camera.  The whole object mask is used as the confidence since
    the prediction is smooth and densely supervised.  Only the shading-normal
    field is supervised; scene geometry is never touched.  Returns the number
    of frames that received a prior.
    """
    device = torch.device(device)
    model = _load_omnidata_normal_model(device)
    zflip = torch.tensor([1.0, 1.0, -1.0], device=device).view(3, 1, 1)
    attached = 0
    for frame_id in frame_ids:
        frame = scene[frame_id]
        if frame.mask_gt is None:
            continue
        image = frame.image.to(device).float()
        h, w = image.shape[-2:]
        if float(image.max()) > 1.5:  # linear HDR -> display range
            image = image / (1.0 + image)
        image = image.clamp(0.0, 1.0)
        inp = F.interpolate(
            image.unsqueeze(0),
            size=(input_size, input_size),
            mode="bicubic",
            align_corners=False,
        )
        out = model(inp).clamp(0.0, 1.0)[0]
        normal_cam = F.interpolate(
            (out * 2.0 - 1.0).unsqueeze(0),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )[0]
        normal_cam = F.normalize(normal_cam * zflip, dim=0, eps=1e-6)

        camera = frame.camera.to(device)
        rotation = camera.c2w[:3, :3]
        normal_world = F.normalize(
            normal_cam.permute(1, 2, 0) @ rotation.T, dim=-1, eps=1e-6
        )
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
        rays_world = F.normalize(
            (pix @ torch.linalg.inv(camera.K).T) @ rotation.T, dim=-1, eps=1e-6
        )
        sign = torch.where(
            (normal_world * rays_world).sum(dim=-1, keepdim=True) > 0, -1.0, 1.0
        )
        normal_world = F.normalize(normal_world * sign, dim=-1, eps=1e-6)

        mask = (frame.mask_gt[0].to(device) > 0.5).float()
        if edge_erode > 0:
            # Erode the mask so only reliable interior pixels are supervised,
            # dropping the grazing-angle silhouette normals that monocular
            # regressors estimate poorly (min-pool = binary erosion).
            k = 2 * int(edge_erode) + 1
            mask = -F.max_pool2d(
                -mask.view(1, 1, h, w), kernel_size=k, stride=1, padding=int(edge_erode)
            ).view(h, w)
        normal_full = torch.zeros(3, h, w, dtype=torch.float32)
        conf_full = torch.zeros(1, h, w, dtype=torch.float32)
        normal_full[:] = normal_world.permute(2, 0, 1).cpu()
        conf_full[0] = mask.cpu()
        scene.frames[frame_id] = replace(
            frame,
            normal_prior_image=normal_full,
            normal_prior_confidence=conf_full,
        )
        attached += 1
    return attached
