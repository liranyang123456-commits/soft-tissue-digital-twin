"""Train a persistent world-space 3D Gaussian BRDF scene from many views."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ..models.pipeline_pro import MVBRDFSHRPro
from .data import MultiViewScene
from .geometry_init import (
    attach_image_photometric_normal_priors,
    attach_monocular_depth_priors,
    attach_monocular_normal_priors,
    attach_omnidata_normal_priors,
    attach_visual_hull_depth_priors,
    initialize_fused_image_photometric_normals,
    initialize_image_photometric_stereo,
    initialize_monocular_fused_points,
    initialize_multiview_mask_points,
    initialize_photometric_normals,
    initialize_visual_hull,
)
from .losses import WorldLossWeights, world_total_loss
from .pipeline import WorldGaussianBRDFPipeline
from .scene import GaussianBRDFField


def initialize_from_first_view(
    dataset: MultiViewScene,
    n_gaussians: int,
    depth_near: float = 1.0,
    depth_far: float = 3.0,
    frame_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback initialization when COLMAP has no sparse points."""
    frame = dataset[frame_id]
    origins, directions = frame.camera.pixel_rays()
    flat_origins = origins.reshape(-1, 3)
    flat_directions = directions.reshape(-1, 3)
    image = frame.image.permute(1, 2, 0).reshape(-1, 3)
    count = min(n_gaussians, flat_directions.shape[0])
    ids = torch.linspace(0, flat_directions.shape[0] - 1, count).long()
    depths = torch.linspace(depth_near, depth_far, count)
    points = flat_origins[ids] + flat_directions[ids] * depths[:, None]
    return points, image[ids]


def _camera_centers_by_view(
    dataset: MultiViewScene, device: torch.device
) -> torch.Tensor:
    """Build a dense view-id to camera-center lookup for soft view locking."""
    entries: list[tuple[int, torch.Tensor]] = []
    for index, frame in enumerate(dataset.frames):
        view_id = (
            frame.view_id
            if frame.view_id is not None
            else getattr(frame.camera, "view_id", index)
        )
        if view_id is not None:
            entries.append((int(view_id), frame.camera.center.detach().to(device)))
    if not entries:
        return torch.empty(0, 3, device=device)
    centers = torch.full(
        (max(view_id for view_id, _ in entries) + 1, 3),
        float("nan"),
        device=device,
    )
    for view_id, center in entries:
        if not torch.isfinite(centers[view_id]).all():
            centers[view_id] = center
    return centers


@torch.inference_mode()
def predict_pseudo_diffuse(
    dataset: MultiViewScene,
    checkpoint: str | Path,
    frame_indices: list[int],
    device: torch.device,
    confidence_floor: float = 0.2,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Predict train-view diffuse priors without touching holdout images."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = state.get("model", state)
    # Auto-detect restorer kind: DHAN checkpoints contain 'restorer.proc.',
    # UNet checkpoints contain 'restorer.stem.'.
    restorer_kind = "dhan" if any("restorer.proc." in k for k in sd) else "unet"
    model = MVBRDFSHRPro(128, 48, 96, restorer_kind=restorer_kind).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    images, confidences = {}, {}
    for frame_id in frame_indices:
        image = dataset[frame_id].image.to(device).unsqueeze(0)
        _, _, h, w = image.shape
        # DHAN + geometry/material U-Nets require square (or evenly-divisible)
        # inputs for their skip-connections. Pad to square, predict, crop back.
        side = max(h, w)
        pad_h, pad_w = side - h, side - w
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=1.0)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(image, use_refine=True, detach_physics=True)
        pred = output["pred"][0, :, :h, :w].float().cpu()
        mask_pred = output["mask_pred"][0, :, :h, :w]
        images[frame_id] = pred
        # Prefer views where the single-image branch reports little ambiguity,
        # but retain a non-zero contribution in repaired highlight regions.
        confidence = confidence_floor + (1 - confidence_floor) * (1 - mask_pred.float())
        confidences[frame_id] = confidence.clamp(0, 1).cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return images, confidences


def build_pipeline(
    dataset: MultiViewScene,
    cfg: dict,
    device: torch.device,
    train_ids: list[int] | None = None,
):
    n_gaussians = int(cfg.get("n_gaussians", 512))
    pseudo_checkpoint = cfg.get("pseudo_diffuse_checkpoint")
    selected_ids = train_ids or list(range(len(dataset)))
    point_normals = None
    point_scales = None
    source_view_id = None

    if bool(cfg.get("image_photometric_stereo_init", False)):
        (
            points,
            point_normals,
            colors,
            point_scales,
            source_view_id,
        ) = initialize_image_photometric_stereo(
            dataset,
            selected_ids,
            max_points=n_gaussians,
            max_views=int(cfg.get("photometric_stereo_max_views", 12)),
            device=device,
            face_camera=bool(cfg.get("photometric_stereo_face_camera", False)),
            hull_resolution=int(cfg.get("image_ps_hull_resolution", 48)),
            fuse_world_surface=bool(cfg.get("image_ps_fuse_world_surface", False)),
            lower_quantile=float(cfg.get("image_ps_lower_quantile", 0.10)),
            upper_quantile=float(cfg.get("image_ps_upper_quantile", 0.55)),
            irls_iterations=int(cfg.get("image_ps_irls_iterations", 0)),
            huber_delta=float(cfg.get("image_ps_huber_delta", 1.5)),
            shadow_threshold=float(cfg.get("image_ps_shadow_threshold", 0.01)),
            saturation_threshold=float(
                cfg.get("image_ps_saturation_threshold", 0.98)
            ),
        )
    elif bool(cfg.get("monocular_fused_init", False)):
        points, point_normals, point_scales = initialize_monocular_fused_points(
            dataset,
            selected_ids,
            max_points=n_gaussians,
            resolution=int(cfg.get("visual_hull_resolution", 64)),
            min_view_fraction=float(
                cfg.get("visual_hull_min_view_fraction", 0.8)
            ),
            normal_smoothing_kernel=int(
                cfg.get("visual_hull_normal_smoothing_kernel", 1)
            ),
            device=device,
            mono_input_size=int(cfg.get("monocular_depth_input_size", 384)),
            depth_points_per_view=int(cfg.get("mono_fused_points_per_view", 6000)),
        )
        colors = dataset.fuse_projected_texture(points, selected_ids)
    elif bool(cfg.get("visual_hull_init", False)):
        points, point_normals, point_scales = initialize_visual_hull(
            dataset,
            selected_ids,
            max_points=n_gaussians,
            resolution=int(cfg.get("visual_hull_resolution", 64)),
            min_view_fraction=float(
                cfg.get("visual_hull_min_view_fraction", 0.8)
            ),
            normal_smoothing_kernel=int(
                cfg.get("visual_hull_normal_smoothing_kernel", 1)
            ),
            coarse_resolution=int(
                cfg.get("visual_hull_coarse_resolution", 0)
            ),
            device=device,
        )
        colors = dataset.fuse_projected_texture(points, selected_ids)
    elif bool(cfg.get("multiview_mask_init", False)):
        points, colors, point_normals = initialize_multiview_mask_points(
            dataset,
            selected_ids,
            max_points=n_gaussians,
            views=int(cfg.get("multiview_mask_views", 8)),
            device=device,
        )
    if dataset.points is not None:
        if not any(
            bool(cfg.get(key, False))
            for key in (
                "image_photometric_stereo_init",
                "visual_hull_init",
                "multiview_mask_init",
            )
        ):
            count = min(n_gaussians, len(dataset.points))
            ids = torch.linspace(0, len(dataset.points) - 1, count).long()
            points = dataset.points[ids]
            projected = dataset.fuse_projected_texture(points, selected_ids)
            point_color_weight = float(cfg.get("point_color_weight", 0.5))
            colors = (
                point_color_weight * dataset.point_colors[ids]
                + (1 - point_color_weight) * projected
                if dataset.point_colors is not None
                else projected
            )
            if dataset.point_normals is not None:
                point_normals = dataset.point_normals[ids]
            if dataset.point_scales is not None:
                point_scales = dataset.point_scales[ids]
    elif not any(
        bool(cfg.get(key, False))
        for key in (
            "image_photometric_stereo_init",
            "visual_hull_init",
            "multiview_mask_init",
        )
    ):
        first_train_id = train_ids[0] if train_ids else 0
        points, colors = initialize_from_first_view(
            dataset, n_gaussians, frame_id=first_train_id
        )
        colors = dataset.fuse_projected_texture(points, selected_ids)
    if pseudo_checkpoint:
        pseudo_images, pseudo_confidences = predict_pseudo_diffuse(
            dataset,
            pseudo_checkpoint,
            selected_ids,
            device,
            float(cfg.get("pseudo_diffuse_confidence_floor", 0.2)),
        )
        pseudo_colors = dataset.fuse_projected_texture(
            points,
            selected_ids,
            source_images=pseudo_images,
            confidence_maps=pseudo_confidences,
        )
        blend = float(cfg.get("pseudo_diffuse_blend", 1.0))
        colors = colors.lerp(pseudo_colors, blend)
    elif cfg.get("initial_albedo") is not None:
        colors = torch.full_like(colors, float(cfg["initial_albedo"]))
    if bool(cfg.get("fused_image_photometric_stereo_init", False)):
        point_normals = initialize_fused_image_photometric_normals(
            dataset,
            points,
            selected_ids,
            device,
            max_views=int(cfg.get("photometric_stereo_max_views", 20)),
            face_camera=bool(cfg.get("photometric_stereo_face_camera", False)),
            lower_quantile=float(cfg.get("image_ps_lower_quantile", 0.10)),
            upper_quantile=float(cfg.get("image_ps_upper_quantile", 0.55)),
            irls_iterations=int(cfg.get("image_ps_irls_iterations", 0)),
            huber_delta=float(cfg.get("image_ps_huber_delta", 1.5)),
            shadow_threshold=float(cfg.get("image_ps_shadow_threshold", 0.01)),
            saturation_threshold=float(
                cfg.get("image_ps_saturation_threshold", 0.98)
            ),
        )
    elif bool(cfg.get("photometric_stereo_init", False)):
        point_normals = initialize_photometric_normals(
            dataset,
            points,
            selected_ids,
            device,
            max_views=int(cfg.get("photometric_stereo_max_views", 16)),
            face_camera=bool(cfg.get("photometric_stereo_face_camera", True)),
        )
    if point_normals is not None and bool(cfg.get("invert_initial_normals", False)):
        point_normals = -point_normals
    normal_mode = str(cfg.get("normal_mode", "learned"))
    scene = GaussianBRDFField.from_point_cloud(
        points.to(device),
        colors.to(device) if colors is not None else None,
        initial_scale=(
            point_scales.to(device) if point_scales is not None else None
        ),
        normal_mode=normal_mode,
    )
    if point_normals is not None:
        normalized = F.normalize(point_normals.to(device), dim=-1, eps=1e-6)
        scene.initialize_surface_normals(normalized)
        scene.register_buffer("normal_prior", normalized.detach().clone())
    else:
        scene.register_buffer("normal_prior", torch.zeros_like(scene.normal_raw))
    if source_view_id is not None:
        scene.register_buffer(
            "source_view_id", source_view_id.to(device=device, dtype=torch.long)
        )
        scene.view_locked_opacity = bool(cfg.get("view_locked_opacity", True))
    else:
        scene.register_buffer(
            "source_view_id",
            torch.full(
                (scene.n_gaussians,), -1, device=device, dtype=torch.long
            ),
        )
        scene.view_locked_opacity = False
    scene.register_buffer(
        "source_confidence",
        torch.ones(scene.n_gaussians, 1, device=device),
    )
    scene.register_buffer(
        "source_camera_centers", _camera_centers_by_view(dataset, device)
    )
    scene.view_lock_neighbors = int(cfg.get("view_lock_neighbors", 0))
    scene.view_lock_mode = str(cfg.get("view_lock_mode", "hard"))
    scene.view_lock_temperature = float(cfg.get("view_lock_temperature", 0.45))
    scene.view_lock_min_weight = float(cfg.get("view_lock_min_weight", 0.02))
    if normal_mode == "covariance":
        scene.normal_raw.requires_grad_(False)
    gaussian_init = str(cfg.get("gaussian_init", "point_cloud"))
    if gaussian_init in {"geometry", "geometry_albedo", "oracle"}:
        prior_path = Path(cfg["data_root"]) / str(
            cfg.get("gaussian_prior", "gaussian_brdf_gt.npz")
        )
        if not prior_path.exists():
            raise FileNotFoundError(f"Gaussian prior not found: {prior_path}")
        scene.initialize_from_npz(str(prior_path), mode=gaussian_init)
    pipeline = WorldGaussianBRDFPipeline(
        scene=scene,
        n_frames=len(dataset),
        lighting_mode=str(cfg.get("lighting_mode", "sh")),
        use_refiner=bool(cfg.get("use_refiner", True)),
        refiner_base=int(cfg.get("refiner_base", 96)),
        chunk_size=int(cfg.get("chunk_size", 32)),
        raster_backend=str(cfg.get("raster_backend", "auto")),
        shared_lighting=bool(cfg.get("shared_lighting", False)),
        background_color=cfg.get("background_color"),
        linear_hdr=bool(cfg.get("linear_hdr", False)),
        shadow_mode=str(cfg.get("shadow_mode", "none")),
        shadow_strength=float(cfg.get("shadow_strength", 1.0)),
        light_ids=[frame.light_id for frame in dataset.frames],
        use_implicit_material=bool(cfg.get("use_implicit_material", False)),
        implicit_material_hidden=int(cfg.get("implicit_material_hidden", 64)),
        implicit_material_layers=int(cfg.get("implicit_material_layers", 3)),
        implicit_material_residual_limit=float(
            cfg.get("implicit_material_residual_limit", 0.10)
        ),
        use_neural_brdf=bool(cfg.get("use_neural_brdf", False)),
        neural_brdf_material_latent=int(cfg.get("neural_brdf_material_latent", 32)),
        neural_brdf_light_latent=int(cfg.get("neural_brdf_light_latent", 16)),
        neural_brdf_hidden=int(cfg.get("neural_brdf_hidden", 96)),
        neural_brdf_strength=float(cfg.get("neural_brdf_strength", 1.0)),
        neural_brdf_mode=str(cfg.get("neural_brdf_mode", "replacement")),
        neural_brdf_residual_limit=float(
            cfg.get("neural_brdf_residual_limit", 0.35)
        ),
        neural_brdf_highlight_gate=bool(
            cfg.get("neural_brdf_highlight_gate", False)
        ),
        neural_brdf_boundary_fallback=bool(
            cfg.get("neural_brdf_boundary_fallback", False)
        ),
    ).to(device)
    pretrained_brdf = cfg.get("neural_brdf_pretrained")
    if pipeline.neural_brdf is not None and pretrained_brdf:
        pretrained_state = torch.load(
            Path(pretrained_brdf), map_location=device, weights_only=False
        )
        pipeline.neural_brdf.load_state_dict(pretrained_state["model"])
        print(f"loaded synthetic neural BRDF: {pretrained_brdf}", flush=True)

    shiq_checkpoint = cfg.get("shiq_refiner_checkpoint")
    if shiq_checkpoint and pipeline.refiner is not None and Path(shiq_checkpoint).exists():
        copied = pipeline.load_shiq_refiner(shiq_checkpoint)
        print(f"transferred {copied} SHIQ Restorer tensors")

    # Initialize known per-frame capture lighting when metadata is available.
    with torch.no_grad():
        for index, frame in enumerate(dataset.frames):
            light_index = int(pipeline.frame_lighting.light_index(index))
            if frame.light_position is not None:
                pipeline.frame_lighting.position[light_index].copy_(
                    frame.light_position.to(device)
                )
            if frame.light_direction is not None:
                pipeline.frame_lighting.direction_raw[light_index].copy_(
                    frame.light_direction.to(device)
                )
            if frame.light_color is not None:
                color = frame.light_color.to(device).clamp(1e-4, 1 - 1e-4)
                pipeline.frame_lighting.color_logits[light_index].copy_(
                    torch.logit(color)
                )
            if frame.point_light_weight is not None:
                weight = min(max(frame.point_light_weight, 1e-4), 1.0 - 1e-4)
                pipeline.frame_lighting.point_light_logits[light_index].fill_(
                    torch.logit(torch.tensor(weight)).item()
                )
            elif frame.light_position is not None and frame.light_direction is None:
                pipeline.frame_lighting.point_light_logits[light_index].fill_(8.0)
            if frame.light_intensity is not None:
                pipeline.frame_lighting.log_intensity[light_index].fill_(
                    max(frame.light_intensity, 1e-4)
                )
                pipeline.frame_lighting.log_intensity[light_index].log_()
            if frame.env_sh is not None:
                pipeline.frame_lighting.env_sh[light_index].copy_(
                    frame.env_sh.to(device)
                )
            pipeline.frame_lighting.log_exposure[index].fill_(
                torch.as_tensor(frame.exposure).clamp_min(1e-4).log()
            )
            if frame.white_balance is not None:
                white_balance = frame.white_balance.to(device).clamp_min(1e-4)
                pipeline.frame_lighting.log_white_balance[index].copy_(
                    white_balance.log()
                )
    return pipeline


def split_frame_indices(n_frames: int, cfg: dict) -> tuple[list[int], list[int]]:
    """Return disjoint train/holdout frame indices from the scene config."""
    stride = int(cfg.get("holdout_stride", 0))
    offset = int(cfg.get("holdout_offset", 0))
    if stride <= 1:
        return list(range(n_frames)), []
    holdout = [index for index in range(n_frames) if index % stride == offset % stride]
    train = [index for index in range(n_frames) if index not in set(holdout)]
    if not train or not holdout:
        raise ValueError("holdout split must contain both train and holdout frames")
    return train, holdout


def split_scene_indices(
    dataset: MultiViewScene, cfg: dict
) -> tuple[list[int], list[int]]:
    """Split by dataset labels, light id, or persistent view id."""
    if bool(cfg.get("use_dataset_split", False)):
        train = [
            index
            for index, frame in enumerate(dataset.frames)
            if (frame.metadata or {}).get("split") == "train"
        ]
        holdout = [
            index
            for index, frame in enumerate(dataset.frames)
            if (frame.metadata or {}).get("split") in {"holdout", "test"}
        ]
        if train:
            return train, holdout
    light_stride = int(cfg.get("holdout_light_stride", 0))
    light_offset = int(cfg.get("holdout_light_offset", 0))
    if light_stride > 1:
        holdout = [
            index
            for index, frame in enumerate(dataset.frames)
            if frame.light_id is not None
            and frame.light_id % light_stride == light_offset % light_stride
        ]
        holdout_set = set(holdout)
        train = [
            index for index in range(len(dataset)) if index not in holdout_set
        ]
        if not train or not holdout:
            raise ValueError(
                "light holdout split must contain both train and holdout frames"
            )
        return train, holdout
    stride = int(
        cfg.get("holdout_view_stride", cfg.get("holdout_stride", 0))
    )
    offset = int(
        cfg.get("holdout_view_offset", cfg.get("holdout_offset", 0))
    )
    if stride <= 1:
        return list(range(len(dataset))), []
    holdout_views = {
        frame.view_id
        for frame in dataset.frames
        if frame.view_id % stride == offset % stride
    }
    holdout = [
        index
        for index, frame in enumerate(dataset.frames)
        if frame.view_id in holdout_views
    ]
    train = [
        index
        for index, frame in enumerate(dataset.frames)
        if frame.view_id not in holdout_views
    ]
    if not train or not holdout:
        raise ValueError("holdout split must contain both train and holdout views")
    return train, holdout


def sample_frame_indices(
    dataset: MultiViewScene,
    candidates: list[int],
    count: int,
    mode: str = "random",
) -> list[int]:
    """Sample frames while preserving requested view/light structure."""
    count = min(int(count), len(candidates))
    if count <= 0:
        return []
    if mode == "random":
        return random.sample(candidates, count)
    key_name = "view_id" if mode == "same_view" else "light_id"
    if mode in {"same_view", "same_light"}:
        groups: dict[int | None, list[int]] = {}
        for index in candidates:
            groups.setdefault(getattr(dataset[index], key_name), []).append(index)
        eligible = [group for group in groups.values() if len(group) >= count]
        if not eligible:
            return random.sample(candidates, count)
        return random.sample(random.choice(eligible), count)
    if mode == "distinct_views":
        groups: dict[int, list[int]] = {}
        for index in candidates:
            groups.setdefault(dataset[index].view_id, []).append(index)
        views = random.sample(list(groups), min(count, len(groups)))
        selected = [random.choice(groups[view]) for view in views]
        if len(selected) < count:
            remaining = [index for index in candidates if index not in selected]
            selected.extend(random.sample(remaining, count - len(selected)))
        return selected
    raise ValueError(
        "view sampling must be random, same_view, same_light, or distinct_views"
    )


def load_dataset(cfg: dict) -> MultiViewScene:
    root = cfg["data_root"]
    kind = str(cfg.get("data_type", "blender")).lower()
    scale = float(cfg.get("image_scale", 1.0))
    if kind == "stanford_orb":
        from .datasets import load_stanford_orb

        canonical = load_stanford_orb(
            root,
            transforms=cfg.get("transforms"),
            source_is_linear=bool(cfg.get("source_is_linear", False)),
            ground_truth_root=cfg.get("ground_truth_root"),
            max_frames_per_split=cfg.get("max_frames_per_split"),
            include_novel=bool(cfg.get("include_novel", False)),
        )
        return MultiViewScene.from_canonical(
            canonical,
            share_light_ids=bool(cfg.get("share_light_ids", True)),
            scale=scale,
            world_scale=float(cfg.get("world_scale", 1.0)),
            use_mesh_points=bool(cfg.get("use_mesh_points", False)),
            max_mesh_points=int(cfg.get("max_mesh_points", 50000)),
            mesh_sampling=str(cfg.get("mesh_sampling", "surface")),
            mesh_normal_scale_ratio=float(
                cfg.get("mesh_normal_scale_ratio", 0.15)
            ),
        )
    if kind == "diligent_mv":
        from .datasets import load_diligent_mv

        canonical = load_diligent_mv(
            root,
            source_is_linear=bool(cfg.get("source_is_linear", True)),
            strict_counts=bool(cfg.get("strict_counts", False)),
            max_views=cfg.get("max_views"),
            max_lights_per_view=cfg.get("max_lights_per_view"),
        )
        return MultiViewScene.from_canonical(
            canonical,
            share_light_ids=bool(cfg.get("share_light_ids", True)),
            scale=scale,
            world_scale=float(cfg.get("world_scale", 1.0)),
            use_mesh_points=bool(cfg.get("use_mesh_points", False)),
            max_mesh_points=int(cfg.get("max_mesh_points", 50000)),
            mesh_sampling=str(cfg.get("mesh_sampling", "surface")),
            mesh_normal_scale_ratio=float(
                cfg.get("mesh_normal_scale_ratio", 0.15)
            ),
        )
    if kind == "blender":
        return MultiViewScene.from_blender(
            root, str(cfg.get("transforms", "transforms.json")), scale
        )
    if kind == "colmap":
        return MultiViewScene.from_colmap(
            root,
            str(cfg.get("sparse_dir", "sparse/0")),
            str(cfg.get("images_dir", "images")),
            scale,
        )
    if kind == "mvsnet":
        return MultiViewScene.from_mvsnet(
            root,
            str(cfg.get("images_dir", "blended_images")),
            str(cfg.get("cams_dir", "cams")),
            str(cfg.get("depth_dir", "rendered_depth_maps")),
            scale,
            int(cfg.get("max_init_points", 50000)),
            int(cfg.get("max_point_views", 16)),
        )
    if kind == "dtu":
        return MultiViewScene.from_dtu(
            root,
            int(cfg.get("scan", 1)),
            int(cfg.get("light", 3)),
            scale,
            int(cfg.get("max_init_points", 50000)),
        )
    raise ValueError(f"unknown data_type {kind}")


def _pipeline_from_state(
    dataset: MultiViewScene,
    cfg: dict,
    state: dict,
    device: torch.device,
) -> WorldGaussianBRDFPipeline:
    """Rebuild a pipeline with checkpoint-sized Gaussian buffers."""
    model = state["model"]
    points = state.get("points", model["scene.means"]).to(device)
    colors = state.get("colors")
    colors = colors.to(device) if isinstance(colors, torch.Tensor) else None
    scene = GaussianBRDFField.from_point_cloud(
        points,
        colors,
        normal_mode=str(cfg.get("normal_mode", "learned")),
    )
    normal_prior = model.get("scene.normal_prior")
    scene.register_buffer(
        "normal_prior",
        (
            normal_prior.to(device)
            if isinstance(normal_prior, torch.Tensor)
            else torch.zeros_like(scene.normal_raw)
        ),
    )
    source = model.get("scene.source_view_id")
    scene.register_buffer(
        "source_view_id",
        (
            source.to(device=device, dtype=torch.long)
            if isinstance(source, torch.Tensor)
            else torch.full(
                (scene.n_gaussians,), -1, device=device, dtype=torch.long
            )
        ),
    )
    source_confidence = model.get("scene.source_confidence")
    scene.register_buffer(
        "source_confidence",
        (
            source_confidence.to(device)
            if isinstance(source_confidence, torch.Tensor)
            else torch.ones(scene.n_gaussians, 1, device=device)
        ),
    )
    source_camera_centers = model.get("scene.source_camera_centers")
    scene.register_buffer(
        "source_camera_centers",
        (
            source_camera_centers.to(device)
            if isinstance(source_camera_centers, torch.Tensor)
            else _camera_centers_by_view(dataset, device)
        ),
    )
    scene.view_locked_opacity = (
        bool(cfg["view_locked_opacity"])
        if "view_locked_opacity" in cfg
        else int(scene.source_view_id.min()) >= 0
    )
    scene.view_lock_neighbors = int(cfg.get("view_lock_neighbors", 0))
    scene.view_lock_mode = str(cfg.get("view_lock_mode", "hard"))
    scene.view_lock_temperature = float(cfg.get("view_lock_temperature", 0.45))
    scene.view_lock_min_weight = float(cfg.get("view_lock_min_weight", 0.02))
    pipeline = WorldGaussianBRDFPipeline(
        scene=scene,
        n_frames=len(dataset),
        lighting_mode=str(cfg.get("lighting_mode", "sh")),
        use_refiner=bool(cfg.get("use_refiner", True)),
        refiner_base=int(cfg.get("refiner_base", 96)),
        chunk_size=int(cfg.get("chunk_size", 32)),
        raster_backend=str(cfg.get("raster_backend", "auto")),
        shared_lighting=bool(cfg.get("shared_lighting", False)),
        background_color=cfg.get("background_color"),
        linear_hdr=bool(cfg.get("linear_hdr", False)),
        shadow_mode=str(cfg.get("shadow_mode", "none")),
        shadow_strength=float(cfg.get("shadow_strength", 1.0)),
        light_ids=[frame.light_id for frame in dataset.frames],
        use_implicit_material=bool(cfg.get("use_implicit_material", False)),
        implicit_material_hidden=int(cfg.get("implicit_material_hidden", 64)),
        implicit_material_layers=int(cfg.get("implicit_material_layers", 3)),
        implicit_material_residual_limit=float(
            cfg.get("implicit_material_residual_limit", 0.10)
        ),
        use_neural_brdf=bool(cfg.get("use_neural_brdf", False)),
        neural_brdf_material_latent=int(cfg.get("neural_brdf_material_latent", 32)),
        neural_brdf_light_latent=int(cfg.get("neural_brdf_light_latent", 16)),
        neural_brdf_hidden=int(cfg.get("neural_brdf_hidden", 96)),
        neural_brdf_strength=float(cfg.get("neural_brdf_strength", 1.0)),
        neural_brdf_mode=str(cfg.get("neural_brdf_mode", "replacement")),
        neural_brdf_residual_limit=float(
            cfg.get("neural_brdf_residual_limit", 0.35)
        ),
        neural_brdf_highlight_gate=bool(
            cfg.get("neural_brdf_highlight_gate", False)
        ),
        neural_brdf_boundary_fallback=bool(
            cfg.get("neural_brdf_boundary_fallback", False)
        ),
    ).to(device)
    has_neural_brdf_state = any(
        key.startswith("neural_brdf.") for key in model
    )
    pretrained_brdf = cfg.get("neural_brdf_pretrained")
    if pipeline.neural_brdf is not None and pretrained_brdf and not has_neural_brdf_state:
        pretrained_state = torch.load(
            Path(pretrained_brdf), map_location=device, weights_only=False
        )
        pipeline.neural_brdf.load_state_dict(pretrained_state["model"])
        print(f"loaded synthetic neural BRDF: {pretrained_brdf}", flush=True)
    missing, unexpected = pipeline.load_state_dict(model, strict=False)
    ignored_missing = {"frame_lighting.frame_to_light"}
    real_missing = [key for key in missing if key not in ignored_missing]
    print(
        f"loaded warm state (gaussians={scene.n_gaussians}, "
        f"missing={len(real_missing)}, unexpected={len(unexpected)}, "
        f"view_lock={scene.view_locked_opacity})",
        flush=True,
    )
    return pipeline


def _checkpoint_payload(
    pipeline: WorldGaussianBRDFPipeline,
    cfg: dict,
    train_ids: list[int],
    holdout_ids: list[int],
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    history: list[dict],
) -> dict:
    return {
        "model": pipeline.state_dict(),
        "config": cfg,
        "points": pipeline.scene.means.detach().cpu(),
        "colors": pipeline.scene.texture.detach().cpu(),
        "train_ids": train_ids,
        "holdout_ids": holdout_ids,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "history": history,
        "execution_audit": {
            "train_count": len(train_ids),
            "holdout_count": len(holdout_ids),
            "visual_hull_init": bool(cfg.get("visual_hull_init", False)),
            "image_photometric_stereo_init": bool(
                cfg.get("image_photometric_stereo_init", False)
            ),
            "view_locked_opacity": bool(
                getattr(pipeline.scene, "view_locked_opacity", False)
            ),
            "view_lock_neighbors": int(
                getattr(pipeline.scene, "view_lock_neighbors", 0)
            ),
            "normal_mode": pipeline.scene.normal_mode,
            "depth_prior_weight": float(
                (cfg.get("loss") or {}).get("depth_prior", 0.0)
            ),
            "densify_every": int(cfg.get("densify_every", 0)),
        },
    }


def train(cfg: dict) -> Path:
    seed = int(cfg.get("seed", 2026))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    requested_device = cfg.get("device")
    device = torch.device(
        requested_device
        if requested_device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = load_dataset(cfg)
    train_ids, holdout_ids = split_scene_indices(dataset, cfg)
    image_normal_prior_weight = float(cfg.get("image_normal_prior_weight", 0.0))
    if image_normal_prior_weight > 0:
        normal_prior_source = str(cfg.get("image_normal_prior_source", "photometric"))
        if normal_prior_source == "monocular":
            attached = attach_monocular_normal_priors(
                dataset,
                train_ids,
                device=device,
                input_size=int(cfg.get("monocular_depth_input_size", 384)),
            )
            print(f"attached monocular normal prior to {attached} train frames")
        elif normal_prior_source == "omnidata":
            attached = attach_omnidata_normal_priors(
                dataset,
                train_ids,
                device=device,
                input_size=int(cfg.get("monocular_depth_input_size", 384)),
                edge_erode=int(cfg.get("omnidata_edge_erode", 0)),
            )
            print(f"attached omnidata normal prior to {attached} train frames")
        else:
            attach_image_photometric_normal_priors(
                dataset,
                train_ids,
                device=device,
                max_views=int(cfg.get("photometric_stereo_max_views", 20)),
                face_camera=bool(cfg.get("photometric_stereo_face_camera", False)),
                lower_quantile=float(cfg.get("image_ps_lower_quantile", 0.10)),
                upper_quantile=float(cfg.get("image_ps_upper_quantile", 0.55)),
                irls_iterations=int(cfg.get("image_ps_irls_iterations", 0)),
                huber_delta=float(cfg.get("image_ps_huber_delta", 1.5)),
                shadow_threshold=float(cfg.get("image_ps_shadow_threshold", 0.01)),
                saturation_threshold=float(
                    cfg.get("image_ps_saturation_threshold", 0.98)
                ),
            )
    output_dir = Path(cfg.get("output_dir", "outputs/world_scene"))
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    init_path = Path(cfg["init_from"]) if cfg.get("init_from") else None
    resume_state = (
        torch.load(latest_path, map_location=device, weights_only=False)
        if bool(cfg.get("auto_resume", False)) and latest_path.exists()
        else None
    )
    warm_state = (
        torch.load(init_path, map_location=device, weights_only=False)
        if resume_state is None and init_path is not None and init_path.exists()
        else None
    )
    if init_path is not None and warm_state is None and resume_state is None:
        raise FileNotFoundError(f"init_from checkpoint not found: {init_path}")
    state = resume_state or warm_state
    pipeline = (
        _pipeline_from_state(dataset, cfg, state, device)
        if state is not None
        else build_pipeline(dataset, cfg, device, train_ids)
    )

    steps = int(cfg.get("steps", 3000))
    refine_start = int(cfg.get("refine_start", steps // 2))
    weights = WorldLossWeights(**cfg.get("loss", {}))
    if weights.depth_prior > 0:
        depth_prior_source = str(cfg.get("depth_prior_source", "visual_hull"))
        if depth_prior_source == "monocular":
            attached = attach_monocular_depth_priors(
                dataset,
                train_ids,
                device=device,
                input_size=int(cfg.get("monocular_depth_input_size", 256)),
            )
            print(f"attached monocular depth prior to {attached} train frames")
        else:
            attached = attach_visual_hull_depth_priors(
                dataset,
                train_ids,
                resolution=int(
                    cfg.get(
                        "depth_prior_hull_resolution",
                        cfg.get("visual_hull_resolution", 96),
                    )
                ),
                device=device,
            )
            print(f"attached visual-hull depth prior to {attached} train frames")

    freeze_geometry = bool(cfg.get("freeze_geometry", False))
    freeze_except_opacity = bool(
        cfg.get("freeze_geometry_except_opacity", False)
    )
    freeze_normals = bool(cfg.get("freeze_normals", False))
    freeze_materials = bool(cfg.get("freeze_materials", False))
    optimize_normals = bool(cfg.get("optimize_normals", True))
    optimize_normals_after_warmup = bool(
        cfg.get("optimize_normals_after_warmup", False)
    )
    optimize_sensors_only = bool(cfg.get("optimize_sensors_only", False))
    geometry_warmup_steps = int(cfg.get("geometry_warmup_steps", 0))
    freeze_after_warmup = bool(
        cfg.get("freeze_geometry_after_warmup", geometry_warmup_steps > 0)
    )
    normal_depth_start = int(cfg.get("normal_depth_consistency_start", 0))
    depth_from_normal_start = int(cfg.get("depth_from_normal_start", 0))
    joint_refine_start = int(cfg.get("joint_refine_start", 0))
    geometry_refine_start = int(cfg.get("geometry_refine_start", 0))

    normal_lr = cfg.get("normal_lr")
    scene_parameters = [
        parameter
        for parameter in pipeline.scene.parameters()
        if normal_lr is None or parameter is not pipeline.scene.normal_raw
    ]
    lighting_parameters: list[torch.nn.Parameter] = []
    trainable_lighting = cfg.get("trainable_lighting_parameters")
    if trainable_lighting is not None:
        allowed_lighting = {str(name) for name in trainable_lighting}
        for name, parameter in pipeline.frame_lighting.named_parameters():
            allow = name in allowed_lighting
            parameter.requires_grad_(allow)
            if allow:
                lighting_parameters.append(parameter)
        if pipeline.neural_light is not None:
            pipeline.neural_light.requires_grad_(False)
    elif optimize_sensors_only:
        for name, parameter in pipeline.frame_lighting.named_parameters():
            allow = name in {"log_exposure", "log_white_balance"}
            parameter.requires_grad_(allow)
            if allow:
                lighting_parameters.append(parameter)
        if pipeline.neural_light is not None:
            pipeline.neural_light.requires_grad_(False)
    elif not bool(cfg.get("freeze_lighting", False)):
        lighting_parameters = list(pipeline.frame_lighting.parameters())
    else:
        pipeline.frame_lighting.requires_grad_(False)
        if pipeline.neural_light is not None:
            pipeline.neural_light.requires_grad_(False)

    parameter_groups = [
        {"params": scene_parameters, "lr": float(cfg.get("scene_lr", 2e-3))}
    ]
    if normal_lr is not None:
        parameter_groups.append(
            {
                "params": [pipeline.scene.normal_raw],
                "lr": float(normal_lr),
            }
        )
    if lighting_parameters:
        parameter_groups.append(
            {
                "params": lighting_parameters,
                "lr": float(cfg.get("light_lr", 1e-3)),
            }
        )
    if (
        pipeline.neural_light is not None
        and not bool(cfg.get("freeze_lighting", False))
        and not optimize_sensors_only
    ):
        parameter_groups.append(
            {
                "params": pipeline.neural_light.parameters(),
                "lr": float(cfg.get("neilf_lr", 1e-4)),
            }
        )
    if pipeline.refiner is not None:
        parameter_groups.append(
            {
                "params": pipeline.refiner.parameters(),
                "lr": float(cfg.get("refiner_lr", 1e-4)),
            }
        )
    if pipeline.implicit_material is not None:
        parameter_groups.append(
            {
                "params": pipeline.implicit_material.parameters(),
                "lr": float(cfg.get("implicit_material_lr", 5e-4)),
            }
        )
    if pipeline.neural_brdf is not None:
        if bool(cfg.get("freeze_neural_brdf_material_encoder", False)):
            pipeline.neural_brdf.material_encoder.requires_grad_(False)
        if bool(cfg.get("freeze_neural_brdf_light_encoder", False)):
            pipeline.neural_brdf.light_encoder.requires_grad_(False)
        if bool(cfg.get("freeze_neural_brdf_decoder", False)):
            pipeline.neural_brdf.decoder.requires_grad_(False)
        neural_brdf_parameters = [
            parameter
            for parameter in pipeline.neural_brdf.parameters()
            if parameter.requires_grad
        ]
        if neural_brdf_parameters:
            parameter_groups.append(
                {
                    "params": neural_brdf_parameters,
                    "lr": float(cfg.get("neural_brdf_lr", 1e-4)),
                }
            )
    optimizer = torch.optim.Adam(parameter_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps, 1)
    )
    start_step = 0
    history: list[dict] = []
    if resume_state is not None:
        try:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
        except (KeyError, ValueError):
            print("resume optimizer state incompatible; continuing with fresh optimizer")
        start_step = int(resume_state.get("step", -1)) + 1
        history = list(resume_state.get("history", []))

    def configure_scene(step: int) -> str:
        geometry_stage = geometry_warmup_steps > 0 and step < geometry_warmup_steps
        allowed: set[int]
        if freeze_except_opacity:
            allowed = {
                *(id(p) for p in pipeline.scene.material_parameters()),
                id(pipeline.scene.opacity_logits),
            }
            stage = "material+opacity"
        elif freeze_geometry:
            allowed = {id(p) for p in pipeline.scene.material_parameters()}
            if optimize_normals and pipeline.scene.normal_mode == "learned":
                allowed.add(id(pipeline.scene.normal_raw))
            stage = "material"
        elif geometry_stage:
            allowed = {id(p) for p in pipeline.scene.geometry_parameters()}
            stage = "geometry"
        else:
            allowed = {id(p) for p in pipeline.scene.parameters()}
            if geometry_refine_start > 0 and step >= geometry_refine_start:
                # Normal-anchored geometry refinement: the shading normals are
                # frozen (anchored by the monocular normal prior), while
                # geometry and materials refine so the surface conforms to the
                # accurate normals through the normal->depth coupling.  Unlike
                # joint refinement, the frozen normals cannot be dragged down
                # by a drifting surface, so geometry must conform to them.
                allowed.discard(id(pipeline.scene.normal_raw))
                stage = "geometry_refine"
            elif joint_refine_start > 0 and step >= joint_refine_start:
                # Joint refinement: reopen geometry so the bidirectional
                # depth/normal coupling can reshape the surface.  The cosine
                # schedule has already decayed the learning rate by this
                # point, keeping the refinement gentle.
                stage = "joint"
            elif freeze_after_warmup and geometry_warmup_steps > 0:
                allowed = {id(p) for p in pipeline.scene.material_parameters()}
                if (
                    optimize_normals_after_warmup
                    and optimize_normals
                    and pipeline.scene.normal_mode == "learned"
                ):
                    allowed.add(id(pipeline.scene.normal_raw))
                stage = "material"
            else:
                stage = "joint"
        if (
            pipeline.scene.normal_mode == "covariance"
            or freeze_normals
            or not optimize_normals
        ):
            allowed.discard(id(pipeline.scene.normal_raw))
        if freeze_materials:
            allowed.difference_update(
                id(parameter) for parameter in pipeline.scene.material_parameters()
            )
            if stage == "material":
                stage = "implicit-material"
        for parameter in pipeline.scene.parameters():
            parameter.requires_grad_(id(parameter) in allowed)
        return stage

    current_stage = configure_scene(start_step)
    train_views = sorted({dataset[index].view_id for index in train_ids})
    holdout_views = sorted({dataset[index].view_id for index in holdout_ids})
    print(
        f"train frames={len(train_ids)} views={train_views}; "
        f"holdout frames={len(holdout_ids)} views={holdout_views}; "
        f"stage={current_stage}",
        flush=True,
    )
    views_per_step = min(int(cfg.get("views_per_step", 1)), len(train_ids))
    checkpoint_every = int(cfg.get("checkpoint_every", 0))
    densify_every = int(cfg.get("densify_every", 0))
    densify_start = int(cfg.get("densify_start", 0))
    densify_end = int(cfg.get("densify_end", steps))
    pipeline.train()
    for step in range(start_step, steps):
        stage = configure_scene(step)
        if stage != current_stage:
            current_stage = stage
            print(f"optimization stage={stage} at step {step}", flush=True)
        frame_ids = sample_frame_indices(
            dataset,
            train_ids,
            views_per_step,
            str(cfg.get("view_sampling", "random")),
        )
        optimizer.zero_grad(set_to_none=True)
        logs: dict[str, float] = {}
        for frame_id in frame_ids:
            frame = dataset[frame_id]
            camera = frame.camera.to(device)
            image = frame.image.to(device)
            has_diffuse = frame.diffuse_gt is not None
            use_refiner = has_diffuse and step >= refine_start
            output = pipeline(camera, frame_id, image=image, refine=use_refiner)
            diffuse_gt = (
                frame.diffuse_gt.to(device)
                if has_diffuse and weights.diffuse_gt > 0
                else None
            )
            specular_gt = (
                image - diffuse_gt
                if diffuse_gt is not None and weights.specular_gt > 0
                else None
            )
            loss, frame_logs = world_total_loss(
                output,
                image,
                pipeline.scene,
                diffuse_gt=diffuse_gt,
                specular_gt=specular_gt,
                albedo_gt=(
                    frame.albedo_gt.to(device)
                    if frame.albedo_gt is not None and weights.albedo_gt > 0
                    else None
                ),
                normal_gt=(
                    frame.normal_gt.to(device)
                    if frame.normal_gt is not None and weights.normal_gt > 0
                    else None
                ),
                depth_gt=(
                    frame.depth_gt.to(device)
                    if frame.depth_gt is not None and weights.depth_gt > 0
                    else None
                ),
                depth_prior=(
                    frame.depth_prior.to(device)
                    if frame.depth_prior is not None and weights.depth_prior > 0
                    else None
                ),
                valid_mask=(
                    frame.mask_gt.to(device) if frame.mask_gt is not None else None
                ),
                weights=weights,
                camera=(
                    camera
                    if (
                        weights.normal_depth_consistency > 0
                        or weights.depth_from_normal > 0
                    )
                    else None
                ),
                step=step,
                normal_depth_start=normal_depth_start,
                depth_from_normal_start=depth_from_normal_start,
            )
            if (
                image_normal_prior_weight > 0
                and frame.normal_prior_image is not None
            ):
                predicted_normal = output["render"].normal
                target_normal = frame.normal_prior_image.to(predicted_normal)
                confidence = (
                    frame.normal_prior_confidence.to(predicted_normal).clamp(0, 1)
                    if frame.normal_prior_confidence is not None
                    else torch.ones_like(target_normal[:1])
                )
                if frame.mask_gt is not None:
                    confidence = confidence * (
                        frame.mask_gt.to(predicted_normal) > 0.5
                    ).to(confidence)
                confidence = confidence * output["render"].alpha.detach().clamp(0, 1)
                if bool((confidence > 0).any()):
                    predicted_unit = F.normalize(
                        predicted_normal, dim=0, eps=1e-6
                    )
                    target_unit = F.normalize(target_normal, dim=0, eps=1e-6)
                    vector_error = (predicted_unit - target_unit).abs().mean(
                        dim=0, keepdim=True
                    )
                    normal_image_loss = (
                        vector_error * confidence
                    ).sum() / confidence.sum().clamp_min(1e-6)
                    loss = loss + image_normal_prior_weight * normal_image_loss
                    frame_logs["normal_image_prior"] = float(
                        normal_image_loss.detach()
                    )
            (loss / views_per_step).backward()
            for key, value in frame_logs.items():
                logs[key] = logs.get(key, 0.0) + value / views_per_step
        residual_reg = float(cfg.get("neural_brdf_residual_reg", 0.0))
        if (
            residual_reg > 0
            and pipeline.neural_brdf is not None
            and pipeline.neural_brdf.mode == "log_residual"
        ):
            final_layer = pipeline.neural_brdf.decoder[-1]
            residual_penalty = final_layer.weight.square().mean()
            if final_layer.bias is not None:
                residual_penalty = residual_penalty + final_layer.bias.square().mean()
            if residual_penalty.requires_grad:
                (residual_reg * residual_penalty).backward()
            logs["neural_residual_reg"] = float(residual_penalty.detach())
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 5.0)
        optimizer.step()
        if (
            densify_every > 0
            and pipeline.scene.means.requires_grad
            and densify_start <= step < densify_end
            and (step + 1) % densify_every == 0
        ):
            slots = pipeline.scene.reallocate_gaussians(
                fraction=float(cfg.get("densify_fraction", 0.01)),
                opacity_threshold=float(
                    cfg.get("prune_opacity_threshold", 0.02)
                ),
            )
            for parameter in pipeline.scene.parameters():
                optimizer_state = optimizer.state.get(parameter, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = optimizer_state.get(key)
                    if (
                        isinstance(value, torch.Tensor)
                        and value.ndim > 0
                        and value.shape[0] == pipeline.scene.n_gaussians
                    ):
                        value[slots].zero_()
            logs["reallocated"] = float(len(slots))
        scheduler.step()
        if step % int(cfg.get("log_every", 25)) == 0 or step == steps - 1:
            print(
                f"[world {step}/{steps}] "
                + " ".join(f"{key}={value:.4f}" for key, value in logs.items()),
                flush=True,
            )
            history.append({"step": step, **logs})
        if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
            torch.save(
                _checkpoint_payload(
                    pipeline,
                    cfg,
                    train_ids,
                    holdout_ids,
                    optimizer,
                    scheduler,
                    step,
                    history,
                ),
                latest_path,
            )

    checkpoint = output_dir / "last.pt"
    torch.save(
        _checkpoint_payload(
            pipeline,
            cfg,
            train_ids,
            holdout_ids,
            optimizer,
            scheduler,
            max(steps - 1, -1),
            history,
        ),
        checkpoint,
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"saved {checkpoint}", flush=True)
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="suffix appended to output_dir (e.g. for seed sweeps)",
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.tag:
        cfg["output_dir"] = str(cfg.get("output_dir", "outputs/world_scene")) + "_" + args.tag
    train(cfg)


if __name__ == "__main__":
    main()

