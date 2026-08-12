"""Unified scene-level inverse-rendering, NVS, relighting, and SHR evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..metrics_ir import (
    aggregate_metrics,
    decomposition_metrics,
    masked_psnr,
    masked_ssim,
    multiview_albedo_consistency,
    normal_angular_metrics,
    optional_lpips,
    orb_image_similarity,
    orb_normal_cosine_distance,
    orb_scale_invariant_albedo_psnr,
    orb_scene_depth_mse,
    orb_shape_chamfer,
    reconstruction_residual,
    scale_aligned_albedo_metrics,
    world_normal_to_orb_camera,
)
from ..viz import save_image
from .geometry_init import _image_photometric_normal_map
from .lighting import WorldLight
from .pipeline import WorldGaussianBRDFPipeline
from .scene import GaussianBRDFField
from .train import load_dataset


def _adapt_checkpoint_state(
    pipeline: WorldGaussianBRDFPipeline, model_state: dict
) -> dict:
    """Allow loading when novel-light frames expand lighting buffers."""
    current = pipeline.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    for key, value in model_state.items():
        if key not in current:
            continue
        if key == "frame_lighting.frame_to_light":
            # This mapping belongs to the currently loaded dataset.  Its shape
            # may match an old checkpoint while its light-id semantics differ.
            continue
        target = current[key]
        if tuple(target.shape) == tuple(value.shape):
            adapted[key] = value
            continue
        if value.ndim >= 1 and target.ndim == value.ndim and value.shape[0] != target.shape[0]:
            merged = target.detach().clone()
            count = min(value.shape[0], target.shape[0])
            merged[:count] = value[:count].to(merged.dtype)
            adapted[key] = merged
    return adapted


def _initialize_novel_lighting(
    pipeline: WorldGaussianBRDFPipeline, dataset, cfg: dict
) -> None:
    """Fill novel-scene light slots from envmap metadata when available."""
    lighting = pipeline.frame_lighting
    env_scale = float(cfg.get("novel_env_scale", 1.0))
    direct_scale = float(cfg.get("novel_direct_scale", 1.0))
    with torch.no_grad():
        for index, frame in enumerate(dataset.frames):
            metadata = getattr(frame, "metadata", None) or {}
            if metadata.get("task") != "novel_scene_relighting":
                continue
            light_index = lighting.light_index(index)
            if frame.env_sh is not None:
                lighting.env_sh[light_index].copy_(
                    frame.env_sh.to(lighting.env_sh.device) * env_scale
                )
                lighting.point_light_logits[light_index].fill_(-8.0)
            if frame.light_direction is not None:
                lighting.direction_raw[light_index].copy_(
                    frame.light_direction.to(lighting.direction_raw.device)
                )
            if frame.light_color is not None:
                color = frame.light_color.to(lighting.color_logits.device).clamp(
                    1e-4, 1 - 1e-4
                )
                lighting.color_logits[light_index].copy_(torch.logit(color))
            if frame.light_intensity is not None:
                lighting.log_intensity[light_index].fill_(
                    max(float(frame.light_intensity) * direct_scale, 1e-8)
                )
                lighting.log_intensity[light_index].log_()
            exposure = float(getattr(frame, "exposure", 1.0) or 1.0)
            lighting.log_exposure[index].fill_(max(exposure, 1e-6))
            lighting.log_exposure[index].log_()
            if frame.white_balance is not None:
                white_balance = frame.white_balance.to(
                    lighting.log_white_balance.device
                ).clamp_min(1e-6)
                lighting.log_white_balance[index].copy_(white_balance.log())


def _environment_override(
    pipeline: WorldGaussianBRDFPipeline,
    frame,
    frame_id: int,
    sample_count: int,
) -> WorldLight | None:
    """Build deterministic lat-long quadrature from the supplied HDR envmap."""
    if frame.envmap_gt is None or sample_count <= 0:
        return None
    device = pipeline.scene.means.device
    dtype = pipeline.scene.means.dtype
    height = max(2, int((sample_count / 2) ** 0.5))
    width = 2 * height
    envmap = F.interpolate(
        frame.envmap_gt.to(device=device, dtype=dtype).unsqueeze(0),
        size=(height, width),
        mode="area",
    )[0]
    phi = (torch.arange(height, device=device, dtype=dtype) + 0.5)
    phi = phi * torch.pi / height
    theta = (torch.arange(width, device=device, dtype=dtype) + 0.5)
    theta = theta * (2 * torch.pi / width) - 0.5 * torch.pi
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")
    sin_phi = torch.sin(phi)
    directions = torch.stack(
        (
            -torch.cos(theta) * sin_phi,
            torch.cos(phi),
            -torch.sin(theta) * sin_phi,
        ),
        dim=-1,
    ).reshape(-1, 3)
    solid_angle = (
        sin_phi * (torch.pi / height) * (2 * torch.pi / width)
    ).reshape(-1)
    base = pipeline.frame_lighting(frame_id)
    return WorldLight(
        env_sh=torch.zeros_like(base.env_sh),
        direction=base.direction,
        intensity=torch.zeros_like(base.intensity),
        color=base.color,
        position=base.position,
        point_weight=torch.zeros_like(base.point_weight),
        exposure=base.exposure,
        white_balance=base.white_balance,
        env_directions=directions,
        env_radiance=envmap.permute(1, 2, 0).reshape(-1, 3),
        env_solid_angle=solid_angle,
    )


def load_world_checkpoint(
    checkpoint: str | Path,
    device: torch.device,
    data_root_override: str | Path | None = None,
) -> tuple[dict, object, WorldGaussianBRDFPipeline]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = dict(state["config"])
    if data_root_override is not None:
        cfg["data_root"] = str(data_root_override)
    dataset = load_dataset(cfg)
    colors = state.get("colors")
    scene = GaussianBRDFField.from_point_cloud(
        state["points"].to(device),
        colors.to(device) if colors is not None else None,
        normal_mode=str(cfg.get("normal_mode", "learned")),
    )
    n_points = int(state["points"].shape[0])
    # Register buffers before load_state_dict so view-locked PS ids are restored.
    source = state["model"].get("scene.source_view_id")
    scene.register_buffer(
        "source_view_id",
        source.to(device).long()
        if source is not None
        else torch.full((n_points,), -1, device=device, dtype=torch.long),
    )
    prior = state["model"].get("scene.normal_prior")
    scene.register_buffer(
        "normal_prior",
        prior.to(device)
        if prior is not None
        else torch.zeros(n_points, 3, device=device),
    )
    scene.view_locked_opacity = bool(cfg.get("view_locked_opacity", False)) or (
        source is not None and int(source.min()) >= 0
    )
    pipeline = WorldGaussianBRDFPipeline(
        scene,
        len(dataset),
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
    pipeline.load_state_dict(
        _adapt_checkpoint_state(pipeline, state["model"]), strict=False
    )
    pipeline.scene.view_locked_opacity = (
        bool(cfg["view_locked_opacity"])
        if "view_locked_opacity" in cfg
        else (
            hasattr(pipeline.scene, "source_view_id")
            and int(pipeline.scene.source_view_id.min()) >= 0
        )
    )
    pipeline.scene.view_lock_neighbors = int(cfg.get("view_lock_neighbors", 0))
    _initialize_novel_lighting(pipeline, dataset, cfg)
    pipeline.eval()
    return state, dataset, pipeline


def _depth_metrics(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> dict[str, float | None]:
    valid = target > 0
    if mask is not None:
        valid &= mask.to(target) > 0.5
    if not valid.any():
        return {"depth_abs_rel": None, "depth_si_log": None}
    p = pred[valid].clamp_min(1e-6)
    t = target[valid].clamp_min(1e-6)
    log_error = p.log() - t.log()
    return {
        "depth_abs_rel": float(((p - t).abs() / t).mean()),
        "depth_si_log": float((log_error - log_error.mean()).square().mean().sqrt()),
    }


def _save_tensor(path: Path, tensor: torch.Tensor, preview: bool = True) -> None:
    value = tensor.detach().float().cpu()
    np.save(path.with_suffix(".npy"), value.numpy())
    if preview:
        display = value
        if display.shape[0] == 1:
            display = display.expand(3, -1, -1)
        if display.shape[0] == 3:
            save_image(display.clamp(0, 1), path.with_suffix(".png"))


@torch.inference_mode()
def evaluate_inverse_rendering(
    checkpoint: str | Path,
    output_dir: str | Path | None = None,
    split: str = "holdout",
    compute_lpips: bool = False,
    shape_mesh: str | Path | None = None,
    chamfer_samples: int = 30_000,
    view_lock_neighbors: int | None = None,
    source_ps_normals: bool | None = None,
) -> dict:
    """Evaluate all available GT channels without inventing missing references."""
    if split not in {"all", "train", "holdout", "test", "novel"}:
        raise ValueError("split must be all, train, holdout, test, or novel")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, dataset, pipeline = load_world_checkpoint(checkpoint, device)
    if view_lock_neighbors is not None:
        if view_lock_neighbors < 0:
            pipeline.scene.view_locked_opacity = False
        else:
            pipeline.scene.view_locked_opacity = True
            pipeline.scene.view_lock_neighbors = int(view_lock_neighbors)
    train_ids = list(state.get("train_ids", range(len(dataset))))
    holdout_ids = list(state.get("holdout_ids", []))
    selected = {
        "all": list(range(len(dataset))),
        "train": train_ids,
        "holdout": holdout_ids,
        "test": [
            index
            for index, frame in enumerate(dataset.frames)
            if frame.metadata.get("split") == "test"
        ],
        "novel": [
            index
            for index, frame in enumerate(dataset.frames)
            if frame.metadata.get("split") == "novel"
        ],
    }[split]
    if not selected:
        raise ValueError(f"checkpoint contains no {split} frames")
    output = Path(output_dir or Path(checkpoint).parent / f"ir_eval_{split}")
    output.mkdir(parents=True, exist_ok=True)
    train_light_ids = {
        dataset[index].light_id
        for index in train_ids
        if dataset[index].light_id is not None
    }
    train_source_light_ids = {
        dataset[index].metadata.get("source_light_id")
        for index in train_ids
        if dataset[index].metadata is not None
        and dataset[index].metadata.get("source_light_id") is not None
    }

    rows: list[dict[str, float | int | None]] = []
    albedo_samples: list[torch.Tensor] = []
    albedo_valid: list[torch.Tensor] = []
    orb_depth_predictions: list[torch.Tensor] = []
    orb_depth_targets: list[torch.Tensor] = []
    orb_depth_masks: list[torch.Tensor] = []
    is_stanford_orb = dataset.metadata.get("adapter") == "stanford_orb"
    use_source_ps_normals = (
        bool(state["config"].get("evaluate_source_ps_normals", False))
        if source_ps_normals is None
        else bool(source_ps_normals)
    ) and dataset.metadata.get("adapter") == "diligent_mv"
    source_ps_normal_maps: dict[int, torch.Tensor] = {}
    if use_source_ps_normals:
        allowed = set(train_ids)
        groups = [
            [index for index in values if index in allowed]
            for values in dataset.group_by_view().values()
        ]
        for values in groups:
            if len(values) < 6:
                continue
            normal_map, _, camera = _image_photometric_normal_map(
                dataset,
                values,
                device,
                face_camera=bool(
                    state["config"].get("photometric_stereo_face_camera", False)
                ),
                lower_quantile=float(
                    state["config"].get("image_ps_lower_quantile", 0.10)
                ),
                upper_quantile=float(
                    state["config"].get("image_ps_upper_quantile", 0.55)
                ),
            )
            if camera.view_id is not None:
                source_ps_normal_maps[int(camera.view_id)] = normal_map
    full_env_samples = int(state["config"].get("novel_full_env_samples", 0))
    for frame_id in selected:
        frame = dataset[frame_id]
        image = frame.image.to(device)
        light_override = (
            _environment_override(
                pipeline,
                frame,
                frame_id,
                full_env_samples,
            )
            if (frame.metadata or {}).get("task") == "novel_scene_relighting"
            else None
        )
        result = pipeline(
            frame.camera.to(device),
            frame_id,
            image=image,
            refine=True,
            light_override=light_override,
        )
        render = result["render"]
        mask = (
            frame.mask_gt.to(device)
            if frame.mask_gt is not None
            else render.alpha > 0.01
        )
        data_range = max(1.0, float(torch.quantile(image, 0.99)))
        row: dict[str, float | int | None] = {
            "frame": frame_id,
            "view_id": frame.view_id,
            "light_id": frame.light_id,
            "full_psnr": masked_psnr(render.full, image, mask, data_range),
            "full_ssim": masked_ssim(render.full, image, mask, data_range),
        }
        if is_stanford_orb:
            official = orb_image_similarity(
                render.full,
                image,
                mask,
                scale_invariant=(
                    frame.metadata.get("task") == "novel_scene_relighting"
                ),
                hdr_available=bool(frame.metadata.get("hdr_available", False)),
                compute_lpips=compute_lpips,
            )
            row.update({f"orb_{key}": value for key, value in official.items()})
        source_light_id = (
            frame.metadata.get("source_light_id") if frame.metadata else None
        )
        if frame_id in holdout_ids and (
            frame.light_id in train_light_ids
            or source_light_id in train_source_light_ids
        ):
            # The light parameters are shared by light_id and therefore learned
            # only from training views; this is task-matched novel-view relighting.
            row["relighting_psnr"] = row["full_psnr"]
            row["relighting_ssim"] = row["full_ssim"]
        if compute_lpips:
            row["full_lpips"] = optional_lpips(
                (render.full / data_range).clamp(0, 1),
                (image / data_range).clamp(0, 1),
                mask,
            )
        residual = reconstruction_residual(image, render.diffuse, render.specular, mask)
        row.update({f"reconstruction_{key}": value for key, value in residual.items()})
        if frame.diffuse_gt is not None:
            target_diffuse = frame.diffuse_gt.to(device)
            row["lobe_off_psnr"] = masked_psnr(
                render.diffuse, target_diffuse, mask, data_range
            )
            row["shr_refined_psnr"] = masked_psnr(
                result["pred"], target_diffuse, mask, data_range
            )
            target_specular = (
                frame.specular_gt.to(device)
                if frame.specular_gt is not None
                else image - target_diffuse
            )
            row.update(
                decomposition_metrics(
                    render.diffuse,
                    target_diffuse,
                    render.specular,
                    target_specular,
                    mask,
                    data_range,
                )
            )
        if frame.albedo_gt is not None:
            target_albedo = frame.albedo_gt.to(device)
            albedo_prefix = (
                "pseudo_albedo"
                if frame.metadata
                and bool(frame.metadata.get("albedo_is_pseudo", False))
                else "albedo"
            )
            aligned = scale_aligned_albedo_metrics(render.albedo, target_albedo, mask)
            row.update(
                {
                    f"{albedo_prefix}_scale_{key}": value
                    for key, value in aligned.items()
                    if isinstance(value, (float, int)) or value is None
                }
            )
            row[f"{albedo_prefix}_psnr"] = masked_psnr(
                render.albedo, target_albedo, mask
            )
            row[f"{albedo_prefix}_ssim"] = masked_ssim(
                render.albedo, target_albedo, mask
            )
            if is_stanford_orb:
                row["orb_material_psnr_ldr"] = orb_scale_invariant_albedo_psnr(
                    render.albedo, target_albedo, mask
                )
            if compute_lpips:
                row[f"{albedo_prefix}_lpips"] = optional_lpips(
                    render.albedo, target_albedo, mask
                )
        if frame.normal_gt is not None:
            normal_thresholds = (
                (5.0, 10.0, 20.0)
                if dataset.metadata.get("adapter") == "diligent_mv"
                else (11.25, 22.5, 30.0)
            )
            predicted_normal = source_ps_normal_maps.get(
                int(frame.view_id) if frame.view_id is not None else -1,
                render.normal,
            )
            normal = normal_angular_metrics(
                predicted_normal,
                frame.normal_gt.to(device),
                mask,
                thresholds=normal_thresholds,
            )
            row.update({f"normal_{key}": value for key, value in normal.items()})
            if is_stanford_orb:
                row["orb_normal_cosine_distance"] = orb_normal_cosine_distance(
                    render.normal, frame.normal_gt.to(device), mask
                )
        if frame.depth_gt is not None:
            row.update(_depth_metrics(render.depth, frame.depth_gt.to(device), mask))
            if is_stanford_orb:
                orb_depth_predictions.append(render.depth.detach().cpu().clone())
                orb_depth_targets.append(frame.depth_gt.detach().cpu().clone())
                orb_depth_masks.append(mask.detach().cpu().clone())

        # Sample the same persistent Gaussian centers in each view to quantify
        # rasterization-induced cross-view albedo variation.
        uv, _, valid = frame.camera.to(device).project(pipeline.scene.means)
        grid = uv.clone()
        grid[:, 0] = 2 * grid[:, 0] / max(frame.camera.width - 1, 1) - 1
        grid[:, 1] = 2 * grid[:, 1] / max(frame.camera.height - 1, 1) - 1
        sample = F.grid_sample(
            render.albedo.unsqueeze(0),
            grid.view(1, 1, -1, 2),
            align_corners=True,
        )[0, :, 0].T
        albedo_samples.append(sample)
        albedo_valid.append(valid)

        frame_dir = output / f"{frame_id:04d}"
        frame_dir.mkdir(exist_ok=True)
        for name, value in {
            "full": render.full,
            "diffuse_lobe_off": render.diffuse,
            "specular": render.specular,
            "albedo": render.albedo,
            "normal": (render.normal + 1) * 0.5,
            "depth": render.depth,
            "mask": render.mask,
            "shr_refined": result["pred"],
        }.items():
            _save_tensor(frame_dir / name, value)
        if is_stanford_orb:
            _save_tensor(
                frame_dir / "normal_camera",
                world_normal_to_orb_camera(render.normal, frame.camera.c2w.to(device)),
                preview=False,
            )
        rows.append(row)

    consistency = multiview_albedo_consistency(
        torch.stack(albedo_samples, dim=1),
        torch.stack(albedo_valid, dim=1),
        view_dim=1,
    )
    summary = aggregate_metrics(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"frame", "view_id", "light_id"}
            }
            for row in rows
        ]
    )
    summary.update(
        {f"cross_view_albedo_{key}": value for key, value in consistency.items()}
    )
    if is_stanford_orb and orb_depth_predictions:
        raw_depth = orb_scene_depth_mse(
            orb_depth_predictions, orb_depth_targets, orb_depth_masks
        )
        summary["orb_depth_mse_scene"] = raw_depth
        summary["orb_depth_mse_scene_x1e3"] = (
            None if raw_depth is None else float(raw_depth) * 1e3
        )
    if is_stanford_orb and shape_mesh is not None:
        target_mesh = dataset.metadata.get("mesh_path")
        summary["orb_bidir_chamfer"] = (
            orb_shape_chamfer(
                str(shape_mesh),
                str(target_mesh),
                num_samples=chamfer_samples,
                seed=0,
            )
            if target_mesh
            else None
        )
    report = {
        "protocol": {
            "method": state["config"].get("protocol_label", "Ours-Unspecified"),
            "split": split,
            "linear_hdr": bool(state["config"].get("linear_hdr", False)),
            "lpips_requested": compute_lpips,
            "shape_mesh": str(shape_mesh) if shape_mesh is not None else None,
            "stanford_orb": (
                "Official 5x5-eroded PSNR-H/PSNR-L and 3x3 SSIM; "
                "novel-light uses per-channel scale invariance. PSNR-H is null "
                "when the reference is not genuine HDR."
                if is_stanford_orb
                else None
            ),
            "nvs": split == "holdout",
            "lobe_off": "physical diffuse component",
            "relighting": "shared light_id evaluated on held-out views",
            "normal_source": (
                "train-light image photometric stereo"
                if use_source_ps_normals
                else "rendered Gaussian field"
            ),
            "view_lock_override": view_lock_neighbors,
            "metric_integrity": (
                "SHR/diffuse metrics are emitted only when diffuse_gt exists; "
                "pseudo albedo is always prefixed pseudo_albedo"
            ),
        },
        "train_ids": train_ids,
        "holdout_ids": holdout_ids,
        "summary": summary,
        "frames": rows,
    }

    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    safe_report = json_safe(report)
    (output / "metrics.json").write_text(
        json.dumps(safe_report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return safe_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--split",
        choices=("all", "train", "holdout", "test", "novel"),
        default="holdout",
    )
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--shape-mesh")
    parser.add_argument("--chamfer-samples", type=int, default=30_000)
    parser.add_argument(
        "--view-lock-neighbors",
        type=int,
        help="evaluation-only override; use -1 to disable view locking",
    )
    parser.add_argument(
        "--source-ps-normals",
        action="store_true",
        default=None,
        help="evaluate DiLiGenT normals from train-light PS maps",
    )
    args = parser.parse_args()
    result = evaluate_inverse_rendering(
        args.checkpoint,
        args.output,
        args.split,
        args.lpips,
        args.shape_mesh,
        args.chamfer_samples,
        args.view_lock_neighbors,
        args.source_ps_normals,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
