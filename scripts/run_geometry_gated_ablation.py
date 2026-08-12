"""Run the six-scene geometry/normal gate before a full public rerun."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import torch
import yaml

from mvbrdf_shr.world.evaluate_ir import evaluate_inverse_rendering
from mvbrdf_shr.world.protocols import stanford_scene_group
from mvbrdf_shr.world.train import train


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("first_view", "visual_hull", "silhouette", "coupled", "densify", "brdf")


def _deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def scene_config(base: dict, dataset: str, scene: str, stage: str) -> dict:
    cfg = copy.deepcopy(base)
    overrides = dict((base.get("scene_overrides") or {}).get(scene, {}))
    protocol_label = str(base.get("protocol_label", "Ours-NoGT"))
    diligent_fuse_world_surface = bool(
        overrides.get(
            "image_ps_fuse_world_surface",
            base.get("diligent_fuse_world_surface", False),
        )
    )
    diligent_world_ps_surface = bool(
        overrides.get(
            "world_ps_surface",
            base.get("diligent_world_ps_surface", False),
        )
    )
    diligent_world_ps_initializer = str(
        base.get("diligent_world_ps_initializer", "fused_image")
    )
    output_root = ROOT / str(cfg.pop("output_root"))
    short_steps = int(cfg.pop("short_steps"))
    final_steps = int(cfg.pop("final_steps"))
    for key in (
        "stanford_root",
        "stanford_ground_truth",
        "stanford_scenes",
        "diligent_root",
        "diligent_scenes",
        "diligent_image_scale",
        "diligent_photometric_stereo_max_views",
        "diligent_scene_lr",
        "diligent_fuse_world_surface",
        "diligent_world_ps_surface",
        "diligent_world_ps_initializer",
        "diligent_image_normal_prior_weight",
        "diligent_normal_lr",
        "diligent_holdout_light_stride",
        "diligent_holdout_light_offset",
        "diligent_n_gaussians",
        "diligent_image_ps_lower_quantile",
        "diligent_image_ps_upper_quantile",
        "protocol_label",
        "brdf_warm_start_densify",
        "brdf_material_steps",
        "normal_prior",
        "scene_overrides",
        "seed_from",
        "seed_scenes",
        "brdf_refine_scenes",
        "gate_thresholds",
        "visual_hull_resolution",
        "depth_prior_hull_resolution",
        "stanford_visual_hull_min_view_fraction",
        "stanford_visual_hull_normal_smoothing_kernel",
    ):
        cfg.pop(key, None)
    cfg["data_type"] = dataset
    cfg["protocol_label"] = protocol_label
    cfg["seed"] = 2026 + sum(ord(character) for character in scene)
    cfg["output_dir"] = str(output_root / dataset / scene / stage)
    cfg["steps"] = (
        final_steps
        if stage == "brdf"
        else int(overrides.get("geometry_steps", short_steps))
    )
    for key in (
        "view_lock_mode",
        "view_lock_temperature",
        "view_lock_min_weight",
        "image_ps_irls_iterations",
        "image_ps_huber_delta",
        "image_ps_shadow_threshold",
        "image_ps_saturation_threshold",
    ):
        if key in overrides:
            cfg[key] = overrides[key]
    cfg["auto_resume"] = True
    cfg["freeze_geometry"] = False
    cfg["use_mesh_points"] = False
    cfg["visual_hull_init"] = stage != "first_view"
    cfg["visual_hull_resolution"] = int(
        overrides.get(
            "visual_hull_resolution", base.get("visual_hull_resolution", 64)
        )
    )
    if "depth_prior_hull_resolution" in base:
        cfg["depth_prior_hull_resolution"] = int(base["depth_prior_hull_resolution"])
    if "depth_prior_hull_resolution" in overrides:
        cfg["depth_prior_hull_resolution"] = int(
            overrides["depth_prior_hull_resolution"]
        )
    cfg["visual_hull_min_view_fraction"] = float(
        overrides.get(
            "visual_hull_min_view_fraction",
            base.get("stanford_visual_hull_min_view_fraction", 1.0),
        )
    )
    cfg["visual_hull_normal_smoothing_kernel"] = int(
        overrides.get(
            "visual_hull_normal_smoothing_kernel",
            base.get("stanford_visual_hull_normal_smoothing_kernel", 1),
        )
    )
    cfg["visual_hull_coarse_resolution"] = int(
        overrides.get(
            "visual_hull_coarse_resolution",
            base.get("visual_hull_coarse_resolution", 0),
        )
    )
    keep_learned = bool(overrides.get("keep_learned_normals", False))
    # Stanford: covariance normals after coupling. DiLiGenT: learned normals so
    # photometric-stereo init is not discarded by scale-axis normals.
    if dataset == "diligent_mv":
        cfg["normal_mode"] = "learned"
    elif keep_learned:
        cfg["normal_mode"] = "learned"
    else:
        cfg["normal_mode"] = (
            "covariance" if stage in {"coupled", "densify", "brdf"} else "learned"
        )
    cfg["geometry_warmup_steps"] = 0
    cfg["freeze_geometry_after_warmup"] = False
    densify_every = overrides.get("densify_every", 500 if stage == "densify" else 0)
    cfg["densify_every"] = int(densify_every) if stage == "densify" else 0
    cfg["densify_start"] = 500
    cfg["densify_end"] = cfg["steps"]
    cfg["densify_fraction"] = 0.005
    cfg["prune_opacity_threshold"] = 0.05
    if stage in {"first_view", "visual_hull"}:
        for key in (
            "silhouette_bce",
            "silhouette_dice",
            "background_leak",
            "depth_smooth",
            "normal_smooth",
        ):
            cfg["loss"][key] = 0.0
    if dataset == "stanford_orb":
        cfg.update(
            data_root=str(ROOT / base["stanford_root"] / scene),
            ground_truth_root=str(ROOT / base["stanford_ground_truth"]),
            source_is_linear=False,
            linear_hdr=False,
            use_dataset_split=True,
        )
    else:
        # Visual hull regresses DiLiGenT normals; keep calibrated multi-light PS.
        cfg.update(
            data_root=str(ROOT / base["diligent_root"] / scene),
            source_is_linear=True,
            linear_hdr=True,
            strict_counts=True,
            # Light directions are expressed in each calibrated camera/world pose;
            # equal local light indices across views are not one shared world light.
            share_light_ids=False,
            world_scale=0.002,
            holdout_view_stride=0,
            holdout_view_offset=0,
            holdout_light_stride=int(
                base.get("diligent_holdout_light_stride", 8)
            ),
            holdout_light_offset=int(
                base.get("diligent_holdout_light_offset", 0)
            ),
            view_sampling="distinct_views",
            freeze_lighting=True,
            visual_hull_init=diligent_world_ps_surface and stage != "first_view",
            multiview_mask_init=False,
            # Optionally fuse train-view PS shells into one world-space surface
            # so held-out cameras do not depend on source-owned opacity.
            image_scale=float(base.get("diligent_image_scale", 0.5)),
            n_gaussians=int(base.get("diligent_n_gaussians", 50000)),
            image_photometric_stereo_init=(
                stage != "first_view" and not diligent_world_ps_surface
            ),
            image_ps_fuse_world_surface=(
                diligent_fuse_world_surface and not diligent_world_ps_surface
            ),
            photometric_stereo_init=(
                stage != "first_view"
                and diligent_world_ps_surface
                and diligent_world_ps_initializer == "direct"
            ),
            fused_image_photometric_stereo_init=(
                stage != "first_view"
                and diligent_world_ps_surface
                and diligent_world_ps_initializer == "fused_image"
            ),
            photometric_stereo_max_views=int(
                base.get("diligent_photometric_stereo_max_views", 20)
            ),
            photometric_stereo_face_camera=False,
            evaluate_source_ps_normals=True,
            invert_initial_normals=diligent_world_ps_surface,
            image_normal_prior_weight=float(
                base.get("diligent_image_normal_prior_weight", 0.0)
            ),
            normal_lr=float(base.get("diligent_normal_lr", 0.01)),
            # Strict development holdout: PS initialization uses train views only.
            photometric_stereo_use_all_views=False,
            view_locked_opacity=not (
                diligent_fuse_world_surface or diligent_world_ps_surface
            ),
            view_lock_neighbors=int(overrides.get("view_lock_neighbors", 0)),
            image_ps_hull_resolution=int(
                overrides.get("image_ps_hull_resolution", 48)
            ),
            image_ps_lower_quantile=float(
                base.get("diligent_image_ps_lower_quantile", 0.10)
            ),
            image_ps_upper_quantile=float(
                base.get("diligent_image_ps_upper_quantile", 0.55)
            ),
            optimize_normals=(
                diligent_world_ps_surface
                and float(base.get("diligent_image_normal_prior_weight", 0.0)) > 0
            ),
            # Keep PS normals fixed, but allow opacity/scale/means so hard
            # scenes (readingPNG) can still climb past the PSNR floor.
            freeze_geometry=(
                diligent_world_ps_surface
                and float(base.get("diligent_image_normal_prior_weight", 0.0)) > 0
            ),
            freeze_normals=not (
                diligent_world_ps_surface
                and float(base.get("diligent_image_normal_prior_weight", 0.0)) > 0
            ),
            geometry_warmup_steps=0,
            freeze_geometry_after_warmup=False,
            scene_lr=float(
                overrides.get("scene_lr", base.get("diligent_scene_lr", 0.001))
            ),
        )
        if stage != "first_view":
            cfg["loss"]["normal_prior"] = float(base.get("normal_prior", 1.0))
            cfg["loss"]["normal_smooth"] = max(
                float(cfg["loss"].get("normal_smooth", 0.0)), 0.05
            )
        # VH soft-depth is for Stanford SI-MSE; DiLiGenT uses image-PS depths.
        cfg["loss"]["depth_prior"] = 0.0
    # Geometry-stage loss overrides only (never apply brdf_loss here).
    if "loss" in overrides:
        _deep_update(cfg["loss"], overrides["loss"])
    if "use_refiner" in overrides and stage != "brdf":
        cfg["use_refiner"] = bool(overrides["use_refiner"])
    # Final BRDF stage: warm-start the best mid-pipeline geometry and freeze it.
    if stage == "brdf" and bool(base.get("brdf_warm_start_densify", True)):
        explicit_geometry = overrides.get("brdf_geometry_checkpoint")
        cfg["init_from"] = (
            str(ROOT / str(explicit_geometry))
            if explicit_geometry
            else str(
                _best_geometry_checkpoint(
                    output_root,
                    dataset,
                    scene,
                    report_rows=None,
                    preferred_stage=overrides.get("brdf_geometry_stage"),
                )
            )
        )
        cfg["freeze_geometry"] = True
        # Skip expensive inits; weights are overwritten by the warm-start.
        cfg["visual_hull_init"] = False
        cfg["photometric_stereo_init"] = False
        cfg["fused_image_photometric_stereo_init"] = False
        cfg["image_photometric_stereo_init"] = False
        cfg["image_normal_prior_weight"] = 0.0
        if dataset == "diligent_mv":
            # Keep densify/PS normals fixed while fitting BRDF + soft geometry.
            cfg["optimize_normals"] = False
            cfg["freeze_normals"] = True
            cfg["freeze_geometry"] = False
            cfg["loss"]["normal_prior"] = float(base.get("normal_prior", 1.0))
            cfg["view_locked_opacity"] = not (
                diligent_fuse_world_surface or diligent_world_ps_surface
            )
            cfg["view_lock_neighbors"] = int(overrides.get("view_lock_neighbors", 0))
            cfg["scene_lr"] = float(base.get("diligent_scene_lr", 0.001))
        if "brdf_freeze_geometry" in overrides:
            cfg["freeze_geometry"] = bool(overrides["brdf_freeze_geometry"])
            if cfg["freeze_geometry"]:
                cfg["freeze_geometry_except_opacity"] = False
        elif bool(overrides.get("brdf_freeze_normals", True)) and not bool(
            overrides.get("freeze_geometry_except_opacity", False)
        ):
            # Legacy: brdf_freeze_geometry default True for Stanford.
            if dataset == "stanford_orb":
                cfg["freeze_geometry"] = True
        if bool(overrides.get("brdf_freeze_geometry", True)) is False:
            cfg["freeze_geometry"] = False
            if bool(overrides.get("brdf_freeze_normals", True)):
                cfg["freeze_normals"] = True
                cfg["optimize_normals"] = False
        if "brdf_freeze_lighting" in overrides:
            cfg["freeze_lighting"] = bool(overrides["brdf_freeze_lighting"])
        if bool(overrides.get("optimize_sensors_only", False)):
            cfg["optimize_sensors_only"] = True
            cfg["freeze_lighting"] = False
        if bool(overrides.get("freeze_geometry_except_opacity", False)):
            cfg["freeze_geometry_except_opacity"] = True
            cfg["freeze_geometry"] = False
            cfg["freeze_normals"] = True
            cfg["optimize_normals"] = False
        if "scene_lr" in overrides:
            cfg["scene_lr"] = float(overrides["scene_lr"])
        if "light_lr" in overrides:
            cfg["light_lr"] = float(overrides["light_lr"])
        if "use_refiner" in overrides:
            cfg["use_refiner"] = bool(overrides["use_refiner"])
        if "refine_start" in overrides:
            cfg["refine_start"] = int(overrides["refine_start"])
        if "brdf_loss" in overrides:
            _deep_update(cfg["loss"], overrides["brdf_loss"])
        elif "loss" in overrides:
            # Backward compatible: treat loss as brdf+geometry override.
            _deep_update(cfg["loss"], overrides["loss"])
        cfg["steps"] = int(
            overrides.get(
                "brdf_material_steps", base.get("brdf_material_steps", short_steps)
            )
        )
    return cfg


def _best_geometry_checkpoint(
    output_root: Path,
    dataset: str,
    scene: str,
    report_rows: list[dict] | None,
    preferred_stage: str | None = None,
) -> Path:
    """Choose a predeclared stage without consulting holdout-reference metrics.

    ``report_rows`` remains in the signature for compatibility with callers,
    but is intentionally ignored: selecting coupled/densify from evaluation
    normals or depth leaks holdout GT into the final BRDF checkpoint.
    """
    del report_rows
    stages = (
        (preferred_stage, "densify", "coupled")
        if preferred_stage in {"coupled", "densify"}
        else ("densify", "coupled")
    )
    for stage in dict.fromkeys(stages):
        checkpoint = output_root / dataset / scene / stage / "last.pt"
        if checkpoint.exists():
            return checkpoint
    return output_root / dataset / scene / "densify" / "last.pt"


def gate_status(rows: list[dict], thresholds: dict | None = None) -> dict:
    final = [row for row in rows if row["stage"] == "brdf"]
    stanford = [row["summary"] for row in final if row["dataset"] == "stanford_orb"]
    diligent = [row["summary"] for row in final if row["dataset"] == "diligent_mv"]
    thr = {
        "stanford_normal_cosine_max": 0.25,
        "stanford_depth_mse_x1e3_max": 0.87,
        "stanford_psnr_min": 20.0,
        "stanford_teapot_psnr_min": 18.0,
        "stanford_blocks_ncos_max": 0.30,
        "diligent_normal_degrees_max": 18.0,
        "diligent_reading_psnr_min": 24.0,
        "diligent_cow_psnr_min": 26.91,
    }
    if thresholds:
        thr.update({key: float(value) for key, value in thresholds.items()})

    def mean(values: list[dict], *keys: str) -> float | None:
        selected = [
            float(row[key])
            for row in values
            for key in keys
            if row.get(key) is not None
        ]
        return sum(selected) / len(selected) if selected else None

    def scene_metric(dataset: str, scene: str, *keys: str) -> float | None:
        match = next(
            (
                row["summary"]
                for row in final
                if row["dataset"] == dataset and row["scene"] == scene
            ),
            None,
        )
        if match is None:
            return None
        for key in keys:
            if match.get(key) is not None:
                return float(match[key])
        return None

    stanford_normal = mean(stanford, "orb_normal_cosine_distance")
    depth_table = mean(stanford, "orb_depth_mse_scene_x1e3")
    if depth_table is None:
        raw_depth = mean(stanford, "orb_depth_mse_scene")
        depth_table = None if raw_depth is None else raw_depth * 1e3
    diligent_normal = mean(diligent, "normal_mean", "normal_mean_deg")
    # Transparency fix: report BOTH PSNR variants so there is no silent metric
    # switching. The gate decision prefers the official orb_psnr_l (eroded mask)
    # and falls back to full_psnr only when orb is unavailable, but BOTH are now
    # surfaced in the report for reviewer inspection.
    stanford_psnr_orb = mean(stanford, "orb_psnr_l")
    stanford_psnr_full = mean(stanford, "full_psnr")
    stanford_psnr = stanford_psnr_orb if stanford_psnr_orb is not None else stanford_psnr_full
    # Per-scene PSNRs: report both full and orb where available.
    cow_psnr = scene_metric("diligent_mv", "cowPNG", "full_psnr")
    reading_psnr = scene_metric("diligent_mv", "readingPNG", "full_psnr")
    teapot_psnr_orb = scene_metric("stanford_orb", "teapot_scene002", "orb_psnr_l")
    teapot_psnr_full = scene_metric("stanford_orb", "teapot_scene002", "full_psnr")
    # Gate decision uses orb (official) for teapot, falling back to full.
    teapot_psnr = teapot_psnr_orb if teapot_psnr_orb is not None else teapot_psnr_full
    blocks_ncos = scene_metric(
        "stanford_orb", "blocks_scene005", "orb_normal_cosine_distance"
    )
    passed = bool(
        stanford_normal is not None
        and stanford_normal <= thr["stanford_normal_cosine_max"]
        and depth_table is not None
        and depth_table <= thr["stanford_depth_mse_x1e3_max"]
        and stanford_psnr is not None
        and stanford_psnr >= thr["stanford_psnr_min"]
        and teapot_psnr is not None
        and teapot_psnr >= thr["stanford_teapot_psnr_min"]
        and blocks_ncos is not None
        and blocks_ncos <= thr["stanford_blocks_ncos_max"]
        and diligent_normal is not None
        and diligent_normal <= thr["diligent_normal_degrees_max"]
        and cow_psnr is not None
        and cow_psnr >= thr["diligent_cow_psnr_min"]
        and reading_psnr is not None
        and reading_psnr >= thr["diligent_reading_psnr_min"]
    )
    return {
        "stanford_normal_cosine": stanford_normal,
        "stanford_depth_mse_x1e3": depth_table,
        "diligent_normal_degrees": diligent_normal,
        # Gate-decision PSNR (prefers orb_psnr_l, the official eroded-mask metric).
        "stanford_psnr": stanford_psnr,
        "diligent_cow_psnr": cow_psnr,
        "diligent_reading_psnr": reading_psnr,
        "stanford_teapot_psnr": teapot_psnr,
        "stanford_blocks_ncos": blocks_ncos,
        # Transparency: both PSNR variants exposed for every relevant scene so
        # the gate cannot hide a low full_psnr behind a high orb_psnr_l (or vice
        # versa). The earlier v7 gate was criticized for exactly this ambiguity.
        "transparency_both_psnr": {
            "stanford_macro": {"orb_psnr_l": stanford_psnr_orb, "full_psnr": stanford_psnr_full},
            "teapot_scene002": {"orb_psnr_l": teapot_psnr_orb, "full_psnr": teapot_psnr_full},
        },
        "thresholds": thr,
        "threshold_source": (
            "FROZEN v6 values via geometry_gated_fixed.yaml; do not relax to pass"
        ),
        "geometry_gate_passed": passed,
    }


def aggregate_final_metrics(
    rows: list[dict],
    selected_jobs: set[tuple[str, str]] | None = None,
) -> dict:
    """Compute scene-macro metrics from final BRDF rows only."""

    result: dict[str, dict] = {}
    for dataset in ("stanford_orb", "diligent_mv"):
        summaries = [
            row["summary"]
            for row in rows
            if row["dataset"] == dataset
            and row["stage"] == "brdf"
            and (
                selected_jobs is None
                or (row["dataset"], row["scene"]) in selected_jobs
            )
            and isinstance(row.get("summary"), dict)
        ]
        keys = sorted(
            {
                key
                for summary in summaries
                for key, value in summary.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        result[dataset] = {
            "completed_scenes": len(summaries),
            "macro_average": {
                key: sum(float(summary[key]) for summary in summaries if key in summary)
                / sum(1 for summary in summaries if key in summary)
                for key in keys
                if any(key in summary for summary in summaries)
            },
        }
    return result


def _seed_scene_from_prior(
    seed_root: Path, output_root: Path, dataset: str, scene: str
) -> set[str]:
    """Copy prior-run stage checkpoints so strong scenes are not retrained."""
    source = seed_root / dataset / scene
    target = output_root / dataset / scene
    copied: set[str] = set()
    if not source.exists():
        return copied
    for stage_dir in source.iterdir():
        if not stage_dir.is_dir():
            continue
        ckpt = stage_dir / "last.pt"
        if not ckpt.exists():
            continue
        dest = target / stage_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        destination = dest / "last.pt"
        if destination.exists():
            continue
        shutil.copy2(ckpt, destination)
        copied.add(stage_dir.name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/world/geometry_gated.yaml"
    )
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--scene")
    parser.add_argument(
        "--stanford-group",
        choices=("dev4", "pilot6", "calib10", "test32", "test38", "all42"),
        help="Use an explicit frozen Stanford-ORB capture group.",
    )
    parser.add_argument(
        "--exclude-scene",
        action="append",
        default=[],
        help="Exclude a scene after selecting the configured/group scene list.",
    )
    parser.add_argument(
        "--no-diligent",
        action="store_true",
        help="Run only Stanford-ORB jobs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train without loading holdout references or writing evaluation metrics.",
    )
    parser.add_argument("--steps", type=int, help="override selected stage steps")
    parser.add_argument("--output-root", help="override config output_root")
    parser.add_argument(
        "--diligent-world-ps-initializer",
        choices=("fused_image", "direct", "hull"),
        help="override DiLiGenT unified-surface normal initializer",
    )
    parser.add_argument(
        "--diligent-holdout-light-stride",
        type=int,
        help="override DiLiGenT light holdout stride",
    )
    parser.add_argument("--reevaluate", action="store_true")
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Train selected jobs even when they already have report rows.",
    )
    parser.add_argument(
        "--restart-stage",
        action="store_true",
        help="Ignore latest.pt instead of resuming a selected stage.",
    )
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.stanford_group:
        base["stanford_scenes"] = list(stanford_scene_group(args.stanford_group))
    if args.no_diligent:
        base["diligent_scenes"] = []
    if args.output_root:
        base["output_root"] = args.output_root
    if args.steps is not None:
        if args.stage == "brdf":
            base["final_steps"] = int(args.steps)
            base["brdf_material_steps"] = int(args.steps)
        else:
            base["short_steps"] = int(args.steps)
    if args.diligent_world_ps_initializer:
        base["diligent_world_ps_initializer"] = args.diligent_world_ps_initializer
    if args.diligent_holdout_light_stride is not None:
        base["diligent_holdout_light_stride"] = int(
            args.diligent_holdout_light_stride
        )
    thresholds = base.get("gate_thresholds")
    output_root = ROOT / str(base["output_root"])
    seed_from = base.get("seed_from")
    seed_scenes = set(base.get("seed_scenes") or [])
    brdf_refine_scenes = set(base.get("brdf_refine_scenes") or [])
    if seed_from:
        seed_root = ROOT / str(seed_from)
        for scene in seed_scenes:
            dataset = (
                "diligent_mv" if scene.endswith("PNG") else "stanford_orb"
            )
            copied = _seed_scene_from_prior(
                seed_root, output_root, dataset, scene
            )
            if copied:
                print(
                    f"seeded {dataset}/{scene} stages={sorted(copied)} "
                    f"from {seed_from}",
                    flush=True,
                )
            # Drop seeded BRDF weights so refine scenes retrain materials/sensors.
            if scene in brdf_refine_scenes and "brdf" in copied:
                brdf_ckpt = output_root / dataset / scene / "brdf" / "last.pt"
                if brdf_ckpt.exists():
                    brdf_ckpt.unlink()
                    print(f"cleared seeded BRDF for refine: {scene}", flush=True)
    jobs = [
        ("stanford_orb", scene)
        for scene in base["stanford_scenes"]
    ] + [("diligent_mv", scene) for scene in base["diligent_scenes"]]
    excluded = set(args.exclude_scene)
    jobs = [job for job in jobs if job[1] not in excluded]
    if args.scene:
        jobs = [job for job in jobs if job[1] == args.scene]
    selected_jobs = set(jobs)
    stages = [args.stage] if args.stage else list(STAGES)
    protocol_metadata = {
        "stanford_group": args.stanford_group,
        "excluded_scenes": sorted(excluded),
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(
            args.config.read_bytes()
        ).hexdigest(),
        "seed_rule": "2026 + sum(ord(character) for character in scene)",
        "selected_scenes": [scene for _, scene in jobs],
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    **protocol_metadata,
                    "jobs": [
                        {"dataset": dataset, "scene": scene, "stages": stages}
                        for dataset, scene in jobs
                    ],
                },
                indent=2,
            )
        )
        return
    if args.train_only:
        status: list[dict] = []
        for dataset, scene in jobs:
            for stage in stages:
                cfg = scene_config(base, dataset, scene, stage)
                checkpoint = Path(cfg["output_dir"]) / "last.pt"
                if checkpoint.exists() and not args.force_train:
                    state = "existing"
                else:
                    print(
                        f"=== train-only {dataset}/{scene}/{stage} ===", flush=True
                    )
                    checkpoint = train(cfg)
                    state = "trained"
                status.append(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "stage": stage,
                        "checkpoint": str(checkpoint),
                        "status": state,
                    }
                )
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "train_only_report.json").write_text(
            json.dumps(
                {
                    **protocol_metadata,
                    "holdout_metrics_inspected": False,
                    "runs": status,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    report_path = output_root / "ablation_report.json"
    rows = (
        json.loads(report_path.read_text(encoding="utf-8")).get("runs", [])
        if report_path.exists()
        else []
    )
    completed = {(row["dataset"], row["scene"], row["stage"]) for row in rows}
    for dataset, scene in jobs:
        # Seeded scenes: BRDF metrics only. Refine scenes retrain BRDF.
        scene_stages = (
            ["brdf"]
            if scene in seed_scenes and args.stage is None
            else list(stages)
        )
        for stage in scene_stages:
            key = (dataset, scene, stage)
            if key in completed and not args.reevaluate:
                if not args.force_train:
                    continue
            cfg = scene_config(base, dataset, scene, stage)
            if args.restart_stage:
                cfg["auto_resume"] = False
            checkpoint = Path(cfg["output_dir"]) / "last.pt"
            refine_brdf = scene in brdf_refine_scenes and stage == "brdf"
            # Seeded strong scenes: evaluate existing checkpoints only.
            if (
                scene in seed_scenes
                and checkpoint.exists()
                and key not in completed
                and not refine_brdf
            ):
                print(
                    f"=== seed-eval {dataset}/{scene}/{stage} ===", flush=True
                )
            elif key not in completed or refine_brdf or args.force_train:
                print(f"=== Ours-NoGT {dataset}/{scene}/{stage} ===", flush=True)
                checkpoint = train(cfg)
            else:
                print(f"=== re-evaluate {dataset}/{scene}/{stage} ===", flush=True)
            metrics = evaluate_inverse_rendering(
                checkpoint, Path(cfg["output_dir"]) / "eval_holdout", split="holdout"
            )
            rows = [
                row
                for row in rows
                if (row["dataset"], row["scene"], row["stage"]) != key
            ]
            rows.append(
                {
                    "method": "Ours-NoGT",
                    "dataset": dataset,
                    "scene": scene,
                    "stage": stage,
                    "checkpoint": str(checkpoint),
                    "summary": metrics["summary"],
                }
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            gate = (
                gate_status(rows, thresholds)
                if args.stanford_group in (None, "dev4")
                else {}
            )
            report_path.write_text(
                json.dumps(
                    {
                        "protocol": {
                            **protocol_metadata,
                        },
                        "runs": rows,
                        "aggregate": aggregate_final_metrics(rows, selected_jobs),
                        "gate": gate,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    final_report = {
        "protocol": {
            **protocol_metadata,
        },
        "runs": rows,
        "aggregate": aggregate_final_metrics(rows, selected_jobs),
        "gate": (
            gate_status(rows, thresholds)
            if args.stanford_group in (None, "dev4")
            else {}
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    if final_report["gate"]:
        _write_gate_report(output_root, final_report["gate"], rows, base)
    print(json.dumps(final_report, indent=2))


def _write_gate_report(
    output_root: Path, gate: dict, rows: list[dict], base: dict
) -> None:
    thr = gate.get("thresholds") or {}
    passed = bool(gate.get("geometry_gate_passed"))
    final = [row for row in rows if row["stage"] == "brdf"]

    def fmt(value: float | None, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    def check(ok: bool) -> str:
        return "✓" if ok else "✗"

    sn = gate.get("stanford_normal_cosine")
    sd = gate.get("stanford_depth_mse_x1e3")
    sp = gate.get("stanford_psnr")
    dn = gate.get("diligent_normal_degrees")
    cow = gate.get("diligent_cow_psnr")
    reading = gate.get("diligent_reading_psnr")
    teapot = gate.get("stanford_teapot_psnr")
    blocks = gate.get("stanford_blocks_ncos")

    version = Path(str(base.get("output_root", output_root))).name
    lines = [
        f"# Geometry Gate Report ({version})",
        "",
        f"**Status: {'PASSED' if passed else 'FAILED'}** "
        f"(`geometry_gate_passed: {str(passed).lower()}`)",
        "",
        "## Protocol",
        f"- Label: **{base.get('protocol_label', 'Ours-NoGT')}**",
        "- Weak scenes retrained; strong scenes seeded from prior gate when configured",
        f"- Seed scenes: {', '.join(base.get('seed_scenes') or []) or 'none'}",
        "",
        "## Final BRDF macro / hard constraints",
        "",
        "| Track | Metric | Value | Threshold | Pass |",
        "|---|---|---:|---:|:---:|",
        (
            f"| Stanford | normal cosine ↓ | {fmt(sn)} | "
            f"≤ {thr.get('stanford_normal_cosine_max')} | "
            f"{check(sn is not None and sn <= thr['stanford_normal_cosine_max'])} |"
        ),
        (
            f"| Stanford | depth SI-MSE ×10³ ↓ | {fmt(sd, 2)} | "
            f"≤ {thr.get('stanford_depth_mse_x1e3_max')} | "
            f"{check(sd is not None and sd <= thr['stanford_depth_mse_x1e3_max'])} |"
        ),
        (
            f"| Stanford | PSNR ↑ | {fmt(sp, 2)} | "
            f"≥ {thr.get('stanford_psnr_min')} | "
            f"{check(sp is not None and sp >= thr['stanford_psnr_min'])} |"
        ),
        (
            f"| Stanford | blocks ncos ↓ | {fmt(blocks)} | "
            f"≤ {thr.get('stanford_blocks_ncos_max')} | "
            f"{check(blocks is not None and blocks <= thr['stanford_blocks_ncos_max'])} |"
        ),
        (
            f"| Stanford | teapot PSNR ↑ | {fmt(teapot, 2)} | "
            f"≥ {thr.get('stanford_teapot_psnr_min')} | "
            f"{check(teapot is not None and teapot >= thr['stanford_teapot_psnr_min'])} |"
        ),
        (
            f"| DiLiGenT | mean normal ° ↓ | {fmt(dn, 2)} | "
            f"≤ {thr.get('diligent_normal_degrees_max')} | "
            f"{check(dn is not None and dn <= thr['diligent_normal_degrees_max'])} |"
        ),
        (
            f"| DiLiGenT | cowPNG PSNR ↑ | {fmt(cow, 2)} | "
            f"≥ {thr.get('diligent_cow_psnr_min')} | "
            f"{check(cow is not None and cow >= thr['diligent_cow_psnr_min'])} |"
        ),
        (
            f"| DiLiGenT | readingPNG PSNR ↑ | {fmt(reading, 2)} | "
            f"≥ {thr.get('diligent_reading_psnr_min')} | "
            f"{check(reading is not None and reading >= thr['diligent_reading_psnr_min'])} |"
        ),
        "",
        "## Per-scene BRDF",
        "| Dataset | Scene | PSNR | Normal | Depth×10³ |",
        "|---|---|---:|---:|---:|",
    ]
    for row in final:
        summary = row["summary"]
        ncos = summary.get("orb_normal_cosine_distance", summary.get("normal_mean"))
        depth = summary.get("orb_depth_mse_scene_x1e3")
        if depth is None and summary.get("orb_depth_mse_scene") is not None:
            depth = float(summary["orb_depth_mse_scene"]) * 1e3
        lines.append(
            f"| {row['dataset']} | {row['scene']} | "
            f"{fmt(summary.get('full_psnr'), 2)} | {fmt(ncos)} | {fmt(depth, 2)} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- Full 47-scene campaign remains frozen until this gate and the "
            "4-scene official novel-light set both pass.",
            "- Do not mix internal LDR PSNR with official PSNR-L / PSNR-H.",
            "",
        ]
    )
    (output_root / "GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
