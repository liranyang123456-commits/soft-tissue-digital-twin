"""Leakage-free HDR material calibration for strict Stanford checkpoints."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from mvbrdf_shr.world.evaluate_ir import evaluate_inverse_rendering
from mvbrdf_shr.world.train import train


ROOT = Path(__file__).resolve().parents[1]
HDR_ROOT = ROOT / "data" / "external" / "stanford_orb" / "blender_HDR"
GT_ROOT = ROOT / "data" / "external" / "stanford_orb" / "ground_truth"
DEFAULT_SCENES = (
    "ball_scene003",
    "blocks_scene005",
    "gnome_scene003",
    "teapot_scene002",
)


def _checkpoint(campaign: Path, scene: str) -> Path:
    for candidate in (
        campaign / "stanford_orb" / scene / "brdf" / "last.pt",
        campaign / "stanford_orb" / scene / "last.pt",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing strict checkpoint for {scene}")


def _calibration_init(checkpoint: Path, destination: Path) -> dict:
    """Copy the strict state while dropping optimizer-only training history."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = copy.deepcopy(state)
    state.pop("optimizer", None)
    state.pop("scheduler", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    return state


def _calibration_config(
    state: dict,
    *,
    scene: str,
    init_from: Path,
    output_dir: Path,
    steps: int,
) -> dict:
    cfg = copy.deepcopy(state["config"])
    cfg.update(
        data_type="stanford_orb",
        data_root=str((HDR_ROOT / scene).resolve()),
        ground_truth_root=str(GT_ROOT.resolve()),
        source_is_linear=True,
        linear_hdr=True,
        include_novel=False,
        use_dataset_split=True,
        output_dir=str(output_dir),
        init_from=str(init_from),
        auto_resume=False,
        steps=int(steps),
        checkpoint_every=max(250, int(steps) // 4),
        freeze_geometry=True,
        freeze_geometry_except_opacity=False,
        freeze_normals=True,
        optimize_normals=False,
        optimize_sensors_only=False,
        shared_lighting=bool(state["config"].get("shared_lighting", False)),
        trainable_lighting_parameters=[
            "log_exposure",
            "log_white_balance",
        ],
        scene_lr=5e-4,
        light_lr=2e-3,
        views_per_step=4,
        densify_every=0,
        use_refiner=False,
    )
    loss = copy.deepcopy(cfg.get("loss") or {})
    loss.update(
        photo_l1=0.35,
        photo_ssim=0.05,
        photo_log=0.75,
        photo_highlight=0.40,
        normal_gt=0.0,
        depth_gt=0.0,
        depth_prior=0.0,
        silhouette_bce=0.05,
        silhouette_dice=0.05,
        background_leak=0.05,
        normal_smooth=0.0,
        normal_edge_aware=0.0,
        normal_planar=0.0,
        exposure_prior=0.002,
        white_balance_prior=0.002,
        material_smooth=0.02,
        normal_axis=0.0,
        scale=0.0,
        opacity_sparse=0.0,
        texture_consistency=0.01,
    )
    cfg["loss"] = loss
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        type=Path,
        default=ROOT / "outputs" / "geometry_gated_softlock_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "hdr_material_calibration_v1",
    )
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    campaign = args.campaign if args.campaign.is_absolute() else ROOT / args.campaign
    output = args.output if args.output.is_absolute() else ROOT / args.output

    rows = []
    for index, scene in enumerate(args.scenes, 1):
        source = _checkpoint(campaign, scene)
        scene_root = output / "stanford_orb" / scene / "brdf"
        adapted = scene_root / "strict_geometry_init.pt"
        state = _calibration_init(source, adapted)
        cfg = _calibration_config(
            state,
            scene=scene,
            init_from=adapted,
            output_dir=scene_root,
            steps=args.steps,
        )
        print(f"[{index}/{len(args.scenes)}] HDR calibration {scene}", flush=True)
        checkpoint = train(cfg)
        report = evaluate_inverse_rendering(
            checkpoint,
            scene_root / "eval_test",
            split="test",
            compute_lpips=False,
        )
        rows.append({"scene": scene, "summary": report["summary"]})

    result = {"scenes": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
