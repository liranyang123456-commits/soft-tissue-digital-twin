"""Evaluate existing Stanford checkpoints on the HDR novel-light split."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from mvbrdf_shr.world.evaluate_ir import (
    evaluate_inverse_rendering,
    load_world_checkpoint,
)
from mvbrdf_shr.world.lighting import WorldLight
from mvbrdf_shr.world.protocols import stanford_scene_group

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "outputs" / "full_external_campaign"
HDR_ROOT = ROOT / "data" / "external" / "stanford_orb" / "blender_HDR"
GT_ROOT = ROOT / "data" / "external" / "stanford_orb" / "ground_truth"
DEFAULT_OUTPUT = ROOT / "outputs" / "hdr_novel_light_eval"
DEFAULT_SCENES = (
    "ball_scene003",
    "blocks_scene005",
    "gnome_scene003",
    "teapot_scene002",
)


@torch.inference_mode()
def _calibrate_known_lighting(
    checkpoint: Path,
    *,
    max_views: int,
) -> dict[str, float | int]:
    """Fit two global light gains using HDR *training* frames only.

    The known environment is split into its low-order SH component and the
    dominant directional lobe already used by the renderer.  Geometry and
    material stay frozen.  No novel frame participates in this fit.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, dataset, pipeline = load_world_checkpoint(checkpoint, device)
    train_ids = [
        index
        for index, frame in enumerate(dataset.frames)
        if (frame.metadata or {}).get("split") == "train"
        and frame.env_sh is not None
        and frame.light_direction is not None
        and frame.light_color is not None
        and frame.light_intensity is not None
    ]
    if not train_ids:
        return {
            "novel_env_scale": 1.0,
            "novel_direct_scale": 1.0,
            "calibration_views": 0,
        }
    if len(train_ids) > max_views:
        positions = torch.linspace(0, len(train_ids) - 1, max_views).long()
        train_ids = [train_ids[int(position)] for position in positions]

    normal = torch.zeros(2, 2, dtype=torch.float64, device=device)
    rhs = torch.zeros(2, dtype=torch.float64, device=device)
    one = torch.ones(1, device=device)
    white = torch.ones(3, device=device)
    zero_sh = torch.zeros(9, 3, device=device)
    zero_intensity = torch.zeros(1, device=device)
    for frame_id in train_ids:
        frame = dataset[frame_id]
        camera = frame.camera.to(device)
        image = frame.image.to(device)
        common = dict(
            direction=frame.light_direction.to(device),
            color=frame.light_color.to(device),
            position=torch.zeros(3, device=device),
            point_weight=torch.zeros(1, device=device),
            exposure=one,
            white_balance=white,
        )
        env_light = WorldLight(
            env_sh=frame.env_sh.to(device),
            intensity=zero_intensity,
            **common,
        )
        direct_light = WorldLight(
            env_sh=zero_sh,
            intensity=torch.tensor(
                [float(frame.light_intensity)], device=device
            ),
            **common,
        )
        env_render = pipeline(
            camera,
            frame_id,
            refine=False,
            apply_sensor=False,
            light_override=env_light,
        )["render"]
        direct_render = pipeline(
            camera,
            frame_id,
            refine=False,
            apply_sensor=False,
            light_override=direct_light,
        )["render"]
        mask = (
            frame.mask_gt.to(device) > 0.5
            if frame.mask_gt is not None
            else env_render.alpha > 0.01
        )
        valid = mask.expand_as(image)
        env = env_render.full[valid].double()
        direct = direct_render.full[valid].double()
        target = image[valid].double()
        normal[0, 0] += (env * env).sum()
        normal[0, 1] += (env * direct).sum()
        normal[1, 1] += (direct * direct).sum()
        rhs[0] += (env * target).sum()
        rhs[1] += (direct * target).sum()
    normal[1, 0] = normal[0, 1]
    ridge = torch.eye(2, dtype=normal.dtype, device=device) * (
        normal.diag().mean().clamp_min(1.0) * 1e-6
    )
    gains = torch.linalg.solve(normal + ridge, rhs).clamp(0.0, 10.0)
    return {
        "novel_env_scale": float(gains[0]),
        "novel_direct_scale": float(gains[1]),
        "calibration_views": len(train_ids),
    }


def _resolve_checkpoint(campaign: Path, scene: str) -> Path | None:
    """Prefer brdf-stage last.pt (gated layouts), else scene/last.pt."""
    candidates = (
        campaign / "stanford_orb" / scene / "brdf" / "last.pt",
        campaign / "stanford_orb" / scene / "last.pt",
        campaign / scene / "brdf" / "last.pt",
        campaign / scene / "last.pt",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        type=Path,
        default=DEFAULT_CAMPAIGN,
        help="Root containing stanford_orb/<scene>/[brdf/]last.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for novel-light metrics and report",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Stanford scenes to evaluate (default: 4 gate scenes)",
    )
    parser.add_argument(
        "--stanford-group",
        choices=("dev4", "pilot6", "calib10", "test32", "test38", "all42"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--calibrate-lighting",
        action="store_true",
        help="Fit global env/direct gains on HDR training frames only",
    )
    parser.add_argument(
        "--calibration-views",
        type=int,
        default=12,
        help="Maximum HDR training views used for leakage-free calibration",
    )
    parser.add_argument(
        "--full-env-samples",
        type=int,
        default=0,
        help="Use deterministic full-environment quadrature (0 keeps SH+dominant)",
    )
    parser.add_argument(
        "--disable-neural-brdf",
        action="store_true",
        help="Evaluate the same checkpoint with the neural BRDF blend disabled",
    )
    parser.add_argument(
        "--neural-brdf-strength",
        type=float,
        help="Override the bounded neural residual strength for development sweeps.",
    )
    parser.add_argument(
        "--neural-brdf-highlight-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable narrow-highlight residual gating.",
    )
    parser.add_argument(
        "--neural-brdf-boundary-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable explicit GGX fallback near silhouettes.",
    )
    args = parser.parse_args()
    if args.scenes and args.stanford_group:
        parser.error("--scenes and --stanford-group are mutually exclusive")
    args.scenes = (
        list(stanford_scene_group(args.stanford_group))
        if args.stanford_group
        else list(args.scenes or DEFAULT_SCENES)
    )

    campaign = args.campaign if args.campaign.is_absolute() else ROOT / args.campaign
    output = args.output if args.output.is_absolute() else ROOT / args.output
    scenes = list(args.scenes)
    if args.limit is not None:
        scenes = scenes[: args.limit]

    rows = []
    for position, scene in enumerate(scenes, 1):
        checkpoint = _resolve_checkpoint(campaign, scene)
        if checkpoint is None:
            print(f"[{position}/{len(scenes)}] {scene}: missing checkpoint, skip", flush=True)
            continue
        out_dir = output / scene / "eval_novel"
        metrics_path = out_dir / "metrics.json"
        print(f"[{position}/{len(scenes)}] {scene} <- {checkpoint}", flush=True)
        calibration: dict[str, float | int] | None = None
        if metrics_path.exists() and not args.force:
            summary = json.loads(metrics_path.read_text(encoding="utf-8"))["summary"]
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            cfg = state["config"]
            cfg["data_root"] = str((HDR_ROOT / scene).resolve())
            cfg["ground_truth_root"] = str(GT_ROOT.resolve())
            cfg["source_is_linear"] = True
            cfg["linear_hdr"] = True
            cfg["include_novel"] = True
            cfg["novel_full_env_samples"] = int(args.full_env_samples)
            if args.disable_neural_brdf:
                cfg["neural_brdf_strength"] = 0.0
            elif args.neural_brdf_strength is not None:
                cfg["neural_brdf_strength"] = float(args.neural_brdf_strength)
            if args.neural_brdf_highlight_gate is not None:
                cfg["neural_brdf_highlight_gate"] = bool(
                    args.neural_brdf_highlight_gate
                )
            if args.neural_brdf_boundary_fallback is not None:
                cfg["neural_brdf_boundary_fallback"] = bool(
                    args.neural_brdf_boundary_fallback
                )
            state["config"] = cfg
            adapted = output / scene / "hdr_cfg.pt"
            adapted.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, adapted)
            calibration = {
                "novel_env_scale": 1.0,
                "novel_direct_scale": 1.0,
                "calibration_views": 0,
            }
            if args.calibrate_lighting:
                calibration = _calibrate_known_lighting(
                    adapted,
                    max_views=args.calibration_views,
                )
                cfg.update(calibration)
                state["config"] = cfg
                torch.save(state, adapted)
                print(f"  train-only light calibration: {calibration}", flush=True)
            report = evaluate_inverse_rendering(
                adapted,
                out_dir,
                split="novel",
                compute_lpips=args.lpips,
            )
            summary = report["summary"]
        rows.append(
            {
                "method": "Ours-NoGT",
                "scene": scene,
                "summary": summary,
                "lighting_calibration": calibration,
            }
        )
        keys = (
            "orb_psnr_h",
            "orb_psnr_l",
            "orb_ssim",
            "orb_lpips",
            "full_psnr",
            "full_ssim",
        )
        macro = {
            key: sum(
                float(row["summary"][key])
                for row in rows
                if row["summary"].get(key) is not None
            )
            / max(1, sum(1 for row in rows if row["summary"].get(key) is not None))
            for key in keys
            if any(row["summary"].get(key) is not None for row in rows)
        }
        aggregate = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "campaign": str(campaign),
            "completed_scenes": len(rows),
            "macro_average": macro,
            "scenes": rows,
            "note": (
                "LDR-trained Ours-NoGT checkpoints evaluated on HDR novel-light "
                "split with envmap-initialized novel lighting. Optional global "
                "env/direct gains use HDR training frames only; novel frames are "
                "evaluation-only."
            ),
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "novel_light_report.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
    if rows:
        print(json.dumps(aggregate["macro_average"], indent=2))
    else:
        raise SystemExit("No scenes evaluated.")


if __name__ == "__main__":
    main()
