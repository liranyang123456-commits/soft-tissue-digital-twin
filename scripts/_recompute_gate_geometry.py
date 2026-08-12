"""Recompute Stanford depth/normal gate metrics on CPU with correct GT arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mvbrdf_shr.metrics_ir import erode_binary_mask, orb_scene_depth_mse
from mvbrdf_shr.world.train import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def normal_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred = F.normalize(pred.float(), dim=0, eps=1e-6)
    target = F.normalize(target.float(), dim=0, eps=1e-6)
    valid = erode_binary_mask(mask).bool()
    if valid.ndim == 3:
        valid = valid[0]
    cosine = (pred * target).sum(dim=0).clamp(-1, 1)
    return float((1.0 - cosine)[valid].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=["brdf"])
    parser.add_argument("--update-report", action="store_true")
    args = parser.parse_args()
    root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    report_path = root / "ablation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scenes = [
        "ball_scene003",
        "blocks_scene005",
        "gnome_scene003",
        "teapot_scene002",
    ]
    for stage in args.stages:
        print(f"=== {stage}")
        for scene in scenes:
            ckpt = root / "stanford_orb" / scene / stage / "last.pt"
            eval_dir = root / "stanford_orb" / scene / stage / "eval_holdout"
            if not ckpt.exists() or not eval_dir.exists():
                print(scene, "missing")
                continue
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            dataset = load_dataset(dict(state["config"]))
            hold = list(state["holdout_ids"])
            preds_d, tgts_d, masks, cosines = [], [], [], []
            incomplete = False
            for index in hold:
                frame_dir = eval_dir / f"{index:04d}"
                depth_path = frame_dir / "depth.npy"
                # Prefer camera-space official normal if present, else world normal.
                normal_path = frame_dir / "normal_camera.npy"
                if not normal_path.exists():
                    normal_path = frame_dir / "normal.npy"
                if not depth_path.exists() or not normal_path.exists():
                    incomplete = True
                    break
                frame = dataset[index]
                pred_depth = torch.from_numpy(np.load(depth_path))
                pred_normal = torch.from_numpy(np.load(normal_path))
                if pred_normal.min() >= 0:
                    pred_normal = pred_normal * 2.0 - 1.0
                target_normal = frame.normal_gt
                # Saved normal_camera is already in ORB camera space; GT from
                # loader is world-space. Prefer world saved normal when present.
                world_normal_path = frame_dir / "normal.npy"
                if world_normal_path.exists():
                    pred_normal = torch.from_numpy(np.load(world_normal_path))
                    if pred_normal.min() >= 0:
                        pred_normal = pred_normal * 2.0 - 1.0
                preds_d.append(pred_depth)
                tgts_d.append(frame.depth_gt)
                masks.append(frame.mask_gt)
                cosines.append(normal_cosine(pred_normal, target_normal, frame.mask_gt))
            if incomplete:
                print(scene, "incomplete eval")
                continue
            raw = orb_scene_depth_mse(preds_d, tgts_d, masks)
            ncos = float(sum(cosines) / len(cosines))
            row = next(
                item
                for item in report["runs"]
                if item["scene"] == scene and item["stage"] == stage
            )
            summary = row["summary"]
            print(
                f"{scene}: psnr={summary.get('full_psnr'):.2f} "
                f"old_ncos={summary.get('orb_normal_cosine_distance')} "
                f"new_ncos={ncos:.4f} "
                f"old_depth={summary.get('orb_depth_mse_scene')} "
                f"new_depth_raw={raw:.6f} new_depth_x1e3={raw * 1e3:.3f}"
            )
            if args.update_report:
                summary["orb_normal_cosine_distance"] = ncos
                summary["orb_depth_mse_scene"] = raw
                summary["orb_depth_mse_scene_x1e3"] = raw * 1e3
                summary["geometry_metrics_recomputed"] = True
    if args.update_report:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_geometry_gated_ablation",
            ROOT / "scripts" / "run_geometry_gated_ablation.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        report["gate"] = module.gate_status(report["runs"])
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("updated gate:", json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
