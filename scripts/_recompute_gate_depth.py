"""Recompute Stanford depth SI-MSE on CPU from saved eval depth maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mvbrdf_shr.metrics_ir import orb_scene_depth_mse
from mvbrdf_shr.world.train import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["densify", "brdf"],
    )
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
            preds, tgts, masks = [], [], []
            incomplete = False
            for index in hold:
                path = eval_dir / f"{index:04d}" / "depth.npy"
                if not path.exists():
                    incomplete = True
                    break
                preds.append(torch.from_numpy(np.load(path)))
                frame = dataset[index]
                tgts.append(frame.depth_gt)
                masks.append(frame.mask_gt)
            if incomplete:
                print(scene, "incomplete eval")
                continue
            raw = orb_scene_depth_mse(preds, tgts, masks)
            row = next(
                item
                for item in report["runs"]
                if item["scene"] == scene and item["stage"] == stage
            )
            summary = row["summary"]
            print(
                f"{scene}: psnr={summary.get('full_psnr'):.2f} "
                f"ncos={summary.get('orb_normal_cosine_distance'):.4f} "
                f"depth_raw={raw:.6f} depth_x1e3={raw * 1e3:.3f} "
                f"old={summary.get('orb_depth_mse_scene')}"
            )


if __name__ == "__main__":
    main()
