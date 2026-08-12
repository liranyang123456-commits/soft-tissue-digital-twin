"""Copy baseline predictions and compute PSNR/SSIM on world_benchmark_v2."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mvbrdf_shr.metrics import psnr_torch, ssim_torch


def eval_subset(gt_dir: Path, pred_dir: Path, names: set[str] | None = None) -> dict:
    psnr, ssim = [], []
    for gt_path in sorted(gt_dir.glob("*.png")):
        if names is not None and gt_path.name not in names:
            continue
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt = torch.from_numpy(
            np.asarray(Image.open(gt_path).convert("RGB")).astype("float32") / 255
        ).permute(2, 0, 1).unsqueeze(0)
        pred = torch.from_numpy(
            np.asarray(Image.open(pred_path).convert("RGB")).astype("float32") / 255
        ).permute(2, 0, 1).unsqueeze(0)
        psnr.append(psnr_torch(pred, gt))
        ssim.append(ssim_torch(pred, gt))
    return {"n": len(psnr), "psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dataset", default="data/world_benchmark_v2")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--skip-copy", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = root / args.dataset
    src = Path(args.src)
    if not src.is_absolute():
        src = root / src
    dst = dataset / "predictions" / args.method
    dst.mkdir(parents=True, exist_ok=True)
    if not args.skip_copy:
        for path in src.glob("*.png"):
            shutil.copy2(path, dst / path.name)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    hard_names = {
        f"scene_{frame['scene']:04d}_v{frame['view']:03d}.png"
        for frame in manifest["frames"]
        if frame.get("split") == "test" and frame.get("hard_test")
    }
    gt_dir = dataset / "flat" / "test" / "gt"
    result = {
        "method": args.method,
        "checkpoint": args.checkpoint,
        "test": eval_subset(gt_dir, dst),
        "hard_test": eval_subset(gt_dir, dst, hard_names),
        "prediction_dir": str(dst.relative_to(root)),
    }
    (dst / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
