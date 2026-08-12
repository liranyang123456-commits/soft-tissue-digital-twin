"""Batch inference on a folder or demo/SHIQ test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mvbrdf_shr.metrics import psnr_torch, ssim_torch
from mvbrdf_shr.models.pipeline import MVBRDFSHR
from mvbrdf_shr.viz import save_image, save_pipeline_viz


def load_rgb(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


@torch.no_grad()
def run_batch(
    input_dir: Path,
    output_dir: Path,
    ckpt: Path | None,
    size: int,
    pattern: str = "*_A.png",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "pred"
    viz_dir = output_dir / "viz"
    pred_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)

    model = MVBRDFSHR(use_generative=True, resolution=size).to(device)
    if ckpt and ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        print(f"loaded {ckpt}")
    model.eval()

    files = sorted(input_dir.glob(pattern))
    if not files:
        files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
    metrics = []
    for path in files:
        img = load_rgb(path, size).unsqueeze(0).to(device)
        out = model(img, use_refine=True)
        stem = path.stem.replace("_A", "")
        save_image(out["pred"], pred_dir / f"{stem}.png")
        # also save with _D naming for eval matching
        save_image(out["pred"], pred_dir / path.name.replace("_A.png", "_D.png"))
        save_pipeline_viz(out, viz_dir / f"{stem}_grid.png", input_img=img)

        gt_path = path.with_name(path.name.replace("_A.png", "_D.png"))
        if gt_path.exists():
            gt = load_rgb(gt_path, size).unsqueeze(0).to(device)
            metrics.append(
                {
                    "file": path.name,
                    "psnr": psnr_torch(out["pred"], gt),
                    "ssim": ssim_torch(out["pred"], gt),
                    "psnr_diffuse": psnr_torch(out["render"].diffuse, gt),
                }
            )

    summary = {"n": len(files), "n_with_gt": len(metrics)}
    if metrics:
        summary["psnr"] = float(np.mean([m["psnr"] for m in metrics]))
        summary["ssim"] = float(np.mean([m["ssim"] for m in metrics]))
        summary["psnr_diffuse"] = float(np.mean([m["psnr_diffuse"] for m in metrics]))
        print(
            f"N={summary['n_with_gt']}  PSNR={summary['psnr']:.3f}  "
            f"SSIM={summary['ssim']:.4f}  PSNR_diff={summary['psnr_diffuse']:.3f}"
        )
    (output_dir / "metrics.json").write_text(json.dumps({"summary": summary, "per_image": metrics}, indent=2), encoding="utf-8")
    # copy preds into baselines folder name for compare
    print(f"wrote predictions -> {pred_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", default="outputs/batch")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--pattern", default="*_A.png")
    args = ap.parse_args()
    run_batch(
        Path(args.input_dir),
        Path(args.output),
        Path(args.ckpt) if args.ckpt else None,
        args.size,
        args.pattern,
    )


if __name__ == "__main__":
    main()
