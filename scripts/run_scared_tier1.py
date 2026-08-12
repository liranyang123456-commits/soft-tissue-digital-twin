"""B2: third-dataset portability check on SCARED (real stereo laparoscopy).

SCARED provides per-keyframe stereo pairs, metric depth maps (mm), and
endoscope calibration. This script runs the Tier-1 optical-geometric
reconstruction on SCARED keyframes (single-view + metric depth), which
exercises the pipeline on a different anatomy (porcine abdomen), a
different camera (stereo laparoscope, 1280x1024), and metric units —
complementing the fixed-camera EndoNeRF experiments.

Outputs: ``outputs/scared_tier1/`` (per-keyframe compare PNG + summary JSON).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.data import MultiViewFrame, MultiViewScene  # noqa: E402
from mvbrdf_shr.world.camera import PerspectiveCamera  # noqa: E402
from scripts.run_soft_tissue_twin import run_tier1  # noqa: E402

SCARED = Path(r"E:\MIS_Datasets\SCARED")


def load_scared_keyframe(kf_dir: Path, image_scale: float = 0.5) -> MultiViewScene:
    """Load one SCARED keyframe as a single-frame MultiViewScene."""
    img_bgr = cv2.imread(str(kf_dir / "Left_Image.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(kf_dir / "left_depth_map.tiff"), cv2.IMREAD_UNCHANGED)
    if img_bgr is None or depth is None:
        raise FileNotFoundError(kf_dir)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    depth = depth[..., 0].astype(np.float32)  # mm, NaN = invalid
    depth = np.nan_to_num(depth, nan=0.0)

    H, W = img.shape[:2]
    # Intrinsics from the calibration yaml (left camera, M1).
    fx = 1035.30810546875
    fy = 1035.087646484375
    cx, cy = 596.9550170898438, 520.4100341796875
    if image_scale != 1.0:
        img = cv2.resize(img, (int(W * image_scale), int(H * image_scale)))
        depth = cv2.resize(depth, (int(W * image_scale), int(H * image_scale)),
                           interpolation=cv2.INTER_NEAREST)
        fx, fy, cx, cy = (fx * image_scale, fy * image_scale,
                          cx * image_scale, cy * image_scale)
    H2, W2 = img.shape[:2]

    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=torch.float32)
    cam = PerspectiveCamera(
        K=K, c2w=torch.eye(4), width=W2, height=H2, frame_id=0, view_id=0,
    )
    frame = MultiViewFrame(
        image=img_t, camera=cam, image_path=kf_dir / "Left_Image.png",
        depth_gt=torch.from_numpy(depth), view_id=0,
    )
    return MultiViewScene(frames=[frame], metadata={"source": "scared"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="dataset_1")
    p.add_argument("--keyframes", type=int, default=2)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--max-points", type=int, default=8000)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--backend", default="gsplat", choices=["auto", "torch", "gsplat"])
    p.add_argument("--output", default=str(ROOT / "outputs" / "scared_tier1"))
    args = p.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    kf_dirs = sorted((SCARED / args.dataset).glob("keyframe_*"))[: args.keyframes]
    rows = []
    for kf in kf_dirs:
        print(f"\n=== {args.dataset}/{kf.name} ===")
        scene = load_scared_keyframe(kf, image_scale=args.scale)
        kf_out = out_root / f"{args.dataset}_{kf.name}"
        tier1 = run_tier1(args, scene, device, kf_out)
        rows.append({
            "keyframe": kf.name,
            "tissue_psnr": tier1["psnr"],
            "all_psnr": tier1["psnr_all"],
            "n_gaussians": tier1["n_gaussians"],
        })
        print(f"  -> tissue-PSNR {tier1['psnr']:.2f} dB")

    summary = {
        "dataset": args.dataset,
        "rows": rows,
        "tissue_psnr_mean": float(np.mean([r["tissue_psnr"] for r in rows])),
        "note": "single-view + metric stereo depth; camera intrinsics from "
                "endoscope_calibration.yaml; depth in mm",
    }
    (out_root / "scared_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nMean tissue-PSNR: {summary['tissue_psnr_mean']:.2f} dB")
    print(f"Saved: {out_root / 'scared_summary.json'}")


if __name__ == "__main__":
    main()
