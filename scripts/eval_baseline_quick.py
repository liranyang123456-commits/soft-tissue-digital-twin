"""Quick standalone evaluator: load a checkpoint, score it on SHIQ val/test/test_full.

Used to back-fill the leakage-fixed baseline (UNet) numbers for the seed0 run
whose in-script final eval crashed on a test_loader NameError. The best.pt is
intact; this just re-runs the three-way eval once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvbrdf_shr.data.shiq_adapter import SHIQAdapter
from mvbrdf_shr.data.synthetic_toy import collate_batch
from mvbrdf_shr.losses_pro import LossWeights  # noqa: F401  (keeps import path stable)
from mvbrdf_shr.metrics import ssim_torch
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHIQ = ROOT.parent / "data" / "raw" / "SHIQ_extracted"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ps, ss = [], []
    for batch in loader:
        img = batch.image.to(device)
        gt = batch.image_clean.to(device)
        pred = model(img, use_refine=True, detach_physics=True)["pred"]
        mse = ((pred - gt) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
        ps.extend((20 * torch.log10(1.0 / mse.sqrt())).tolist())
        for i in range(pred.shape[0]):
            ss.append(ssim_torch(pred[i:i + 1], gt[i:i + 1]))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "n": len(ps)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shiq-root", default=str(DEFAULT_SHIQ))
    ap.add_argument("--resolution", type=int, default=200)
    ap.add_argument("--restorer-kind", choices=["unet", "dhan"], default="unet")
    ap.add_argument("--label", default="baseline")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MVBRDFSHRPro(
        resolution=args.resolution, base_ch=48, restorer_base=96,
        restorer_kind=args.restorer_kind,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)

    out = {"label": args.label, "ckpt": args.ckpt}
    for split in ("val", "test", "test_full"):
        ds = SHIQAdapter(args.shiq_root, args.resolution, split)
        loader = DataLoader(
            ds, batch_size=8, shuffle=False,
            collate_fn=collate_batch, num_workers=0,
        )
        m = evaluate(model, loader, device)
        out[split] = m
        print(f"  {split:9s} n={m['n']:5d}  PSNR={m['psnr']:.3f}  SSIM={m['ssim']:.4f}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
