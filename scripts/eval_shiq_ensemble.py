"""Ensemble + TTA evaluation on SHIQ for the SOTA push.

Combines two free PSNR gains that need no retraining:
1. Multi-checkpoint ensemble: average predictions of several DHAN checkpoints
   (longer 34.17, a_dhan_only_s0 33.94, a_dhan_only_s1 33.95). Diverse training
   trajectories reduce uncorrelated errors.
2. 4-flip test-time augmentation: predict each image under 4 flip combos and
   average the inverse-transformed predictions.

Reports the gain over the single-best checkpoint (longer).
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
from mvbrdf_shr.metrics import ssim_torch
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro


def transform(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode & 1:
        x = x.flip(-1)
    if mode & 2:
        x = x.flip(-2)
    return x


@torch.no_grad()
def predict(model, image, tta_modes):
    """Return the TTA-averaged prediction for a batch."""
    preds = []
    for mode in tta_modes:
        aug = transform(image, mode)
        pred = model(aug, use_refine=True, detach_physics=True)["pred"]
        preds.append(transform(pred, mode))
    return torch.stack(preds).mean(0)


def load_model(ckpt, device, restorer_kind="dhan"):
    model = MVBRDFSHRPro(200, 48, 96, restorer_kind=restorer_kind).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="One or more DHAN checkpoints to ensemble.")
    ap.add_argument("--shiq-root", default="../data/raw/SHIQ_extracted")
    ap.add_argument("--split", default="test_full")
    ap.add_argument("--tta", type=int, default=4, choices=[1, 2, 4],
                    help="Number of flip modes for TTA (1=no TTA).")
    ap.add_argument("--output", default="outputs/ensemble_tta_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SHIQAdapter(args.shiq_root, 200, args.split)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_batch, num_workers=0)
    modes = list(range(args.tta))

    models = [load_model(c, device) for c in args.checkpoints]
    print(f"loaded {len(models)} checkpoints; TTA modes={len(modes)}; split={args.split} n={len(ds)}")

    psnrs, ssims = [], []
    for batch in loader:
        image = batch.image.to(device)
        target = batch.image_clean.to(device)
        # Ensemble: average TTA predictions across all checkpoints.
        per_model = [predict(m, image, modes) for m in models]
        pred = torch.stack(per_model).mean(0)
        mse = (pred - target).square().mean((1, 2, 3)).clamp(min=1e-12)
        psnrs.extend((20 * torch.log10(1 / mse.sqrt())).cpu().tolist())
        for i in range(len(pred)):
            ssims.append(ssim_torch(pred[i:i+1], target[i:i+1]).item())

    result = {
        "checkpoints": args.checkpoints,
        "n_models": len(models),
        "tta_modes": args.tta,
        "split": args.split,
        "n": len(psnrs),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
