"""Evaluate SHIQ with flip test-time augmentation."""
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shiq-root", default="../data/raw/SHIQ_extracted")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--modes", type=int, default=4, choices=[1, 2, 4])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SHIQAdapter(args.shiq_root, 200, "test")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_batch, num_workers=0
    )
    model = MVBRDFSHRPro(200, 48, 96).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    psnr, ssim = [], []
    for batch in loader:
        image = batch.image.to(device)
        target = batch.image_clean.to(device)
        predictions = []
        for mode in range(args.modes):
            augmented = transform(image, mode)
            prediction = model(
                augmented, use_refine=True, detach_physics=True
            )["pred"]
            predictions.append(transform(prediction, mode))
        pred = torch.stack(predictions).mean(0)
        mse = (pred - target).square().mean((1, 2, 3)).clamp(min=1e-12)
        psnr.extend((20 * torch.log10(1 / mse.sqrt())).cpu().tolist())
        for index in range(len(pred)):
            ssim.append(
                ssim_torch(pred[index : index + 1], target[index : index + 1])
            )
    result = {
        "n": len(psnr),
        "modes": args.modes,
        "psnr": float(np.mean(psnr)),
        "ssim": float(np.mean(ssim)),
    }
    output = Path(args.checkpoint).parent / "tta_results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

