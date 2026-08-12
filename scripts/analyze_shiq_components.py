"""Diagnose which SHIQ pipeline component limits final quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvbrdf_shr.data.shiq_adapter import SHIQAdapter
from mvbrdf_shr.data.synthetic_toy import collate_batch
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro


def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    mse = (pred - target).square().mean((1, 2, 3)).clamp(min=1e-12)
    return (20 * torch.log10(1 / mse.sqrt())).cpu().tolist()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shiq-root", default="../data/raw/SHIQ_extracted")
    parser.add_argument("--batch-size", type=int, default=8)
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
    scores = {
        "input_psnr": [],
        "physics_diffuse_psnr": [],
        "coarse_psnr": [],
        "final_psnr": [],
        "specular_psnr": [],
    }
    intersections = unions = true_positive = predicted_positive = gt_positive = 0.0
    for batch in loader:
        image = batch.image.to(device)
        clean = batch.image_clean.to(device)
        specular = batch.specular_gt.to(device)
        mask = batch.mask_gt.to(device)
        output = model(image, use_refine=True, detach_physics=True)
        scores["input_psnr"] += batch_psnr(image, clean)
        scores["physics_diffuse_psnr"] += batch_psnr(
            output["render"].diffuse, clean
        )
        scores["coarse_psnr"] += batch_psnr(output["coarse"], clean)
        scores["final_psnr"] += batch_psnr(output["pred"], clean)
        scores["specular_psnr"] += batch_psnr(output["spec_pred"], specular)
        prediction = output["mask_pred"] >= 0.5
        target = mask >= 0.5
        intersections += float((prediction & target).sum())
        unions += float((prediction | target).sum())
        true_positive += float((prediction & target).sum())
        predicted_positive += float(prediction.sum())
        gt_positive += float(target.sum())
    result = {key: float(np.mean(value)) for key, value in scores.items()}
    result.update(
        mask_iou=intersections / max(unions, 1),
        mask_precision=true_positive / max(predicted_positive, 1),
        mask_recall=true_positive / max(gt_positive, 1),
        mask_f1=2
        * true_positive
        / max(predicted_positive + gt_positive, 1),
    )
    output_path = Path(args.checkpoint).parent / "component_analysis.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

