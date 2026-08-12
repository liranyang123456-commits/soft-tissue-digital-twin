"""Evaluate MVBRDF-SHR Pro on SHIQ or SSHR official paired splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from mvbrdf_shr.data.shiq_adapter import (
    NSHAdapter,
    PSDAdapter,
    SHIQAdapter,
    SSHRAdapter,
)
from mvbrdf_shr.data.synthetic_toy import collate_batch
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro
from mvbrdf_shr.viz import save_image


def batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float().clamp(0, 1) - target.float().clamp(0, 1)).square()
    mse = mse.flatten(1).mean(1).clamp(min=1e-12)
    return -10 * torch.log10(mse)


def batch_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    prediction = F.pad(prediction, (5, 5, 5, 5), mode="reflect")
    target = F.pad(target, (5, 5, 5, 5), mode="reflect")
    mean_p = F.avg_pool2d(prediction, 11, stride=1)
    mean_t = F.avg_pool2d(target, 11, stride=1)
    var_p = F.avg_pool2d(prediction.square(), 11, stride=1) - mean_p.square()
    var_t = F.avg_pool2d(target.square(), 11, stride=1) - mean_t.square()
    covariance = F.avg_pool2d(prediction * target, 11, stride=1) - mean_p * mean_t
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_p * mean_t + c1) * (2 * covariance + c2)) / (
        (mean_p.square() + mean_t.square() + c1)
        * (var_p + var_t + c2)
        + 1e-12
    )
    return score.flatten(1).mean(1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("shiq", "sshr", "nsh", "psd"), required=True
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="outputs/shiq_mask/best.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--restorer-kind", choices=["unet", "dhan"], default="dhan",
        help="Restorer backbone of the checkpoint (path A). Auto-detect defaults "
             "to dhan for the SOTA-push checkpoints.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if args.dataset == "shiq":
        dataset = SHIQAdapter(root, args.size, args.split)
    elif args.dataset == "sshr":
        dataset = SSHRAdapter(
            root, args.size, args.split, tone_corrected=True
        )
    elif args.dataset == "nsh":
        dataset = NSHAdapter(root, args.size, args.split)
    else:
        # PSD supports test (full) or ft_test (held-out groups after FT split).
        dataset = PSDAdapter(root, args.size, args.split)
    if args.max_images > 0:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MVBRDFSHRPro(args.size, 48, 96, restorer_kind=args.restorer_kind).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # strict=False: DHAN checkpoints have a different restorer layout than unet,
    # and aux buffers (e.g. num_batches_tracked) may differ.
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()

    output = Path(args.output)
    predictions = output / "predictions"
    if args.save_predictions:
        predictions.mkdir(parents=True, exist_ok=True)
    scores = {"ours_psnr": [], "ours_ssim": [], "input_psnr": [], "input_ssim": []}
    image_index = 0
    for batch in loader:
        image = batch.image.to(device, non_blocking=True)
        clean = batch.image_clean.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(
                image, use_refine=True, detach_physics=True
            )["pred"]
        scores["ours_psnr"].extend(
            batch_psnr(prediction, clean).cpu().tolist()
        )
        scores["ours_ssim"].extend(
            batch_ssim(prediction, clean).cpu().tolist()
        )
        scores["input_psnr"].extend(batch_psnr(image, clean).cpu().tolist())
        scores["input_ssim"].extend(batch_ssim(image, clean).cpu().tolist())
        for index in range(len(image)):
            if args.save_predictions:
                save_image(
                    prediction[index].float(),
                    predictions / f"{image_index:06d}.png",
                )
            image_index += 1
        if image_index % 500 == 0:
            print(f"{image_index}/{len(dataset)}", flush=True)

    result = {
        "dataset": args.dataset.upper(),
        "split": args.split,
        "n": image_index,
        "checkpoint": args.checkpoint,
        **{key: float(np.mean(values)) for key, values in scores.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
