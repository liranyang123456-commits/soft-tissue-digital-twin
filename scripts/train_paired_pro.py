"""Fine-tune MVBRDF-SHR Pro on SHIQ or SSHR paired training data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from mvbrdf_shr.data import (
    DATASET_DOMAINS,
    NSHAdapter,
    PSDAdapter,
    SHIQAdapter,
    SSHRAdapter,
    collate_batch,
)
from mvbrdf_shr.losses_pro import LossWeights, total_loss
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro


def batch_metrics(prediction: torch.Tensor, target: torch.Tensor):
    prediction = prediction.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    mse = (prediction - target).square().flatten(1).mean(1).clamp(min=1e-12)
    psnr = -10 * torch.log10(mse)
    p = F.pad(prediction, (5, 5, 5, 5), mode="reflect")
    t = F.pad(target, (5, 5, 5, 5), mode="reflect")
    mp, mt = F.avg_pool2d(p, 11, 1), F.avg_pool2d(t, 11, 1)
    vp = F.avg_pool2d(p.square(), 11, 1) - mp.square()
    vt = F.avg_pool2d(t.square(), 11, 1) - mt.square()
    cov = F.avg_pool2d(p * t, 11, 1) - mp * mt
    ssim = ((2 * mp * mt + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mp.square() + mt.square() + 0.01**2) * (vp + vt + 0.03**2)
        + 1e-12
    )
    return psnr, ssim.flatten(1).mean(1)


@torch.inference_mode()
def evaluate(model, loader, device, domain_id: int = 0) -> dict:
    model.eval()
    psnr, ssim = [], []
    for batch in loader:
        image = batch.image.to(device, non_blocking=True)
        target = batch.image_clean.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            domains = torch.full(
                (len(image),), domain_id, dtype=torch.long, device=device
            )
            prediction = model(
                image,
                use_refine=True,
                detach_physics=True,
                domain_ids=domains,
            )["pred"]
        p, s = batch_metrics(prediction, target)
        psnr.extend(p.cpu().tolist())
        ssim.extend(s.cpu().tolist())
    return {"n": len(psnr), "psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("shiq", "sshr", "nsh", "psd"), required=True
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Adapt only output heads for fast cross-dataset calibration",
    )
    args = parser.parse_args()

    if args.dataset == "shiq":
        train_set = SHIQAdapter(args.root, args.size, "train")
        test_set = SHIQAdapter(args.root, args.size, "test")
        val_set = Subset(test_set, range(min(args.val_size, len(test_set))))
    elif args.dataset == "sshr":
        train_set = SSHRAdapter(
            args.root, args.size, "train", tone_corrected=True
        )
        validation_count = min(args.val_size, max(1, len(train_set) // 20))
        val_set = Subset(
            train_set, range(len(train_set) - validation_count, len(train_set))
        )
        train_set = Subset(train_set, range(len(train_set) - validation_count))
    elif args.dataset == "psd":
        # Public PSD archive has only PSD_Test; use group-level 70/10/20 split.
        train_set = PSDAdapter(args.root, args.size, "train")
        val_set = PSDAdapter(args.root, args.size, "val")
    else:
        train_set = NSHAdapter(args.root, args.size, "train")
        val_set = NSHAdapter(args.root, args.size, "val")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(args.batch_size, 8),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MVBRDFSHRPro(args.size, 48, 96).to(device)
    state = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    if args.head_only:
        model.requires_grad_(False)
        model.decomposer.spec_ratio.requires_grad_(True)
        model.decomposer.mask_logit.requires_grad_(True)
        model.restorer.out_rgb.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )
    weights = LossWeights(
        photo=0.1,
        diffuse=0.8,
        refine=1.5,
        refine_mse=8.0,
        refine_ssim=0.5,
        refine_grad=0.2,
        specular=2.0,
        coarse=1.0,
        mask_pred=0.3,
        mask_dice=0.2,
        hl_weight=2.0,
        smooth=0.001,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best = {"psnr": -1.0}
    iterator = iter(train_loader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        image = batch.image.to(device, non_blocking=True)
        clean = batch.image_clean.to(device, non_blocking=True)
        specular = batch.specular_gt.to(device, non_blocking=True)
        mask = batch.mask_gt.to(device, non_blocking=True)
        model.train()
        domains = torch.full(
            (len(image),),
            DATASET_DOMAINS[args.dataset],
            dtype=torch.long,
            device=device,
        )
        prediction = model(
            image,
            use_refine=True,
            detach_physics=True,
            domain_ids=domains,
        )
        loss, logs = total_loss(
            prediction,
            image_clean=clean,
            mask_gt=mask,
            weights=weights,
            specular_gt=specular,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            metrics = evaluate(
                model, val_loader, device, DATASET_DOMAINS[args.dataset]
            )
            print(
                f"[{args.dataset} {step}] loss={logs['total']:.4f} "
                f"val={metrics['psnr']:.3f}/{metrics['ssim']:.4f}",
                flush=True,
            )
            if metrics["psnr"] > best["psnr"]:
                best = {**metrics, "step": step}
                torch.save(
                    {"model": model.state_dict(), "metrics": best},
                    output / "best.pt",
                )
    (output / "training_metrics.json").write_text(
        json.dumps({"best": best}, indent=2), encoding="utf-8"
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
