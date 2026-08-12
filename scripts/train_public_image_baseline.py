"""Train the matched pure-image UNet under the same public-data protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.data import DomainDataset, PairedDomainAugment, balanced_concat, collate_batch
from mvbrdf_shr.losses_pro import _grad_loss, _mask_boundary, _ssim_loss, _weighted_mean
from mvbrdf_shr.models.restorer import PureUNetBaseline
from train_public_multidataset import (
    build_dataset,
    load_compatible_state,
    seed_everything,
    split_names,
)
from train_paired_pro import batch_metrics


def baseline_loss(pred, image, target, mask):
    boundary = _mask_boundary(mask)
    weight = 1 + 3 * mask
    return (
        ((pred - target).abs() * weight).mean()
        + 0.5 * _ssim_loss(pred, target)
        + 0.2 * _grad_loss(pred, target)
        + 0.2 * _weighted_mean((pred - target).abs(), boundary)
        + 0.2 * _weighted_mean((pred - image).abs(), 1 - mask)
    )


@torch.inference_mode()
def evaluate(model, loaders, device):
    model.eval()
    result = {}
    for name, loader in loaders.items():
        psnr, ssim = [], []
        for batch in loader:
            pred = model(batch.image.to(device))
            p, s = batch_metrics(pred, batch.image_clean.to(device))
            psnr.extend(p.cpu().tolist())
            ssim.extend(s.cpu().tolist())
        result[name] = {
            "psnr": float(np.mean(psnr)),
            "ssim": float(np.mean(ssim)),
            "n": len(psnr),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/public_multidataset.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--resume")
    parser.add_argument("--output", default="outputs/public_image_baseline")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["datasets"] = list(args.datasets or cfg["datasets"])
    seed_everything(int(cfg["seed"]))
    augment = PairedDomainAugment(**cfg.get("augmentation", {}))
    train_sets, val_loaders = [], {}
    for name in cfg["datasets"]:
        train_split, val_split = split_names(name)
        train = build_dataset(name, cfg["roots"].get(name), int(cfg["size"]), train_split)
        val = build_dataset(name, cfg["roots"].get(name), int(cfg["size"]), val_split)
        if name in {"shiq", "sshr"}:
            count = min(int(cfg["validation_size"]), max(1, len(train) // 20))
            val = Subset(train, range(len(train) - count, len(train)))
            train = Subset(train, range(len(train) - count))
        if len(train) == 0 or len(val) == 0:
            raise RuntimeError(f"{name} has empty train/validation split")
        train_sets.append(DomainDataset(train, name, augment))
        val_loaders[name] = DataLoader(
            DomainDataset(val, name),
            batch_size=int(cfg["eval_batch_size"]),
            collate_fn=collate_batch,
            num_workers=int(cfg["workers"]),
            pin_memory=True,
        )
    combined, sampler = balanced_concat(
        train_sets, int(cfg["samples_per_epoch"]), int(cfg["seed"])
    )
    loader = DataLoader(
        combined,
        batch_size=int(cfg["batch_size"]),
        sampler=sampler,
        collate_fn=collate_batch,
        num_workers=int(cfg["workers"]),
        pin_memory=True,
        drop_last=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PureUNetBaseline(base=int(cfg["restorer_channels"])).to(device)
    load_report = {}
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        load_report = load_compatible_state(model, state)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    iterator, best = iter(loader), {"mean_psnr": -1.0}
    history, global_step = [], 0
    for phase_cfg in cfg["phases"]:
        phase = phase_cfg["name"]
        steps = int(phase_cfg["steps"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(phase_cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        for local_step in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            image = batch.image.to(device, non_blocking=True)
            target = batch.image_clean.to(device, non_blocking=True)
            mask = batch.mask_gt.to(device, non_blocking=True)
            model.train()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(image)
                loss = baseline_loss(pred, image, target, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % int(cfg["eval_every"]) == 0 or local_step == steps - 1:
                metrics = evaluate(model, val_loaders, device)
                mean_psnr = float(np.mean([value["psnr"] for value in metrics.values()]))
                record = {
                    "step": global_step,
                    "phase": phase,
                    "mean_psnr": mean_psnr,
                    "validation": metrics,
                }
                history.append(record)
                print(json.dumps(record), flush=True)
                state = {
                    "model": model.state_dict(),
                    "model_type": "pure_unet",
                    "best": best,
                    "config": cfg,
                }
                torch.save(state, output / "latest.pt")
                if mean_psnr > best["mean_psnr"]:
                    best = record
                    state["best"] = best
                    torch.save(state, output / "best.pt")
    (output / "training_report.json").write_text(
        json.dumps(
            {
                "best": best,
                "history": history,
                "load_report": load_report,
                "config": cfg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
