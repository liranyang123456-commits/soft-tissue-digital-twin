"""Reproducible multi-dataset training for single-image physics-guided SHR."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.data import (
    DATASET_DOMAINS,
    DomainDataset,
    NSHAdapter,
    PSDAdapter,
    PairedDomainAugment,
    SHIQAdapter,
    SSHRAdapter,
    balanced_concat,
    collate_batch,
)
from mvbrdf_shr.losses_pro import LossWeights, total_loss
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro
from train_paired_pro import batch_metrics


def load_compatible_state(model: torch.nn.Module, state: dict) -> dict[str, list[str]]:
    """Load matching tensors while reporting new or shape-incompatible parameters."""
    source = state.get("model", state)
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    skipped = [
        key
        for key, value in source.items()
        if key not in target or target[key].shape != value.shape
    ]
    return {
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
        "skipped": skipped,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(name: str, root: str | None, size: int, split: str):
    if name == "synthetic":
        return SHIQAdapter(root, size, split)
    if name == "shiq":
        return SHIQAdapter(root, size, split)
    if name == "sshr":
        return SSHRAdapter(root, size, split, tone_corrected=True)
    if name == "nsh":
        return NSHAdapter(root, size, split)
    if name == "psd":
        return PSDAdapter(root, size, split)
    raise ValueError(name)


def split_names(name: str) -> tuple[str, str]:
    if name == "synthetic":
        return "train", "val"
    if name in {"shiq", "sshr"}:
        return "train", "train"
    if name == "psd":
        return "train", "val"
    return "train", "val"


def configure_phase(model: MVBRDFSHRPro, phase: str) -> None:
    model.requires_grad_(phase == "joint")
    if phase == "adapter":
        if model.uncertainty_fusion is not None:
            model.uncertainty_fusion.requires_grad_(True)
        model.decomposer.spec_ratio.requires_grad_(True)
        model.decomposer.mask_logit.requires_grad_(True)
        model.restorer.out_rgb.requires_grad_(True)
    elif phase != "joint":
        raise ValueError("phase must be adapter or joint")


@torch.inference_mode()
def evaluate(model, loaders, device) -> dict:
    model.eval()
    results = {}
    for name, loader in loaders.items():
        psnr, ssim, mask_psnr = [], [], []
        domain_id = DATASET_DOMAINS[name]
        for batch in loader:
            image = batch.image.to(device, non_blocking=True)
            target = batch.image_clean.to(device, non_blocking=True)
            mask = batch.mask_gt.to(device, non_blocking=True)
            domains = torch.full(
                (len(image),), domain_id, dtype=torch.long, device=device
            )
            output = model(
                image,
                use_refine=True,
                detach_physics=True,
                domain_ids=domains,
            )
            p, s = batch_metrics(output["pred"], target)
            psnr.extend(p.cpu().tolist())
            ssim.extend(s.cpu().tolist())
            error = ((output["pred"] - target).square() * mask).flatten(1).sum(1)
            denom = mask.expand_as(target).flatten(1).sum(1).clamp(min=1)
            mask_psnr.extend((-10 * torch.log10((error / denom).clamp(1e-12))).cpu().tolist())
        results[name] = {
            "n": len(psnr),
            "psnr": float(np.mean(psnr)),
            "ssim": float(np.mean(ssim)),
            "highlight_psnr": float(np.mean(mask_psnr)),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/public_multidataset.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--phases", nargs="+", choices=("adapter", "joint"))
    parser.add_argument("--resume")
    parser.add_argument("--output", default="outputs/public_multidataset")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.phases:
        cfg["phases"] = [
            phase for phase in cfg["phases"] if phase["name"] in set(args.phases)
        ]
        if not cfg["phases"]:
            raise ValueError("no configured phase matched --phases")
    seed_everything(int(cfg["seed"]))
    size = int(cfg["size"])
    names = list(args.datasets or cfg["datasets"])
    cfg["datasets"] = names
    roots = cfg.get("roots", {})
    augmentation = PairedDomainAugment(**cfg.get("augmentation", {}))

    train_domains = []
    validation_loaders = {}
    for name in names:
        train_split, val_split = split_names(name)
        train_set = build_dataset(name, roots.get(name), size, train_split)
        val_set = build_dataset(name, roots.get(name), size, val_split)
        if name in {"shiq", "sshr"}:
            count = min(int(cfg["validation_size"]), max(1, len(train_set) // 20))
            val_set = Subset(train_set, range(len(train_set) - count, len(train_set)))
            train_set = Subset(train_set, range(len(train_set) - count))
        if len(train_set) == 0 or len(val_set) == 0:
            raise RuntimeError(f"{name} has empty train/validation split")
        train_domains.append(DomainDataset(train_set, name, augmentation))
        validation_loaders[name] = DataLoader(
            DomainDataset(val_set, name),
            batch_size=int(cfg["eval_batch_size"]),
            shuffle=False,
            num_workers=int(cfg["workers"]),
            pin_memory=True,
            collate_fn=collate_batch,
        )

    combined, sampler = balanced_concat(
        train_domains,
        samples_per_epoch=int(cfg["samples_per_epoch"]),
        seed=int(cfg["seed"]),
    )
    train_loader = DataLoader(
        combined,
        batch_size=int(cfg["batch_size"]),
        sampler=sampler,
        num_workers=int(cfg["workers"]),
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MVBRDFSHRPro(
        size,
        int(cfg["base_channels"]),
        int(cfg["restorer_channels"]),
        use_uncertainty_fusion=bool(cfg.get("use_uncertainty_fusion", True)),
    ).to(device)
    load_report = {}
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        load_report = load_compatible_state(model, state)

    weights = LossWeights(**cfg.get("loss", {}))
    phases = cfg["phases"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, global_step, best = [], 0, {"mean_psnr": -1.0}
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda",
        init_scale=256.0,
        growth_interval=2000,
    )
    iterator = iter(train_loader)
    for phase_cfg in phases:
        phase = phase_cfg["name"]
        configure_phase(model, phase)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(phase_cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        steps = int(phase_cfg["steps"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        for local_step in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            image = batch.image.to(device, non_blocking=True)
            clean = batch.image_clean.to(device, non_blocking=True)
            specular = batch.specular_gt.to(device, non_blocking=True)
            mask = batch.mask_gt.to(device, non_blocking=True)
            domains = batch.domain_ids.to(device, non_blocking=True)
            model.train()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(
                    image,
                    use_refine=True,
                    detach_physics=phase == "adapter",
                    domain_ids=domains,
                )
                loss, logs = total_loss(
                    prediction,
                    image_clean=clean,
                    mask_gt=mask,
                    specular_gt=specular,
                    weights=weights,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at step {global_step + 1} in phase {phase}"
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nonfinite_parameters = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            if nonfinite_parameters:
                raise FloatingPointError(
                    f"non-finite gradient at step {global_step + 1} in phase "
                    f"{phase}: {nonfinite_parameters[:20]}"
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(cfg["grad_clip"]), error_if_nonfinite=False
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite gradient norm at step {global_step + 1} "
                    f"in phase {phase}"
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            if global_step % int(cfg["eval_every"]) == 0 or local_step == steps - 1:
                metrics = evaluate(model, validation_loaders, device)
                mean_psnr = float(np.mean([item["psnr"] for item in metrics.values()]))
                record = {
                    "step": global_step,
                    "phase": phase,
                    "train": logs,
                    "validation": metrics,
                    "mean_psnr": mean_psnr,
                }
                history.append(record)
                print(json.dumps(record), flush=True)
                state = {
                    "model": model.state_dict(),
                    "best": best,
                    "config": cfg,
                    "domains": DATASET_DOMAINS,
                }
                torch.save(state, output_dir / "latest.pt")
                if np.isfinite(mean_psnr) and mean_psnr > best["mean_psnr"]:
                    best = record
                    state["best"] = best
                    torch.save(state, output_dir / "best.pt")

    report = {"best": best, "history": history, "load_report": load_report, "config": cfg}
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
