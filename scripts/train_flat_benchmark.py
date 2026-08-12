"""Train/evaluate the single-image track on the local synthetic benchmark."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from mvbrdf_shr.losses_pro import LossWeights, total_loss
from mvbrdf_shr.metrics import psnr_torch, ssim_torch
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro
from mvbrdf_shr.types import Batch
from mvbrdf_shr.viz import save_image


def rgb(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1)


def gray(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255
    return torch.from_numpy(array).unsqueeze(0)


class FlatDataset(Dataset):
    def __init__(self, root: Path, split: str, flat: str = "flat", augment=False):
        self.root = root / flat / split
        self.paths = sorted((self.root / "input").glob("*.png"))
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = rgb(path)
        clean = rgb(self.root / "gt" / path.name)
        specular = rgb(self.root / "specular" / path.name)
        mask = gray(self.root / "mask" / path.name)
        if self.augment:
            if random.random() < 0.5:
                image, clean, specular, mask = [
                    tensor.flip(-1) for tensor in (image, clean, specular, mask)
                ]
            if random.random() < 0.5:
                image, clean, specular, mask = [
                    tensor.flip(-2) for tensor in (image, clean, specular, mask)
                ]
        return image, clean, specular, mask, path.name


def collate(samples):
    image, clean, specular, mask, names = zip(*samples)
    return (
        torch.stack(image),
        torch.stack(clean),
        torch.stack(specular),
        torch.stack(mask),
        names,
    )


@torch.no_grad()
def evaluate(model, loader, device, save_dir: Path | None = None):
    model.eval()
    psnr, ssim = [], []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for image, clean, _, _, names in loader:
        image, clean = image.to(device), clean.to(device)
        prediction = model(image, use_refine=True, detach_physics=True)["pred"]
        for index in range(len(prediction)):
            psnr.append(psnr_torch(prediction[index : index + 1], clean[index : index + 1]))
            ssim.append(ssim_torch(prediction[index : index + 1], clean[index : index + 1]))
            if save_dir is not None:
                save_image(prediction[index], save_dir / names[index])
    return {"n": len(psnr), "psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/world_benchmark_v2")
    parser.add_argument("--resume", default="outputs/shiq_mask/best.pt")
    parser.add_argument("--output", default="outputs/world_benchmark_single")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=300)
    args = parser.parse_args()
    root, output = Path(args.dataset), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = FlatDataset(root, "train", augment=True)
    val_set = FlatDataset(root, "val")
    hard_test = FlatDataset(root, "test", flat="flat_hard")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(hard_test, batch_size=args.batch_size, collate_fn=collate)
    model = MVBRDFSHRPro(128, 48, 96).to(device)
    checkpoint = Path(args.resume)
    if checkpoint.exists():
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=False)["model"]
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
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
    best = {"psnr": -1.0}
    iterator = iter(train_loader)
    for step in range(args.steps):
        try:
            image, clean, specular, mask, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            image, clean, specular, mask, _ = next(iterator)
        image, clean = image.to(device), clean.to(device)
        specular, mask = specular.to(device), mask.to(device)
        prediction = model(image, use_refine=True, detach_physics=True)
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
            metrics = evaluate(model, val_loader, device)
            print(
                f"[flat {step}] loss={logs['total']:.4f} "
                f"val={metrics['psnr']:.3f}/{metrics['ssim']:.4f}",
                flush=True,
            )
            if metrics["psnr"] > best["psnr"]:
                best = {**metrics, "step": step}
                torch.save(
                    {"model": model.state_dict(), "metrics": best},
                    output / "best.pt",
                )
            model.train()
    state = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test_metrics = evaluate(
        model, test_loader, device, root / "predictions" / "mvbrdf_shr_trained"
    )
    result = {"validation_best": best, "hard_test": test_metrics}
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

