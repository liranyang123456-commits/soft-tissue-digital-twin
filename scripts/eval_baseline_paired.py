"""Evaluate DHAN-SHR / TSHRNet on SHIQ, SSHR, or PSD using local adapters.

All metrics are computed against the same GT the Ours track uses.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvbrdf_shr.data import (  # noqa: E402
    NSHAdapter,
    PSDAdapter,
    SHIQAdapter,
    SSHRAdapter,
    collate_batch,
)


def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred.float().clamp(0, 1) - target.float().clamp(0, 1)).square()
    mse = mse.flatten(1).mean(1).clamp(min=1e-12)
    return -10 * torch.log10(mse)


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    pred = F.pad(pred, (5, 5, 5, 5), mode="reflect")
    target = F.pad(target, (5, 5, 5, 5), mode="reflect")
    mp = F.avg_pool2d(pred, 11, 1)
    mt = F.avg_pool2d(target, 11, 1)
    vp = F.avg_pool2d(pred.square(), 11, 1) - mp.square()
    vt = F.avg_pool2d(target.square(), 11, 1) - mt.square()
    cov = F.avg_pool2d(pred * target, 11, 1) - mp * mt
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mp * mt + c1) * (2 * cov + c2)) / (
        (mp.square() + mt.square() + c1) * (vp + vt + c2) + 1e-12
    )
    return score.flatten(1).mean(1)


def load_dhan(device: torch.device) -> torch.nn.Module:
    dhan_root = ROOT / "baselines_ext" / "DHAN-SHR"
    sys.path.insert(0, str(dhan_root))
    from models import Model  # type: ignore

    weight = dhan_root / "weights" / "model.pth"
    model = Model()
    state = torch.load(weight, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = OrderedDict(
            (k.removeprefix("module."), v) for k, v in state.items()
        )
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def load_tshrnet(device: torch.device) -> list[torch.nn.Module]:
    tshr_root = ROOT / "baselines_ext" / "TSHRNet"
    sys.path.insert(0, str(tshr_root))
    from models.UNet import UNet  # type: ignore

    ckpt_dir = tshr_root / "checkpoints_mix_SSHR_SHIQ_PSD"

    def _load(ch: int, name: str) -> torch.nn.Module:
        model = UNet(input_channels=ch, output_channels=3)
        state = torch.load(ckpt_dir / name, map_location="cpu", weights_only=True)
        state = OrderedDict(
            (k.removeprefix("module."), v) for k, v in state.items()
        )
        model.load_state_dict(state)
        return model.to(device).eval()

    return [
        _load(3, "UNet1_60.pth"),
        _load(3, "UNet2_60.pth"),
        _load(6, "UNet3_60.pth"),
        _load(9, "UNet4_60.pth"),
    ]


@torch.inference_mode()
def predict_dhan(model, image: torch.Tensor) -> torch.Tensor:
    return model(image).clamp(0, 1)


def _next_tshr_size(h: int, w: int) -> tuple[int, int]:
    # TSHRNet UNet uses 4x4 kernels deep in the encoder; keep >=256 and multiple of 32.
    def _fix(v: int) -> int:
        v = max(256, v)
        return ((v + 31) // 32) * 32

    return _fix(h), _fix(w)


@torch.inference_mode()
def predict_tshrnet(networks, image: torch.Tensor) -> torch.Tensor:
    # TSHRNet official training uses [-1, 1] inputs and power-of-two-ish sizes.
    _, _, h, w = image.shape
    th, tw = _next_tshr_size(h, w)
    x = image
    if th != h or tw != w:
        x = F.interpolate(x, (th, tw), mode="bilinear", align_corners=False)
    x = x * 2 - 1
    diffuse = networks[0](x)
    specular = networks[1](x)
    refined = networks[2](torch.cat([diffuse, x], dim=1))
    prediction = networks[3](torch.cat([refined, specular, x], dim=1))
    out = prediction.add(1).mul(0.5).clamp(0, 1)
    if out.shape[-2:] != (h, w):
        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=False)
    return out


def build_dataset(name: str, root: Path, size: int, split: str, max_images: int):
    if name == "shiq":
        ds = SHIQAdapter(root, size, split)
    elif name == "sshr":
        ds = SSHRAdapter(root, size, split, tone_corrected=True)
    elif name == "nsh":
        ds = NSHAdapter(root, size, split)
    else:
        ds = PSDAdapter(root, size, split)
    if max_images > 0:
        ds = Subset(ds, range(min(max_images, len(ds))))
    return ds


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dhan_shr", "tshrnet"), required=True)
    parser.add_argument("--dataset", choices=("shiq", "sshr", "nsh", "psd"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--size", type=int, default=0, help="0 = method default")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    if args.size <= 0:
        # Match Ours SHIQ protocol (200); TSHRNet internally pads to >=256.
        args.size = 200 if args.dataset == "shiq" else 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(
        args.dataset, Path(args.root), args.size, args.split, args.max_images
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    if args.method == "dhan_shr":
        model = load_dhan(device)
        predict = lambda img: predict_dhan(model, img)
        ckpt = str(ROOT / "baselines_ext" / "DHAN-SHR" / "weights" / "model.pth")
    else:
        networks = load_tshrnet(device)
        predict = lambda img: predict_tshrnet(networks, img)
        ckpt = str(ROOT / "baselines_ext" / "TSHRNet" / "checkpoints_mix_SSHR_SHIQ_PSD")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pred_dir = output / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(exist_ok=True)

    psnrs, ssims = [], []
    index = 0
    for batch in tqdm(loader, desc=f"{args.method}/{args.dataset}"):
        image = batch.image.to(device)
        target = batch.image_clean.to(device)
        if image.shape[-1] != args.size or image.shape[-2] != args.size:
            image = F.interpolate(image, (args.size, args.size), mode="bilinear", align_corners=False)
            target = F.interpolate(target, (args.size, args.size), mode="bilinear", align_corners=False)
        pred = predict(image)
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
        p = batch_psnr(pred, target)
        s = batch_ssim(pred, target)
        psnrs.extend(p.cpu().tolist())
        ssims.extend(s.cpu().tolist())
        if args.save_predictions:
            from mvbrdf_shr.viz import save_image

            for b in range(pred.shape[0]):
                save_image(pred[b], pred_dir / f"{index:06d}.png")
                index += 1

    metrics = {
        "method": args.method,
        "dataset": args.dataset,
        "split": args.split,
        "size": args.size,
        "checkpoint": ckpt,
        "n": len(psnrs),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "protocol": "local adapter GT; same PSNR/SSIM as Ours eval",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
