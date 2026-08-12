"""Training entry for MVBRDF-SHR (Phase A/B/C)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from mvbrdf_shr.data import (
    DemoMultiViewDataset,
    SHIQAdapter,
    SSHRAdapter,
    SyntheticMultiViewToy,
    collate_batch,
)
from mvbrdf_shr.losses import LossWeights, total_loss
from mvbrdf_shr.models.pipeline import MVBRDFSHR


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_dataset(cfg: dict):
    name = str(cfg.get("dataset", "toy")).lower()
    res = int(cfg.get("resolution", 128))
    if name in ("toy", "synthetic", "phase_a"):
        return SyntheticMultiViewToy(
            size=int(cfg.get("dataset_size", 32)),
            resolution=res,
            n_views=int(cfg.get("n_views", 4)),
        )
    if name in ("demo", "demo_mv"):
        root = cfg.get("data_root") or cfg.get("demo_root")
        ds = DemoMultiViewDataset(
            root=root,
            split=str(cfg.get("split", "train")),
            resolution=res,
            group_views=bool(cfg.get("group_views", True)),
        )
        if len(ds) == 0:
            raise FileNotFoundError(
                f"Demo dataset empty at {ds.root}. Run: python scripts/generate_demo_data.py"
            )
        return ds
    if name == "shiq":
        root = cfg.get("data_root") or cfg.get("shiq_root")
        ds = SHIQAdapter(root=root, resolution=res, split=str(cfg.get("split", "train")))
        if len(ds) == 0:
            raise FileNotFoundError(
                f"SHIQ empty at {ds.root}. See scripts/prepare_datasets.md "
                "or use --dataset demo"
            )
        return ds
    if name == "sshr":
        root = cfg.get("data_root") or cfg.get("sshr_root")
        ds = SSHRAdapter(
            root=root,
            resolution=res,
            split=str(cfg.get("split", "train")),
            tone_corrected=bool(cfg.get("tone_corrected", True)),
        )
        if len(ds) == 0:
            raise FileNotFoundError(f"SSHR empty at {ds.root}")
        return ds
    if name == "mix":
        parts: list = [
            SyntheticMultiViewToy(
                size=int(cfg.get("dataset_size", 32)),
                resolution=res,
                n_views=int(cfg.get("n_views", 4)),
            )
        ]
        demo = DemoMultiViewDataset(
            root=cfg.get("demo_root"),
            split=str(cfg.get("split", "train")),
            resolution=res,
            group_views=True,
        )
        if len(demo) > 0:
            parts.append(demo)
        shiq = SHIQAdapter(
            root=cfg.get("shiq_root"), resolution=res, split=str(cfg.get("split", "train"))
        )
        if len(shiq) > 0:
            parts.append(shiq)
        return ConcatDataset(parts) if len(parts) > 1 else parts[0]
    raise ValueError(f"Unknown dataset: {name}")


def train(cfg: dict) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = int(cfg.get("resolution", 128))
    steps = int(cfg.get("steps", 50))
    lr = float(cfg.get("lr", 1e-3))
    batch_size = int(cfg.get("batch_size", 2))
    out_dir = Path(cfg.get("output_dir", "outputs/phase_a"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(cfg)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, drop_last=True
    )
    print(f"dataset={cfg.get('dataset', 'toy')} size={len(ds)} device={device}")

    use_gen = bool(cfg.get("use_generative", True))
    model = MVBRDFSHR(use_generative=use_gen, resolution=res).to(device)

    resume = cfg.get("resume")
    if resume:
        ckpt_path = Path(resume)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"], strict=False)
            print(f"resumed from {ckpt_path}")

    if cfg.get("freeze_physics"):
        for name, p in model.named_parameters():
            if not name.startswith("refiner"):
                p.requires_grad = False
        print("froze physics fields; training refiner only")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    weights = LossWeights(**cfg.get("loss", {}))

    model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            image = batch.image.to(device)
            clean = batch.image_clean.to(device) if batch.image_clean is not None else None
            mask = batch.mask_gt.to(device) if batch.mask_gt is not None else None
            light = batch.light_ints.to(device) if batch.light_ints is not None else None
            light0 = light[:, 0] if (light is not None and light.dim() == 2) else light

            out = model(image, light_int=light0, use_refine=use_gen)
            loss, logs = total_loss(out, image_clean=clean, mask_gt=mask, weights=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if step % max(1, steps // 10) == 0 or step == steps - 1:
                msg = " | ".join(f"{k}={v:.4f}" for k, v in logs.items())
                print(f"[{step}/{steps}] {msg}")
            step += 1

    ckpt = out_dir / "last.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt)
    print(f"saved {ckpt}")
    return ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.resume is not None:
        cfg["resume"] = args.resume
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    train(cfg)


if __name__ == "__main__":
    main()
