"""Train confidence gating between Pro and world-space physical diffuse."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..metrics import psnr_torch, ssim_torch
from ..models.pipeline_pro import MVBRDFSHRPro
from ..viz import save_image
from .fusion import HybridMVBRDFSHR
from .pipeline import WorldGaussianBRDFPipeline
from .scene import GaussianBRDFField
from .train import load_dataset


def load_world(checkpoint: str, device: torch.device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = state["config"]
    dataset = load_dataset(cfg)
    scene = GaussianBRDFField.from_point_cloud(
        state["points"].to(device), state["colors"].to(device)
    )
    pipeline = WorldGaussianBRDFPipeline(
        scene,
        len(dataset),
        lighting_mode=str(cfg.get("lighting_mode", "sh")),
        use_refiner=bool(cfg.get("use_refiner", True)),
        refiner_base=int(cfg.get("refiner_base", 96)),
        chunk_size=int(cfg.get("chunk_size", 32)),
        raster_backend=str(cfg.get("raster_backend", "auto")),
        shared_lighting=bool(cfg.get("shared_lighting", False)),
        background_color=cfg.get("background_color"),
    ).to(device)
    pipeline.load_state_dict(state["model"], strict=False)
    pipeline.eval().requires_grad_(False)
    return pipeline, dataset, state


@torch.no_grad()
def evaluate(
    hybrid: HybridMVBRDFSHR,
    world: WorldGaussianBRDFPipeline,
    dataset,
    frame_ids: list[int],
    device: torch.device,
    output: Path,
) -> dict:
    hybrid.eval()
    output.mkdir(parents=True, exist_ok=True)
    scores = {
        "hybrid_psnr": [],
        "hybrid_ssim": [],
        "world_psnr": [],
        "single_psnr": [],
        "gate_mean": [],
    }
    for frame_id in frame_ids:
        frame = dataset[frame_id]
        image = frame.image.to(device).unsqueeze(0)
        target = frame.diffuse_gt.to(device).unsqueeze(0)
        world_output = world(
            frame.camera.to(device), frame_id, image=image, refine=False
        )
        result = hybrid(image, world_output, detach_backbones=True)
        scores["hybrid_psnr"].append(psnr_torch(result["pred"], target))
        scores["hybrid_ssim"].append(ssim_torch(result["pred"], target))
        scores["world_psnr"].append(
            psnr_torch(result["world_diffuse"], target)
        )
        scores["single_psnr"].append(
            psnr_torch(result["single_pred"], target)
        )
        scores["gate_mean"].append(float(result["gate"].mean()))
        save_image(result["pred"][0], output / f"{frame_id:03d}_hybrid.png")
        save_image(result["gate"][0], output / f"{frame_id:03d}_gate.png")
    return {
        "n": len(frame_ids),
        **{key: float(np.mean(value)) for key, value in scores.items()},
    }


def train(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world, dataset, world_state = load_world(args.world_checkpoint, device)
    single = MVBRDFSHRPro(128, 48, 96).to(device)
    single_state = torch.load(
        args.single_checkpoint, map_location=device, weights_only=False
    )
    single.load_state_dict(single_state["model"], strict=False)
    single.eval().requires_grad_(False)
    hybrid = HybridMVBRDFSHR(single=single, gate_hidden=args.gate_hidden).to(
        device
    )
    train_ids = list(world_state.get("train_ids", range(len(dataset))))
    holdout_ids = list(world_state.get("holdout_ids", []))
    if not holdout_ids:
        raise ValueError("world checkpoint has no strict holdout split")
    optimizer = torch.optim.AdamW(
        hybrid.fusion_parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )
    history = []
    for step in range(args.steps):
        frame_id = random.choice(train_ids)
        frame = dataset[frame_id]
        image = frame.image.to(device).unsqueeze(0)
        target = frame.diffuse_gt.to(device).unsqueeze(0)
        with torch.no_grad():
            world_output = world(
                frame.camera.to(device), frame_id, image=image, refine=False
            )
        hybrid.gate.train()
        hybrid.correction.train()
        result = hybrid(image, world_output, detach_backbones=True)
        world_error = (result["world_diffuse"] - target).abs().mean(1, keepdim=True)
        single_error = (result["single_pred"] - target).abs().mean(1, keepdim=True)
        gate_target = (world_error < single_error).to(result["gate"])
        reconstruction = F.l1_loss(result["pred"], target)
        mse = F.mse_loss(result["pred"], target)
        gate_loss = F.binary_cross_entropy(result["gate"], gate_target)
        correction_anchor = F.l1_loss(result["pred"], result["fused"].detach())
        # Reconstruction quality is primary; the oracle source label is only
        # a weak stabilizer because either branch can be locally biased.
        loss = reconstruction + 4 * mse + 0.01 * gate_loss + 0.05 * correction_anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(hybrid.fusion_parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "reconstruction": float(reconstruction.detach()),
                "gate_loss": float(gate_loss.detach()),
                "gate_mean": float(result["gate"].detach().mean()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gate": hybrid.gate.state_dict(),
            "correction": hybrid.correction.state_dict(),
            "world_checkpoint": args.world_checkpoint,
            "single_checkpoint": args.single_checkpoint,
            "train_ids": train_ids,
            "holdout_ids": holdout_ids,
        },
        output / "fusion.pt",
    )
    metrics = evaluate(
        hybrid,
        world,
        dataset,
        holdout_ids,
        device,
        output / "holdout_predictions",
    )
    result = {"holdout": metrics, "history": history}
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--single-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gate-hidden", type=int, default=48)
    parser.add_argument("--log-every", type=int, default=100)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
