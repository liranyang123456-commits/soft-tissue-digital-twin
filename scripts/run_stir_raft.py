"""Reference 2D point tracker: RAFT optical flow chained over the sampled frames.

Uses the torchvision RAFT-Large weights and the same sampled, shifted
half-resolution frames as the Gaussian methods. Each start point is advected
frame to frame by bilinearly sampled forward flow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"E:\MIS_Datasets\STIRC2024\endonerf_format")
OUT = ROOT / "outputs" / "stir_tracking" / "raft"


def load(path: Path, device) -> torch.Tensor:
    image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).permute(2, 0, 1).float()
    return (image / 127.5 - 1.0).unsqueeze(0).to(device)


def sample_flow(flow: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    _, _, h, w = flow.shape
    grid = torch.stack([2 * points[:, 0] / (w - 1) - 1, 2 * points[:, 1] / (h - 1) - 1], dim=-1)
    sampled = F.grid_sample(flow, grid.view(1, -1, 1, 2), align_corners=True)
    return sampled.view(2, -1).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    index = json.loads((args.data / "index.json").read_text())
    for entry in index:
        name = entry["name"]
        target = args.output / name / "points.npz"
        if target.exists():
            continue
        meta = json.loads((args.data / name / "stir_meta.json").read_text())
        shift = np.asarray(meta["shift_half_res"])
        frames = sorted((args.data / name / "images").glob("*.png"))
        points = torch.from_numpy(
            np.asarray(meta["start_points_full_res"], dtype=np.float32) / 2.0 + shift.astype(np.float32)
        ).to(device)
        with torch.no_grad():
            previous = load(frames[0], device)
            for path in frames[1:]:
                current = load(path, device)
                flow = model(previous, current)[-1]
                points = points + sample_flow(flow, points)
                previous = current
        end = (points.cpu().numpy() - shift) * 2.0
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, end_full_res=end)
        print(f"{name}: {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
