"""3D reference trackers on STIR: chained RAFT lifted by stereo depth.

The official 3D query points (``gt3d.json``) are advected by torchvision
RAFT-Large over the same sampled half-resolution frames as the Gaussian
methods. The end points are lifted to 3D in two ways:

    raft_sgm     semi-global-matching depth of the last sampled frame
                 (the depth supervising the Gaussian methods)
    raft_stereo  disparity of the official STIR RAFT-Stereo model on the
                 last original stereo frame, back-projected with the
                 rectified geometry as in the STIR 3D baseline

All positions are in millimetres in the left camera frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

ROOT = Path(__file__).resolve().parents[1]
STIR = Path(r"E:\MIS_Datasets\STIRC2024")
DATA = STIR / "endonerf_format"
OUT = ROOT / "outputs" / "stir_tracking"
STEREO_MODEL = STIR / "STIRMetrics" / "src" / "models" / "iraftstereo_rvc_ontrackingpointsSTIR.pt"


def load(path: Path, device) -> torch.Tensor:
    image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).permute(2, 0, 1).float()
    return (image / 127.5 - 1.0).unsqueeze(0).to(device)


def sample_flow(flow: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    _, _, h, w = flow.shape
    grid = torch.stack([2 * points[:, 0] / (w - 1) - 1, 2 * points[:, 1] / (h - 1) - 1], dim=-1)
    return F.grid_sample(flow, grid.view(1, -1, 1, 2), align_corners=True).view(2, -1).T


def chain_raft(model, frames: list[Path], points: torch.Tensor, device) -> np.ndarray:
    with torch.no_grad():
        previous = load(frames[0], device)
        for path in frames[1:]:
            current = load(path, device)
            flow = model(previous, current)[-1]
            points = points + sample_flow(flow, points)
            previous = current
    return points.cpu().numpy()


def sgm_depth(depth_png: Path, uv_half: np.ndarray, window: int = 2) -> np.ndarray:
    depth = np.asarray(Image.open(depth_png), dtype=np.float32)
    h, w = depth.shape
    values = []
    for u, v in uv_half:
        x, y = int(round(u)), int(round(v))
        patch = depth[max(0, y - window): min(h, y + window + 1), max(0, x - window): min(w, x + window + 1)]
        patch = patch[patch > 0]
        values.append(float(np.median(patch)) if patch.size else np.nan)
    return np.asarray(values)


def last_frame(video: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    frame = None
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frame = image
    capture.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def stereo_disparity(model, left: np.ndarray, right: np.ndarray, points_full: np.ndarray, pad_full: float, device) -> np.ndarray:
    resize = lambda image: F.interpolate(
        torch.from_numpy(image).permute(2, 0, 1)[None].float().to(device) / 255.0, size=(512, 640),
        mode="bilinear", align_corners=False,
    )
    left_t, right_t = resize(left), resize(right)
    pad = int(pad_full) // 2
    padded = F.pad(left_t, (pad, 0, 0, 0), mode="replicate")[:, :, :, :-pad] if pad > 0 else left_t
    points = torch.from_numpy(points_full / 2.0).float().to(device)[None]
    locations = torch.stack((points[:, :, 0] + pad, points[:, :, 1]), dim=-1)
    with torch.no_grad():
        disparity = model(locations, padded, right_t)
    return (-disparity * 2.0).squeeze(0).reshape(-1).cpu().numpy()


def backproject(points_full: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    x = (points_full[:, 0] - K[0, 2]) * depth / K[0, 0]
    y = (points_full[:, 1] - K[1, 2]) * depth / K[1, 1]
    return np.stack([x, y, depth], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    gt = json.loads((OUT / "gt3d.json").read_text())
    index = {entry["name"]: entry for entry in json.loads((DATA / "index.json").read_text())}
    flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    stereo_model = torch.jit.load(str(STEREO_MODEL)).to(device)
    for name, labels in gt.items():
        target = OUT / "raft3d" / name / "points3d.npz"
        if target.exists():
            continue
        meta = json.loads((DATA / name / "stir_meta.json").read_text())
        shift = np.asarray(meta["shift_half_res"], dtype=np.float32)
        K = np.asarray(labels["K_full"], dtype=np.float64)
        start_full = np.asarray(labels["start_2d"], dtype=np.float32)
        frames = sorted((DATA / name / "images").glob("*.png"))
        end_half = chain_raft(flow_model, frames, torch.from_numpy(start_full / 2.0 + shift).to(device), device)
        end_full = (end_half - shift) * 2.0

        depth_sgm = sgm_depth(sorted((DATA / name / "depth").glob("*.png"))[-1], end_half)

        left_dir = STIR / index[name]["clip"]
        right_dir = left_dir.parent.parent / left_dir.parent.name.replace("left", "right", 1) / left_dir.name
        calib = json.loads((left_dir.parent.parent / "calib.json").read_text())
        pad_full = float(np.asarray(calib["rightcameramat"])[0][2] - np.asarray(calib["leftcameramat"])[0][2])
        baseline_mm = abs(float(np.asarray(calib["translation"])[0]) * 1000.0)
        left = last_frame(next(left_dir.glob("frames/*.mp4")))
        right = last_frame(next(right_dir.glob("frames/*.mp4")))
        disparity = stereo_disparity(stereo_model, left, right, end_full, pad_full, device)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth_stereo = np.where(disparity > 0, K[0, 0] * baseline_mm / disparity, np.nan)

        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target, end_full=end_full,
            raft_sgm=backproject(end_full, depth_sgm, K),
            raft_stereo=backproject(end_full, depth_stereo, K),
        )
        print(f"{name}: {len(start_full)} points, depth sgm {np.nanmedian(depth_sgm):.0f} "
              f"stereo {np.nanmedian(depth_stereo):.0f} mm", flush=True)


if __name__ == "__main__":
    main()
