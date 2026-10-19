"""STIR query tracker: RAFT chain + reference refresh.

Fallback only: fuse_stir_gated.py may replace an identity neighbourhood with
these tracks when that neighbourhood is inconsistent. This is not the
paper-wide STIR method.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"E:\MIS_Datasets\STIRC2024\endonerf_format")
OUT = ROOT / "outputs" / "stir_tracking"

def load(path, device):
    image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).permute(2, 0, 1).float()
    return (image / 127.5 - 1.0).unsqueeze(0).to(device)

def sample_flow(flow, points):
    _, _, h, w = flow.shape
    grid = torch.stack([2 * points[:, 0] / (w - 1) - 1, 2 * points[:, 1] / (h - 1) - 1], dim=-1)
    return F.grid_sample(flow, grid.view(1, -1, 1, 2), align_corners=True).view(2, -1).T

def sgm_depth(depth_png, uv, window=2):
    depth = np.asarray(Image.open(depth_png), dtype=np.float32)
    h, w = depth.shape
    values = []
    for u, v in uv:
        x, y = int(round(float(u))), int(round(float(v)))
        patch = depth[max(0, y - window): min(h, y + window + 1), max(0, x - window): min(w, x + window + 1)]
        patch = patch[patch > 0]
        values.append(float(np.median(patch)) if patch.size else 0.0)
    return np.asarray(values)

def advect(model, src, dst, uv, fb_abs=1.0, fb_rel=0.05):
    forward = model(src, dst)[-1]
    backward = model(dst, src)[-1]
    h, w = forward.shape[-2:]
    motion = sample_flow(forward, uv)
    uv_new = uv + motion
    clamp = uv_new.clone()
    clamp[:, 0].clamp_(0, w - 1)
    clamp[:, 1].clamp_(0, h - 1)
    back = sample_flow(backward, clamp)
    fb = (motion + back).norm(dim=-1)
    ok = fb <= fb_abs + fb_rel * motion.norm(dim=-1)
    inside = (uv_new[:, 0] >= 0) & (uv_new[:, 0] <= w - 1) & (uv_new[:, 1] >= 0) & (uv_new[:, 1] <= h - 1)
    return uv_new, fb, ok & inside

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", default="query_refresh_v1")
    parser.add_argument("--refresh", type=int, default=5)
    parser.add_argument("--jump", type=float, default=12.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    index = json.loads((DATA / "index.json").read_text())
    dest = OUT / args.method_name
    for entry in index:
        name = entry["name"]
        target = dest / name / "tracks.npz"
        if target.exists():
            print(name, "skip", flush=True)
            continue
        meta = json.loads((DATA / name / "stir_meta.json").read_text())
        shift = np.asarray(meta["shift_half_res"], dtype=np.float32)
        frames = sorted((DATA / name / "images").glob("*.png"))
        uv0 = torch.from_numpy(np.asarray(meta["start_points_full_res"], dtype=np.float32) / 2.0 + shift).to(device)
        uv = uv0.clone()
        with torch.no_grad():
            images = [load(p, device) for p in frames]
            for i in range(1, len(images)):
                uv_chain, fb_chain, ok_chain = advect(model, images[i - 1], images[i], uv)
                uv_new, fb_error = uv_chain, fb_chain
                if i % args.refresh == 0:
                    uv_ref, fb_ref, ok_ref = advect(model, images[0], images[i], uv0)
                    jump = (uv_ref - uv_chain).norm(dim=-1)
                    prefer = ok_ref & ((~ok_chain) | ((fb_ref <= fb_chain) & (jump <= args.jump)))
                    uv_new = torch.where(prefer[:, None], uv_ref, uv_chain)
                    fb_error = torch.where(prefer, fb_ref, fb_chain)
                    ok_chain = ok_chain | prefer
                lost = ~ok_chain
                uv = torch.where(lost[:, None], uv_chain, uv_new)
        uv_np = uv.cpu().numpy()
        uv0_np = uv0.cpu().numpy()
        depths = sorted((DATA / name / "depth").glob("*.png"))
        z0 = sgm_depth(depths[0], uv0_np)
        zT = sgm_depth(depths[-1], uv_np)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            uv0=uv0_np.astype(np.float64),
            z0=z0.astype(np.float64),
            uvT=uv_np.astype(np.float64),
            zT=zT.astype(np.float64),
            opacity=np.ones(len(uv_np), dtype=np.float64),
        )
        end_full = (uv_np - shift) * 2.0
        np.savez_compressed(target.parent / "points.npz", end_full_res=end_full)
        print(name, len(frames), "frames", len(uv_np), "pts", flush=True)

if __name__ == "__main__":
    main()