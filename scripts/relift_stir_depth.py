"""Recompute last-frame depth for existing fused STIR tracks."""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mvbrdf_shr.world.endonerf_dataset import load_endonerf_scene
DATA = Path(r"E:\MIS_Datasets\STIRC2024\endonerf_format")
OUT = ROOT / "outputs" / "stir_tracking"
def median_depth(depth, uv, radius=2):
    h, w = depth.shape[-2:]
    u = np.clip(np.rint(uv[:, 0]), 0, w - 1).astype(int)
    v = np.clip(np.rint(uv[:, 1]), 0, h - 1).astype(int)
    out = np.zeros(len(uv), dtype=np.float64)
    for i, (ui, vi) in enumerate(zip(u, v)):
        patch = depth[max(0, vi - radius): vi + radius + 1, max(0, ui - radius): ui + radius + 1]
        vals = patch[patch > 0.01]
        out[i] = float(np.median(vals)) if vals.size else 0.0
    return out
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="present_fused_v4")
    parser.add_argument("--dest", default="present_fused_v5")
    args = parser.parse_args()
    src, dst = OUT / args.source, OUT / args.dest
    clips = sorted(x.name for x in src.iterdir() if (x / "tracks.npz").exists())
    for name in clips:
        packed = dict(np.load(src / name / "tracks.npz"))
        scene = load_endonerf_scene(DATA / name, image_scale=1.0)
        frames = sorted(scene.frames, key=lambda f: int(f.view_id))
        d0 = frames[0].depth_gt.cpu().numpy().astype(np.float64)
        dT = frames[-1].depth_gt.cpu().numpy().astype(np.float64)
        z0_med = median_depth(d0, packed["uv0"])
        zT_med = median_depth(dT, packed["uvT"])
        offset = np.where(z0_med > 0.01, packed["z0"] - z0_med, 0.0)
        stereo = zT_med + offset
        pose = packed["zT"]
        agree = np.abs(stereo - pose) <= 0.15 * np.maximum(pose, 1e-3)
        packed["zT"] = np.where((zT_med > 0.01) & agree, stereo, np.where(zT_med > 0.01, stereo, pose))
        out = dst / name
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "tracks.npz", **packed)
        src_run = src / name / "run.json"
        if src_run.exists():
            shutil.copy(src_run, out / "run.json")
        print(name, flush=True)
    print("relifted", len(clips), "clips ->", dst)
if __name__ == "__main__":
    main()
