"""Build present_fused_v7: gated uv (v6) + SGM/pose consensus depth."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_stir_tracking import DATA, OUT
from scripts.fuse_stir_gated import neighbourhood, prefer_query

def sgm(depth_png, uv, window=2):
    depth = np.asarray(Image.open(depth_png), dtype=np.float32)
    h, w = depth.shape
    out = []
    for u, v in np.atleast_2d(uv):
        x, y = int(round(float(u))), int(round(float(v)))
        patch = depth[max(0, y - window): min(h, y + window + 1), max(0, x - window): min(w, x + window + 1)]
        patch = patch[patch > 0]
        out.append(float(np.median(patch)) if patch.size else 0.0)
    return np.asarray(out)

def main():
    dest = OUT / "present_fused_v7"
    dest.mkdir(parents=True, exist_ok=True)
    index = json.loads((DATA / "index.json").read_text())
    for entry in index:
        name = entry["name"]
        meta = json.loads((DATA / name / "stir_meta.json").read_text())
        shift = np.asarray(meta["shift_half_res"])
        start_full = np.asarray(meta["start_points_full_res"], dtype=np.float64)
        start_half = start_full / 2.0 + shift
        fused = dict(np.load(OUT / "present_fused_v4" / name / "tracks.npz"))
        fused5 = dict(np.load(OUT / "present_fused_v5" / name / "tracks.npz"))
        query = np.load(OUT / "query_refresh_v1" / name / "points.npz")["end_full_res"]
        query_tr = dict(np.load(OUT / "query_refresh_v1" / name / "tracks.npz"))
        depth_last = sorted((DATA / name / "depth").glob("*.png"))[-1]
        pred, zT, z0, flags = [], [], [], []
        for i, point in enumerate(start_half):
            st = neighbourhood(point, fused)
            st5 = neighbourhood(point, fused5)
            fused_full = (st["end"] - shift) * 2.0 if st is not None else start_full[i]
            use_q = prefer_query(st, start_full[i], fused_full, query[i])
            if use_q:
                end_h = query[i] / 2.0 + shift
                z_pose = float(query_tr["zT"][i])
                flags.append(1)
            else:
                end_h = (st5 or st)["end"]
                z_pose = float((st5 or st)["z"])
                flags.append(0)
            z_s = float(sgm(depth_last, end_h)[0])
            if z_s > 0.01 and abs(z_s - z_pose) <= 0.15 * max(abs(z_pose), 1e-3):
                z = 0.5 * (z_s + z_pose)
            elif z_s > 0.01:
                z = z_s
            else:
                z = z_pose
            pred.append((end_h - shift) * 2.0)
            zT.append(z)
            z0i = float(query_tr["z0"][i])
            z0.append(z0i if z0i > 0.01 else z)
        pred = np.asarray(pred)
        out = dest / name
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "tracks.npz",
            uv0=start_half,
            z0=np.asarray(z0),
            uvT=pred / 2.0 + shift,
            zT=np.asarray(zT),
            opacity=np.ones(len(pred)),
            gated_query=np.asarray(flags, dtype=np.uint8),
        )
        np.savez_compressed(out / "points.npz", end_full_res=pred)
        print(name, "switch", sum(flags), "/", len(flags), flush=True)
    print("wrote", dest)

if __name__ == "__main__":
    main()