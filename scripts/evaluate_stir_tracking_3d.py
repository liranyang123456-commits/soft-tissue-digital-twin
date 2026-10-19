"""STIR 3D endpoint evaluation (official protocol, millimetres).

Query points are the stereo-matched start tattoos of ``gt3d.json``. Gaussian
methods move each query by the anchored primitives exactly as in the 2D
evaluation and take the depth of the anchored primitives in the last frame;
the end point is back-projected with the full-resolution left intrinsics.
Reference trackers provide 3D end points directly. Errors are distances to
the nearest labelled 3D end point, with accuracy at 2, 4, 8, 16, and 32 mm.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"E:\MIS_Datasets\STIRC2024\endonerf_format")
OUT = ROOT / "outputs" / "stir_tracking"
THRESHOLDS = (2, 4, 8, 16, 32)
RADII = (3.0, 6.0, 12.0)


def anchor3d(start_half: np.ndarray, tracks: dict, depth_band: float = 0.05):
    uv0, z0, uvT, zT, opacity = tracks["uv0"], tracks["z0"], tracks["uvT"], tracks["zT"], tracks["opacity"]
    usable = (z0 > 0) & (zT > 0) & (opacity > 0.05) & np.isfinite(uvT).all(1)
    ends, depths = [], []
    for point in start_half:
        distance = np.linalg.norm(uv0 - point, axis=1)
        candidates = np.empty(0, dtype=int)
        for radius in RADII:
            candidates = np.flatnonzero(usable & (distance < radius))
            if len(candidates) >= 3:
                break
        if len(candidates) == 0:
            ends.append(point)
            depths.append(np.nan)
            continue
        front = candidates[z0[candidates] <= np.percentile(z0[candidates], 10) * (1 + depth_band)]
        weight = opacity[front] * np.exp(-0.5 * (distance[front] / (radius / 2)) ** 2)
        weight = weight / weight.sum()
        ends.append(point + (weight[:, None] * (uvT[front] - uv0[front])).sum(0))
        depths.append(float((weight * zT[front]).sum()))
    return np.asarray(ends), np.asarray(depths)


def backproject(points_full: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    x = (points_full[:, 0] - K[0, 2]) * depth / K[0, 0]
    y = (points_full[:, 1] - K[1, 2]) * depth / K[1, 1]
    return np.stack([x, y, depth], axis=-1)


def errors3d(predicted: np.ndarray, end: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    predicted = np.where(np.isfinite(predicted).all(1, keepdims=True), predicted, fallback)
    distance, _ = cKDTree(end).query(predicted)
    return distance


def summarize(errors: np.ndarray) -> dict:
    accuracy = {f"acc@{t}mm": float(np.mean(errors <= t)) for t in THRESHOLDS}
    return {"points": int(errors.size), **accuracy, "acc_avg": float(np.mean(list(accuracy.values()))),
            "error_median_mm": float(np.median(errors)), "error_mean_mm": float(errors.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+",
                        default=["present", "present_flow", "endogaussian", "raft_sgm", "raft_stereo"])
    parser.add_argument("--clips-file", type=Path)
    parser.add_argument("--tag", default="all")
    args = parser.parse_args()
    gt = json.loads((OUT / "gt3d.json").read_text())
    if args.clips_file:
        keep = set(json.loads(args.clips_file.read_text()))
        gt = {k: v for k, v in gt.items() if k in keep}

    def available(method: str) -> bool:
        if method.startswith("raft_"):
            return all((OUT / "raft3d" / n / "points3d.npz").exists() for n in gt)
        return all((OUT / method / n / "tracks.npz").exists() for n in gt)

    methods = [m for m in args.methods if available(m)]
    pooled = {m: [] for m in ["control", *methods]}
    rows = []
    for name, labels in gt.items():
        meta = json.loads((DATA / name / "stir_meta.json").read_text())
        shift = np.asarray(meta["shift_half_res"])
        K = np.asarray(labels["K_full"])
        start_2d, start_3d = np.asarray(labels["start_2d"]), np.asarray(labels["start_3d"])
        end_3d = np.asarray(labels["end_3d"])
        row = {"clip": name, "points": len(start_3d)}
        control = errors3d(start_3d, end_3d, start_3d)
        pooled["control"].append(control)
        row["control_mm"] = float(np.median(control))
        for method in methods:
            if method.startswith("raft_"):
                predicted = np.load(OUT / "raft3d" / name / "points3d.npz")[method]
            else:
                tracks = dict(np.load(OUT / method / name / "tracks.npz"))
                end_half, depth = anchor3d(start_2d / 2.0 + shift, tracks)
                predicted = backproject((end_half - shift) * 2.0, depth, K)
            errors = errors3d(predicted, end_3d, start_3d)
            pooled[method].append(errors)
            row[f"{method}_mm"] = float(np.median(errors))
        rows.append(row)

    summary = {"clips": len(gt), "methods": {m: summarize(np.concatenate(v)) for m, v in pooled.items()}}
    (OUT / f"summary3d_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    with (OUT / f"per_clip3d_{args.tag}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for method, values in summary["methods"].items():
        print(f"{method:14s} n={values['points']:4d}  avg acc {values['acc_avg']:.3f}  "
              f"median {values['error_median_mm']:.2f} mm  "
              + "  ".join(f"{t}mm {values[f'acc@{t}mm']:.2f}" for t in THRESHOLDS))


if __name__ == "__main__":
    main()
