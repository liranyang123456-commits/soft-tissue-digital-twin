"""Common STIR endpoint evaluation for Gaussian trackers and reference trackers.

Gaussian methods (present fixed-identity tracker, EndoGaussian) are anchored
identically: for every labelled start point, the primitives whose first-frame
projection lies within a small radius and in the front depth layer are
selected, and the point is moved by their opacity- and distance-weighted
mean projected displacement between the first and the last frame. Point
trackers provide end positions directly. Errors follow the STIR protocol:
distance from each predicted point to the nearest labelled end point, with
accuracy at 4, 8, 16, 32, and 64 px (full resolution).
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
THRESHOLDS = (4, 8, 16, 32, 64)
RADII = (3.0, 6.0, 12.0)


def anchor(start: np.ndarray, tracks: dict, depth_band: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    uv0, z0, uvT, opacity = tracks["uv0"], tracks["z0"], tracks["uvT"], tracks["opacity"]
    usable = (z0 > 0) & (tracks["zT"] > 0) & (opacity > 0.05) & np.isfinite(uvT).all(1)
    predictions, anchored = [], []
    for point in start:
        distance = np.linalg.norm(uv0 - point, axis=1)
        candidates = np.empty(0, dtype=int)
        for radius in RADII:
            candidates = np.flatnonzero(usable & (distance < radius))
            if len(candidates) >= 3:
                break
        if len(candidates) == 0:
            predictions.append(point)
            anchored.append(False)
            continue
        depth = z0[candidates]
        front = candidates[depth <= np.percentile(depth, 10) * (1 + depth_band)]
        weight = opacity[front] * np.exp(-0.5 * (distance[front] / (radius / 2)) ** 2)
        motion = (weight[:, None] * (uvT[front] - uv0[front])).sum(0) / weight.sum()
        predictions.append(point + motion)
        anchored.append(True)
    return np.asarray(predictions), np.asarray(anchored)


def endpoint_errors(predicted: np.ndarray, end: np.ndarray) -> np.ndarray:
    distance, _ = cKDTree(end).query(predicted)
    return distance


def summarize(errors: np.ndarray) -> dict:
    accuracy = {f"acc@{t}": float(np.mean(errors <= t)) for t in THRESHOLDS}
    return {
        "points": int(errors.size),
        **accuracy,
        "acc_avg": float(np.mean(list(accuracy.values()))),
        "epe_mean": float(errors.mean()),
        "epe_median": float(np.median(errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--methods", nargs="+", default=["present", "present_cv", "endogaussian", "raft"])
    parser.add_argument("--clips-file", type=Path, help="JSON list of clip names to evaluate")
    parser.add_argument("--tag", default="all")
    args = parser.parse_args()
    index = json.loads((args.data / "index.json").read_text())
    if args.clips_file:
        keep = set(json.loads(args.clips_file.read_text()))
        index = [e for e in index if e["name"] in keep]

    available = [
        m for m in args.methods
        if all((args.output / m / e["name"] / ("tracks.npz" if m != "raft" else "points.npz")).exists() for e in index)
    ]
    clips = [e["name"] for e in index]
    pooled = {m: [] for m in ["control", *available]}
    rows = []
    for name in clips:
        meta = json.loads((args.data / name / "stir_meta.json").read_text())
        shift = np.asarray(meta["shift_half_res"])
        start_full = np.asarray(meta["start_points_full_res"], dtype=np.float64)
        end_full = np.asarray(meta["end_points_full_res"], dtype=np.float64)
        start_half = start_full / 2.0 + shift
        row = {"clip": name, "source": meta["clip"], "points": len(start_full),
               "video_frames": meta["video_frames"]}
        control = endpoint_errors(start_full, end_full)
        pooled["control"].append(control)
        row["control_epe"] = float(np.median(control))
        for method in available:
            if method == "raft":
                predicted_full = np.load(args.output / method / name / "points.npz")["end_full_res"]
                anchored = np.ones(len(predicted_full), dtype=bool)
            else:
                tracks = dict(np.load(args.output / method / name / "tracks.npz"))
                predicted_half, anchored = anchor(start_half, tracks)
                predicted_full = (predicted_half - shift) * 2.0
            errors = endpoint_errors(predicted_full, end_full)
            pooled[method].append(errors)
            row[f"{method}_epe"] = float(np.median(errors))
            row[f"{method}_anchored"] = float(anchored.mean())
        rows.append(row)

    summary = {
        "clips": len(clips),
        "methods": {m: summarize(np.concatenate(v)) for m, v in pooled.items() if v},
        "per_clip_median_epe": {
            m: float(np.median([r[f"{m}_epe"] for r in rows])) for m in ["control", *available]
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    with (args.output / f"per_clip_{args.tag}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
