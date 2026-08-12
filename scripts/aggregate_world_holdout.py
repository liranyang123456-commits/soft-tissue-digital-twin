"""Aggregate per-view world holdout metrics across MANY scenes.

Why this exists
---------------
``eval_world_ablation.py`` evaluates a single scene (scene_0070, 3 holdout
views) and reports the *mean* PSNR/SSIM. On scene_0070 one of the three views
reaches 57.24 dB (because that view is nearly diffuse-free), which inflates the
3-view mean to 36.82 dB. Reporting that single mean as a headline number is
misleading.

This script collects per-view numbers from *all* available scene ablation
directories (e.g. ``outputs/world_ablation_scene_00NN/``) and reports robust
statistics — median, IQR, min, max, and the full per-view distribution — so the
reported number is no longer dominated by a single easy view.

Usage
-----
    python scripts/aggregate_world_holdout.py \
        --glob "outputs/world_ablation_scene_*" \
        --output outputs/world_holdout_aggregate.json

After each scene has been trained (stage 2.3) and evaluated with
``eval_world_ablation.py --world-checkpoint .../scene_00NN_strict/last.pt
--output outputs/world_ablation_scene_00NN``, this script turns those per-scene
results into a single cross-scene summary with honest statistics.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def robust_stats(values: list[float]) -> dict:
    """Median + IQR + min/max + count, robust to outliers (unlike mean)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "median": None, "mean": None, "iqr_lo": None,
                "iqr_hi": None, "min": None, "max": None, "values": []}
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "iqr_lo": float(q1),
        "iqr_hi": float(q3),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "values": [float(v) for v in arr],
    }


def collect(pattern: str) -> dict:
    """Read every ablation_metrics.json matching the glob; gather per-view data."""
    variants: dict[str, dict[str, list[float]]] = {}
    scenes_found: list[str] = []
    for path in sorted(glob.glob(pattern)):
        if not path.endswith("ablation_metrics.json"):
            candidate = Path(path) / "ablation_metrics.json"
            if candidate.exists():
                path = str(candidate)
            else:
                continue
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # derive scene id from path
        p = Path(path).parent.name  # e.g. world_ablation_scene_0070
        scenes_found.append(p)
        for variant, info in data.get("variants", {}).items():
            variants.setdefault(variant, {"psnr": [], "ssim": [], "per_scene": []})
            for pv in info.get("per_view", []):
                variants[variant]["psnr"].append(pv["psnr"])
                variants[variant]["ssim"].append(pv["ssim"])
            variants[variant]["per_scene"].append(
                {"scene": p, "psnr": info.get("psnr"), "ssim": info.get("ssim")}
            )

    summary = {}
    for variant, data in variants.items():
        summary[variant] = {
            "psnr": robust_stats(data["psnr"]),
            "ssim": robust_stats(data["ssim"]),
            "per_scene": data["per_scene"],
        }
    return {"scenes_evaluated": scenes_found, "n_scenes": len(scenes_found),
            "n_views_total": len(summary.get("ours_phys", {}).get("psnr", {}).get("values", [])),
            "variants": summary}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="outputs/world_ablation_scene_*/ablation_metrics.json",
                    help="Glob pattern for per-scene ablation_metrics.json files.")
    ap.add_argument("--output", default="outputs/world_holdout_aggregate.json")
    args = ap.parse_args()

    result = collect(args.glob)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Human-readable summary focusing on the key variants.
    print(f"Scenes evaluated: {result['n_scenes']} | total holdout views: {result['n_views_total']}")
    print(f"{'variant':24s} {'median PSNR':>12s} {'IQR':>14s} {'n':>4s}")
    print("-" * 60)
    for variant in ["ours_phys", "ours_pro", "fusion_full", "fusion_analytic",
                    "ours_refined", "input"]:
        if variant in result["variants"]:
            st = result["variants"][variant]["psnr"]
            if st["median"] is not None:
                print(f"{variant:24s} {st['median']:12.3f} "
                      f"[{st['iqr_lo']:.2f},{st['iqr_hi']:.2f}] {st['n']:4d}")
    print(f"\nWrote {out}")
    print("NOTE: median + IQR are robust to the single-easy-view inflation that")
    print("      affected the scene_0070-only 3-view mean (36.82 dB).")


if __name__ == "__main__":
    main()
