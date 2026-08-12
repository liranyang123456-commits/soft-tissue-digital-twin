"""Scene-blocked paired statistics for two evaluation reports."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _scene_summaries(report: dict) -> dict[str, dict]:
    records = report.get(
        "runs", report.get("rows", report.get("scenes", []))
    )
    result = {}
    for row in records:
        if row.get("stage", "brdf") != "brdf":
            continue
        summary = row.get("summary", row.get("metrics"))
        if isinstance(summary, dict) and "scene" in row:
            result[str(row["scene"])] = summary
    return result


def _sign_test_two_sided(deltas: np.ndarray) -> float:
    nonzero = deltas[deltas != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    successes = int((nonzero > 0).sum())
    extreme = min(successes, n - successes)
    probability = sum(math.comb(n, k) for k in range(extreme + 1)) / (2**n)
    return min(1.0, 2 * probability)


def paired_statistics(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    metric: str,
    *,
    higher_is_better: bool,
    samples: int,
    seed: int,
) -> dict:
    scenes = sorted(set(baseline) & set(candidate))
    scenes = [
        scene
        for scene in scenes
        if isinstance(baseline[scene].get(metric), (int, float))
        and isinstance(candidate[scene].get(metric), (int, float))
    ]
    raw = np.asarray(
        [
            float(candidate[scene][metric]) - float(baseline[scene][metric])
            for scene in scenes
        ],
        dtype=np.float64,
    )
    improvement = raw if higher_is_better else -raw
    rng = np.random.default_rng(seed)
    if len(improvement):
        indices = rng.integers(0, len(improvement), size=(samples, len(improvement)))
        means = improvement[indices].mean(axis=1)
        ci = np.quantile(means, [0.025, 0.975]).tolist()
    else:
        ci = [None, None]
    return {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "scene_count": len(scenes),
        "scenes": scenes,
        "mean_improvement": (
            float(improvement.mean()) if len(improvement) else None
        ),
        "median_improvement": (
            float(np.median(improvement)) if len(improvement) else None
        ),
        "bootstrap_95_ci": ci,
        "sign_test_p_two_sided": (
            _sign_test_two_sided(improvement) if len(improvement) else None
        ),
        "improved_scene_count": int((improvement > 0).sum()),
        "unchanged_scene_count": int((improvement == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--higher", nargs="*", default=["psnr", "ssim"])
    parser.add_argument("--lower", nargs="*", default=["lpips"])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    baseline = _scene_summaries(
        json.loads(args.baseline.read_text(encoding="utf-8"))
    )
    candidate = _scene_summaries(
        json.loads(args.candidate.read_text(encoding="utf-8"))
    )
    results = [
        paired_statistics(
            baseline,
            candidate,
            metric,
            higher_is_better=higher,
            samples=args.bootstrap_samples,
            seed=args.seed + index,
        )
        for index, (metric, higher) in enumerate(
            [(metric, True) for metric in args.higher]
            + [(metric, False) for metric in args.lower]
        )
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "baseline": str(args.baseline),
                "candidate": str(args.candidate),
                "bootstrap_unit": "scene",
                "seed": args.seed,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
