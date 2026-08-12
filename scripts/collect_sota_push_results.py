"""Collect SOTA-push experiment results into a ranked comparison table.

Gathers every shiq_results.json produced by train_shiq_pro.py (each config x
seed) and merges them with the literature SOTA numbers, producing:
  - a ranked SHIQ table (ours configs interleaved with literature)
  - per-config median + IQR across seeds (robust to single-seed noise)
  - a clear PASS/FAIL verdict on whether each config beats the SOTA thresholds
    (NeuralDRM 34.50, DHAN-SHR 33.81)

Usage:
  python scripts/collect_sota_push_results.py
  python scripts/collect_sota_push_results.py --glob "outputs/sota_*s*/shiq_results.json"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Literature SOTA on the official SHIQ 1000-image test set.
LITERATURE = [
    {"method": "NeuralDRM", "psnr": 34.501, "ssim": 0.979, "venue": "ICCV'25", "source": "literature"},
    {"method": "JSHDR", "psnr": 34.131, "ssim": 0.86, "venue": "CVPR'21", "source": "literature"},
    {"method": "DHAN-SHR", "psnr": 33.81, "ssim": 0.975, "venue": "MM'24", "source": "literature"},
    {"method": "HighlightRNet", "psnr": 30.231, "ssim": 0.93, "venue": "MM'24", "source": "literature"},
    {"method": "TSHRNet", "psnr": 25.575, "ssim": 0.933, "venue": "ICCV'23", "source": "literature"},
]
SOTA_PSNR = 34.501  # NeuralDRM
DHAN_PSNR = 33.81   # DHAN-SHR


def collect_ours(pattern: str) -> dict[str, list[dict]]:
    """Group our experiment results by config name; collect test_full numbers."""
    by_config: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(pattern)):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Derive a config label from the output dir name (strip seed suffix).
        out_dir = Path(path).parent.name  # e.g. sota_a_dhan_only_s0
        label = out_dir.rsplit("_s", 1)[0] if "_s" in out_dir else out_dir
        seed = data.get("seed", 0)
        # Prefer the literature-comparable test_full number; fall back to strict test.
        tf = data.get("mvbrdf_pro_test_full") or data.get("mvbrdf_pro") or {}
        ts = data.get("mvbrdf_pro_test_strict") or {}
        entry = {
            "seed": seed,
            "psnr": tf.get("psnr"),
            "ssim": tf.get("ssim"),
            "test_strict_psnr": ts.get("psnr"),
            "path": str(path),
        }
        by_config.setdefault(label, []).append(entry)
    return by_config


def robust(values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"median": None, "iqr_lo": None, "iqr_hi": None, "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "iqr_lo": float(q1),
        "iqr_hi": float(q3),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="outputs/*s*/shiq_results.json")
    ap.add_argument("--output", default=str(ROOT / "outputs" / "sota_push_summary.json"))
    args = ap.parse_args()

    ours = collect_ours(args.glob)

    # Build the ranked table: literature + our configs (median test_full PSNR).
    table = list(LITERATURE)
    ours_summary = {}
    for label, entries in sorted(ours.items()):
        psnrs = [e["psnr"] for e in entries]
        ssims = [e["ssim"] for e in entries]
        p_stats = robust(psnrs)
        s_stats = robust(ssims)
        ours_summary[label] = {
            "psnr": p_stats, "ssim": s_stats,
            "seeds": [{"seed": e["seed"], "psnr": e["psnr"], "ssim": e["ssim"],
                       "test_strict_psnr": e["test_strict_psnr"]} for e in entries],
        }
        if p_stats["median"] is not None:
            table.append({
                "method": f"OURS-{label} (median)",
                "psnr": p_stats["median"],
                "ssim": s_stats["median"] if s_stats["median"] is not None else None,
                "venue": f"ours n={p_stats['n']}",
                "source": "local official SHIQ test_full",
            })

    # Sort by PSNR descending.
    table.sort(key=lambda r: -(r["psnr"] or -1))

    verdict = {}
    for label, s in ours_summary.items():
        med = s["psnr"]["median"]
        if med is None:
            continue
        verdict[label] = {
            "median_psnr": med,
            "beats_dhan": med > DHAN_PSNR,
            "beats_neuraldrm": med > SOTA_PSNR,
            "margin_vs_dhan": round(med - DHAN_PSNR, 3),
            "margin_vs_neuraldrm": round(med - SOTA_PSNR, 3),
        }

    result = {
        "shiq_ranking": table,
        "ours_configs": ours_summary,
        "sota_verdict": verdict,
        "thresholds": {"neuraldrm": SOTA_PSNR, "dhan_shr": DHAN_PSNR},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Human-readable ranked table.
    print(f"{'rank':4s} {'method':42s} {'PSNR':>8s} {'SSIM':>7s} {'source'}")
    print("-" * 80)
    for i, row in enumerate(table, 1):
        psnr = f"{row['psnr']:.3f}" if row["psnr"] is not None else "  -  "
        ssim = f"{row['ssim']:.3f}" if row["ssim"] is not None else "  -  "
        flag = " <-- SOTA" if row["psnr"] == SOTA_PSNR else ""
        ours_flag = " *OURS*" if row["method"].startswith("OURS-") else ""
        print(f"{i:4d} {row['method']:42s} {psnr:>8s} {ssim:>7s} {row['source']}{ours_flag}{flag}")

    print("\n=== SOTA verdict (median PSNR across seeds) ===")
    for label, v in verdict.items():
        d = "BEATS" if v["beats_dhan"] else "below"
        n = "BEATS" if v["beats_neuraldrm"] else "below"
        print(f"  {label}: {v['median_psnr']:.3f} dB | DHAN({d}, {v['margin_vs_dhan']:+.2f}) "
              f"| NeuralDRM({n}, {v['margin_vs_neuraldrm']:+.2f})")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
