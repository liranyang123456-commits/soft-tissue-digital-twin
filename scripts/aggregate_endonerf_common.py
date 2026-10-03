"""Aggregate common-protocol EndoNeRF metrics and draw a comparison figure."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def collect(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/*/evaluation/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "scene": report["scene"],
                "method": report["method"],
                **report["metrics"],
                "path": str(path),
            }
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    methods = sorted({row["method"] for row in rows})
    metrics = ("full_psnr", "full_ssim", "tissue_psnr", "tissue_ssim")
    result = {"per_run": rows, "per_method": {}}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        total = sum(row["n"] for row in selected)
        result["per_method"][method] = {
            "scenes": len(selected),
            "frames": total,
            **{
                metric: float(
                    sum(row[metric] * row["n"] for row in selected) / max(total, 1)
                )
                for metric in metrics
            },
        }
    return result


def plot(summary: dict, output: Path) -> None:
    methods = list(summary["per_method"])
    if not methods:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=170)
    x = np.arange(len(methods))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        [summary["per_method"][m]["full_psnr"] for m in methods],
        width,
        label="full image",
    )
    axes[0].bar(
        x + width / 2,
        [summary["per_method"][m]["tissue_psnr"] for m in methods],
        width,
        label="tissue",
    )
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_xticks(x, methods, rotation=20, ha="right")
    axes[0].legend()
    axes[0].set_title("Strict 1-in-8 holdout")

    axes[1].bar(
        x - width / 2,
        [summary["per_method"][m]["full_ssim"] for m in methods],
        width,
        label="full image",
    )
    axes[1].bar(
        x + width / 2,
        [summary["per_method"][m]["tissue_ssim"] for m in methods],
        width,
        label="tissue",
    )
    axes[1].set_ylabel("SSIM")
    axes[1].set_xticks(x, methods, rotation=20, ha="right")
    axes[1].legend()
    axes[1].set_title("Same held-out frames and masks")
    figure.tight_layout()
    figure.savefig(output / "endonerf_common_comparison.png", bbox_inches="tight")
    figure.savefig(output / "endonerf_common_comparison.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs" / "endonerf_common",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "endonerf_common",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = collect(args.root)
    summary = aggregate(rows)
    (args.output / "common_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if rows:
        with (args.output / "common_per_run.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    plot(summary, args.output)
    print(json.dumps(summary["per_method"], indent=2))


if __name__ == "__main__":
    main()
