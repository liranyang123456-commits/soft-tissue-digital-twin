"""Visualize the complete or interim FEM60 inversion campaign."""
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


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            {
                key: (value if key == "scenario" else float(value))
                for key, value in row.items()
            }
            for row in csv.DictReader(stream)
        ]


def plot(rows: list[dict], summary: dict, output: Path) -> None:
    if not rows:
        raise ValueError("no completed FEM scenarios")
    output.mkdir(parents=True, exist_ok=True)
    bg_gt = np.asarray([row["E_bg_gt"] for row in rows]) / 1000.0
    bg_est = np.asarray([row["E_bg_est"] for row in rows]) / 1000.0
    bg_error = 100 * np.asarray([row["E_bg_rel_err"] for row in rows])
    incl_error = 100 * np.asarray([row["E_incl_rel_err"] for row in rows])
    contrast_error = 100 * np.asarray([row["contrast_rel_err"] for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=180)
    axis = axes[0, 0]
    points = axis.scatter(bg_gt, bg_est, c=bg_error, cmap="viridis", s=28)
    limits = [min(bg_gt.min(), bg_est.min()), max(bg_gt.max(), bg_est.max())]
    axis.plot(limits, limits, "k--", linewidth=1)
    axis.set_xlabel("true background modulus (kPa)")
    axis.set_ylabel("estimated background modulus (kPa)")
    axis.set_title("Background modulus")
    figure.colorbar(points, ax=axis, label="relative error (%)")

    axis = axes[0, 1]
    axis.boxplot(
        [bg_error, incl_error, contrast_error],
        labels=["background E", "inclusion E", "contrast"],
        showfliers=True,
    )
    axis.set_ylabel("relative error (%)")
    axis.set_title("Error distributions")
    axis.tick_params(axis="x", rotation=15)

    axis = axes[1, 0]
    axis.scatter(bg_gt, bg_error, color="#4C72B0", s=28)
    axis.set_xlabel("true background modulus (kPa)")
    axis.set_ylabel("background-modulus error (%)")
    axis.set_title("Error versus stiffness")

    axis = axes[1, 1]
    thresholds = np.asarray([1, 2, 5, 10, 20, 30, 50])
    fraction = [(bg_error <= threshold).mean() for threshold in thresholds]
    axis.plot(thresholds, fraction, marker="o")
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("error threshold (%)")
    axis.set_ylabel("fraction of scenarios")
    axis.set_title("Background-modulus success curve")
    axis.grid(alpha=0.25)

    complete = summary.get("complete", False)
    figure.suptitle(
        f"FEM inversion campaign: {len(rows)}/{summary.get('expected_scenarios', 60)} "
        f"scenarios ({'complete' if complete else 'interim'})"
    )
    figure.tight_layout()
    figure.savefig(output / "fem60_summary.png", bbox_inches="tight")
    figure.savefig(output / "fem60_summary.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs" / "fem_validation_60",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "fem_validation_60",
    )
    args = parser.parse_args()
    rows = load_rows(args.input / "fem60_per_scenario.csv")
    summary = json.loads(
        (args.input / "fem60_summary.json").read_text(encoding="utf-8")
    )
    plot(rows, summary, args.output)
    print(args.output / "fem60_summary.png")


if __name__ == "__main__":
    main()
