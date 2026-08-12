"""Append group-blocked paired statistics to an existing public-suite report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate_public_suite import (
    _group_id,
    dataset_for,
    paired_group_statistics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report_path = args.output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cfg = report["protocol"]
    rows_by_checkpoint = {}
    for name in report["checkpoints"]:
        with (args.output / f"{name}_per_image.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        datasets = {
            dataset_name: dataset_for(
                dataset_name,
                cfg.get("roots", {}).get(dataset_name),
                int(cfg["size"]),
                cfg["splits"][dataset_name],
            )
            for dataset_name in cfg["datasets"]
        }
        numeric = {
            "psnr",
            "ssim",
            "highlight_psnr",
            "non_highlight_psnr",
            "boundary_psnr",
            "decomposition_l1",
        }
        for row in rows:
            row["group"] = _group_id(
                datasets[row["dataset"]], int(row["index"])
            )
            for metric in numeric:
                row[metric] = float(row[metric])
        rows_by_checkpoint[name] = rows

    names = list(report["checkpoints"])
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            key = f"{left}_minus_{right}"
            report["comparisons"][key]["paired_group_statistics"] = (
                paired_group_statistics(
                    rows_by_checkpoint[left], rows_by_checkpoint[right]
                )
            )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
