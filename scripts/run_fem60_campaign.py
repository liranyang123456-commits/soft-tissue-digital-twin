"""Resumable 60-scenario FEM inversion campaign.

Each scenario is written atomically to an individual JSON file. Re-running
the command skips completed scenarios, which makes the campaign safe to
resume after interruption. Aggregation is deterministic and can be run
without launching additional inversions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATA = Path(r"E:\Diff_Rending_Re_3D\dataset\ion_ct_synthetic_mechanics60")
DEFAULT_OUTPUT = ROOT / "outputs" / "fem_validation_60"
HOLDOUT_CASES = ("press_right", "shear_x", "shear_y")


def select_scenarios(
    data_root: Path,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
    max_scenarios: int | None = None,
) -> list[Path]:
    """Return a deterministic lexical shard of scenario directories."""
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= index < num_shards")
    scenarios = sorted(path for path in data_root.iterdir() if path.is_dir())
    scenarios = scenarios[shard_index::num_shards]
    return scenarios[:max_scenarios] if max_scenarios is not None else scenarios


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_scenario(
    scenario_dir: Path,
    *,
    n_iters: int,
    with_holdout: bool,
) -> dict:
    # Imported lazily so aggregation and unit tests do not require the sibling
    # FEM repository.
    from scripts.run_fem_validation_suite import (
        case_tensors,
        invert_two_param_fem,
        load_case,
        predict_load_case,
    )

    started = time.time()
    training = case_tensors(load_case(scenario_dir, "press_left"))
    inversion = invert_two_param_fem(training, n_iters=n_iters)
    row = {
        "scenario": scenario_dir.name,
        "status": "completed",
        "protocol": {
            "training_load": "press_left",
            "frames": training["frames"],
            "iterations": n_iters,
            "measured_force": True,
            "surface_motion": "noisy_observation",
        },
        "E_bg_gt": inversion["E_bg_gt"],
        "E_bg_est": inversion["E_bg_est"],
        "E_bg_rel_err": abs(inversion["E_bg_est"] - inversion["E_bg_gt"])
        / inversion["E_bg_gt"],
        "E_incl_gt": inversion["E_incl_gt"],
        "E_incl_est": inversion["E_incl_est"],
        "E_incl_rel_err": abs(inversion["E_incl_est"] - inversion["E_incl_gt"])
        / inversion["E_incl_gt"],
        "contrast_gt": inversion["contrast_gt"],
        "contrast_est": inversion["contrast_est"],
        "contrast_rel_err": abs(inversion["contrast_est"] - inversion["contrast_gt"])
        / max(abs(inversion["contrast_gt"]), 1e-12),
        "objective_final": inversion["loss_curve"][-1],
        "runtime_s": inversion["runtime_s"],
    }

    if with_holdout:
        held_out = []
        for case in HOLDOUT_CASES:
            test = case_tensors(load_case(scenario_dir, case))
            estimated = predict_load_case(
                test, inversion["E_bg_est"], inversion["E_incl_est"]
            )
            oracle = predict_load_case(test, test["E_bg"], test["E_incl"])
            held_out.append(
                {
                    "case": case,
                    "estimated_rel_err": estimated["err_relative"],
                    "oracle_rel_err": oracle["err_relative"],
                    "estimated_mean_err_mm": estimated["err_mean_mm"],
                    "oracle_mean_err_mm": oracle["err_mean_mm"],
                }
            )
        row["held_out"] = held_out

    row["wall_runtime_s"] = time.time() - started
    return row


def load_completed(scenario_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not scenario_dir.exists():
        return rows
    for path in sorted(scenario_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") == "completed":
            rows.append(row)
    return rows


def _describe(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate(rows: list[dict], expected_scenarios: int = 60) -> dict:
    """Aggregate completed per-scenario records."""
    rows = sorted(rows, key=lambda row: row["scenario"])
    summary = {
        "protocol": "press_left inversion; measured force; noisy surface motion",
        "expected_scenarios": expected_scenarios,
        "completed_scenarios": len(rows),
        "complete": len(rows) == expected_scenarios,
        "scenario_ids": [row["scenario"] for row in rows],
        "E_bg_rel_err": _describe(row["E_bg_rel_err"] for row in rows),
        "E_incl_rel_err": _describe(row["E_incl_rel_err"] for row in rows),
        "contrast_rel_err": _describe(row["contrast_rel_err"] for row in rows),
        "runtime_s": _describe(row["runtime_s"] for row in rows),
    }
    holdout_rows = [
        (item["case"], item["estimated_rel_err"], item["oracle_rel_err"])
        for row in rows
        for item in row.get("held_out", [])
    ]
    if holdout_rows:
        summary["held_out"] = {
            case: {
                "estimated_rel_err": _describe(
                    estimated for name, estimated, _ in holdout_rows if name == case
                ),
                "oracle_rel_err": _describe(
                    oracle for name, _, oracle in holdout_rows if name == case
                ),
            }
            for case in HOLDOUT_CASES
        }
    return summary


def write_aggregate(output: Path, expected_scenarios: int = 60) -> dict:
    rows = load_completed(output / "scenarios")
    summary = aggregate(rows, expected_scenarios=expected_scenarios)
    _atomic_json(output / "fem60_summary.json", summary)
    with (output / "fem60_per_scenario.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = (
            "scenario",
            "E_bg_gt",
            "E_bg_est",
            "E_bg_rel_err",
            "E_incl_gt",
            "E_incl_est",
            "E_incl_rel_err",
            "contrast_gt",
            "contrast_est",
            "contrast_rel_err",
            "runtime_s",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--with-holdout", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(args.data_root)
    args.output.mkdir(parents=True, exist_ok=True)
    scenario_output = args.output / "scenarios"
    scenario_output.mkdir(exist_ok=True)

    if not args.aggregate_only:
        scenarios = select_scenarios(
            args.data_root,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            max_scenarios=args.max_scenarios,
        )
        for index, scenario in enumerate(scenarios, start=1):
            destination = scenario_output / f"{scenario.name}.json"
            if destination.exists() and not args.overwrite:
                print(f"[{index}/{len(scenarios)}] {scenario.name}: skipped")
                continue
            print(f"[{index}/{len(scenarios)}] {scenario.name}: running", flush=True)
            try:
                row = _run_scenario(
                    scenario,
                    n_iters=args.iters,
                    with_holdout=args.with_holdout,
                )
            except Exception as exc:
                _atomic_json(
                    args.output / "errors" / f"{scenario.name}.json",
                    {
                        "scenario": scenario.name,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            _atomic_json(destination, row)
            print(
                f"  E_bg error={100*row['E_bg_rel_err']:.2f}% "
                f"contrast error={100*row['contrast_rel_err']:.2f}% "
                f"runtime={row['runtime_s']:.1f}s",
                flush=True,
            )

    summary = write_aggregate(args.output)
    print(
        f"completed {summary['completed_scenarios']}/"
        f"{summary['expected_scenarios']} scenarios"
    )


if __name__ == "__main__":
    main()
