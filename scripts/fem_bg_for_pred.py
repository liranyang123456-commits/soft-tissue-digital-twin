"""Improve E_bg for force-driven holdout prediction (resumable)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fem_validation_suite import (  # noqa: E402
    case_tensors,
    invert_two_param_fem,
    load_case,
    predict_load_case,
)

DATA = Path(r"E:\Diff_Rending_Re_3D\dataset\ion_ct_synthetic_mechanics60")
HOLDOUT = ("press_right", "shear_x", "shear_y")
BASE = ROOT / "outputs" / "fem_validation_60" / "scenarios"
OUT = ROOT / "outputs" / "fem_bg_for_pred"
OUT.mkdir(parents=True, exist_ok=True)

# Focus on the two recipes that already beat paper20 on scenario_000/001.
RECIPES = {
    "iters40": dict(n_iters=40, soft_width=0.02, frames=[1, 2, 4]),
    "iters40_allf": dict(n_iters=40, soft_width=0.02, frames=[1, 2, 3, 4, 5, 6]),
    "bg_only40": dict(n_iters=40, soft_width=0.02, frames=[1, 2, 4], optimize_incl=False),
}


def eval_pred(scen, E_bg, E_incl):
    errs = []
    for case in HOLDOUT:
        data = case_tensors(load_case(scen, case))
        errs.append(
            predict_load_case(
                data, E_bg, E_incl, soft_width=None, against="true"
            )["err_relative"]
        )
    return float(np.median(errs))


def run_scenario(scen: Path) -> dict:
    saved = json.loads((BASE / f"{scen.name}.json").read_text())
    row = {"scenario": scen.name, "status": "completed"}
    row["paper20"] = {
        "bg_err": saved["E_bg_rel_err"],
        "c_err": saved["contrast_rel_err"],
        "pred": eval_pred(scen, saved["E_bg_est"], saved["E_incl_est"]),
        "pred_homog": eval_pred(scen, saved["E_bg_est"], saved["E_bg_est"]),
        "E_bg_est": saved["E_bg_est"],
        "E_incl_est": saved["E_incl_est"],
    }
    p20 = row["paper20"]
    print(
        "  paper20 bg={:.1f}% pred={:.1f}% homog={:.1f}%".format(
            100 * p20["bg_err"], 100 * p20["pred"], 100 * p20["pred_homog"]
        ),
        flush=True,
    )
    for name, kw0 in RECIPES.items():
        kw = copy.deepcopy(kw0)
        frames = kw.pop("frames")
        data = case_tensors(load_case(scen, "press_left"), frames=tuple(frames))
        r = invert_two_param_fem(data, frames=frames, **kw)
        bg = abs(r["E_bg_est"] - r["E_bg_gt"]) / r["E_bg_gt"]
        c = abs(r["contrast_est"] - r["contrast_gt"]) / max(abs(r["contrast_gt"]), 1e-12)
        pred = eval_pred(scen, r["E_bg_est"], r["E_incl_est"])
        pred_h = eval_pred(scen, r["E_bg_est"], r["E_bg_est"])
        row[name] = {
            "bg_err": bg,
            "c_err": c,
            "pred": pred,
            "pred_homog": pred_h,
            "E_bg_est": r["E_bg_est"],
            "E_incl_est": r["E_incl_est"],
        }
        print(
            "  {:14s} bg={:.1f}% pred={:.1f}% homog={:.1f}%".format(
                name, 100 * bg, 100 * pred, 100 * pred_h
            ),
            flush=True,
        )
    return row


def aggregate(rows: list[dict]) -> dict:
    names = ["paper20", *RECIPES.keys()]
    summary = {}
    for name in names:
        if not all(name in r for r in rows):
            continue
        summary[name] = {
            "n": len(rows),
            "median_bg": float(np.median([r[name]["bg_err"] for r in rows])),
            "median_pred": float(np.median([r[name]["pred"] for r in rows])),
            "median_pred_homog": float(np.median([r[name]["pred_homog"] for r in rows])),
        }
    return summary


def main():
    scens = sorted(p for p in DATA.iterdir() if p.is_dir())[:8]
    rows = []
    for scen in scens:
        dest = OUT / "scenarios" / f"{scen.name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            row = json.loads(dest.read_text(encoding="utf-8"))
            if row.get("status") == "completed" and all(k in row for k in RECIPES):
                print(scen.name[-12:], "skip", flush=True)
                rows.append(row)
                continue
        print(scen.name[-12:], flush=True)
        row = run_scenario(scen)
        dest.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)

    summary = aggregate(rows)
    (OUT / "summary.json").write_text(
        json.dumps({"summary": summary, "scenario_ids": [r["scenario"] for r in rows]}, indent=2),
        encoding="utf-8",
    )
    print("=== MEDIAN ===")
    for key, value in summary.items():
        print(
            "{:14s} n={} bg {:.1f}%  pred {:.1f}%  homog {:.1f}%".format(
                key,
                value["n"],
                100 * value["median_bg"],
                100 * value["median_pred"],
                100 * value["median_pred_homog"],
            )
        )


if __name__ == "__main__":
    main()
