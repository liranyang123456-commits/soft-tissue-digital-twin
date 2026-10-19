"""Expand bg_only40 E_bg + holdout prediction to all 60 FEM scenarios (resumable)."""
from __future__ import annotations

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
OUT = ROOT / "outputs" / "fem_bg_only40_60"
OUT.mkdir(parents=True, exist_ok=True)
PILOT = ROOT / "outputs" / "fem_bg_for_pred" / "scenarios"


def eval_pred(scen, E_bg, E_incl):
    by_case = {}
    errs = []
    for case in HOLDOUT:
        data = case_tensors(load_case(scen, case))
        e = predict_load_case(
            data, E_bg, E_incl, soft_width=None, against="true"
        )["err_relative"]
        by_case[case] = float(e)
        errs.append(e)
    return float(np.median(errs)), by_case


def run_one(scen: Path) -> dict:
    saved = json.loads((BASE / f"{scen.name}.json").read_text())
    paper_pred, paper_by = eval_pred(scen, saved["E_bg_est"], saved["E_incl_est"])
    data = case_tensors(load_case(scen, "press_left"), frames=(1, 2, 4))
    r = invert_two_param_fem(
        data, frames=[1, 2, 4], n_iters=40, soft_width=0.02, optimize_incl=False
    )
    bg = abs(r["E_bg_est"] - r["E_bg_gt"]) / r["E_bg_gt"]
    pred, by = eval_pred(scen, r["E_bg_est"], r["E_incl_est"])
    pred_h, _ = eval_pred(scen, r["E_bg_est"], r["E_bg_est"])
    return {
        "scenario": scen.name,
        "status": "completed",
        "paper20": {
            "bg_err": saved["E_bg_rel_err"],
            "c_err": saved["contrast_rel_err"],
            "pred": paper_pred,
            "pred_by_case": paper_by,
            "E_bg_est": saved["E_bg_est"],
            "E_incl_est": saved["E_incl_est"],
        },
        "bg_only40": {
            "bg_err": bg,
            "c_err": abs(r["contrast_est"] - r["contrast_gt"])
            / max(abs(r["contrast_gt"]), 1e-12),
            "pred": pred,
            "pred_homog": pred_h,
            "pred_by_case": by,
            "E_bg_est": r["E_bg_est"],
            "E_incl_est": r["E_incl_est"],
        },
    }


def seed_from_pilot() -> None:
    dest_dir = OUT / "scenarios"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not PILOT.exists():
        return
    for f in PILOT.glob("*.json"):
        dest = dest_dir / f.name
        if dest.exists():
            continue
        row = json.loads(f.read_text(encoding="utf-8"))
        if "bg_only40" not in row or row.get("status") != "completed":
            continue
        slim = {
            "scenario": row["scenario"],
            "status": "completed",
            "paper20": row["paper20"],
            "bg_only40": row["bg_only40"],
            "seeded_from_pilot": True,
        }
        dest.write_text(json.dumps(slim, indent=2), encoding="utf-8")
        print("seed", row["scenario"][-12:], flush=True)


def main():
    seed_from_pilot()
    scens = sorted(p for p in DATA.iterdir() if p.is_dir())
    rows = []
    for scen in scens:
        dest = OUT / "scenarios" / f"{scen.name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            row = json.loads(dest.read_text(encoding="utf-8"))
            if row.get("status") == "completed" and "bg_only40" in row:
                print(scen.name[-12:], "skip", flush=True)
                rows.append(row)
                continue
        print(scen.name[-12:], flush=True)
        row = run_one(scen)
        dest.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(
            "  paper20 bg={:.1f}% pred={:.1f}% | bg_only40 bg={:.1f}% pred={:.1f}%".format(
                100 * row["paper20"]["bg_err"],
                100 * row["paper20"]["pred"],
                100 * row["bg_only40"]["bg_err"],
                100 * row["bg_only40"]["pred"],
            ),
            flush=True,
        )
        rows.append(row)

    summary = {}
    for name in ("paper20", "bg_only40"):
        summary[name] = {
            "n": len(rows),
            "median_bg": float(np.median([r[name]["bg_err"] for r in rows])),
            "median_pred": float(np.median([r[name]["pred"] for r in rows])),
        }
        if all("pred_by_case" in r[name] for r in rows):
            for case in HOLDOUT:
                summary[name][f"median_{case}"] = float(
                    np.median([r[name]["pred_by_case"][case] for r in rows])
                )
    (OUT / "summary.json").write_text(
        json.dumps({"summary": summary, "n": len(rows)}, indent=2), encoding="utf-8"
    )
    print("=== MEDIAN ===", flush=True)
    for key, value in summary.items():
        print(
            "{:14s} n={} bg {:.1f}%  pred {:.1f}%".format(
                key, value["n"], 100 * value["median_bg"], 100 * value["median_pred"]
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()