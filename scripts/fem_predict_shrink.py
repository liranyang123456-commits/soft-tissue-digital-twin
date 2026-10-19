"""Tune prediction: hard E-field + contrast shrinkage on training load."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fem_validation_suite import case_tensors, load_case, predict_load_case, invert_two_param_fem

DATA = Path(r"E:\Diff_Rending_Re_3D\dataset\ion_ct_synthetic_mechanics60")
EST = ROOT / "outputs" / "fem_validation_60" / "scenarios"
HOLDOUT = ("press_right", "shear_x", "shear_y")


def shrink(E_bg, E_incl, w):
    return E_bg, E_bg + w * (E_incl - E_bg)


def score_train(left, E_bg, E_incl, w, soft_width):
    Eb, Ei = shrink(E_bg, E_incl, w)
    # hold out last training frame of press_left as proxy
    return predict_load_case(
        left, Eb, Ei, frame=left["frames"][-1],
        soft_width=soft_width, against="observed",
    )["err_relative"]


def main():
    paths = sorted(EST.glob("*.json"))
    ws = [0.0, 0.25, 0.5, 0.75, 1.0]
    # Collect predictions
    buckets = {
        "base_soft": [], "hard": [], "shrink_cv": [], "oracle_hard": [],
        "base_true": [], "hard_true": [], "shrink_true": [], "oracle_true": [],
    }
    per_case = {c: {k: [] for k in ["base_soft", "hard", "shrink_cv", "oracle_hard"]} for c in HOLDOUT}
    for i, path in enumerate(paths):
        est = json.loads(path.read_text())
        scen = DATA / est["scenario"]
        left = case_tensors(load_case(scen, "press_left"))
        E_bg, E_incl = est["E_bg_est"], est["E_incl_est"]
        E_bg_gt, E_incl_gt = est["E_bg_gt"], est["E_incl_gt"]
        # choose w on training load frame
        best_w, best_s = 1.0, 1e9
        for w in ws:
            s = score_train(left, E_bg, E_incl, w, soft_width=None)
            if s < best_s:
                best_s, best_w = s, w
        Eb_s, Ei_s = shrink(E_bg, E_incl, best_w)
        for case in HOLDOUT:
            data = case_tensors(load_case(scen, case))
            vals = {
                "base_soft": predict_load_case(data, E_bg, E_incl, soft_width=0.02, against="observed")["err_relative"],
                "hard": predict_load_case(data, E_bg, E_incl, soft_width=None, against="observed")["err_relative"],
                "shrink_cv": predict_load_case(data, Eb_s, Ei_s, soft_width=None, against="observed")["err_relative"],
                "oracle_hard": predict_load_case(data, E_bg_gt, E_incl_gt, soft_width=None, against="observed")["err_relative"],
                "base_true": predict_load_case(data, E_bg, E_incl, soft_width=0.02, against="true")["err_relative"],
                "hard_true": predict_load_case(data, E_bg, E_incl, soft_width=None, against="true")["err_relative"],
                "shrink_true": predict_load_case(data, Eb_s, Ei_s, soft_width=None, against="true")["err_relative"],
                "oracle_true": predict_load_case(data, E_bg_gt, E_incl_gt, soft_width=None, against="true")["err_relative"],
            }
            for k, v in vals.items():
                buckets[k].append(v)
            for k in ["base_soft", "hard", "shrink_cv", "oracle_hard"]:
                per_case[case][k].append(vals[k])
        if (i + 1) % 10 == 0:
            print(i + 1, "w_hist", flush=True)
    print("=== POOLED MEDIAN ===")
    summary = {}
    for k, v in buckets.items():
        summary[k] = float(np.median(v))
        print(f"{k:14s} {100*summary[k]:.2f}%")
    print("=== PER CASE ===")
    pc = {}
    for case, d in per_case.items():
        pc[case] = {k: float(np.median(v)) for k, v in d.items()}
        print(case, {k: f"{100*v:.2f}%" for k, v in pc[case].items()})
    out = ROOT / "outputs" / "fem_predict_shrink.json"
    out.write_text(json.dumps({"summary": summary, "per_case": pc}, indent=2))


if __name__ == "__main__":
    # Patch predict to allow soft_width=None with python floats (hard step)
    main()