"""Post-hoc force/modulus scale calibration from the training load, then predict."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fem_validation_suite import case_tensors, load_case, predict_load_case

DATA = Path(r"E:\Diff_Rending_Re_3D\dataset\ion_ct_synthetic_mechanics60")
EST = ROOT / "outputs" / "fem_validation_60" / "scenarios"
HOLDOUT = ("press_right", "shear_x", "shear_y")


def train_scale(left, E_bg, E_incl, frame=None):
    frame = left["frames"][len(left["frames"]) // 2] if frame is None else frame
    # predict on training load mid frame
    pred = predict_load_case(left, E_bg, E_incl, frame=frame, soft_width=None, against="observed")
    u_p = pred["u_pred_surf"]
    u_o = left["u_obs"][list(left["frames"]).index(frame)]
    # robust scale: median |u_obs| / |u_pred| per-node norms
    num = u_o.norm(dim=-1)
    den = u_p.norm(dim=-1).clamp_min(1e-12)
    ratio = (num / den).detach().cpu().numpy()
    s = float(np.median(ratio[np.isfinite(ratio)]))
    return s


def main():
    buckets = {k: [] for k in [
        "base", "scale_E", "scale_E_true", "oracle", "oracle_true", "base_true"
    ]}
    per_case = {c: {k: [] for k in ["base", "scale_E", "oracle"]} for c in HOLDOUT}
    scales = []
    for i, path in enumerate(sorted(EST.glob("*.json"))):
        est = json.loads(path.read_text())
        scen = DATA / est["scenario"]
        left = case_tensors(load_case(scen, "press_left"))
        E_bg, E_incl = est["E_bg_est"], est["E_incl_est"]
        s = train_scale(left, E_bg, E_incl)
        scales.append(s)
        # u ~ F/E => to match larger observed motion, lower E by s... 
        # if s = |u_obs|/|u_pred| > 1, pred too stiff (E too high) => E' = E/s
        E_bg_s, E_incl_s = E_bg / s, E_incl / s
        for case in HOLDOUT:
            data = case_tensors(load_case(scen, case))
            vals = {
                "base": predict_load_case(data, E_bg, E_incl, soft_width=None, against="observed")["err_relative"],
                "scale_E": predict_load_case(data, E_bg_s, E_incl_s, soft_width=None, against="observed")["err_relative"],
                "scale_E_true": predict_load_case(data, E_bg_s, E_incl_s, soft_width=None, against="true")["err_relative"],
                "oracle": predict_load_case(data, est["E_bg_gt"], est["E_incl_gt"], soft_width=None, against="observed")["err_relative"],
                "oracle_true": predict_load_case(data, est["E_bg_gt"], est["E_incl_gt"], soft_width=None, against="true")["err_relative"],
                "base_true": predict_load_case(data, E_bg, E_incl, soft_width=None, against="true")["err_relative"],
            }
            for k, v in vals.items():
                buckets[k].append(v)
            for k in ["base", "scale_E", "oracle"]:
                per_case[case][k].append(vals[k])
        if (i + 1) % 10 == 0:
            print(i + 1, "median_s", float(np.median(scales)), flush=True)
    print("scale median/mean", float(np.median(scales)), float(np.mean(scales)))
    print("=== POOLED ===")
    summary = {k: float(np.median(v)) for k, v in buckets.items()}
    for k, v in summary.items():
        print(f"{k:14s} {100*v:.2f}%")
    print("=== PER CASE ===")
    pc = {}
    for case, d in per_case.items():
        pc[case] = {k: float(np.median(v)) for k, v in d.items()}
        print(case, {k: f"{100*v:.2f}%" for k, v in pc[case].items()})
    (ROOT / "outputs" / "fem_predict_scale.json").write_text(
        json.dumps({"summary": summary, "per_case": pc, "scale_median": float(np.median(scales))}, indent=2)
    )


if __name__ == "__main__":
    main()