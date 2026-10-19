"""Ablate held-out deformation prediction without re-inverting."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DIFF = Path(r"E:\Diff_Rending_Re_3D")
sys.path.insert(0, str(DIFF))
from physics.fem import make_heterogeneous_E_field, newton_solve, node_E_to_per_elem_mu_lam, solve_nh_heterogeneous
from scripts.run_fem_validation_suite import case_tensors, load_case, predict_load_case, NU, _nu_t

DATA = Path(r"E:\Diff_Rending_Re_3D\dataset\ion_ct_synthetic_mechanics60")
EST = ROOT / "outputs" / "fem_validation_60" / "scenarios"
HOLDOUT = ("press_right", "shear_x", "shear_y")


def legacy_predict(data, E_bg, E_incl, frame=2):
    nodes, elems, surf = data["nodes"], data["elems"], data["surf"]
    E_field = make_heterogeneous_E_field(nodes, data["center"], data["radius"], E_bg, E_incl)
    mu, lam, _ = node_E_to_per_elem_mu_lam(E_field, elems, NU)
    u_pred = newton_solve(nodes, elems, mu, lam, data["forces_m"][frame], data["fixed_dofs"], 3).view(-1, 3)
    u_gt = data["u_obs"][frame]
    err = (u_pred[surf] - u_gt).norm(dim=-1)
    mag = u_gt.norm(dim=-1).mean().clamp_min(1e-9)
    return float((err.mean() / mag).detach())


def main():
    rows = []
    recipes = [
        "legacy_obs", "matched_obs", "matched_true",
        "homog_obs", "homog_true", "oracle_obs", "oracle_true",
        "adaptive_obs", "adaptive_true",
    ]
    pooled = {k: [] for k in recipes}
    scenarios = sorted(EST.glob("*.json"))[:20]  # 20 materials / first geom block enough? use all 60 for predict-only
    scenarios = sorted(EST.glob("*.json"))
    for path in scenarios:
        est = json.loads(path.read_text())
        scen = DATA / est["scenario"]
        E_bg, E_incl = est["E_bg_est"], est["E_incl_est"]
        E_bg_gt, E_incl_gt = est["E_bg_gt"], est["E_incl_gt"]
        contrast = E_incl / max(E_bg, 1e-12)
        # Adaptive: trust inclusion only when estimated contrast is moderate; else homogeneous.
        if 1.15 <= contrast <= 2.8:
            E_ad_bg, E_ad_incl = E_bg, E_incl
        else:
            E_ad_bg, E_ad_incl = E_bg, E_bg
        row = {"scenario": est["scenario"], "contrast_est": contrast, "contrast_gt": est["contrast_gt"]}
        for case in HOLDOUT:
            data = case_tensors(load_case(scen, case))
            vals = {
                "legacy_obs": legacy_predict(data, E_bg, E_incl),
                "matched_obs": predict_load_case(data, E_bg, E_incl, against="observed")["err_relative"],
                "matched_true": predict_load_case(data, E_bg, E_incl, against="true")["err_relative"],
                "homog_obs": predict_load_case(data, E_bg, E_bg, against="observed")["err_relative"],
                "homog_true": predict_load_case(data, E_bg, E_bg, against="true")["err_relative"],
                "oracle_obs": predict_load_case(data, E_bg_gt, E_incl_gt, against="observed")["err_relative"],
                "oracle_true": predict_load_case(data, E_bg_gt, E_incl_gt, against="true")["err_relative"],
                "adaptive_obs": predict_load_case(data, E_ad_bg, E_ad_incl, against="observed")["err_relative"],
                "adaptive_true": predict_load_case(data, E_ad_bg, E_ad_incl, against="true")["err_relative"],
            }
            for k, v in vals.items():
                pooled[k].append(v)
                row[f"{case}_{k}"] = v
        rows.append(row)
        if len(rows) % 10 == 0:
            print(len(rows), "done", flush=True)

    summary = {k: {
        "median": float(np.median(v)),
        "mean": float(np.mean(v)),
        "q75": float(np.quantile(v, 0.75)),
    } for k, v in pooled.items()}
    # per holdout case medians for matched/adaptive
    per_case = {}
    for case in HOLDOUT:
        per_case[case] = {
            "matched_obs": float(np.median([r[f"{case}_matched_obs"] for r in rows])),
            "matched_true": float(np.median([r[f"{case}_matched_true"] for r in rows])),
            "adaptive_obs": float(np.median([r[f"{case}_adaptive_obs"] for r in rows])),
            "oracle_obs": float(np.median([r[f"{case}_oracle_obs"] for r in rows])),
            "oracle_true": float(np.median([r[f"{case}_oracle_true"] for r in rows])),
            "legacy_obs": float(np.median([r[f"{case}_legacy_obs"] for r in rows])),
            "homog_obs": float(np.median([r[f"{case}_homog_obs"] for r in rows])),
        }
    out = ROOT / "outputs" / "fem_predict_ablation.json"
    out.write_text(json.dumps({"summary": summary, "per_case": per_case, "n": len(rows)}, indent=2))
    print("=== POOLED MEDIAN ===")
    for k, s in summary.items():
        print(f"{k:16s} median {100*s['median']:.2f}%  mean {100*s['mean']:.2f}%")
    print("=== PER CASE MEDIAN ===")
    for case, s in per_case.items():
        print(case, {k: f"{100*v:.2f}%" for k, v in s.items()})


if __name__ == "__main__":
    main()