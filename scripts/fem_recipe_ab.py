"""Quick A/B of FEM inversion recipes on the first N scenarios."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fem_validation_suite import case_tensors, invert_two_param_fem, load_case
DATA = Path(r"E:\Diff_Rending_Re_3D\dataset\ion_ct_synthetic_mechanics60")

def err(r):
    bg = abs(r["E_bg_est"] - r["E_bg_gt"]) / r["E_bg_gt"]
    c = abs(r["contrast_est"] - r["contrast_gt"]) / max(abs(r["contrast_gt"]), 1e-12)
    inc = abs(r["E_incl_est"] - r["E_incl_gt"]) / r["E_incl_gt"]
    return bg, c, inc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--iters", type=int, default=16)
    args = p.parse_args()
    scenarios = sorted(x for x in DATA.iterdir() if x.is_dir())[: args.n]
    recipes = {
        "baseline": dict(staged=False, ratio_weight=0.0, focus_gain=0.0, soft_width=0.02),
        "staged": dict(staged=True, ratio_weight=0.0, focus_gain=0.0, stage2_ratio=0.25, stage2_focus=1.0, soft_width=0.02),
        "staged_soft": dict(staged=True, ratio_weight=0.0, focus_gain=0.0, stage2_ratio=0.15, stage2_focus=2.0, soft_width=0.01),
    }
    rows = []
    for scen in scenarios:
        data = case_tensors(load_case(scen, "press_left"))
        print(scen.name, f"gt contrast={data['E_incl']/data['E_bg']:.2f}", flush=True)
        for name, kw in recipes.items():
            t0 = time.time()
            r = invert_two_param_fem(data, n_iters=args.iters, **kw)
            bg, c, inc = err(r)
            print(f"  {name:12s} bg={100*bg:5.1f}% c={100*c:5.1f}% incl={100*inc:5.1f}% "
                  f"est={r['E_incl_est']/1e3:.1f}/{r['E_incl_gt']/1e3:.1f}kPa  {time.time()-t0:.0f}s", flush=True)
            rows.append({"scenario": scen.name, "recipe": name, "E_bg_rel_err": bg,
                         "contrast_rel_err": c, "E_incl_rel_err": inc})
    out = ROOT / "outputs" / "fem_recipe_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # medians
    summary = {}
    for name in recipes:
        sub = [r for r in rows if r["recipe"] == name]
        summary[name] = {
            "median_bg": float(np.median([r["E_bg_rel_err"] for r in sub])),
            "median_contrast": float(np.median([r["contrast_rel_err"] for r in sub])),
            "median_incl": float(np.median([r["E_incl_rel_err"] for r in sub])),
        }
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("=== MEDIAN ===")
    for k, v in summary.items():
        print(f"{k:12s} bg {100*v['median_bg']:.1f}%  contrast {100*v['median_contrast']:.1f}%  incl {100*v['median_incl']:.1f}%")

if __name__ == "__main__":
    main()