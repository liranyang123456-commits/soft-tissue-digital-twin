"""Which material error hurts holdout prediction: E_bg or E_incl?"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fem_validation_suite import case_tensors, load_case, predict_load_case
DATA = Path(r"E:\Diff_Rending_Re_3D\dataset\ion_ct_synthetic_mechanics60")
EST = ROOT / "outputs" / "fem_validation_60" / "scenarios"
HOLDOUT = ("press_right", "shear_x", "shear_y")
keys = ["both_est", "bg_est_incl_gt", "bg_gt_incl_est", "both_gt", "homog_bg_est"]
pool = {k: [] for k in keys}
for i, path in enumerate(sorted(EST.glob("*.json"))):
    est = json.loads(path.read_text())
    scen = DATA / est["scenario"]
    mats = {
        "both_est": (est["E_bg_est"], est["E_incl_est"]),
        "bg_est_incl_gt": (est["E_bg_est"], est["E_incl_gt"]),
        "bg_gt_incl_est": (est["E_bg_gt"], est["E_incl_est"]),
        "both_gt": (est["E_bg_gt"], est["E_incl_gt"]),
        "homog_bg_est": (est["E_bg_est"], est["E_bg_est"]),
    }
    for case in HOLDOUT:
        data = case_tensors(load_case(scen, case))
        for k, (bg, inc) in mats.items():
            pool[k].append(predict_load_case(data, bg, inc, soft_width=None, against="true")["err_relative"])
    if (i+1) % 15 == 0:
        print(i+1, flush=True)
print("=== against TRUE surface, hard field ===")
for k, v in pool.items():
    print(f"{k:18s} median {100*np.median(v):.2f}%  mean {100*np.mean(v):.2f}%")