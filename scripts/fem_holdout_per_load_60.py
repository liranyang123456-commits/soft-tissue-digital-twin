"""Per-load unseen-load errors on all 60 FEM scenarios for true, 20-iteration, background-only, and residual-selected moduli."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fem_validation_suite import case_tensors, load_case, predict_load_case

DATA = Path(r"E:\Diff_Rending_Re_3D\dataset\ion_ct_synthetic_mechanics60")
HOLDOUT = ("press_right", "shear_x", "shear_y")
RUN = ROOT / "outputs" / "fem_bg_only40_60"
BASE = ROOT / "outputs" / "fem_validation_60" / "scenarios"
select = {r["scenario"]: r for r in json.loads((RUN / "select_train.json").read_text(encoding="utf-8"))["rows"]}

def errs(scen, E_bg, E_incl):
    return {c: float(predict_load_case(case_tensors(load_case(scen, c)), float(E_bg), float(E_incl), soft_width=None, against="true")["err_relative"]) for c in HOLDOUT}

rows = []
for f in sorted((RUN / "scenarios").glob("*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    scen = DATA / r["scenario"]
    base = json.loads((BASE / f.name).read_text(encoding="utf-8"))
    s = select[r["scenario"]]
    chosen = r[s["chosen"]]
    row = {
        "scenario": r["scenario"],
        "true": errs(scen, base["E_bg_gt"], base["E_incl_gt"]),
        "paper20": errs(scen, r["paper20"]["E_bg_est"], r["paper20"]["E_incl_est"]),
        "bg_only40": errs(scen, r["bg_only40"]["E_bg_est"], r["bg_only40"]["E_incl_est"]),
        "selected": errs(scen, chosen["E_bg_est"], chosen["E_incl_est"]),
        "chosen": s["chosen"],
    }
    rows.append(row)
    print(len(rows), r["scenario"][-12:], {k: round(100 * np.median(list(row[k].values())), 1) for k in ("true", "paper20", "bg_only40", "selected")}, flush=True)

summary = {}
for k in ("true", "paper20", "bg_only40", "selected"):
    summary[k] = {
        "median_over_scenarios_of_case_median": float(np.median([np.median(list(r[k].values())) for r in rows])),
        **{c: float(np.median([r[k][c] for r in rows])) for c in HOLDOUT},
    }
out = {"n": len(rows), "summary": summary, "rows": rows}
(RUN / "holdout_per_load_60.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))