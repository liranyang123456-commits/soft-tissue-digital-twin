import json
import glob
from pathlib import Path

for name in ["ours_constant_light_mononormal_w2", "nerfactor"]:
    p = Path(f"outputs/scene0070_same_protocol/{name}/eval_holdout/metrics.json")
    if not p.exists():
        c = glob.glob(f"outputs/scene0070_same_protocol/{name}/**/metrics.json", recursive=True)
        p = Path(c[0]) if c else None
    if p is None or not p.exists():
        print(name, "NOT FOUND")
        continue
    d = json.load(open(p))
    print("===", name, "===")
    if "macro" in d:
        print("  macro NMAE:", d["macro"].get("normal_mae_deg"), "RGB:", d["macro"].get("full_psnr"))
    pv = d.get("per_view", {})
    if isinstance(pv, dict):
        for v, m in pv.items():
            print("  view", v, "NMAE", m.get("normal_mae_deg"), "RGB", m.get("full_psnr"))
    elif isinstance(pv, list):
        for m in pv:
            print("  view", m.get("view_id"), "NMAE", m.get("normal_mae_deg"), "RGB", m.get("full_psnr"))
