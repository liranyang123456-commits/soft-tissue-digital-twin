"""B3: assemble the ablation table from verified runs.

Ablations (all numbers from local verified outputs):
1. Tool-mask supervision fix (Tier 1): tool-mask vs tissue-mask supervision.
2. Temporal model (Tier 2): inertia-only vs constant-velocity + rollback.
3. Inversion parameterization (Tier 3): per-node field vs 2-region.
4. Forward model (Tier 3): FEM-in-the-loop vs mass-spring surrogate
   (filled from fem_validation_summary.json when available).

Writes ``outputs/ablation_table.json`` + ``outputs/ablation_table.md``.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main():
    rows = []

    # --- Ablation 1: tool-mask fix (Tier 1, pulling scene) -----------------
    rows.append({
        "ablation": "Tier-1 supervision mask",
        "variant": "tool mask (mistaken as tissue)",
        "metric": "tissue-PSNR (dB)", "value": 11.13,
        "source": "outputs/soft_tissue_twin (first run)",
    })
    rows.append({
        "ablation": "Tier-1 supervision mask",
        "variant": "tissue mask = depth>0 (correct)",
        "metric": "tissue-PSNR (dB)", "value": 36.79,
        "source": "outputs/soft_tissue_twin",
    })

    # --- Ablation 2: temporal model (Tier 2, pulling scene) ----------------
    rows.append({
        "ablation": "Tier-2 temporal model",
        "variant": "inertia-only",
        "metric": "worst-frame loss", "value": 37.23,
        "source": "outputs/soft_tissue_twin (frame 5 blow-up)",
    })
    rows.append({
        "ablation": "Tier-2 temporal model",
        "variant": "const-velocity + accel prior + step cap + rollback",
        "metric": "worst-frame loss", "value": 1.55,
        "source": "outputs/soft_tissue_twin_v2 (frame 5 rolled back)",
    })

    # --- Ablation 3: inversion parameterization (Tier 3, synthetic) --------
    rows.append({
        "ablation": "Tier-3 inversion parameterization",
        "variant": "per-node E field + smoothness",
        "metric": "inclusion contrast (GT 4.0x)", "value": 1.69,
        "source": "tissue_sim per-node run",
    })
    rows.append({
        "ablation": "Tier-3 inversion parameterization",
        "variant": "2-region (E_bg, E_incl)",
        "metric": "inclusion contrast (GT 4.0x)", "value": 3.19,
        "source": "outputs/soft_tissue_twin/tier3_elasticity",
    })

    # --- Ablation 4: forward model (filled from FEM suite if present) ------
    fem_summary = OUT / "fem_validation" / "fem_validation_summary.json"
    if fem_summary.exists():
        s = json.loads(fem_summary.read_text(encoding="utf-8"))
        if "A1_fem_inversion" in s:
            rows.append({
                "ablation": "Tier-3 forward model (FEM dataset)",
                "variant": "FEM-in-the-loop (matches generator)",
                "metric": "E_bg rel err",
                "value": round(s["A1_fem_inversion"]["E_bg_rel_err_mean"], 4),
                "source": "outputs/fem_validation",
            })
        if "B1_surrogate" in s:
            rows.append({
                "ablation": "Tier-3 forward model (FEM dataset)",
                "variant": "mass-spring surrogate (model mismatch)",
                "metric": "inclusion contrast est",
                "value": round(s["B1_surrogate"]["inversion"]["contrast_est"], 3),
                "source": "outputs/fem_validation",
            })

    doc = {"ablations": rows}
    (OUT / "ablation_table.json").write_text(
        json.dumps(doc, indent=2), encoding="utf-8"
    )
    md = ["# Ablation table (B3)\n",
          "| Ablation | Variant | Metric | Value |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['ablation']} | {r['variant']} | {r['metric']} | {r['value']} |")
    (OUT / "ablation_table.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT / 'ablation_table.json'} and .md ({len(rows)} rows)")


if __name__ == "__main__":
    main()
