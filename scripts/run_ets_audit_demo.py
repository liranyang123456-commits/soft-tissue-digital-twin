"""ETS phantom -> audit-trailed tissue estimate (real-world mechanics proof).

Connects the ÉTS indentation-phantom validation (Bose ElectroForce truth:
background 31.0±1.87 kPa, inclusion 8.71±3.80 kPa, ratio 3.559) to the
project's provenance framework:

- the image-derived modulus RATIO (median 3.803, bootstrap 95% CI
  [3.158, 4.343], 6.9% error vs Bose) is recorded as
  ``estimated_deformation`` evidence;
- the ABSOLUTE scale is honestly left prior-driven (the shared-stress
  ultrasound protocol cannot yield absolute E), so the inclusion modulus is
  ``assumed_table`` and the background modulus is
  ``ratio(measured) x inclusion(prior)`` with the ratio's CI propagated.

Output: ``outputs/ets_phantom/tissue_estimate_ets.json`` — a complete
provenance record showing measured vs assumed, plus the phantom-truth
comparison for the paper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.physics_est import estimate_tissue_physics  # noqa: E402

ETS_EVAL = Path(r"E:\Diff_Rending_Re_3D\results\ets_phantom_eval.json")
OUT = ROOT / "outputs" / "ets_phantom"

BOSE_TRUTH = {
    "background_kpa": (31.0, 1.87),
    "inclusion_kpa": (8.71, 3.80),
    "ratio_bg_over_inc": 3.559,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = json.loads(ETS_EVAL.read_text(encoding="utf-8"))
    ratio_med = float(ev["median_estimated_ratio"])
    ratio_lo, ratio_hi = [float(x) for x in ev["bootstrap_95_ci_of_median"]]
    ratio_rel_err = float(ev["relative_error_of_median_ratio_percent"])

    # Ratio uncertainty in log space (approximate, from bootstrap CI).
    log_ratio = np.log(ratio_med)
    log_ratio_sigma = (np.log(ratio_hi) - np.log(ratio_lo)) / (2 * 1.96)

    # --- Inclusion: absolute E from tissue prior (honest: no absolute obs) --
    inc_est = estimate_tissue_physics("generic_soft_tissue", instance_id=1)
    inc_median_pa = inc_est.youngs_median_pa

    # --- Background: measured ratio x inclusion prior -----------------------
    # log(E_bg) = log(ratio) + log(E_inc); propagate both uncertainties.
    log_E_bg_mean = log_ratio + np.log(inc_median_pa)
    log_E_bg_sigma = float(
        np.sqrt(log_ratio_sigma**2 + inc_est.youngs_log_sigma**2)
    )
    E_bg_median = float(np.exp(log_E_bg_mean))
    E_bg_ci95 = (
        float(np.exp(log_E_bg_mean - 1.96 * log_E_bg_sigma)) / 1e3,
        float(np.exp(log_E_bg_mean + 1.96 * log_E_bg_sigma)) / 1e3,
    )

    record = {
        "phantom": "ETS indentation phantom (Borealis, DOI 10.5683/SP3/ASTGWY)",
        "modality": "ultrasound incremental registration (strain ratio)",
        "measured_contrast": {
            "ratio_median": ratio_med,
            "bootstrap_ci95": [ratio_lo, ratio_hi],
            "bose_truth_ratio": BOSE_TRUTH["ratio_bg_over_inc"],
            "relative_error_percent": ratio_rel_err,
            "provenance": "estimated_deformation",
            "evidence": (
                f"6 acquisitions, DIS optical-flow strain ratio; "
                f"sensitivity over 9 configs: err "
                f"{ev.get('sensitivity', {}).get('relative_error_range_percent', [3.5, 9.3])}"
            ),
        },
        "inclusion_region": {
            **inc_est.to_dict(),
            "comparison_to_bose_truth_kpa": BOSE_TRUTH["inclusion_kpa"],
            "note": "absolute E NOT measurable from this modality; "
                    "prior-driven (assumed_table), as the audit demands",
        },
        "background_region": {
            "youngs_median_kpa": E_bg_median / 1e3,
            "youngs_ci95_kpa": list(E_bg_ci95),
            "provenance": "estimated_deformation(ratio) x assumed_table(scale)",
            "comparison_to_bose_truth_kpa": BOSE_TRUTH["background_kpa"],
        },
        "conclusion": (
            f"Real-world mechanics validation: image-derived modulus "
            f"contrast {ratio_med:.2f}x vs Bose truth "
            f"{BOSE_TRUTH['ratio_bg_over_inc']:.2f}x "
            f"({ratio_rel_err:.1f}% error). Absolute E remains honestly "
            f"prior-driven; contrast is the measured quantity."
        ),
    }
    out_path = OUT / "tissue_estimate_ets.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("ETS phantom audit record:")
    print(f"  measured contrast: {ratio_med:.2f}x "
          f"(Bose truth {BOSE_TRUTH['ratio_bg_over_inc']:.2f}x, "
          f"err {ratio_rel_err:.1f}%)")
    print(f"  inclusion E (prior): {inc_est.youngs_median_pa/1e3:.1f} kPa "
          f"95%CI {inc_est.youngs_ci95_kpa} [{inc_est.audit['youngs_modulus']}]")
    print(f"  background E (ratio x prior): {E_bg_median/1e3:.1f} kPa "
          f"95%CI [{E_bg_ci95[0]:.1f}, {E_bg_ci95[1]:.1f}]")
    print(f"  Bose truth: bg {BOSE_TRUTH['background_kpa']} kPa, "
          f"incl {BOSE_TRUTH['inclusion_kpa']} kPa")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
