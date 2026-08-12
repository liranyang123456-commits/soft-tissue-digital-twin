"""Assemble a one-page summary figure from soft-tissue twin intermediate results.

Writes ``outputs/soft_tissue_twin/summary_panel.png`` with:
  row 1: Tier-1 input | reconstruction | tissue-masked
  row 2: Tier-2 deformation quiver
  row 3: Tier-3 Exp A (uniform E) | Exp B (tumor inclusion)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "soft_tissue_twin"


def main():
    summary = json.loads((OUT / "twin_summary.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(14, 12), dpi=140)

    # --- Row 1: Tier 1 ---
    ax = fig.add_subplot(3, 1, 1)
    img = mpimg.imread(OUT / "tier1_reconstruction" / "final_compare.png")
    ax.imshow(img)
    ax.set_title(
        f"Tier 1 — Optical-geometric twin   "
        f"tissue-PSNR {summary['tier1']['psnr']:.1f} dB   "
        f"({summary['tier1']['n_gaussians']} Gaussians)   "
        f"[left: input | mid: render | right: tissue only]",
        fontsize=11,
    )
    ax.axis("off")

    # --- Row 2: Tier 2 ---
    ax = fig.add_subplot(3, 1, 2)
    img = mpimg.imread(OUT / "tier2_deformation" / "deformation_quiver.png")
    ax.imshow(img)
    ax.set_title(
        "Tier 2 — Kinematic twin: tracked deformation field (displacement BC)",
        fontsize=11,
    )
    ax.axis("off")

    # --- Row 3: Tier 3 ---
    ax = fig.add_subplot(3, 2, 5)
    img = mpimg.imread(OUT / "tier3_elasticity" / "synthetic_uniform.png")
    ax.imshow(img)
    e = summary["tier3"]["synthetic_uniform"]
    ax.set_title(
        f"Tier 3a — Uniform E: {e['E_gt_kpa']:.1f}→{e['E_est_kpa']:.2f} kPa "
        f"(err {e['relative_error_pct']:.2f}%)",
        fontsize=10,
    )
    ax.axis("off")

    ax = fig.add_subplot(3, 2, 6)
    img = mpimg.imread(OUT / "tier3_elasticity" / "synthetic_inclusion.png")
    ax.imshow(img)
    b = summary["tier3"]["synthetic_inclusion"]
    ax.set_title(
        f"Tier 3b — Tumor inclusion: contrast {b['contrast_gt']:.1f}×→"
        f"{b['contrast_est']:.2f}×  MAE {b['field_mae_kpa']:.2f} kPa",
        fontsize=10,
    )
    ax.axis("off")

    fig.tight_layout()
    out_path = OUT / "summary_panel.png"
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
