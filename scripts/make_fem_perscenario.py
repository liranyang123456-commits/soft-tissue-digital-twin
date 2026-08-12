"""Per-scenario FEM-benchmark results figure (Fig 6) for the CMPB paper.

Grouped bars per scenario: (a) E_bg est vs gt; (b) inclusion contrast est vs
gt. Directly shows how many scenarios and how well each is recovered.
Writes ``submission/cmpb/figures/fig_fem_perscenario.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "fem_validation" / "fem_validation_summary.json"
OUT = ROOT / "submission" / "cmpb" / "figures" / "fig_fem_perscenario.png"


def main():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = s["A1_fem_inversion"]["per_scenario"]
    labels = [r["scenario"][-3:] for r in rows]
    E_gt = np.array([r["E_bg_gt"] for r in rows]) / 1e3
    E_est = np.array([r["E_bg_est"] for r in rows]) / 1e3
    c_gt = np.array([r["contrast_gt"] for r in rows])
    c_est = np.array([r["contrast_est"] for r in rows])

    x = np.arange(len(labels))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)

    ax = axes[0]
    ax.bar(x - w / 2, E_gt, w, label="ground truth", color="#888888")
    ax.bar(x + w / 2, E_est, w, label="recovered", color="#4C86C6")
    for xi, (g, e) in enumerate(zip(E_gt, E_est)):
        err = abs(e - g) / g * 100
        ax.text(xi, max(g, e) * 1.02, f"{err:.0f}%", ha="center", fontsize=8,
                color="#B03030" if err > 10 else "#227722")
    ax.set_xticks(x)
    ax.set_xticklabels([f"sc.{l}" for l in labels])
    ax.set_ylabel(r"$E_{bg}$ (kPa)")
    ax.set_title("(a) background modulus, est vs gt (error % on top)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - w / 2, c_gt, w, label="ground truth", color="#888888")
    ax.bar(x + w / 2, c_est, w, label="recovered", color="#6BAE6B")
    ax.set_xticks(x)
    ax.set_xticklabels([f"sc.{l}" for l in labels])
    ax.set_ylabel(r"inclusion contrast ($\times$)")
    ax.set_title("(b) inclusion contrast, est vs gt")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("FEM-benchmark inversion across six scenarios", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
