"""Generate the FEM-benchmark distribution figure for the CMPB paper.

Two panels from the 6-scenario A1 inversion results:
  (a) background-modulus relative error vs ground-truth background
      stiffness, showing the regime dependence (worse on soft backgrounds);
  (b) recovered inclusion contrast vs ground-truth contrast (identity
      line), showing contrast recovery across scenarios.

Writes ``submission/cmpb/figures/fig_fem_distribution.png``.
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
OUT = ROOT / "submission" / "cmpb" / "figures" / "fig_fem_distribution.png"


def main():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = s["A1_fem_inversion"]["per_scenario"]
    E_bg_gt = np.array([r["E_bg_gt"] for r in rows]) / 1e3
    E_bg_est = np.array([r["E_bg_est"] for r in rows]) / 1e3
    err = np.abs(E_bg_est - E_bg_gt) / E_bg_gt * 100
    c_gt = np.array([r["contrast_gt"] for r in rows])
    c_est = np.array([r["contrast_est"] for r in rows])
    labels = [r["scenario"][-3:] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)

    # (a) E_bg error vs stiffness
    ax = axes[0]
    sc = ax.scatter(E_bg_gt, err, s=90, c=err, cmap="RdYlGn_r",
                    edgecolors="k", linewidths=0.6, zorder=3)
    for x, y, lb in zip(E_bg_gt, err, labels):
        ax.annotate(lb, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color="0.3")
    ax.axhline(np.median(err), color="navy", ls="--", lw=1.2,
               label=f"median {np.median(err):.1f}%")
    ax.set_xlabel(r"ground-truth background stiffness $E_{bg}^{gt}$ (kPa)")
    ax.set_ylabel(r"$E_{bg}$ relative error (%)")
    ax.set_title("(a) background-modulus error vs stiffness regime")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3, zorder=0)
    fig.colorbar(sc, ax=ax, label="error (%)")

    # (b) contrast est vs gt
    ax = axes[1]
    lim = [0, max(c_gt.max(), c_est.max()) * 1.15]
    ax.plot(lim, lim, "r--", lw=1.2, label="ideal")
    ax.scatter(c_gt, c_est, s=90, c="steelblue", edgecolors="k",
               linewidths=0.6, zorder=3)
    for x, y, lb in zip(c_gt, c_est, labels):
        ax.annotate(lb, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color="0.3")
    ax.set_xlabel(r"ground-truth inclusion contrast ($\times$)")
    ax.set_ylabel(r"recovered contrast ($\times$)")
    ax.set_title("(b) inclusion-contrast recovery")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, zorder=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"  E_bg err: median {np.median(err):.1f}%, mean {np.mean(err):.1f}%, "
          f"range [{err.min():.1f}, {err.max():.1f}]")
    print(f"  contrast: est mean {c_est.mean():.2f}x vs gt mean {c_gt.mean():.2f}x")


if __name__ == "__main__":
    main()
