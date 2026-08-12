"""ETS phantom per-acquisition figure for the CMPB paper.

Per-acquisition estimated modulus contrast vs the compression-test truth
(3.56x), showing the video-derived contrast is consistent across all six
acquisitions. Writes ``submission/cmpb/figures/fig_ets.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ETS = Path(r"E:\Diff_Rending_Re_3D\results\ets_phantom_eval.json")
OUT = ROOT / "submission" / "cmpb" / "figures" / "fig_ets.png"


def main():
    ev = json.loads(ETS.read_text(encoding="utf-8"))
    seqs = ev["sequences"]
    acq = [s["acquisition"] for s in seqs]
    ratio = [s["estimated_modulus_ratio_bg_over_inc"] for s in seqs]
    truth = ev["ground_truth"]["modulus_ratio_bg_over_inc"]
    med = ev["median_estimated_ratio"]
    lo, hi = ev["bootstrap_95_ci_of_median"]

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    x = np.arange(len(acq))
    ax.bar(x, ratio, color="#4C86C6", width=0.6, label="image-derived contrast")
    ax.axhline(truth, color="#B03030", ls="--", lw=1.5,
               label=f"compression-test truth {truth:.2f}x")
    ax.axhspan(lo, hi, color="#888888", alpha=0.2,
               label=f"95% CI of median [{lo:.2f}, {hi:.2f}]")
    ax.axhline(med, color="#444444", ls=":", lw=1.2, label=f"median {med:.2f}x")
    for xi, r in zip(x, ratio):
        err = abs(r - truth) / truth * 100
        ax.text(xi, r + 0.05, f"{err:.0f}%", ha="center", fontsize=8, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([f"acq {a}" for a in acq])
    ax.set_ylabel(r"modulus contrast $E_{bg}/E_{incl}$ ($\times$)")
    ax.set_title("\'ETS phantom: per-acquisition modulus contrast vs truth")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
