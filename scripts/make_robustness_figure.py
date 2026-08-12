"""Mechanics-robustness 3-panel figure for the CMPB paper.

(a) Force-scale sweep: force scale vs recovered E_bg ratio (theory 1:1).
(b) Noise robustness: added motion noise vs E_bg error.
(c) Held-out load case: displacement error, inverted vs oracle material.
Data from fem_validation_summary.json + force_propagation.json.
Writes ``submission/cmpb/figures/fig_robustness.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEM = ROOT / "outputs" / "fem_validation" / "fem_validation_summary.json"
FORCE = ROOT / "outputs" / "force_propagation" / "force_propagation.json"
OUT = ROOT / "submission" / "cmpb" / "figures" / "fig_robustness.png"


def main():
    fem = json.loads(FEM.read_text(encoding="utf-8"))
    force = json.loads(FORCE.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=150)

    # (a) force sweep
    ax = axes[0]
    sw = force["force_scale_sweep"]
    xs = [r["force_scale"] for r in sw]
    ys = [r["E_bg_ratio"] for r in sw]
    ax.plot(xs, xs, "r--", lw=1.2, label="theory $E_{est}/E_{gt}=\\beta$")
    ax.plot(xs, ys, "o-", color="#4C86C6", label="measured")
    ax.set_xlabel(r"force scale $\beta$")
    ax.set_ylabel(r"$E_{bg}$ ratio (est/gt)")
    ax.set_title("(a) force-scale propagation")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # (b) noise robustness
    ax = axes[1]
    runs = fem["A3_noise"]["runs"]
    by_noise: dict[float, list] = {}
    for r in runs:
        by_noise.setdefault(r["noise_std"], []).append(r["E_bg_rel_err"] * 100)
    ns = sorted(by_noise)
    means = [np.mean(by_noise[n]) for n in ns]
    stds = [np.std(by_noise[n]) for n in ns]
    ax.errorbar([n * 1e3 for n in ns], means, yerr=stds, fmt="o-",
                color="#B03030", capsize=4)
    ax.set_xlabel("added motion noise (mm)")
    ax.set_ylabel(r"$E_{bg}$ error (%)")
    ax.set_title("(b) noise robustness")
    ax.grid(alpha=0.3)

    # (c) held-out load case
    ax = axes[2]
    ho = fem["A4_holdout_load"]["held_out"]
    cases = [h["case"] for h in ho]
    est = [h["err_relative_est"] * 100 for h in ho]
    ora = [h["err_relative_oracle"] * 100 for h in ho]
    x = np.arange(len(cases))
    w = 0.38
    ax.bar(x - w / 2, ora, w, label="oracle material", color="#888888")
    ax.bar(x + w / 2, est, w, label="inverted material", color="#6BAE6B")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=8)
    ax.set_ylabel("displacement error (%)")
    ax.set_title("(c) held-out load-case prediction")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Mechanical-inversion robustness", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
