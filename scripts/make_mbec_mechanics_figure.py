"""Mechanical-validation figure for MBEC, drawn directly from result files.

(a) FEM60 background modulus, estimate versus ground truth
(b) FEM60 relative-error distributions
(c) FEM60 stiffness contrast, estimate versus ground truth
(d) force-scale sweep on a representative scenario
(e) displacement prediction for unseen load cases
(f) ETS phantom contrast against the compression-test reference
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEM60 = ROOT / "outputs" / "fem_validation_60" / "fem60_per_scenario.csv"
FEM = ROOT / "outputs" / "fem_validation" / "fem_validation_summary.json"
FORCE = ROOT / "outputs" / "force_propagation" / "force_propagation.json"
HOLDOUT = ROOT / "outputs" / "fem_model_mismatch" / "holdout_scenario000.json"
ETS = Path(r"E:\Diff_Rending_Re_3D\results\ets_phantom_eval.json")
OUT = ROOT / "submission" / "mbec" / "figures" / "fig_mechanics.png"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6.3,
        "axes.titlesize": 6.6,
        "axes.labelsize": 6.3,
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 5.7,
        "legend.fontsize": 5.4,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
    }
)


def tidy(axis, letter: str) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.25, linewidth=0.4)
    axis.text(-0.27, 1.06, letter, transform=axis.transAxes, fontsize=7.5, fontweight="bold", va="bottom")


def main() -> None:
    with FEM60.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    value = lambda key: np.asarray([float(row[key]) for row in rows])
    bg_gt, bg_est = value("E_bg_gt") / 1e3, value("E_bg_est") / 1e3
    c_gt, c_est = value("contrast_gt"), value("contrast_est")
    errors = [100 * value("E_bg_rel_err"), 100 * value("E_incl_rel_err"), 100 * value("contrast_rel_err")]

    force = json.loads(FORCE.read_text(encoding="utf-8"))
    fem = json.loads(FEM.read_text(encoding="utf-8"))
    ets = json.loads(ETS.read_text(encoding="utf-8"))

    figure, axes = plt.subplots(2, 3, figsize=(4.72, 3.25), dpi=300)
    figure.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.115, wspace=0.52, hspace=0.62)

    axis = axes[0, 0]
    axis.scatter(bg_gt, bg_est, s=7, c="#3B5B7A", alpha=0.45, linewidths=0)
    limits = [1.8, 8.2]
    axis.plot(limits, limits, "k--", linewidth=0.6)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel(r"true $E_{bg}$ (kPa)")
    axis.set_ylabel(r"estimated $E_{bg}$ (kPa)")
    axis.set_title("background modulus")
    tidy(axis, "a")

    axis = axes[0, 1]
    positions = np.arange(1, 4)
    axis.boxplot(errors, positions=positions, widths=0.5, showfliers=False,
                 medianprops={"color": "#B5651D", "linewidth": 0.9},
                 boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6})
    rng = np.random.default_rng(1)
    for position, sample in zip(positions, errors):
        axis.scatter(position + rng.uniform(-0.16, 0.16, len(sample)), sample, s=2.2, c="#475569", alpha=0.6, linewidths=0)
    axis.set_xticks(positions, [r"$E_{bg}$", r"$E_{incl}$", "contrast"])
    axis.set_ylabel("relative error (%)")
    axis.set_title("benchmark error")
    tidy(axis, "b")

    axis = axes[0, 2]
    axis.scatter(c_gt, c_est, s=7, c="#3B5B7A", alpha=0.45, linewidths=0)
    limits = [0.9, 4.3]
    axis.plot(limits, limits, "k--", linewidth=0.6)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("true contrast")
    axis.set_ylabel("estimated contrast")
    axis.set_title("inclusion contrast")
    tidy(axis, "c")

    axis = axes[1, 0]
    sweep = force["force_scale_sweep"]
    beta = np.asarray([entry["force_scale"] for entry in sweep])
    ratio = np.asarray([entry["E_bg_ratio"] for entry in sweep])
    axis.plot(beta, beta, "--", color="#6B7280", linewidth=0.7, label=r"exact scaling")
    axis.plot(beta, ratio, "o-", color="#3B5B7A", markersize=2.4, linewidth=0.8, label=r"$E_{bg}$ estimate")
    axis.set_xlabel(r"force scale $\beta$")
    axis.set_ylabel(r"$E_{bg}^{est}/E_{bg}^{true}$")
    axis.set_title("force-scale sweep")
    axis.legend(frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.1)
    tidy(axis, "d")

    axis = axes[1, 1]
    held = json.loads(HOLDOUT.read_text(encoding="utf-8"))["conditions"]
    cases = [entry["case"].replace("_", "\n") for entry in held["matched"]]
    oracle = [100 * entry["oracle_rel_err"] for entry in held["matched"]]
    inverted = [100 * entry["estimated_rel_err"] for entry in held["matched"]]
    refined = [100 * entry["estimated_rel_err"] for entry in held["fine"]]
    x = np.arange(len(cases))
    axis.bar(x - 0.27, oracle, 0.25, color="#9CA3AF", label="true material")
    axis.bar(x, inverted, 0.25, color="#3B5B7A", label="inverted")
    axis.bar(x + 0.27, refined, 0.25, color="#C9A36A", label="inverted, refined-mesh data")
    axis.set_xticks(x, cases)
    axis.set_ylabel("displacement error (%)")
    axis.set_title("unseen load cases")
    axis.legend(frameon=False, loc="upper left", handlelength=1.0, borderaxespad=0.1)
    axis.set_ylim(0, 58)
    tidy(axis, "e")

    axis = axes[1, 2]
    sequences = ets["sequences"]
    estimate = [entry["estimated_modulus_ratio_bg_over_inc"] for entry in sequences]
    reference = ets["ground_truth"]["modulus_ratio_bg_over_inc"]
    low, high = ets["bootstrap_95_ci_of_median"]
    median = ets["median_estimated_ratio"]
    x = np.arange(1, len(estimate) + 1)
    axis.axhspan(low, high, color="#C9D6E3", alpha=0.35, linewidth=0, label="95% CI of median")
    axis.bar(x, estimate, 0.6, color="#3B5B7A")
    axis.axhline(reference, color="#B5651D", linestyle="--", linewidth=0.8, label=f"compression test {reference:.2f}")
    axis.axhline(median, color="#111827", linestyle=":", linewidth=0.8, label=f"median {median:.2f}")
    axis.set_xticks(x)
    axis.set_xlabel("acquisition")
    axis.set_ylabel(r"$E_{bg}/E_{incl}$")
    axis.set_ylim(0, 7.4)
    axis.set_title("phantom modulus ratio")
    axis.legend(frameon=False, loc="upper center", ncol=1, handlelength=1.2, borderaxespad=0.05, labelspacing=0.15)
    tidy(axis, "f")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=300, facecolor="white")
    plt.close(figure)
    print(OUT)


if __name__ == "__main__":
    main()
