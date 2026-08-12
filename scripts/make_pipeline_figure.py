"""Generate the overall pipeline figure (Fig 1) for the CMPB paper.

Vertical flow: endoscopic video -> three tiers -> simulable twin -> use,
with the provenance audit as a side bar spanning the three tiers.
Writes ``submission/cmpb/figures/fig_pipeline.png``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "cmpb" / "figures" / "fig_pipeline.png"

C_IN = "#EAF2FB"
C_T1 = "#D6E9D6"
C_T2 = "#FCE8C9"
C_T3 = "#F6D5D5"
C_TWIN = "#E3D9F2"
C_USE = "#D9E8E8"
C_AUDIT = "#FFF3B0"
EDGE = "#333333"


def box(ax, x, y, w, h, text, fc, fs=11, bold=False, ec=EDGE):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs,
        fontweight="bold" if bold else "normal", zorder=3,
    )


def arrow(ax, x1, y1, x2, y2, text="", fs=9):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
        linewidth=1.6, color=EDGE, zorder=1,
    )
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2 + 0.02, (y1 + y2) / 2, text,
                ha="left", va="center", fontsize=fs, style="italic", color="#222222")


def main():
    fig, ax = plt.subplots(figsize=(7.4, 9.2), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cx, w = 0.44, 0.56
    x = cx - w / 2

    # Input
    box(ax, x, 0.925, w, 0.062,
        "Endoscopic video\n$\\{I_t, D_t, M_t\\}$ (image, depth, tool mask)",
        C_IN, fs=11, bold=True)

    # Tier 1
    box(ax, x, 0.795, w, 0.085,
        "TIER 1  Optical-geometric field\n"
        "canonical Gaussian BRDF field $G$\n"
        "(geometry + appearance)",
        C_T1, fs=10.5, bold=True)
    arrow(ax, cx, 0.925, cx, 0.880, "reconstruct")

    # Tier 2
    box(ax, x, 0.660, w, 0.085,
        "TIER 2  Kinematic tracking\n"
        "deformation sequence $\\{\\Delta_t\\}$\n"
        "(replayable boundary conditions)",
        C_T2, fs=10.5, bold=True)
    arrow(ax, cx, 0.795, cx, 0.745, "track")

    # Tier 3
    box(ax, x, 0.525, w, 0.085,
        "TIER 3  Mechanical inversion\n"
        "elastic parameters $\\Theta$ + forward operator $\\mathcal{S}$\n"
        "(simulation-in-the-loop)",
        C_T3, fs=10.5, bold=True)
    arrow(ax, cx, 0.660, cx, 0.610, "invert")

    # Twin
    box(ax, x, 0.375, w, 0.105,
        "Simulable soft-tissue digital twin\n"
        "$\\mathcal{T} = (G, \\{\\Delta_t\\}, \\Theta, \\mathcal{A})$",
        C_TWIN, fs=11.5, bold=True)
    arrow(ax, cx, 0.525, cx, 0.480, "assemble")

    # Use
    box(ax, x, 0.225, w, 0.095,
        "Rehearsal & prediction\n"
        "stiffness-contrast map, held-out load cases,\n"
        "surgical planning",
        C_USE, fs=10.5, bold=True)
    arrow(ax, cx, 0.375, cx, 0.320, "drive")

    # Audit side bar
    axw = 0.155
    ax_x = 0.825
    box(ax, ax_x, 0.525, axw, 0.355,
        "Provenance\naudit\n$\\mathcal{A}$\n\nmeasured /\nestimated /\nassumed\n+ confidence",
        C_AUDIT, fs=9.5, bold=True)
    # connect audit to three tiers
    for yy in [0.838, 0.703, 0.568]:
        arrow(ax, ax_x, yy, x + w, yy, "")
    arrow(ax, ax_x + axw / 2, 0.525, ax_x + axw / 2, 0.480, "")

    # caption note at bottom
    ax.text(0.5, 0.12,
            "One canonical Gaussian field is shared by all three tiers;\n"
            "the audit tags every physical parameter with its provenance.",
            ha="center", va="center", fontsize=10, style="italic",
            color="#333333")

    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
