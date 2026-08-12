"""Generate the model architecture figure (Fig 2) for the CMPB paper.

Left-to-right "shared substrate" diagram: the canonical Gaussian field at
the center, fed by the three tiers' inputs, driving the renderer and the
simulator, with the provenance audit wrapping the parameters.
Writes ``submission/cmpb/figures/fig_architecture.png``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "cmpb" / "figures" / "fig_architecture.png"

EDGE = "#333333"
C_INPUT = "#EAF2FB"
C_FIELD = "#E3D9F2"
C_T1 = "#D6E9D6"
C_T2 = "#FCE8C9"
C_T3 = "#F6D5D5"
C_OUT = "#D9E8E8"
C_AUDIT = "#FFF3B0"


def box(ax, x, y, w, h, text, fc, fs=10, bold=False):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.4, edgecolor=EDGE, facecolor=fc, zorder=2,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p1, p2, text="", fs=8.5, rad=0.0):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=16, linewidth=1.5,
        color=EDGE, zorder=1,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs,
                style="italic", color="#222222",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def main():
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- central substrate ----
    box(ax, 0.40, 0.42, 0.22, 0.20,
        "Canonical Gaussian field $G$\n"
        "position, covariance, normal,\n"
        "BRDF $(\\rho_d,\\rho_s,\\alpha,m,a)$",
        C_FIELD, fs=10.5, bold=True)

    # ---- Tier inputs (left) ----
    box(ax, 0.02, 0.70, 0.26, 0.16,
        "TIER 1 inputs\nimage $I_t$, depth $D_t$,\ntool mask $M_t$",
        C_T1, fs=10, bold=True)
    box(ax, 0.02, 0.42, 0.26, 0.16,
        "TIER 2 inputs\nsubsequent frames\n$\\{I_t,D_t,M_t\\}_{t>1}$",
        C_T2, fs=10, bold=True)
    box(ax, 0.02, 0.14, 0.26, 0.16,
        "TIER 3 inputs\ndeformation $\\{\\Delta_t\\}$\n+ load $f$ (measured/est.)",
        C_T3, fs=10, bold=True)

    # ---- Tier modules (middle) ----
    box(ax, 0.40, 0.74, 0.22, 0.14,
        "TIER 1\noptical-geometric fitting", C_T1, fs=10, bold=True)
    box(ax, 0.40, 0.06, 0.22, 0.14,
        "TIER 2 + 3\ndeformation tracking\n+ mechanical inversion", C_T3, fs=10, bold=True)

    # ---- outputs (right) ----
    box(ax, 0.74, 0.60, 0.24, 0.16,
        "Differentiable renderer\n$\\hat{I}, \\hat{D}, \\hat{\\alpha}$\n(lobe-off diffuse)",
        C_OUT, fs=10, bold=True)
    box(ax, 0.74, 0.20, 0.24, 0.16,
        "Differentiable simulator\n$\\mathcal{S}:(G,\\Theta,f)\\mapsto u$\n(FEM / mass--spring)",
        C_OUT, fs=10, bold=True)

    # ---- audit (bottom) ----
    box(ax, 0.30, -0.02, 0.56, 0.13,
        "Provenance audit $\\mathcal{A}$: every parameter tagged\n"
        "measured / estimated / assumed + confidence",
        C_AUDIT, fs=10, bold=True)

    # ---- arrows: inputs -> tiers ----
    arrow(ax, (0.28, 0.78), (0.40, 0.80), "")
    arrow(ax, (0.28, 0.50), (0.40, 0.50), "")
    arrow(ax, (0.28, 0.22), (0.40, 0.16), "")

    # tiers -> field
    arrow(ax, (0.51, 0.74), (0.51, 0.62), "build $G$")
    arrow(ax, (0.51, 0.20), (0.51, 0.42), "track + invert")

    # field -> outputs
    arrow(ax, (0.62, 0.55), (0.74, 0.66), "render")
    arrow(ax, (0.62, 0.48), (0.74, 0.30), "simulate")

    # outputs -> back (supervision)
    arrow(ax, (0.74, 0.60), (0.62, 0.50), "photometric loss", rad=0.25)

    # audit connections
    arrow(ax, (0.51, 0.42), (0.51, 0.11), "")
    arrow(ax, (0.86, 0.20), (0.86, 0.11), "")

    ax.text(0.5, 0.97,
            "One shared Gaussian field carries appearance, motion, and mechanics",
            ha="center", va="top", fontsize=11.5, style="italic", color="#222222")

    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
