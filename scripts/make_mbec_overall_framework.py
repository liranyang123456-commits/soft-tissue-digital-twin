"""Create the consolidated MBEC digital-twin framework figure.

The figure combines the end-to-end visual sequence of the former
architecture figure with the force-evidence branch of the former R3D-18
figure.  It deliberately separates the core twin from the auxiliary force
estimator, which is used only for sensitivity analysis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "submission" / "mbec" / "figures"


def box(
    axis,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    detail: str,
    color: str,
    *,
    title_size: float = 10.5,
    detail_size: float = 8.2,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.2,
        edgecolor="#334155",
        facecolor=color,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="#0f172a",
    )
    axis.text(
        x + width / 2,
        y + height * 0.30,
        detail,
        ha="center",
        va="center",
        fontsize=detail_size,
        color="#1e293b",
        linespacing=1.15,
    )


def arrow(axis, start: tuple[float, float], end: tuple[float, float], label: str = "") -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#475569",
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        axis.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.016,
            label,
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#334155",
        )


def main() -> None:
    architecture = Image.open(FIGURES / "fig_architecture_results.png").convert("RGB")

    figure = plt.figure(figsize=(14.8, 8.4), dpi=180, facecolor="white")
    grid = figure.add_gridspec(2, 1, height_ratios=(0.78, 1.35), hspace=0.08)

    visual = figure.add_subplot(grid[0])
    visual.imshow(architecture)
    visual.set_axis_off()
    visual.text(
        0.005,
        1.02,
        "(a) Observable sequence and twin outputs",
        transform=visual.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#0f172a",
    )

    axis = figure.add_subplot(grid[1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.005,
        0.985,
        "(b) Co-registered computational graph and force-dependent identifiability",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        color="#0f172a",
    )

    y = 0.53
    h = 0.25
    w = 0.155
    xs = (0.015, 0.205, 0.395, 0.585, 0.775)
    box(
        axis,
        (xs[0], y),
        w,
        h,
        "Endoscopic\nevidence",
        r"RGB $I_t$, depth $D_t$," "\n" r"tool mask $M_t$",
        "#dbeafe",
    )
    box(
        axis,
        (xs[1], y),
        w,
        h,
        "Gaussian surface\nand BRDF",
        "geometry, opacity,\n"
        "albedo, roughness, specular\n"
        "tool-excluded supervision",
        "#dcfce7",
    )
    box(
        axis,
        (xs[2], y),
        w,
        h,
        "Fixed-identity\nmotion",
        r"displacement $\Delta_t$" "\n"
        "graph regularization\n"
        "prediction + rollback",
        "#fef3c7",
    )
    box(
        axis,
        (xs[3], y),
        w,
        h,
        "Surface-to-volume\ncoupling",
        "trimmed similarity ICP\n"
        "inverse-distance transfer\n"
        "tetrahedral boundary data",
        "#ede9fe",
    )
    box(
        axis,
        (xs[4], y),
        w,
        h,
        "Differentiable\nmechanics",
        "Neo-Hookean FEM\n"
        r"$E$, contrast, stress/strain" "\n"
        "forward load simulation",
        "#fee2e2",
    )
    for left, right in zip(xs[:-1], xs[1:]):
        arrow(axis, (left + w, y + h / 2), (right, y + h / 2))

    # Force-evidence branch, condensed from the former R3D-18 architecture.
    box(
        axis,
        (0.585, 0.12),
        0.17,
        0.19,
        "Force evidence",
        "measured force: absolute modulus\n"
        "unknown force: contrast only",
        "#f8fafc",
        title_size=10,
        detail_size=7.8,
    )
    box(
        axis,
        (0.79, 0.12),
        0.16,
        0.19,
        "Auxiliary R3D-18",
        r"video clip $\rightarrow$ 3D ResNet" "\n"
        r"$\rightarrow\,\hat f$ error model" "\n"
        "sensitivity analysis only",
        "#fff7ed",
        title_size=10,
        detail_size=7.7,
    )
    arrow(axis, (0.79, 0.215), (0.755, 0.215))
    arrow(axis, (0.67, 0.31), (0.83, y), "force scale")

    box(
        axis,
        (0.015, 0.09),
        0.49,
        0.23,
        "Audited simulable twin",
        r"$\mathcal{T}=(G,\{\Delta_t\},\Theta,\mathcal{A})$" "\n"
        "shared tissue identity links rendering, motion, and mechanics\n"
        "each parameter: measured / estimated / assumed + uncertainty",
        "#e0f2fe",
        title_size=11,
        detail_size=8.2,
    )
    axis.add_patch(
        FancyArrowPatch(
            (xs[4] + w * 0.72, y),
            (0.49, 0.32),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#475569",
            connectionstyle="arc3,rad=-0.18",
        )
    )

    axis.text(
        0.015,
        0.015,
        "Core contribution: an explicit observation-to-simulation interface; "
        "the renderer, R3D-18 backbone, and constitutive law are established components.",
        fontsize=8.4,
        color="#334155",
        ha="left",
        va="bottom",
    )

    output_png = FIGURES / "fig_overall_framework.png"
    output_pdf = FIGURES / "fig_overall_framework.pdf"
    figure.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(output_png)
    print(output_pdf)


if __name__ == "__main__":
    main()
