"""Result-driven horizontal architecture figure (Fig 2) for the CMPB paper.

Each stage shows its ACTUAL result image, with the method label below the
image and arrows between stages. No "TIER" text. Horizontal layout.
Writes ``submission/cmpb/figures/fig_architecture.png``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "submission" / "cmpb" / "figures"
OUT = FIG / "fig_architecture.png"


def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def crop_third(img, k):
    """Take the k-th third of a 3-panel compare image."""
    h, w = img.shape[:2]
    t = w // 3
    return img[:, k * t:(k + 1) * t]


def main():
    # gather result images
    recon = crop_third(load(FIG / "fig_tier1_pulling.png"), 1)  # reconstruction
    deform = load(FIG / "fig_tier2_quiver.png")                  # deformation quiver
    material = crop_third(load(FIG / "fig_material_pulling_3panel.png"), 1)  # albedo
    rehearsal = load(FIG / "fig_twin_rehearsal_single.png")      # twin driven forward
    input_img = crop_third(load(FIG / "fig_tier1_pulling.png"), 0)  # input

    stages = [
        ("Endoscopic video", input_img),
        ("Canonical scene\nreconstruction", recon),
        ("Deformation\ntracking", deform),
        ("Mechanical\ninversion", material),
        ("Simulable twin\n+ rehearsal", rehearsal),
    ]

    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(16, 4.6), dpi=150)
    for i, (label, img) in enumerate(stages):
        ax = axes[i]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(1.4)
        ax.set_xlabel(label, fontsize=10, fontweight="bold")
        ax.xaxis.set_label_position("bottom")
        # arrow to next
        if i < n - 1:
            ax.annotate(
                "", xy=(1.06, 0.5), xytext=(1.0, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6),
            )

    fig.suptitle("One canonical Gaussian field carries appearance, motion, "
                 "and mechanics; the audit tags every parameter",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
