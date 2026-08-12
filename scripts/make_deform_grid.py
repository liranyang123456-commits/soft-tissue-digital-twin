"""Assemble a 2-scene Tier-2 deformation figure (Fig 4): pulling + cutting."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "submission" / "cmpb" / "figures"
OUT = FIG / "fig_deform_grid.png"

SCENES = [
    ("EndoNeRF pulling", FIG / "fig_tier2_quiver.png"),
    ("EndoNeRF cutting",
     ROOT / "outputs" / "soft_tissue_twin_cutting" / "tier2_deformation" / "deformation_quiver.png"),
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=150)
    for ax, (label, path) in zip(axes, SCENES):
        if path.exists():
            ax.imshow(Image.open(path))
            ax.set_title(f"Tier-2 tracked deformation — {label}", fontsize=11)
        else:
            ax.text(0.5, 0.5, f"{label}\n(pending)", ha="center", va="center",
                    transform=ax.transAxes)
        ax.axis("off")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
