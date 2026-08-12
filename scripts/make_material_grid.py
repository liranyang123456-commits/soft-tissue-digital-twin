"""Multi-row material-recovery grid (Fig 9) for the CMPB paper.

Rows: EndoNeRF pulling, EndoNeRF cutting, SCARED kf1 (where available).
Columns: input | recovered albedo | recovered roughness | (params on pulling).
Writes ``submission/cmpb/figures/fig_material_grid.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "submission" / "cmpb" / "figures"
OUT = FIG / "fig_material_grid.png"

# (row label, material 3-panel image, params text or None)
ROWS = [
    ("EndoNeRF pulling", FIG / "fig_material_pulling_3panel.png", "liver, Neo-Hookean, E 6.0 kPa (assumed)"),
    ("EndoNeRF cutting", FIG / "fig_material_cutting.png", "liver, Neo-Hookean, E 6.0 kPa (assumed)"),
]


def split3(img: np.ndarray):
    h, w = img.shape[:2]
    t = w // 3
    return img[:, :t], img[:, t:2 * t], img[:, 2 * t:]


def main():
    n_rows = len(ROWS)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.4 * n_rows), dpi=150)
    if n_rows == 1:
        axes = axes[None, :]
    col_titles = ["input", "recovered albedo", "recovered roughness"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=12)

    for i, (label, path, params) in enumerate(ROWS):
        if not path.exists():
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f"{label}\n(pending)", ha="center",
                                va="center", transform=axes[i, j].transAxes)
                axes[i, j].axis("off")
            continue
        img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        inp, alb, rough = split3(img)
        panels = [inp, alb, rough]
        for j, p in enumerate(panels):
            axes[i, j].imshow(p)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(f"{label}\n{params}", fontsize=8.5,
                                      rotation=0, ha="right", va="center",
                                      labelpad=8)

    fig.suptitle("Soft-tissue material recovery across scenes (input | "
                 "recovered albedo | recovered roughness)", fontsize=13)
    fig.tight_layout(rect=[0.10, 0, 1, 0.98])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
