"""Assemble a 4-row reconstruction-result grid (Fig 3) for the CMPB paper.

Rows: EndoNeRF pulling, EndoNeRF cutting, SCARED kf1, SCARED kf2.
Columns: input | reconstruction | tissue-only | error map.
Uses the per-scene final_compare images (input|recon|tissue) and computes a
per-row error map from them. Writes
``submission/cmpb/figures/fig_recon_grid.png``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "submission" / "cmpb" / "figures"
OUT = FIG / "fig_recon_grid.png"

# (row label, compare image path, psnr)
ROWS = [
    ("EndoNeRF pulling", FIG / "fig_tier1_pulling.png", "36.8"),
    ("EndoNeRF cutting", FIG / "fig_tier1_cutting.png", "34.1"),
    ("SCARED kf1", FIG / "fig_scared_kf1.png", "27.0"),
    ("SCARED kf2", FIG / "fig_scared_kf2.png", "26.2"),
    # kf3--kf5 have no checked summary json. Do not print a PSNR for them.
    ("SCARED kf3", FIG / "fig_scared_kf3.png", ""),
    ("SCARED kf4", FIG / "fig_scared_kf4.png", ""),
    ("SCARED kf5", FIG / "fig_scared_kf5.png", ""),
]


def split3(img: np.ndarray):
    """Split a 3-panel compare image (input|recon|tissue) into thirds."""
    h, w = img.shape[:2]
    t = w // 3
    return img[:, :t], img[:, t:2 * t], img[:, 2 * t:]


def main():
    n_rows = len(ROWS)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13, 2.5 * n_rows), dpi=150)
    col_titles = ["input", "reconstruction", "tissue only", "error map"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=12)

    n_done = 0
    for i, (label, path, psnr) in enumerate(ROWS):
        if not path.exists():
            for j in range(4):
                axes[i, j].text(0.5, 0.5, f"{label}\n(pending)", ha="center",
                                va="center", transform=axes[i, j].transAxes)
                axes[i, j].axis("off")
            continue
        img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        inp, rec, tis = split3(img)
        err = np.abs(inp - tis).mean(axis=-1)
        err = err / max(err.max(), 1e-6)
        panels = [inp, rec, tis, err]
        for j, p in enumerate(panels):
            if j == 3:
                axes[i, j].imshow(p, cmap="hot", vmin=0, vmax=1)
            else:
                axes[i, j].imshow(p)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(f"{label}\n{psnr} dB", fontsize=10,
                                      rotation=0, ha="right", va="center",
                                      labelpad=10)
        n_done += 1

    fig.suptitle("Tier-1 reconstruction across datasets "
                 "(input | reconstruction | tissue only | error map)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0.06, 0, 1, 0.99])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT} ({n_done}/4 rows present)")


if __name__ == "__main__":
    main()
