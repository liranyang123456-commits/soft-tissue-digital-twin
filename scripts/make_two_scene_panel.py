"""Two-scene summary panel: pulling + cutting side by side."""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "outputs" / "soft_tissue_twin"          # pulling
B = ROOT / "outputs/soft_tissue_twin_cutting"      # cutting


def psnr_tissue(compare_png: Path) -> float:
    import numpy as np
    from PIL import Image
    img = np.array(Image.open(compare_png)).astype(np.float32) / 255
    h, w, _ = img.shape
    t = w // 3
    tgt, masked = img[:, :t], img[:, 2 * t:]
    tissue = masked.sum(-1) > 0.05
    mse = np.mean((tgt[tissue] - img[:, t:2 * t][tissue]) ** 2)
    return float(20 * np.log10(1 / max(mse**0.5, 1e-6)))


def main():
    fig = plt.figure(figsize=(15, 11), dpi=140)

    for row, (d, name) in enumerate([(A, "pulling_soft_tissues"),
                                     (B, "cutting_tissues_twice")]):
        p = psnr_tissue(d / "tier1_reconstruction" / "final_compare.png")
        ax = fig.add_subplot(3, 2, row * 2 + 1)
        ax.imshow(mpimg.imread(d / "tier1_reconstruction" / "final_compare.png"))
        ax.set_title(f"{name} — Tier 1 (tissue-PSNR {p:.1f} dB)", fontsize=10)
        ax.axis("off")

        ax = fig.add_subplot(3, 2, row * 2 + 2)
        ax.imshow(mpimg.imread(d / "tier2_deformation" / "deformation_quiver.png"))
        ax.set_title(f"{name} — Tier 2 deformation", fontsize=10)
        ax.axis("off")

    ax = fig.add_subplot(3, 1, 3)
    ax.imshow(mpimg.imread(A / "tier3_elasticity" / "synthetic_inclusion.png"))
    ax.set_title("Tier 3 — synthetic closed loop (tumor inclusion, surface-only)",
                 fontsize=10)
    ax.axis("off")

    fig.tight_layout()
    out = ROOT / "outputs" / "two_scene_summary.png"
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
