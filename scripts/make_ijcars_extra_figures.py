"""IJCARS reconstruction grid and synthetic closed-loop figures (no in-figure titles)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submission" / "mbec" / "figures"
OUT = ROOT / "submission" / "ijcars" / "figures"
ROWS = [
    ("EndoNeRF pulling\n36.8 dB", SRC / "fig_tier1_pulling.png"),
    ("EndoNeRF cutting\n34.1 dB", SRC / "fig_tier1_cutting.png"),
    ("SCARED keyframe 1\n27.0 dB", SRC / "fig_scared_kf1.png"),
    ("SCARED keyframe 2\n26.2 dB", SRC / "fig_scared_kf2.png"),
]


def recon_grid() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(len(ROWS), 4, figsize=(7.0, 1.42 * len(ROWS) + 0.3), dpi=300)
    for j, title in enumerate(["input", "reconstruction", "tissue only", "|input - tissue only|"]):
        axes[0, j].set_title(title, fontsize=8)
    for i, (label, path) in enumerate(ROWS):
        img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        third = img.shape[1] // 3
        inp, rec, tis = img[:, :third], img[:, third:2 * third], img[:, 2 * third:3 * third]
        err = np.abs(inp - tis).mean(axis=-1)
        err = err / max(float(err.max()), 1e-6)
        for j, panel in enumerate([inp, rec, tis, err]):
            ax = axes[i, j]
            ax.imshow(panel, cmap="hot" if j == 3 else None, vmin=0 if j == 3 else None, vmax=1 if j == 3 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[i, 0].set_ylabel(label, fontsize=7.5, rotation=90, labelpad=4)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.95, bottom=0.01, wspace=0.03, hspace=0.06)
    fig.savefig(OUT / "Fig4.png", dpi=300, facecolor="white")
    plt.close(fig)


def synthetic_loop() -> None:
    base = ROOT / "outputs" / "soft_tissue_twin_seeded" / "tier3_elasticity"
    panels = []
    for name in ("synthetic_uniform.png", "synthetic_inclusion.png"):
        im = Image.open(base / name).convert("RGB")
        arr = np.asarray(im)
        dark = (arr.mean(axis=2) < 200)
        rows = np.where(dark.any(axis=1))[0]
        first = rows[0]
        gap = first
        while gap < arr.shape[0] and dark[gap].any():
            gap += 1
        im = im.crop((0, gap + 4, im.width, im.height))
        panels.append(im)
    width = max(p.width for p in panels)
    height = sum(p.height for p in panels) + 10
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height + 10
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    except OSError:
        font = ImageFont.truetype("arialbd.ttf", 26)
    half = width // 2
    for k, (x, yy) in enumerate([(4, 2), (half + 4, 2), (4, panels[0].height + 12), (half + 4, panels[0].height + 12)]):
        draw.text((x, yy), "abcd"[k], fill="black", font=font)
    sheet.save(OUT / "Fig7.png", dpi=(150, 150))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    recon_grid()
    synthetic_loop()
    print("ok")