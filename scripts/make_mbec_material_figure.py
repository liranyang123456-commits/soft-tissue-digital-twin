"""Optical-material figure for MBEC.

Re-lays out the 4x4 material grid (input | albedo | roughness | specular for
two EndoNeRF scenes and two SCARED keyframes): the generic title is removed
and each row receives its scene label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "submission" / "mbec" / "figures"
SOURCE = FIGURES / "fig_material_4x4.png"
OUT = FIGURES / "fig_material.png"
ROW_BANDS = ((139, 544), (608, 1013), (1077, 1482), (1546, 1951))
ROW_LABELS = ("EndoNeRF pulling", "EndoNeRF cutting", "SCARED keyframe 1", "SCARED keyframe 2")


def font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


ROWS = (0, 2)
GAP = 24


def main() -> None:
    source = Image.open(SOURCE).convert("RGB")
    header = 58
    margin = 70
    bands = [ROW_BANDS[k] for k in ROWS]
    height = sum(bottom - top for top, bottom in bands) + GAP * (len(bands) - 1)
    canvas = Image.new("RGB", (source.width + margin, height + header), "white")
    draw = ImageDraw.Draw(canvas)
    typeface = font(34)

    gray = np.asarray(source.convert("L"))[ROW_BANDS[0][0] : ROW_BANDS[0][1]]
    occupied = (gray < 245).any(axis=0)
    edges = np.flatnonzero(np.diff(occupied.astype(int)))
    starts, stops = edges[::2] + 1, edges[1::2] + 1
    for (start, stop), text in zip(zip(starts, stops), ("input", "albedo", "roughness", "specular")):
        width = draw.textlength(text, font=typeface)
        draw.text((margin + (start + stop) / 2 - width / 2, 10), text, fill="#111827", font=typeface)
    y = header
    for k, (top, bottom) in zip(ROWS, bands):
        canvas.paste(source.crop((0, top, source.width, bottom)), (margin, y))
        label = Image.new("RGBA", (bottom - top, margin), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label)
        width = label_draw.textlength(ROW_LABELS[k], font=typeface)
        label_draw.text(((label.width - width) / 2, 12), ROW_LABELS[k], fill="#111827", font=typeface)
        rotated = label.rotate(90, expand=True)
        canvas.paste(rotated, (0, y), rotated)
        y += bottom - top + GAP
    canvas.save(OUT, dpi=(300, 300))
    print(OUT, canvas.size)


if __name__ == "__main__":
    main()
