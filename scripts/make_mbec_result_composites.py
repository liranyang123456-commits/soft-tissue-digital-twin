"""Assemble page-efficient result figures for the official MBEC template."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "submission" / "mbec" / "figures"
WIDTH = 2200
MARGIN = 55
LABEL_HEIGHT = 72


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    )
    selected = next((path for path in candidates if path.exists()), None)
    if selected is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(selected), size=size)


def fit(image: Image.Image, width: int) -> Image.Image:
    ratio = width / image.width
    return image.resize((width, round(image.height * ratio)), Image.Resampling.LANCZOS)


def labelled_panel(image: Image.Image, label: str, title: str) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + LABEL_HEIGHT), "white")
    panel.paste(image, (0, LABEL_HEIGHT))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 10), f"({label})", fill="#111827", font=font(34, True))
    draw.text((78, 10), title, fill="#111827", font=font(32, True))
    return panel


def stack(panels: list[Image.Image], gap: int = 35) -> Image.Image:
    height = MARGIN * 2 + sum(panel.height for panel in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (WIDTH, height), "white")
    y = MARGIN
    for panel in panels:
        x = (WIDTH - panel.width) // 2
        canvas.paste(panel, (x, y))
        y += panel.height + gap
    return canvas


def reconstruction_material() -> None:
    comparison = fit(
        Image.open(FIGURES / "endonerf_common_comparison.png").convert("RGB"),
        1700,
    )
    material = fit(
        Image.open(FIGURES / "fig_material_4x4.png").convert("RGB"),
        WIDTH - 2 * MARGIN,
    )
    output = stack(
        [
            labelled_panel(
                comparison,
                "a",
                "Strict common-holdout reconstruction metrics",
            ),
            labelled_panel(
                material,
                "b",
                "Optical-material decomposition across four endoscopic scenes",
            ),
        ],
        gap=28,
    )
    output.save(FIGURES / "fig_reconstruction_material_results.png", dpi=(300, 300))


def tracking() -> None:
    source = Image.open(FIGURES / "fig_deform_4x4.png").convert("RGB")
    source = source.crop((0, 58, source.width, source.height))
    source = fit(source, WIDTH - 2 * MARGIN)
    output = stack(
        [
            labelled_panel(
                source,
                "a",
                "Fixed-identity motion: displacement, rendering, and error",
            )
        ]
    )
    output.save(FIGURES / "fig_tracking_results.png", dpi=(300, 300))


def mechanics() -> None:
    fem = fit(Image.open(FIGURES / "fem60_summary.png").convert("RGB"), 1500)
    robustness = fit(
        Image.open(FIGURES / "fig_robustness.png").convert("RGB"),
        WIDTH - 2 * MARGIN,
    )
    phantom = fit(Image.open(FIGURES / "fig_ets.png").convert("RGB"), 1250)
    output = stack(
        [
            labelled_panel(fem, "a", "Complete 60-scenario FEM inversion"),
            labelled_panel(
                robustness,
                "b",
                "Force-scale, motion-noise, and held-out-load sensitivity",
            ),
            labelled_panel(
                phantom,
                "c",
                "Physical phantom contrast against compression testing",
            ),
        ],
        gap=26,
    )
    output.save(FIGURES / "fig_mechanics_results.png", dpi=(300, 300))


def main() -> None:
    reconstruction_material()
    tracking()
    mechanics()
    for name in (
        "fig_reconstruction_material_results.png",
        "fig_tracking_results.png",
        "fig_mechanics_results.png",
    ):
        print(FIGURES / name)


if __name__ == "__main__":
    main()
