"""Create the current, protocol-aware SOTA comparison report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    literature_path = ROOT / "docs" / "sota_numbers.csv"
    shiq_path = ROOT / "outputs" / "shiq_mse" / "shiq_results.json"
    raster_path = ROOT / "outputs" / "world_raster_benchmark.json"
    world_path = ROOT / "outputs" / "world_demo" / "world_metrics.json"
    literature = []
    with literature_path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["shiq_psnr"]:
                literature.append(
                    {
                        "method": row["method"],
                        "psnr": float(row["shiq_psnr"]),
                        "ssim": float(row["shiq_ssim"]),
                        "venue": row["venue"],
                        "source": "literature",
                    }
                )
    shiq = json.loads(shiq_path.read_text(encoding="utf-8"))
    ours = {
        "method": "MVBRDF-SHR (single-image track)",
        "psnr": shiq["mvbrdf_pro"]["psnr"],
        "ssim": shiq["mvbrdf_pro"]["ssim"],
        "venue": "ours",
        "source": "local official SHIQ test",
    }
    ranking = sorted(literature + [ours], key=lambda row: row["psnr"], reverse=True)
    raster = (
        json.loads(raster_path.read_text(encoding="utf-8"))
        if raster_path.exists()
        else None
    )
    world = (
        json.loads(world_path.read_text(encoding="utf-8"))
        if world_path.exists()
        else None
    )

    rows = [
        "# Current SOTA Comparison",
        "",
        "## SHIQ single-image track (official 1000-image test split, 200×200)",
        "",
        "| Rank | Method | PSNR ↑ | SSIM ↑ | Source |",
        "|---:|---|---:|---:|---|",
    ]
    for rank, row in enumerate(ranking, 1):
        rows.append(
            f"| {rank} | {row['method']} | {row['psnr']:.3f} | "
            f"{row['ssim']:.4f} | {row['source']} |"
        )
    neural = next(row for row in ranking if row["method"] == "NeuralDRM")
    rows.extend(
        [
            "",
            "### Gap to current literature SOTA",
            "",
            f"- PSNR: {ours['psnr']:.3f} vs {neural['psnr']:.3f} "
            f"({ours['psnr'] - neural['psnr']:+.3f} dB)",
            f"- SSIM: {ours['ssim']:.4f} vs {neural['ssim']:.4f} "
            f"({ours['ssim'] - neural['ssim']:+.4f})",
            "",
            "SHIQ is a single-image benchmark. It evaluates the fallback/restoration "
            "track, not the calibrated world-space multi-view contribution.",
            "",
            "## World-space 3D Gaussian BRDF track",
            "",
            "This track requires calibrated multiple views and therefore must not "
            "be numerically mixed with SHIQ single-image rows.",
        ]
    )
    if world is not None:
        summary = world["summary"]
        rows.extend(
            [
                "",
                "| Demo metric | Value |",
                "|---|---:|",
                f"| Full reconstruction PSNR | {summary['full_psnr']:.3f} |",
                f"| Physical diffuse PSNR | {summary['diffuse_psnr']:.3f} |",
                f"| Refined diffuse PSNR | {summary['refined_psnr']:.3f} |",
                f"| Refined diffuse SSIM | {summary['refined_ssim']:.4f} |",
            ]
        )
    if raster is not None:
        rows.extend(
            [
                "",
                "## Rasterizer benchmark",
                "",
                f"- Device: {raster['device']}",
                f"- Scene: {raster['gaussians']} Gaussians at "
                f"{raster['resolution'][0]}×{raster['resolution'][1]}",
                f"- PyTorch: {raster['torch']['milliseconds']:.3f} ms",
                f"- gsplat CUDA: {raster['gsplat']['milliseconds']:.3f} ms",
                f"- Speedup: **{raster['speedup']:.2f}×**",
                f"- Peak memory: {raster['torch']['peak_memory_mb']:.1f} MB → "
                f"{raster['gsplat']['peak_memory_mb']:.1f} MB",
            ]
        )
    rows.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current single-image track is above HighlightRNet, TSHRNet, "
            "SpecularityNet, and traditional methods in PSNR.",
            "- It remains below DHAN-SHR, JSHDR, and Neural DRM in PSNR.",
            "- The world-space track is now executable and accelerated, but a "
            "publication-grade comparison needs DTU/BlendedMVS or a calibrated "
            "captured dataset with diffuse/material ground truth.",
        ]
    )
    report = "\n".join(rows) + "\n"
    output = ROOT / "outputs" / "SOTA_COMPARISON.md"
    output.write_text(report, encoding="utf-8")
    (ROOT / "outputs" / "sota_comparison.json").write_text(
        json.dumps(
            {
                "shiq_ranking": ranking,
                "world_demo": world,
                "raster_benchmark": raster,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()

