"""A2: assemble the comparison table (reconstruction + mechanics).

Reconstruction numbers for EndoNeRF-family methods are taken from their
publications (novel-view-synthesis protocol); ours is a canonical-field
single-frame fit — a DIFFERENT protocol, marked explicitly. Mechanics
numbers come from PAC-NeRF / PhysDreamer papers, the sibling repository's
TMI-line results (60-scene FEM benchmark), and this repository's runs.
The table is written to ``outputs/comparison_table.json`` and a markdown
copy for the paper draft.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

RECONSTRUCTION = [
    # method, venue, protocol, PSNR, SSIM, source
    ("EndoNeRF", "MICCAI'22", "NVS (per-scene, 2 held-out views)", 37.61, 0.955, "Wang et al. 2022, Tab.1 (mean of 2 scenes)"),
    ("EndoSurf", "MICCAI'23", "NVS (per-scene)", 37.24, 0.958, "Zha et al. 2023, Tab.1"),
    ("EndoGaussian", "AAAI'24", "NVS (per-scene)", 37.89, 0.962, "Liu et al. 2024, Tab.1"),
    ("LGS", "IROS'24", "NVS (per-scene)", 37.42, 0.959, "Zhu et al. 2024"),
    ("Prior work (Diff_Rending_Re_3D)", "—", "NVS (EndoNeRF, 14-frame train)", 24.80, 0.762, "results/FINAL_REPORT.md (mean of 2 scenes)"),
    ("Ours Tier-1 (pulling)", "—", "canonical fit, tissue-masked", 36.79, None, "outputs/soft_tissue_twin"),
    ("Ours Tier-1 (cutting)", "—", "canonical fit, tissue-masked", 34.93, None, "outputs/soft_tissue_twin_cutting"),
]

MECHANICS = [
    # method, venue, setting, E error / contrast, source
    ("PAC-NeRF", "ICLR'24", "system ID from multi-view video, known force", "E err ~10-30% (object-dependent)", "Li et al. 2024"),
    ("PhysDreamer", "ECCV'24", "video-diffusion prior, static objects", "relative stiffness only", "Zhang et al. 2024"),
    ("Prior TMI-line (60-scene FEM bench)", "—", "image+force joint inversion", "E err 30.2% mean / 21.3% median", "Diff_Rending_Re_3D results/FINAL_REPORT.md"),
    ("Ours synthetic closed loop (mass-spring)", "—", "uniform E, known force", "E err 0.01%", "outputs/soft_tissue_twin"),
    ("Ours synthetic inclusion (mass-spring)", "—", "2-region, surface-only obs", "contrast 4.0x -> 3.19x", "outputs/soft_tissue_twin"),
    ("Ours FEM-dataset inversion", "—", "2-region, measured force (5% err) + noisy motion", "see fem_validation_summary.json", "outputs/fem_validation"),
]


def main():
    doc = {
        "reconstruction": [
            {
                "method": m, "venue": v, "protocol": p,
                "psnr_db": psnr, "ssim": ssim, "source": src,
            }
            for m, v, p, psnr, ssim, src in RECONSTRUCTION
        ],
        "mechanics": [
            {"method": m, "venue": v, "setting": s, "result": r, "source": src}
            for m, v, s, r, src in MECHANICS
        ],
        "honesty_notes": [
            "EndoNeRF-family PSNRs are novel-view synthesis on held-out views; "
            "ours is a canonical-field tissue-masked fit. Protocols differ; "
            "the table is indicative, not a leaderboard.",
            "Mechanics rows span different identifiability settings (known "
            "force vs force-free, absolute E vs relative contrast).",
        ],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "comparison_table.json").write_text(
        json.dumps(doc, indent=2), encoding="utf-8"
    )

    md = ["# Comparison tables (A2)\n", "## Reconstruction\n",
          "| Method | Venue | Protocol | PSNR | SSIM |", "|---|---|---|---|---|"]
    for m, v, p, psnr, ssim, _ in RECONSTRUCTION:
        md.append(f"| {m} | {v} | {p} | {psnr if psnr else '—'} | {ssim if ssim else '—'} |")
    md += ["\n## Mechanics\n",
           "| Method | Venue | Setting | Result |", "|---|---|---|---|"]
    for m, v, s, r, _ in MECHANICS:
        md.append(f"| {m} | {v} | {s} | {r} |")
    md += ["", "*Notes:* " + " ".join(doc["honesty_notes"])]
    (OUT / "comparison_table.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT / 'comparison_table.json'} and .md")


if __name__ == "__main__":
    main()
