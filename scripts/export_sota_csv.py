"""Export literature SOTA numbers as CSV for paper tables."""
from __future__ import annotations

import csv
from pathlib import Path

# Neural DRM Solver (ICCV 2025) Table 1 protocol: SHIQ / PSD / SSHR
ROWS = [
    # method, shiq_psnr, shiq_ssim, psd_psnr, psd_ssim, sshr_psnr, sshr_ssim, venue, code
    ("Shen13", 13.923, 0.428, 13.886, 0.610, 24.388, 0.904, "traditional", ""),
    ("SpecularityNet", 23.420, 0.920, 21.801, 0.880, 25.731, 0.894, "TMM", "https://github.com/jianweiguo/SpecularityNet-PSD"),
    ("JSHDR", 34.131, 0.860, 21.516, 0.883, 26.979, 0.895, "CVPR'21", "https://github.com/fu123456/SHIQ"),
    ("TSHRNet", 25.575, 0.933, 22.759, 0.903, 28.633, 0.940, "ICCV'23", "https://github.com/fu123456/TSHRNet"),
    ("HighlightRNet", 30.231, 0.930, 29.787, 0.922, 30.066, 0.956, "MM'24", "https://github.com/zz0223/HRNet"),
    ("DHAN-SHR", 33.810, 0.975, 25.280, 0.883, 34.103, 0.959, "MM'24", "https://github.com/CXH-Research/DHAN-SHR"),
    ("NeuralDRM", 34.501, 0.979, 29.803, 0.943, 34.637, 0.965, "ICCV'25 SOTA", "(pending)"),
    ("MVBRDF-SHR", "", "", "", "", "", "", "ours", ""),
]


def main():
    out = Path(__file__).resolve().parents[1] / "docs" / "sota_numbers.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "shiq_psnr",
                "shiq_ssim",
                "psd_psnr",
                "psd_ssim",
                "sshr_psnr",
                "sshr_ssim",
                "venue",
                "code",
            ]
        )
        w.writerows(ROWS)
    print(f"wrote {out}")
    print("\nSHIQ leaderboard (from Neural DRM ICCV'25 Table 1):")
    ranked = sorted(
        [r for r in ROWS if isinstance(r[1], float)], key=lambda r: r[1], reverse=True
    )
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r[0]:<16} PSNR={r[1]:6.3f}  SSIM={r[2]:.3f}  [{r[7]}]")


if __name__ == "__main__":
    main()
