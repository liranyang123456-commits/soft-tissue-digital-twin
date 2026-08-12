"""Compare multiple prediction folders against the same GT (PSNR / SSIM / LPIPS-opt)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def _load(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img).astype(np.float64) / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return 99.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Channel-averaged SSIM (Gaussian window approx via local 11x11 means)."""
    def _ssim_gray(x: np.ndarray, y: np.ndarray) -> float:
        c1, c2 = (0.01) ** 2, (0.03) ** 2
        # box filter via cumulative sum for speed
        k = 11
        pad = k // 2
        xp = np.pad(x, pad, mode="reflect")
        yp = np.pad(y, pad, mode="reflect")
        # integral images
        def box(arr: np.ndarray) -> np.ndarray:
            integ = np.pad(arr, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
            H, W = x.shape
            out = np.empty_like(x)
            for i in range(H):
                for j in range(W):
                    i0, j0 = i, j
                    i1, j1 = i + k, j + k
                    out[i, j] = (
                        integ[i1, j1] - integ[i0, j1] - integ[i1, j0] + integ[i0, j0]
                    ) / (k * k)
            return out

        mu_x, mu_y = box(xp), box(yp)
        # crop pad region already handled by indexing into padded — simplify:
        # use numpy convolve style
        kernel = np.ones((k, k)) / (k * k)
        from numpy.lib.stride_tricks import sliding_window_view

        def mean_filter(arr: np.ndarray) -> np.ndarray:
            sw = sliding_window_view(arr, (k, k))
            return sw.mean(axis=(-2, -1))

        mx = mean_filter(xp)
        my = mean_filter(yp)
        mx2 = mean_filter(xp * xp)
        my2 = mean_filter(yp * yp)
        mxy = mean_filter(xp * yp)
        sx = mx2 - mx * mx
        sy = my2 - my * my
        sxy = mxy - mx * my
        num = (2 * mx * my + c1) * (2 * sxy + c2)
        den = (mx * mx + my * my + c1) * (sx + sy + c2)
        return float((num / (den + 1e-12)).mean())

    vals = [_ssim_gray(a[:, :, c], b[:, :, c]) for c in range(3)]
    return float(np.mean(vals))


def eval_folder(pred_dir: Path, gt_dir: Path) -> dict:
    scores_p, scores_s = [], []
    gt_files = sorted(list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg")))
    for gt in gt_files:
        # match by stem; also try replacing _D -> _A naming
        candidates = [
            pred_dir / gt.name,
            pred_dir / gt.name.replace("_D", "_A"),
            pred_dir / (gt.stem + ".png"),
        ]
        pred = next((c for c in candidates if c.exists()), None)
        if pred is None:
            continue
        g = _load(gt)
        p = _load(pred, size=g.shape[1::-1])
        scores_p.append(psnr(p, g))
        scores_s.append(ssim(p, g))
    n = len(scores_p)
    return {
        "n": n,
        "psnr": float(np.mean(scores_p)) if n else float("nan"),
        "ssim": float(np.mean(scores_s)) if n else float("nan"),
    }


# Literature reference numbers (Neural DRM Solver ICCV'25 Table 1 protocol)
LITERATURE = {
    "neural_drm": {"shiq_psnr": 34.501, "shiq_ssim": 0.979, "note": "ICCV'25 SOTA"},
    "dhan_shr": {"shiq_psnr": 33.810, "shiq_ssim": 0.975, "note": "MM'24"},
    "highlightrnet": {"shiq_psnr": 30.231, "shiq_ssim": 0.930, "note": "MM'24"},
    "tshrnet": {"shiq_psnr": 25.575, "shiq_ssim": 0.933, "note": "ICCV'23"},
    "jshdr": {"shiq_psnr": 34.131, "shiq_ssim": 0.860, "note": "CVPR'21"},
    "specularitynet": {"shiq_psnr": 23.420, "shiq_ssim": 0.920, "note": "TMM"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument(
        "--methods",
        default="neural_drm,dhan_shr,highlightrnet,tshrnet,jshdr,specularitynet,mvbrdf_shr",
    )
    ap.add_argument("--out", default="outputs/baselines/summary.csv")
    ap.add_argument(
        "--include-literature",
        action="store_true",
        help="Also print literature SHIQ numbers for methods without local preds",
    )
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_root = Path(args.pred_root)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = []
    print(f"{'method':<16} {'N':>5} {'PSNR':>8} {'SSIM':>8}  note")
    print("-" * 56)
    for m in methods:
        pred_dir = pred_root / m
        if pred_dir.exists() and gt_dir.exists():
            r = eval_folder(pred_dir, gt_dir)
            note = "local"
            print(f"{m:<16} {r['n']:5d} {r['psnr']:8.3f} {r['ssim']:8.4f}  {note}")
            rows.append({"method": m, **r, "note": note})
        elif args.include_literature and m in LITERATURE:
            lit = LITERATURE[m]
            print(
                f"{m:<16} {'—':>5} {lit['shiq_psnr']:8.3f} {lit['shiq_ssim']:8.4f}  "
                f"lit:{lit['note']}"
            )
            rows.append(
                {
                    "method": m,
                    "n": 0,
                    "psnr": lit["shiq_psnr"],
                    "ssim": lit["shiq_ssim"],
                    "note": f"literature:{lit['note']}",
                }
            )
        else:
            print(f"{m:<16} {'miss':>5}")
            rows.append({"method": m, "n": 0, "psnr": "", "ssim": "", "note": "missing"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n", "psnr", "ssim", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
