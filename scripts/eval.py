"""Simple PSNR/SSIM eval helper (read-only on GT dirs)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    # Lightweight structural proxy (not full SSIM); enough for smoke eval.
    a_g = a.mean(axis=2)
    b_g = b.mean(axis=2)
    mu_a, mu_b = a_g.mean(), b_g.mean()
    sig_a, sig_b = a_g.var(), b_g.var()
    sig_ab = ((a_g - mu_a) * (b_g - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(
        ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2))
        / ((mu_a ** 2 + mu_b ** 2 + c1) * (sig_a + sig_b + c2) + 1e-8)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--gt-dir", required=True)
    args = p.parse_args()
    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    scores = []
    for gt in sorted(gt_dir.glob("*.png")):
        pred = pred_dir / gt.name
        if not pred.exists():
            continue
        g = np.asarray(Image.open(gt).convert("RGB")).astype(np.float32) / 255.0
        pr = np.asarray(Image.open(pred).convert("RGB").resize(g.shape[1::-1])).astype(
            np.float32
        ) / 255.0
        scores.append((psnr(pr, g), ssim_simple(pr, g)))
    if not scores:
        print("no matching pairs")
        return
    ps = np.mean([s[0] for s in scores])
    ss = np.mean([s[1] for s in scores])
    print(f"N={len(scores)}  PSNR={ps:.3f}  SSIM~={ss:.4f}")


if __name__ == "__main__":
    main()
