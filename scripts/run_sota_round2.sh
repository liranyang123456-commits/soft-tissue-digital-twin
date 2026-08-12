#!/usr/bin/env bash
# SOTA-push round 2: hyperparameter variants to close the 0.55 dB gap to
# NeuralDRM (34.50). Current best: a_dhan_only median 33.947 (beats DHAN 33.81).
#
# Variants (each builds on the proven a_dhan_only recipe: DHAN+physics+LPIPS/FFT,
# pure SHIQ, batch 3). Pick based on round-1 findings:
#   - longer   : 12000 steps (val curve still rising at 8000, not saturated)
#   - lpips_hi : LPIPS weight 0.5 (sharpen highlight-edge texture more)
#   - distill  : resume from world-distilled physics heads (stage E)
#
# Usage: bash scripts/run_sota_round2.sh <variant>
set -euo pipefail
V="${1:-longer}"
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PHYSGEN_SF3D_BACKEND=stub PHYSGEN_DIFFUSION_BACKEND=tiny

case "$V" in
  longer)
    # Val curve rose +0.15 dB over 6400->8000; extend to 12000 for a bit more.
    python scripts/train_shiq_pro.py \
      --steps 12000 --batch-size 3 --eval-every 800 \
      --restorer-kind dhan --refine-lpips 0.3 --refine-fft 0.1 \
      --output-dir outputs/sota_r2_longer_s0 --seed 0
    ;;
  lpips_hi)
    # Stronger perceptual loss targets the high-frequency highlight edges that
    # L1/SSIM miss (the residual gap to NeuralDRM is largely texture detail).
    python scripts/train_shiq_pro.py \
      --steps 8000 --batch-size 3 --eval-every 800 \
      --restorer-kind dhan --refine-lpips 0.5 --refine-fft 0.15 \
      --output-dir outputs/sota_r2_lpips_hi_s0 --seed 0
    ;;
  distill)
    # Stage E: resume world-distilled physics heads (better normal/albedo prior)
    # then DHAN overwrites the restorer. Tests if physics-prior quality helps.
    python scripts/train_shiq_pro.py \
      --steps 8000 --batch-size 3 --eval-every 800 \
      --restorer-kind dhan --refine-lpips 0.3 --refine-fft 0.1 \
      --resume outputs/physics_distilled.pt \
      --output-dir outputs/sota_a_distill_s0 --seed 0
    ;;
  longer_distill)
    # Combine: distilled physics heads + longer training.
    python scripts/train_shiq_pro.py \
      --steps 12000 --batch-size 3 --eval-every 800 \
      --restorer-kind dhan --refine-lpips 0.3 --refine-fft 0.1 \
      --resume outputs/physics_distilled.pt \
      --output-dir outputs/sota_r2_longer_distill_s0 --seed 0
    ;;
  *)
    echo "Unknown variant: $V (use longer | lpips_hi | distill | longer_distill)"; exit 1 ;;
esac
