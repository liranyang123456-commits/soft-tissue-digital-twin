#!/usr/bin/env bash
# SOTA-push experiment launcher (paths A + B + C combined).
#
# Runs after the leakage-fixed baseline (shiq_leakage_fixed_s0) completes.
# Three configurations, each 8000 steps, to be compared against SOTA 34.5:
#
#   1. BC_only   : current UNet + LPIPS/FFT loss + SSHR joint data + aug
#   2. A_full    : DHAN restorer + physics + LPIPS/FFT + SSHR joint + aug  (main SOTA push)
#   3. A_dhan_only: DHAN + physics + LPIPS/FFT, SHIQ-only (isolate architecture effect)
#
# Run from mvbrdf_shr/ with PYTHONPATH=. and the physgen_shr env active.
# Usage: bash scripts/run_sota_experiments.sh <config_name>
#   where <config_name> in {bc_only, a_full, a_dhan_only}
set -euo pipefail
CFG="${1:-a_full}"
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PHYSGEN_SF3D_BACKEND=stub PHYSGEN_DIFFUSION_BACKEND=tiny

case "$CFG" in
  bc_only)
    # Path B+C only (isolate loss+data effect on the existing UNet).
    python scripts/train_shiq_pro.py \
      --steps 8000 --batch-size 8 --eval-every 800 \
      --refine-lpips 0.3 --refine-fft 0.1 \
      --aux-datasets sshr \
      --output-dir outputs/sota_bc_only_s0 --seed 0
    ;;
  a_full)
    # Path A+B+C: DHAN + physics + LPIPS/FFT + SSHR joint + aug (MAIN SOTA PUSH).
    # batch 6 (DHAN 4.5M params + VGG LPIPS is heavier than the UNet).
    python scripts/train_shiq_pro.py \
      --steps 8000 --batch-size 6 --eval-every 800 \
      --restorer-kind dhan \
      --refine-lpips 0.3 --refine-fft 0.1 \
      --aux-datasets sshr \
      --output-dir outputs/sota_a_full_s0 --seed 0
    ;;
  a_dhan_only)
    # Path A+B: DHAN + physics + LPIPS/FFT, SHIQ-only (isolate architecture).
    python scripts/train_shiq_pro.py \
      --steps 8000 --batch-size 6 --eval-every 800 \
      --restorer-kind dhan \
      --refine-lpips 0.3 --refine-fft 0.1 \
      --output-dir outputs/sota_a_dhan_only_s0 --seed 0
    ;;
  *)
    echo "Unknown config: $CFG (use bc_only | a_full | a_dhan_only)"; exit 1 ;;
esac
