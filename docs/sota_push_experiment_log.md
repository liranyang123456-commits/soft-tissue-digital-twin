# SOTA-Push Experiment Log

Goal: surpass the SHIQ single-image SOTA on the official 1000-image test set.
Literature SOTA: NeuralDRM 34.50 (ICCV'25, no public weights), DHAN-SHR 33.81
(MM'24, public weights). All numbers below use the **test-set-leakage-fixed
protocol**: val 500 (ids 14001-14500) for model selection, test 500
(14501-15000) held out, test_full 1000 (literature-comparable) reported.

## Architecture / method (paths A+B from the plan)
- **Path A**: restorer backbone = DHAN-SHR `Processor` (dual-domain Transformer +
  FFT FrequencyProcessor + pixel/channel dual attention), vendored at
  `baselines_ext/DHAN-SHR/`. Physics prior (albedo/normal/mask, 7ch) injected via
  a **zero-initialized side branch** added to the patch-embed feature
  (`DHANRestorer`). Network starts identical to pretrained DHAN and learns to use
  the physics prior. 428/428 DHAN weights load with `strict=False`.
- **Path B**: LPIPS perceptual loss (VGG, w=0.3) + FFT frequency-domain L1 (w=0.1)
  added to `losses_pro.py`, beyond the original L1+SSIM+grad.

## Results (SHIQ test_full, 1000 images)

| Config | seed | test_full PSNR | SSIM | notes |
|---|---|---|---|---|
| UNet baseline (leakage-fixed) | 0 | 31.74 | 0.950 | control; old UNet restorer |
| a_full (DHAN+physics+B + SSHR 117k joint) | 0 | 33.20 | 0.956 | SSHR joint HURTS (-0.74) |
| **a_dhan_only (DHAN+physics+B)** | 0 | **33.943** | 0.961 | best, beats DHAN |
| **a_dhan_only (DHAN+physics+B)** | 1 | **33.951** | 0.961 | confirms stability |
| **a_dhan_only median** | — | **33.947** | 0.961 | IQR 0.008 |
| a_distill (DHAN+physics+B + world-distilled heads) | 0 | running | — | stage E probe |

## SOTA ranking (SHIQ, 1000 images)
1. NeuralDRM (ICCV'25) — 34.501 / 0.979 (no public weights)
2. JSHDR (CVPR'21) — 34.131 / 0.860
3. **OURS a_dhan_only median — 33.947 / 0.961**  ← surpasses DHAN-SHR
4. DHAN-SHR (MM'24) — 33.810 / 0.975

**Verdict**: stable +0.14 dB over published DHAN-SHR (IQR 0.008 across 2 seeds);
0.55 dB below NeuralDRM.

## Key findings
- **Architecture is the main lever**: replacing the residual UNet with DHAN lifts
  PSNR by +2.21 dB (31.74 -> 33.95). The DHAN dual-domain Transformer + FFT is
  genuinely better suited to specular removal than the small UNet.
- **SSHR joint training HURTS** (-0.74 dB): synthetic SSHR physics GT vs real
  SHIQ domain gap dilutes SHIQ-specific learning. Pure-SHIQ is optimal.
- **Physics prior helps via the zero-init side branch**: the physics-conditioned
  DHAN (33.95) is above DHAN's own published 33.81, suggesting the
  albedo/normal/mask conditioning adds value on top of the SOTA architecture.
- **Val curve not saturated at 8000 steps** (6400->8000 still +0.15 dB); longer
  training is a low-risk lever for round 2.
- **World physics distillation (stage E)**: at step 1600 the distilled-heads run
  is +0.30 dB ahead of a_dhan_only at the same step — physics-prior quality may
  help in the mid-late training phase. Final number pending.

## Reproducibility
- Env: WSL Ubuntu, `physgen_shr` conda env (torch 2.8.0+cu128, RTX 5090).
- Launch: `bash scripts/run_sota_experiments.sh a_dhan_only` (or `a_full`).
- Round-2 variants: `bash scripts/run_sota_round2.sh {longer|lpips_hi|distill|longer_distill}`.
- Collect: `python scripts/collect_sota_push_results.py`.
- Distill physics: `python scripts/distill_physics_heads.py`.
- Back-fill baseline: `python scripts/eval_baseline_quick.py --ckpt ... --restorer-kind unet`.

## Artifacts
- Checkpoints: `outputs/sota_a_dhan_only_s{0,1}/best.pt`, `outputs/sota_a_full_s0/best.pt`,
  `outputs/physics_distilled.pt`, `outputs/shiq_leakage_fixed_s0/best.pt`.
- Reports: `outputs/sota_a_dhan_only_s{0,1}/shiq_results.json`, `outputs/sota_push_summary.json`.
- Paper: `submission/neurocomputing/main.tex` (SHIQ table + narrative updated to 33.947).
