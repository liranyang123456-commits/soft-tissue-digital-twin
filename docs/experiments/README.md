# Strengthening experiment protocols

## Strict EndoNeRF holdout

The common split matches the official EndoGaussian loader:

```text
holdout: (frame_id - 1) % 8 == 0
train:   all remaining frame IDs
```

Both methods use the same 28 held-out frames, source images, tissue masks,
and evaluator. The present method excludes holdout images from
reconstruction and tracking, then linearly interpolates displacement at
holdout timestamps. EndoGaussian uses the official implementation at
commit `8d12793838a1595b299df0696c8149c07329e980` and its 3000-iteration
configuration.

Across the two scenes:

- EndoGaussian: tissue PSNR 36.589 dB; tissue SSIM 0.9386.
- Present method: tissue PSNR 16.714 dB; tissue SSIM 0.5640.

Tissue PSNR is computed only on selected pixels. Tissue SSIM is the mean
of the local 11-by-11 SSIM map over those pixels; excluded black regions
do not contribute. The result does not support a reconstruction-SOTA
claim for the present method.

Commands:

```powershell
python scripts/run_endonerf_holdout_twin.py ...
python scripts/evaluate_endonerf_common.py ...
python scripts/run_endogaussian_common.py ...
python scripts/aggregate_endonerf_common.py
```

## Complete FEM campaign

`run_fem60_campaign.py` uses the frozen protocol from the original
six-scenario run:

- `press_left` load for inversion;
- measured force with calibration error;
- noisy surface displacement;
- frames 1, 2, and 4;
- 20 optimization iterations;
- deterministic lexical ordering of all 60 scenarios.

Every scenario is written independently, allowing interrupted shards to
resume without recomputation. All 60 scenarios are complete:

- background-modulus error: median 9.59%, mean 9.95%;
- inclusion-modulus error: median 32.02%, mean 30.08%;
- stiffness-contrast error: median 32.44%, mean 33.83%.

The complete campaign supports moderate recovery of background stiffness
but not accurate recovery of deep-inclusion modulus or contrast.

## Gaussian surface to volumetric mesh

`build_gaussian_volume_map.py` constructs a shared-node tetrahedral liver
volume from 82 de-identified CT slices, performs trimmed similarity ICP,
and maps tracked Gaussian displacement to covered boundary nodes with
inverse-distance interpolation.

Current cross-subject demonstration:

- 28,523 volume nodes;
- 149,028 tetrahedra;
- 7,111 boundary nodes;
- 2,755 mapped boundary nodes (38.743% coverage);
- normalized median alignment distance 0.02373.

The Gaussian and CT data come from different subjects, and CT spacing is
unavailable. This result validates the software interface, not anatomical
registration or clinical validity.
