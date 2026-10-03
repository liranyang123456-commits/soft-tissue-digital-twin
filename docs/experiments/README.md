# Strengthening experiment protocols

## EndoNeRF common holdout

The frozen split matches the official EndoGaussian loader:

```text
holdout: (frame_id - 1) % 8 == 0
train:   all remaining frame IDs
```

The present method reconstructs frame 0, tracks training frames only, and
linearly interpolates tracked displacement for holdout times. Holdout
images are used only by `evaluate_endonerf_common.py`.

Commands:

```powershell
python scripts/run_endonerf_holdout_twin.py ...
python scripts/evaluate_endonerf_common.py ...
python scripts/run_endogaussian_common.py ...
python scripts/run_endonerf_common_baseline.py ...
python scripts/aggregate_endonerf_common.py
```

Metrics are full-image and tissue-masked PSNR/SSIM on identical frame IDs.
Current present-method aggregate: 28 held-out frames, tissue PSNR
16.714 dB, tissue SSIM 0.7026. External baseline rows are added only after
the official implementation completes.

## FEM60

`run_fem60_campaign.py` freezes the protocol used in the earlier
six-scenario run:

- press-left load for inversion;
- measured force with calibration error;
- noisy surface displacement;
- frames 1, 2, and 4;
- 20 optimization iterations;
- deterministic lexical ordering of all 60 scenarios.

Each scenario is written independently, so interrupted shards resume
without recomputation.

## Patient-specific volumetric mapping

`build_gaussian_volume_map.py` constructs a tetrahedral liver volume from
82 de-identified CT slices, performs trimmed similarity ICP, and maps
covered volume-boundary nodes to tracked Gaussians with inverse-distance
weights.

The current cross-subject demonstration contains 28,523 nodes, 149,028
tetrahedra, 7,111 boundary nodes, and 2,755 mapped boundary nodes
(38.74% coverage). Because the CT and endoscopic data are from different
subjects and CT spacing is unavailable, this is a software/geometry
demonstration rather than anatomical or clinical validation.
