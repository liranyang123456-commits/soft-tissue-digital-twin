# DTU / BlendedMVS Evaluation Protocol

DTU and BlendedMVS are multi-view reconstruction datasets, not paired
highlight-removal datasets. They must not be mixed into the SHIQ/SSHR/PSD/NSH
diffuse-PSNR table.

## Supported input

`MultiViewScene.from_mvsnet` reads the native BlendedMVS/MVSNet layout:

```text
PID/
  blended_images/00000000.jpg
  cams/00000000_cam.txt
  rendered_depth_maps/00000000.pfm
```

The loader reads calibrated intrinsics/extrinsics and unprojects rendered depth
maps to initialize the world-space Gaussian cloud. Holdout images are excluded
from projected texture and material initialization.

## Metrics

For datasets without diffuse GT, report only:

- held-out full-radiance PSNR/SSIM (`render.full` versus observed RGB);
- train/holdout frame IDs;
- geometry coverage and optional DTU point-cloud distance;
- qualitative world diffuse/specular decomposition.

Do not label held-out RGB reconstruction as highlight-removal PSNR. Diffuse
PSNR/SSIM is reported only when a separately captured or rendered diffuse GT is
available.

## Commands

```powershell
python -m mvbrdf_shr.world.train --config configs/world/blendedmvs.yaml
python -m mvbrdf_shr.world.evaluate `
  --checkpoint outputs/blendedmvs_scene/last.pt --split holdout
```

DTU official SampleSet requires conversion of its calibration/images into
COLMAP or MVSNet format before using the same protocol.
