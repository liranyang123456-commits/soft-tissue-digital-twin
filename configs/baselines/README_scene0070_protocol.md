# Scene 0070 same-input inverse-rendering protocol

This protocol first renders all cameras under one fixed distant illumination. It then
uses train views `1,2,3,5,6,7,9,10,11` and test views `0,4,8`. Every method
receives only sRGB images, foreground masks, camera intrinsics, and camera
poses. Diffuse, specular, albedo, normal, depth, BRDF, point-cloud, and
lighting ground truth are isolated under `evaluation_only`.

Prepare the data:

```powershell
python -m scripts.make_scene0070_constant_light `
  --checkpoint outputs/world_benchmark_v2_scene_0070_strict/last.pt `
  --source ../mvbrdf_shr/data/world_benchmark_v2/scenes/test/scene_0070 `
  --output data/world_benchmark_v2_constant_light/scene_0070 `
  --light-frame 1

python scripts/prepare_scene0070_inverse_baselines.py `
  --source data/world_benchmark_v2_constant_light/scene_0070 `
  --output data/baseline_protocols/scene0070_rgb_only
```

Train the compared methods with:

```powershell
python -m mvbrdf_shr.world.train `
  --config configs/world/benchmark_scene_0070_rgb_only.yaml

python train.py `
  --config ../../configs/baselines/nvdiffrec_scene0070_rgb_only.json

python train.py `
  --config ../../configs/baselines/nvdiffrecmc_scene0070_rgb_only.json
```

The two NVIDIA commands are run from their respective repositories under
`baselines_ext` in the `kaolin` environment and a Visual Studio x64 developer
shell. NeRFactor uses its original NeRF, geometry extraction, shape
pretraining, and joint factorization stages. It runs under WSL with
`TF_USE_LEGACY_KERAS=1`; this preserves the TensorFlow 2 graph behavior
expected by the released code.

Evaluate every exported prediction with
`scripts/evaluate_scene0070_inverse_baseline.py`. The common report contains
foreground-masked sRGB PSNR/SSIM, per-channel scale-aligned albedo metrics,
and mean normal angular error. Full-frame PSNR printed by the original
baseline trainers is not substituted for the common foreground metric.
