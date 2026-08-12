# Public-dataset optimization track

This directory is the post-submission development line. It reuses the single
data tree at `../data` and must not copy public datasets into this project.
Package modules under `mvbrdf_shr/data/*.py` are tracked; only dataset
directories (`data/raw`, `data/generated`, etc.) remain ignored.

Implemented components:

1. Equal-mass SHIQ/SSHR/NSH/PSD/synthetic sampling with domain IDs.
2. Dataset-conditioned uncertainty fusion between physical diffuse and the
   image restorer.
3. Additive decomposition consistency, heteroscedastic uncertainty,
   highlight-region, boundary, non-highlight identity, and texture losses.
4. Adapter-first then low-rate joint domain adaptation.
5. Camera-response, exposure, white-balance, blur, noise, JPEG, material and
   lighting randomization for generated paired data.
6. Image-wise public-suite evaluation with highlight/non-highlight/boundary
   metrics and uncertainty-error correlation.
7. A matched pure-image UNet trained with the same data sampler, augmentation,
   optimization budget and evaluation implementation.

Generate synthetic pretraining data:

```powershell
python scripts/generate_domain_randomized_synth.py `
  --out "../data/generated/domain_randomized_shr"
```

Train the matched pure-image baseline:

```powershell
python scripts/train_public_image_baseline.py `
  --config configs/train/public_multidataset.yaml
```

Train the proposed model:

```powershell
python scripts/train_public_multidataset.py `
  --config configs/train/public_multidataset.yaml `
  --resume outputs/shiq_mse/best.pt
```

Domain-adapt a mixed checkpoint by changing `datasets` and `roots` in
`configs/train/public_domain_adapt.yaml`, then:

```powershell
python scripts/train_public_multidataset.py `
  --config configs/train/public_domain_adapt.yaml `
  --resume outputs/public_multidataset/best.pt `
  --output outputs/public_domain_adapt
```

Evaluate under one metric implementation:

```powershell
python scripts/evaluate_public_suite.py `
  --checkpoint ours=outputs/public_multidataset/best.pt `
  --checkpoint image=outputs/public_image_baseline/best.pt
```

Report zero-shot and adapted checkpoints separately. PSD adaptation results
must use `ft_test`; the full `test` split is reserved for zero-shot reporting.

Run the complete campaign (full tests, 14k randomized synthetic pairs, matched
mixed training, matched SSHR/NSH/PSD adaptation, zero-shot/adapted evaluation,
and four full-budget component ablations):

```powershell
python scripts/run_public_campaign.py
```

The proposed checkpoint is explicitly reported as `ours_pretrained`: it starts
from the existing SHIQ checkpoint, while the matched image model starts from
scratch. The report therefore discloses this initialization difference instead
of presenting it as an equal-total-compute comparison. Both models use the same
10k mixed-training steps and the same 2k steps for each adaptation experiment.
