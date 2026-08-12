# Physics-Audited Soft-Tissue Digital Twins from Endoscopic Video

Official code for the paper:

> **Physics-Audited Soft-Tissue Digital Twins from Endoscopic Video:
> Canonical Gaussian Fields, Simulation-in-the-Loop Elasticity Inversion,
> and Parameter Provenance.**
> Ranyang Li, Nan Wei, Zhipeng Lin, Wufeng Liu, Chao Fan, Junjun Pan.
> *Computer Methods and Programs in Biomedicine* (under review), 2026.

The pipeline converts a monocular endoscopic video into a **physically
simulable soft-tissue digital twin** in which every physical parameter
carries an explicit, auditable **provenance** (measured / estimated /
assumed, with confidence).

```
endoscopic video ──► canonical Gaussian field ──► deformation tracking
                   ──► mechanical inversion ──► simulable, audited twin
```

## What the twin contains

A twin is `T = (G, {Δ_t}, Θ, A)`:

| Symbol | Meaning | Provenance |
|---|---|---|
| `G` | canonical Gaussian BRDF field (geometry + appearance) | measured |
| `{Δ_t}` | deformation sequence on the fixed topology | measured |
| `Θ` | elastic / viscous / inertial parameters | measured / estimated / assumed |
| `A` | audit map: a provenance tag + confidence per parameter | — |

## Installation

```bash
git clone https://github.com/liranyang123456-commits/soft-tissue-digital-twin.git
cd mvbrdf_shr_next
pip install -e .
# CUDA rasterizer (recommended; falls back to a PyTorch reference otherwise)
pip install -e ".[cuda]"
```

Requires Python ≥ 3.10 and PyTorch. Tested on Windows / Linux with an
NVIDIA GPU (RTX 5090).

## Reproduce the paper

Every number in the paper is produced by a released end-to-end script with
a fixed seed (`--seed 0`).

```bash
# Full three-stage twin on an EndoNeRF scene (reconstruction + tracking +
# mechanical inversion + audit). Writes a machine-readable summary.
python scripts/run_soft_tissue_twin.py --scene pulling_soft_tissues --seed 0
python scripts/run_soft_tissue_twin.py --scene cutting_tissues_twice --seed 0

# Portability test on the SCARED stereo dataset
python scripts/run_scared_tier1.py

# FEM-benchmark mechanical validation (staged validation)
python scripts/run_fem_validation_suite.py

# Force-provenance study (force error -> modulus error)
python scripts/run_force_propagation.py

# Real phantom (ETS) audit demo
python scripts/run_ets_audit_demo.py

# Hospital CT liver geometry + classification check
python scripts/run_hospital_ct_liver.py
```

Figures are assembled by the `scripts/make_*.py` scripts into
`submission/cmpb/figures/`.

## Datasets

Datasets are **not** redistributed here (size / licence). Please obtain them
from the original sources and place them under `datasets/`:

- **EndoNeRF** — <https://github.com/med-air/EndoNeRF>
- **SCARED** — <https://endovissub2019-scared.grand-challenge.org/>
- **ETS phantom** — Borealis, DOI `10.5683/SP3/ASTGWY`
- **Hospital CT** — subject to institutional consent; available on request.

## Repository layout

```
mvbrdf_shr/            Python package
  world/               digital-twin modules
    scene.py           canonical Gaussian BRDF field
    renderer.py        differentiable rasterizer (gsplat / torch)
    deform.py          deformation tracking (const-velocity + rollback)
    tissue_sim.py      differentiable mass-spring / FEM inversion
    physics_est.py     Bayesian parameter estimation + audit
    tissue_db.py       soft-tissue material priors
scripts/               end-to-end reproduction + figure scripts
configs/               experiment configs
submission/cmpb/       the manuscript (LaTeX) and figures
tests/                 unit tests
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2026twin,
  title   = {Physics-Audited Soft-Tissue Digital Twins from Endoscopic Video:
             Canonical Gaussian Fields, Simulation-in-the-Loop Elasticity
             Inversion, and Parameter Provenance},
  author  = {Li, Ranyang and Wei, Nan and Lin, Zhipeng and Liu, Wufeng and
             Fan, Chao and Pan, Junjun},
  journal = {Computer Methods and Programs in Biomedicine},
  year    = {2026},
  note    = {under review}
}
```

## License

[MIT](LICENSE). The force estimator used as a prior in the force-provenance
study is prior work and is not part of this contribution.

---

*This repository also contains the earlier MVBRDF-SHR specular-highlight
removal line; see `docs/README_shr_legacy.md` for that documentation.*
