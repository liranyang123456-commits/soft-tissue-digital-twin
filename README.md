# Simulable Soft-Tissue Digital Twins from Endoscopic Video

Code for the manuscript

> **Simulable soft-tissue digital twins from endoscopic video: fixed-identity
> Gaussian tracking and force-scale-aware finite-element inversion.**
> Ranyang Li, Zhuoyi Li, Zhipeng Lin, Nan Wei, Chao Fan, Junjun Pan.
> Manuscript under review, 2026.

The pipeline converts an endoscopic image sequence and a separately acquired
CT volume into a soft-tissue digital twin that can be driven forward under
loads that were not observed:

1. Fixed-identity Gaussian tracking turns the endoscopic sequence into
   Lagrangian surface motion.
2. The motion is prescribed on a CT-derived Neo-Hookean finite-element mesh,
   and Young's modulus is estimated by differentiable inversion.
3. Absolute modulus is reported only when force is measured; otherwise the
   twin reports stiffness contrast. Every parameter carries a provenance tag
   (measured, estimated, or assumed) and a confidence.

## What the twin contains

A twin is `T = (G, {Δ_t}, Θ, A)`:

| Symbol | Meaning |
|---|---|
| `G` | canonical geometric and optical representation (3D Gaussian-BRDF field) |
| `{Δ_t}` | displacement sequence of the fixed-identity primitives |
| `Θ` | physical parameters and the forward operator `S: (G, Θ, f) -> u` from a load to a displacement field |
| `A` | provenance tag and confidence for every parameter |

## Installation

```bash
git clone https://github.com/liranyang123456-commits/soft-tissue-digital-twin.git
cd soft-tissue-digital-twin
pip install -e .
# CUDA rasterizer (recommended; falls back to a PyTorch reference otherwise)
pip install -e ".[cuda]"
```

Requires Python ≥ 3.10 and PyTorch. Tested on Windows / Linux with an
NVIDIA GPU (RTX 5090).

## Reproducing the reported results

Each script writes a machine-readable summary. The summaries behind the
numbers reported in the manuscript are collected in
[`results/`](results/README.md), together with the script that produced each
file. All scripts are in `scripts/`; run them with `--help` for their options.

| Result | Scripts |
|---|---|
| Canonical reconstruction, tracking, inversion, and audit on EndoNeRF | `run_soft_tissue_twin.py`, `run_endonerf_holdout_twin.py`, `aggregate_endonerf_common.py` |
| Canonical reconstruction on SCARED | `run_scared_tier1.py` |
| STIR tattoo tracking (2D on 60 clips, 3D on 59 clips) | `prepare_stir_sequences.py`, `run_stir_tracking.py`, `run_stir_query_refresh.py`, `fuse_stir_gated.py`, `run_stir_raft.py`, `run_stir_raft3d.py`, `evaluate_stir_tracking.py`, `evaluate_stir_tracking_3d.py` |
| 60-scenario finite-element benchmark | `run_fem60_campaign.py` |
| Unseen-load prediction | `fem_bg_for_pred.py`, `fem_bg_only40_expand.py`, `fem_holdout_per_load_60.py` |
| Model mismatch (constitutive law, mesh refinement) | `run_fem_model_mismatch.py`, `run_fem_mismatch_holdout.py` |
| Force-error propagation | `run_force_propagation.py` |
| Same-animal DePoLL registration | `run_depoll_registration.py`, `depoll_shape_ambiguity.py` |
| Force-free stiffness contrast on the ETS phantom | `run_ets_audit_demo.py` |
| Patient-specific CT volume and surface-to-volume mapping | `run_ct_liver_dicom.py`, `build_gaussian_volume_map.py` |
| Figures | `make_ijcars_mechanics_figure.py`, `make_ijcars_extra_figures.py`, `make_mbec_*_figure.py` |

## Datasets

Datasets are not redistributed here. Obtain them from their creators and
place them under `datasets/`:

- EndoNeRF: <https://github.com/med-air/EndoNeRF>
- SCARED: <https://endovissub2019-scared.grand-challenge.org/>
- STIR: Schmidt et al., IEEE Transactions on Medical Imaging 43(7):2634-2645, 2024, <https://doi.org/10.1109/TMI.2024.3372828>
- DePoLL: Modrzejewski et al., IJCARS 14:1237-1245, 2019, <https://doi.org/10.1007/s11548-019-02001-4>
- Small-bowel retraction force dataset: Wang et al., IJCARS 21(4):793-803, 2026, <https://doi.org/10.1007/s11548-026-03583-6>
- ETS indentation phantom: Borealis, <https://doi.org/10.5683/SP3/ASTGWY>

The hospital CT series is not shared because of patient-privacy
restrictions. The finite-element benchmark and the other experimental data
are available from the corresponding author (pan_junjun@buaa.edu.cn) upon
reasonable request.

## Repository layout

```
mvbrdf_shr/                  Python package
  world/                     digital-twin modules
    scene.py                 canonical 3D Gaussian field with BRDF materials
    renderer.py              differentiable Gaussian splatting with BRDF separation
    deform.py                fixed-identity deformation tracking with rollback
    correspondence.py        persistent 2D correspondences for tracking
    flow_init.py             optical-flow initialization
    appearance_layer.py      appearance Gaussians for tissue exposed in later frames
    volume_mapping.py        tracked surface to patient-specific tetrahedral volume
    surface_registration.py  registration of a volumetric organ model to a surface
    tissue_sim.py            differentiable soft-tissue simulation and modulus inversion
    physics_est.py           Bayesian parameter estimation and provenance audit
    tissue_db.py             soft-tissue material priors
scripts/                     reproduction and figure scripts
configs/                     experiment configurations
results/                     result summaries behind the reported numbers
tests/                       unit tests
```

## Citation

```bibtex
@misc{li2026simulable,
  title  = {Simulable soft-tissue digital twins from endoscopic video:
            fixed-identity {G}aussian tracking and force-scale-aware
            finite-element inversion},
  author = {Li, Ranyang and Li, Zhuoyi and Lin, Zhipeng and Wei, Nan and
            Fan, Chao and Pan, Junjun},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

[MIT](LICENSE). The R3D-18 force estimator that defines the force-error model
follows Wang et al. (2026) and is trained on their public small-bowel
retraction force dataset; it is not part of this repository. The aggregate
error parameters used for force-error propagation are in
`configs/force_error_model_summary.json`.

---

*This repository also contains the earlier MVBRDF-SHR specular-highlight
removal line; see `docs/README_shr_legacy.md` for that documentation.*
