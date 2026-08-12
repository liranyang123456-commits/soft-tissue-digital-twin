# CMPB Submission Package

Target journal: **Computer Methods and Programs in Biomedicine** (Elsevier,
CAS Q2 TOP, JCR Q1).

## Manuscript

**Physics-Audited Soft-Tissue Digital Twins from Endoscopic Video:
Canonical Gaussian Fields, Simulation-in-the-Loop Elasticity Inversion,
and Parameter Provenance**

Compiled: `main.pdf` — elsarticle `review` format (single column,
double-spaced, line numbers), **40 pages**, matching CMPB initial-submission
requirements. **37 references, all cited.**

## Structure

- Structured abstract (Objective / Methods / Results / Conclusion)
- Introduction (continuous prose, no subsections): motivation, observability
  gap, position, contributions, organization
- Related work: medical digital twins, endoscopic reconstruction,
  physics-aware reconstruction, elastography, differentiable simulation,
  gaps and our response
- Method: the soft-tissue digital twin (Definition 1), canonical scene
  reconstruction, deformation tracking, mechanical parameter inversion,
  force-scale propagation (Proposition 1 + proof), the force estimator
  (network diagram + Eq.), the provenance audit (Definition 2), using the
  twin, network-free remark (Remark)
- Experiments: setup (datasets / metrics / comparison / implementation),
  the complete twin result (Fig. twin_complete), then per-stage results —
  canonical reconstruction, deformation tracking, mechanical inversion
  (staged validation), uncertainty honesty, comparison with related
  approaches, component-wise comparison, threats to validity, ablations
- Discussion (continuous prose): measured-vs-assumed, a worked example,
  limits and failure modes, clinical translation, ethics and governance,
  reproducibility
- Conclusion
- Declaration of competing interest; Data and code availability (GitHub)

## Figures (13)

pipeline (TikZ), model architecture (result-driven), force-estimator
network (PlotNeuralNet), the complete digital twin (input→twin→output),
reconstruction grid (7 rows), deformation 4x4, synthetic closed loop
(uniform + inclusion), FEM per-scenario, FEM distribution, mechanics
robustness, material recovery (4x4), ETS phantom, hospital CT liver.

## Tables (12)

position vs related work, notation, tissue priors, datasets, reconstruction,
reconstruction comparison, deformation-tracking ablation, staged validation,
per-scenario FEM, mechanics comparison, ablations, (and one more).

## Verified results (all reproducible from released scripts, `--seed 0`)

| Claim | Value | Source artifact |
|---|---|---|
| Reconstruction PSNR (pulling) | 36.78 dB | `outputs/soft_tissue_twin_seeded/` |
| Reconstruction PSNR (cutting) | 34.09 dB | `outputs/soft_tissue_twin_cutting_seeded/` |
| Reconstruction PSNR (SCARED) | 26.99 / 26.17 dB | `outputs/scared_tier1/scared_summary.json` |
| Deformation worst-frame loss | 37.23 → 1.55 | `outputs/ablation_table.json` |
| Uniform modulus inversion | 0.01% error | `outputs/soft_tissue_twin_seeded/` |
| Inclusion contrast (synthetic) | 4.0× → 3.19× | `outputs/soft_tissue_twin_seeded/` |
| FEM E_bg error | median 4.3% / mean 9.4% (6 scenarios) | `outputs/fem_validation/` |
| Held-out load case | 11.1 / 3.7 / 14.9% vs oracle | `outputs/fem_validation/` |
| Force sweep | E_bg ratio 0.846–1.110 | `outputs/force_propagation/` |
| Force MC E_bg error | 4.1 ± 4.0% | `outputs/force_propagation/` |
| ETS phantom contrast | 3.80× vs 3.56× (6.9%) | `ets_phantom_eval.json` |

## Files in this package

- `main.tex` / `main.pdf` — manuscript (40 pp review format)
- `references.bib` — 37 references
- `figures/` — all figure sources (TikZ `.tex` + `.pdf`, and `.png`)
- `cover_letter.txt` — cover letter
- `highlights.txt` — highlights (5 bullets)
- `author_contributions.txt` — CRediT author contributions

## Checklist

- [x] elsarticle `review` format (single column, double-spaced, line numbers)
- [x] 40 pages, 0 undefined refs, 0 significant overfull
- [x] All numbers match seeded, reproducible artifacts (`--seed 0`)
- [x] References complete and cited (37 refs)
- [x] 13 figures, 12 tables, 12 equations
- [x] Proposition 1 + proof, 2 Definitions, 2 Remarks
- [x] Structured abstract, highlights, cover letter, author contributions
- [x] Data and code availability statement with GitHub link
- [x] No "tier/rung/ladder" jargon; academic stage names throughout
- [ ] Final proofread by co-authors
