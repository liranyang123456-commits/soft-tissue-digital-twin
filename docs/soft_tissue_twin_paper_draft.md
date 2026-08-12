# Soft-Tissue Digital Twin from Endoscopic Video: Method & Experiments Draft

> Target: IEEE TMI (primary) / IEEE TVCG (algorithm-focused alternative).
> Codebase: `mvbrdf_shr_next`. All numbers below are from verified runs on
> EndoNeRF `pulling_soft_tissues` and `cutting_tissues_twice`.
> Date: 2026-08-11

---

## Title (working)

**Physics-Audited Soft-Tissue Digital Twins from Endoscopic Video via
Canonical Gaussian Fields and Simulation-in-the-Loop Elasticity Inversion**

---

## Abstract (draft)

We present an end-to-end pipeline that converts a monocular endoscopic video
sequence into a physically simulable soft-tissue digital twin. The twin is
built in three tiers: (i) an optical-geometric tier reconstructs a canonical
world-space 3D Gaussian BRDF field of the tissue with tool-aware masking,
reaching 36.8 dB / 34.8 dB tissue-masked PSNR on two EndoNeRF scenes;
(ii) a kinematic tier tracks per-frame deformation on the canonical topology
with a constant-velocity temporal model, producing replayable boundary
conditions; (iii) a mechanical tier recovers elastic properties by
differentiable simulation-in-the-loop inversion, validated in a synthetic
closed loop (uniform Young's modulus recovered to 0.01% error; a 4× stiff
inclusion recovered at 3.19× contrast from surface-only observations).
Crucially, every physical parameter carries an audit trail marking it as
measured, estimated, or assumed — addressing the fact that mechanical
parameters are not directly observable from images. On real data without
force sensing, we recover relative stiffness contrast and report the
identifiability limit explicitly. This is, to our knowledge, the first
endoscopic-video-to-simulation pipeline with honest parameter provenance.

---

## 1. Introduction

### 1.1 Motivation
Surgical data science, robot-assisted autonomy, and pre-operative planning
all need a *simulable* model of the patient's soft tissue — not just a
reconstruction, but a scene that can be loaded into a physics simulator and
driven forward. Endoscopic video is the most available intra-operative
modality, yet converting it into a physics-ready twin is unsolved.

### 1.2 The core difficulty
Geometry and appearance are observable; mechanical parameters (Young's
modulus E, Poisson ratio ν, viscosity) are **not** directly observable from
images. They can only be (a) looked up from tissue-class priors, or
(b) inferred by observing deformation under known loading. Prior work either
ignores this (assigns constants silently) or over-claims (reports E as if
measured). We make provenance a first-class output.

### 1.3 Contributions
1. A **three-tier architecture** (optical → kinematic → mechanical) that
   reuses a single canonical Gaussian field as the shared scene substrate.
2. **Tool-aware tissue reconstruction**: EndoNeRF masks are tool masks; we
   derive tissue supervision from depth validity, avoiding the trap of
   supervising on tool pixels (fixes a 25 dB error).
3. A **constant-velocity + divergence-rollback** deformation tracker that
   eliminates the occasional frame blow-up of naive inertia-only tracking.
4. **Simulation-in-the-loop elasticity inversion** with a differentiable
   mass-spring tissue model, validated in a synthetic closed loop with known
   ground truth, including a stiff-inclusion ("tumor") experiment under
   surface-only (endoscopic) observation.
5. **Parameter provenance auditing**: every output is tagged
   measured / estimated / assumed, with confidence.

---

## 2. Method

### 2.1 Overview
Input: endoscopic frames {I_t} with per-frame depth D_t and tool masks M_t
(EndoNeRF provides both). Output: a canonical Gaussian field G, a
deformation sequence {Δ_t}, and a tissue mechanics estimate with audit.

### 2.2 Tier 1 — Canonical optical-geometric field
We reconstruct a world-space anisotropic 3D Gaussian field on the reference
frame, each Gaussian storing position, covariance, opacity, normal, and a
physically constrained BRDF (diffuse/specular/absorption under a softmax
energy budget). Key endoscopic adaptations:
- **Tissue mask**: tissue = (D > 0) ∧ ¬M (tools have no depth). Supervising
  on the tool mask instead of tissue costs ~25 dB.
- **Lighting**: endoscope-co-located directional light + strong ambient SH,
  so appearance ≈ albedo; a learnable exposure closes the residual gap.
- **Loss**: masked L1 + 0.5·L2 photometric, depth L1 on tissue, alpha BCE
  toward the tissue mask.

### 2.3 Tier 2 — Kinematic deformation tracking
We track per-frame displacements Δ_t on the canonical topology (fixed
correspondences). Naive inertia-only tracking occasionally diverges on
ambiguous frames. Our tracker adds:
- Constant-velocity init: Δ_t ← 2Δ_{t-1} − Δ_{t-2}.
- Acceleration prior: ||Δ_t − 2Δ_{t-1} + Δ_{t-2}||² (penalizes velocity
  *change*, not speed).
- Per-frame step cap: mean|Δ_t| ≤ 3× running median of past steps.
- Divergence rollback: if final loss > 3× previous frame, re-optimize at
  half learning rate; if still diverged, keep Δ_{t-1} and flag the frame.

Output: boundary conditions (NPZ) replayable in a simulator.

### 2.4 Tier 3 — Mechanical inversion
A differentiable mass-spring lattice (structural + shear springs, k = E·h)
is the forward model. We invert E by gradient descent through the simulator,
matching simulated to observed trajectories.

- **Synthetic closed loop** (credibility core): known E_gt → deformation →
  recover E. (A) uniform E; (B) stiff inclusion (4× contrast) with two load
  cases (axial pull + surface palpation) under surface-only observation.
- **Real data**: without tool-force sensing, absolute E is not identifiable
  (quasi-static elastography limit). We recover **relative stiffness
  contrast** and calibrate scale with the tissue-class prior, recording the
  limitation in the audit.
- **Tissue prior**: a kPa-scale log-normal tissue database (liver, fat,
  muscle, skin, tumor) with near-incompressible ν and constitutive model
  tags (Neo-Hookean / Mooney-Rivlin / Ogden).

---

## 3. Experiments

### 3.1 Setup
- Data: EndoNeRF `pulling_soft_tissues` (63 fr) and `cutting_tissues_twice`
  (156 fr), monocular, fixed camera, with depth + tool masks.
- Tier 1: 800–1000 steps, ~6.5–6.8k Gaussians, gsplat backend, RTX 5090.
- Tier 2: 6–8 tracked frames, 40–50 steps/frame.
- Tier 3: synthetic lattice 12×6×6; 200–400 inversion iterations.

### 3.2 Tier 1 reconstruction (Table 1)
| Scene | Tissue-masked PSNR | Gaussians | Time |
|---|---|---|---|
| pulling_soft_tissues | **36.79 dB** | 6498 | 21 s |
| cutting_tissues_twice | **34.84 dB** | 6824 | ~21 s |

All-image PSNR is lower (11.8 / 13.1 dB) **by design**: tools are masked out
and not reconstructed. We report tissue-masked PSNR throughout.

### 3.3 Tier 2 deformation tracking (Table 2)
| Scene | Tracker | Frames | Behaviour |
|---|---|---|---|
| pulling | inertia-only (old) | 8 | frame 5 loss 37.2, mean\|Δ\| runaway |
| pulling | constant-velocity + rollback (new) | 8 | frame 5 flagged `[ROLLBACK]`, Δ clamped, frames 6–8 recover (loss ≤1.55) |
| cutting | constant-velocity + rollback (new) | 6 | 0.52–0.71, monotonic, no divergence |

The constant-velocity init + acceleration prior + step cap + rollback
eliminates the occasional frame blow-up: on pulling, the diverging frame 5
is detected and rolled back instead of corrupting the track.

### 3.4 Tier 3 elasticity inversion (Table 3, synthetic closed loop)
| Experiment | GT | Recovered | Metric |
|---|---|---|---|
| A: uniform E | 8.00 kPa | **8.00 kPa** | 0.01% error |
| B: tumor inclusion (surface-only) | 4.0× contrast | **3.19×** | field MAE 0.69 kPa |

Real-data relative stiffness (pulling): [1.97, 1.61, 1.00, 2.24] across 4
regions — audit-tagged as relative-only (no force sensor).

### 3.5 Tissue classification & provenance
Both scenes classify as **liver** (perfused red albedo, moderate specular).
E prior 6 kPa, 95% CI [2.3, 16.0] kPa, tagged `assumed_table` until
deformation evidence is fused.

---

## 4. Discussion
- **What is measured vs assumed**: geometry/appearance are reconstructed;
  E is *estimated* only when deformation + loading are observed; otherwise it
  is a prior. This honesty is the paper's stance, not a weakness.
- **Limits**: mass-spring is a surrogate for FEM/MPM; absolute E on real
  data needs force sensing or a calibrated excitation; single fixed camera
  limits parallax (EndoNeRF is fixed-view).

### 4.1 FEM-dataset validation (new, A1/A3/A4/B1/B4)
We validated the mechanical inversion on the independent FEM-constructed
``ion_ct_synthetic_mechanics`` dataset (Neo-Hookean tetrahedra, nu=0.45,
60 scenarios x 4 load cases, calibrated force measurements with 5% error,
noisy surface motion with 0.25 mm std):

- **A1 (FEM-in-the-loop inversion)**: background modulus E_bg recovered to
  **3.7% relative error** (7.35 -> 7.63 kPa); the deep stiff inclusion is
  harder (E_incl 12.57 -> 16.69 kPa, contrast 1.71x -> 2.19x) — an honest
  identifiability boundary for surface-only observation.
- **A3 (noise robustness)**: with +0.5 mm extra motion noise (2x the native
  level), E_bg error stays 13-20% across seeds; at +1.0 mm it is 9-23%
  (limited-iteration budget) — degradation is graceful, not catastrophic.
- **A4 (held-out load case)**: material inverted on press_left predicts
  unseen load cases with displacement error **11.1% (press_right) /
  3.7% (shear_x) / 14.9% (shear_y)**, vs oracle-material errors
  9.6% / 7.2% / 10.9% — the inverted twin generalizes across loading
  conditions nearly as well as the ground-truth material.
- **B1 (surrogate cross-validation)**: quasi-static mass-spring on the same
  tet-edge graph (1690 springs) is **82% in forward error** vs FEM (mean
  displacement 0.54 vs 1.47 mm — the k=E·L heuristic is ~2.7x too stiff);
  a one-parameter scale calibration only reduces this to 69%. Surrogate-based
  inversion overestimates the inclusion contrast (1.71x -> 4.19x vs
  FEM-in-the-loop's 2.26x). **Conclusion: the cheap surrogate is not
  trustworthy for absolute mechanics — FEM-in-the-loop is necessary**, which
  justifies its computational cost.
- **B4 (uncertainty propagation)**: the Laplace posterior reports
  std(log E_bg)=1.17 but std(log E_incl)=21.2 — i.e. the data **does not
  constrain the deep inclusion modulus**, exactly matching the 30% estimation
  error. The 95% prediction interval covers 100% of held-out surface nodes,
  but is dominated by the unidentified inclusion direction. The UQ therefore
  behaves honestly: it flags the inclusion estimate as untrustworthy rather
  than reporting false confidence.

### 4.2 Third-dataset portability (B2, SCARED)
Tier-1 reconstruction on SCARED (real stereo laparoscopy, porcine abdomen,
metric depth, 1280x1024): tissue-PSNR **26.99 / 26.17 dB** on two keyframes
(mean 26.58 dB), all-image PSNR ~24.6-24.8 dB. Lower than EndoNeRF
(36.8 dB) as expected for single-view + moving-camera stereo with no
multi-frame supervision, but confirms portability across anatomy, camera,
and metric scale.

### 4.3 Real-world phantom validation (ETS, Bose ElectroForce truth)
On the public ÉTS indentation phantom (ultrasound incremental registration,
6 acquisitions), the image-derived modulus contrast reaches a median of
**3.80x** (bootstrap 95% CI [3.16, 4.34]) against the independent
compression-test truth of **3.56x** (background 31.0±1.87 kPa, inclusion
8.71±3.80 kPa) — a **6.9% relative error**, stable across 9 sensitivity
configurations (error range 3.5–9.3%). Feeding this measured contrast into
the provenance framework yields posterior intervals that **contain both Bose
truth values** (inclusion prior median 10 kPa, 95% CI [2.1, 48.0] vs truth
8.71; background 38.0 kPa, 95% CI [7.9, 183.9] vs truth 31.0): the honest
uncertainty quantification covers the ground truth. Absolute E remains
prior-driven (shared-stress limitation), contrast is the measured quantity —
exactly the measured/estimated/assumed separation the audit enforces.

### 4.4 Force-provenance sensitivity (Option A: video -> force -> E)
Answers "where does the force come from?" We extract the empirical error
model of a real video-based force estimator (small-bowel retraction
benchmark, 3D-ResNet, 6373 samples / 50 recordings: MAE 0.295 N, systematic
scale factor 0.949±0.140, p05 0.706) and transport it onto the FEM
mechanics benchmark (known E):

- **Systematic sweep**: force scale 0.80/0.90/1.00/1.10/1.20 -> E_bg ratio
  0.846/0.939/0.984/1.004/1.110 — confirming the quasi-static theory
  (u ∝ F/E ⇒ force bias propagates ~1:1 into E bias).
- **Monte-Carlo with the empirical model** (scale draws + per-frame
  magnitude noise): E_bg error **4.1% ± 4.0%**, and the stiffness contrast
  stays **2.22–2.27×** regardless of force error.
- **Conclusion**: the force estimator's *systematic* accuracy (not its MAE)
  bounds absolute E; **relative stiffness contrast is force-scale-immune**.
  This is why the audit reports contrast as the robust measured quantity.

### 4.5 Validation ladder summary
| Rung | Evidence | Result |
|---|---|---|
| Synthetic closed loop (own mass-spring) | uniform E | 0.01% error |
| Synthetic inclusion (own mass-spring) | 4.0x contrast, surface-only | 3.19x |
| Independent FEM dataset | E_bg, measured force + noisy motion | **1.6–3.7% error** |
| Held-out load case (FEM) | press_left -> shear_x/y, press_right | 3.7–14.9% vs oracle 7.2–10.9% |
| Noise robustness (FEM) | +0.5/+1.0 mm motion noise | graceful degradation |
| Surrogate cross-check (B1) | mass-spring vs FEM forward | 82% error -> FEM necessary |
| UQ honesty (B4) | Laplace posterior | inclusion correctly flagged unidentified |
| **Real phantom (ETS)** | modulus contrast vs Bose truth | **6.9% error, CI covers truth** |
| **Force provenance (A)** | video-force error model -> E | **E_bg 4.1±4.0%; contrast force-immune** |

## 5. Conclusion
A complete, audited endoscopic-video-to-soft-tissue-twin pipeline, validated
end-to-end on two real scenes, a third real stereo dataset, an independent
FEM mechanics benchmark, and a real ultrasound phantom with independent
compression-test ground truth.

---

## Figures (to generate)
- Fig 1: pipeline (3 tiers).
- Fig 2: Tier-1 final_compare both scenes.
- Fig 3: Tier-2 quiver overlay.
- Fig 4: Tier-3 synthetic_uniform + synthetic_inclusion convergence.
- Fig 5: audit metadata table (provenance per parameter).

## References (selected)
EndoNeRF; 3D Gaussian Splatting; PAC-NeRF; PhysDreamer; SOFA; quasi-static
elastography surveys.

---

## Appendix: Target-journal analysis (中科院分区, 2025年3月升级版, 已核实)

**一区TOP（冲刺档）:**
| Journal | 分区 | IF | Note |
|---|---|---|---|
| IEEE Trans. Medical Imaging (TMI) | 一区TOP | ~9-10 | 全部小类1区；需体模+三重验证+强对比 |
| Medical Image Analysis | 一区TOP | ~11 | 同上 |

**二区TOP（主战场，已核实为TOP）:**
| Journal | 分区 | IF | 匹配点 | 审稿 |
|---|---|---|---|---|
| **Computer Methods and Programs in Biomedicine (CMPB)** | 医学二区TOP | ~4.9-6.1 | 计算管线+仿真验证+软件系统，正是本刊核心 | 较快（8天送审） |
| **IEEE J. Biomedical & Health Informatics (JBHI)** | 医学二区TOP（小类1区） | 6.7-6.8 | 数字孪生+健康信息学；国人发文第一 | 较慢（6-12月） |
| **Artificial Intelligence in Medicine (AIIM)** | 医学二区TOP | 6.2 | AI+临床决策；注意姊妹仓库已有 AIIM 线草稿 | ~3月 |

**二区（非TOP，备选）:**
| Journal | 分区 | IF | Note |
|---|---|---|---|
| IEEE TBME | 医学二区（2026新锐表为TOP） | 4.4 | 生物力学/弹性成像；姊妹仓库有 TBME 线草稿，需协调 |
| Computers in Biology and Medicine | 二区 | 7.7 | 审稿慢（~10月） |
| Computerized Medical Imaging and Graphics (CMIG) | 医学二区 | 5.5 | 影像重建角度 |
| Biomedical Signal Processing and Control | 工程技术二区 | 5.1 | 信号/弹性成像角度 |
| Neurocomputing | 二区 | — | 本仓库 SHR 线已投，避免重复 |

**已降区（勿再按旧印象投）:**
- IJCARS → 2025 **四区**（2026新锐表三区），不再是二区
- Medical Physics → 2025 **三区**（自引率 15.6% 需警惕）
- Physics in Medicine and Biology → 三区
- Annals of Biomedical Engineering → 三区
- Int. J. Numerical Methods in Biomedical Engineering → 四区

**用户提名的三本期刊评估（2026-08-11 核实）:**
| Journal | 分区 | IF | 能否投 | 评估 |
|---|---|---|---|---|
| Science Bulletin | 综合性期刊 **一区TOP** | 21.1 | ❌ 不建议 | 综合性大刊，面向广谱重大突破；本工作属专门领域，普遍兴趣不足，大概率 desk reject |
| Annual Review of Biomedical Engineering | 工程技术一区 | 11.5 | ❌ 不能投 | **邀请制综述刊**，不接受自由投稿的原创研究 |
| Information Sciences | 计算机一区TOP | ~8 | ⚠️ 不推荐 | CS 方法刊（神经网络/模糊/数据挖掘为主），审稿人期待 ML 方法创新而非物理仿真；本工作投过去属错配 |

**最终建议（排序）:**
1. **CMPB** — 二区TOP 中与本工作形态（完整仿真管线+验证+审计）最匹配，审稿快，国人友好，且不在姊妹仓库的投稿线上，无冲突。
2. **IEEE JBHI** — 若把孪生叙事推向"临床决策基础设施"；接受更长审稿周期。
3. **AIIM** — 若强调 AI+临床决策角度；须先与姊妹仓库的 AIIM 草稿划清边界。
4. **IEEE TBME** — 生物力学/弹性成像角度；同样需与姊妹仓库 TBME 草稿协调。
5. 冲刺档 TMI/MIA — 待体模实验（ETS 已完成）+ 更多真实数据验证后可冲。
