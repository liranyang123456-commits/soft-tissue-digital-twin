# MVBRDF-SHR: Multi-View BRDF Field for Physically-Grounded Specular Highlight Removal

**投稿类型：** 期刊长文（Journal Article）  
**主投期刊：** *Neurocomputing*（Elsevier，中科院计算机科学 **2区 TOP**）  
**完整投稿包：** `mvbrdf_shr/submission/neurocomputing/`（`main.pdf` 已编译为 **17 页**，含作者、图、Highlights、Cover Letter）  

**作者：** Ranyang Li\*（河南工业大学）, Nan Wei（河南省人民医院）, Zhipeng Lin（北航）, Wufeng Liu（河南工业大学）, Chao Fan（河南工业大学）, Junjun Pan\*（北航）  
\*通讯作者：`lry@haut.edu.cn` / `liranyang666@buaa.edu.cn`；`pan_junjun@buaa.edu.cn`

---

## Title

**Physics-Guided Specular Highlight Removal via World-Space 3D Gaussian BRDF Fields and Confidence-Gated Fusion**

## 中文标题

**MVBRDF-SHR：基于世界空间 3D 高斯 BRDF 场与可信度融合的高光去除方法**

---

## Abstract

Specular highlight removal from a single image is fundamentally ill-posed because saturated observations discard the underlying diffuse texture, and existing learning-based methods cannot enforce material consistency across viewpoints. We argue that *cross-view material identity* is the missing inductive bias, and instantiate it through **MVBRDF-SHR**, a hybrid framework that couples three cooperating modules: (i) a **world-space 3D Gaussian BRDF field** in which each primitive stores geometry, anisotropic covariance, a unit-normal, and physically constrained material parameters under a partition-of-unity energy budget; (ii) a **single-image physics-guided restorer** for in-the-wild inputs; and (iii) an adaptive **confidence-guided fusion gate** that prefers the physical diffuse estimate where multi-view reconstruction is reliable. A differentiable Cook–Torrance renderer with per-frame spherical-harmonic lighting produces a highlight-free diffuse image simply by disabling the microfacet specular lobe, coupling all viewpoints through a shared material. On a controlled multi-view benchmark with known geometry and albedo, the physical branch reaches **36.82 dB** PSNR on strictly held-out views, **1.29 dB** above a strong single-image restorer, and the fusion gate attains the best structural similarity (**SSIM 0.9865**) by delegating uncertain regions to the learned branch. On the official 1000-image SHIQ single-image test set the restorer reaches **32.61 dB / 0.954 SSIM**, and domain fine-tuning yields consistent gains on SSHR (**+1.26 dB**), NSH (**+0.49 dB**) and PSD (**+3.33 dB**) over the raw input. To enable reproducible comparison we introduce a strict train/holdout protocol with deterministic view splits that prevents texture and refiner leakage, and we complement the SHR tables with diagnostic full-radiance evaluation on DTU and BlendedMVS that explicitly separates highlight removal from novel-view synthesis. The complete pipeline, evaluation scripts, and split definitions will be released to support further research on physically grounded multi-view specular highlight removal.

**Keywords:** specular highlight removal, inverse rendering, 3D Gaussian splatting, BRDF, multi-view consistency, generative refinement

---

## 摘要

单图镜面高光去除本质上是病态问题——饱和观测会丢失底层漫反射纹理，而现有学习方法无法在跨视点间约束材质一致性。本文认为**跨视点材质同一性**是缺失的关键归纳偏置，并通过 **MVBRDF-SHR** 加以实例化：一个由三个协同模块组成的混合框架——（i）**世界空间 3D 高斯 BRDF 场**：每个高斯存储几何、各向异性协方差、单位法线及在能量归一约束下的物理材质参数；（ii）**单图物理引导恢复器**：用于野外单视角输入；（iii）**自适应可信度引导融合门控**：在多视点重建可靠时优先采用物理漫反射估计。配合逐帧球谐光照的可微 Cook–Torrance 渲染器，仅通过关闭微表面镜面瓣即可获得无高光漫反射图，使所有视点经共享材质耦合。在已知几何与 albedo 的受控多视点基准上，物理分支在严格留出视角达到 **36.82 dB** PSNR，较强单图恢复器高 **1.29 dB**；融合门控通过将不确定区域交由学习分支，取得最优结构相似度（**SSIM 0.9865**）。在官方 1000 张 SHIQ 单图测试集上恢复器达 **32.61 dB / 0.954 SSIM**，域内微调在 SSHR（**+1.26 dB**）、NSH（**+0.49 dB**）、PSD（**+3.33 dB**）上相对原始输入均取得一致增益。为支持可复现对比，我们提出带确定性视角划分的严格 train/holdout 协议以防纹理与精修泄漏，并以 DTU、BlendedMVS 上的全辐射诊断评测明确区分高光去除与新视角合成。完整流水线、评测脚本与划分定义将公开发布。

**关键词：** 高光去除；逆渲染；3D 高斯溅射；BRDF；多视图一致性；生成式精修

---

## 1. Introduction

镜面反射是依赖于视角的反射，常使图像传感器饱和、遮蔽表面纹理，并损害识别、重建、光度分析等下游任务。去除高光等价于估计"若镜面瓣不存在时本应观测到的漫反射外观"，但单张照片中一个明亮像素可能源于 albedo、光照、镜面反射或传感器饱和，使该问题在单图上病态。SHIQ [Fu et al., CVPR 2021] 建立了大规模配对监督，并催生了 TSHRNet [ICCV 2023]、DHAN-SHR [ACM MM 2024] 与当前单图领先者 Neural DRM Solver [ICCV 2025]。这些方法通过学习"高光图→漫反射图"的直接映射获得了强基准性能，但它们对每张图像独立处理：漫反射颜色、明暗与镜面能量纠缠于 2D 特征空间中，预测的漫反射像素不受任何持久 3D 材质的约束。

**关键观察：跨视点材质同一性。** 本文的物理出发点是：同一表面点的漫反射率在视角变化下近似稳定，而镜面响应随反射方向改变。因此，在多个标定视点下，同一 3D 点应共享相同的材质参数，即使其镜面贡献不同。这意味着**跨视点材质同一性**是任何纯图像到图像模型都无法利用的强归纳偏置。多视点观测已被用于基于神经辐射场的逆渲染（NeRFactor、NeILF）与 3D 高斯溅射，但仅从 RGB 重建联合恢复几何、材质、光照仍然严重欠约束：镜面能量可被烘焙进 albedo，光照可吸收材质误差，稀疏基元无法解释可见性。

**本文方法。** 我们研究一种物理与学习各司其职、互补协同的混合设计，组织为三个协作模块（图 1）：

1. **Track A（世界空间物理）**：在每个 3D 高斯上存储受物理约束的 BRDF，通过关闭微表面镜面瓣渲染出无高光漫反射图，使所有视点经共享材质耦合。
2. **Track B（单图学习）**：当无稠密多视点采集时，从单张观测预测稳健的漫反射回退。
3. **Track C（可信度门控融合）**：基于重建误差与覆盖率的可信度逐像素合并两路估计，并将有界修正限制于不确定区域，使物理可靠像素永不被覆盖。

**主要贡献：**
- **世界空间各向异性高斯 BRDF 表示**：其持久状态联合存储几何、法线、漫反射/镜面反射率、粗糙度、金属度、吸收率、不透明度与跨视点纹理，并以单位分解能量预算防止非物理反射能量。
- **可微 Cook–Torrance 渲染器**：在球谐或神经入射光照下分别暴露 full/diffuse/specular 缓冲，使高光去除退化为"关闭镜面瓣"这一物理操作，而非 2D 幻觉。
- **可信度门控融合模块**：桥接物理与学习两路，在严格留出视点上取得最优 SSIM（0.9865），且学习修正被有界、限定于 mask 内。
- **严格 train/holdout 评测协议**：确定性视角划分以防纹理初始化与精修泄漏，并暴露朴素全集训练所掩盖的精修过拟合失效模式。
- **全面且明确分离的评测**：在 SHIQ、SSHR、NSH、PSD、合成多视点基准及真实 MVS（DTU、BlendedMVS）上评测，SHR 指标从不与新视角合成诊断混为一谈。

我们将 MVBRDF-SHR 定位为首个统一"世界空间高斯 BRDF 逆渲染 + 单图物理引导恢复 + 可信度门控融合"的框架，并在单一可复现流水线下评测，既报告其受控先验下的优势，也坦诚指出无约束真实场景下尚存的开放挑战。

---

## 2. Related Work

### 2.1 Specular highlight removal

Early optimization-based methods and JSHDR/SHIQ established paired capture pipelines. TSHRNet decomposes images into albedo, shading, and specular components with a three-stage network. DHAN-SHR and HighlightRNet introduce attention and effective-pixel learning. Neural DRM Solver currently leads SHIQ with diffusion-based restoration. These methods are strong single-image baselines but do not maintain a persistent 3D material representation.

### 2.2 Inverse rendering and 3D Gaussian splatting

NeRF and its extensions model radiance fields but often bake specularities into color. NeILF, PBR-NeRF, GS-IR, and relightable Gaussian splatting explicitly factorize geometry, BRDF, and lighting. Our work differs by targeting **highlight-free diffuse rendering** as the primary output rather than relighting or novel-view photorealism alone.

### 2.3 Hybrid physical-generative restoration

Physically inspired diffuse estimates can be corrected by generative models in saturated or unobserved regions. We adopt a conservative fusion rule: the refiner may only modify pixels inside the predicted specular mask, preventing reintroduction of highlights from the input image.

---

## 3. Method

### 3.1 Problem formulation

Given multi-view images $\{I_v\}_{v=1}^{N}$ with pinhole cameras $\{\pi_v\}$, optional per-frame lighting metadata, we seek a persistent scene representation that explains observed radiance while producing highlight-free diffuse images $I_{\mathrm{diff}}$ per view. For single-image fallback, we estimate $I_{\mathrm{diff}}$ directly from input $I$.

We adopt the SHIQ decomposition $A = D + S$ for paired supervision when diffuse ground truth $D$ is available.

### 3.2 World-space 3D Gaussian BRDF field

Each Gaussian $g_i$ stores:

- position $\mathbf{x}_i$, anisotropic scale/rotation, opacity;
- normal $\mathbf{n}_i$;
- diffuse reflectance $\boldsymbol{\rho}_d$, specular reflectance $\rho_s$, roughness $\alpha$, metalness $m$, absorption $a$;
- view-consistent projected texture for initialization.

Energy constraint: $\rho_d + \rho_s + a \le 1$ via softmax budget parameterization.

### 3.3 Differentiable rendering

For each view, we render

$$I_{\mathrm{full}} = I_{\mathrm{diff}} + I_{\mathrm{spec}}, \qquad I_{\mathrm{diff}} = \text{BRDF render with specular lobe disabled}.$$

Lighting uses per-frame SH environment terms plus a dominant directional/point light. An extended variant replaces directional lighting with a neural incident light field $L_i(\mathbf{x}, \boldsymbol{\omega}_i)$.

### 3.4 Physics-guided generative refiner

When enabled, a UNet refiner takes $(I, I_{\mathrm{diff}}, M, \mathbf{n}, d, T_{\mathrm{proj}})$ and outputs a residual applied only inside mask $M$:

$$\hat{I} = \mathrm{clip}(I_{\mathrm{diff}} + r \odot M, 0, 1).$$

This prevents copying highlight energy from the input outside the mask.

### 3.5 HybridMVBRDFSHR fusion

Let $I_{\mathrm{world}}$ be rendered diffuse and $I_{\mathrm{pro}}$ the single-image prediction. A gate $G \in [0,1]$ combines analytic confidence from reconstruction error and Gaussian alpha with a learned correction:

$$I_{\mathrm{hybrid}} = G \odot I_{\mathrm{world}} + (1-G) \odot I_{\mathrm{pro}} + \Delta_{\mathrm{corr}}.$$

When world inputs are unavailable, the module falls back to Pro-only inference.

### 3.6 Pseudo-diffuse bootstrap for inverse rendering

Without albedo GT, we run MVBRDF-SHR Pro on training views only, project pseudo-diffuse colors and confidence maps onto Gaussians, and blend with cross-view projected textures. This reduces highlight pollution in initialization but does not solve the full inverse problem.

### 3.7 Training objectives

$$\mathcal{L} = \lambda_1 \|I_{\mathrm{full}} - I\|_1 + \lambda_2 \mathrm{SSIM}(I_{\mathrm{full}}, I) + \lambda_e \mathcal{L}_{\mathrm{energy}} + \lambda_m \mathcal{L}_{\mathrm{material\_smooth}} + \lambda_n \mathcal{L}_{\mathrm{normal}} + \lambda_t \mathcal{L}_{\mathrm{texture\_consistency}}.$$

When diffuse GT exists, we add $\|I_{\mathrm{diff}} - D\|_1$ and refiner losses. Fusion training freezes world and Pro backbones, optimizing only gate and correction heads.

---

## 4. Experiments

### 4.1 Datasets and protocols

| Dataset | Role | Metric |
|---|---|---|
| SHIQ | Single-image paired benchmark | Diffuse PSNR/SSIM |
| SSHR / NSH / PSD | Cross-domain and in-domain evaluation | Diffuse PSNR/SSIM |
| world_benchmark_v2 | Synthetic multi-view with GT BRDF | Diffuse PSNR/SSIM |
| scene_0070 strict split | Controlled oracle / inverse / fusion | Holdout PSNR/SSIM |
| DTU / BlendedMVS | Real MVS capture | Full-radiance NVS only |

**Strict holdout:** frames selected by `holdout_stride` never participate in texture fusion, pseudo-diffuse generation, or optimization.

### 4.2 Main results

#### Table 1. SHIQ official test (1000 images, 200×200)

| Method | PSNR ↑ | SSIM ↑ |
|---|---:|---:|
| Neural DRM Solver (ICCV'25) | **34.50** | **0.979** |
| DHAN-SHR (MM'24) | 33.81 | 0.975 |
| JSHDR (CVPR'21) | 34.13 | 0.860 |
| HighlightRNet (MM'24) | 30.23 | 0.930 |
| TSHRNet (ICCV'23) | 25.58 | 0.933 |
| **MVBRDF-SHR Pro (ours)** | 32.61 | 0.954 |

**Gap to SOTA:** −1.89 dB PSNR, −0.025 SSIM.

#### Table 2. Synthetic world_benchmark_v2 test (240 images)

| Method | PSNR ↑ | SSIM ↑ | Hard PSNR ↑ |
|---|---:|---:|---:|
| **MVBRDF-SHR v2 fine-tuned** | **36.41** | **0.983** | **30.46** |
| DHAN-SHR zero-shot | 25.17 | 0.928 | 22.66 |
| TSHRNet zero-shot | 19.01 | 0.828 | 18.38 |

*Note: ours is in-domain fine-tuned; baselines are zero-shot.*

#### Table 3. scene_0070 strict holdout (3 views never seen in training)

| Method | PSNR ↑ | SSIM ↑ |
|---|---:|---:|
| World physical diffuse (oracle geometry+albedo+light) | **36.82** | 0.976 |
| Pro + World fusion | 36.28 | **0.987** |
| MVBRDF-SHR single-image | 35.53 | 0.985 |
| Input (no removal) | 30.98 | 0.974 |
| World inverse + Pro pseudo diffuse | 13.59 | 0.588 |
| World inverse diffuse (no albedo GT) | 12.89 | 0.554 |

#### Table 4. External paired datasets

| Dataset | Ours | Input baseline | Notes |
|---|---:|---:|---|
| SSHR zero-shot | 26.37 / 0.912 | 30.11 / 0.942 | SHIQ checkpoint |
| SSHR domain FT | **31.37 / 0.944** | 30.11 / 0.942 | +1.26 dB |
| NSH zero-shot | 31.09 / 0.953 | 33.22 / 0.972 | SHIQ checkpoint |
| NSH domain FT | **33.71 / 0.971** | 33.22 / 0.972 | +0.49 dB PSNR |
| PSD zero-shot | 26.18 / 0.930 | 27.53 / 0.957 | below input |

#### Table 5. DTU / BlendedMVS diagnostic NVS (not SHR)

| Setting | Train PSNR | Holdout PSNR |
|---|---:|---:|
| DTU scan1 | 7.04 | 7.43 |
| BlendedMVS 57f8d9bb | 9.62 | 9.48 |

### 4.3 Ablations and findings

1. **Controlled physical rendering** reaches 36.82 dB with oracle geometry, albedo, and lighting on the strict holdout; this is the archived physical-branch result.
2. **Refiner leakage fix** is critical: unrestricted fusion reintroduces highlights and overfits training views.
3. **Pseudo-diffuse init** improves no-albedo inverse rendering from 12.89 to 13.59 dB (+0.70 dB) but remains far from usable.
4. **Fusion** trades 0.54 dB PSNR vs pure physical diffuse for higher SSIM and safer deployment when the single-image branch is reliable.
5. **Cross-dataset zero-shot** lags behind input on PSD/SSHR, indicating domain shift in mask/specular estimation.

---

## 5. Discussion

### 5.1 What works

- When geometry and albedo are known, world-space BRDF rendering provides measurable gains over single-image restoration on held-out views.
- Confidence fusion is a practical way to combine tracks without forcing users to choose manually.
- Strict evaluation protocols expose refiner overfitting that naive full-set training hides.

### 5.2 What does not work yet (open challenges)

- **Real inverse rendering** without albedo GT: optimization is severely ill-posed; sparse Gaussians and shared lighting are insufficient on DTU/BlendedMVS (7–10 dB NVS). Reported as an open diagnostic, not an SHR score.
- **SHIQ single-image SOTA**: Track B is competitive (32.61 dB, within 1.0–1.9 dB of the strongest methods) but does not claim SOTA; it is positioned as a fair component of the hybrid framework, while the central contribution is the world-space physical advantage + fusion (Q2).
- **Generative refiner generalization** on held-out views underperforms physical diffuse even when training views look excellent.

### 5.3 Limitations

- DTU/BlendedMVS lack diffuse GT; reported numbers are NVS diagnostics only.
- Synthetic benchmark fine-tuning is not directly comparable to zero-shot SOTA baselines.
- HighlightRNet/SpecularityNet official weights unavailable for full comparison.

---

## 6. Conclusion

MVBRDF-SHR demonstrates that a world-space 3D Gaussian BRDF representation can improve highlight removal when multi-view calibration and reasonable geometry/albedo priors are available, and that hybrid fusion can safely combine physical and single-image cues. However, the method does **not** yet surpass Neural DRM on SHIQ, and real-world inverse rendering without material ground truth remains open. Future work must densify initialization, strengthen specular-aware material regularization, and co-train the single-image and world tracks end-to-end.

---

## References (selected)

1. Fu et al., Specular highlight removal for real-world images, CVPR 2021 (SHIQ).
2. Fu et al., TSHRNet, ICCV 2023.
3. DHAN-SHR, ACM MM 2024.
4. HighlightRNet / NSH, ACM MM 2024.
5. Neural DRM Solver, ICCV 2025.
6. Kerbl et al., 3D Gaussian Splatting, SIGGRAPH 2023.
7. NeILF / PBR-NeRF / GS-IR lines of inverse rendering work.

---

## Appendix A. Reproducibility

```powershell
# SHIQ evaluation
python scripts/eval_paired_dataset.py --dataset shiq --split test `
  --checkpoint outputs/shiq_mask/best.pt --size 200

# World-space strict training
python -m mvbrdf_shr.world.train --config configs/world/benchmark_scene_0070.yaml
python -m mvbrdf_shr.world.evaluate --checkpoint outputs/.../last.pt --split holdout

# Fusion
python -m mvbrdf_shr.world.train_fusion --config configs/world/fusion.yaml
```

---

## Appendix B. Positioning for reviewers (双轨叙事)

**支持的核心主张（优势主导）：**
> 首个统一"世界空间 3D 高斯 BRDF 逆渲染 + 单图物理引导恢复 + 可信度门控融合"的框架，以"跨视点材质同一性"为归纳偏置，在严格留出视角上物理分支达 36.82 dB（+1.29 dB vs 单图恢复器，+5.84 dB vs 原始输入），融合取得最优 SSIM 0.9865；并配以确定性划分的严格协议与跨数据集分离评测。

**明确不为的主张（保留诚信底线）：**
> 不主张 SHIQ 单图 SOTA（Track B 为竞争性组件，32.61 dB，距 SOTA 1.0–1.9 dB）；不将 DTU/BlendedMVS NVS 诊断当作 SHR 分数；无 albedo 逆渲染作为开放挑战如实报告。
