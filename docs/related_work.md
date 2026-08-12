# MVBRDF-SHR 相关论文与数据集调研

## 1. 肖春霞团队（武汉大学）高光去除脉络

| 工作 | 会议/期刊 | 核心贡献 | 与本方法关系 |
|---|---|---|---|
| Specular highlight removal for real-world images | CGF 2019 | 真实场景高光去除 | 早期物理/优化基线 |
| **JSHDR** + **SHIQ** | CVPR 2021 | 联合检测与去除；约 16K 真实四元组数据集 | 主评估基准；多光源拍摄启发多视点/多光照 |
| **TSHRNet** + **SSHR** | ICCV 2023 | 三阶段：物理分解→精修→色调校正；大规模合成数据 | 物理分解（albedo/shading/specular）可对照；SSHR 可复用 |
| **HighlightRemover** + **NSH** | ACM MM 2024 | 有效像素学习；多光源采集系统 NSH | NSH 对齐好、非高光亮度稳，适合验证材质一致性 |
| **PHR-DIFF** | AAAI 2025 | 肖像高光、patch 扩散 | 生成式修复对照；本方法面向通用物体+物理场 |
| HAFNet | CVPR 2025 | 文本场景大面积高光 | 不同域 |

**研究空白：** 肖团队工作以 **2D 图像分解 / 生成修复** 为主；尚未把 **多视点连续观测 → 逐点 BRDF（吸收/反射）场 → 只渲染漫反射** 作为高光去除主路径。

## 2. 物理反渲染 / NeRF 系（与本思路最贴近）

| 工作 | 年份 | 要点 |
|---|---|---|
| NeRF | 2020 | 多视点体积渲染，外观烘焙进 radiance |
| NeILF / NeILF++ | 2022–23 | 入射光场 + Disney BRDF，材质-光照解耦 |
| **PBR-NeRF** | CVPR 2025 | 能量守恒 + NDF 加权镜面损失，纠正 albedo 中“烤进”的高光 |
| GS-IR / GIR | 2024 | 3DGS 反渲染：几何+材质+光照 |
| MatSpray | 2025 | 2D 扩散 PBR 图融合到 3D 几何 |

本方法吸收：NeILF 入射光场、Disney/GGX BRDF、PBR-NeRF 物理约束；目标从“可重打光”转为 **显式高光消除（I_diffuse）**。

## 3. 推荐生成底座（中型、可训练/微调）

| 模型 | 体量 | 微调方式 | 适用性 |
|---|---|---|---|
| **SD-Turbo** | ~1B UNet | LoRA + ControlNet | 一步/少步，端到端友好（推荐默认） |
| SDXL-Turbo | ~2.6B | LoRA | 更高质量，显存更大 |
| Stable Cascade (Stage C) | 中型 | LoRA | 潜空间紧凑 |
| ControlNet / T2I-Adapter | 附加 | 全参或 LoRA | 注入 I_diff / mask / normal |

**策略：** Stage A–B 只训材质/光场；Stage C 用物理渲染结果作条件，对 SD-Turbo + ControlNet 做 LoRA 微调，修复纹理空洞而不破坏物理一致性。

## 4. 数据集清单（可复用现有 `data/`）

| 数据集 | 类型 | 规模 | 下载/来源 | 用途 |
|---|---|---|---|---|
| **SHIQ** | 真实四元组 | ~10k–16k | [KosRud/SHIQ](https://github.com/KosRud/SHIQ) / Fu 等 | 监督 I_d、mask；本仓库 `data/raw/SHIQ_*` |
| **SSHR** | 合成物体级 | ~5GB | [TSHRNet](https://github.com/fu123456/TSHRNet) | 物理 GT（A/S/D）；`data/raw/SSHR_*` |
| **NSH** | 真实多光源 | 大规模 | HighlightRemover / HRNet | 多光照材质一致性验证 |
| **PSD** | 真实 | — | SpecularityNet-PSD | 补充真实评估 |
| NeRF / Blender 合成多视点 | 合成多视点 | 自渲染 | Blender + Objaverse | **本方法主训练**：连续视点 + 可控光强 |
| DTU / BlendedMVS | 多视点 | 中 | 公开 | 几何初始化 / 跨域泛化 |

## 5. SOTA 定量对比（投稿必比）

完整表与代码链接见 [`sota_baselines.md`](sota_baselines.md)。摘要（Neural DRM Solver, ICCV 2025, SHIQ）：

| Method | PSNR | SSIM |
|---|---:|---:|
| SpecularityNet | 23.42 | 0.920 |
| TSHRNet | 25.58 | 0.933 |
| HighlightRNet | 30.23 | 0.930 |
| DHAN-SHR | 33.81 | 0.975 |
| JSHDR | 34.13 | 0.860 |
| **Neural DRM (当前 SOTA)** | **34.50** | **0.979** |

易复现优先: [DHAN-SHR](https://github.com/CXH-Research/DHAN-SHR)、[TSHRNet](https://github.com/fu123456/TSHRNet)、[HRNet/HighlightRNet](https://github.com/zz0223/HRNet)、[SpecularityNet-PSD](https://github.com/jianweiguo/SpecularityNet-PSD)。

## 6. 与 PhysGen-SHR（旧目录）的差异

| 维度 | PhysGen-SHR | **MVBRDF-SHR（本目录）** |
|---|---|---|
| 主路径 | 单图 SF3D → PBR → 扩散融合 | **多视点材质场**（吸收/反射系数）→ 渲染分离 |
| 表示 | 2D 像素材质图 + mesh 初始化 | 3D hash 场 / 屏幕空间材质场 + 光场 |
| 高光消除 | 扩散主修 + 物理先验 | **物理主修**：关掉镜面瓣得 I_d，生成仅精修 |
| 数据 | 单图对为主 | **连续多视点 + 多光强** 为主，SHIQ/SSHR 适配 |
| 代码 | `physgen_shr/`（勿改） | `mvbrdf_shr/`（全新） |
