# MVBRDF-SHR 对比方法与 SOTA 清单

更新日期: 2026-07-12  
指标默认: **PSNR ↑ / SSIM ↑**（与 JSHDR / TSHRNet / Neural-DRM 论文一致）  
主评估集: **SHIQ**（真实）、**PSD**（真实）、**SSHR**（合成）

> 注意：不同论文的数据划分/分辨率不完全一致，表中数字来自原文报告，复现时需统一协议（见 §4）。

---

## 1. 当前 SOTA 总览（必比）

| 优先级 | 方法 | 会议 | 类型 | 公开代码 | 为何必比 |
|---|---|---|---|---|---|
| ★★★ | **Neural DRM Solver** | ICCV 2025 | 物理(DRM)+学习 | 待作者开源（跟进 Fu） | **现公开论文中 SHIQ/PSD/SSHR 综合最强** |
| ★★★ | **DHAN-SHR** | ACM MM 2024 | 纯学习注意力 | [CXH-Research/DHAN-SHR](https://github.com/CXH-Research/DHAN-SHR) | 统一基准、预训练权重大、易复现 |
| ★★★ | **HighlightRNet** (HighlightRemover) | ACM MM 2024 | 有效像素学习 | [zz0223/HRNet](https://github.com/zz0223/HRNet) | 肖团队；NSH 多光源数据 |
| ★★☆ | **TSHRNet** | ICCV 2023 | 物理分解+精修 | [fu123456/TSHRNet](https://github.com/fu123456/TSHRNet) | 肖/Fu 三阶段物理基线；SSHR 数据源 |
| ★★☆ | **JSHDR** | CVPR 2021 | 多任务检测+去除 | [fu123456/SHIQ](https://github.com/fu123456/SHIQ) / [KosRud/SHIQ](https://github.com/KosRud/SHIQ) | 经典基线 + SHIQ 数据集 |
| ★★☆ | **SpecularityNet** | TMM | GAN+检测引导 | [jianweiguo/SpecularityNet-PSD](https://github.com/jianweiguo/SpecularityNet-PSD) | PSD 数据集来源 |

### 1.1 Neural DRM Solver 报告数字（ICCV 2025 Table 1）

列顺序按原文: **SHIQ | PSD | SSHR**

| Method | SHIQ PSNR | SHIQ SSIM | PSD PSNR | PSD SSIM | SSHR PSNR | SSHR SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Shen13 (传统) | 13.92 | 0.428 | 13.89 | 0.610 | 24.39 | 0.904 |
| SpecularityNet | 23.42 | 0.920 | 21.80 | 0.880 | 25.73 | 0.894 |
| JSHDR | 34.13 | 0.860 | 21.52 | 0.883 | 26.98 | 0.895 |
| TSHRNet | 25.58 | 0.933 | 22.76 | 0.903 | 28.63 | 0.940 |
| HighlightRNet | 30.23 | 0.930 | 29.79 | 0.922 | 30.07 | 0.956 |
| DHAN-SHR | 33.81 | 0.975 | 25.28 | 0.883 | 34.10 | 0.959 |
| **Neural DRM (SOTA)** | **34.50** | **0.979** | **29.80** | **0.943** | **34.64** | **0.965** |
| **MVBRDF-SHR（单图轨，本地实测）** | **32.613** | **0.9536** | TBD | TBD | TBD | TBD |

当前 SHIQ 排名位于 DHAN-SHR 与 HighlightRNet 之间；距 Neural DRM 为
1.888 dB / 0.0254 SSIM。完整的协议分轨结果见
`outputs/SOTA_COMPARISON.md`。多视点 world-space 轨不与 SHIQ 单图数字混排。

### 1.2 DHAN-SHR 统一基准数字（ACM MM 2024，另一协议）

列: **PSD | SHIQ | SSHR**（与上表协议不同，勿直接横向混比）

| Method | PSD | SHIQ | SSHR |
|---|---:|---:|---:|
| JSHDR* | 22.78 / 0.811 | 37.97 / 0.980 | 26.43 / — |
| SpecularityNet | 23.58 / 0.838 | 30.92 / 0.963 | 31.07 / 0.941 |
| TSHRNet | 23.30 / 0.826 | 34.57 / 0.972 | 33.32 / 0.950 |
| **DHAN-SHR** | **25.28 / 0.883** | 33.81 / 0.975 | **36.48 / 0.964** |

\* JSHDR 无完整开源训练代码，部分数字来自作者可执行文件。

---

## 2. 扩展对比（论文附录 / 相关域）

| 方法 | 会议 | 说明 | 代码 |
|---|---|---|---|
| PHR-DIFF | AAAI 2025 | 肖像高光；仅肖像子集对比 | 跟进肖团队 |
| HAFNet / HMAFNet | CVPR 2025 | 文本图像大面积高光；不同域 | — |
| One-Step SHIR Diffusion | ICCV 2025 | 一步扩散去高光；生成类最接近 | 跟进开源 |
| DiffBIR | ECCV 2024 | 通用修复上界参考 | 官方 DiffBIR |
| InstructPix2Pix | CVPR 2023 | 指令编辑弱基线 | diffusers |
| Yang15 / Shen13 / Fu19 | 传统 | 物理假设下限 | 经典实现 |

---

## 3. 本方法差异化（写论文时怎么讲）

| 对比对象 | 他们做什么 | MVBRDF-SHR |
|---|---|---|
| Neural DRM / TSHRNet | 2D DRM 变量或 albedo×shading | **多视点 BRDF 场**（ρ_d/ρ_s/α/m/a）+ 关镜面瓣 |
| DHAN / HighlightRNet | 纯 2D 网络映射 | 物理可解释 I_diff + 可选生成精修 |
| SpecularityNet / JSHDR | 检测引导 / 多任务 2D | 材质-光照解耦，支持多光强序列 |

额外可报（纯生成方法难以报）：albedo / roughness / normal 一致性、跨视点材质一致性。

---

## 4. 统一复现协议（强烈建议）

1. **测试集**: SHIQ / PSD / SSHR 官方 test split（与 TSHRNet README 的 `.lst` 对齐）。  
2. **分辨率**: 与 Neural DRM 对齐时用其默认；报告时写明（如 256² 或原论文 50p 缩放策略）。  
3. **指标**: PSNR、SSIM；可选 LPIPS。高光区域 masked-PSNR（用 GT mask 或 `|I-GT|`）。  
4. **训练公平性**: 对有代码的方法（DHAN、TSHRNet、SpecularityNet）尽量同数据混合再训；无代码则引原文数字并注明。  
5. 输出目录约定见 `scripts/baselines.md`。

---

## 5. 推荐对比实验表格（投稿用）

**主表**: SHIQ / PSD / SSHR 上 vs Neural-DRM、DHAN、HighlightRNet、TSHRNet、JSHDR、SpecularityNet。  
**消融**: w/o multi-view agg | w/o L_cons | w/o L_spec_ndf | w/o generative refine | diffuse-only vs full pipeline。  
**定性**: 强高光 / 彩色高光 / 白材质 / 多光源（NSH 若可获取）。
