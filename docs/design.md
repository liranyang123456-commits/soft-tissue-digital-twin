# MVBRDF-SHR 设计文档

**方法全称:** Multi-View BRDF Field Specular Highlight Removal  
**代号:** MVBRDF-SHR  
**日期:** 2026-07-12  
**原则:** 不修改仓库内 `physgen_shr/` 及原有 configs/scripts；本目录独立实现。

---

## 0. 核心思想（用户思路形式化）

观察：物体表面每一点对入射光的 **吸收** 与 **反射** 随材质与角度变化，产生漫反射与镜面反射。

$$
L_o(\mathbf{x},\omega_o) = \int_{\Omega} f_r(\mathbf{x},\omega_i,\omega_o)\, L_i(\mathbf{x},\omega_i)\, (\mathbf{n}\cdot\omega_i)\, d\omega_i
$$

其中 BRDF 分解为：

$$
f_r = \underbrace{f_d(\rho_d)}_{\text{漫反射（与视点弱相关）}} + \underbrace{f_s(\rho_s,\alpha,m)}_{\text{镜面（视点相关 → 高光）}}
$$

能量约束（吸收）：

$$
\underbrace{a(\mathbf{x})}_{\text{吸收}} + \underbrace{\rho_d(\mathbf{x})}_{\text{漫反射率}} + \underbrace{\rho_s(\mathbf{x})}_{\text{镜面反射率}} \approx 1
$$

**高光消除：** 用多视点连续图像优化场参数后，仅用 $f_d$ 渲染得到 $I_d$（无高光）；可选生成模型精修。

类似 NeRF：沿射线积分 / 表面查询；更复杂处在于显式材质-光照解耦与物理约束。

---

## 1. 总体管线

```
输入: 多视点图像 {I_v} + 相机 {π_v}  (+ 可选光强调制)
        │
        ▼
┌───────────────────────────────────────┐
│ Stage 1  Geometry Field               │
│   hash-grid / MLP → 深度、法线 n(x)   │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ Stage 2  Material Field（逐点材质）     │
│   ρ_d, ρ_s, α(roughness), m(metal)    │
│   a = softplus(absorb) 能量守恒正则    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ Stage 3  Incident Light Field         │
│   L_i(x, ω_i) 或 SH 环境光 + 主光强  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ Stage 4  Differentiable BRDF Render   │
│   I_full = f_d + f_s                  │
│   I_diff = f_d only  ← 物理无高光     │
│   M = ||I - I_diff||                  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ Stage 5  Generative Refine (可选)      │
│   SD-Turbo + ControlNet               │
│   cond = {I, I_diff, M, n}            │
│   → Î 最终无高光图                    │
└───────────────────────────────────────┘
```

**单图推理：** 用预训练单图编码器初始化材质/法线场（屏幕空间），再跑 Stage 4–5；多视点时物理主导。

---

## 2. 模块细节

### 2.1 Geometry Field
- 实现：`HashGridEncoder` + 小型 MLP → σ / 深度；法线由深度有限差分或独立预测。
- Stub：直接预测屏幕空间法线（便于无 COLMAP 时单图/玩具数据测试）。

### 2.2 Material Field（核心）
每个表面点输出：
| 参数 | 含义 | 范围 |
|---|---|---|
| `albedo` ρ_d | 漫反射（吸收补） | [0,1]^3 |
| `specular` ρ_s | 镜面反射强度 | [0,1] |
| `roughness` α | GGX 粗糙度 | [0.05,1] |
| `metalness` m | 金属度 | [0,1] |
| `absorption` a | 吸收系数 | ≥0 |

多视点下同一 3D 点材质应一致 → 在世界坐标查询 hash 场。

### 2.3 Light Field
- 默认：3 阶 SH 环境光 + 可学习主光方向/强度（支持多光强序列）。
- 扩展：NeILF 式 5D MLP `L_i(x, ω_i)`。

### 2.4 BRDF 与渲染
Cook-Torrance / GGX（与 PhysGen 可对照，但场表示不同）：
- Diffuse: `ρ_d/π * E_d`
- Specular: `D·G·F / (4 n·l n·v)`
- 关掉 specular → `I_diff`

物理损失（借鉴 PBR-NeRF）：
- `L_cons`：能量守恒（反射不超过入射）
- `L_spec`：NDF 加权，惩罚高光区 diffuse 过大（防高光烤进 albedo）

### 2.5 生成精修
中型底座 **SD-Turbo + 多条件 ControlNet + LoRA**；训练用 noise-prediction，不展开长采样链。

---

## 3. 训练阶段

| Phase | 内容 | 数据 |
|---|---|---|
| A | 玩具/合成多视点：只训 Geometry+Material+Light+Render | 自渲染 / SSHR 适配 |
| B | 加入 SHIQ 等 paired 监督：`||I_diff - GT||` | SHIQ / SSHR |
| C | 冻结物理场，训生成精修 LoRA | 合成 + 真实 |

损失：

$$
\mathcal{L} = \mathcal{L}_{\text{photo}}(I_{\text{full}}, I_v)
 + \lambda_d \mathcal{L}_{\text{diff}}(I_{\text{diff}}, I_d^*)
 + \lambda_c \mathcal{L}_{\text{cons}}
 + \lambda_s \mathcal{L}_{\text{spec}}
 + \lambda_g \mathcal{L}_{\text{gen}}
$$

---

## 4. 与 PhysGen-SHR 的借鉴（只读）

可复用思路/数据（**不修改**旧代码）：
- SHIQ / SSHR 目录布局与评测指标（PSNR/SSIM）
- Cook-Torrance/GGX 公式与 SH 辐照度
- SD-Turbo + ControlNet 融合范式
- stub 后端便于无权重测试

本目录全部新实现，包名 `mvbrdf_shr`。

---

## 5. 成功标准

1. ~~多视点玩具场景：`I_full` 拟合输入，`I_diff` 无高光且纹理保留。~~ ✅ Phase A  
2. SHIQ 测试集：对标 Neural DRM / DHAN（需下载真实数据后填表）。Demo 合成测试已跑通。  
3. 消融配置已备：`configs/train/ablation_no_gen.yaml`、`ablation_no_phys.yaml`。  
4. ~~`pytest` 通过；一键全流程 `scripts/run_full_pipeline.py --quick`~~ ✅  

## 6. SOTA 对比入口

- 数字与必比方法: [`sota_baselines.md`](sota_baselines.md)  
- 本地复现脚手架: [`../scripts/baselines.md`](../scripts/baselines.md)  
- 汇总评测: `python scripts/eval_compare.py --include-literature ...`  
- **一键全流程**: `python scripts/run_full_pipeline.py --quick` → 报告 `outputs/PIPELINE_REPORT.md`

## 7. 严格 world-space 主创新实现

原设计中的 `ScreenGeometryField` 仅用于单图/SHIQ 退化模式，不代表真正的
多视点实现。严格版本已独立实现于 `mvbrdf_shr/world/`：

- `camera.py`：K、camera-to-world 位姿、射线和投影 Jacobian；
- `scene.py`：世界空间各向异性 3D Gaussian 及逐点 BRDF/吸收参数；
- `renderer.py`：协方差投影、深度排序 alpha 合成、GGX full/diffuse/specular；
- `lighting.py`：逐帧 SH+主光以及 NeILF `Li(x,ωi)`；
- `data.py`：COLMAP、Blender、跨视点纹理投影融合；
- `refiner.py`：使用 RGB/diffuse/mask/normal/depth/跨视点纹理的最终精修；
- `train.py` / `inference.py`：多视点联合优化和目标视角去高光。

完整说明与运行命令见 [`world_gaussian_brdf.md`](world_gaussian_brdf.md)。

## 8. MVBRDF-SHR Pro + World-space 融合

统一入口实现于 `mvbrdf_shr/world/fusion.py::HybridMVBRDFSHR`：

1. Pro 分支从当前 RGB 产生 `single_pred/spec_pred/mask_pred`，保证没有
   多视点、相机或 world checkpoint 时仍可直接推理。
2. World 分支产生 `diffuse/specular/mask/normal/albedo/alpha`，并计算
   `|I-(I_diff+I_spec)|` 作为物理重建置信度。
3. 门控网络逐像素预测 `g∈[0,1]`：
   `I_fused = g·I_world_diffuse + (1-g)·I_Pro`。
4. 仅在 `4g(1-g)` 较高的来源不确定区域允许最大 0.05 的有界残差修正，
   防止生成网络覆盖可信的物理 diffuse。
5. `world=None` 时自动返回 Pro 输出，不需要单独部署两套入口。

门控训练实现于 `mvbrdf_shr/world/train_fusion.py`。world 和 Pro backbone
默认冻结，只训练 gate/correction；训练视角监督门控，留出视角仅评测。
当前 scene_0070 严格留出结果为 36.28 dB / 0.9865，PSNR 介于 Pro 和
physical diffuse 之间，但 SSIM 为三者最高。
