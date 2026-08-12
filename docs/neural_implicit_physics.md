# 神经隐式物理仿真方案设计

> 从高斯泼溅回归物理仿真、但用网络学习物理规律隐式表征的方案。
> 作者: 基于 `mvbrdf_shr_next` 当前工作的失败教训设计。
> 日期: 2026-08-10

## 0. TL;DR

当前工作（`world/`）证明了一件事：**显式 Cook-Torrance/GGX + 单次弹射 EWA 高斯泼溅，在 NoGT 真实场景下物理上是不够的**。它把"高光"建模成了单次镜面反射瓣，但真实高光里混着互反射（interreflection）、软影、次表面泄漏、环境多次弹射——这些都是**多弹射光传输现象**，单次弹射渲染器原理上无法表达。

本方案提出**神经传输场（Neural Transport Field, NTF）**：用一个网络隐式表征"给定场景状态 → 完整光传输解"的映射，监督来自物理仿真器（Mitsuba 3 / Cycles 路径追踪），推理时零 Monte-Carlo 开销。物理规律被网络**隐式学习**而非显式编码；高光去除通过在 NTF 内部关闭镜面通路实现，保持物理可解释性。

与当前 `DualLatentNeuralBRDF` 的本质区别：后者只是 GGX 的**残差修正**（已证明中性、被拒），而 NTF **替代整个渲染积分**，把多弹射物理纳入表征——这是当前工作做不到的。

---

## 1. 当前工作的物理瓶颈（诊断）

### 1.1 渲染器是单次弹射的

读 `world/renderer.py:261-314`（`_shade_sh`）和 `:360-475`（`_shade_environment`）：

```
diffuse = albedo/π · (SH_irradiance + direct_light · n·l · visibility)
specular = specular_weight · D_GGX · G_Smith · F_Schlick / (4·n·l·n·v) · direct_light
```

这是**直接光照模型**（direct illumination only）：
- `visibility` 是单条射线的软影透射（`_direct_visibility`），不是多弹射；
- 没有任何**互反射项**（indirect bounce）；
- 没有次表面、没有焦散、没有环境多次弹射。

这意味着：如果物体表面有凹腔（如 Stanford-ORB 的 blocks、gnome），凹腔壁之间的互反射会被错误地归到 albedo 或 specular weight 里——这正是论文 §V.D 报告的 "material-lighting ambiguity"（`main.tex:1390-1411`）的物理根源。

### 1.2 Mitsuba 验证只覆盖了最简单情形

`scripts/verify_renderer_mitsuba.py` 验证的是：**单材质球、纯漫反射、单点光**（`:122-130`）。这个场景没有互反射、没有镜面、没有复杂几何。它证明的是"我们的漫反射积分在无弹射场景下正确"，**不能**推广到"我们的渲染器物理准确"。论文自己也承认（`:150-154`）：只把 circularity 担心从"GT 可能错"收窄到"GT 正确但合成"。

### 1.3 神经残差已尝试且失败

`DualLatentNeuralBRDF`（`models/dual_latent_brdf.py`）是当前工作对"用网络补物理"的尝试。论文 Table `tab:latent_ablation`（`main.tex:1055-1070`）结论：

| 配置 | PSNR-H | PSNR-L |
|---|---:|---:|
| Explicit GGX (β=0) | 20.375 | 27.058 |
| Bounded residual | 20.374 | 27.058 |

**差异在噪声量级，被显式拒绝**。原因（论文 `:121-123`、`:1390-1411`）：
1. 它是 **GGX 的乘性 log 残差**，只能微调已有镜面瓣形状，不能引入 GGX 表达不了的现象（互反射、多次弹射）；
2. 在 NoGT 下，增加神经容量会"fit 训练图但不改善未见光渲染"——容量变成了过拟合通道，不是物理通道；
3. 解耦不彻底：material latent 和 light latent 共享 decoder，没有物理守恒约束保证残差不吸收本该是漫反射的能量。

### 1.4 反向渲染病态的根本原因

论文最坦诚的负面结果：无 albedo GT 的反向渲染在 scene_0070 只有 **12.89 dB**（`verified_metrics.json`）。根因不是"网络不够大"，而是**单次弹射模型本身就是病态的**：当互反射能量存在但模型没有互反射项时，优化器只能把它塞进 albedo（导致高光烘焙进漫反射）或 specular weight（导致视点错误）。**加大网络容量解决不了模型缺失项的问题。**

这是本方案的出发点：**不是让网络修残差，而是让网络学到那个缺失的"多弹射传输项"本身。**

---

## 2. 核心思想：神经传输场（NTF）

### 2.1 物理仿真要回归什么

完整渲染方程（Kajiya）：

```
L_o(x, ω_o) = L_e(x,ω_o) + ∫_Ω f_r(x,ω_i,ω_o) L_i(x,ω_i) (n·ω_i) dω_i
```

其中 `L_i(x,ω_i)` 本身满足递推 `L_i = L_o(·, ω_i)`——这就是**多弹射**。传统物理仿真（路径追踪）通过递归采样求解，计算量随弹射数指数增长。

关键观察：**对于固定场景（几何+材质+光源固定），`L_o(x,ω_o)` 是一个关于 (x, ω_o) 的确定性函数**。物理仿真求解它；网络可以**隐式表征**它。这就是 NTF。

### 2.2 NTF 定义

```
NTF: Φ_θ(scene_state, x, ω_o) → L_o(x, ω_o)
```

其中 `scene_state` 是场景的紧凑编码（几何 + 材质 + 光场），`Φ_θ` 是网络。训练时，用 Mitsuba 路径追踪生成的 `(scene_state, x, ω_o, L_o)` 对监督 `Φ_θ`。推理时，一次前向即得完整多弹射解。

**为什么这是"学习物理规律的隐式表征"而非黑盒拟合：**
- 物理规律（能量守恒、亥姆霍兹互易性、可视性）被作为**约束和不变量**注入训练目标，而非让网络自由拟合；
- NTF 不是从图像到图像的端到端映射，而是从**物理状态**到**辐射场**的映射——中间表征是物理量，可审计；
- 高光去除 = 在 NTF 内部关闭镜面散射通路（见 §3.4），输出仍是物理量，不是生成结果。

### 2.3 与当前 `NeuralIncidentLightField` 的区别

当前 `NeuralIncidentLightField`（`lighting.py:188-221`）学的是 `L_i(x,ω_i)`——入射辐射场，但它**不包含散射过程**：它给出"到达 x 的光"，然后还是套显式 GGX 做单次散射。所以它只是把"光源"换成神经场，没把"散射+多次弹射"换成神经场。

NTF 学的是**完整的 `L_o`（含散射+弹射）**，把 GGX 单次散射这个瓶颈整体替换掉。这是本质区别。

---

## 3. 方案设计

### 3.1 总体架构

```
                    ┌─────────────────────────────────────┐
  多视点图像 ──→ [反向物理场恢复] ──→ scene_state = {几何, 材质, 光场}
                    │  (复用当前 world/ 的 NoGT 管线)      │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  神经传输场 Φ_θ                      │
                    │  输入: scene_state + (x, ω_o)        │
                    │  输出: L_o^full(x, ω_o) [多弹射]     │
                    │  训练: Mitsuba path-traced GT 监督    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  高光分离 (lobe-off in NTF)          │
                    │  L_o^diff = Φ_θ with specular gate=0 │
                    │  L_o^spec = L_o^full - L_o^diff      │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  (可选) 生成精修 (mask-confined)      │
                    └─────────────────────────────────────┘
```

### 3.2 scene_state 编码（解耦设计，解决 §1.3 的第3点）

当前 `DualLatentNeuralBRDF` 失败的一个原因是 material/light latent 耦合。NTF 的 scene_state 强制解耦：

| 分量 | 编码 | 物理约束 |
|---|---|---|
| 几何 `G` | 高斯means+scales+ normals（复用 `GaussianBRDFField`） | 法线单位长、EWA投影 |
| 材质 `M` | per-Gaussian (ρd, ρs, α, m, a) + softmax能量守恒 | `ρd+ρs+a=1`（复用当前 `materials()`） |
| 光场 `L` | SH环境 + 主光（复用 `PerFrameLighting`） | 非负、能量有界 |

**关键设计：scene_state 的每一项都有独立的物理约束损失，NTF 只在这些约束的流形上工作。** 这避免了"网络吸收本该是物理量的能量"。

### 3.3 NTF 网络结构

采用**解耦的两分支 + 融合**结构，而非单一黑盒 MLP：

```
分支A: 散射传输支 Φ_scat
  输入: (x, ω_o, normal, material_code)
  输出: 单次散射+漫反射贡献 L_scat
  物理先验: 近似 GGX 但允许学习偏离（含互反射一次弹射）

分支B: 间接传输支 Φ_indirect  
  输入: (x, ω_o, geometry_code, light_code)
  输出: 多次弹射间接光照 L_indirect
  物理先验: 可视性感知、能量递减

融合: L_o = L_scat + L_indirect
  显式加法保证能量可分解
```

**为什么解耦：** 高光去除只需关闭 `Φ_scat` 的镜面部分，`Φ_indirect` 不受影响——这对应物理上"关掉直接镜面反射，保留互反射漫射"。如果耦合在一个网络里，无法干净地做 lobe-off。

### 3.4 高光去除（lobe-off）机制

```
L_o^diff(x,ω_o) = Φ_scat(x,ω_o, spec_gate=0) + Φ_indirect(x,ω_o)
L_o^spec(x,ω_o) = L_o^full - L_o^diff
```

`spec_gate` 是一个连续可微的门（sigmoid），训练时 =1（学完整物理），推理时设 0 得到无高光结果。这比当前"硬关 `specular_weight`"更物理：当前做法连镜面引起的互反射都关不掉，NTF 的 `Φ_indirect` 会保留正确的互反射漫射成分。

### 3.5 训练数据与监督

**这是本方案最关键、也最需要诚实的部分。** 当前工作的核心教训是 NoGT 真实场景病态。NTF 需要物理仿真 GT，这带来数据策略选择：

**阶段1（预训练，合成）：**
- 用 Mitsuba 3 / Blender Cycles 路径追踪生成大量合成场景的多弹射 GT；
- 场景：Objaverse 子集 + 程序化材质 + HDR 环境；
- 每场景存：scene_state + 采样的 (x, ω_o, L_o) 对；
- 监督：`||Φ_θ(scene_state, x, ω_o) - L_o^Mitsuba||²` + 物理约束。

**阶段2（适配，真实）：**
- 真实多视点（Stanford-ORB / DiLiGenT-MV）无 L_o GT；
- 用**自监督 + 物理一致性**：
  - 重投影一致性：`||splat(Φ_θ) - I_observed||`（复用当前 `photo_l1` + `photo_ssim`）；
  - 互易性约束：`Φ_θ(x,ω_i,ω_o) ≈ Φ_θ(x,ω_o,ω_i)`（亥姆霍兹）；
  - 能量守恒：`∫_Ω Φ_θ · (n·ω) dω ≤ ∫ L_i · (n·ω) dω`；
  - **预训练先验**：NTF 在真实场景的输出不应偏离合成预训练流形太远（KL 正则）。

**阶段3（lobe-off 评测）：**
- PSD 测量的漫反射参考（已用，`main.tex:1364-1382`）作为真实高光去除 GT；
- 这是验证 NTF 高光去除物理正确性的**唯一真实 GT**。

### 3.6 物理约束损失（防止退化为黑盒）

| 约束 | 形式 | 目的 |
|---|---|---|
| 能量守恒 | `||∫ Φ_θ (n·ω) dω|| ≤ ||L_in||` | 防止网络放大能量 |
| 亥姆霍兹互易 | `||Φ_θ(s,x,ω_i→ω_o) - Φ_θ(s,x,ω_o→ω_i)||` | 强制物理对称性 |
| 可视性一致 | NTF 的可视性项与几何 ray-trace 一致 | 防止穿墙光照 |
| lobe-off 物理性 | `L_o^diff` 必须满足 Lambertian 上界 | 防止高光泄漏进漫反射 |
| 分解唯一性 | `L_o^full - L_o^diff = L_o^spec` 单调性 | 防止 spec 吸收 diffuse |

这些约束**正是当前 `DualLatentNeuralBRDF` 缺失的**——它只有一个 `tanh` bound，没有物理不变量。

---

## 4. 与当前工作的差异和继承

### 4.1 继承（复用已验证有效的部分）

| 当前组件 | NTF 方案中的角色 |
|---|---|
| `GaussianBRDFField`（scene.py） | **保留**：几何+材质的高斯表征仍是最优的初始化和 scene_state 载体 |
| `PerFrameLighting`（lighting.py） | **保留**：SH+主光的光场参数化作为 L 分支输入 |
| NoGT 初始化（geometry_init.py） | **保留**：visual hull / photometric stereo 的 provenance-aware 初始化 |
| staged 训练（train.py） | **保留并扩展**：几何冻结→材质→NTF 三阶段 |
| `verify_renderer_mitsuba.py` | **扩展**：从单球漫反射扩展到多弹射场景验证 NTF |
| Mitsuba 多视点生成脚本 | **重用**：作为 NTF 预训练数据生成器 |

### 4.2 核心差异（替换的部分）

| 当前 | NTF 方案 |
|---|---|
| `_shade_sh` / `_shade_environment` 显式 GGX 单次弹射 | NTF 隐式多弹射传输 |
| `DualLatentNeuralBRDF`（GGX log 残差，已拒） | NTF 替代整个散射积分（非残差） |
| 硬关 `specular_weight` 做 lobe-off | `spec_gate` 连续门 + 保留互反射 |
| 无物理不变量约束（仅 softmax 能量） | 互易性+守恒+可视性约束 |
| 仅合成 GT 验证单球 | 多弹射 GT 验证 + 互易性验证 |

### 4.3 直面当前工作的失败教训

本方案的设计选择**直接回应**当前工作报告的每一个负面结果：

| 当前失败 | NTF 如何回应 |
|---|---|
| DualLatent 残差中性（§1.3） | NTF 不是残差，是替代；解决"残差只能微调GGX形状"的局限 |
| 无 albedo GT 病态（12.89 dB, §1.4） | NTF 用合成预训练注入多弹射先验，真实场景自监督不依赖 albedo GT |
| Omnidata 法线更准但作为先验更差 | NTF 不依赖单目法线先验；几何来自多视点，NTF 学的是传输不是几何 |
| 几何重开 redistribute 而非 reduce error | NTF 不重开几何；几何冻结后 NTF 只学传输 |
| material-light 互泄 | scene_state 解耦 + 互易性/守恒约束 |
| DTU/BlendedMVS 7-10 dB NVS 失败 | NTF 的多弹射表征理论上能改善凹腔互反射（待验证） |

---

## 5. 创新点与可发表性评估

### 5.1 潜在创新点

1. **神经传输场作为多弹射物理的隐式表征**：区别于 NeRF（学 radiance）、NeILF（学入射光）、当前工作（学 GGX 残差），NTF 学的是**完整光传输解**。这是一个未被占据的概念位置。

2. **lobe-off 在隐式表征内的物理实现**：现有高光去除要么硬关显式镜面瓣（当前工作），要么端到端生成（DHAN/NeuralDRM）。NTF 的 `spec_gate` 是第三条路：在学到的物理场内连续调节镜面通路，保留互反射。

3. **物理约束驱动的神经仿真监督**：互易性+能量守恒作为不变量损失，把"学物理"从"拟合数据"提升到"满足物理律"。

### 5.2 风险与诚实评估

**高风险点：**
- **预训练到真实的域泛化**：合成 Mitsuba 场景训出的 NTF，在真实 Stanford-ORB 上是否有效？当前工作的 cross-dataset zero-shot 已经暴露域移问题（PSD/SSHR 落后 input）。这是最大风险。
- **计算成本**：Mitsuba 生成多弹射 GT 的开销不小（当前 verify 脚本 spp=256 单球就要数秒；多场景×多视点×多弹射可能需 GPU-天级）。
- **NTF 容量 vs 病态**：当前工作证明"加容量不解决病态"。NTF 也是加容量——除非物理约束+预训练先验真能把容量导向物理通道而非过拟合通道，否则可能重蹈 DualLatent 覆辙。**这是必须在阶段1就验证的核心假设。**

**低风险点（已有基础）：**
- 几何/材质/光场的参数化和初始化（复用当前 world/）；
- Mitsuba 集成（已有脚本）；
- 评测协议（复用 TEST32/DiLiGenT/PSD）。

### 5.3 与 SOTA 的差异化叙事

如果投 TMM/CVPR/ICCV 后续：
- vs NeuralDRM（2D DRM 分解）：NTF 是 3D 多弹射物理，可解释互反射；
- vs GS-IR/Relightable-3DGS（可重新打光）：NTF 目标是高光去除的物理正确性，lobe-off 是一等目标；
- vs PBR-NeRF：PBR-NeRF 加物理损失修 albedo，NTF 用神经场替代单次弹射模型本身——更激进。
- vs 当前 MVBRDF-SHR 投稿：NTF 是其"物理仿真回归"的自然演进，正面回应审稿人"单次弹射不够"的潜在质疑。

---

## 6. 实施路线（分阶段，可中止）

### 阶段0：可行性验证（1-2 周，go/no-go）

**目标：验证 NTF 能否在合成场景上学到多弹射传输，且优于显式单次弹射。**

- 用 Mitsuba 生成 10 个凹腔场景（blocks/gnome 类），每场景 16 视点，存 path-traced `L_o`；
- 训练一个小 NTF（仅 `Φ_scat + Φ_indirect`，无 scene_state 解耦）；
- 对比：NTF vs 当前 `_shade_environment` 在**未见视点**的 PSNR；
- **go 条件**：NTF 在凹腔场景超过单次弹射 ≥3 dB，证明多弹射被学到；
- **no-go 条件**：NTF 不优于单次弹射 → 说明多弹射无法被隐式表征有效学习，方案止步。

### 阶段1：物理约束与 lobe-off（2-3 周）

- 加入互易性/能量守恒损失；
- 实现 `spec_gate` 连续门；
- 验证 lobe-off 的 `L_o^diff` 满足 Lambertian 上界（物理性检查）；
- 在合成场景验证 lobe-off 输出与 Mitsuba 漫反射 GT 的一致性。

### 阶段2：真实场景适配（3-4 周）

- scene_state 解耦 + 合成预训练；
- Stanford-ORB 真实自监督适配；
- PSD 测量漫反射参考评测 lobe-off；
- **go 条件**：PSD 适配后 ≥32 dB（超过当前 30.38）且 lobe-off 物理性检查通过。

### 阶段3：完整评测与论文（3-4 周）

- TEST32 / DiLiGenT-MV / scene_0070 holdout；
- 与当前投稿结果对比；
- 消融：NTF vs 单次弹射 vs DualLatent；
- 论文撰写。

---

## 7. 关键设计决策记录

1. **为什么不用端到端"神经仿真器"（输入场景→直接输出图像）？**
   因为它失去了 lobe-off 的物理可分解性——高光去除需要能在表征内部关掉镜面通路，端到端黑盒做不到。当前工作的 `PhysicsGuidedWorldRefiner` 无门控版已证明会丢 9.95 dB（supplementary 负控制）。

2. **为什么保留高斯而非换 NeRF/网格？**
   高斯的 EWA 投影 + per-Gaussian 材质参数化是 scene_state 的天然载体，且当前 `GaussianBRDFField` 已验证。换表征是无关的工程开销。

3. **为什么 NTF 不是 NeILF 的扩展？**
   NeILF 学 `L_i(x,ω_i)`（入射光），仍套显式 BRDF 做散射。NTF 学 `L_o(x,ω_o)`（含散射+弹射的完整解）。前者是"神经光源"，后者是"神经求解器"——层次不同。

4. **为什么预期 NTF 不会重蹈 DualLatent 覆辙？**
   DualLatent 是 GGX 的**乘性残差**（`f_s · exp(δ)`），上限是"微调 GGX 形状"；NTF 是**替代整个散射积分**，能表达 GGX 表达不了的多弹射。但这一预期**必须在阶段0验证**，不能假设。

---

## 8. 待确认的开放问题

- 预训练数据规模：多少场景×视点够 NTF 泛化？（参考 NeILF/PBR-NeRF 的数据量）
- NTF 推理成本：是否比当前 `_shade_environment` 的 512 样本积分更便宜？（应该是，但需测）
- 真实场景无 L_o GT 时，互易性约束是否足够防止退化？（阶段2核心验证）
- 是否存在公开的多弹射 GT 数据集可省去合成生成？（待调研）

---

*本文档基于对 `mvbrdf_shr_next` 代码、TMM/Neurocomputing 论文、实验日志的完整阅读。所有引用的文件路径和数字均来自仓库实际内容。*
