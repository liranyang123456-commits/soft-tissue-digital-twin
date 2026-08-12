# MVBRDF-SHR

> `mvbrdf_shr_next` 是投稿版本之后的公开数据集优化线。投稿版本保留在
> `../mvbrdf_shr`，本目录只读复用 `../data`，不复制原始数据。新的训练与
> 评测流程见 [`docs/public_dataset_optimization.md`](docs/public_dataset_optimization.md)。

**Multi-View BRDF Field Specular Highlight Removal**

通过多视点（或多光照）图像，学习物体表面每一点对光线的**吸收 / 漫反射 / 镜面反射**系数，再用可微 BRDF 渲染关掉镜面瓣，得到无高光结果；可选中型生成模型（SD-Turbo + ControlNet）做纹理精修。

> 本目录为独立实现，**不修改**仓库内原有 `physgen_shr/` 及配套文件。可只读复用 `../data/raw/SHIQ_*`、`SSHR_*`。

- 设计文档: [`docs/design.md`](docs/design.md)
- 论文与数据集调研: [`docs/related_work.md`](docs/related_work.md)
- **SOTA 对比清单**: [`docs/sota_baselines.md`](docs/sota_baselines.md)（必读）
- **严格 3D Gaussian BRDF 主创新轨**:
  [`docs/world_gaussian_brdf.md`](docs/world_gaussian_brdf.md)

## 思路一句话

类似 NeRF，但每个表面点存的是材质 BRDF（ρ_d, ρ_s, α, m, a），而不是烘焙 radiance；高光 = 视点相关镜面项，去掉后即高光消除。

```
多视点 {I_v} → Geometry + Material + Light 场
            → I_full = diffuse + specular
            → I_diff = diffuse only   ← 物理无高光
            → (可选) 生成精修 → Î
```

## 环境

建议复用仓库根目录的 conda 环境，或：

```bash
cd mvbrdf_shr_next
pip install torch torchvision pyyaml pillow numpy pytest
# 可选生成底座
# pip install diffusers transformers accelerate
```

将包加入 PYTHONPATH：

```bash
# Windows PowerShell
$env:PYTHONPATH = "$PWD;$PWD\.."
# 或在本目录
pip install -e .
```

## 快速开始

```powershell
cd f:\SpecularReflectionRemoving\mvbrdf_shr_next
$env:PYTHONPATH = "$PWD"

# 单元测试
pytest tests -q

# 完整公开数据集优化实验（训练、域适配、公平对比、消融）
python scripts/run_public_campaign.py

# 一键跑通全流程（生成 demo 数据 → A/B/C 训练 → 推理 → 评测 → SOTA 表）
python scripts/run_full_pipeline.py --quick
```

分步：

```powershell
python scripts/generate_demo_data.py --out data/demo_mv
python -m mvbrdf_shr.train --config configs/train/phase_a.yaml --steps 50
python -m mvbrdf_shr.train --config configs/train/phase_b_demo.yaml --resume outputs/phase_a/last.pt
python -m mvbrdf_shr.train --config configs/train/phase_c_demo.yaml --resume outputs/phase_b/last.pt
python -m mvbrdf_shr.batch_infer --input-dir data/demo_mv/test --ckpt outputs/phase_c/last.pt --output outputs/batch_test
python -m mvbrdf_shr.inference --input data/demo_mv/test/scene_0000_v0_A.png --output outputs/infer --ckpt outputs/phase_c/last.pt
```

## 目录结构

```
mvbrdf_shr/
  docs/                 设计与调研
  mvbrdf_shr/           Python 包
    models/
      geometry_field.py   法线/深度场
      material_field.py   逐像素吸收与反射系数（核心）
      light_field.py      SH + 主光（支持多光强）
      brdf.py / renderer.py / envlight.py
      generative_refine.py
      pipeline.py
    data/               玩具多视点 + SHIQ/SSHR 只读适配
    losses.py train.py inference.py
  configs/ tests/ scripts/
```

## 与 PhysGen-SHR 对比

| | PhysGen-SHR（旧） | MVBRDF-SHR（本目录） |
|---|---|---|
| 主监督 | 单图 SF3D + 扩散 | **多视点材质场** |
| 高光去除 | 扩散主导 | **物理关掉镜面瓣为主** |
| 表示 | 2D 精修材质图 | 吸收/反射系数场 + 光场 |

## 数据集（只读复用）

见 `docs/related_work.md`：SHIQ、SSHR、NSH、自渲染多视点。适配器：`mvbrdf_shr.data.SHIQAdapter` / `SSHRAdapter`。

## 生成底座建议

默认 stub（`TinyRefineUNet`）。正式推理可开 SD-Turbo：

```powershell
$env:MVBRDF_DIFFUSION_BACKEND = "sdturbo"   # 需 pip install diffusers
```

## SOTA 对比（投稿主表）

当前公开论文综合最强：**Neural DRM Solver (ICCV 2025)**；易复现强基线：**DHAN-SHR**、**HighlightRNet**、**TSHRNet**、**JSHDR**。

```powershell
# 仅打印文献数字（无需本地预测）
python scripts/eval_compare.py --gt-dir . --pred-root outputs/baselines --include-literature

# 有本地预测后汇总
python scripts/eval_compare.py --gt-dir ../data/raw/SHIQ_extracted/test --pred-root outputs/baselines --out outputs/baselines/summary.csv
```

详见 [`docs/sota_baselines.md`](docs/sota_baselines.md) 与 [`scripts/baselines.md`](scripts/baselines.md)。

## Phase B（SHIQ）

```powershell
# 先按 scripts/prepare_datasets.md 下载并解压 SHIQ
python -m mvbrdf_shr.train --config configs/train/phase_b.yaml --dataset shiq
```

## World-space 多视点主创新

```powershell
# 无外部软件的小型验证
python scripts/generate_world_demo.py --out data/world_demo
python -m mvbrdf_shr.world.train --config configs/world/demo.yaml
python -m mvbrdf_shr.world.inference --checkpoint outputs/world_demo/last.pt --frame 0

# 真实视频：抽帧并由 COLMAP 恢复 K/P_i
python scripts/prepare_multiview.py --video capture.mp4 --out data/my_scene

# Blender/Objaverse：生成带相机、光源和 diffuse GT 的训练数据
blender -b -P scripts/blender_render_multiview.py -- `
  --asset asset.glb --out data/object_01 --views 60

# CUDA 3DGS（自动回退到 PyTorch 参考实现）
pip install -e ".[cuda]"
python scripts/benchmark_rasterizers.py --gaussians 2000 --size 256
```
