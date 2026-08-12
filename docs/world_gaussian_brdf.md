# World-space 3D Gaussian BRDF 主创新轨

这是用户原始思路的严格实现，和用于 SHIQ 排名的单图分解轨相互独立。

## 1. 输入与坐标

输入为连续多视点图像、内参和 camera-to-world 位姿：

```text
{I_i, K_i, P_i}_{i=1..N}
可选：{light_direction_i, light_intensity_i, env_SH_i}
```

- 视频入口：`scripts/prepare_multiview.py` 调用 FFmpeg + COLMAP。
- Blender/Objaverse：`scripts/blender_render_multiview.py` 输出 full/diffuse 配对、
  相机矩阵和逐帧光源参数。
- 异源渲染：`blender_render_multiview_cycles.py` 与
  `mitsuba_render_multiview.py` 输出线性 HDR 的 full/diffuse/specular/albedo/
  normal/depth/mask，并生成固定 train/val/test 划分。
- 真实数据：支持 Stanford-ORB（Blender/LLFF 布局）与 DiLiGenT-MV
  （20 视角、逐光源标定）；所有入口统一转换为 OpenCV camera-to-world。
- 数据加载：`mvbrdf_shr/world/data.py`。

## 2. 每个 Gaussian 的世界空间状态

`GaussianBRDFField` 为每个三维表面 Gaussian 持久保存：

```text
中心 x ∈ R³
各向异性尺度 s ∈ R³
四元数旋转 q
透明度 o
法线 n
基础颜色 c
漫反射权重 ρd
镜面反射权重 ρs
粗糙度 α
金属度 m
吸收率 a
跨视点融合纹理 t
```

能量预算使用 softmax 参数化，因此严格满足：

```text
ρd + ρs + a = 1
```

同一个 Gaussian 被所有相机共同查询，材质不再绑定图像像素；这从表示层面保证
跨视点材质一致。

## 3. 各向异性 Gaussian 投影与遮挡

世界协方差：

```text
Σ_world = R(q) diag(s²) R(q)ᵀ
```

投影至相机：

```text
Σ_camera = R_cw Σ_world R_cwᵀ
Σ_2D = J_projection Σ_camera J_projectionᵀ
```

每个 Gaussian 形成二维椭圆核，按深度排序后执行 front-to-back alpha compositing，
产生可微的颜色、法线、深度、反照率和跨视点纹理 G-buffer。

实现：`mvbrdf_shr/world/renderer.py`。

- `raster_backend: gsplat`：CUDA tile rasterization，用于大规模训练；
- `raster_backend: torch`：按深度有序分块的可微参考实现；
- `raster_backend: auto`：CUDA 可用时选择 gsplat，编译/运行失败自动回退。

RTX 5090 Laptop、2000 Gaussians、256² 实测：gsplat 7.04 ms，PyTorch
208.58 ms，约 29.6× 加速；峰值显存 22.4 MB 对 86.7 MB。

大规模验证：50,000 Gaussians、512²、18 个联合 G-buffer 通道，gsplat
平均 9.55 ms，峰值显存 89.8 MB。

## 4. 两级光照

### 4.1 第一版

`PerFrameLighting` 将“照明”和“成像响应”拆开：相同 `light_id` 的视图共享
SH、主光方向、强度与颜色，每个 capture 单独优化曝光和白平衡。已知标定光源
可以初始化并冻结，未知光照则联合优化。

- 三阶 SH 环境光；
- 主光方向；
- 主光强度和颜色。
- Stanford HDR lat-long 环境图投影到同一世界坐标 SH，并提取最亮 0.5% 的
  dominant lobe 驱动 GGX 镜面项；
- 线性 HDR 曝光与白平衡（带尺度/色度规范化先验）；
- Gaussian 密度上的可微软阴影透射率。

### 4.2 完整版

`NeuralIncidentLightField` 实现：

```text
Li(x, ωi) → RGB
```

输入为世界坐标和单位入射方向。渲染器使用 Fibonacci 球面方向进行 Monte Carlo
积分，从而表示空间变化照明、间接光和遮挡形成的复杂入射分布。

## 5. 物理分离

对每个 Gaussian 使用 Cook–Torrance/GGX：

```text
f = f_diffuse(ρd) + f_specular(ρs, α, m)
I_full = splat(f_diffuse + f_specular)
I_diffuse = splat(f_diffuse)
I_specular = splat(f_specular)
```

最终高光消除的物理结果就是 `I_diffuse`，而不是由生成网络凭空翻译。

## 6. 最终生成精修

`PhysicsGuidedWorldRefiner` 的条件为：

```text
原图 RGB                 3
物理 diffuse             3
specular mask            1
normal                    3
depth                     1
跨视点投影纹理            3
albedo                    3
```

共 17 通道。非高光区域优先保留输入；高光/饱和区域以物理 diffuse 为基础修复。
`load_shiq_restorer()` 可迁移当前 SHIQ Restorer，新增的深度/跨视点通道单独学习。

## 7. 无 GT 几何与法线修复

主结果统一标为 `Ours-NoGT`，只读取训练图像、相机、训练 mask 和已标定光源：

- Stanford-ORB 由训练视角 mask 执行 voxel carving，提取 visual-hull 表面；
- Gaussian 使用切平面宽、法向薄的各向异性初始尺度；
- `normal_mode: covariance` 从最短 covariance axis 导出法线，禁止独立
  `normal_raw` 漂移；
- silhouette BCE/Dice、背景泄漏、深度/法线平滑约束跨视角覆盖；
- 几何 warm-up 后冻结稳定几何，再优化 BRDF 与光照；
- fixed-budget densify/prune 将低 opacity/低梯度 splat 重新分配到高屏幕梯度区域；
- DiLiGenT-MV 使用 96 个标定光源的 robust photometric stereo 初始化世界法线，
  不读取 GT normal。

`Ours-GT` 只作为受控上界：按三角形面积采样 GT mesh 表面并插值法线，不得并入
无 GT 主结果。

```powershell
# 六场景、六阶段门控消融；通过后才允许启动 47 场景全量训练
python -m scripts.run_geometry_gated_ablation
python -m scripts.run_full_external_campaign
```

## 8. 训练

```powershell
# 1. 无 Blender 的小型端到端验证
python scripts/generate_world_demo.py --out data/world_demo
python -m mvbrdf_shr.world.train --config configs/world/demo.yaml
python -m mvbrdf_shr.world.inference `
  --checkpoint outputs/world_demo/last.pt --frame 0
python -m mvbrdf_shr.world.evaluate `
  --checkpoint outputs/world_demo/last.pt
python -m mvbrdf_shr.world.evaluate_ir `
  --checkpoint outputs/world_demo/last.pt --split holdout
python -m mvbrdf_shr.world.export `
  --checkpoint outputs/world_demo/last.pt --output outputs/world_export

# 2. 视频 + COLMAP
python scripts/prepare_multiview.py `
  --video capture.mp4 --out data/my_scene --fps 5
# 将 configs/world/gaussian_brdf_sh.yaml 的 data_type 改为 colmap，
# sparse_dir 改为 sparse_txt。

# 3. Blender / Objaverse
blender -b -P scripts/blender_render_multiview.py -- `
  --asset asset.glb --out data/object_01 --views 60

# 4. SH 光照版本
python -m mvbrdf_shr.world.train `
  --config configs/world/gaussian_brdf_sh.yaml

# 5. NeILF 完整光场版本
python -m mvbrdf_shr.world.train `
  --config configs/world/gaussian_brdf_neilf.yaml

# gsplat / PyTorch 性能对比
pip install -e ".[cuda]"
python scripts/benchmark_rasterizers.py --gaussians 2000 --size 256

# 6. Cycles/Mitsuba 异源合成
blender -b -P scripts/blender_render_multiview_cycles.py -- `
  --asset asset.glb --out data/object_cycles --views 60 --seed 7
python scripts/mitsuba_render_multiview.py `
  --asset asset.obj --out data/object_mitsuba --views 60 --seed 7
python -m mvbrdf_shr.world.train `
  --config configs/world/cross_renderer_hdr.yaml

# 7. 真实多视角数据
python -m mvbrdf_shr.world.train --config configs/world/stanford_orb.yaml
python -m mvbrdf_shr.world.train --config configs/world/diligent_mv.yaml

# 8. 外部逆渲染基线（需显式提供第三方仓库和权重）
python scripts/run_inverse_baseline.py nerfactor `
  --scene-id object --scene-root data/object `
  --images-dir data/object/images --camera-file data/object/transforms.json `
  --repo external/nerfactor --checkpoint checkpoints/nerfactor.ckpt --dry-run
```

统一评测 `evaluate_ir.py` 只计算存在真实或明确伪 GT 的通道，输出 albedo
尺度对齐误差、normal 角误差、NVS、共享光源 holdout relighting、diffuse/
specular 分解、物理 lobe-off SHR、LPIPS（可选）和对应三维点的跨视角 albedo
一致性。缺失 GT 返回 `null`，不会以输入图像替代 GT。

指标诚信规则：只有数据集提供独立 diffuse-only target 时才生成
`lobe_off_psnr`/diffuse SHR 指标。Stanford-ORB 的估计反照率始终使用
`pseudo_albedo_*` 前缀；DiLiGenT-MV、DTU 和 BlendedMVS 不生成 SHR
PSNR。跨视角 albedo 方差是栅格化一致性诊断；跨方法比较时必须把各方法输出
用同一几何、可见性和 correspondence 反投影到同一表面点。

## 9. 两条实验轨

| 轨道 | 目的 | 数据 |
|---|---|---|
| World 3D Gaussian BRDF | 论文主创新、材质/光照/几何可解释 | 视频、DTU、BlendedMVS、Blender |
| SHIQ 单图分解 | 与 Neural DRM/DHAN 等公平比较 | SHIQ |

SHIQ 四元组没有相机位姿，不能用来证明世界空间多视点材质场；它只训练最终
Restorer 和单图退化模式。主创新必须在有标定多视点数据上报告新视角、
材质参数和重打光结果。

