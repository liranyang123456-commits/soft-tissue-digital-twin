# 基线运行与对比脚手架

所有基线**在仓库外或 `mvbrdf_shr/baselines_ext/` 克隆**，推理结果 PNG 写入统一目录后，用本目录 `scripts/eval_compare.py` 汇总。

## 目录约定

```
mvbrdf_shr/outputs/baselines/
  neural_drm/     # ICCV'25 SOTA（有权重后）
  dhan_shr/       # ACM MM'24
  highlightrnet/  # ACM MM'24 HighlightRemover
  tshrnet/        # ICCV'23
  jshdr/          # CVPR'21
  specularitynet/ # TMM / PSD
  mvbrdf_shr/     # 本方法
```

GT 目录（只读，指向仓库 data 或下载位置）:
```
--gt-dir <SHIQ_test_gt>
```

## 一键汇总

```bash
cd mvbrdf_shr
python scripts/eval_compare.py \
  --gt-dir ../data/raw/SHIQ_extracted/test_gt \
  --pred-root outputs/baselines \
  --methods neural_drm,dhan_shr,highlightrnet,tshrnet,jshdr,specularitynet,mvbrdf_shr \
  --out outputs/baselines/summary.csv
```

## 各基线获取

| 方法 | Clone / 权重 |
|---|---|
| DHAN-SHR | `git clone https://github.com/CXH-Research/DHAN-SHR` + [Releases](https://github.com/CXH-Research/DHAN-SHR/releases) |
| TSHRNet | `git clone https://github.com/fu123456/TSHRNet` + Google Drive checkpoints |
| HighlightRNet | `git clone https://github.com/zz0223/HRNet` |
| SpecularityNet | `git clone https://github.com/jianweiguo/SpecularityNet-PSD` |
| JSHDR | SHIQ 仓库可执行文件 / 作者发布 |
| Neural DRM | ICCV 2025；代码未公开时先引用 Table 1 数字 |

数据集下载见 `scripts/prepare_datasets.md`。

数字参考与论文表: [`docs/sota_baselines.md`](../docs/sota_baselines.md)。
