# 数据集准备（只读复用 / 下载）

不修改仓库根目录已有 `physgen_shr`；数据可放在仓库 `data/raw/` 或任意路径，用配置指向。

## SHIQ（真实，主评估）

- https://github.com/fu123456/SHIQ  
- https://github.com/KosRud/SHIQ  
- Google Drive（SHIQ README）: 约 1GB

期望结构（适配器可扫 `*_A.png` / `*_D.png`）:
```
data/raw/SHIQ_extracted/
  train/  *_A.png *_D.png *_S.png
  test/   ...
  train.lst  test.lst   # 可选
```

```bash
python scripts/prepare_data.py --dataset shiq --url <drive_or_zip> --out ../data/raw/SHIQ_extracted
```

## SSHR（合成，TSHRNet）

- https://github.com/fu123456/TSHRNet  
- Google Drive ~5GB

## PSD（真实）

- https://github.com/jianweiguo/SpecularityNet-PSD

## NSH（多光源，HighlightRemover）

- 见 HRNet / HighlightRemover 论文与仓库说明

## DHAN 统一基准（可选）

- Kaggle: https://www.kaggle.com/datasets/xuhangc/acm-mm-2024-dehighlight-dataset
