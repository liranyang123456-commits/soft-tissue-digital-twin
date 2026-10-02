# BioMedical Engineering OnLine submission package

Journal: **BioMedical Engineering OnLine** (Springer Nature / BMC), SCIE, open access.
2026 新锐：医学 **3 区**（工程生物医学 3 区）。2025 年中科院升级版：医学 **4 区**。
不在现行预警名单。录用后 APC 约 **2790 美元**。

投稿：https://submission.nature.com/new-submission/12938/3

题目：**Audited Soft-Tissue Digital Twins from Monocular Endoscopic Video**

## 作者

1. Ranyang Li（通讯，河南工业大学）— 博士，2022 年 6 月毕业于北京航空航天大学 — lry@haut.edu.cn
2. Haotian Ma（河南工业大学）— 学士
3. Zhipeng Lin（北京航空航天大学）— 博士在读
4. Nan Wei（河南省人民医院）— 博士，2021 年 6 月毕业于郑州大学

## 上传

| 文件 | 用途 |
|---|---|
| `main.pdf` | 稿件 PDF（Springer Nature 模板，双倍行距，行号） |
| `manuscript_source.zip` | `main.tex`、`sn-jnl.cls`、`sn-vancouver.bst`、`references.bib`、`figures/` |
| `cover_letter.txt` | 投稿信 |

系统里再填三位合作者的邮箱。只有通讯作者邮箱写进了稿件。

## 和现有方法比，没有重建 SOTA

同一协议下变好的只有两处：跟踪消融（最差帧损失 37.23 到 1.55）和组织掩膜消融（11.13 dB 到 36.79 dB）。组织掩膜 PSNR、6 个场景的有限元中位误差、体模对比度是各自协议上的结果，不能和 EndoNeRF 家族的留出视角 PSNR 排成一张榜。无测力的真实内镜绝对模量没有做。对照图是 `figures/fig_task_comparison.png`。

## 范围风险

期刊说明不收「只有仿真、没有真实患者数据」或「只用公开数据集」的稿。本稿力学主结果来自公开内镜数据、有限元基准和超声体模；医院 CT 是去标识的真实影像，用于几何来源验证。投稿信已写明这一点。若编辑按「仅公开数据」预审，仍可能直接拒稿。
