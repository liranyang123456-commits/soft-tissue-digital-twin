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

## 上传（只传这三份）

| 文件 | 系统里放哪里 |
|---|---|
| `main.pdf` | Manuscript |
| `manuscript_source.zip` | Manuscript source。内含 `main.tex`、`references.bib`、`sn-jnl.cls`、`sn-vancouver.bst`、`figures/` |
| `cover_letter.txt` | 粘贴到 Cover letter |

稿件以该刊官方 Research 作者指南为准：结构化摘要仅含
**Background / Results / Conclusions**，正文为
**Background（含相关工作）→ Results → Discussion（含 Limitations）→
Methods → Conclusion**；文末包含缩略语和全部 Declarations。Methods
中按照该刊要求披露了 AI 辅助英文编辑。不要上传 `compile*.txt`、
`_sn/`、`sn.zip`。系统里还需填写三位合作者的邮箱。代码链接已写进
数据声明和投稿信。

## 提交前必须人工确认

- 按官方指南，人类影像数据即使获得豁免，也应写出作出豁免决定的
  伦理委员会名称及适用的批准号或豁免号。当前稿件只有伦理和知情同意
  获得豁免的事实，没有委员会名称和编号；请在提交前向河南省人民医院
  确认后补入。
- 确认医院 CT 的“可向通讯作者合理申请”与原始机构许可一致。
- 在投稿系统中补齐 Haotian Ma、Zhipeng Lin 和 Nan Wei 的邮箱。

## 和现有方法比，没有重建 SOTA

同一协议下变好的只有两处：跟踪消融（最差帧损失 37.23 到 1.55）和组织掩膜消融（11.13 dB 到 36.79 dB）。组织掩膜 PSNR、6 个场景的有限元中位误差、体模对比度是各自协议上的结果，不能和 EndoNeRF 家族的留出视角 PSNR 排成一张榜。无测力的真实内镜绝对模量没有做。对照图是 `figures/fig_task_comparison.png`。

## 范围风险

期刊说明不收「只有仿真、没有真实患者数据」或「只用公开数据集」的稿。本稿力学主结果来自公开内镜数据、有限元基准和超声体模；医院 CT 是去标识的真实影像，用于几何来源验证。投稿信已写明这一点。若编辑按「仅公开数据」预审，仍可能直接拒稿。
