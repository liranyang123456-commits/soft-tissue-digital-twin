# NeuroComputing 投稿论文逐节优化设计

**日期：** 2026-07-16
**目标文件：** `mvbrdf_shr/submission/neurocomputing/main.tex`（1545 行，17 页 PDF，elsarticle 模板）
**同步文件：** `mvbrdf_shr/docs/manuscript_draft.md`
**期刊：** NeuroComputing (Elsevier, 中科院 2 区 TOP)

---

## 1. 背景与事实纠正

### 1.1 文件定位（已确认）
- `paper/main.tex`（443 行）= 旧简短存根（sn-jnl 模板），**不再作为权威源**。
- `submission/neurocomputing/main.tex`（1545 行）= **真实投稿版**（elsarticle 官方模板），含完整公式、TikZ 流程图、8 表 18 图、Highlights、CRediT、致谢。**在此文件上优化**。

### 1.2 当前论文现状评估
**优势：**
- 方法公式完整严谨（Cook-Torrance、能量约束 softmax、各向异性投影、可微 splatting）。
- 已有 TikZ 流程图（Track A/B/C 三轨五列）。
- 评测诚实、协议严格（holdout stride 公式化）。
- 数据集覆盖广（SHIQ/SSHR/NSH/PSD/world_v2/DTU/BlendedMVS）。

**问题（优化点）：**
1. **定位过于自谦**：摘要/结论反复强调 "1.89 dB below SOTA"、"do not claim SHIQ SOTA"、"failed fitting"。NeuroComputing 审稿人会读成"贡献不足"。
2. **主表混入失败实验**：DTU/BlendedMVS（7-10 dB）与核心贡献结果同列主表，削弱整体观感。
3. **贡献主张与 SHIQ SOTA 对撞**：贡献第 5 条把"跨数据集统一评测"当卖点，但实际未达 SOTA，叙事错位。
4. **流程图已有但可优化**：当前三轨布局信息密、Bus 路由略隐晦，可提升清晰度。
5. **Highlights 末条用 "honest limitations"**：投稿系统里这会削弱卖点。

---

## 2. 总原则（贯穿全文）

**双轨叙事：优势主导 + 严格诚实。**

- **主表/主叙事**走优势主导：以 world-space BRDF 的可控增益（+1.29 dB）、融合 SSIM 最优（0.9865）、严格评测协议、跨数据集域适配增益为核心贡献。
- **失败实验不藏不删**：DTU/BlendedMVS NVS、无 albedo 逆渲染移入**附录/诊断小节**，重新框定为"开放挑战 / 诚实诊断"，不与核心贡献同表。
- **SHIQ 差距如实陈述**，但贡献主张锚定到 world-space + 融合 + 协议，不与单图 SOTA 直接对撞。
- 保留学术诚信底线，不夸大、不伪造。

---

## 3. 逐节优化计划（逐节确认节奏）

每改完一节，停下展示 diff 与说明，等用户 OK 再进下一节。

| 顺序 | 节 | 当前行号 | 优化重点 |
|---|---|---|---|
| 1 | **Abstract** | 82-112 | 优势前置；把 SHIQ 差距从"否定"改写为"定位说明"；末句改为贡献收尾 |
| 2 | **Introduction** | 123-173 | 强化动机；贡献列表重排（world+融合+协议在前，跨数据集诊断降级）；增加 novelty 段 |
| 3 | **Related Work** | 175-208 | 补充差异化定位句；每节末明确"我们的区别" |
| 4 | **Method（含流程图）** | 210-832 | 流程图优化（清晰度）；notation 表保留；公式仅微调 |
| 5 | **Experiments** | 834-1459 | 主表重组：DTU/BlendedMVS 移诊断区；Q1-Q3 叙事重写为优势主导 |
| 6 | **Discussion** | 1463-1494 | "What works"前置详写；"What fails"压缩并重新框定为"open challenges" |
| 7 | **Conclusion** | 1496-1510 | 收尾改为贡献总结 + 未来方向，去掉自谦 |
| 8 | **Highlights + 末段** | highlights.txt | 5 条改写为正向卖点 |
| 9 | **投稿文件生成** | — | 编译验证 PDF；整理 cover letter / highlights / 图清单 |

---

## 4. 流程图优化（Method 节内，第 4 步细化）

当前：三轨 × 五列矩阵 + 右侧 Bus 路由（line 225-343）。
**优化方向（不重写，渐进改进）：**
- 提升色彩区分度（当前 7 色 chip 略杂）。
- Bus 路由用更明显的视觉提示（避免 dashed 线被忽略）。
- 关键输出 $I_d, I_{pro}, \hat{I}$ 加粗高亮。
- caption 补充一句"阅读顺序建议"。
- 确保 `\resizebox{\textwidth}` 不溢出。

**约束：** 不改三轨五列的总体结构（已经是合理的 hybrid 表达），只优化可读性。

---

## 5. 范围边界（不做什么）

- ❌ 不伪造任何数字（所有结果来自已有 `verified_metrics.json` / 本地 JSON）。
- ❌ 不删除失败实验（DTU/BlendedMVS/inverse）——只重新定位与框定。
- ❌ 不改方法本身（公式、模块定义）。
- ❌ 不动 `paper/main.tex`（已确认是废弃存根）。
- ❌ 不重新生成已有的 18 张结果图 PNG（除非优化节需要）。

---

## 6. 投稿文件清单（优化完成后生成）

1. `main.tex` —— 优化后正文（含 TikZ）
2. `main.pdf` —— 重新编译验证
3. `references.bib` —— 保持现有
4. `figures/*.png` —— 保持现有 18 张
5. `highlights.txt` —— 重写为正向卖点
6. `cover_letter.txt` —— 更新以匹配优化后定位
7. `CRediT / Acknowledgments / Data availability` —— 已有，校对

---

## 7. 验证标准

- 每节改完后 LaTeX 仍可编译（无新错误）。
- 数字与本地 JSON 一致（不篡改）。
- 优化后摘要、贡献、结论三处叙事自洽、不再自相矛盾。
- 流程图在 PDF 中清晰、无溢出。
