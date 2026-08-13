# Figure 5 Idea Mining：两个编号实例的真实分诊（2026-07-21）

## 结论

EasyICU 自己的 Idea Mining 已在既有完整 MIIV export 上运行五条真实 PubMed 路线，未重新提取六库数据。系统共产生 29 条可审计 discovery rows，其中 8 条完成 predictor/outcome 可执行映射，但 **0 条达到 `go/recommend`**。因此本轮没有把任何候选冒充为新科学发现，也没有启动下游分析。

两个由 EasyICU 自动排序并冻结的 outcome-resolved 实例为：

1. `FIG5-DISC-001`：辅助升压药启动时机 → 脓毒性休克患者死亡；`hold`，独立复核找到同题大型公共数据库观察研究及最新 meta-analysis，已拒绝作为新发现。
2. `FIG5-DISC-002`：高钠血症钠纠正速度 → 死亡；`hold`，独立复核找到 MIMIC-III/eICU 及其他大队列的直接同题研究，已拒绝作为新发现。

这两条目前只能作为“系统能拒绝伪创新”的论文过程实例，不能写成科学发现结果。复核凭证见各编号目录下 `review/prior_art_review.json`。

## 数据与运行边界

- 输入：`/Volumes/外置硬盘/easyicu_data/full6_20260717/miiv`
- 复用的 outcome-blind 宽队列：`research_output/experiments/FIG5-DISC-001/triage/miiv_wide_idea.parquet`
- ICU stays：94,458
- 可用概念：279
- 文献：真实 PubMed title/abstract；PMC full text 首次网络读取不完整后，使用工具正式支持的 `--no-fulltext` 路径重跑。
- 没有手工指定候选、变量、效应方向或结果。

## 五条路线

| 编号 | 路线 | discovery rows | 唯一可执行假设 | 结果 |
|---|---|---:|---:|---|
| FIG5-DISC-001 | routine predictor → non-death outcome | 5 | 3 | 3 hold / 2 db-cannot-do |
| FIG5-DISC-002 | prognostic | 4 | 2 | 2 hold / 2 db-cannot-do |
| FIG5-DISC-003 | diverse outcomes | 6 | 0 | 6 db-cannot-do |
| FIG5-DISC-004 | routine lab marker | 3 | 0 | 3 db-cannot-do |
| FIG5-DISC-005 | concept-scoped reviews | 11 | 3 | 3 hold / 8 db-cannot-do |

## 权威产物

- `FIG5-DISC-001` triage SHA-256：`46031e83036cddc4cc1c32f4601b4784af634f320bbf7bb0eb0002c7abd0f6a4`
- `FIG5-DISC-001` handoff SHA-256：`a95cb60ea134063c7ea32dffced8165f0d120db17b4a3f9d7a9168f20bb15fe4`
- `FIG5-DISC-002` triage SHA-256：`55e0925f7012c9741443cfe20a1858aba9a81578d7347bbe20de5eb63f71d3a5`
- `FIG5-DISC-002` handoff SHA-256：`2a6d962f09e7de0b42b279a1d6f584a23fab87dcf3eb747124b29233eef6ca9d`

## 本轮暴露并修复的通用缺口

`run_discovery_to_manuscript.py` 原本可能自动选中一个高排名 concept-set audit，但 handoff schema 又要求冻结 outcome，导致自动交接崩溃。提交 `7dbedd2` 增加了 outcome-resolved 自动选择约束；25 项 discovery-package 回归、Ruff、Black、architecture gate 与 module graph 均通过。

## 下一步

1. 保留两个 `hold` handoff，进行独立人工 prior-art / clinical differentiation review；未通过前不进入分析。
2. 把已经存在的 `idea_mining_data_first.py` 接成正式 discovery route，从六库可测概念反向寻找可迁移、文献稀疏的候选；仍须经过同一 prior-art 与 human gate。
3. 只有 `go/recommend + human confirmation` 的候选才运行 `tools/run_discovery_to_manuscript.py --run-analysis` 并成为 Figure 5 科学结果。
