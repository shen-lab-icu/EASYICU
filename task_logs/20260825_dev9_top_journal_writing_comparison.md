# Dev9 与已发表高质量 ICU 论文的写作对照及通用修复

> 日期：2026-08-25  
> 隔离分支：`codex/writer-planner-integration-20260825`  
> 修复前基线：`60abcd7897b65833b43a49d8902f1e04368c8d15`  
> 比较范围：Dev9 九个开发稿；14 篇可访问 PMC 全文；其中 11 组公开补充材料已核查  
> 权威边界：已发表论文只作为设计、写作和呈现标尺，其效应量不是 EasyICU 的标准答案

## Review setup

本轮不是以摘要或题名作主观模仿。对照材料来自
`figure2_dev9_anchor_shadow_review_84f31fd_refresh_20260825/anchor_source_pack.json`，
每条全文记录具有 PMID/PMCID、主文 XML SHA-256、章节结构、图表数和补充材料数；九题另有
`run_bound_reviews/*.json` 的七维、run-bound 比较。当前稿件先以旧合同审计，再以本轮增强后的
`manuscript-quality-audit-v3` 进行零 Provider 重放。

评价问题为：稿件是否让临床读者清楚知道研究对象、时间零点、变量定义、主要估计量、图表所承载的证据和结论边界；不是结果是否与已发表论文数值一致。

## Reviewer 1 — technical and reporting integrity

### Major findings

1. **正文与已有图表脱节。** 九稿中，多数 Results 没有明确的 `Figure 1` 或 `Table 1` callout，尽管八题已有注册主图、多个题已有 Table 1。对照全文通常在结果首次报告相应证据时指向图或表；缺少 callout 会让图件像附加产物，而不是论证链的一部分。
2. **Writer 续写会丢失 host-owned 行政段落。** E1、E2、E3、M1、M2、H3 缺 Data and code availability、Funding、Ethics、Conflicts of interest 和 Supplementary artifact release。根因不是作者尚未签字，而是旧的 section migration 只重新组装八个科学章节，直接丢弃行政章节。
3. **内部术语仍能进入读者正文。** M3 暴露 `stay_id`、`los_icu`，并把 `0.621428` 写成未命名的 point estimate；H1 暴露 `mort_28d` 和 “host-bound”；H2 暴露 `H2_VERIFIED_NON_USE_UNAVAILABLE`。这些内容可保留在审计文件，不应进入摘要、结果或讨论。
4. **现有主要科学边界总体没有被 Writer 越权改写。** H2 无真实非暴露对照组时拒绝因果估计，H3 在 BIC 边界最优时拒绝选择 K，均比强行生成阳性结论更符合高质量方法论文的报告原则。

### Technical recommendation

要求 revision。先关闭 display callout、行政段落和内部术语三个通用报告缺口；不要用润色掩盖外部复现、positivity、informative censoring 或聚类稳定性不足。

## Reviewer 2 — originality, positioning, and contribution

### Major findings

1. 九稿不是逐句复制同一文章；现有检查没有发现大段科学句子的机械复用。问题主要是**章节骨架过于统一**：预测、静态表型、时间变化生存和正确失败的因果可行性稿都使用近似的 “Primary association / Sensitivity” 叙事。
2. 已发表锚点显示，不同研究家族的贡献中心不同：M2 应围绕区分度、校准、Brier、决策曲线和验证层级；M3/H3 应围绕聚类选择、稳定性、算法一致性和外部迁移；H1 应围绕时间零点、风险集、时间变化效应与删失；H2 首先是 estimand 是否被识别，而不是常规 association 结果。
3. 当前通用 Writer 已包含问题特异叙事要求，但尚未在九稿中形成稳定可验收的 family-specific prose。此缺口可以通过本轮有限章节重写验证，不应立即重构八个稳定章节标题。

### Originality recommendation

保留统一、可审计的章节合同，但要求每个 Results 段落围绕本题实际注册的 display 和 estimand 组织。若重写后仍可无差别移植到另一题，再将 analysis-family-specific narrative 作为下一 owner-scoped 改进；本轮不扩大成全 Writer 重构。

## Reviewer 3 — interdisciplinary readability and editorial usability

### Major findings

1. 对临床和跨学科读者，当前最明显的阅读障碍不是词句“不够华丽”，而是不知道一段结果对应哪张表、哪张图，以及内部代码为何出现在正文。
2. 明确写出 “Figure 1 / Table 1” 比暴露 evidence id 更符合投稿阅读路径；evidence id 应继续由 audit/bound manuscript 保存，reader view 可保持干净。
3. 未经作者核验的伦理、经费、冲突与发布声明不能由模型补写。高质量的临时稿应明确标为 “requires author verification”，而不是直接删除，也不是生成看似完整的常规套话。
4. 对 H2/H3，跨学科可读性的正确做法是用临床语言解释“为什么不能估计/不能命名”，同时保留失败原因的机器码在 sidecar，而不是把 reason code 当结论。

### Editorial recommendation

修复后可作为开发级 article package 复审，但不能称 publication-ready。行政占位符、`analysis_only`、单数据库限制和未完成外部验证仍必须清楚可见。

## Cross-review synthesis

三位 reviewer 一致认为，以下是本轮可关闭的通用写作缺口：

| 缺口 | 修复 owner | 本轮修改 | 验收方式 |
|---|---|---|---|
| Results 不引用已有 Table 1/Figure 1 | manuscript quality + Writer section contract | 仅当相应 evidence id 已注册时要求 callout；无图的 H2 不伪造 Figure 1 | `MANUSCRIPT_DISPLAY_NOT_CALLED_OUT` 必须为 0 |
| section migration 丢行政声明 | administrative authority + manuscript sections | resume/migration 加载 run-bound authority；无 authority 时追加明确待作者核验的五段 | 五个行政章节均存在；不得生成虚假已核验声明 |
| 历史/退休稿重新进入 resume | Writer resume | 从 append-only 全记录改为 current + digest-verified 证据视图 | focused resume tests |
| 内部变量、reason code、未命名指标进入正文 | manuscript quality + bounded section repair | 只重写错误所属章节；Methods 中为复现所需的精确变量保留 warning 而非全局删除 | M3/H1/H2 error 为 0 |

修复前增强门禁的零 Provider 结果为 0/9 pass，这不表示科学执行从 6/9 “退步”，而是新门禁把此前未检查的 display/行政问题显式化：E1/E2/E3/M1/M2/H3 主要新增行政与 display 错误；M3/H1/H2 另有既存内部术语或未命名指标错误。证据位于
`figure2_dev9_writer_planner_integration_60abcd7_audit_v2_20260825/summary.json`。

## Per-task published-anchor comparison

| 题目 | 主要开放全文锚点 | 修复前写作缺口 | 仍不能靠写作关闭的科学缺口 |
|---|---|---|---|
| E1 | PMC11192388；PMC8508729 | 缺 Figure 1 callout；缺五个行政章节 | 暴露机会/早期事件、重复住院依赖、外部复现 |
| E2 | PMC2875540 | 缺 Table 1/Figure 1 callout；缺五个行政章节 | measurement-by-indication、外部复现 |
| E3 | PMC9543500 | 缺 Figure 1 callout；缺五个行政章节 | AKI 发生窗口与早期事件机会、外部复现 |
| M1 | PMC9322581；PMC8757589 | 缺 Figure 1 callout；缺五个行政章节 | measurement-by-indication、外部复现 |
| M2 | PMC6572845；PMC7223438 | 缺 Figure 1 callout；缺五个行政章节 | 时间/外部验证与 recalibration |
| M3 | PMC6537818；PMC13227316 | 缺 Figure 1 callout；内部变量；未命名且过精确指标 | 稳定性低、第二 robustness axis 和外部复现 |
| H1 | PMC7906666 | 缺 Figure 1 callout；内部 outcome/runtime 用语 | informative censoring 与外部复现 |
| H2 | PMC7023737；PMC10106031 | reason code 进入 Abstract/Results | 无可验证非暴露对照、positivity 未建立；应继续 fail closed |
| H3 | PMC9250715 | 缺 Figure 1 callout；缺五个行政章节 | 无 interior BIC optimum、缺替代算法一致性与外部复现 |

## Risk, unsupported claims, and decision

- published effects 没有进入 expected answer、门禁或 Writer prompt。
- 本轮修复不改变任何 analysis plan、cohort、effect estimate、FigureContract 或 source data。
- “九稿写作门禁通过”只代表当前通用结构、术语、图表导航和证据绑定检查通过；不能推出临床正确、外部有效、专家认可或论文可投稿。
- 只有 agent pipeline 重新生成并通过零 Provider 审计的稿件才计入本轮修复结果；手工改稿不作为验收证据。

综合决定：**major revision at the reporting layer；retain analysis-only authority。**

## Final bounded Writer migration outcome

本轮最终统一验收目录为
`/Volumes/外置硬盘/easyicu_data/figure2_dev9_writer_only_topjournal_2481c89_20260825`。
九题的 `writer_only_migration_receipt.json`、`manuscript_quality_audit.json` 与
`manuscript_literature_audit.json` 均为 pass；`manuscript_bound.md` 和
`manuscript_reader.md` 均不再含未展开的 claim/evidence token，且所有运行均记录
`analysis_steps_executed=0`、`source_run_modified=false`、
`claim_ceiling=analysis_only`、`publication_authorized=false`。

| 题目 | 当前正文导航 | 本轮 Writer 结果 | 与锚点相比仍需保留的限制 |
|---|---|---|---|
| E1 | Table 1 + Figure 1 | 摘要、Methods、Results 与行政段落已收口；非因果结论不再复制 Results | 注册 claim 仍用 generic group wording；重复住院依赖、外部复现和人工定义验证未关闭 |
| E2 | Table 1 + Figure 1 | Methods 内部术语已移除，display callout 与行政段落齐全 | measurement-by-indication、非线性外推和外部复现未关闭 |
| E3 | Table 1 + Figure 1 | 机器精度已转为出版精度，Methods 与 display 已收口 | 时间窗机会偏倚、外部复现；注册 claim 的组标签仍不够读者友好 |
| M1 | Table 1 + Figure 1 | 缺失/测量过程语言转为临床可读，行政段落齐全 | measurement-by-indication 与跨中心复现未关闭 |
| M2 | Table 1 + Figure 1 | AUROC、Brier、校准截距/斜率和 patient-level split 均明确命名 | 仅内部/重复拆分验证；缺时间或外部验证及 recalibration |
| M3 | Table 1 + Figure 1 | 低 silhouette、低重采样 ARI 和低算法一致性被明确限制为 candidate phenotypes | 稳定性不足，不能命名为已建立亚型；旧 plan 仅允许 report-only 零调用封装 |
| H1 | Figure 1 | 时间零点、风险集和时间变化分析语言已读者化 | informative censoring、外部复现和更完整生存诊断未关闭 |
| H2 | 无已注册主表/主图 | 继续明确报告“不可构造 verified non-use 对照，因此不估计效应” | positivity 与 treatment contrast 未识别；这是正确 fail-closed，不是阴性研究结果 |
| H3 | Figure 1 | 固定窗口、缺失状态和候选六类写清，未伪造结局效应 | 未提供可报告的稳定性数值/通过判定；不能称稳定轨迹亚型 |

这 9 篇通过的是当前写作、文献和证据绑定合同，不是完整投稿组合。与开放全文及补充材料相比，
当前 article package 仍普遍只有一个注册主结果 display；H1/H3 只有 Figure 1，H2 因科学不可识别而
没有主结果图表。是否需要新增主文 Figure 2/3、Table 2 及补充图表，必须由各题已执行结果和
FigureContract 决定，不能为了模仿顶刊篇数而凭空补图。缺失率/测量过程应默认位于补充材料或
质量控制 display，除非它本身就是研究问题（如 M1），不应自动占用所有文章的主结果图。

Provider 使用必须按整个修复过程而非最终零调用封装统计：共 35 次 Writer 调用；CLI 未返回
provider-metered usage，启发式记录为 529,710 tokens。durable hard-stop ledger 的保守上界为
6,628,360 tokens、US$245.48360。该上界包含 E1 合同调试期间所有失败且未发布的候选；最终统一
目录的封装本身为零调用。没有 Planner、Executor、Coder 或 Figure Provider 调用。

最终判断：**九篇写作层 9/9 通过；论文级科学与图表充分性 0/9 获得授权。** 下一阶段应按上述
题目特异缺口补科学执行或图表，而不是继续对已通过的文字做无边界润色，也不能把公开论文的
效应量当作目标答案。
