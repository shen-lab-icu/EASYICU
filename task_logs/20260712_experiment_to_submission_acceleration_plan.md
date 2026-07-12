# EasyICU 实验到投稿加速执行方案

> Task ID: `SUBMISSION-ACCEL-20260712`
> 日期: 2026-07-12
> 状态: active plan；本文件定义后续执行顺序和停止规则，不代表尚未完成的实验已经通过。
> 当前代码基线: `fix/easyicu-concept-bounds-enforcement` @ `a9fce77`；架构修复尚未 push。
> 当前实验真相: Figure 2 冻结 6/9，Fresh E3 完成 9/12；剩余 E2/E3/H3。

## 1. 目标与成功定义

目标不是把 9 个题目调到“看起来都成功”，而是尽快得到一套可投稿、可追溯、能诚实说明 EasyICU 通用能力边界的完整证据包，并把论文写作从实验的串行尾部移到并行主线。

本计划把“完成”分成两层：

1. **实验冻结**：每个问题都有唯一 artifact-of-record、固定协议/模型/代码/数据快照、可复核的 reportability 状态和 source data。`publication_ready` 与安全 fail-closed 必须分开报告；安全拦截不能伪装成分析成功，分析失败也不能通过继续加 case-specific runner 被强行洗成成功。
2. **投稿冻结**：正文、图注、表格、补充材料和 source-data bundle 中的每个主张都与当前证据一致，完成 integrity → adversarial review → revision → final integrity 门禁。

## 2. 唯一事实源

后续不得再从旧稿或旧实验反推当前状态：

- Figure 2 / agent 实验事实：`项目进度/benchmark实验/CURRENT.md`。
- Idea Mining / Figure 5 事实：`项目进度/idea-mining/CURRENT.md`。
- 六库数据与 Table 1 事实：`项目进度/数据底座/CURRENT.md`。
- 当前中文正文唯一工作源：`easyicu写作/10_写作工作区_20260610/02_草稿/EasyICU_中文正文_20260623.md`。
- 文章故事、图号和投稿优先级：`EasyICU_当前投稿主控计划.md`。
- 历史 Fig3/4/5 包只从 `EASYICU/research_output/ARCHIVE_INDEX.md` 指向的外置盘归档恢复；`research_output/` 本身是可清理 scratch，不是永久事实源。

当前正文中的“九题 9/9”“全部 gate-reportable”“统一 gpt-5.4”“旧 Fig5 已成立”等表述均视为**待证据重绑的过期声明**，在新 evidence-to-claim 对账完成前不得继续传播到摘要、图注或英文稿。

## 3. 需要先冻结的一项实验设计决策

### 推荐方案：platform capability suite

Figure 2 定位为 **protocol-versioned platform capability suite**：

- 保留 6 个已冻结题，不为追求表面上的单模型整齐度重跑；
- E2/E3/H3 只从当前失败步骤精确续跑；
- 每题披露模型、provider、运行日期、commit、数据快照和 protocol version；
- Figure 2 证明的是同一 EasyICU 架构跨方法族完成/安全拦截任务的能力，不声称比较模型优劣，也不声称单一模型的总体成功率。

这是最快且与当前证据最一致的选择。若改为“统一模型、统一 commit 的 9 题 canonical benchmark”，就必须重跑已冻结 6 题，预计会重新成为全文最大关键路径；在正式 protocol freeze 前应由作者明确选择，不能在正文里同时使用两种口径。

## 4. 双轨并行执行

### Track A — Result Freeze（机器重任务通道）

同一时间只允许一个 bench 或 discovery 重任务。

#### A0. 建立干净实验快照

- 在 Web/Copilot 并发修改之外建立干净 worktree，基于经审阅的 agent commit；不要在当前 dirty worktree 直接启动 canonical run。
- clean worktree 通过显式绝对路径读取现有 universe/run artifact，或把 `OUT_ROOT` 指向 durable 实验盘；不要假设 gitignored 的 `research_output/` 会自动出现在新 worktree。
- 冻结并记录：commit、所有 tracked diff、prompt/rubric version、模型字符串、provider、数据 manifest、运行参数、失败重试规则。
- `.env.local` 和 API key 只在本机注入，不写入计划、日志或提交。
- 先运行最小 completion probe；`/models` 返回 200 不等于 completion 可用。额度/上游不可用时数秒退出，不做长时间 patient retry。

#### A1. E3 最短路径

1. 从当前 Fresh E3 的 `06_secondary_adjusted_association` 精确 resume，并 `stop-after`。
2. 单独检查 estimator family、effect scale、fitted term/CI、penalized/convergence/separation、current evidence authority 和 source data。
3. Step 06 只有在 `status=ok` 且所有合同门通过后，才继续 Step 07；Step 07 通过后再生成独立 figure。
4. 不从 Step 01 重跑，不复用旧 Step 06/07 为论文证据。

#### A2. E2 / H3

- **E2 当前没有可 resume 的 canonical run，且 canonical JSONL 尚缺失**。API 阻塞期先核验 current export/provenance，并从已有 94,458-row discovery universe 生成、冻结 E2 输入和 SHA，避免重扫完整原始库；随后做一次 fresh aware run。
- **H3 universe/JSONL 已具备，但没有真实 artifact-of-record**。反管线化后 trajectory features、方法和聚类数必须由 agent 规划；做一次 fresh aware run，旧 deterministic clustering 不得抢 primary。
- E3 后，若 E2 输入尚未冻结则先跑 H3；若 E2 输入已冻结，优先跑 H3 暴露最高风险的纵向方法边界，再以 E2 作为低风险收尾。默认队列因此为 **E3 → H3 → E2**，但输入未就绪的题不得占住机器通道。
- fresh run 仍逐步 `stop-after` 验收；已有成功 checkpoint 的后续补跑只从第一个当前失败步骤 resume，禁止凭旧日志猜失败步。
- 只跑完整 EasyICU `aware` workflow；不跑 historical `naive` arm。

#### A3. 每题冻结包

每个题在记分前必须一次性固化：

- run plan、ResearchContext、protocol/model/commit/data snapshot；
- final + partial manifest、run status、audit log；
- 当前 producer 的 code/evidence id/SHA-256；
- 表/图 source data、figure contract、PNG/SVG/PDF/TIFF；
- 五维 scorecard 与 reportability/fail-closed 理由；
- durable artifact manifest，不能只留在可清理 scratch。

#### A4. Fig3 / Fig4 / Fig5 与数据门

- **Fig3/4**：先从归档恢复 source data、绘图代码、caption provenance 和多格式图，做数据版本、SHA、轴/单位/不确定性和当前定义复核。审计相符就复用，不盲目重算；只有 provenance 或数据定义不一致才重跑。
- Fig3 的六库 spot-check 若数字不变，直接由冻结 JSON/source 重建当前 2-panel 图，不重提六库；Fig4 若必须重算，只提取其当前比较所需的 MIMIC-IV/SICdb，不为该图重提其余四库。
- **Table 1 / Fig3**：完成真实六库 export bounds/profile spot-check 和 daggered mapping/source-data 复核后才锁数字。
- **Fig5**：先做 FiO2 分母、采样窗口和缺失结构的 outcome-blind feasibility gate；候选通过后，只能由 `tools/run_discovery_to_manuscript.py` 全链生成。手算只作 oracle，不能进入论文。

### Track B — Paper Assembly（并行轻任务通道）

实验等待 API 或运行时，持续推进正文，不等 9 题全部结束才开始写。

#### B0. 先做 stale-claim quarantine

- 以当前中文正文为唯一 base，建立 claim–evidence matrix。
- 所有数字、模型版本、run date、成功率、成本、图号和“已完成”表述先绑定 evidence id/SHA；无法绑定的改成结果槽位或删除。
- 旧 Supplement、旧 Figure 2–5 caption 和旧引用清单只作素材，不直接合并。

#### B1. 可立即完成的稳定内容

- title、故事定位、Introduction / Related Work；
- R1 平台架构与 human-confirmed workflow；
- Methods：概念层、prepared-data contract、agent-owned scientific planning、EvidenceStore/current-record authority、repair governance、三态 gate、benchmark rubric、Idea Mining protocol；
- Discussion 的原则、边界和限制骨架；
- Fig1 1a–1c caption，以及 Fig2–5 无数字 caption/schema；
- 新 Supplementary Methods、reproducibility envelope；
- Data/Code Availability、Ethics、CRediT、COI、Funding、AI disclosure；
- DOI/题录核验后的 `.bib`。

稳定段落可先完成中文结构并同步英文；不要先润色仍在变化的结果段。

#### B2. 必须等待证据冻结的内容

- Abstract 的全部结果数字；
- R2 / Figure 2 的完成数、模型/协议和记分；
- R3 / Table 1 / Fig3 数字；
- R4 / Fig4 的跨库误插补结果；
- R5 / Fig5 的候选、数字与结论；
- Fig1 1d 真实拦截例；
- 结果型 captions、成本、运行日期、模型版本和最终 commit/hash；
- Discussion 中对结果的综合判断。

#### B3. 一次性结果合并

全部 artifact 冻结后，只进行一次 evidence-to-claim merge：从冻结 manifest/source data 写入正文、图注、表格和补充材料；随后锁数字，避免在多个版本中重复手工同步。

#### B4. 投稿质量流水线

结构与证据合并完成后，顺序固定为：

1. Integrity audit：数字、引用、证据、图表和 provenance；
2. 独立 adversarial paper review；
3. 按审阅意见 revision；
4. Final integrity audit；
5. 生成 submission bundle、SHA manifest、冻结 tag/DOI 或投稿前归档包。

## 5. 加速治理与停止规则

### WIP 限制

- 16 GB 机器同一时间最多 1 个 bench/discovery 重任务。
- bench 等待/运行时可以并行做正文、引用、归档审计和轻量测试；不能再启动第二个 bench。
- 运行中不编辑会被 step worker 重新导入的 research-agent 源文件。

### 修复预算

一个真实失败点最多进入以下循环：

1. 判断是外部/API、数据/benchmark contract、还是共享架构问题；
2. 只有能表述为 case-neutral invariant 的问题才改 shared engine；
3. 一个通用修复 + 一次定点重跑；
4. 若仍是 case-specific 信息不足，修 benchmark item/rubric 或保持 fail-closed，不再新增 runner/关键词路由。

任何需要放松 `test_meta_benchmark_spec`、capability-drift 或 evidence-authority 探针才能通过的修复都立即停止。

### 不再重做的工作

- 不重跑 6 个已冻结题，除非发现影响其 artifact validity 的明确代码/数据 bug。
- 明确不重跑 E1、M1、M2、M3、H1、H2，也不重跑 E3 已通过的 Steps 01–05 及其已绿图件。
- 不全量重跑 9 题来验证一个局部修复。
- 不为题目新增决定暴露、结局、队列或模型的 deterministic runner。
- 不把 launcher `rc=0`、`/models=200` 或旧 manifest `status=ok` 当成科学成功。
- 不在 canonical protocol 中途换模型并隐去差异；若必须换，逐题披露。
- 不手算论文结果，不用自由图像生成重画数值图。
- 不在结果未冻结时反复润色摘要和 Discussion 数字段。

## 6. 阶段看板

| Phase | 机器通道 | 写作/审计通道 | 出口条件 |
|---|---|---|---|
| P0 事实源冻结 | 建立干净 worktree、冻结 protocol/model/data manifest | 建 claim–evidence matrix、隔离旧声明 | 唯一代码/数据/正文事实源明确 |
| P1 Fig2 收口 | E3 Step06→07；随后 E2/H3 定点 resume | 完成稳定正文、Methods、Supplement 骨架 | 9 题各有冻结且诚实的最终状态 |
| P2 平台结果 | 仅在审计不符时重生 Fig3/4 | 核对 Table1、caption、结果槽位 | Fig3/4 durable bundle + 数据门通过 |
| P3 Discovery | FiO2 gate → full discovery-to-manuscript | 写 Fig5 protocol/限制，不预写结果 | Fig5 full-pipeline provenance + source data |
| P4 全文合并 | 不再改实验，除非 integrity blocker | 一次性 evidence-to-claim merge，中英稿统一 | 正文/图/表/补充零数字冲突 |
| P5 投稿门禁 | 只做可复现性复核 | integrity → review → revision → final integrity | submission bundle + SHA manifest |

## 7. 近期立即动作

在 completion API 尚不可用时：

1. 建立正文 stale-claim quarantine 和 claim–evidence matrix；
2. 恢复并只读审计 Fig3/4 归档包；
3. 完成六库 spot-check / Table 1 映射复核；
4. 做 Fig5 FiO2 outcome-blind feasibility 复核；
5. 完成可稳定的 Methods、Supplement 和投稿声明。

completion API 恢复后的第一项重任务：在干净 worktree 中从 Fresh E3 Step 06 精确续跑并 stop-after；通过才继续 Step 07。该动作优先于任何新的架构重构或全题重跑。

## 8. 投稿完成门禁

缺少任一项都不标记“论文完成”：

- protocol/model/commit/data snapshot 已冻结；
- 每个主文数字有 agent-produced 或允许的平台确定性 artifact、current producer、evidence id 与 SHA；
- Fig2 无 placeholder，Fig3/4 通过数据审计，Fig5 有 full-pipeline provenance；
- main text、caption、table、supplement 数字交叉一致；
- 引用 DOI/题录已核验且无 citation orphan；
- 每张结果图有 source data、plot code、单位/CI 定义和 SVG/PDF/TIFF；
- Data/Code Availability、Ethics、CRediT、COI、Funding、AI disclosure 完整；
- submission bundle、复现清单和 SHA manifest 已归档。
