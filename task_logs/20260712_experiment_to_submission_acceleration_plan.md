# EasyICU 实验到投稿加速执行方案

> Task ID: `SUBMISSION-ACCEL-20260712`
> 日期: 2026-07-12
> 状态: active plan；2026-07-12 用户将优先级改为“先完成全部实验，再进入正文装配”。
> 当前代码基线: `fix/easyicu-concept-bounds-enforcement` @ `a9fce77`；架构修复尚未 push。
> 当前实验真相: Figure 2 development 状态 6/9，Fresh E3 完成 9/12；剩余 E2/E3/H3。

## 0. 2026-07-12 优先级修订

本修订覆盖本文较早的“已冻结 6 题不重跑、立即并行组稿”建议：

1. 先把 E2/E3/H3 作为 **development stress runs** 跑完，只修它们暴露的 case-neutral 框架问题。
2. 随后冻结代码、模型、prompt、rubric、输入 SHA、hidden oracle 和 retry policy。
3. 在同一冻结配置下，对全部九题做 fresh paper-facing canonical evaluation；开发期的 6 个旧 artifact 可作诊断历史，但不再拼接成最终 Figure 2。
4. 增加从未用于修框架的 held-out tasks/variants 和人工盲审，避免“九题既是开发集又是测试集”。
5. 正文实质写作暂缓到 canonical/held-out 实验冻结后；API 等待期只做实验输入、评价协议、oracle 与执行准备，不进入结果段装配。

评价设计审计与 npj 对照见 `EASYICU/task_logs/20260712_fig2_evaluation_protocol_audit.md`。

## 1. 目标与成功定义

目标不是把 9 个题目调到“看起来都成功”，而是先得到一套可投稿、可追溯、能诚实说明 EasyICU 能力边界的完整实验包；实验冻结后再一次性装配论文，避免用开发期结果提前写稿。

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

## 3. Figure 2 两层评价设计

### 开发层：protocol-rich capability suite

当前九题已经参与框架迭代，因此 E2/E3/H3 与此前 6 题都属于 development/capability suite。它们用于发现通用架构缺口，不能单独证明 untouched generalization。

### 论文层：frozen canonical + held-out validation

- 架构修复结束后，冻结同一 commit、模型/provider、prompt/rubric、数据 SHA、预算与 retry policy。
- 全部九题在冻结配置下 fresh 运行；最终 Figure 2 不混用不同 commit/模型的开发 artifact。
- 推荐每题 3 次独立 fresh run；若资源限制，则至少完成九题 pass@1 并对预先指定的高随机性任务重复，且把结论限定为 case-suite evidence。
- 另加 3–6 个未参与开发的 held-out tasks/variants，包含可完成正控和应 fail-closed 负控。
- 每题披露 task-specific oracle、hazard/forbidden-claim key、人工盲审与一致性；不把内部 validator 通过替代为科学正确性。
- Figure 2 仍称 **prespecified nine-task ICU research capability suite**，不称 comprehensive/general benchmark。

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

### Track B — Paper Assembly（canonical 冻结后启动）

按照用户最新优先级，正文实质装配推迟到 Figure 2 canonical/held-out 实验冻结后。下列结构保留为后续写作顺序，不在当前实验阶段执行结果段写作。

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

- development 阶段不为每个局部修复全量重跑九题；只定点验证失败步骤和相关回归。
- 旧 E1/M1/M2/M3/H1/H2 与 E3 Steps 01–05 可作为开发诊断历史复用，但不再直接充当最终 paper-facing canonical artifact。
- 只在框架与评价协议冻结后进行一次统一的九题 fresh canonical batch；若冻结后发现真实框架 bug，升级 protocol version 并明确作废受影响的结果。
- 不为题目新增决定暴露、结局、队列或模型的 deterministic runner。
- 不把 launcher `rc=0`、`/models=200` 或旧 manifest `status=ok` 当成科学成功。
- 不在 canonical protocol 中途换模型并隐去差异；若必须换，逐题披露。
- 不手算论文结果，不用自由图像生成重画数值图。
- 不在结果未冻结时反复润色摘要和 Discussion 数字段。

## 6. 阶段看板

| Phase | 机器通道 | 写作/审计通道 | 出口条件 |
|---|---|---|---|
| P0 开发 stress | E3 Step06→07；随后 H3/E2 | 只准备 E2 input、oracle、rubric 与 run manifest | 剩余三题完成或诚实 fail-closed；通用 blocker 收口 |
| P1 Protocol freeze | 冻结 clean worktree/model/prompt/data/retry | ICU+统计专家预审 task/oracle/hazard | 不再按终态结果调 shared engine |
| P2 Canonical9 | 同一冻结配置下九题 fresh runs/repeats | 盲审 plan 与 conclusion safety | 九题同版本 artifact + 重复性/资源数据 |
| P3 Held-out | 运行 3–6 个未参与修复的正/负控 | 评分一致性与 adjudication | generalization evidence 不来自开发九题 |
| P4 其余结果 | Fig3/4 数据复核；Fig5 full pipeline | 建 claim–evidence matrix、开始稳定正文 | 全部主图 durable bundle |
| P5 全文与投稿 | 不再改实验，除非 integrity blocker | merge → integrity → review → revision → final integrity | submission bundle + SHA manifest |

## 7. 近期立即动作

在 completion API 尚不可用时：

1. 冻结 Figure 2 评价协议草案、per-task oracle/hazard/forbidden-claim schema；
2. 生成并冻结 E2 canonical universe 与输入 SHA；
3. 建立最终 clean worktree、模型/prompt/rubric/data manifest 与 retry policy；
4. 设计 3–6 个不参与修复的 held-out 正/负控，但不向 agent 暴露 evaluator gold；
5. 不启动正文结果段装配。

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
