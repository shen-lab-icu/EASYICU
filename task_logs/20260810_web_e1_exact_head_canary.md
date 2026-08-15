# Web E1 exact-head engineering canary（2026-08-10）

## 结论

`fix/pi-workspace-review-20260809@e19980177b0cb9b07aeab4343cb5f93510c147a2` 已从 Web Copilot 用一句短命令完成一次 development-only、aware-only E1 engineering canary。真实 `gpt-5.6-luna` 通过本机 OpenAI-compatible provider 执行；11/11 科学步骤、分析验证、EvidenceStore 完整性、数值核验和稿件生成五道技术门全部通过。外层 gate 保持 `analysis_only`、`reportable=false`、`draft_unlocked=false`，因为本轮没有真人签署或论文 authority。

这不是 Canonical9 正式 Provider batch，也不改变冻结评分的 4/9。没有运行 naive、E2--E9 或 27 次正式实验，没有修改 Canonical9 问题、shared prompt、frozen paper rubric 或 case protocol。

## Task identity

- Task ID：`WEB-E1-EXACT-HEAD-CANARY`
- 分支：`fix/pi-workspace-review-20260809`
- exact SHA：`e19980177b0cb9b07aeab4343cb5f93510c147a2`
- exact image：`easyicu-research-agent:web-e1-e199801`
- image ID：`sha256:bd2d7ea0b52f69cdb91c359102b89fa699c77407bd55a3fee5095a116f0df058`
- Web job：`15e7105f6f19`
- Pipeline run：`run_20260811T030843_4d45a8`
- 运行目录：`/Users/haibo/easyicu/projects/study_9dbcdc7967f5f5b6/run_15e7105f6f19/pipeline/run_20260811T030843_4d45a8`
- Provider：真实本地 OpenAI-compatible endpoint；model=`gpt-5.6-luna`；`used_mock_llm=false`

## 本轮通用修复

上一轮 Web E1 在 robustness figure 的 typed input closure 上失败：同一个 `primary_or.json` 同时注册为 `statistic:primary_or` 与规范化后的 `statistic:primary_effect`，而文件内部只声明前者。typed loader 正确以 `statistic_evidence_value_missing` fail closed。

`e199801` 将统计身份归还 robustness owner：

- deterministic runner 独立写出 `primary_or.json` 与 `primary_effect.json`，每份文件的 `statistic` 字段与注册 identity 一致；
- 对 Planner 声明的其他统计标签，由 owner 生成 identity-safe sidecar，不再让多个身份别名指向同一个声明不相符的文件；
- robustness figure executor 同时兼容规范 `primary_effect` 与 legacy `primary_or`，二者并存但数值冲突时 fail closed；
- 新增正向、冲突和任意标签回归。

前两个相邻提交也属于本次 E1 闭环：

- `ec14035 fix(agent): seal data-quality figure lineage`
- `65ea7d3 fix(agent): bind live missingness audit input`

它们让数据质量图只消费 digest-bound missingness audit typed input，并在真实运行中由确定性 renderer 生成。

## 精确运行结果

### Pipeline readiness

- `required_step_count=11`
- `completed_step_count=11`
- `missing_steps=[]`
- `failed_steps=[]`
- 每个 step 的 `execution_ok=true`、`scientific_requirement_complete=true`
- `execution_complete=true`
- `analysis_validated=true`
- `evidence_complete=true`
- `numeric_verified=true`
- `manuscript_ready=true`
- `missing_evidence_count=0`
- `numeric_error_count=0`
- `analysis_error_count=0`
- `writer_probe_failed_steps=[]`

关键执行事件真实出现：

- `Using deterministic robustness runner for 08_robustness_replay.`
- `Step 8/11 complete: 08_robustness_replay.`
- `Using planner-scoped robustness figure executor for 08_robustness_replay_figure.`
- `Step 9/11 complete: 08_robustness_replay_figure.`
- `Using digest-bound measurement-missingness figure renderer for 10_data_quality_figure.`
- `Step 11/11 complete: 10_data_quality_figure.`
- `Research-agent run complete.`

### Scientific boundary

外层 `quality_gate.json` 的五项检查全部通过，但终态故意保持：

- `status=analysis_only`
- `reason=research_agent_pipeline_complete_human_interpretation_required`
- `reportable=false`
- `draft_unlocked=false`
- `paper_authorized=false`

因此这次运行只能证明 Web→Pi→ResearchAgentPipeline→EvidenceStore→结果解读/预览的工程和科学门闭环，不能作为 Canonical9 正式得分或论文结论。

## Web 用户路径 QA

1. 用户在既有 E1 项目中只输入“重新运行当前 E1 完整分析；保持 development-only、aware-only，任何阻断 fail closed。”并授权一次完整分析。
2. 对话的真实生命周期依次显示 provider、typed materialization、trajectory authority、runtime validation、11-step plan、每步 executor/repair/complete、figure audit、publication bundle、literature、manuscript 和 terminal event。
3. 运行完成后，用户只需再问“读取最新 E1 运行状态并解释结果”，Pi 自动读取最新 run、quality gate、evidence ledger、plan 和结果解读 owner。
4. Pi 在同一对话给出 140 ICU stays 的有界结果解读、OR 1.50（95% CI 0.49--4.60）及观察性/样本量/残余混杂/单库/24 h 窗口等限制；同时明确区间跨 1、不得作因果解释或论文授权。
5. 对话列出 9 个白名单产物。点击“图件画廊”后右侧打开 `figure_gallery.json`，可见 1 张 primary 与 4 张 supporting 图件：absolute risk、两张 robustness 和 data-quality missingness；产物状态明确“仅供分析 / 需要人工签署”。
6. 桌面三栏同时可用，对话和右侧预览均未见横向溢出、遮挡或裁切。稿件全文直接投影因体积超过 Pi 有界投影上限而以 `pi_projection_too_large` fail closed；受治理的“锁定论文草稿”产物入口仍保留。

## 验证

- robustness/data-quality 聚焦与邻接回归：`184 passed in 22.31s`
- Ruff：`All checks passed!`
- `py_compile`：通过
- `git diff --check`：通过
- exact image container smoke：`primary_effect.json` 与 `statistic:primary_effect` capability 均存在
- 浏览器：Web job `done`、科研流程 `7/7`、Pi 结果解读完成、图件右侧预览打开

### 推送后架构门收口

首次推送后的 push/PR Research Agent CI 同时指出 `agents/core.py.loc` 相对架构棘轮为 `+18`。这不是运行结果或科学合同失败，而是 `26f57cf`/`8f128b1` 的 Planner retry 指导仍留在 orchestration god module。`5383c45` 未刷新架构基线，也未提高提示词预算；它把 schema/retry guidance 移到已有 `agents/plan_payload.py` owner，使 `agents/core.py.loc` 从基线 `+18` 收缩为 `-1`。本地架构 diff 通过，Planner/重规划/固定提示预算相关 175 条回归全绿，Ruff 与 diff-check 通过。

## 下一步

- 保持本次 run 为 development-only engineering evidence，不写入 Canonical9 正式 ledger 或 Figure 2 score。
- `21410d2` 的 UNSIGNED package 仍 non-authorizing；由于后续技术 HEAD 已前进到 `5383c45` 及其证据提交，真人签署前应在新的冻结 exact SHA 上重新生成 digest、CI 和 unsigned package。
- 正式 E1→E9×3 仍等待新的 exact-head scientific package 与真人 clinical+methods 双签；不得因本次 Web canary 成功而提前启动。
