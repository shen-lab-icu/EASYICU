# Figure 2 M1 framework iteration

- 日期：2026-07-10
- 模块 / phase：`benchmark实验` + `agent` / `FIG2-CANONICAL9-GATE`
- 测试题：`M1_hepatobiliary_missingness`
- 模型：`gpt-5.6-luna`
- 执行臂：`aware` only
- 代码基线：`e220715` + 本轮 working-tree 框架修复

## 目标与决策

M1 是 Figure 2 九问中的缺失数据 / 测量过程 / 结局关联题。本轮不为追求一次“全绿”而覆盖失败证据，而是先复现旧 run 的每个失败点，将案例现象拆成通用框架问题，再以断点测试和 fresh M1 分层验收。

API 凭据只注入当前交互 shell，未写入脚本、日志、进度文档或 git。

## 权威失败基线

首轮路径：

`research_output/_canonical9_20260710/bench_m1_gpt-56-luna/M1_hepatobiliary_missingness/aware/run_20260710T154537_df4d68`

- 耗时 `9756.28 s`（约 2 h 42 min 36 s）。
- 12 个步骤中 7 个完成、5 个失败，最终 `diagnostic_only`，`manuscript_ready=false`。
- deterministic missingness 表确认加载队列 `n=74,829`，bilirubin 缺失 `40,754 / 54.462842%`，有测量 `34,075 / 45.537158%`。
- 原始 `df4d68` 保持不变；所有定点复测均在独立副本运行。

## 根因

### 1. 百分比序列化舍入被误判为图表溯源失败

Figure source 由整数计数重算全精度百分比，上游 CSV 保留 6 位小数。两处失败的最大差值仅 `2.95888e-7`，不是数据矛盾。原 validator 对所有 numeric 列统一使用 `1e-9`，错把显示精度当成科学结果不一致。

### 2. SDK 重试将单次 timeout 放大为一小时

旧环境为 `EASYICU_LLM_TIMEOUT=300` 且 `EASYICU_LLM_MAX_RETRIES=12`。首次请求加 12 次重试约为 `13 × 300 s`，与 `06_absolute_risk_context` 约 63:32、`08_sensitivity_comparison` 约 62:23 的失败时长一致。这不是 Luna 单次生成需要一小时。

### 3. replan cap 在 primary-preservation guard 之后再删除关键角色

旧 run 的 raw revised plan 仍包含 Table 1 和 genuine primary model，但后续 cap 只保护 completed step 和 figure-parent 单元，可再次删除 baseline/Table 1 与 primary-estimand owner。最终 revision 8 因此无法支持完整文章契约。

### 4. 过宽 preflight matcher 造成“假成功”

- 复杂 exposure/source repair 步骤被简化 missingness 计数 runner 抢走，却未产生 joint availability、invalid range、model availability 和 source reconciliation 契约。
- 负责生成 cohort overlap/attrition 的 owner 步骤被下游 sensitivity comparator 抢走，缺少上游输入后 clean-skip，外层却仍可记录为 `ok`。

### 5. 通用描述 / robustness 角色缺少确定性步骤级入口

06 的首次 Luna 代码约 93 KB / 1,100+ 行，仅因一个括号缺失再花费一次长修复请求。08 也生成约 78 KB 脚本，首次执行后因 `primary_or` 为空进入 contract repair。这两类角色均可从已锁定 plan/context/spec 确定性执行，不应每次生成超长代码。

## 通用框架修复

- `audits/validators.py`：仅 `_pct` 列使用 `1e-6` 绝对容差；计数、效应量、CI 和 P 值仍为 `1e-9`。
- `plan_utils.py`：plan cap 自动保护第一个 genuine primary-estimand model 和第一个结构化 baseline/Table 1 owner，且保持 cap 精确长度。
- `pipeline_execute.py`：
  - 复杂 exposure/source repair 不再被简化 missingness runner 抢走；
  - cohort attrition owner 不再被下游 comparator 抢走；
  - 按结构化输出契约精确路由 absolute-risk 和 robustness preflight；
  - primary owner 和 figure step 均有负向路由保护。
- `deterministic_descriptive.py`：通用 exposure prevalence / binary absolute-risk runner，包含 Wilson CI、ordinal/categorical 水平、continuous median/IQR 与 source-state 分类；不做事后分箱，不将缺失编码为 0。
- `deterministic_robustness.py`：读取 locked robustness specs、已验证 primary estimate 和 universe/primary cohort，产生 robustness matrix、membership change、outcome executability、specification grid 与 statistics artifacts；同一 stay-level scalar outcome 的 first/any 不冒充独立验证。
- `run_case_bench.sh`：默认 `timeout=900` / `SDK retries=0`，只信任精确 loopback URL，调用者注入凭据优先，curl 不在 argv 暴露 key，代码指纹包含 tracked diff 与 untracked 文件内容。

## 自动化验证

- 75 项 validator / replan / resume / deterministic-runner / route-ownership 聚焦测试通过。
- 20 项 association figure rescue 测试通过。
- 4 项 plan-cap 测试通过（228 项未选）。
- 所有本轮 Python 改动 Ruff 通过；`run_case_bench.sh` 通过 `bash -n`。
- 旧真实 figure source 产物在新 validator 下返回 `source_subset_matches`；超过 `1e-6` 的百分比偏差仍被拒绝。

## 定点真实运行

### 06 — Luna 旧生成路径诊断

`research_output/_diagnostic_m1_targeted_20260710/bench_m1_gpt-56-luna/M1_hepatobiliary_missingness/aware/run_20260710T154537_df4d68_t06`

- 首次 coder 约 10.5 min 返回，脚本因一处括号不匹配失败。
- 一次 repair 后生成 exposure/outcome summary、complete-case denominator、cohort flow、PNG/SVG 及 figure contract。
- 步骤 `complete`，退出码 0，Critic `pass`。

### 08 — Luna 旧生成路径诊断

`research_output/_diagnostic_m1_targeted_20260710/bench_m1_gpt-56-luna/M1_hepatobiliary_missingness/aware/run_20260710T154537_df4d68_t08`

- 首次 coder 约 10 min 返回，脚本执行后 `primary_or` 为空，触发 contract repair。
- repair 后步骤 `complete`，退出码 0，表中 OR `2.049` (95% CI `1.924–2.182`)。
- Critic 仍为 `needs_revision`；且该 OR 是 robustness 步骤临时补算，不能代替独立 primary step。

### 06 — 新确定性路径验收

`research_output/_diagnostic_m1_targeted_20260710/bench_m1_gpt-56-luna/M1_hepatobiliary_missingness/aware/run_20260710T154537_df4d68_det06`

- audit 明确记录 `fallback_reason=absolute_risk_context_preflight`。
- 读取 `n=74,829`，只解析 plan 结构化输入中的 `sofa2_liver_max` 和 `bili_max`，产生 19 行可溯源汇总。
- 步骤执行约 5 s，无 8787 连接，无 code repair，退出码 0。

## Fresh M1 验收

路径：

`research_output/_diagnostic_m1_fresh_20260710/bench_m1_gpt-56-luna/M1_hepatobiliary_missingness/aware/run_20260710T203942_f5ea9e`

- fresh plan 共 11 步，已包含 `03_table_one_baseline`、`04_absolute_risk_context`、独立 `05_primary_missingness_aware_association` 及 `06_cohort_definition_sensitivity`。
- 在 step 1 前发现合并 robustness 角色仍可被旧 overlap matcher 抢走，因此安全中断，完成路由与显式 artifacts 修复后从同一 run 续跑。
- 当前状态：`in_progress`。只有 fresh run 完成后才能判定 baseline、primary result figure、robustness 和整体 `manuscript_ready` gate。

## 边界

- 定点 resume 只是失败点诊断，不是 canonical/fresh 验收。
- 不将手工计算数字写入论文；最终可用结果必须由 research-agent pipeline 产生。
- 不运行 historical `naive` arm。
- M1 达标前不扩大到其余八问；继续优先断点修复。

## 2026-07-11：Step 8 reconciliation 与 Step 9 主关联闭环

### Step 8 最终验收

同一 fresh run 的 `04_absolute_risk_context_reconciliation` 最终为 `ok`，代码复用模式 `resumed_code_reuse`。详细 reconciliation CSV 共 16 行：10 行逐行绑定 registered parent table，6 行为明确记录理由的本地补算；cohort lock、registered output、fraction、source-status 与 reconciliation trace 五类门禁均无 error。

### Step 9 首轮真实失败

resume22 恢复本地 Codex Tools proxy 后从 Step 9 定点生成。初始脚本成功计算分母和模型 summary，但 `primary_adjusted_association.csv` 只有表头、0 行。直接根因是成功拟合的 `model_info` 漏写 `converged=true`，coefficient/risk helper 因默认 false 静默返回；同一 helper 还存在 latent intercept 索引错位。新 `PrimaryModelContractValidator` 正确将步骤拦为 contract violation，而不是接受 return code 0。

初轮数值锚点正确：source-aware baseline-complete `94,456 / 9,465`，bilirubin/SOFA complete-case `41,209 / 6,046`；bilirubin `log1p(bili_max)` 为唯一 primary，SOFA 为分开拟合的 secondary，普通调整项仅 age/sex/adm。

### Repair 反馈链根因与通用修复

旧 repair log 只传 `finding.message`，丢弃 `finding.detail.issues`；同时 coder prompt 只列字段名、不公开 canonical enum，导致模型反复用自然语言猜 `analysis_set`、`baseline_missing_policy`、`fit_status` 和 `exposure_role`。此外 validator 会把宽格式 `figure_source_data.csv` 的非模型行误识别成系数行。

本轮修复：

- repair prompt 尾部写入紧凑结构化 finding JSON，保留 validator、model_id、issue、expected/reported/allowed。
- 公开并强校验 canonical machine fields：`exposure_role`、`analysis_role`、`analysis_set`、`baseline_missing_policy`、`fit_status`；atomic exposure expression 禁止混入 prose。
- coefficient discovery 过滤 model_id/term/term_role 为空的宽表行；模型 summary 与逐项 coefficient/CI 做跨产物数值一致性验证。
- 对分类 baseline covariate 做零事件/零非事件单元检查；M1 的 `adm=EYE` 2/0、complete-case 1/0 因此不能被“converged”掩盖。
- separation fallback 若使用 statsmodels ridge，必须在 `fit_method` 报告 alpha，且 per-observation alpha 不得大于 `1/n`；sklearn 必须报告 `C>=1`。独立 oracle 证实旧 `alpha=0.01` 会把主 OR 从约 1.93 任意压到 1.33，故不能通过。
- StatisticalGuard 只有在至少两个有限 p 值存在时才提示 multiplicity；全空 `p_value` 占位列不再产生假阳性。

### Step 9 最终验收

resume23 复用最新 Step 9 code，结构化反馈经两轮修复后状态 `ok`。resume24 再次复用同一代码稳定复跑：约 11 秒、0 repair、0 contract finding。

最终机器合同：

- primary：`bili_source_aware_full`，`log1p(bili_max)`，source-aware，explicit baseline missing category，`n=94,458`、events `9,466`。
- bilirubin complete-case sensitivity：`n=41,209`、events `6,046`。
- SOFA source-aware/complete-case 均为 secondary/sensitivity，未与 bilirubin 互相调整。
- 四模型均明确 `separation_detected=true`、`penalized=true`；source-aware alpha `5.29335789451e-06`，complete-case alpha `1e-05`，均小于各自 `1/n` 上限。
- Agent 产出主 OR `1.9259885602`，95% CI `1.8555496206–1.9991014484`；coefficient CSV 与 summary 逐值一致。
- `coefficients.csv` 38 行，普通 adjustment source 仅 age/sex/adm；无 mutual exposure adjustment，无 `_n`/measured/LoS ordinary adjustment。
- standardized marginal-risk table 18 行，覆盖四个模型；PNG/SVG、figure contract、figure source data 均存在。
- Critic 无 unsupported claim、无 missing evidence；唯一剩余提示是复用代码时跳过可选 LLM concept audit，deterministic audits 已执行。

验证：新增/扩展主模型合同测试 17 passed；contract/primary/repair 聚焦回归 146 passed；StatisticalGuard 聚焦测试 4 passed；Python compile 通过。

下一步只跑 `05_primary_missingness_aware_association_figure`，不重跑前九步。

## 2026-07-11：Step 10–12 与最终 publication bundle

### Step 10：四角色关联结果图

同一 fresh run 的 `05_primary_missingness_aware_association_figure` 最终输出四面板 `publication_figure`：绝对结局风险（`descriptive_result`）、主调整效应（`primary_estimand`）、complete-case 缺失敏感性（`robustness`）和信号缺失率（`data_quality`）。图契约声明 `event_rate_panel / forest / dot_interval / bar` 四种图型；PNG/SVG、figure contract 与 7 个 trace tables 均闭合，contract/source/visual findings 为 0。

### Step 11：精确 registered-model robustness replay

初版 robustness matrix 只携带显示数值，缺少模型身份、事件数、系数来源和脚本指纹，且把同一 stay-level outcome 的两个编码变体画成 `N=0` 的独立结果。本轮将通用确定性 replay 改为逐行绑定 exact `spec_id × model_id × coefficient_term`，并携带 analysis set、exposure expression、fit method/status、model-contract N、event N、coefficient source 和 script SHA；任何 converged 行无法唯一绑定合同即 fail closed。

最终 `06_cohort_definition_sensitivity` 为 `ok`、0 repair。主分析为 `94,458 / 9,466`，LoS ≥6 h、≥1 d、≥2 d 分别为 `93,224 / 9,210`、`74,829 / 7,397`、`46,337 / 5,709`，complete-case 为 `41,209 / 6,046`。完整证据位于：

- `steps/06_cohort_definition_sensitivity/outputs/robustness_matrix.csv`
- `steps/06_cohort_definition_sensitivity/outputs/model_replay_index.json`
- `steps/06_cohort_definition_sensitivity/outputs/robustness_variant_coefficients.csv`
- `steps/06_cohort_definition_sensitivity/outputs/step_summary.json`

### Step 12：只画可估计行，分开报告 estimability

渲染器现在只从 direct parent 的声明产物读取数据，CSV 使用 round-trip float parser。6 个已估计 OR 行进入 `sensitivity_forest_source_data.csv`；2 个非独立 outcome 编码行不再复制主估计或伪造 `N=0`，而是进入 `sensitivity_estimability_source_data.csv`。输入只有 OR 时动态生成 OR forest + denominator/event panel，不再创建空白 risk-difference panel。最终 PNG/SVG/PDF/TIFF、合同和两个 source CSV 齐全，0 repair、0 contract/source/visual finding。

### 最终 publication bundle 根因与验收

Steps 1–12 首次全绿后，旧 run-level 主图仍只覆盖两个角色，导致 `publication_ready=false`。根因不是科学步骤，而是 `PublicationFigureSkill` 单遍、顺序依赖地发现 bundle：内容寻址去重会把 Step 10 复制的 source tables 登记回 parent step，选择器因此误判正确四面板 bundle 无 source data，随后退回到混合多模型表生成稀疏主图。

通用修复改为按 figure contract 中明确声明的 evidence id / basename / stem 跨 EvidenceStore 解析 source closure，候选优先 child step、direct parent，再按注册顺序；不再无条件绑定同 step 的未声明表。新增 parent-step source resolution 回归。`test_publication_figures.py` 44 项通过，robustness/source-trace/primary-model 相关 61 项通过，Python compile 与 `git diff --check` 均通过。

resume33 显式绑定原 run，只重跑 Step 12 与 finalization，返回码 0。最终结果：

- 12/12 steps complete；`numeric_verified=true`、`evidence_complete=true`、`analysis_validated=true`。
- `manuscript_ready=true`、`publication_ready=true`、`manuscript_text_ready=true`。
- numeric/evidence/analysis errors 均为 0；display suite 与 article figure strategy 均 complete、errors 为空。
- 最新 publication figure summary 为 `generation_mode=promoted_step_publication_figure`，`promoted_from_step_id=05_primary_missingness_aware_association_figure`，`audit_findings=[]`。
- run-level 主图覆盖 `descriptive_result / primary_estimand / robustness / data_quality` 四角色；PNG/SVG SHA 与 Step 10 源图逐字节一致。

最终证据：

- `manifest.json`
- `run_status.json`
- `article_figure_strategy_audit.json`
- `publication_figures/easyicu_publication_figure.figure_contract.json`
- `evidence/publication_figure_skill_summary_v4__publication_figure_skill_summary.json`
- `steps/05_primary_missingness_aware_association_figure/outputs/`
- `steps/06_cohort_definition_sensitivity_figure/outputs/`

M1 因此从 fail-closed 翻为 publication-ready 并冻结；Figure 2 九问计分由 5/9 更新为 6/9。后续只在 E2/E3/H3 中选择一个真实失败 step 继续 `aware`-only 定点迭代，不重复运行 M1。
