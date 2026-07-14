# research_agent 结构去债 · 拆分清单(freeze 后执行) · 2026-07-14

> **状态:已审阅，延期执行。** H2 Step 06 已于 2026-07-14 收口，但 H2 后续步骤及 E2/E3/H3 development runs 尚未全部完成；本文件仍只作为 freeze 后的候选迁移方案，任何源码移动都不得插入当前实验。2026-07-14 复审同时发现，原方案中的同名模块/目录、固定测试数、测试同步搬迁和“纯 git mv”表述均不够安全，以下已按可执行约束修正。

> 触发:2026-07-14 架构审阅。一次快照显示 research_agent 约占全仓 **57% 源码 / 73% 测试**,包根约 **134–135 个 .py**,含 3 个万行 monolith。这些是规模快照而非验收常量。不同静态审阅对 import cycle 的结论不一致，因此不能预设“无真循环”，必须先生成可复验的 import-graph/SCC 基线。这一任务只处理**结构债**，不夹带逻辑修复。
>
> **执行前置(硬约束)**:
> - 只在 **E2/E3/H2/H3 收口 + 引擎 freeze 之后** 动。freeze 前重构 = 违反 agent/CURRENT.md「⚠️不要做:bench 运行期间编辑源文件」。
> - 普通平铺模块允许 **纯 `git mv` + 改 import + 加 façade 再导出**；`validators.py` 这类单文件内拆分类不是 `git mv`，必须定义为“逻辑保持的代码提取”，单独审阅，禁止顺手改行为。
> - **保留向后兼容**:`research_agent/__init__.py` 与被移动的旧模块名保留 `from .<newpkg>.<mod> import *` 再导出,避免全仓 + 测试 + 外部 import 大面积改路径。
> - **禁止新增同名模块/包并存**:已有 `pipeline.py`、`evidence.py`，不得直接再创建同级 `pipeline/`、`evidence/`。本方案改用 `orchestration/`、`evidence_store/`；旧模块仅作为兼容 façade。仓内已经存在 `prompts.py` + `prompts/` 的历史冲突，应在动态导入基线后优先消除，不能把这种状态复制到别的模块族。
> - 与 Codex 错峰:动前 `git status` 确认目标文件不在 Codex dirty 列表。
> - 开工前记录三份动态基线:完整 test collection、公开 import/符号清单、包导入图/SCC；不得依赖本文中的历史数量。

---

## A. validators.py(11,256 行)沿 seam 拆 6 份 → `audits/`

按顶层 class 边界切,`audits/validators.py` 退化为 thin façade(保留 `dedupe_findings` + 共享 AST helper `_call_function_name/_extract_column_names/...` + 从各子模块 re-export)。

| 新文件 | 装什么(class/seam) | 约行范围 |
|---|---|---|
| `audits/validators_concept.py` | CohortAuditor · ConceptUsageAuditor · LLMConceptAuditor · parse/downgrade helpers | L102–1269 |
| `audits/validators_crossstep.py` | CrossStepCohortLock · CrossStepRegisteredOutput · StepSummaryFraction · CrossStepReconciliationTrace · CrossStepSourceStatus | L1271–1632, L4034–5235 |
| `audits/validators_model.py` | **PrimaryModelContractValidator**(单类 ~1,875 行,自成一文件) | L2159–4034 |
| `audits/validators_statistics.py` | StatisticalValidator · StatisticalGuard · ClinicalConstraintValidator | L5235–5566, L10684–11018 |
| `audits/validators_figure.py` | **FigureSourceDataValidator**(~4,700 行,最大 seam)· FigureContractQualityValidator | L5566–10684 |
| `audits/validators_replication.py` | ReplicationDesignAuditor · ReplicationResultComparator · PublicationClaimAuditor | L11018–11215 |
| `audits/validators.py`(保留) | `dedupe_findings` + 共享 AST helpers + 全部 re-export(façade) | 薄 |

> 注:`FigureSourceDataValidator` 自己就 4,700 行,是本轮复审 fail-open 最难定位的地方;拆出后它仍偏大,可二次沿其内部审计族(source/statistic/structural-accounting)再切,但**第一刀先按 class 边界**,别一次切太碎。

## B. pipeline.py(10,603)/ pipeline_execute.py(9,670)

`pipeline_execute` 已从 `pipeline` 拆出,是好先例。第二刀:把 `pipeline.py` 里的 **resume 加载 / plan 兼容 / authority 兜底** 抽到 `orchestration/resume_authority.py`(与现有 `pipeline_resume.py`、`runtime_artifacts.py` 归拢)。优先级低于 validators,单独一轮。不能使用 `pipeline/`,否则会与 `pipeline.py` 发生 Python 导入名冲突。

---

## C. 包根 134 文件 → ~11 个子包(bin map)

已有子包:`audits/ methods/ figures/ providers/ replication/ prompts/ case_plugins/ fallback/ runner_image/`。下面把根目录平铺文件按语义归箱(判断项已标 ⚠)。

**`orchestration/`(编排核心,18;旧 `pipeline.py` 保留 façade)**
pipeline · pipeline_cache · pipeline_config · pipeline_cross_db · pipeline_execute · pipeline_package · pipeline_phases · pipeline_primary_effect · pipeline_profiles · pipeline_report · pipeline_resume · pipeline_state · pipeline_write · pipeline_writer_aux · plan_utils · graph · runner · cost

**`gates/`(契约/合同门,10)**
contracts · article_contract · declared_product_contract · figure_contract · figure_contracts · ordered_stratified_contract · robustness_execution_contract · methodological_rigor · reporting_checklist · trajectory_plan_contract

**`evidence_store/`(证据/出处/权威,6;旧 `evidence.py` 保留 façade)**
evidence · provenance · lock_authority · runtime_artifacts · concept_audit_cache · review_artifacts

**`coder/`(coder/前检/修复,13)**
agentic_coder · coder_context · context · context_numeric · code_hygiene · code_patch · code_preflight · code_repair · code_repair_helpers · cohort_repair · summary_repair · repair_registry · structured_retry

**`executors/`(deterministic 执行器,12)**
deterministic_causal · deterministic_clustering · deterministic_cohort_flow · deterministic_descriptive · deterministic_missingness · deterministic_ordinal · deterministic_robustness · deterministic_sensitivity · deterministic_survival · estimators · robustness_panel · trajectory_stability_executor

**`routing/`(方法路由/能力/可行性,9)**
analysis_blueprint · analysis_method_suite · analysis_types · capability_registry · method_capabilities · method_compatibility · causal_audit · validity_signals · viability

**`cohort/`(队列/概念可用性,8)**
cohort_materializer · cohort_schema · concept_availability · concept_catalog · concept_dict_audit · concept_proposal · data_catalog · data_foundation

**`trajectory/`(H3 轨迹,5)**
trajectory_bundle · trajectory_contract · trajectory_resume_schema · temporal_features · temporal_semantics
⚠ `trajectory_plan_contract`→gates、`trajectory_stability_executor`→executors(或全部 trajectory_* 归一包,二选一,建议后者更内聚)

**`idea_mining/`(选题挖掘/发现,18)**
idea_mining · idea_mining_eval · idea_mining_extended_feasibility · idea_mining_feasibility_tier · idea_mining_funnel · idea_mining_priorart · idea_mining_pubmed · idea_mining_schema · idea_registry · idea_scope · hypothesis_generator · literature · tier2_jury · tier2_rubric · side_findings · discovery_handoff · discovery_package · discovery_story_figure

**`manuscript/`(成稿/渲染/评审,11)**
latex · bibtex · manuscript_post · pdf_render · publication_figures · figure_skill · figure_strategy · display_suite · visual_qa · reviewer · evaluation_scorecard

**`llm/`(LLM/记忆/面板,6)**
llm · llm_mocks · cross_model_panel · prompts · memory · experience

**`cases_bench/`(案例/基准,6)⚠**
icu_agent_bench · experiment_spec · case_contexts · easyicu_case_builder · icu_rules · study_design(+ study_design_playbook)
⚠ 与已有 `case_plugins/` 合并考虑;`icu_agent_bench` 是 prototype framework,别当冻结公共 benchmark。

**保留在根(跨包共享/入口 + 兼容 façade)**
`__init__.py` · cli · replication_cli · mcp_server · agents · architecture · schema · scalar_utils · projection · temporal_semantics(⚠或 trajectory)· skills · step_summary
> 这些是 CLI 入口 / 跨包 orchestrator / 通用 schema,拆包收益低,先留根。由于兼容 façade 也必须留在根，不能再把“根目录只剩 ~8 个 .py”作为硬验收；应统计**实现承载模块**是否完成归箱，并单独列出 façade 白名单。

---

## D. 执行顺序建议(每个可回滚批次一 commit)
1. validators.py 6-切(A)——**收益最大、复审最痛**,先做。
2. `evidence_store/` + `gates/`——安全门集中,后续复审最常 grep 的两类。
3. `executors/` + `routing/`——deterministic runner 与路由归位。
4. `orchestration/` + `coder/`——体量大、import 面广,放后面,façade 兜底充分再动。
5. `idea_mining/` + `manuscript/` + `llm/` + `cohort/` + `trajectory/` + `cases_bench/`——按需。

**分层验收**:

1. 开工基线:`pytest tests/research_agent --collect-only -q` 保存 pytest 最终 collection summary；另保存仓内公开 import 路径和 `research_agent.__all__`/约定公共符号。禁止用 `wc -l` 推断测试数。
2. 每个机械移动批次:导入 smoke、旧路径 façade/API 兼容测试、对应子系统 focused tests、`test_meta_benchmark_spec.py`、`ruff check`、`black --check`、`git diff --check`。
3. 每个逻辑保持的代码提取批次(尤其 validators):除第 2 层外，必须跑该 validator 全部反例/正控；提取 diff 由独立审阅确认没有条件或严重度变化。
4. 每个 durable commit/里程碑:完整 `pytest tests/research_agent/ -q`，collection 数与基线一致，pass/skip 变化均有解释。完整套件当前约需 30 分钟，不应在同一未提交批次里按单文件重复运行。
5. 每批确认导入 SCC 没有新增，并实际执行旧外部 import smoke；仅写 `import *` façade 不能自动证明兼容，尤其下划线私有符号、monkeypatch 路径与 pickled qualified name。

最终验收以“实现模块归箱 + façade 白名单 + 无新增循环 + API/collection 基线保持”为准，不使用不真实的“根目录约 8 文件”目标。

---

## E. 测试目录整理（独立、可选、最后执行）

> **测试本身不是债**:2026-07-14 的一次快照曾统计约 3,950 个测试函数 / 312 文件，但该数字会持续变化，不能作为迁移契约。正式开工时必须以 pytest 的动态 collection baseline 为准。测试是资产，不清理、不合并、不删。
>
> 源码拆包时**不强制同步搬测试**。pytest 测试路径不是公共 API，和源码同 commit 搬迁会扩大 diff、降低 blame 可读性，并可能同时触发 collection/plugin/fixture 问题。先保证源码兼容；测试镜像化只在所有源码迁移稳定后作为独立可回滚工程评估。

- 若后续确有导航收益，再按一个测试域一个 commit 使用 `git mv` 镜像化；每批先验证相对导入、conftest 作用域、pytest plugin/marker 与精确 collection node id 依赖。
- research_agent 测试占比高是源码规模与 fail-close 契约的自然映射，不是测试侧缺陷——不要因为「测试也多」去删测试换整洁。
- **可选优化(优先级最低,别现在动)**:当前 conftest 较少而测试规模持续增长,各文件可能自造 cohort/evidence/stub 有 setup 重复;真要收敛可抽公共 fixture 进 `tests/research_agent/conftest.py`,但这是独立小工程,不与拆包混做。
- 验收补一条:镜像迁移前后 pytest collection summary 与 node-id 清单都要比对；测试总数不减仍不足以发现“旧测试漏收集 + 新测试恰好补齐”的抵消。

---

## F. 当前决策与回主线条件

- **现在不执行 A–E。** 当前唯一动作是把本方案修成 freeze 后可安全执行的计划，不改共享引擎结构。
- 回到主线:H2 Step 06 已用同一 run 的 step-level resume 收口；下一步从 Step 07 继续，00–06 与正文图不重跑。证据见 `task_logs/20260714_h2_step06_framework_iteration.md`。
- 允许启动本计划的证据:E2/E3/H2/H3 CURRENT 均明确收口、无 benchmark 进程、工作树清洁、完整 research-agent suite 绿、collection/API/import-graph 基线已归档。
- 若拆分需要改变 validator 条件、严重度、runner 路由、prompt 或 evidence authority，立即停止并另开逻辑修复任务；不得把行为变化伪装成结构迁移。
