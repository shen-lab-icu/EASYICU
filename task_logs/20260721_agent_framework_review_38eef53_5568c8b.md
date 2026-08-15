# Agent 框架与近期修改审阅（38eef53..5568c8b，14 commits）

**日期**：2026-07-21 21:37 EDT
**审阅对象**：`refactor/agent-control-plane` 上 `38eef53`（CURRENT.md 记录基线）之后到 `5568c8b`（HEAD）的 14 个提交（约 +2,550 行），以及 research_agent 框架当前架构状态。
**方法**：逐 diff 精读 + 新文件全文精读；手动运行 CI 不覆盖的三道门禁；聚焦测试矩阵重跑；对失败测试做基线归因（临时 worktree + PYTHONPATH 强制源码 + git bisect）；对照 `baselines/easyicu_vs_baselines_systematic_comparison_20260528.md` 的 25-基线定位审计刷新差异化判断。
**结论**：**本批 14 个提交无需要回滚的回归；方向正确（可确定性计算的步骤持续从 LLM Coder 手中收回给 host）；发现 2 条遗留问题（非本批引入）与 3 条设计层建议。**

---

## 1. 手动门禁验证（记忆中标记"CI 不强制、需手动跑"的项）

| 门禁 | 命令 | 结果 |
|---|---|---|
| 架构 lower-is-better | `.venv/bin/python tools/arch_measure.py --diff tools/arch_baselines/execution_phase.json` | **OK 无回退**；`execution/phase.py` 实际比基线 −78 行 |
| 模块图（zero-SCC/canonical surface） | `.venv/bin/python tools/research_agent_module_graph.py --diff tools/arch_baselines/research_agent_module_graph.json` | **exit 0 无漂移** |
| Golden 特征化 | `pytest tests/research_agent/test_char_golden_run_bundle.py` | **3/3 通过** |
| 新模块 ruff | 6 个新/改模块 | 全绿 |
| 聚焦测试矩阵 | 本批 15 个相关测试文件 | **762 passed / 2 failed**（2 条失败归因见 §3.1，非本批引入） |

注意 `8ff19dc` 在同一提交里刷新了 `tools/arch_baselines/execution_phase.json`（`gates/preflight.py` 8300→8364，+64 行即新增的 carve-out）。方向上属合法基线刷新（同提交其它文件下降、HEAD 复测无回退），但"改门禁的同一提交里改基线"仍是需要评审时专门看一眼的模式。

## 2. 本批 14 个提交的主题与逐项判断

### 2.1 主题 A：missingness / source-availability 审计确定性化（7eb8209→bde215c→d02b613→c6ba6fb→5568c8b）
- `execution/runners/deterministic_missingness.py`：host 生成自包含审计脚本，零 LLM Coder 调用。直接落实两条既有教训：missingness-audit coder 曾系统性 27.6 分钟超时；纯计数步骤不该交给 LLM。**逐行审阅通过**：脚本为纯静态字符串（无注入面）；`_measured` 指示语义推断（binary_event_presence）的四重同意条件（值全 0/1、flag/值/计数三方逐行一致、无缺失）足够窄；分母合同缺失时写 `blocked` 摘要而非假成功。
- 选择器 `source_availability_audit_executor_owns_step` 是**精确三元合同**（method + 精确 outputs 集 + 精确 typed inputs 集），不是关键词路由——符合"gate allowlist 反模式"教训的正确侧；不匹配时降级回普通 Coder 路径，是可接受的降级方向。
- `figures/missingness_source.py` + `figures/sealed_registry.py`：sealed renderer 消费 digest-bound 快照、从密封整数计数重算百分比、不选变量不读队列。**adapter registry 是正确的结构起点**（此前 5 个 repair_id 是 pipeline.py 里的 if/elif 链，本次只迁移了 missingness 一个）。
- 小缺口（见 §4 建议 B）：`_validated_source_frame` 要求所有行 `missingness_kind == "measurement_missing"`、`indicator_semantics == "measurement_availability"`，任何 `structural_no_source` / `binary_event_status_complete` / `measurement_flag_conflict` 行都会使整个密封渲染退出（返回 None 走回非密封路径）。结构性无来源恰是该审计的科学卖点与跨库常态，当前测试只锁定了纯 measurement_missing 的正例，没有锁定混合场景的回退行为。

### 2.2 主题 B：host cohort 物化权威（96116e8、59980b6、2d0c63f、f10169d）
- `authority/run_input.py` 新增 `_seal_host_cohort_materialization` 等：host 物化的 cohort + 流水账（attrition ledger）以单一封闭权威登记，且**封存前自验**（`_host_cohort_materializer_authority_error` 校验 canonical parquet SHA vs sealed evidence、flow CSV 摘要、首行 universe/末行 n_cohort/逐行 `n_before−n_excluded==n_remaining` 对账、symlink/containment）。这一自验强度在同类 agent 框架中未见对应物。
- `cohort/schema.py` `_build_cohort_with_flow`：纳排按锁定顺序逐条计数（CONSORT 式顺序 attrition），确定性可复现。
- `execution/cohort_adoption.py`：resume/plan-phase 采纳已物化 cohort 而不再调度 Coder（perf 正确）。采纳校验＝定义 SHA + flow/provenance 全等 + parquet 行数；任何解析失败都**禁用采纳而非放水**（fail-safe 方向正确）。
- resume plan-scope 等价（`plan_scope.py`）：`legacy_host_checkpoint_may_inherit_plan_scope` 只允许"无 scope 的 host-materializer checkpoint"在 scope 无歧义（候选计划 scope 唯一，或其它 completed record 携带 scope 可判）时一次性迁移，且 `_selectively_revalidate_resume_successes` 重审时立即补章 `plan_scientific_signature`——**迁移是单调收敛的，不是放宽 fail-close**。审阅通过。

### 2.3 主题 C：其余修复（逐项验证）
- `9162b93` + 前序 Docker 提交：runtime 探针 `docker run` 去掉 `--rm`，改为显式 `_teardown_container` 确认；成功路径也先 inspect 再收产物。与 Docker Desktop 卡死的既有事故史一致。`assert container_ref is not None` 在传入 fallback_name 时安全（风格上可改为显式 raise，非阻断）。
- `70e105f` Table One `_token`：int/float 统一为 "number" token（`1.0`→`1`），bool 分支在 int 之前（bool 是 int 子类，顺序正确）；声明侧与数据侧走同一函数，一致性成立。修复真实的 1 vs 1.0 分类层级失配。
- `7345b7b`：`_declared_output_scope_contract` 改用统一谓词 `_step_expects_figure`，修掉 `fig:` 别名步骤收到"本步未声明图产物"矛盾指令的问题。正确且最小。
- `a49967a` `research_context/prompt_scope.py`（251 行新模块）：Planner 上下文投影 = 全量目录（每变量一行含 role/dtype/source/missing/observed/caveat + 全 roster SHA-256 回执）+ 约 36 个变量的全量元数据。"transport 调度而非科学选择"的边界在 docstring 与实现上都成立；被省略变量仍可被 Planner 显式选择并在步骤级附回全量元数据。`_guide_segments` 的 24 个锚点对 prompt-pack 编辑 fail-close（抛错而非静默发全文），是有意设计，但构成 prompt-pack 与代码的强耦合维护点。
- `8ff19dc` preflight `_proven_unavailable_audit_return`：**方向是减少误阻断**（接受"来源列不存在→显式 unavailable 审计行"的正确代码），但实现是又一个精确 AST 形状模板（要求 `x = col in df.columns` 赋值 + `if not x:` + 首语句为 status="unavailable" 的字典 append + return）。同语义的其它写法（内联条件、`not in`、walrus）仍会被阻断。与"preflight shape-template over-block"既有记忆同族——是补丁不是治本。

## 3. 发现（按严重度）

### 3.1 【遗留，非本批】`test_provider_budget.py` 2 条测试 committed-RED 且不稳定
- `test_pipeline_resume_restores_durable_budget_and_blocks_without_new_call`、`test_pipeline_default_budget_executes_two_semantic_repairs_and_final_audit`。
- **归因证据**：在临时 worktree（PYTHONPATH 强制该 commit 源码，已打印 `easyicu.__file__` 验证）上，`38eef53` 与 7-19 冻结点 `c1032ae` 均复现同样失败 → **不是本批 14 个提交的回归**。
- **不稳定性证据**：同一 HEAD 同一测试两种失败模式交替——(a) `ValueError: stop_after_step_id='01_summary' is not in the active analysis plan`（快速失败）；(b) 走进**真实 Docker runtime 依赖探针**并 60s 超时（`Docker execution-runtime dependency capture timed out`）。git bisect 在该区间非单调（判出的"first bad" `04c6718` 是纯文档提交，随后其 good 端点直接复测为红），坐实 flaky。本机 `docker info`/交互式起容器正常（29.2.1，约 1s），超时疑与并行 agent 会话争用 Docker 有关。
- **建议**：(1) 给离线 mock 测试提供 hermetic runtime 探针（mock 掉真实 `docker run` 依赖捕获）；(2) runtime 探针降级时应产出 typed blocked 状态，而不是让 `stop_after_step_id` 校验以令人误导的 ValueError 爆炸；(3) 在 CURRENT.md ⚠️ 登记为已知红，避免未来会话重复归因（本次归因花了约 8 个工具调用）。

### 3.2 【LOW】cohort 采纳路径的 parquet 内容锚定存在窄残留窗口
`load_materialized_analysis_cohort_result` 校验定义 SHA + flow 全等 + 行数，但 provenance JSON 里没有 parquet **内容**摘要（`cohort_sha256` 是定义哈希）。内容锚定依赖两条后续链：evidence record 已存在时的 sealed-SHA 比对，或 `materialized_cohort_authority_ref` 存在时的 typed authority 校验。残留窗口 = 无 authority_ref 且 evidence 首次登记的场景下，plan-phase 写盘与 execute-phase 采纳之间的同行数篡改不可检。**建议**：物化时把 `cohort_parquet_sha256` 写入 `*_provenance.json` 并在 loader 校验，一行即可闭环，与项目"处处内容寻址"的自我标准对齐。

### 3.3 【设计确认项】密封 missingness 渲染器拒绝混合 missingness_kind（见 §2.1）
若属有意保守（v1 只接管干净场景），至少补一条测试锁定"含 structural_no_source 时回退到非密封路径"的行为；否则建议扩展渲染器支持 structural_no_source（仍是纯计数，可渲染为 0% available + 标注）。

### 3.4 【债务延续】preflight 精确 AST 形状模板继续累积（见 §2.3 末条）
既有记忆已录两例 E3 误阻断同源。durable 修法仍是既定方向：从语法形状转向 artifact/结构证据（如以 host helper 的实际调用回执、审计行 schema 校验代替语法模板）。本批不阻断。

## 4. 框架层评价与基线对照刷新（vs 2026-05-28 定位审计）

5-28 审计列出的 10 个工程问题，截至本 HEAD 的状态：
- **#1 strict pilot 从未跑完 → 已解决**（E2/E3/H2 真实 provider 多步收敛，Figure 2 权威 6/9）。
- **#4 code repair 浅 → 已解决**（typed `repairs/` 包：exact patch/full-rewrite transport、typed reasons、确定性 source repair）。
- **#7 Docker 很少用 → 已反转**（Docker 为 canonical 执行路径，teardown/lifecycle 有专门权威与回归）。
- **#8 primary 依赖 LLM 代码 → 本批正是此方向**（table-one / trajectory-stability / missingness-audit / host cohort 物化均已 host 确定性接管；同时 B7 判定保持"primary 科学决策 agent-owned、runner 只做辅助计算/渲染"的边界，未越界）。
- **#9 pipeline.py 5k 行 → 结构上大改善**（227 modules / 0 SCC / arch 测量门 + 模块图门在线；`pipeline.py`/`execution/phase.py` 仍是两个 ~11k 行的诚实剩余债务，且有 lower-is-better 门看守）。
- **仍开放**：#2 外部 scorecard（canonical9 是内部协议，MedAgentBench adapter 未做）；#3 Tier-2/3 评审执行（`evaluation/` 有 adapter，canonical 运行未跑）；#6 literature grounding 弱。
- 三项差异化（概念层、value-level 证据绑定、manuscript 耦合的可复现信封）自 5-28 后**进一步拉开**：本批的 host 自验封存（checkpoint 发布前用自身权威校验器重查 SHA/对账）与 digest-bound sealed renderer，在 AI-Scientist-v2 / HealthFlow / OpenLens-AI 等对标中均无对应物——那些框架的 reviewer/critic 全部是 LLM 化、非确定性的。
- 对标项目仍领先的位置：外部基准（HealthFlow 的 EHRFlowBench 是冻结公开集）、技能/记忆广度（Voyager/HealthFlow）——后者是 EasyICU 有意不做（审计风险），前者是真实差距。

## 5. 复现命令

```bash
# 门禁
.venv/bin/python tools/arch_measure.py --diff tools/arch_baselines/execution_phase.json
.venv/bin/python tools/research_agent_module_graph.py --diff tools/arch_baselines/research_agent_module_graph.json
.venv/bin/python -m pytest tests/research_agent/test_char_golden_run_bundle.py -q
# 聚焦矩阵（762/764；2 条 provider_budget 红为遗留 flaky，归因方法见 §3.1）
.venv/bin/python -m pytest tests/research_agent/test_deterministic_missingness_runner.py \
  tests/research_agent/test_association_figure_rescue.py tests/research_agent/test_repair_registry.py \
  tests/research_agent/test_coder_output_scope.py tests/research_agent/test_cohort_materialize_from_prose.py \
  tests/research_agent/test_cohort_schema.py tests/research_agent/test_resume_model_roster_migration.py \
  tests/research_agent/test_resume_revalidation.py tests/research_agent/test_char_resume_revalidation.py \
  tests/research_agent/test_provider_budget.py tests/research_agent/test_planner_prompt_resource_scope.py \
  tests/research_agent/test_coder_context_repair_preflight.py tests/research_agent/test_table_one_method.py \
  tests/research_agent/test_docker_runner.py tests/research_agent/test_preflight_role_ownership.py -q
```

---

## 6. 【第二轮】整包结构独立审视（2026-07-21 22:05 EDT 补充）

覆盖说明：以下基于全包尺寸/fan-in 清单、词表与边界方向 grep 查证、门禁盲区核对；`validators.py`/`phase.py`/`preflight.py` 函数体为抽样读非逐行。包总量 212,794 行、21 个顶层文件、27 个子包。

### 6.1 尺寸治理盲区：全包最大文件不在架构门里
`audits/validators.py` **12,989 行 / 51 个顶级定义**（全包第一大，超过 phase.py），但 `arch_measure` baseline 只钉 12 个历史重构文件，未含它；同样未钉：`plan_utils.py` 4,977、`agents/core.py` 4,059、`contracts/declared_product.py` 3,509、`figures/skill.py` 3,458、`discovery/idea_mining.py` 3,350。前端有明文软预算（JS 1,500 / CSS 600），agent 侧只有"被修过的文件不许回退"。建议：arch_measure 增加包级 top-N 报告并把 validators.py 纳入 baseline。

### 6.2 方法词表散布 ≥11 处（系统性风险源）
`*_METHODS` frozenset 类词表分布在 repair_registry / plan_utils / prompt_scope / gates/{numeric_reduction,contract,preflight} / reporting/readiness / audits/validators / contracts/declared_product / execution/phase / trajectory/contract **11 个文件**；单个方法串 `missingness_and_source_availability_audit` 出现在 **4 个文件**（runner 合同、phase compact 集、repair_registry planner-methods、renderer CONTROLLED_METHOD）。每新增一个确定性能力需同步 3-4 处注册表——canonical9 逐题爆 family-specific gate blocker、"unregistered figure trace key" 正是此散布的直接后果。`planning/capability_registry.py` 只集中了 **family 级**能力（有 drift 测试），**method-string 级**词表未集中。建议：method 串归一到 `planning/analysis_types` 或 capability_registry 的 per-family alias 表，消费方引用常量；加"同一 method 串唯一定义处"drift 测试。

### 6.3 生成式字符串脚本模式两头不占
4 个 runner（robustness/missingness/descriptive/table_one）以 `textwrap.dedent` 内嵌数百行 Python 字符串（`deterministic_robustness.py` 2,335 行主体即脚本）。字符串代码不吃 ruff/coverage/类型检查，只能整体 `exec` 测试（test_deterministic_missingness_runner.py:81 即如此）；而脚本内部又 `import easyicu.research_agent.icu_rules`——**既非自包含（依赖镜像内包版本），又失去真模块的全部工具链**。项目已有 `implementation_bundle_sha256` 内容寻址机制可封存真模块。建议：审计逻辑入 runner_image 真模块，生成脚本退化为 thin invocation，SHA 对模块源码封存。另：生成脚本内 `analysis_plan.json`/`research_context.json` 解析失败为 `except Exception: pass`，审计范围静默从"计划声明输入"漂移为"全部 _measured 概念"——host-owned 文件解析失败应 hard-fail。

### 6.4 溯源词汇被历史路径污染
标准 host cohort 物化：evidence_id=`analysis_cohort_execute_repair`、producer=`cohort_repair`、evidence 层 generation_mode=`"llm"`，而 checkpoint 层 generation_mode=`"deterministic_cohort_materializer"`（run_input.py:1426）——同一产物两层两词，且"repair"实为常规路径。"repair" 在包内至少四义（repair_registry 的 sealed renderer / repairs/ 的 LLM patch / cohort/repair.py / producer 名）。对外部审稿人读 evidence 账本有实际误导。短期在 schema 文档显式记录历史名语义；长期随 protocol version 升级统一。

### 6.5 gates→execution 方向违例（1 处，图门查不到）
`gates/contract.py:62` `from ..execution.runners.deterministic_robustness import replay_locked_memberships`。无环，但只读 gate 依赖 execution runner 模块方向反了；replay helper 应住叶子（methods/ 或 contracts/）。建议给 arch 门加"包→包允许方向表"（当前图门只查环与 canonical surface）。其余方向抽查干净：authority→execution 0、repairs→execution 0、figures→pipeline 0。

### 6.6 pipeline.py "入口"名不副实（已知债的精确化）
自述"只作入口并延迟委派"，实测 67 个顶级定义 / 10,786 行，仍拥有 sealed-renderer seal/render 双 if-elif 派发（~7 个 repair_id 分支，9908-9934 等）与 resume 兼容加载。本批 sealed_registry 只迁 1/7。phase.py + pipeline.py 双 ~11k 构成事实双头编排（有 lower-is-better 门看住）。建议固定搭配：每次触碰该函数迁一个分支进 sealed_registry。

### 6.7 prompt-pack 锚点耦合
`prompt_scope._guide_segments` 以 24 个精确英文散文子串定位 guide 分段，措辞编辑即运行时 ValueError（有意 fail-close、有测试，但"代码匹配散文"本质脆弱）。建议 guide 内放显式 `<!-- SECTION:x -->` 标记，锚点匹配标记。

### 6.8 小件
- 生产代码 115 处中文（8+ 文件）；提示词内或有意，evidence description（如 "prose 纳排"）会进审计账本/复现包，投稿仓库建议统一英文。
- docker runner 成功路径新代码用 `assert container_ref is not None`（改显式 raise 更稳）。
- learning/（自动经验，默认关）、know_how/（人审卡片，默认关）、discovery/（假设挖掘）三个"知识"包并存，机制不同但命名界限不直观，README 一句话说清即可。

### 6.9 看似不合理但经查证合理的（勿"修"）
`case_plugins/` 休眠 runner（B7 六通道证实死代码但按协议保留）；`mcp_server.py` 平行外部面（有意）；21 个顶层文件（均为公共/跨域入口）；`data/*.bak_before_*` 快照（有意历史检查点）；providers/mocks.py 1,995 行（离线确定性 floor，合理）。

---

## 7. 【修复批】即时安全修复收口、协议升级债务延期（2026-07-22，未提交，工作区待审）

用户指示"全部修复"。本批完成所有可在当前协议下独立验证的即时修复，并把 3 项需要独立协议升级/架构重构的债务明确保留在 §7.3；因此不得把本节误读成整个审阅清单已经 12/12 物理完成。

### 7.1 修复清单（全部经测试验证）

1. **provider_budget 两条红测（§3.1 定性修正 + 修复）**。真实根因不是环境 flaky：mock 计划缺少 2026-07-19 `835d3e3` 起强制的 `planned_analysis_role` → Planner 拒绝 → 计划被兜底为仅剩 host 补挂的 `01_audit_panel` → `stop_after_step_id='01_summary'` 校验爆炸（证据：失败 run 的 `analysis_plan.json` 步骤列表 = `['01_audit_panel']`）。Docker 60s 超时是 `runner_kind="auto"` 撞真实 DockerRunner 的叠加模式（机器忙时）。修复：两个 mock 步骤补 `planned_analysis_role: "primary"`；三处 pipeline 构造加 `runner_kind="subprocess"`（hermetic）；`orchestration/resume.py` 的 ValueError 现在列出实际活动步骤 id（本次诊断如果有这个信息可省 6 个工具调用）。44/44 + 三连跑稳定 2/2×3。
2. **cohort provenance 内容锚定（§3.2）**：`cohort/schema.py` 物化时写 `cohort_parquet_sha256`（分块哈希，legacy 分支）；`load_materialized_analysis_cohort_result` 要求摘要存在且精确匹配。缺摘要的旧账本无法证明原始 parquet 字节，现直接拒绝采纳；fresh run 可确定性重物化，旧 run 不再以同一行数冒充内容权威。新增篡改拒绝 + pre-digest fail-close 两条测试。
3. **密封 missingness 渲染器（§3.3）**：`_validated_source_frame` 接受 `structural_no_source`（附加内部矛盾校验：structural 行 measured_one_n 必须为 0）；图上该类概念用独立纹理、颜色和 `No source` 图例，不再进入普通 `Measurement missing` 堆叠；`measurement_flag_conflict` 仍拒绝且回退行为已有测试锁定。新增 SVG 语义断言。
4. **生成脚本静默吞错（§3 之 S3b）**：`deterministic_missingness` 脚本对 `research_context.json`/`analysis_plan.json` 改为"缺失=合法旧态、存在但坏=响亮失败"；其余 3 个 runner 核查无同类模式。新增损坏计划 hard-fail + 缺失文件旧态两条测试。
5. **Docker runner assert→raise（S8a）**：4 处 `assert container_ref is not None` 收敛为 `_required_container_reference()` 显式 raise。64/64。
6. **gates→execution 方向违例（S5a）**：`replay_locked_memberships`+`_membership_audit`+`_identifier_column` 以 AST 逐字等价证明迁入新叶子 `robustness/membership.py`；`gates/contract.py` 与 `execution/runners/deterministic_robustness.py` 均改用 canonical 路径，对象 identity 保持。182 项相关回归绿。
7. **包间方向回归测试（S5b）**：新 `test_package_dependency_directions.py`（5 项）：execution 仅入口面+documented agents 边可导入；任何子包不得导入 pipeline；gates 不依赖行动层；methods 保持叶子。含阴性对照（解析器修正过一个 off-by-one——初版会漏检原违例形态，已证明能抓到）。顺带发现并显式放行 `agents→execution.method_capabilities`（Coder 提示需真实沙箱能力快照，理由已注释）。
8. **arch_measure 扩监控面（S1）**：TARGET_FILES 追加 6 个稳定脱管大文件（validators/plan_utils/agents.core/declared_product/figures.skill/evidence_store）；新增每次 diff 打印的包级 top-N 信息报告（不 gate，防下一个失控大文件）。`discovery/idea_mining.py` 正由独立工作流修改，暂只在 top-N 报告以 `*` 暴露，不把另一会话尚未提交的字节写进本批 baseline；由 Idea Mining 自身提交在工作树干净后纳管。工具 v1.8.0，基线重发。
9. **方法词表归一（S2）**：新纯常量叶子 `planning/method_vocabulary.py`（9 个确定性能力方法串）；4 个 figure 渲染器 CONTROLLED_METHOD、repair_registry planner-methods、missingness 选择合同、phase compact 集全部改引常量；新 `test_method_vocabulary_registry.py`（5 项）锁跨注册表一致 + 禁止在 6 个注册表文件里重打字面量。范围诚实说明：`cohort_definition_sensitivity` 等在 11 个文件的更广散布未一次扫平（启发式 prompt 集合等语义不同），留后续批次。
10. **sealed renderer 分支迁移（S6，收敛范围）**：distribution-v1/continuous-v1/ordered-v2 三个分支（seal+render 三处调用点语义可证等价）迁入 `figures/sealed_registry.py`；pipeline.py −71 行。**有意不迁**：ordered-v1（上游选择处无 seal 探测，adapter 路径会附加之=行为变化）与 absolute_risk/cohort_flow/sensitivity/association（实现仍是 pipeline 本地函数，注册会造成 figures→pipeline 禁止方向）——原因已注释在 registry，迁移前置条件=先抽出这些实现。测试 repoint 2 处（identity 测试改锁 registry↔figures）。
11. **prompt-pack 锚点（S7 安全替代）**：不改 pack 字节（在跑 run 的 receipt/resume 身份绑定 pack），新 `test_coder_guide_anchor_integrity.py`（6 项）对真实 `_CODER_GUIDE` 验证 25 段锚点解析、锚点唯一性、三个代表方法族的选段、坏锚点 fail-close——把运行时爆炸前移到测试时。显式标记：标记化迁移排入下次 protocol 升级。
12. **文档三件套（S4/S8b/S8c）**：`research_agent/README.md` 新增"溯源账本历史名"表（cohort_repair/execute_repair/generation_mode=llm 的真实语义）与三个知识包（learning/know_how/discovery）权威模型区分；evidence 账本唯一中文描述串改英文（golden 无绑定，验证通过）。

### 7.2 验证总账
- `arch_measure --diff`：绿（基线重发理由：phase.py +3=词表导入、pipeline.py −71=分支迁移、新增 6 个稳定受控文件；Idea Mining 脏字节未混入）。
- `research_agent_module_graph --diff`：exit 0（新增 2 模块未破坏零环/canonical surface）。
- golden 3/3；25 文件组合矩阵 **902/902**；sealed/publication figure 分片 **147/147**；provider_budget 三连跑稳定；ruff/black 全绿。
- 新增测试 5 文件 21 项 + 既有文件内新增 7 项。

### 7.3 显式未做（防止误读为遗漏）
- **S3a 字符串脚本→真模块**：4 个 runner 的整体模块化涉及 sealed script SHA 身份与 golden 刷新，需独立批次；本批已修其正确性缺口（吞错）。
- **S4 改名执行**：`analysis_cohort_execute_repair` 等改名绑定 golden/resume/capsule 身份，排入下次 protocol version 升级；本批以 README 语义表止损。
- **R4 preflight 形状模板治本**：架构级重构，方向已记录（从语法形状转 artifact/结构证据），不属本批。

### 7.4 Codex 收口复核（2026-07-21 23:30 EDT）

- 修正一处跨会话 baseline 污染：Claude 首次重发 baseline 时把另一会话尚未提交的 `discovery/idea_mining.py` SHA 写入其中。该文件现暂不进入 `TARGET_FILES`，但继续在 top-N 报告以 `*` 暴露；由其自身工作流在提交后纳管。
- cohort 旧账本兼容从“无摘要仍采纳”收紧为 fail-close；这是 pre-v1 retirement 下的内容权威要求，不再用行数/flow 一致替代 parquet 字节证明。
- `structural_no_source` 不再共享普通缺失的堆叠颜色与图例，现使用独立纹理、边框与 `No source` 图例；SVG 文本回归锁定两种状态同时存在。
- 复核验证：cohort+figure 专项 **81/81**；cohort/figure/provider-budget/依赖方向/词表/架构组合 **239/239**；golden **3/3**；arch self-diff、module graph、Ruff、Black、diff-check 全绿。未运行任何在线实验。
