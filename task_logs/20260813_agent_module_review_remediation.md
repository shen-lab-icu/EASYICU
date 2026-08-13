# Agent 模块审阅与整改（2026-08-13）

分支 `fix/pi-workspace-review-20260809`，基线 `69fe7d1`（审阅开始时为 `ca1d168`；期间并行会话
落了 `dc42e32`/`f2f26ab`/`69fe7d1` 三个提交，其中 `dc42e32` 正是本轮 P1 所依赖的
`materialization_window` 改动，因此本轮所有验证已在新 HEAD 上重跑）。本轮是对 `src/easyicu/research_agent/`
的一次代码审阅，五项发现全部整改并逐条做了变异验证。**没有调用 Provider、没有读取患者数据、
没有构建镜像、没有跑 M3**；通过的是代码与离线合同，不是科研结果。

## 发现与整改

### P1 `data_constraints` 字符串截断 → 用户确认与执行窗静默丢失，repeated-stay 门 fail-open

`webserver/agent_pipeline_runs.py` 把 `cohort` / `confirmations` / `analysis_design` /
`materialization_window` 序列化后 `[:2_400]` **切字符串**。`sort_keys=True` 使
`materialization_window` 排在最后、`confirmations` 倒数第二，超限时它们整块消失，且剩下的
字符串不再是合法 JSON。

实测（`include_diagnoses` 28 条，StudyContext schema 上限 64，完全合法）：

```
stored length              : 2400
'repeat' token present     : False
'readmission' token present: False
materialization_window kept: False
confirmations kept         : False
```

危害链：`planning/scientific_review.py::repeat_units_possible()` 在这段文本里扫
`repeat` / `readmission` token → 返回 False → `required_method_layers_for_plan` 不加
`dependence` 层 → `reporting/scientific_maturity.py` 的
`REPEATED_STAY_DEPENDENCE_UNRESOLVED` / `REPEATED_STAY_METHOD_NOT_DECLARED` 两条 major
不产出。Web 侧 typed 主门 `study_contexts.py::analysis_dependence_finding` 只在
`cohort.exclude_readmissions` **显式**为 `False` 时触发，键缺失时沉默 —— 那种配置下被截断的
文本 backstop 是唯一防线。

`[:2_400]` 本身源自 `7922951`（旧债）。审阅时它还是未提交改动、后由并行会话提交为 `dc42e32` 的
那一步——把 `time_window` 从自带 `[:1_000]` 的独立字段挪进这个共享封顶 blob 的**末位**——把旧债
踩爆：`materialization_window` 按字典序恰好排最后，成为超限时第一个被切掉的键。

**整改**：新增 `_compile_data_constraints()` / `_elide_constraint_lists()`。按**结构**收敛而不是
切字符：超限时逐级缩短列表值（保留头部 + 显式 `[N omitted]` 标记，标记本身不含任何门会扫的
token），保证每个顶层键都在、值始终是合法 JSON；连全部列表清空后仍超限则以
`research_pipeline_data_constraints_too_large` typed fail-closed，details 里逐段给出字节数。

整改后同一输入：长度 1944、JSON 可解析、四个顶层键齐全、`repeat` token 保留。

### P2 `execution_environment_failed` 是全仓库唯一一处出现的孤儿状态值

`execution/failure_classification.py` 新增的 isolation 失败类写入
`status="execution_environment_failed"`，而同文件另外三个 class 都写 `execution_failed`
（8 个文件消费）。全仓库 grep 只有产出这一处；`test_runtime_failure_classification.py` 新增的
两条测试也只断言 `runtime_failure_class` / `runtime_repair_route` / `llm_repair_used` /
validator / progress_message，**没有断言 status**。

主门 `readiness.py` 的 `failed_steps` 用 `!= "ok"`，所以不会漏计（方向是 fail-closed）；但
`contracts/declared_product.py::_FAILED_STATUSES` 是显式白名单，且
`is_failed_step_status` 的前缀兜底只认 `fail_` / `failed_` 开头 —— host 拼写落在两者之外。

**整改**：把 `execution_environment_failed` 登记进 `_FAILED_STATUSES` 并加注释说明 host 拼写
必须显式登记；新增参数化测试 `test_every_terminal_status_is_recognised_as_a_failure`，对每个
terminal class 断言其 status 既 `!= "ok"` 又被 `is_failed_step_status` 认出。

### P3 `validate_run_against_article_figure_strategy` 从未接线

`planning/figure_strategy.py` 的这个 run 级校验器在 src 与 tests 里都是零引用。它包装的
`summarize_article_figure_strategy_coverage` 本身**已经**接进 `reporting/readiness.py` 并参与
`publication_authorized`，所以不是门缺失 —— 缺的是：figure 覆盖不足只存在于那份 projection 里，
**不产出任何审阅者能读到的 finding**。

**整改**：给校验器加 `analysis_family` 参数（必须由调用方从 final plan 解析，否则 finding 会和
它所报告的门不一致），并在 `execution/phase.py` 的 article-contract 块之后按同样的
try/except 模式接线。`build_article_figure_strategy` 对未知 family 会抛 `KeyError`，因此外层
try 会把它降级成 warning finding 而不是打断 execute phase。

`render_article_figure_strategy_for_prompt` 同样只有测试引用，但**不是缺陷**：figure strategy
经 `build_analysis_blueprint` 已经到达 Planner，该函数是重复渲染器；按 CLAUDE.md「不因零生产
引用就删除真实实现」保留不动。

（本轮先前误报过 `PlanScientificFinding` / `executable_scientific_step` /
`remediation_route_for_finding` 零引用 —— 那是探测脚本排除了定义文件本身导致的，三者在
`scientific_review.py` 内部分别被用 31 / 6 / 4 次，不是死代码，已更正。）

### P4 isolation probe 的空 argv 分支会抛未捕获异常

`execution/runner.py::_trusted_isolation_probe_command` 在 `len(command) < 2` 时返回 `[]`，
随后 `_run_capturing_with_descendant_reaping([])` 抛 `IndexError`（实测），而调用方的
except 只接 `(OSError, subprocess.TimeoutExpired)` —— 一个不可探测的命令会从「保留子进程原始
失败」变成崩溃。真实 CodeRunner 命令不会短于 2，属理论路径。

**整改**：返回 `Optional[List[str]]`，短 argv 返回 `None`；调用方在 `probe_cmd is None` 时直接
记 `probe_error="ProbeCommandUnavailable"`，走既有的「探针未完成、保留子进程原始结果」分支。

### P5 格式

`failure_classification.py` 顶层 `@dataclass` 前被 `dbd5521` 删成只剩 1 个空行，已补回。
（ruff 当前规则集不管；规范 venv 里没装 black，`black --check` 跑不了。）

## 变异验证（每条都做了，不是只断言本来就成立的事）

| 整改 | 变异 | 结果 |
| --- | --- | --- |
| P1 | 恢复 `json.dumps(...)[:2_400]` | 4 条新断言变红（`JSONDecodeError` ×3、fail-closed 未触发 ×1）；`inc=0/12/18` 保持绿——它们本来就不超限，是「小 study 无回归」用例 |
| P2 | 从 `_FAILED_STATUSES` 删掉新登记值 | isolation 用例变红（`execution_environment_failed`），timeout 用例保持绿（它用 `execution_failed`） |
| P4 | 恢复 `return []` | 复现 `IndexError: list index out of range` |

## 验证命令与结果

规范环境 `.venv/bin/python`（Python 3.11.15）。**以下为并行会话三个提交落地后、在新 HEAD
`69fe7d1` 上的重跑结果**（先前在 `ca1d168` 上的一轮数字已作废，不再引用）。

- `pytest tests/test_pi_copilot_research_workflow.py tests/test_pi_copilot_contract.py tests/research_agent/test_runtime_failure_classification.py tests/research_agent/test_runner.py tests/research_agent/test_family_figures.py tests/research_agent/test_package_dependency_directions.py tests/research_agent/test_plan_literature_bindings.py`
  → **230 passed, 3 skipped**（含 6 条包依赖方向门，确认 `execution → planning` 是既有合法方向）
- `pytest tests/test_agent_run_documents.py tests/test_webserver_route_contracts.py tests/test_webserver_security_hardening.py tests/test_scientific_readiness_projection.py tests/research_agent/test_declared_product_contract.py tests/research_agent/test_execution_phase_contract.py tests/research_agent/test_display_suite.py tests/research_agent/test_publication_figures.py tests/research_agent/test_scientific_maturity.py tests/research_agent/test_plan_scientific_review.py`
  → **449 passed**
- `ruff check src/easyicu/research_agent src/easyicu/webserver tests` → All checks passed
- `python tools/lint_progress.py` → OK（6 个 CURRENT.md 通过，3 warning）

## 边界（不要写成别的东西）

- 本轮**没有**跑 full exact-head CI，也没有跑全仓库套件；按 CLAUDE.md 的
  development test/CI checkpoint policy，full CI 留到 E1 11/11 后的冻结 checkpoint。
- 本轮**没有**解冻 Canonical9、没有改 shared prompt、没有动 frozen paper rubric；
  paper authority 仍 0/9，E1 仍停在待用户确认的科学配置门。
- P3 的接线会让 figure 覆盖不足新产出一条 **warning** finding。它不改变
  `publication_authorized` 的判定（那条路径本来就读同一份 coverage），只是让缺口对审阅者可见。
- CURRENT.md 里那条已知未修 P1（Analyzer 只看得见 `step_summary`、看不见本步 typed 产物）
  本轮**未动**：本轮复核确认属实（`agents/core.py::AnalyzerAgent.run` 只收 `step_summary` +
  evidence id 列表，`project_outbound_step_summary` 也只投影脚本自己写进 step_summary.json 的
  内容），但它需要先设计 outbound 隐私投影，属独立 owner 决定。

## 全量套件（诚实记录：未取得结果）

`tests/research_agent/` 全量跑了两次都**没有取回结果**：第一次被并行会话的提交改动了源码树、
第二次被我主动终止（同样原因——它读的源码在运行途中已经不是它开始时那份）。两次都以 exit 144
结束，**不得写成"全量通过"**。焦点+邻接共 679 项已在新 HEAD 上重跑通过；全量与 full exact-head CI
按 CLAUDE.md 的 development test/CI checkpoint policy，留到 E1 11/11 后的冻结 checkpoint。
