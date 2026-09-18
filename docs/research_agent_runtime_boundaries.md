# Research Agent 现有运行时接口边界（A11 地图）

> 目的：在替换 runtime / 抽接口之前，先写清现有各层的所有权与扩展点。本页是**描述**，不是新接口；
> 核对坐标：`codex/dev9-web-acceptance-20260906@71b287377`（2026-09-14）。路径会漂移，引用以文件为单位。

## 1. 分层与所有权

| 层 | 所有者文件 | 输入 | 输出 | 不得越界 |
|---|---|---|---|---|
| Web / Copilot 外壳 | `webserver/pi_copilot/`（`service.py`、`gateway.py`、`turn_authority.py`）、`webserver/routes/pi_copilot.py`、`webserver/agent_pipeline_runs.py` | 用户消息、来源授权、计划决定 | 会话投影、任务提交、受控产物入口 | 不定义科学结果；只投影 Host 生成的回执 |
| 通用 agent runtime | `webserver/pi_copilot/node_app`（`package.json` 固定 Pi 0.84.1） | Host 工具定义 | 对话/工具调用 | 不持有证据或科学权限 |
| 模型客户端与预算 | `research_agent/providers/`（client/factory/mocks）、`authority/provider_budget.py`、`authority/provider_budget.py` 的 receipt | 请求、账本 | 模型输出、扣费回执 | 预算/防重复付费门不可被上层绕过 |
| Planner（计划所有者） | `research_agent/agents/planner.py`、`agents/progressive_planner.py`、`agents/plan_payload.py`、`planning/prompt_projection.py` | 研究上下文、文献、能力目录 | `AnalysisPlan`（wire schema 受 30KB 级门约束） | 不读患者行；不执行分析 |
| 计划权威与编译 | `schema.py`（`AnalysisPlan`/`AnalysisStep`）、`authority/plan_scope.py`、`planning/scientific_action_catalog.py`、`planning/method_adapter_catalog.py`、`planning/analysis_method_suite.py` | 计划候选 | 编译后的可执行步骤/动作 | 字段必须恰有一类权威（`plan_scope` drift 检查） |
| 执行内核 | `execution/phase.py`、`phase_support.py`、`candidate_loop.py`、`runtime/`、`execution/runners/selection.py`、`execution/step_executor_registry.py` | 已批准计划、typed 输入 | 步骤产物、审计、回执 | 确定性执行器只按精确 action/spec/输入认领（`StepExecutor.owns`） |
| 沙箱运行镜像 | `research_agent/runner_image/`（Dockerfile、`requirements.lock`、`base-image.lock`） | 源码闭包 | 每步容器执行 | 禁网、只读挂载；生成代码不能装包 |
| 证据与身份 | `authority/evidence_store.py`、`authority/run_receipt.py`、`execution/kernel_identity.py` | 产物字节 | digest 化证据、收据、内核身份 | 发布门 fail-closed；不得由渲染器回写 |
| 报告与读者 | `reporting/write_phase.py`、`reporting/manuscript_provenance.py`、`tools/build_manuscript_reader.py` | 已核验步骤记录 | 证据绑定稿件、读者投影 | 只投影，不计算/升级结果 |

## 2. 主要扩展点（现有接线口）

1. **新增一个宿主科学能力**：`planning/scientific_action_catalog.py` 的 `_RUNTIME_CONTRACTS`（typed outputs/executor key）
   → `planning/method_adapter_catalog.py`（owner 模块、验证引用）→ `execution/runners/<owner>.py`
   → `execution/runners/selection.py` 注册 `StepExecutor`（`owns` 只按精确 action + 完整 spec + expected_outputs）。
2. **新增一个计划字段**：`schema.py` → `authority/plan_scope.py` 的两处分类集合（drift 检查强制）
   → 若该字段由 Planner 经 wire schema 产生，必须核算 `agents/plan_payload.py` 的传输字节（当前 10 源 run-bound 仅余 ~57B；
   常量子字段可像 `TrajectoryStabilitySpec` 一样从传输 schema 裁剪、解码后恢复）。
3. **新增一条读者证据链**：`reporting/manuscript_provenance.py`（claim 投影）→ `tools/build_manuscript_reader.py`
   → `webserver/static/js/screens-agent-render.js` + `screens-guided-pi-preview.js`（事件委托）。
4. **依赖/镜像变更**：见 `docs/dependency_model_change_regression_policy.md`。

## 3. 身份与部署

- 执行内核身份：`execution/kernel_identity.py::build_execution_kernel_identity`（闭包 + 依赖锁摘要），运行前在镜像内复核。
- 运行收据：`authority/run_receipt.py`，记录代码版本、镜像、门状态。
- Web 启动预检：`tests/webserver/test_web_execution_runtime_preflight.py`（docker daemon/镜像在花钱前具名拒绝）。

## 4. 明确不做（本地图对应的计划）

- 不在此批次抽通用 runtime 端口（A11 实施后置），不更换持久执行库（A9 需先有故障证据），
  不接外部 MCP（A3 later）。
