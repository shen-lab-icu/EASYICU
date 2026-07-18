# research_agent 架构总控板 / 剩余债务台账 — 当前单一执行视图

> 更新：2026-07-18 06:41 EDT
> 分支 / 生产代码基线：`refactor/agent-control-plane@e54f675`
> 当前策略：**先完成有边界的 Track B-Core 架构整理，再 fresh 重跑 E3/H2/E2，随后执行 3–6 个 held-out 全流程。**

## 文档权威关系

| 层 | 文件 | 只回答什么 |
|---|---|---|
| 当前状态 | `项目进度/agent/CURRENT.md` | 现在在哪个 commit、正在做什么、下一步和 blocker；这是日常唯一入口 |
| **组件总控（本文件）** | `task_logs/20260717_remaining_debt_register.md` | 每个架构组件完成/部分/待做、验收证据和下一动作 |
| 安全设计 | `task_logs/20260715_agent_freeze_refactor_safety_plan.md` | 规范化等价、单调 authority、职责边界、freeze 不变量 |
| 历史设计细则 | `task_logs/20260716_agent_architecture_optimization_checklist.md` | rev.2 原始设计、测试矩阵和性能口径；**不再用于判断实时进度** |

若四者冲突，按上表从上到下取最新状态；安全不变量不能被进度文档放松。

## 当前量化快照

| 指标 | 冻结基线 | 当前 `e54f675` | 变化 |
|---|---:|---:|---:|
| `_execute_one_step` 行数 | 6,694 | 6,029 | −665（−9.9%） |
| `run_execute_phase` 行数 | 8,451 | 7,631 | −820（−9.7%） |
| `pipeline_execute.py` 行数 | 14,631 | 11,048 | −3,583（−24.5%） |
| `pipeline_execute.py + authority/{typed_binding,plan_authority}.py` | — | 12,613 | 新职责模块带来净 +75 行显式类型/边界/兼容代码；这是职责迁移，不冒充代码删除 |
| `research_agent` 模块 / 顶层兼容路径 | — | 214 / 157 | canonical 实现已归入 3 个职责子包；旧顶层 façade 为归档/API 兼容而保留 |
| package / import edge | 8 / 674 | 11 / 704 | 新增 `gates/`、`execution/`、`authority/`；由自动 graph gate 约束 |
| 潜在 import SCC / 循环模块 | 1 个・最大 103 / 103 | 3 个・最大 24 / 31 | parent-artifact cycle cut 将 103 模块巨环拆为 24/5/2；该静态图含 lazy/local import，不能冒充 import-time SCC 全消除 |
| E3 Step02 性能验收 | 旧 6 calls / 373.5s | 1 call / 0 repair / 26.8s | active wall −92.8% |
| fresh E3/H2/E2 | — | 未启动 | Track B-Core freeze 后执行 |
| held-out 全流程 | — | 未启动 | fresh 三题后执行 3–6 个 |

`tools/arch_measure.py` + `tools/arch_baselines/pipeline_execute.json` 是 LOC/闭包的可重跑门；运行性能仍以真实 run receipt/audit log 为准。

## 组件 scoreboard

| 组件 / 工作流 | 状态 | 当前证据 | 下一动作 / 完成定义 |
|---|---|---|---|
| A1 失败分类 / provider 总账 / repair budget | **完成** | schema-v5 总账、attempt-owned accounting、typed RepairReason、E3 真实 6→1 call | 不扩 case-specific reason；只修真实通用漏洞 |
| StepAuthorityCapsule / CheckpointAuthority | **完成** | content-addressed capsule、checkpoint 显式选择、resume/revalidation 回归 | 保持 checkpoint 唯一选择器；不得扫描“较新”候选 |
| Evidence authority / success commit | **完成（当前边界）** | canonical `authority/registration.py` + EvidenceStore 严格快照 + `StepEvidenceCommit`；旧 `evidence_registration` 是同一 module object | 不扩大成第二套 current authority；未来 prepare/seal/commit 变化须单独事务审查 |
| Visual GateEvaluator | **完成（职责子包）** | canonical `gates/visual.py`；typed `VisualGateResult/Decision`；旧路径同对象 alias | 保持 read-only；不得吸收 provider、repair 或 authority mutation |
| Deterministic / figure contract gate | **完成（职责子包）** | canonical `gates/contract.py`；read-only findings；pre/post canonicalization 顺序锁 | 不得把会写文件的 preparation 混回只读 gate |
| Concept gate / concept audit execution | **完成（职责子包）** | `gates/concept.py` + `execution/concept_audit.py`；policy 与 provider/cache/receipt 生命周期分离 | 保持两层分离；旧路径只作同对象 alias |
| Figure contract preparation | **完成（职责子包）** | `execution/figure_preparation.py`；8 个塑形/规范化 helper | 只处理已授权 figure 产品，不选择科学设计 |
| Publication figure execution | **完成（职责子包）** | `execution/publication_figure.py` + `SealedRendererState` | 保持 rendering-only 边界；旧路径同对象 alias |
| RepairCoordinator | **完成限定职责** | `repair_coordination.py` 只承接 patch→可选 rewrite transport；事务/分类属于 provider ledger/A1 | 不把 gate/science 塞进 coordinator；名称按限定职责理解 |
| Execution state | **部分完成** | `StepWorkerProgress`、`ConceptQuarantineState`、`SealedRendererState` | 只在真实跨边界读写集需要时继续值对象化；不造万能 state bag |
| StepExecutor / RunCoordinator | **部分完成** | 已有 seam/coordination 模块，核心 orchestration 仍在 `pipeline_execute.py` | 继续从主体直线控制流提取可测职责，不以“新建文件”冒充完成 |
| PlanAuthority | **完成（纯 candidate authority 边界）** | `authority/plan_authority.py` 用冻结 typed result 承接 completed-step snapshot、estimand/figure 保留、plan cap、robustness lock 投影、typed/trajectory/companion shaping 与 scientific no-op 判定；`e54f675` 独立对抗审阅 ACCEPT | provider 调用、revision/evidence 注册、cohort mutation、runner 重建、replan budget 继续由 orchestrator 单一持有；不得扩成第二个 Planner |
| TypedBindingResolver | **完成（3/3）** | `authority/plan_scope.py` + `authority/typed_binding.py` 承接 scientific signature、lineage/binding/schema receipt/manifest/resume 与 resolver；`dfb76b6` 用单一 resolver 替换 step 内闭包，每次显式传当前 plan，旧路径 identity 不变 | 保持 evidence fail-close、exact unpublished ID/alias 边界和 Planner 科学所有权；不再为凑 LOC 继续切本职责 |
| 目录 / import cycle 治理 | **当前批完成，持续门禁** | 7 个稳定实现及 PlanAuthority 归入职责子包；parent artifact authority 进 `authority/parent_artifact.py`，distribution seal 回 renderer；graph gate 当前 214 modules / 157 top-level / 704 edges / SCC 24/5/2 / 0 literal dynamic import | 后续模块随真实职责提取归位；每批以 cyclic-module count + largest SCC 为风险门，SCC 个数仅报告；不机械降低顶层文件数 |
| B2 canonical 跨-run memory | **完成** | `a9cb05c`；新 profile 显式 off，旧 profile canonical JSON 不变 | canonical 永不重开；非 canonical 才允许显式 opt-in |
| B3 step 并发 | **关闭，无需实现** | canonical 三题因 replanning + primary cohort + typed deps 被正确强制串行 | 保留 serial-gate 契约；不拆安全守卫追求伪加速。跨库 replicate 另属非关键路径 |
| B6 跨库 metadata 契约 | **blocked（数据会话）** | 数据字典/回调仍有并行 dirty 修改 | 等数据层提交并 re-lock profile；不得把 extraction bounds 混成 physiological `valid_range` |
| B7 dormant primary runners | **已达成，不物理删** | `_PRIMARY_DETERMINISTIC_RUNNERS` 空集 + registry lock | 投稿实验前不做化妆性删除；live `figures/*.py` 不得误删 |
| B7-3 display labels | **后置独立变更** | 会改变 display contract、source SHA 和可见文字 | 单独审稿图合同变更；不得当“零风险清理”顺手做 |
| B8 middleware/hooks | **待判定，当前不做** | empty middleware 只会增加抽象层 | 只有出现两个以上真实 hook consumer 才引入 |
| fresh 三题 + held-out | **待做** | shared engine 尚未 freeze | Track B-Core freeze → fresh E3/H2/E2 → 3–6 held-out；结果不得反向诱导 case-specific shared patch |

## 接下来三个可验收 bundle

1. **剩余控制面关账 + B6**：只读审计仍标“部分完成”的 Execution state 与 StepExecutor/RunCoordinator；freeze 前必须把每项变成“完成”或有测试支持的“必要 orchestrator 剩余”，不得把 partial 静默带过。同时等待并行数据会话提交干净字典/profile，只核验/补齐 source-concept、单位、来源库/表、伴随列与 native export intake，不把 extraction bounds 混成 physiological `valid_range`。
2. **冻结门**：串行分片回归 + meta/capability + capsule/resume/provider/evidence authority + arch/graph diff，并锁定唯一 commit/profile/dictionary/model/prompt/rubric/retry policy。PlanAuthority 已令 `run_execute_phase` 达到原 `≤7,660` 结构目标；`_execute_one_step` 仍比诊断目标 `5,980` 高 49 行，但没有新的职责边界时不为凑数抽取。
3. **fresh 三题 + held-out**：唯一 freeze 版本 fresh 跑 E3/H2/E2，再跑 3–6 个未参与修复的全流程任务；结果不得反向诱导 shared-engine case patch。

## 架构 freeze 的完成定义

- Agent 继续拥有 exposure、outcome、cohort、method、estimand；确定性组件只执行/审计/渲染已锁定规格。
- `pipeline_execute.py` 不再承载已识别的 gate/concept/figure 具体实现；canonical imports 不经旧 façade。
- 没有新增 import SCC；现有 SCC 数量/最大规模不回退，并有自动基线。
- 旧 import 路径、sealed replay/public API 保持兼容；fresh run **不等于**可以删除 shim 或 legacy migration。
- `test_meta_benchmark_spec.py`、capability drift、evidence authority、resume/revalidation、golden 全绿。
- scoreboard 不得残留未解释的“部分完成”：要么完成职责边界，要么明确证明剩余逻辑是 orchestrator 必需胶水并写入契约测试。
- 性能不回退：E3 Step02 仍为 1 call / 0 repair / active wall 约 26.8s 量级；架构整理不承诺再复制 93% 提速。
- freeze 后 fresh 跑 E3/H2/E2；再跑 3–6 个未参与修复的 held-out 全流程。meta lint 不能替代 held-out。

## 硬约束 / 不要做

- 不在 shared engine 加 H2/E2/E3、KDIGO、MIMIC 或九题特定路由/提示规则。
- 不新增 primary deterministic runner；不以裸关键词或 validator 文案驱动科学路由。
- 不放松 provenance、fail-close、meta/capability 探针来换题目通过。
- 不机械搬完所有顶层文件；文件整理必须是职责边界形成后的物理归位。
- 不因 fresh run 删除旧路径：归档 run、sealed scripts、公共 API 和未来 replay 仍要求 façade/retirement 策略。
- 不触碰并行数据会话持有的 `concept/callbacks.py`、`data/concept-dict.json`、`data/sofa2-dict.json`。
- 不在实验进程运行时编辑 research-agent 源码。

## 历史文档说明

- 2026-07-16 rev.2 checklist 的 9 步、未勾 checkbox 和“Track A 后才改 Track B”是当时设计记录；7 月 17 日用户已改为 arch-first，实时状态以本表和 `CURRENT.md` 为准。
- 2026-07-15 freeze plan 的等价、authority、职责边界仍有效；其中“先跑实验、现在不拆”的执行顺序已被 7 月 17 日决定取代。
- 详细 commit/test 证据仍见 `task_logs/20260718_bundle3_concept_audit_boundary.md` 及 `CURRENT.md` 的日志指针，本表不复制长流水账。
