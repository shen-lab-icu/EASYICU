# research_agent 架构总控板 / 剩余债务台账 — 当前单一执行视图

> 更新：2026-07-18 05:51 EDT
> 分支 / HEAD：`refactor/agent-control-plane@daebd49`
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

| 指标 | 冻结基线 | 当前 `daebd49` | 变化 |
|---|---:|---:|---:|
| `_execute_one_step` 行数 | 6,694 | 6,028 | −666（−10.0%） |
| `run_execute_phase` 行数 | 8,451 | 7,785 | −666（−7.9%） |
| `pipeline_execute.py` 行数 | 14,631 | 12,527 | −2,104（−14.4%） |
| `research_agent` 模块 / 顶层兼容路径 | — | 211 / 157 | canonical 实现已归入 3 个职责子包；旧顶层 façade 为归档/API 兼容而保留 |
| package / import edge | 8 / 674 | 11 / 686 | 新增 `gates/`、`execution/`、`authority/`；由自动 graph gate 约束 |
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
| PlanAuthority | **待做（Typed binding 后）** | 完整 replanning 边界仍捕获 25 个外围名并写 plan/cohort/runner，当前不宜先搬 | Typed binding 收口后先抽纯 candidate normalize/validate，provider/注册/cohort mutation 留 orchestrator |
| TypedBindingResolver | **下一核心 bundle** | 约 1.15k LOC 自包含 lineage/binding/manifest 家族、约 101 个直接回归；cycle cut 后反事实图显示 typed module 不再进 SCC | 建 `authority/typed_binding.py` + typed resolver seam；resolved-input JSON 字节一致、唯一 current/path/SHA/identity-row 继续 fail-close |
| 目录 / import cycle 治理 | **当前批完成，持续门禁** | 7 个稳定实现归入职责子包；parent artifact authority 进 `authority/parent_artifact.py`，distribution seal 回 renderer；graph gate 当前 211 modules / 157 top-level / SCC 24/5/2 / 0 literal dynamic import | 后续模块随真实职责提取归位；每批以 cyclic-module count + largest SCC 为风险门，SCC 个数仅报告；不机械降低顶层文件数 |
| B2 canonical 跨-run memory | **完成** | `a9cb05c`；新 profile 显式 off，旧 profile canonical JSON 不变 | canonical 永不重开；非 canonical 才允许显式 opt-in |
| B3 step 并发 | **关闭，无需实现** | canonical 三题因 replanning + primary cohort + typed deps 被正确强制串行 | 保留 serial-gate 契约；不拆安全守卫追求伪加速。跨库 replicate 另属非关键路径 |
| B6 跨库 metadata 契约 | **blocked（数据会话）** | 数据字典/回调仍有并行 dirty 修改 | 等数据层提交并 re-lock profile；不得把 extraction bounds 混成 physiological `valid_range` |
| B7 dormant primary runners | **已达成，不物理删** | `_PRIMARY_DETERMINISTIC_RUNNERS` 空集 + registry lock | 投稿实验前不做化妆性删除；live `figures/*.py` 不得误删 |
| B7-3 display labels | **后置独立变更** | 会改变 display contract、source SHA 和可见文字 | 单独审稿图合同变更；不得当“零风险清理”顺手做 |
| B8 middleware/hooks | **待判定，当前不做** | empty middleware 只会增加抽象层 | 只有出现两个以上真实 hook consumer 才引入 |
| fresh 三题 + held-out | **待做** | shared engine 尚未 freeze | Track B-Core freeze → fresh E3/H2/E2 → 3–6 held-out；结果不得反向诱导 case-specific shared patch |

## 接下来三个可验收 bundle

1. **TypedBindingResolver（已由依赖测量选定）**：先迁共享科学签名 kernel，再迁 lineage/binding/manifest/resume 函数，最后用 typed resolver seam 替换 step 内解析；目标 `pipeline_execute.py ≤11,450`、`run_execute_phase ≤7,660`、`_execute_one_step ≤5,980`，且 resolved-input JSON 字节一致。
2. **PlanAuthority 的纯决策边界**：只抽 replanner candidate normalize/validate typed result；provider 调用、revision 注册和 cohort mutation 留 orchestrator。
3. **B6 → 冻结门**：数据层提交后 re-lock metadata 契约；最后串行分片回归 + meta/capability + capsule/resume/provider/evidence authority + arch/graph diff，随后 freeze 并 fresh 跑三题。

## 架构 freeze 的完成定义

- Agent 继续拥有 exposure、outcome、cohort、method、estimand；确定性组件只执行/审计/渲染已锁定规格。
- `pipeline_execute.py` 不再承载已识别的 gate/concept/figure 具体实现；canonical imports 不经旧 façade。
- 没有新增 import SCC；现有 SCC 数量/最大规模不回退，并有自动基线。
- 旧 import 路径、sealed replay/public API 保持兼容；fresh run **不等于**可以删除 shim 或 legacy migration。
- `test_meta_benchmark_spec.py`、capability drift、evidence authority、resume/revalidation、golden 全绿。
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
