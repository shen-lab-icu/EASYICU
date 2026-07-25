# research_agent 架构优化 · 详细执行清单 · **rev.2（Codex 复核后）** · 2026-07-16

> **状态说明（2026-07-18）**：本文件保留为历史设计与验收细则，**不再是实时进度表**。其中未勾 checkbox 不表示尚未完成；原“先 Track A / E3 resume、再 Track B”顺序已被用户 2026-07-17 的 **arch-first → fresh E3/H2/E2** 决定取代。当前状态看 `项目进度/agent/CURRENT.md`，组件 scoreboard 看 `task_logs/20260717_remaining_debt_register.md`。

> **用途**：可逐条勾选、可逐条验收的执行项。供用户过目 + Codex 复核后正式执行。
> **rev.2 修订（Codex 复核）**：①新增 **P0-5 性能验收**（最大缺口——原清单只证正确不证变快）；②A2 增加 **StepAuthorityCapsule**（真正解 resume/context 耗时）；③修正 RepairTransaction 为**分支**状态图；④重定 Track B 风险（B1-1/B3/B6-2/B7-4）；⑤里程碑改**分片串行完整回归**而非永久跳过；⑥baseline commit 范围含 freeze plan。五项开放问题已决（见 §决策）。
> **两轨关系（硬）**：**Track A 全部先完成、验收、合并、冻结；Track B 在 Track A 期间只允许只读核验，源码修改一律后置。两条线不同时改源码。**
> **总原则**：只改执行治理控制面 + agent 设计层；**不碰护城河**（evidence store / NumericClaim / validators / restricted-AST evaluator / replay）。
> 图例：🟢低风险纯净 · 🟡中（碰路由/预算/控制面，需回归） · 🔴高（碰 validator 条件/严重度/evidence 权威） · 🔵先只读核验再定改法。
> 配套：方案 `20260716_agent_architecture_fix_proposal.md`；审阅证据 `20260716_agent_architecture_review_overfit_integration.md`。

---

## 北极星 · 通用性不变量 + 借鉴口径（贯穿所有 Phase）

> 最终目标是 **ICU 通用 agent**；9+3 题只是 harness，不是目标。以下两条是所有改动的验收透镜。

**通用性不变量（每个控制面改动都要过这关）**：
- 控制面（plan/route/repair/gate/render）**零 per-question 代码**：新 ICU 问题应无需新增 bespoke 分支/renderer/gate/prompt 字面量。
- 路由靠**结构/产物证据**（source_table、method family、product-contract 角色），不靠问题关键词。
- 失败分类是**通用类别**（code_bug / data_limitation / policy），适用任意问题，非 per-case handler。
- 图渲染 keyed on **figure-data contract 角色**——新问题的图是新 contract 实例，不是新代码。
- deterministic runner 只做「接受 Agent 已锁定参数的计算器」，不选暴露/结局/方法（选了即 per-question）。
- **成功口径不是「12/12 通过」，而是「控制面无 per-question 分支」**。真正的通用性证明是 **freeze 后跑 3–6 个完全未参与修复的全流程 held-out 任务**（一个第 13 题只算 smoke test）；`meta_generalization` bench 只是**防过拟合探针**，不能替代真实 held-out 执行。保留并强化 `capability_registry` + meta_generalization 作探针，但 held-out 全流程才是判据。

**借鉴口径：借模式，不借运行时**（延续 freeze 计划结论；通用性目标使其更成立——越是未见问题越需要领域不变量，通用框架恰恰没有）：
- **可借的模式**（作为模式非依赖）：显式状态机/图（A2 的 StepExecutor/RepairCoordinator/GateEvaluator 即其领域化）；typed tool 层（MCP，内部步应走同一 typed 契约——B5）；reflection/self-repair 环（RepairTransaction=其有原则版）；plan-execute 分离（A2 形式化）；episodic vs semantic memory 区分（保运行/resume 记忆，不建未验证跨-run 学习）；结构化路由（capability_registry 去字符串版）；**tracing/observability（正交、低风险，正是 P0-5 性能 harness 所需）**。
- **永远自建 / 不可借**：evidence store / NumericClaim / deterministic validators / fail-closed gates / 单调 authority / StepAuthorityCapsule——通用框架都没有，是护城河。
- **不整体迁 LangGraph 等通用运行时**：checkpoint/resume/evidence-authority 语义本身就是领域逻辑，通用 checkpointer 不强制这些不变量。

---

## 0 · 前置（一切之前，硬约束）

- [ ] **P0-1 基线提交** 🟢：单独一个 baseline commit，**只含** `task_logs/20260715_agent_freeze_refactor_safety_plan.md` + 五个 `tests/research_agent/test_char_*.py` + `fixtures/char_golden_run_bundle.json`；**不含** `benchmarks/extension3/`、`test_extension3_benchmark_spec.py`、任何生产源码。先跑绿这批 char 测试再提交。所有分支从此 commit 拉。
- [ ] **P0-2 错峰** 🟢：动源码前 `git status` 确认目标文件不在并行会话 dirty 列表。
- [ ] **P0-3 分账** 🟢：Track A 源文件（`repair_reasons.py`/`provider_budget.py`/`code_preflight.py`/`agents.py`/`pipeline_execute.py`）请另一会话这轮别动。
- [ ] **P0-4 回归政策** 🟡（rev.2 修正，不再是「永不跑全套」）：
  - 每个小 commit：focused tests；
  - **每个 Phase 合并前**：`tests/research_agent` **按模块串行分片**运行，`OMP/OPENBLAS/MKL_NUM_THREADS=1`；
  - **meta / characterization / resume / evidence-authority 每批必跑**；
  - 不得因 16GB 就永久跳过整合回归。
- [ ] **P0-5 性能验收** 🟡（rev.2 新增，**最重要**——否则只拆漂亮不变快）：
  - [ ] **P0-5a 先建 A/B 测量 harness**：对同一 E3 run 记录旧流程的 provider calls / input tokens / wall time / 重复执行次数 / resume 准备耗时，作为对照基线（没有基线就无法验收「下降」）。
  - **硬指标（每条单独验收）**：
    - clean step 总 LLM calls **≤3**；
    - 一次 repair **≤4**；patch 失败回退 rewrite **≤5**；
    - 相同 code/context/input/validator SHA → **0 次重复执行、0 次重复 concept audit**（= StepAuthorityCapsule 命中，A2-批2 的验收）；
    - resume 到当前步骤的本地准备 **<10 秒**；
    - initial Coder 输入 **≤12k tokens**，repair 输入 **≤8k tokens**；
    - E3 Step01 **不重新执行**；
    - **与旧流程对照**：provider calls / input tokens / wall time 在**有浪费的路径**（repair/resume/重复执行审计）上 **≥50%↓**（clean step 本就接近 ≤3，不强求同幅）。

---

## Track A — 执行/修复控制面（先完成并冻结）

### A1 · Phase 1：失败分类 + 原子修复事务（含 Codex 前几轮四项收口）

- [ ] **A1-1 AST preflight 生成稳定 reason** 🔴：确定性 AST 前检直接产出 `detail.reason="lossy_numeric_coercion"`；检测两类缺口——①算了 coercion-loss 计数却未在 >0 时 fail-close；②域校验只覆盖 post-coercion 非空值。**不得**从 `issue_code=other`+文案猜。位置 `code_preflight.py`；样本 `analysis.py:45,~340`。验收 T1、T3。
- [ ] **A1-2 精确分类替换一刀切** 🔴：改 `repair_reasons.py:170-171`（现把所有 `llm_concept_auditor` finding 一律判 `SCIENTIFIC_SEMANTICS_VIOLATION`）为按 AST reason / 严格 schema `issue_code` 枚举精确命中；旧 quarantine `issue_code=other` 由新 AST 重识别。验收 T1、T2。
- [ ] **A1-3 引导 coder fail-close** 🟡：用既有 `strict_numeric_input` 或加 `if newly_invalid>0: raise`。验收 T4。
- [ ] **A1-4 RepairTransaction 状态机（分支，rev.2 修正）** 🟡：绑定 `step_id / attempt_id / repair-ticket SHA / before-code SHA / typed-input+context SHA / provider-history SHA / engine+validator SHA / final-audit token`。状态图为**分支**：
  ```text
  patch_pending ───────────────→ audit_pending → completed
        └→ fallback_pending ───→ audit_pending → completed
  pending 状态（patch/fallback/audit）验证完整后可**恢复**，不自动 fail-close；
  只有篡改 / 无法验证 / 终态失败才 → failed_closed。（completed 之后不得再进 failed_closed）
  ```
  位置 `provider_budget.py` + repair 段。验收 T5、T13、T14。
- [ ] **A1-5 原子预算 + 保护 final audit** 🟡：开事务原子预留整笔（patch+可选 rewrite+**强制 final audit**）；**当只剩一个非审计额度、且 final-audit 额度已单独保留时**才直接 full rewrite（不浪费在 patch，也不动 audit 额度）；patch 成功后未用 rewrite 容量安全释放，**final-audit 容量始终不可侵占**。验收 T5、T6、T7、T11、T12。
- [ ] **A1-6 单一总账** 🟡：repair 与 audit 可分账，并入一个 durable aggregate ledger。
- [ ] **A1-7 receipt 单调 + 版本化 continuation** 🔴：receipt 永不删/不因 engine identity 重置；仅一次结构绑定、版本标记 continuation；E3 保持 logical attempts=3、禁止 attempt 4。验收 T8、T13。
- [ ] **A1-8 真脏数据终态** 🔴：真实域外值 → `failed_closed_data_invalid`，**无 current evidence/alias**；limitation 可登记但**不获 current authority**。验收 T9、T10。
- [ ] **A1-9 测试矩阵 T1–T14 先写后实现** 🟡（见方案 rev.3 §4）。
  - **验收（Codex 判）**：E3 Step 02 finding 由 **AST** 判 `LOSSY_NUMERIC_COERCION`，coder 加 guard 后**通过**——不是猜、不是改判 limitation、不是放行、不是删 receipt。

### A2 · Phase 2：控制面分解（四批，含 StepAuthorityCapsule）

> 不是一次造八个类；按下列**四批**逐批平移，每批 `char_*` golden 锁**规范化后行为等价** + 该职责独立单测。

- [ ] **A2-批1 `RepairCoordinator` + `GateEvaluator`** 🟡：承接 A1 的事务/分类；gate 一次返回全部 finding，绑定 attempt/checkpoint/artifact digest。
- [ ] **A2-批2 `StepAuthorityCapsule` + `CheckpointAuthority`** 🟡（rev.2 新增 Capsule，**这才解 resume/context 耗时**）：Capsule 打包并 SHA 绑定——`scoped context / typed bindings+SHA / Planner 科学规格 / product contracts / code+ticket SHA / validator+prompt+engine fingerprints / audit cache identity`；**上游 SHA 未变时直接加载 Capsule，不重建 context/plan**。
  - **验收**：相同 digest 0 重复执行/审计（P0-5）；resume 本地准备 <10s；E3 Step01 不重跑。
- [ ] **A2-批3 `EvidenceRegistrar`** 🔴：validate → seal → register current；失败尝试只进历史、不获 current authority（不碰 validator 条件，只搬边界）。
- [ ] **A2-批4 `StepExecutor` + `RunCoordinator`** 🟡：StepExecutor 只在 sandbox 执行已锁定分析；RunCoordinator 只推进状态，不含任何统计/临床规则。`ExecutionState` 值对象收拢原 ~40 个共享闭包局部。`PlanAuthority`/`TypedBindingResolver` 先识别边界划线。

---

## Track B — agent 设计/方法层（Track A 冻结后才改源码；期间只读核验）

### Track A 进行期间：**只允许只读核验**（不改源码）
- [ ] **B1-1v** 🔵 核实 `methods/survival.py:132` numpy Cox 是否被 agent 生成/replay 脚本**动态 import**（静态 grep 不够）。
- [ ] **B4-1** 🔵 核实 `skills.py`(915) / deterministic runners / planner 三处是否重叠表达「跑什么分析」。
- [ ] **B5-1** 🔵 核实内部 agent 步是否复用 `mcp_server.py` 的 tool 抽象 vs 私有平行路径。
- [ ] **B6-1** 🔵 核实 `icu_rules.py:126-422` `valid_range`（生理合理域）与 dict min/max 是否**同一个量**（可能本就不同，非漂移）。

### Track A 冻结后：源码修改
- [ ] **B1 方法去重**（字节/SHA 一致口径）：B1-1 删/替 numpy Cox → statsmodels PHReg（**仅当确认无脚本依赖**）；B1-2 IRLS×4 → 一 helper；B1-3 KM×3 → `_km_step`；B1-4 🟢 保留项加「为何不用包」注释 + 改 `missing_data.py:17` 假措辞。**若合并会改数字 → 不算去重，须另立 method-version（容差验收 + fresh 重跑 canonical）**。
- [ ] **B2 memory**（决策已定）：保留 Step/Run/resume 运行记忆（= StepAuthorityCapsule 线）；**StrategyCard 跨-run 自学习暂不接线**（`validate/retire/record_retrieval` 保持未接，或删死字段+bonus 并诚实标注）；**ExperienceBank 继续关闭**。
- [ ] **B3 并行跨库复制** 🟡（**必须在 A2 状态隔离之后**，rev.2 降级）：`pipeline.py:4080` 串行→有界并发；先确认 LLM/cost meter/repro envelope/memory 无竞态（A2 隔离后才成立），`max_concurrency≈2`。
- [ ] **B6-2**（**依赖 B6-1**）🟢：若 min/max 语义可比，扩上游 `get_concept_info`(api.py:2015) additive 补 min/max/levels；B6-3 若需单源 → `context.py:581-587` 优先 dict + 漂移测试。
- [ ] **B7 过拟合归位**（决策已定）：**六 runner 不得接成 primary**；只保留「接受 Agent 已锁定参数的通用计算器」，**会选择暴露/结局/队列/协变量/方法的 runner 删除或归档**（连带 `deterministic_causal.py:239` 案例协变量、`deterministic_clustering.py:116` SOFA-2 正则）；`pipeline.py:10299-10365` 标签外置 `display_label`；**B7-4 删死 shim/死门延后**（对速度无益，不抢主线）。
- [ ] **B8 hooks seam** 🟡（**A2 之后**）：`_execute_one_step` 外围引入 empty-default `StepMiddleware`。

---

## 决策（五项，已定）

| 问题 | 决定 |
|---|---|
| **Memory loop** | 保留运行/resume 记忆（Capsule）；生产环境删除/禁用未验证的跨-run 自学习。StrategyCard 学习循环暂不接线，ExperienceBank 继续关闭。 |
| **六个 runner** | 不得接成 primary runner。只保留「接受 Agent 已锁定参数」的通用计算器；会选择暴露/结局/队列/协变量/方法的 runner 删除/归档。 |
| **A2 批次** | RepairCoordinator+GateEvaluator → StepAuthorityCapsule+CheckpointAuthority → EvidenceRegistrar → StepExecutor+RunCoordinator。非一次八类。 |
| **数值等价** | 行为保持/去重必须**字节或 SHA 一致**；更换算法/第三方包只能作为**单独 method-version 变更**，容差验收并 fresh 重跑 canonical。 |
| **A/B 顺序** | Track A 全部先完成、验收、合并、冻结；Track B 期间只读核验；B7/B8 等源码修改必须 A2 之后。 |

---

## 最终执行顺序（9 步）

1. **P0** characterization baseline commit（+ 建 P0-5a 性能对照 harness）。
2. **A1 T1–T14 测试先行**。
3. **A1** RepairTransaction + coercion preflight。
4. **A2 批1**：RepairCoordinator + GateEvaluator。
5. **A2 批2**：StepAuthorityCapsule + CheckpointAuthority + resume 快路径。
6. **A2 批3/批4**：EvidenceRegistrar，然后 StepExecutor + RunCoordinator 边界。
7. **完整分片回归**（P0-4）**+ 性能对照**（P0-5 全部硬指标）。
8. 同一 E3 run **只 resume Step02**（验证端到端解锁）。
9. **Track A 冻结后**再进入 Track B（B1/B2/B3/B6-2/B7/B8）。

---

## 验收总则

- 行为**必须不变**处（科学产物 SHA、evidence 权威、numeric claim、receipt 单调）→ golden/replay 字节/SHA 锁死。
- 行为**应当改变**处（事务原子性、finding 精确分类、脏数据 fail-close）→ 明标「行为修复」，验收=正确终态 + **性能硬指标达标**，而非「与旧输出一致」。
- **性能门**：无 P0-5 达标不得宣布 Phase 完成（防「只拆漂亮、速度没变」）。
- 三条红线：不得把行为修复伪装成结构迁移；不得把结构迁移偷改行为；不得把真实代码缺陷洗成 limitation。
