# research_agent · 行为保持的 vNext 结构提取 · 安全计划 · 2026-07-15

> **状态说明（2026-07-18）**：§1–5 的规范化等价、单调 authority、职责边界与 characterization 要求仍是当前安全规范；本文件中“现在不拆、先跑实验”的执行顺序已被用户 2026-07-17 的 **先完成有边界架构整理、再 fresh 跑 E3/H2/E2** 决定取代。实时状态看 `项目进度/agent/CURRENT.md`，组件台账看 `task_logs/20260717_remaining_debt_register.md`。

> **这是一份「行为保持的 vNext 结构提取计划」——不是 canonical 引擎重设计计划,也不是 LangGraph 迁移计划。**
>
> 由 Claude↔Codex 2026-07-15 架构讨论收敛而来。它是叠在打包/目录拆分计划
> `task_logs/20260714_agent_structural_debt_split_plan.md` **之上的语义/安全层**:
> 那份说「哪个文件进哪个子包、怎么安全 git-mv、façade 兼容、import-graph 验收」;
> 本份说「拆动时行为必须以什么口径保持不变、为什么连移动函数都要当引擎身份变更、
> 巨型执行函数应按什么职责边界分解、freeze 生命周期怎么排、框架借到哪为止」。
> 两份互补,都不在实验进行中执行。

---

## 0. 一句话定位

近期先把「每步多次长链循环」压缩成一次聚合执行(已在做,见今晚硬化日志);
实验 freeze 后再把重型控制面按职责边界**平移**成清晰组件;最后才考虑让框架接管
*正交的* tracing。EasyICU 自己保留的核心是 **ICU 科学所有权 + 可发表证据链**,
不是通用任务调度代码。

目标改写为(取代早前「以迁移框架为目标」的措辞):

> **不以「迁移框架」为目标,而以「降低集中度、锁住行为、保持科学所有权」为目标。**

---

## 1. 行为保持的验收口径 = **规范化后等价**,不是逐字节一致

拆分/提取前后,拿同一冻结输入跑同一 run,比较的是**规范化后等价**——
时间戳、run ID、绝对路径、wall-clock、进程内存地址等天然会变,不纳入比较。

**必须锁死(语义等价,任一漂移即视为回归):**

1. step 状态与依赖传播(ok / failed / quarantined 及其向下游的传播集合);
2. typed input bindings 及其 SHA;
3. current evidence / alias authority(谁是当前权威、别名解析结果);
4. numeric claims(值—证据绑定、authoritative_numeric_claims 集合);
5. validator findings(条目、reason code、severity,逐条绑定 attempt/checkpoint/artifact digest);
6. seal / register 顺序(draft → validate → seal → register current 的次序与幂等性);
7. publication / readiness 结果(STRICT 门的通过/拒绝判定与理由);
8. 稳定产物 SHA:表、图源数据、模型输出。

**比较方式:**
- **确定性产物**(表 / 图源数据 / 模型输出 parquet 等):要求**字节一致**(SHA 相等)。
- **含运行元数据的文件**(manifest、step record 等):**去除允许变化字段**
  (时间戳 / run_id / 绝对路径 / 耗时 / 进程信息)后再比较结构等价。
- 建立一份「易变字段白名单」并纳入比较脚本,任何不在白名单里的差异都是回归。

> 这条纠正了早前「逐字节一致」的过严表述——那会把无害的运行元数据变化误报成行为漂移。

---

## 2. 「纯谓词/纯函数抽取零风险」并不成立 —— 它会改 **engine identity**

EasyICU 已把 **validator / engine implementation SHA** 用于 audit cache 键、resume
身份和 revalidation(复核已确认 `environment_sha256` 绑定 engine+validator+prompt+llm,
`concept_audit_cache` 以它为 key)。因此**即使只移动一个函数、不改任何逻辑,也会改变
引擎实现 SHA**,可能使旧 checkpoint 需要重新验证、audit cache 失效。

结论:纯谓词抽取**可以**在 freeze 前做(收益:缩小巨型函数 + 买到可单测覆盖),但只能:

- 在**没有任何 run 活跃**时进行(不得一边 resume H2 一边抽取);
- 明确**当作 engine-identity 变更**记录,不假装「零变化」;
- 抽取后跑完 **resume / revalidation 回归**,确认旧 checkpoint 的重验行为符合预期;
- 与打包拆分同样受 `agent/CURRENT.md` 「⚠️不要做:bench/实验运行期间编辑源文件」约束。

---

## 3. 目标职责边界(是「边界」,不是「立刻造 8 个大类」)

把现在隐含在巨型 `pipeline_execute.py` / `pipeline.py` 里的状态、职责和权威边界**显式化**。
每个边界职责要窄;科学决策始终留在 Agent(Planner/Scientist),确定性层只执行/校验/封存/恢复。

| 组件 | 职责 | 明确不做 |
|---|---|---|
| **PlanAuthority** | 对 Planner/Replanner 返回的 candidate plan 应用 completed-step immutable snapshot、max-step cap、estimand/figure 保留、锁定规格投影与科学签名/no-op 判定 | 不调用 provider、不注册 revision/evidence、不修改 cohort/runner/budget，也不替 Planner 选择暴露、结局、方法、队列或 estimand |
| **TypedBindingResolver** | typed product → 精确 evidence/path/SHA、权威列与 companion metadata、row identity/alignment、consumption receipt | 不选科学设计 |
| **StepExecutor** | 在 sandbox 执行**已锁定**的分析 | 不决定模型/暴露/结局/队列 |
| **RepairCoordinator** | 聚合 RepairTicket、管理 patch/预算/quarantine | 不按错误文案关键词路由科学方法 |
| **GateEvaluator** | deterministic / statistical / clinical / concept gates,一次返回全部 finding,绑定 attempt/checkpoint/artifact digest | — |
| **EvidenceRegistrar** | validate → seal → register current;失败尝试只进历史,不获 current authority | — |
| **CheckpointAuthority** | run identity、current successful checkpoint、validator drift 与选择性重审;最新 checkpoint 损坏时 fail-close,不回退旧 authority | — |
| **RunCoordinator** | 只推进上述状态 | 不含任何统计/临床/科学规则 |

> 这些是**职责边界**,不代表要立刻创造八个巨类。第一刀先把这些职责从执行函数里
> *识别并划线*,配合 `20260714` 打包计划的文件归箱逐步落位。
> 早前遗漏的 **PlanAuthority** 与 **TypedBindingResolver** 是 load-bearing 的,必须显式命名归属。

---

## 4. freeze 生命周期六阶段

1. **Freeze 前(现在可做)**:金标准 / characterization 测试;纯谓词抽取(受 §2 约束);
   调用数 / 耗时 tracing。**不做任何改行为的拆分。**
2. **Freeze 点**:锁定唯一 commit、模型、prompt、数据、rubric、retry policy;归档
   collection / API / import-graph 基线(见 `20260714` §D)。
3. **Canonical 实验**:用冻结引擎产出 canonical / held-out 结果;**引擎不再被反向优化**。
4. **Freeze 后**:从冻结 commit 开**独立 vNext 分支**,按模块逐次平移(打包见 `20260714` §A–C 顺序)。
5. **每次提取的验收**:§1 规范化输出等价 **且** authority graph 等价 **且**
   fail-open/fail-close 反例全绿(尤其 `FigureSourceDataValidator`、resume-authority、
   provider-budget、evidence 门)。任一不满足即回滚该批次。
6. **canonical 结果绑定**:原 canonical 结果**继续绑定旧冻结 commit**;只有确认**真实 bug**
   时才升级 protocol version 并作废受影响结果,不得因结构迁移悄悄改动已发布数值。

---

## 5. 外部框架:只借正交的 tracing,不借运行时

- 收回早前「未来优先考虑迁 LangGraph」的倾向。对 fail-closed 证据引擎,通用运行时的
  「省事」部分是幻觉:checkpoint / resume / evidence authority 的语义**本身就是领域逻辑**
  (单调 authority、SHA 绑定身份、evidence closure),通用 checkpointer 不强制这些,
  迁过去要么丢不变量要么照样自己包一层。
- **当前最值得借、且低风险高 ROI 的,只有正交的 tracing / observability**(step 级追踪、
  调用/耗时/预算可视化)——它与控制流正交,可单独接入,不碰证据链。
- checkpoint、resume、evidence authority、typed products、validators、科学契约**全部保留自建**。
- 不同时引入多个 Agent 框架;若真要评估状态图,先做**非主线 prototype** 对比现有运行时,
  绝不直接改 canonical 引擎。抽取后的自建状态机若已稳定,可以**不迁**——换框架本身不是成功指标。

---

## 6. 历史执行顺序（已被 2026-07-17 arch-first 决定取代）

> 本节仅保留 2026-07-15 当时的 freeze 决策，**不是当前任务队列**。
> 当前顺序为：完成有边界的 Track B-Core → freeze → fresh E3/H2/E2 → 3–6 held-out。

- **（历史动作）当时不执行 §3–§4 的任何源码拆分。** 当时第一优先级是**跑完实验**
  (H2 Step 07 resume → E2/E3/H3),重构推迟到 freeze 之后。
- freeze 前唯一允许的架构性动作:写本类计划文档、加金标准测试、(实验空闲窗口内、
  受 §2 约束的)纯谓词抽取、加 tracing。
- 允许启动 §4 阶段 4 的证据:E2/E3/H2/H3 CURRENT 均明确收口、无 benchmark/实验进程、
  工作树清洁、完整 research-agent suite 绿、collection/API/import-graph 基线已归档。
- 若拆分需要改变 validator 条件/严重度、runner 路由、prompt 或 evidence authority,
  **立即停止**并另开逻辑修复任务——不得把行为变化伪装成结构迁移。

---

## 附录 A · Freeze 前 characterization 测试清单(evidence-authority / resume 重审路径)

> 目的:把**当前 HEAD 的可观测行为**钉成快照,作为 §1 规范化等价的**验收底座**——freeze 后每次结构提取都拿它 diff。这是加测试,**不改引擎源码、不改 engine SHA、不碰任何 in-flight run**。
> 这些测试**必须在当前 HEAD 全绿**(它们描述现状);若某条现在就红,说明发现了现状与预期不符——**surface 出来,不得改引擎让它变绿**。
> 优先钉这条路径,因为它既是今晚硬化的重点(immutable run identity / current evidence authority / 选择性重审 / 单调 authority),也是复核里出过 resume-authority 回归的地方——最高价值。

**建议放新文件**(命名可识别为 freeze 基线,交叉引用本文档),别混进现有测试:
`tests/research_agent/test_char_evidence_authority.py`、`test_char_resume_revalidation.py`、`test_char_runartifact_authority.py`、`test_char_audit_cache_identity.py`、`test_char_golden_run_bundle.py`。

**G1 · evidence 当前权威与 provenance**(`evidence.py`、`manuscript_post.py`)
- `current_verified_records(per_step_records)`:同一步先成功后失败时,失败步记录**不进**当前权威;latest successful producer 胜出。
- `authoritative_numeric_claims`:跨 ok/failed 步的 claim,只有 ok 步存活;evidence-id-scoped 去重结果锁定。
- `bind_manuscript`:STRICT 下未解析 `{evidence:}` 抛 `EvidenceEnforcementError`(非 STRICT 留 `[evidence missing]`)——两种模式各钉一例。
- `enforce_evidence_bound_scaffold`:bullet 行内 result-like 且无 `{evidence:}` 在 STRICT 下被删/抛;**无数字定性方向结论的 bullet 现在会被抓**(锁 #4 修复后的行为)。
- alias 权威解析:publish_aliases → 当前 alias → record 的解析结果锁定。

**G2 · run-artifact 单调 authority**(`runtime_artifacts.py`)
- `load_run_artifact_authority`:newest-sequence 胜出;newest 损坏 / 无 ledger 时 **fail-close**(不回退旧 authority);旧 success 不能覆盖新 failure。用带 sequence 的 fixture ledger 构造。
- `active_step_evidence_ids_by_step`、`run_level_evidence_matches_claim_owner`:对 fixture 的映射输出锁定。

**G3 · resume 重审门**(`run_input_capsule.py`)—— *本任务核心*
- `_host_probe_authority_error`:合规 probe 记录通过;各类畸形各自返回**精确 reason 串**。
- `_host_cohort_materializer_authority_error`(锁 #1 修复):合规 materializer checkpoint 通过;逐个违约(错 evidence_id / 多列 evidence_ids / 缺 receipt / cohort 计数非法 / authority 缺失或不匹配 / 带 script 或 inputs / SHA 不符 / 行数不符)各自返回精确 reason。
- 通用步:缺 `step_summary_evidence_id` / `script_evidence_id` → invalidated。
- `_migrated_legacy_step_authority`:只迁闭合 legacy 链,绝不伪造。
- 下游定点传播:invalidate 上游步 → 其 consumer 全部失效。
- `resume_from_step_id` 排在已失效上游之后 → 抛 `RunInputIdentityError`。

**G4 · 环境绑定 audit cache**(`concept_audit_cache.py`)
- cache key 绑定 `environment_sha256`(engine+validator+prompt+llm)+ auditor_identity + per run_dir;换 environment_sha → cache miss(无跨环境污染)。**这条同时把「移动代码为何会失效旧 checkpoint」钉成可执行文档。**

**G5 · golden 规范化等价快照**(§1 预言机)
- 用现有最小 end-to-end pipeline fixture 跑一次,抓取归一化 §1 bundle:{step 状态 + 依赖传播集、typed input bindings + SHA、当前 evidence/alias 权威映射、authoritative numeric claims、validator findings(reason+severity)、seal/register 顺序、publication/readiness 判定、确定性产物 SHA}。
- 归一化时剥离易变字段(timestamp / run_id / 绝对路径 / 耗时 / pid),存成 golden JSON,断言相等。**这份 golden 就是 freeze 后重构的验收预言机。**

**本任务验收**
1. 上述新测试在当前 HEAD **全绿**;运行时确认只跑这几个新文件(内存紧,勿全套 pytest,见「full suite OOM — split it」)。
2. 结束时 `git diff --name-only` 应**只有测试文件 + 本文档**,`src/easyicu/research_agent/**` 零改动。
3. 回报:新增文件、测试数、pass/fail、耗时;若发现任何现状 discrepancy,列出但**不改引擎**。

### 附录 A 执行记录 · 2026-07-15 · `main@de4af7f`

- 新增 5 个独立 characterization 文件和 1 份 golden fixture，覆盖 G1–G5；没有编辑
  `src/easyicu/research_agent/**`，因此 engine / validator identity 未改变。
- 聚焦验收命令只运行上述 5 个新文件：`47 passed in 1.96s`（`/usr/bin/time`：
  real 2.53s / user 2.72s / sys 0.96s）。新增文件 Ruff 全绿，Black check 全绿；
  未运行完整 pytest。
- G3 精确锁定 host probe 与 host cohort materializer 的合规路径及 27 个畸形/reason
  组合，并锁定 generic authority、legacy migration、A→B→C 定点失效传播和 late
  resume-cut 写前阻断。
- G1 的 numeric-claim 用例同时锁定：同一 evidence 的重复 claim 会折叠，但不同
  evidence 即使 field/value 完全相同也必须分别保留，避免重构时误变成全局数值去重。
- G5 使用既有最小 4 步 typed-product pipeline（24 行、本地 controlled LLM/runner、
  无网络/LaTeX/visual QA/concept-auditor）生成
  `tests/research_agent/fixtures/char_golden_run_bundle.json`。快照锁定 5 个当前 step
  状态（含 host probe）、14 条 declared dependency edge、typed binding identity/product
  contract/SHA、29 条 current evidence、60 个 current alias、28 条 authoritative numeric
  claim、17 条 normalized finding、11 个确定性表 SHA、readiness fail-close 结果及每步
  完整的 authority event 序列。事件以 run-length 形式保存顺序和每类调用次数，不会再
  折叠重复 seal/register/final-gate/numeric-registration 调用。

**按现状记录的 discrepancy / nuance（未改引擎）：**

1. 附录原句“通用步缺 `script_evidence_id` → invalidated”并非无条件成立：若 legacy
   summary→script 链唯一、闭合且 digest-valid，当前 HEAD 会追加迁移后的 `ok` checkpoint；
   只有链不闭合或同一步存在多个 code authority、无法唯一证明时才以
   `successful checkpoint is missing required script_evidence_id` 失效。测试同时锁住两条。
2. 单独一个合法 pre-ledger manifest 当前返回 legacy `None`；“无 ledger fail-close”只适用于
   新 ledger-less boundary 试图回放更旧 modern ledger 的情形。
3. `run_dir` 不进入 concept-audit key digest；per-run 隔离来自每个 run 自己的
   `.cache/llm_concept_audit.json`。同参数 key 相同，但不同 run 目录不会自然共享文件。
4. 当前 `ValidationFinding` 没有统一 `reason_code`，golden 只能按
   `detail.reason → detail.issue_code → detail.issue → message` 归一化；finding 也没有普遍
   直接绑定 artifact/result-seal digest，只能按 owner step 关联 `result_seal_sha256`。
5. `audit_log.jsonl` 当前不记录 seal/register 时序事件；G5 因而用只观察、不替换行为的
   runtime instrumentation 锁定现行顺序。完成后的静态 artifact 本身不能独立证明时序。

## 关联文档
- 打包/目录拆分(文件归箱、git-mv、façade、import-graph 验收):`task_logs/20260714_agent_structural_debt_split_plan.md`
- 今晚 resume-authority / provider-budget 硬化 + Claude 复核 + Codex 4 项修复:`task_logs/20260715_agent_resume_authority_provider_budget_hardening.md`
- H2 Step 06 收口:`task_logs/20260714_h2_step06_framework_iteration.md`
- 四层架构:`src/easyicu/research_agent/README.md`
