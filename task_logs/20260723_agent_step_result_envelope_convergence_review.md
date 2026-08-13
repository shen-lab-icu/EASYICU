# Research Agent `StepResultEnvelope` 收敛审查

> 日期：2026-07-23 EDT  
> 分支：`refactor/agent-control-plane`  
> 审查范围：`c58df96..186d53c`  
> 审查时 HEAD：`599b046e19eb4f8562f772f2acf602a694f405af`  
> 任务：暂停新增单点 repair / 提高 repair cap / 新 formal authority，独立审查近期增长并设计唯一结果规范化边界。

## 1. 独立核对后的当前真相

- 工作树在开始审查时 clean，当前分支相对远端 `ahead 1`。
- 没有 `run_discovery_to_manuscript`、Canonical9 或 E2 实验进程。
- E1 development canary：
  - 路径：`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_dev_ad4d826_canary/e1_sepsis3_prevalence_mortality/aware/run_20260723T211020_5733af`
  - producer commit：`748b3524ac10ffeb7f7ecbb29e980671a074acde`
  - 7/7 required steps complete；
  - `manuscript_ready=true`；
  - numeric / evidence / analysis error 均为 0；
  - locked scorer `tristate=gate_reportable`；
  - 仍为 `forced_diagnostic_only=true`、`paper_authorized=false`、`publication_ready=false`，不能晋升为论文 authority。
- E2 development run：
  - 路径：`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_dev_748b352_remaining8/e2_lactate_mortality/aware/run_20260723T235937_f4d63c`
  - 最近记录的 producer commit：`f496da6f185b4a6d58f22fc38591f7a635f8fe90`
  - 6/9 required steps complete；
  - `06_complete_case_robustness` 为 `blocked_by_concept_audit`；
  - 其 figure child 为 `skipped_dependency_failed`；
  - `manuscript_ready=false`，保持 diagnostic-only。
- E3 路径只含 pre-plan / blueprint / input authority 文件，没有 execution manifest；不得记为已执行。
- 因用户要求暂停，最新 E2 development 进程已终止；本审查没有启动新 Provider、Docker、患者数据或 authority。

### 与交接数字的差异

本地 exact refs 的结果是：

```text
git rev-list --count c58df96..186d53c
43

git diff --shortstat c58df96..186d53c
117 files changed, 11070 insertions(+), 2366 deletions(-)
```

交接中的 45 commits / 119 files / `+11317` 很可能来自稍晚或不同 base 的快照。本报告使用可复现的 exact local refs，不为差异倒推口径。

## 2. 增长分类

以下是 `src/easyicu/research_agent` 与 `tests/research_agent` 的独立 numstat 分类。一个提交可能横跨多类，表按文件最终归属统计，不把测试行数算作生产逻辑。

| 类别 | 新增 | 删除 | 净增 | 判断 |
|---|---:|---:|---:|---|
| tests | 4,892 | 874 | +4,018 | 大部分是有价值的负向/回归资产，应保留 |
| repairs | 2,425 | 50 | +2,375 | 主要膨胀源，必须收敛 |
| execution / reporting / evaluation / figures | 1,031 | 158 | +873 | 多个消费者仍自行解释任意文件 |
| other production | 927 | 328 | +599 | 含 schema、pipeline、routing、runtime |
| scientific validation | 1,116 | 829 | +287 | 有重写迁移；科学 fail-close 不应随格式 repair 删除 |
| evidence / authority / contracts | 247 | 10 | +237 | 大多应保留，但必须绑定唯一 envelope version |

`repairs/` 本范围最大增长：

| 文件 | `c58df96..186d53c` 增量 | 当前 LOC | 主导问题 |
|---|---:|---:|---|
| `repairs/source.py` | `+730/-43` | 7,527 | 多种代码形状、JSON 转换、figure/source/statistic 特例混居 |
| `repairs/typed_input.py` | `+360/-5` | 635 | 输入 authority 与输出格式修复边界混杂 |
| `repairs/provenance_summary.py` | `+352/-2` | 2,034 | generated-code provenance 适配应迁到 host binder |
| `repairs/nullable_validation.py` | `+226` | 226 | nullable 表达差异 |
| `repairs/percentage_identity.py` | `+212` | 212 | 通过字段命名猜 numerator/denominator/percentage |
| `repairs/rendering_role.py` | `+193` | 193 | hard-code robustness/complete-case 行角色 |
| `repairs/rendering_summary.py` | `+173` | 173 | render-only summary 投影 |
| `repairs/nonfinite_audit.py` | `+170` | 603 | 同时包含科学非有限值拒绝与表示层修复 |

### 43 个提交的主导分类

这里只标主导目的，用于判断迁移归属，不表示提交只能属于一类。

- 序列化 / 格式规范化（16）：
  `3a9442f`, `f87f10b`, `92770db`, `b9ad840`, `a09d6df`,
  `9e1d4f0`, `8e25560`, `e856f86`, `1c0fb4a`, `c987dfd`,
  `2694c23`, `e851359`, `2c2f508`, `be866f1`, `75ec0ad`,
  `a841ba0`。
- 证据绑定 / authority（14）：
  `6782345`, `a9c5404`, `e5318c1`, `7cb99dd`, `8dafe4a`,
  `798ec2b`, `ab86631`, `60e9a71`, `8fa79ee`, `717316d`,
  `637b394`, `9025bc5`, `748b352`, `5a27e8a`。
- 科学语义 / 科学 fail-close（8）：
  `ac6b2a4`, `efb6305`, `c20f91c`, `559dc3c`, `b58098d`,
  `ad4d826`, `28811ee`, `186d53c`。
- 题目形状或运行适配（5）：
  `c6f80e3`, `054106c`, `ece2638`, `f1efaa6`, `ac6e065`。

## 3. Findings

### P1 — 当前没有可强制执行的唯一结果合同

`schema.StepRecord` 明确说明结构“organically”增长，使用
`extra="allow"`，而 `step_summary` 仍是 `Optional[Dict[str, Any]]`。这不是
`StepResultEnvelope`，只是允许任意 payload 继续流动的外壳。

静态盘点有 **76 个生产模块**直接接触 `step_summary`。Validator、Writer、
readiness、scorecard 和 figure 层仍可分别解释字典、CSV、JSON 与文件名。
因此同一科学结果没有一个唯一、可版本化、可校验的内存表示。

影响：

1. 格式等价性由多个消费者重复实现；
2. 新任务出现新表达时，最容易的局部修法仍是再加 repair；
3. 不同下游可能对同一 artifact 得出不同语义；
4. 不能证明 E1/E2 的行为等价迁移只改变表示、不改变数值或科学结论。

### P1 — 已有 normalizer 不是唯一边界，也不是最后写入者

`execution/output_files.py::normalize_typed_statistic_sidecars` 是正确方向，
但当前只处理 exact `statistic:<name>` JSON identity，并且直接改写生成文件。
随后执行链仍有其他路径重新读取并写回 `step_summary.json`；例如 detached
figure binding 使用 `json.dumps(..., default=str)`。这会把未知对象静默变成
字符串，而且发生在早期 statistic normalizer 之后。

同时：

- Writer 仍直接 `pd.read_csv` 并猜列名；
- readiness 仍有 filesystem fallback，遍历 `steps/*/outputs/step_summary.json`；
- scorecard 仍有无 manifest 时的 raw summary glob；
- validators 中存在多处独立 CSV/JSON 解析。

所以当前的 “host output normalizer” 不能保证所有下游消费相同值。

### P1 — 通用输出绑定中存在结果/题目特例

`execution/output_files.py` 的通用路径硬编码：

- `statistic:primary_or`
- `declared_name == "primary_or"`
- `table:adjusted_association_estimates`
- odds-ratio 的正数与区间约束

这些约束本身可能科学合理，但应来自 Planner-owned typed product contract
中的 `effect_scale` / `estimand` / interval contract，而不是写死在通用 I/O
helper 中。否则后续 risk ratio、hazard ratio、coefficient、difference、
calibration 或 trajectory statistic 会再次要求新 helper。

`repairs/rendering_role.py` 又固定读取
`robustness_summary["restriction"]`，并内置 `primary/full/none/all` 与
`complete_case` 标签集合；`repairs/percentage_identity.py` 通过
`pct/n_total/total_n/event_n` 等字段名猜角色。这两类都是 typed envelope
缺失后的症状。

### P1 — E1 artifact 已出现 current status 与证据副本漂移

E1 根目录最终 `run_status.json`：

```text
manuscript_ready=true
analysis_error_count=0
evidence_error_count=0
numeric_error_count=0
```

但 `evidence/run_status__run_status.json` 仍为：

```text
manuscript_ready=false
analysis_error_count=2
evidence_error_count=0
numeric_error_count=0
```

这没有改变 E1 development scorer 的 `gate_reportable` 事实，但证明
“文件名 + 目录位置”不足以表达 current authority。最终 envelope 必须带：

- envelope version / SHA；
- source artifact SHA；
- current ledger coordinate；
- supersedes / superseded-by；
- exact consumer binding。

任何 Writer/scorer 都不得自己选择根文件、evidence 副本或旧 alias。

### P2 — 格式规范化被计入 repair，错误分类仍驱动预算

host 侧为 sidecar 补 product identity 后调用 `_record_repair`，repair id 为
`typed_output_normalization_v1`。这让无 LLM、无数值改变的 deterministic
canonicalization 与代码修复共享账本语义。

建议拆为四个互斥终态：

1. `normalized`：安全、确定性、零 LLM，不消耗 repair budget；
2. `code_error`：最多 1 次普通 Coder repair；只有明确可修的机械错误才允许第 2 次；
3. `scientific_contract_error`：Replanner 或 fail-close；
4. `authority_error`：直接 fail-close，禁止 Coder repair。

### P2 — NumPy/nullable/path/provenance 修复发生得太晚

`75ec0ad` 通过修改生成代码解决 NumPy scalar JSON；
nullable、host/container path、provenance 与 figure source 也各有独立
AST repair。它们都属于“将原始执行产物编译成 canonical envelope”的职责。

正确做法不是让每个生成脚本学习一套 `to_jsonable`，而是：

- 原始 artifact append-only 保存；
- host 读取允许的 registered product；
- 统一转换 Python/NumPy/Pandas scalar、nullable 与 path；
- 写一份新的 `step_result.envelope.json`；
- 记录 normalization receipt；
- 不改原始数值和原始 artifact bytes。

### P2 — 现有测试证明局部逻辑正确，但没有锁住唯一消费边界

本轮复跑：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/research_agent/test_primary_estimate_output_binding.py \
  tests/research_agent/test_bound_percentage_identity_repair.py \
  tests/research_agent/test_char_golden_run_bundle.py \
  tests/research_agent/test_writer_digest_v2.py \
  tests/research_agent/test_stale_step_summary_authority.py
```

结果：`36 passed`。

这证明现有 bind/repair/stale-authority 逻辑没有被本审查破坏；它不证明
Validator、Writer、readiness 与 scorer 已只消费同一个 canonical object。

## 4. 目标 `StepResultEnvelope`

建议新增版本化、`extra="forbid"` 的 immutable Pydantic schema，并与
`StepRecord` 生命周期信息分开：

```text
StepRecord
├── lifecycle / runner / repair / audit state
└── result_envelope_ref -> StepResultEnvelope

StepResultEnvelope
├── schema_version
├── envelope_id / sha256 / supersedes
├── step_id / planned_analysis_role / product_contract_ref
├── population
│   ├── eligible_n / analyzed_n
│   ├── group counts
│   └── denominator identities
├── variables
│   ├── exposure / outcome / covariates
│   └── typed source-column refs
├── statistics[]
│   ├── statistic_id / estimand / scale / unit
│   ├── value / interval / p_value
│   ├── numerator_ref / denominator_ref
│   └── source_artifact_ref
├── missing_data
│   ├── declared plan
│   ├── executed policy
│   └── before/after N and sensitivity role
├── model_diagnostics
│   ├── convergence status + controlled evidence source
│   └── warnings
├── artifacts[]
│   ├── product id / kind / media type
│   ├── raw path / canonical path
│   ├── sha256 / evidence id
│   └── figure source-data refs and panel roles
├── provenance
│   ├── input authority refs
│   ├── script / image / runner identity
│   └── normalization receipts
└── limitations / warnings
```

关键规则：

- `StepResultEnvelope` 不保存任意自由 JSON；
- 扩展字段必须先升级 schema version，不使用 `extra="allow"`；
- 原始 `step_summary.json` 只作为 untrusted/raw producer artifact；
- canonical envelope 是唯一 downstream contract；
- evidence store 注册 raw artifact 和 canonical envelope 两者，各自 SHA
  独立，不覆盖原文件；
- `StepRecord.step_summary` 在迁移期只读兼容，完成迁移后改为
  `result_envelope_ref`，legacy adapter 仅用于历史 run；
- current envelope 由 ledger coordinate 选择，不由文件名、mtime 或目录顺序选择。

## 5. 目标 `OutputNormalizer`

唯一入口：

```text
normalize_step_result(
    raw_artifacts,
    planner_product_contract,
    resolved_input_manifest,
    execution_identity,
    current_ledger_coordinate,
) -> NormalizationResult
```

严格阶段：

1. **安全发现**：只读取 Planner 声明且 runner 注册的 product；路径 containment、
   symlink、SHA、media type 先验证。
2. **表示规范化**：Python/NumPy/Pandas scalar、nullable boolean/number、
   host/container path、UTF-8、JSON finite number。
3. **typed product parse**：Table、Statistic、Rate、Figure、SourceData、
   ModelDiagnostic 各一个 parser；不根据题目名选择 parser。
4. **identity binding**：statistic name、effect scale、numerator/denominator、
   figure panel role、provenance 从 typed product contract 绑定。
5. **cross-product consistency**：百分比与 count、estimate 与 CI、figure 与
   source data、summary 与 sidecar 一次校验。
6. **evidence registration**：注册 raw artifact、canonical envelope、
   normalization receipt；不改 raw bytes。
7. **emit immutable envelope**：原子写 `step_result.envelope.json`，返回 exact
   envelope ref。

不属于 normalizer 的内容：

- 暴露/结局是否科学合理；
- cohort 是否保留 comparator；
- 是否应插补；
- 模型是否适合 estimand；
- convergence 是否来自受控 optimizer signal；
- authority 是否允许 paper-facing。

这些继续由 scientific contract / authority gate fail-close，不能被格式
normalizer“修好”。

## 6. 准备合并 / 删除的 repair 清单

### 第一批：envelope consumer 切换后整文件删除

| 模块 | 当前 LOC | 替代机制 |
|---|---:|---|
| `repairs/nullable_validation.py` | 226 | typed nullable scalar parser |
| `repairs/percentage_identity.py` | 212 | `RateStatistic` + numerator/denominator refs |
| `repairs/rendering_role.py` | 193 | typed figure panel/artifact role |
| `repairs/rendering_summary.py` | 173 | canonical render-only projection |

可直接退出生产路径合计 **804 LOC**，测试迁移为 normalizer 负向测试后保留。

### 第二批：按职责拆除，不整文件删除

- `repairs/source.py`
  - 删除重复 `to_jsonable` 注入；
  - 删除 statistic sidecar identity 补丁；
  - 删除 figure source-data 形状修补；
  - 删除只为 summary key/path alias 服务的分支；
  - 保留真正改变错误分析代码且跨任务通用的机械 repair。
- `repairs/provenance_summary.py`
  - provenance mapping / receipt shape 迁到 host evidence binder；
  - 保留对伪造来源、患者级泄漏、source SHA 不一致的 fail-close validator。
- `repairs/typed_input.py`
  - 输入 authority 继续独立保留；
  - 与输出共享 scalar/path primitives，但不把 input contract 塞进
    `StepResultEnvelope`。
- `repairs/nonfinite_audit.py`
  - 保留 model input 中非有限值的科学 fail-close；
  - 删除单纯 JSON/nullable/表示层补丁。
- `repairs/serialization.py`
  - JSON-safe scalar/path/null 转入 OutputNormalizer；
  - sklearn optimizer diagnostic 的受控来源验证保留在 model diagnostic gate。

### 明确保留

- cohort / exposure / comparator 科学合同；
- outcome 与 primary estimand authority；
- missing-data plan 及禁止插补 exposure/outcome 的科学门；
- evidence SHA、producer、current-ledger、resume/capsule 绑定；
- nonfinite analysis input、模型收敛来源、未来信息泄漏等科学负向测试；
- Provider/outbound/PHI 边界。

## 7. 净删减目标

保守估计：

- 删除/合并旧生产逻辑：2,300–3,200 LOC；
- 新增 envelope schema、normalizer、typed parsers、legacy adapter：
  800–1,100 LOC；
- **生产代码净删减目标：1,400–2,100 LOC**；
- tests 不以删行为目标，只合并完全重复 fixture；预计保留绝大多数现有
  4,000+ 新测试行。

这不是承诺一次提交完成的数字。每一批必须先证明 E1/E2 envelope 与旧
artifact 在允许字段上行为等价，才能删旧消费者。

## 8. 迁移与回归顺序

### M0 — 冻结开发回放坐标

- E1：上述 `run_20260723T211020_5733af`。
- E2：上述 `run_20260723T235937_f4d63c`。
- 保存 raw artifact SHA、current ledger record、E1 scorer、E2 failed-step
  状态；不签发新 authority。

### M1 — dual-read / shadow envelope

- 新 normalizer 对 E1/E2 只生成 shadow envelope；
- 旧 Validator/Writer/scorer 仍决定结果；
- 比较样本 N、组别 N、统计值、CI、P 值、artifact SHA、step status；
- 不允许数值差异；路径表示和字段顺序差异必须由 receipt 解释。

### M2 — consumer 逐层切换

顺序固定：

1. Validator；
2. Writer evidence digest；
3. readiness / completion；
4. scorer / Jury；
5. figure/source-data consumer。

每切一层都增加静态门：该层不得再 glob/raw-read
`step_summary.json`、任意 CSV/JSON 或猜字段名。

### M3 — 删除 point repairs

先删四个整文件候选，再按功能块缩减 source/provenance/serialization。
不提高 repair cap，不放松 evidence gate，不增加 Canonical9 题目词或变量词。

### M4 — 恢复开发实验

只有 E1/E2 shadow/authoritative envelope 回归一致后：

1. 从 E2 当前失败步骤继续 diagnostic run；
2. 再顺序诊断 E3–H3；
3. 九题 development 全通后冻结唯一 commit/image/profile/rubric；
4. 只做一次 final fresh authority run。

## 9. 回归命令

当前基线：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/research_agent/test_primary_estimate_output_binding.py \
  tests/research_agent/test_bound_percentage_identity_repair.py \
  tests/research_agent/test_char_golden_run_bundle.py \
  tests/research_agent/test_writer_digest_v2.py \
  tests/research_agent/test_stale_step_summary_authority.py
```

每个迁移增量至少执行：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/research_agent/test_primary_estimate_output_binding.py \
  tests/research_agent/test_char_golden_run_bundle.py \
  tests/research_agent/test_writer_digest_v2.py \
  tests/research_agent/test_stale_step_summary_authority.py \
  tests/research_agent/test_evaluation_scorecard.py \
  tests/research_agent/test_validators_figure_source_trace.py \
  tests/research_agent/test_resume_revalidation.py

.venv/bin/python tools/arch_measure.py \
  --diff tools/arch_baselines/execution_phase.json
.venv/bin/python tools/research_agent_module_graph.py \
  --diff tools/arch_baselines/research_agent_module_graph.json
PYTHONPATH=src .venv/bin/python \
  tools/research_agent_resource_baseline.py \
  --diff tools/arch_baselines/research_agent_resource_context.json

.venv/bin/python -m ruff check src/easyicu/research_agent tests/research_agent
.venv/bin/python -m black --check src/easyicu/research_agent tests/research_agent
```

新增的 envelope 专项测试必须覆盖：

- NumPy/Pandas scalar、nullable、NaN/Inf；
- hostile/symlink/container/host path；
- conflicting statistic identity；
- percentage/count identity；
- figure/source-data SHA 与 panel role；
- stale/superseded envelope；
- root/evidence status 版本漂移；
- E1/E2 archived artifact shadow equivalence；
- scientific/authority error 绝不进入 Coder repair；
- normalizer 不消耗 repair budget。

## 10. 本轮结论

路线没有走歪：E1 development canary 证明受控 Agent 全链能够产出
manuscript-ready、scorer-reportable 的结果；E2 也已到真实 robustness
阻断，而不是停在 mock。

但继续按“一种输出形状一个 repair”推进会把 E3–H3 变成新的补丁来源。
当前应冻结 point repair，先把 `StepResultEnvelope + OutputNormalizer` 做成
唯一边界。完成行为等价迁移前，不启动新正式 authority，不提高 repair cap，
也不把 E1 development artifact 写成 paper result。
