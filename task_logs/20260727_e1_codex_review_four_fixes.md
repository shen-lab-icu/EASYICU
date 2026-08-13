# E1 复核四处修复：提示词被插断、确定性修复认不出真实形状、第二处假绿、执行器负测缺口

日期：2026-07-27
分支：`fix/external-review-20260724-p0-p1`
来源：Codex 对上一轮 `eaa84ce`/`0905f67`/`ede84d0`/`53ab365` 的复核（8 条）
本轮提交：`e45c006` / `7780e43` / `e4f4d6b` / `c4272d3`

真实 run：
`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260726_luna_miiv_dev_1d5743b_e1_r25/e1_sepsis3_prevalence_mortality/aware/run_20260726T235648_51cac6`

---

## 〇、先核验，再动手

Codex 的 8 条全部拿真实 artifact 复核过，**没有一条是幻觉**。核验结果：

| # | 说法 | 核验 | 证据 |
|---|---|---|---|
| 1 | `53ab365` 提示词仍被插断 | ✅ 属实 | 见下一节 |
| 2 | Step06 已 9/9 provider call，普通 resume 卡住 | ✅ 属实 | `.runtime/provider_call_budgets/e38b18ff…json` 恰好 9 个 category |
| 3 | 确定性 `retain_and_flag` 修复匹配不到真实代码 | ✅ 属实 | 拿真实 quarantine 草稿实跑：`CHANGED: False` |
| 4 | 不要简单 `raise` 换 `pass` | ✅ 合理 | 已在文档与测试中写明该修复只完成「保留」，未完成「标记」 |
| 5 | 执行器 fail-closed 有 4 处缺口 | ✅ 属实 | 见第四节 |
| 6 | 非 JSONL 入口仍无条件 `return 0` | ✅ 属实 | `main()` 结尾 4012 行 |
| 7 | `susp_inf` 需要 typed `absence_semantics` | ✅ 属实 | 仍未修，属科学口径决定 |
| 8 | `CURRENT.md` 与实际 8/12 不符 | ✅ 属实 | 本轮已同步 |

---

## 一、提示词被自己的补丁插断（`e45c006`）

### 真实渲染文本

`53ab365` 想补的是「host 的 `out_of_range_action` 是有约束力的」。这句话被插进了**上一句的中间**：

```
… never index it as a list or use
`plausibility_policy.out_of_range_action` is binding, not a suggestion: …
`lower`/`upper` aliases. Do not rediscover metadata …
```

两处 raw-input 提示（`_typed_input_scope_contract`、`_compact_repair_scope_contract`）都是这个形状。

### 为什么既有测试没抓到

既有断言查的是**片段**：`"analysis_plausibility_range + plausibility_policy" in contract`。
片段两端都还在，所以插断不影响断言。**模型读到的是渲染后的整句，测试查的是拼装前的碎片。**

### 修法

- 恢复 `use `lower`/`upper` aliases.` 为一整句，policy 另起一句放在其后。
- 第三处 `bind_execution_cohort_runtime` 此前只写了字段名、**没写它有约束力**，一并补上（同一缺陷类）。
- 新回归断言**渲染后的字符串**：policy 从句前面必须是 `". "` 或换行 —— 这正是「插进句子中间」会违反的性质，而片段检查看不见。

三个面 × 2 项测试，79 passed。

---

## 二、确定性修复认不出真实的 `float(bound)`（`7780e43`）

### 真实被阻断的代码

`llm_concept_auditor` 的原话（`.quarantine/concept_draft.json`）：

> The script raises on finite age values outside the host plausibility range,
> implementing exclusion rather than the binding retain-and-flag policy.

真实脚本（generic helper，不是 per-variable）：

```python
if lower is not None and (numeric < float(lower)).any():
    raise RuntimeError(f"{column} contains values below analysis plausibility minimum")
```

host 的 age 合同：`{"out_of_range_action": "retain_and_flag", "range_policy": "flag_only"}`。

### 为什么仓库里已有的修复没接住

`_sealed_comparison` 只匹配 `Name` 对 `Name`。真实代码把 JSON bound 在比较处收窄成 `float(lower)`，
右操作数是 `ast.Call` —— **一次都匹配不上**。拿真实草稿实跑：`CHANGED: False`。

### 修法

只解包**一层朴素 `float(...)`**：单个位置参数、无关键字、callee 恰为 `float`。
序列侧必须仍是裸 `Name`（`float(series) < bound` 不是这个形状）。
`float(lower, 2)` / `float(lower * 2)` / `round(lower)` / `float(x=lower)` 一律不认。

### 验证

拿**真实 quarantine 草稿**实跑：

```
CHANGED: True
-            raise RuntimeError(f"{column} contains values below analysis plausibility minimum")
+            pass  # _easyicu_flag_only_plausibility_range_retained_v1
-            raise RuntimeError(f"{column} contains values above analysis plausibility maximum")
+            pass  # _easyicu_flag_only_plausibility_range_retained_v1
PARSES OK
```

恰好两处改写，其余（比较式、bound 读取、类型转换）逐字不动。17 项测试通过。

### 关于 Codex 第 4 条：`pass` 不等于 `retain_and_flag`

同意。`retain_and_flag` 是**两件事**：保留 + 标记。
确定性补丁只能证明第一件（删掉终止守卫 → 没有行被排除）；
它无法凭空造出结构化 flag/count —— 那属于生成脚本的声明输出。
本轮把这句话写进 docstring，并加了一条专门测试锁住它：
**只有这个 marker 的 run，完成的是保留，不是标记。**
因此**不**用这条修复去把既有 run「洗绿」。

---

## 三、第二处假绿：非 JSONL 入口（`e4f4d6b`）

`ede84d0` 的 exit 4 只落在 `_run_ehrflowbench_jsonl`。
`main()`（`--bench-kind rule|analysis`）结尾仍是无条件 `return 0` —— **同一个假绿，另一个门。**

修法：同样的 `_score_execution_failures` 检查 + 同一个 exit code 4。

回归**断的是性质而不是今天这两个函数**：任何能 `return 0` 的入口都必须调用执行完成检查
（AST 扫描 `main` / `_run_ehrflowbench_jsonl`，将来新增入口自动被覆盖）。12 项测试通过。

---

## 四、执行器的 fail-closed 缺口（`c4272d3`）

上一轮的行校验只查 `q1 <= q3`，于是以下四种坏数据都能过：

1. median 落在自己的 q1–q3 之外（等于「抄了别的变量的中位数」）；
2. `iqr` 的 `summary_value` 不等于 `q3 - q1`；
3. 同一变量同时报 `valid_observed` 和 level 明细，两者**对不上**（同一批观测被报了两次）；
4. 重复的**无 level** measurement-process 单元格（有 level 的由 partition 检查覆盖，没 level 的没人管）。

四条真实 E1 数据全部满足 —— **这正是问题所在**：这个执行器直接把上游表发布成图，
它默认成立却没人检查的不变量，就是没人检查的不变量。

补齐后拿**真表复验**：55 行审计 → 14 变量、21 行 process → 21 单元格，仍全部通过。
24 项测试（含 7 项新负测）通过。

---

## 五、验证与归因

| 项 | 结果 |
|---|---|
| 提示词渲染（3 个面） | **79 passed** |
| flag-only plausibility repair | **17 passed** |
| bench 执行语义（两个入口） | **12 passed** |
| missingness 图执行器 | **24 passed** |
| 四组 + 相邻提示词/raw-contract | **164 passed** |
| ruff / `git diff --check` | 全绿 |
| black | **逐文件与 HEAD 基线逐数相等** |

### black 漂移逐文件对账

| 文件 | HEAD | 本轮 | 说明 |
|---|---|---|---|
| `agents/core.py` | 5 | 5 | 不变 |
| `resources/coder.py` | 0 | 0 | 不变 |
| `repairs/plausibility.py` | 12 | 0 | HEAD 的 12 行漂移**恰在我重写的那个块内**，重写后为 black-clean |
| `runners/missingness_..._executor.py` | 0 | 0 | 不变 |
| `tools/run_research_agent_bench.py` | 57 | 57 | 不变（57 行是既有漂移，未卷入） |
| 四个测试文件 | 11/6/0/0 | 11/6/0/0 | 不变 |

### ⚠️ 37 项失败为分支既有，不是本轮引入

宽组（`-k "repair or prompt or coder or executor or runner or bench or plausibility or contract"`）跑出 38 failed。
逐项归因方法：

1. 用**工作树源码**重建镜像 → `check_agent_runtime` = `ready` / `network: none`；
2. 在 `git worktree add --detach <path> HEAD` 的**干净 HEAD** 上**另建一个 HEAD 镜像**，
   `PYTHONPATH=<worktree>/src` 跑同一批文件。

结果：**干净 HEAD 也是 37 failed，测试名完全相同。**
（宽组的第 38 项是镜像未重建时的 source-mismatch，重建后消失。）

即：`test_pipeline.py`(12)、`test_resume.py`(11)、`test_post_repair_concept_gate.py`(3)、
`test_visual_repair_governance.py`(4)、`test_runner_trajectory_contract.py`(2)、
`test_provider_protocol_boundaries.py`(2) 等 37 项，是本分支进入本轮时**已经红**的。

> 教训（本轮新增）：**归因 docker 相关失败必须同时对齐源码和镜像。**
> 只重建工作树镜像会把「HEAD 本来就红」误读成「我改红的」，
> 因为 HEAD 的源码与新镜像不匹配，会先抛 source-mismatch 盖住真实原因。
> 正确做法是给 HEAD worktree 也建一个镜像，两边各自自洽再比。

---

## 六、E1 现状与下一步建议

### 现状（真实 `run_status.json`）

```
status                                = diagnostic_only
gates.execution_complete              = false
gates.step_scientific_requirements_complete = false
gates.completed_step_count            = 8 / 12
gates.failed_steps  = [{"step_id": "06_primary_adjusted_association",
                        "status": "blocked_by_concept_audit"}]
gates.missing_steps = ["06_primary_adjusted_association_figure",
                       "07_robustness_sensitivity",
                       "07_robustness_sensitivity_figure"]
gates.paper_authorized                = false      ← development diagnostic，预期
```

上一轮的 Step05 图修复**已在真实 run 上生效**（7/12 → 8/12）。

### 为什么不建议继续 resume 这个 run

`.runtime/provider_call_budgets/e38b18ff…json` 的 Step06 已消费**恰好 9 个** category：

```
initial_generation, contract_repair_patch, contract_repair_patch,
contract_repair_full_rewrite, concept_audit, initial_generation,
runtime_repair_patch, initial_generation, concept_audit
```

`max_step_provider_calls` 默认 9 —— **额度已满**，且预算跨 resume 累加
（列表里已有 3 次 `initial_generation`，即 3 次续跑）。
本轮的提示词修复只对**新生成**有效；这个 run 不会再生成新代码。
即使确定性修复现在能改代码，改完的新 digest 仍需一次 concept audit，而那也要额度。

`final_reservation_state` 还停在 `released: false` / `completed_token: null`。

### 建议（与 Codex 一致）

归档这个 8/12 development diagnostic 作为诊断证据，用本轮 HEAD 重建镜像后
**fresh 跑 E1**，不扩额度、不绕过 final audit、不手改 run 目录。

---

## 七、仍未修（待用户定夺）

- **`susp_inf` 的 positive-only 编码需要 typed `absence_semantics`**。
  `susp_inf_max/_min/_mean/_first` 只有唯一值 `1.0`、47.5% 非空 —— 缺失即「未怀疑」而非「未测量」。
  当前 05 审计把 `susp_inf_max` 记为 `missing=49,543`，**读起来像「没测」**。
  这是上游审计口径问题；渲染器如实渲染上游表是对的。
  不能按数值猜、也不能静默填零 —— 需要一个 host 侧的 typed 字段说明「缺失的语义是什么」。
- **二值旗标的 8 列机械展开冗余**：`sep3_sofa2` 的六列实测为同一个 0/1 向量。
- **本分支 37 项既有失败**：不是本轮引入，但也没修；需要单独一轮处理。
