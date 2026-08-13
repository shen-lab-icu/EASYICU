# E1 真实流程三处修复：missingness 图执行器、raw-contract 形状、外层假绿

日期：2026-07-27
分支：`fix/external-review-20260724-p0-p1`
接手自：Codex 交接（`9301521` 之后）
本轮提交：`eaa84ce` / `0905f67` / `ede84d0`

真实 run：
`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260726_luna_miiv_dev_1d5743b_e1_r25/e1_sepsis3_prevalence_mortality/aware/run_20260726T235648_51cac6`

---

## 一、`05_missingness_measurement_audit_figure` — 把「中位数」当成「计数」

### 真实失败

```
status: failed
ValueError: table:missingness_measurement_audit contains invalid structural
accounting rows at source_row_index values [2, 3, 6, 7, 10, 11, ... 50, 51]
```

被判「非法」的行索引全部 ≡ 2 或 3 (mod 4)。对照真表：每个数值变量占 4 行 ——
`missing` / `valid_observed` / `median` / `iqr`。**被拒的正好是全部 median/iqr 行。**

生成代码要求每一行都能用 `count / denominator` 对账。但中位数不是计数，
它的 `count` 按 schema 本来就是空的；它通过 `summary_value`/`q1`/`q3` 对账。
**一个「每行都必须有计数」的校验，把 schema 规定的正确形状判成了坏账。**

### 修法

新增 sealed executor
`execution/runners/missingness_measurement_figure_executor.py`：

- **owner 只看结构合同**：`auxiliary` + `visualization` + 两个 `all_rows`
  table 输入 + 单一 figure product。不看 intent 散文，不看 E1 名字。
- 每个 binding 逐项校验：路径包含于 run_dir、SHA-256（读前 + 读后各一次）、
  声明列序、行数。
- **按 metric 声明的行类分别校验**：计数行对 `count/denominator/percentage`，
  分布行对 `summary_value/q1/q3` 且必须没有 count。
- 逐变量对账：`missing + (valid_observed | Σlevel_distribution) == denominator`。
- `level` 非空的行按 (variable, measure) 分组必须划分该 measure 的分母 ——
  **按 level 列的结构判定，不按 measure 名字**（避免又一个字符串白名单）。
- chart_type 用文章策略 `data_quality` 词表里的 `availability_panel` +
  `coverage_heatmap`。

### 中途被 validator 抓到的一个真实错误（我自己引入的）

第一版我另外写了两个「派生」source data：availability 补数（`available_n`/
`available_pct`）和重排后的 coverage 网格。`FigureSourceDataValidator` 两条都拒了：

```
these source-data value columns were not verified against any row-aligned
upstream value vector: ['available_n', 'available_pct', 'missing_n']
source-data values disagree with measurement_process_audit.csv (key: variable)
```

**validator 是对的**：派生列无法回溯到上游任何一行；而 coverage 网格按
`variable` 连接会多对多（`lact_n` 有 2 行）。

改成：两个 panel 的 source data 都是**上游行的逐字子集**，带 `source_row_index`
（validator 已支持的位置键）。Panel A 直接画上游 `missing` 行自己的
`percentage`，因此也更诚实 —— 缺失审计图就该画缺失。

### 验证

- 对**真表实跑**：55 + 21 行全部校验并消费，PNG/SVG/PDF/TIFF 四格式齐全。
- 17 项聚焦测试 + 48 项相邻 runner 测试通过。
- 在真实 plan 上跑 `select_standard_executor`：只认领
  `05_missingness_measurement_audit_figure` 一步，无串台。

---

## 二、`06_primary_adjusted_association` — 合同「缺失」其实是形状认错

### 真实失败

```
ValueError: raw_input_contracts.contracts is missing
```

而真实 manifest 里 6 个合同**全都在**：

```
contracts TYPE: dict
contract keys: ['age','death','sep3_sofa2_max','sep3_sofa2_measured',
                'sep3_sofa2_n','sex']
```

生成代码写的是：

```python
if not isinstance(contracts, list):
    raise ValueError("raw_input_contracts.contracts is missing")
for contract in contracts:
    by_name[contract["column"]] = contract
```

host 发出的 `contracts` **本来就是按列名 keyed 的对象**，代码却先断言它是 list，
然后再把它「重建成」按列 keyed 的映射 —— 断言先炸，于是把形状不符报成了缺失。

### 为什么既有修复没接住

`patch_raw_contract_mapping_iteration` 的触发条件是
`AttributeError: 'str' object has no attribute 'get'` —— 那是**直接遍历 dict
才会出现的 traceback**。本例在到达那一步之前就被类型断言拦下了，
**那条 traceback 永远不会出现**。

### 修法

1. **提示词**：dict-keyed 这件事此前只写在**一个** Coder 附件里
   （materialized execution receipt），而 step 06 根本没收到那个附件。
   另外两处 raw-input 提示和 `bind_execution_cohort_runtime` 只说了
   「contracts 是唯一可执行元数据」，**没说它是什么形状**，模型只能猜。三处都补上。
2. **确定性修复** `patch_raw_contract_list_type_assertion`：只认一种明确形状 ——
   唯一一个 `isinstance(<contracts>, list)` 守卫 + 唯一一个按 `column` 重建键的
   循环 —— 把 `list` 改成 `dict`、把遍历改成 `.values()`。
   **触发键绑在 host schema 路径 `raw_input_contracts.contracts` 上**，
   不绑模型自己写的错误文案（那是散文，不是稳定触发器）。
   不动 cohort / 变量 / 模型 / estimand / 任何数值；每列的存在性检查全部保留，
   所以真正缺失的合同仍然 fail closed。

### 验证

拿**真实 analysis.py + 真实 run.log** 跑这条修复：

```
CHANGED: True
-    if not isinstance(contracts, list):
+    if not isinstance(contracts, dict):
-    for contract in contracts:
+    for contract in contracts.values():

REPAIRED READER OK -> columns:
['age','death','sep3_sofa2_max','sep3_sofa2_measured','sep3_sofa2_n','sex']
```

11 项聚焦测试通过（含：wrapped/unwrapped manifest、缺列仍 fail closed、
无关报错不修、不重建键的循环不修、两处候选歧义不修）。

---

## 三、外层假绿 —— 「调用返回了」被当成「任务成功了」

### 真实矛盾

`ehrflowbench_progress.json`：`status=completed`、`completed_tasks=1`、
`failed_or_blocked_tasks=0`、exit 0。
同一次 run 的 `run_status.json`：`execution_complete=false`、7/12 步、
两个失败步。

### 根因

外层在 `_run_one_item_from_cohort` **没抛异常**时就调用
`task_hard_stop.finish(score=score)`。失败两步的 run 照样会返回 score，
于是「没抛异常」被读成了「任务成功」。而 score payload 里**根本没有执行轴字段**，
只有一个派生的 `gate_status` 字符串，下游想区分也无从区分。

### 修法

- `_score_arm` 显式输出 `execution_complete`、
  `step_scientific_requirements_complete`、`failed_step_ids`、
  `missing_step_ids`、步数。
- `_arm_execution_succeeded` 只读这些，**刻意不读 `status` 和
  `paper_authorized`** —— development diagnostic 本来就该以
  `diagnostic_only` + `paper_authorized=false` 收尾，那是合法的「执行完成」；
  不合法的是自己的门报告有未完成/失败步。
- ledger 在该情形下带 error 收尾（外层 totals 随之翻转）、pending 记一条、
  `main()` 返回**新退出码 4**（与 paper-acceptance 的 3 区分开，
  这样 development diagnostic 仍能区分「跑完了但没授权」和「根本没跑完」）。

7 项契约测试，覆盖交接单要求的四种情况，含真实的 7/12 形状。

---

## 四、验证与归因

| 项 | 结果 |
|---|---|
| 新图执行器 focused | **17 passed** |
| 相邻 runner 组（prevalence / owns_contract / deterministic missingness） | **48 passed** |
| raw-contract repair | **11 passed** |
| bench 执行语义 | **7 passed** |
| ruff / py_compile / `git diff --check` | 全绿 |
| black | **未引入任何新漂移**（逐文件与 HEAD 基线逐数相等） |

### ⚠️ 13 项失败已归因为既有问题，不是本轮引入

宽组跑出 13 failed。在**干净 HEAD 的 worktree**（不含批 2/3）复跑同一批文件：
**12 failed，同名**。主因是

```
RuntimeError: Docker execution-runtime dependency capture failed:
EasyICU research-agent source mismatch: expected 6baa666..., observed 0faeb3a...
```

—— 即**镜像源码哈希与工作树不一致**，正是交接单要求「改代码后必须重建镜像」的那条。
另一条 `test_production_prompt_calls_use_the_authorized_delivery_boundary`
指向 `src/easyicu/research_agent/providers/hard_stop.py:115`，**本轮未碰过该文件**。

### 教训（本轮新增）

- **黑色格式化会把别人的历史漂移卷进你的提交。** `black` 一跑，
  `repairs/source.py` 从 +7 变成 +142/-99，其中 137 行是 HEAD 上就存在的漂移。
  做法：先在干净 HEAD 量一遍每个文件的 black 漂移基线，改完再量一遍，
  **两个数必须相等**，不等就说明卷进了别人的东西。
- **不要用 `git stash` 做归因。** 本轮 `git stash && pytest && git stash pop`
  在 pytest 超时时被打断，`pop` 没执行，工作区一度只剩未跟踪文件。
  改用 `git worktree add --detach <path> HEAD`：**只读、隔离、不动工作树**。

---

## 五、下一步

1. 用本轮 HEAD 重建镜像并 `python tools/check_agent_runtime.py --image <tag>`。
2. 从 `05_missingness_measurement_audit_figure` 精确恢复，不重跑前面 7 步成功记录。
3. 目标：12/12、`missing_steps=[]`、`failed_steps=[]`、
   `execution_complete=true`、`step_scientific_requirements_complete=true`，
   `paper_authorized=false`（development diagnostic，预期如此）。

## 六、附带发现（**未修**，待用户定夺）

宽表每个概念机械展开 8 列（`_max/_min/_mean/_first/_n/_measured/_first_time/_last_time`）。
对真正的重复测量（lact：max 376 / min 279 / mean 3824 / first 340 个不同值）这套展开合理。
但对**二值旗标**属于过度展开，并且产生了一个语义问题：

- `sep3_sofa2`：`_max/_min/_mean/_first/_n/_measured` 六列**是同一个 0/1 向量**
  （实测 `identical(max, measured) == True`）。
- `susp_inf`：`_max/_min/_mean/_first` 四列都只有唯一值 `1.0`、47.5% 非空 ——
  positive-only 编码，缺失即「未怀疑」而非「未测量」。

因此当前 05 审计把 `susp_inf_max` 记为 `missing=49,543`，**读起来像「没测」，
实际是「没怀疑」**。这是上游 05 审计的语义问题，不是渲染器的问题；
渲染器如实渲染上游表是正确的。是否要给 positive-only 列改标签/改口径，
属于科学口径决定，**留给用户**，不在本轮静默变更。
