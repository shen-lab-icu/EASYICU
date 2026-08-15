# 2026-07-24 外部 code review（`main@6ddd0e4` / PR #5）逐条核实与修复

## 结论摘要

外部审查提出 19 条（P0×4 / P1×6 / P2×6 + 3 条流程/治理）。逐条读源码核实：

- **18 条属实** —— 已修复其中全部 P0/P1 与 4 条 P2。
- **1 条与代码不符（P2-6 HiRID shard 数量校验）** —— 反证见下。
- **另发现 1 条审查未提的 red test**：`main` 上 `test_full_concept_dictionary_audit.py::test_public_database_source_mappings_have_no_structural_errors` 本身就是红的（stash 对比确认为既有，非本次引入）。

## 逐条核实结果

| 编号 | 结论 | 证据 |
|---|---|---|
| P0-1 PR 带红合并 | 属实（流程） | PR #5 自述 20 red + 12 arch drift，无 reviewer；`agent/CURRENT.md` 亦记录 |
| P0-2 `dur_var` 单位靠分布猜 | **属实** | `api.py` `duration_is_hours = q95<=48 and median<=24`；产出端 8 处单位不一致（`callback_utils.py:2129` 小时 / `callback_apply.py:956` 分钟 / `callback_utils.py:2384` 小时） |
| P0-3 chunk size 改变 SOFA | 属实 | `api.py` docstring 与 warning 自述；且低内存档位 1000 vs 其余 2000 → **同命令跨机器结果可不同** |
| P0-4 全局 loader 竞态 | 属实 | `_get_global_loader` 对模块级 `_global_loader` 做无锁 check-then-act |
| P1-1 MIMIC-III → MIMIC-IV | 属实 | `dataio.py:222` `"mimic" in s → "miiv"`；`_CORE_TABLES` 无 `miii` |
| P1-2 未知库静默回退 miiv | 属实 | `_database_profile_or_default` 吞 KeyError |
| P1-3 runner_image 未打包 | **属实（已实测）** | HEAD wheel 内 `runner_image` 条目为空；修复后 3 个文件均在 wheel 内 |
| P1-4 PID 可杀错进程 | 属实 | `stop_app` 读 PID 直接 SIGTERM，无身份校验；`/tmp/easyicu` 无 UID 隔离 |
| P1-5 嵌套产物漏登记 | 属实 | `_collect_safe_output_artifacts` 用 `out_dir.iterdir()`，子目录内容既不收集也不删除 |
| P1-6 中继无预算/白名单 | 属实 | 无 body 上限、无 max_tokens/n/messages 上限；`_RATE_LIMIT_STATE` 无淘汰 |
| P2-1 lint 豁免过宽 | 属实（未改，见下） | `--isolated` 实测 research_agent 265 条（F401×256 / F841×9） |
| P2-2 CI 依赖漂移 | 属实 | pyproject `pyarrow>=23` vs agent CI `pyarrow>=14` |
| P2-3 镜像不可重建 | 属实 | base image 未 pin digest；OCI label 为占位 `your-org/EASYICU` |
| P2-4 三个版本号 | 属实 | package 1.0.0 / FastAPI 0.1.0 |
| P2-5 采样失败回退全库 | 属实 | `_sample_patient_ids` 异常吞掉返回 None |
| **P2-6 HiRID 只看文件数** | **不成立** | 见下 |

### P2-6 反证

审查称"没有验证连续编号、逐文件映射、footer"，并举例 `1,2,4,5` 缺 `3` 不会被发现。
实际 `data_converter.py:384-402` `_valid_numbered_parquet_shard_count`：

```python
if shard_nums[0] != 1 or shard_nums != list(range(1, len(shard_nums) + 1)):
    return 0                                      # 连续性校验
if not all(self._has_parquet_footer(shard_paths[n]) for n in shard_nums):
    return 0                                      # footer 校验
```

举例中 `1,2,4,5` 返回 **0**（不是 4），`existing_count >= len(part_files)` 不成立 → 不跳过转换。
未做的只是 per-file SHA-256 / size 映射，与所述失败场景无关。**该条未修改。**

## 修复内容

### P0-2 `dur_var` 显式单位契约（新文件 `src/easyicu/table/duration.py`）

把单位从"由数值推断"改为"由产出端声明"：

- 产出端 6 处调用 `set_dur_var_unit(frame, UNIT_MINUTES|UNIT_HOURS|UNIT_TIMEDELTA)`
  （`callback_utils.ts_to_win_tbl` / `grp_mount_to_rate` / eICU 速率、`callback_apply` 内联 `ts_to_win_tbl`、`callbacks.py` win_dur）。
- 消费端 `resolve_dur_var_hours()`：timedelta 自描述 → 精确；有声明 → 精确；无声明 → 旧启发式 + **WARNING 点名概念**；
  `EASYICU_STRICT_DUR_VAR_UNIT=1` 时无声明直接 `DurationUnitError`（供论文级运行 fail-close）。
- 顺带修两处同族缺陷：
  - `concept/__init__.py` `_align_time_to_admission` 原先对**任何**数值 dur_var 无条件 `/60`，
    对本来就是小时的产出（`ts_to_win_tbl` 数值分支 / HiRID `grp_mount_to_rate`）会**除短 60 倍**；改为按声明单位换算。
  - 同文件 `pd.to_numeric(combined['dur_var'])` 对 timedelta 列会得到**纳秒**；改为先走 timedelta→小时。
- 新增 `MAX_WINDOW_EXPANSION_POINTS = 10_000` 展开上限：单窗口超限抛 `WindowExpansionError` 并打印原始行
  （原先 1e9 小时会一路 append 直到进程被杀）。

实测（`test_review_20260724_fixes.py`）：10 分钟输注声明 minutes → 1 点；声明 hours → 11 点。旧代码两者都是 11 点。

### 其余

- **P0-3**：SOFA 档位改为固定 `SOFA_FIXED_CHUNK_SIZE = 2000`，**去掉内存分档**；低内存需显式 `EASYICU_AUTO_CHUNK_SIZE` 并收到 warning。
- **P0-4**：`_LOADER_LOCK`（RLock）覆盖 check→create→publish→return 全段；换出的旧 loader 释放缓存。
- **P1-1**：`_detect_database` 先测 MIMIC-III 再测 MIMIC-IV；路径歧义（裸 `mimic`）改用表结构判定，判不出返回 `unknown` 而非猜一个；`_CORE_TABLES` 补 `miii`；`data-sources.json` 给 `mimic` profile 加 `miii` 别名（原先 webserver 用 `miii` 但 registry 不认）。
- **P1-2**：具名但不认识的库 → `ValueError` 并列出支持列表；仅 `None`/空保留 miiv 默认。
- **P1-3**：`pyproject.toml` package-data + `MANIFEST.in` 补 `runner_image`；实测 wheel 内已含 3 个文件。
- **P1-4**：PID 文件改 JSON（pid + create_time + cmdline marker），原子 `os.replace` 写入；`stop_app` 发信号前用 psutil 校验 cmdline 与启动时间，不匹配就拒发；runtime dir 走 `$XDG_RUNTIME_DIR`，`/tmp` 回退加 UID 且 `chmod 0700`；兼容旧裸整数格式。
- **P1-5**：产物收集改为限深（8 层）递归，symlink/hardlink/特殊文件仍拒收并删除。
- **P1-6**：流式 body 上限 2 MB、messages ≤200、max_tokens ≤8192、n ≤1；转发字段改**白名单**（`provider`/`route`/`transforms` 等被剥离）；限流状态改 `OrderedDict` + 过期清理 + LRU 上限 4096。
- **P2-2**：agent CI `pyarrow>=14` → `>=23`；新增 `tests/test_ci_dependency_floors.py` 锁死 CI floor ≥ pyproject floor（含负向对照验证）。
- **P2-3**：base image 改 `ARG BASE_IMAGE` 便于按 digest 固定（默认仍为 tag，并注明"tag 默认不是可复现构建"）；OCI label 改真实仓库地址。**未伪造 digest。**
- **P2-4**：FastAPI `version` 改由 `importlib.metadata.version("easyicu")` 单一来源。
- **P2-5**：新增 `allow_unbounded_fallback=False`（默认）——传了 `max_patients` 却采样失败时报错，不再把"取 100 例预览"悄悄变成全库提取。

## 验证

| 项 | 结果 |
|---|---|
| `tests/test_review_20260724_fixes.py` | **46 passed**（新增，逐条对应 finding） |
| `tests/test_ci_dependency_floors.py` | **2 passed** + 负向对照确认能抓漂移 |
| `pytest tests/ --ignore=tests/research_agent` | **1940 passed / 39 skipped**，2 failed |
| ↳ failed #1 `test_load_concepts_bounded_flag_preserves_legacy_positional_tail` | 本次签名新增 `allow_unbounded_fallback`，已按"只许追加"更新锁并通过 |
| ↳ failed #2 `test_public_database_source_mappings_have_no_structural_errors` | **既有红**：`git stash` 后在干净 `main` 上同样失败 |
| `pytest tests/research_agent/test_runner.py` | **41 passed** |
| `pytest tests/research_agent -k "evidence or artifact or execution or phase_contract"` | **732 passed** |
| `ruff check`（12 个改动文件） | All checks passed |
| `black --check` | 改动文件均干净（`runner.py` 的 black 差异为既有，且不落在本次新增块内） |
| wheel 打包 | HEAD wheel 无 `runner_image`；修复后 3 文件均在 |

## 未做 / 留作后续

1. **P0-3 根因未消除**。已消除"同命令跨机器结果不同"，但 chunk size 与 SOFA 窗口展开的耦合本身仍在。
   真正的 partition-invariance 证明需要真实库（`/Volumes/外置硬盘/databases`）跑
   `chunk_size ∈ {None,250,1000,2000,4000}` 的 `assert_frame_equal(check_exact=True)`，本次未跑。
   **在该测试通过前，不建议用自动分块结果生成论文级统计量。**
2. **P2-1 lint 未收紧**。`--isolated` 实测 research_agent 265 条（F401×256 / F841×9）。
   F401 多在 re-export 面且架构 gate 依赖模块身份，批量 `--fix` 属于大范围重构，与 "prefer small reviewable patches" 冲突，故未动。
   9 条 F841 中 `agents/core.py:1815` 的 `outputs` 计算后丢弃，紧邻 figure predicate 注释，**疑似逻辑残留**，值得单独看。
3. **P2-3 剩余**：apt 包未固定版本、pip 无 wheel hash、lock 只固定直接依赖。
4. **P0-1 流程项**（分支保护、CODEOWNERS、拆分巨型 PR、把 20 red + 12 arch drift 转成 issue）非代码可改，需仓库管理员操作。
5. 既有红测试 `test_public_database_source_mappings_have_no_structural_errors`（`mech_circ_support` 等 itemid 缺失）未修，属数据字典问题，与本次审查无关。

---

# 第二轮审查（2026-07-25）：逐条核实与修复

审查对第一轮修复提出 3 个 P0 + 5 个 P1 + 3 项发布门禁。**核实结果：绝大部分属实**，其中 3 条是我自己引入或漏掉的：

## 我引入/漏掉的（最该认的三条）

| 条目 | 性质 | 复现证据 |
|---|---|---|
| **datetime 分支忽略声明单位** | **第一轮漏修** | `api.py` 旧第 215 行硬编码 `pd.to_timedelta(dur_numeric, unit="m")`；我只修了 numeric 索引分支。声明 HOURS + datetime 索引 → 被当分钟 → **同样 60 倍误差** |
| **loader 活动期被清缓存** | **第一轮引入的回归** | 我加的 `_release_loader(previous)` 会在 B 线程切库时清掉 A 线程正在用的 loader 缓存 |
| **扁平 MIMIC-IV 被判 miii** | **第一轮引入的回归** | 我加的 `icustays./d_items.` 规则命中转换后扁平布局；实测转换目录 → `miii`（两代都有 icustays，表名不可判版本） |

## 逐条处置

1. **P0-1.1 默认失败开放** → 采纳。默认翻转为**失败关闭**；旧 `EASYICU_STRICT_DUR_VAR_UNIT` 保留兼容，新增反向开关 `EASYICU_ALLOW_DUR_VAR_UNIT_GUESS=1` 才允许猜。主套件 1963 passed 未因此破。
2. **P0-1.2 datetime 分支** → 采纳。两条分支合并到 `_resolve_duration_hours()`，声明单位不可能被一条读到、另一条忽略。
3. **P0-1.3 坏值静默变 0** → 采纳。负值/±inf **直接抛 `DurationValueError`**（原先经 `max(x,0)` 变成合法暴露点）；NaN 按行丢弃并计数告警，不再变成"零长度但仍发一个点"的窗口。
4. **P0-1.4 attrs 脆弱** → 部分采纳。已把两条消费分支收敛到单一入口并让 timedelta 自描述路径优先；**未**把单位提升为 `WinTbl` 结构化字段、未做 concat 单位冲突检测与 parquet 往返——列为后续。
5. **P0-2 SOFA 分块不变性** → **已用真实数据关闭，见下。**
6. **P0-3 loader 生命周期** → 采纳。改为**按配置分键的有界缓存**，切库不再主动清理别人在用的对象；`clear_global_loader()` 这种显式拆除仍然释放。新增"A 持有中 B 切库"的并发测试。
7. **P1-1 MIMIC 版本识别** → 采纳并加强。改为 **schema 判定优先于路径名**（`stay_id` vs `icustay_id` / `anchor_year` vs `dob`），只读 parquet footer 不扫行。测试还暴露出旧实现吃**整条路径**字符串——pytest tmp 目录名含 `mimic_iv` 就会劫持判定，现已由"内容优先"消除。
8. **P1-2 超深目录静默漏产物** → 采纳。改为 `OutputArtifactPolicyError` 失败关闭，并加文件数/目录数/单文件大小/总大小四道上限。
9. **P1-3 转发未规范化值** → 采纳。`_strict_int` 拒收 `8192.9` / `"8192"` / `True`，并把规范化值**写回 payload**，检查的和发出的是同一个值。
10. **P1-4 异步路由内同步 HTTP** → 采纳。`run_in_threadpool` + `asyncio.Semaphore` 并发上限（默认 8），不再让一个 180s 上游调用堵住 `/health` 和其他请求。
11. **P1-5 上限当默认值** → 采纳。拆分 `HOSTED_DEFAULT_OUTPUT_TOKENS=2048` 与 `HOSTED_MAX_OUTPUT_TOKENS=8192`。**未做**：输入 token 计价、按用户/按日预算。
12. **发布门禁 1（wheel 未真构建）** → 采纳。新增 `tests/test_packaging_runner_image.py`：真实 `pip wheel` → 干净 venv 安装 → `importlib.resources` 读取 → `pip check`，实测 **2 passed (73s)**；并给主 CI 加 `packaging` job（wheel + sdist + pip check + package-data 校验）。
13. **发布门禁 2（镜像 digest）** → 部分。`ARG BASE_IMAGE` 已可注入 digest，仓库默认仍是 tag。**不伪造 digest**；固定为默认值需要联网取真实 digest，属仓库管理动作。
14. **发布门禁 3（PR 级 CI 证据）** → 未做，需要推分支开 PR（对外动作，待授权）。

## P0-2 已关闭：真实数据实测分块不变性

审查称"仅验证了默认 chunk 在不同内存下相同，未比较不同 chunk 的实际 SOFA 输出"。**该实验已在本机真实库 `/Volumes/外置硬盘/databases` 上完成**，`pd.testing.assert_frame_equal(check_exact=True)`：

| 库 | 队列 | 条件 | 结果 |
|---|---|---|---|
| MIMIC-IV | 1,000 | chunk 250 / 500 / 1000 | **逐字节相同**（240,115 行） |
| MIMIC-IV | 3,000 | chunk 250/500/1000/2000/4000 + workers 1/4 | **逐字节相同**（sofa 706,167 行；sofa2 634,727 行） |
| MIMIC-IV | 10,000 | chunk 500 / 2000 / 4000 | **逐字节相同**（2,351,306 行） |
| eICU | 3,000 | chunk 500 / 2000 | **逐字节相同**（456,899 行） |

结论有两层：

1. **审查的 P0-2 顾虑在 ≤10k 队列上不成立**——SOFA/SOFA-2 确实是 partition-invariant。
2. 更要紧的是：`api.py` 里那句"chunk size can change large-cohort window expansion results"**是没有证据的**，我第一轮还照抄了它去论证固定档位。现已按实测改写注释，并保留固定档位作为便宜的保险（**全库 ~94k 规模仍未实测**）。

harness 固化为 `tests/test_sofa_partition_invariance.py`（`needs_real_data`），经 pytest 对真实数据实跑 **11 passed (118s)**。

## 第二轮验证

| 项 | 结果 |
|---|---|
| `tests/test_review_20260724_fixes.py` | **66 passed**（新增 r2_* 组：loader 占用中不被清、扁平 MIMIC-IV、超深/超量产物、relay 严格类型/默认值/不阻塞事件循环） |
| `tests/test_sofa_partition_invariance.py --run-real` | **11 passed**（真实 MIMIC-IV） |
| `tests/test_packaging_runner_image.py --run-packaging` | **2 passed**（真实 wheel 构建 + 干净 venv 安装 + pip check） |
| `pytest tests/ --ignore=tests/research_agent` | **1963 passed / 54 skipped**，1 failed = 既有红（干净 `main` 同样红） |
| `tests/research_agent/test_runner.py` | 41 passed |
| ruff / black | clean |

## 第二轮仍未关闭

1. `dur_var` 单位**未**成为 `WinTbl` 结构化字段；无 concat 单位冲突检测、无 parquet 往返保存。attrs 仍是唯一载体。
2. SOFA 不变性**未**在全库 ~94k 规模实测。
3. Relay 无输入 token 计价与按用户/按日预算。
4. 镜像默认未 pin digest；apt 未固定版本、pip 无 wheel hash。
5. PR 级 CI 证据（3.10/3.11/3.12 全绿、完整 research_agent、20 red resume、12 arch drift）——需推分支开 PR。
6. 完整 `tests/research_agent` 套件本轮未跑完（上轮跑到 56% 无失败后被我主动中止，因为改了代码结果会失效）。

---

# 第三轮审查（2026-07-25）：逐条核实与修复

审查撤回了对 SOFA 分块的 P0 判定（承认第二轮的真实数据结果），保留 2 个 P0 + 3 个 P1 + 证据/流程项。**全部核实属实并修复**，另外发现 1 条审查未提的既有缺陷。

## 我必须自己更正的两处表述

1. **"逐字节相同"是我用词不准。** `assert_frame_equal(check_exact=True)` 比的是**规范排序后**的表格内容、dtype 与数值精确相等，**不是** Parquet 文件或序列化字节的比较，harness 也没有比对文件 SHA-256。审查这条纠正正确。已在证据 JSON 的 `comparison` 字段、`docs/evidence/.../README.md` 和 commit message 中改用准确表述。
2. **"11 passed"来自 `-k "chunk_sizes or boundary"` 的过滤运行**，不是全量。该文件定义 13 个用例。已在当前 HEAD 无过滤重跑：**13 passed (230.83s)**。

## 逐条处置

| 编号 | 结论 | 处置 |
|---|---|---|
| **P0-1 `change_dur_unit()` 留下错误单位声明** | **属实** | 改为按**声明的源单位**换算；换算后调 `set_dur_var_unit(new_data, unit)`；未声明的 numeric 直接抛 `DurationUnitError`（原注释 "Already numeric, assume minutes" 正是本契约要消灭的猜测）。换算改走单一 seconds-per-unit 比值，顺带支持 seconds/days |
| ↳ **额外发现（审查未提）** | **既有缺陷** | 该函数内 `from .table.meta import` 前缀错误（模块在 `easyicu/io/`，`easyicu.io.table` 不存在）→ **函数作为公共 API 导出但一调即抛 ModuleNotFoundError**。审查描述的 60 倍路径因此实际跑不到。已修为 `..table` |
| **P0-2 证据扫描 OSError 静默跳过** | **属实** | `except OSError: continue` → 抛 `OutputArtifactPolicyError`。两个测试：monkeypatch `Path.iterdir` 抛 `PermissionError`，以及真实 `chmod 000` 目录 |
| **P1 CacheManager 强引用** | **属实** | `list` → `weakref.WeakSet`（不可弱引用对象回落强引用，不静默失败）；新增 `unregister_memory_cache()`；`clear_memory_cache` 先快照再迭代。测试用 `weakref.ref` 验证淘汰后可回收 |
| **P1 流式绕过并发上限** | **属实** | semaphore 原先只包住 `_post_upstream`，`stream=True` 时连接一建立就释放，正文尚未消费 → 长流可无限并发。改为 `_bounded_streaming_response()` 让槽位覆盖整个流生命周期，`finally` 中关闭上游并释放（客户端断连也走到）。**实测**：limit=2、8 并发流 → 峰值 2、槽位全部归还 |
| **P1 陈旧注释** | **属实** | `api.py` 函数体内仍有 "Chunk size can still change SOFA window expansion" 与刚建立的证据冲突，已删改；加测试 `test_r3_api_has_no_stale_chunk_invariance_claim` 防回潮 |
| **`load_win()` 无单位契约** | **属实** | 新增必填 `duration_unit`（numeric duration 未给则抛错）；timedelta 自描述免填 |
| **`WinTbl` 无单位字段** | **属实** | 新增结构化 `dur_unit`，构造时与 frame attrs 双向同步；timedelta 列自动识别为 `timedelta` |
| **证据固化为 JSON** | 采纳 | 新增 `tools/record_partition_invariance_evidence.py`，产出 commit / tree-clean / python / pandas / prepared-manifest sha / 每配置 canonical frame sha256 / cohort_id_sha256。**无 PHI**（队列只以排序后 stay id 的 SHA-256 出现）。已生成并提交到 **`docs/evidence/partition_invariance/`**（`research_output/` 是 gitignored scratch，放那里带不走） |

## 第三轮验证

| 项 | 结果 |
|---|---|
| `tests/test_review_20260724_fixes.py` | **81 passed**（新增 r3_* 组 12 项） |
| `tests/test_sofa_partition_invariance.py --run-real`（**无过滤，全量**） | **13 passed (230.83s)** |
| 证据 JSON（clean tree @ `2471137`） | miiv sofa 7/7、sofa2 7/7；eicu sofa 4/4，全部 `matches_reference=true` |
| `pytest tests/ --ignore=tests/research_agent` | **1978 passed / 54 skipped**，1 failed = 既有红 |
| `tests/research_agent/test_runner.py` | 41 passed |
| `ruff check src tests`（CI gate） | All checks passed（`tools/` 的 5 条为既有，不在 gate 内） |
| black | 我的新增块均合规；`data_load.py` / `table/__init__.py` / `data_tools.py` / `cache_manager.py` 的 black 差异为**基线既有**，未整体重排（避免噪音 + 会使 figure2 scorer tree 摘要失效） |

## 第三轮仍未关闭

1. **完整 `tests/research_agent` 仍未跑完**。本轮后台重启，13 分钟报告 414 项、**0 失败**、进度 6%；按此速率需 3 小时以上。这解释了前两轮为何从未跑完。covering 子集（runner 41 + evidence/artifact/execution 732）全绿。
2. 全库 ~94k 规模的分块不变性未测。
3. 镜像默认未 pin digest；apt 未固定版本、pip 无 wheel hash。
4. Relay 无输入 token 计价与按用户/按日预算。
5. `rbind_tbl` / `cbind_tbl` 的 `dur_unit` 保存与冲突拒绝**未实现**（审查要求项 4 的一部分）；当前只覆盖 `WinTbl.__init__` / `change_dur_unit` / `load_win`。
6. PR 未创建、未推送，GitHub Actions 无证据。
