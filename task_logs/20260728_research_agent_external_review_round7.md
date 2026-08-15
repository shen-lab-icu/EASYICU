# Research-agent 深入复审（第七轮）—— 逐条核实与修复

- 日期：2026-07-25
- 分支：`fix/external-review-20260724-p0-p1`（基线 `main@6ddd0e4`）
- 审查基线：`82383c5..6b5815b`（第六轮 9 个提交）
- 审查结论：REQUEST CHANGES —— 6 个 P0 + 2 个 P1
- **未推送、未合并、未调用 Provider / Docker / 患者数据 / Canonical9**
- 编号说明：审查方称其为「第六轮」（他对本分支的第 6 次复审），本文件沿用我方序号「第七轮」（这是你转交给我的第 7 份报告）。同一份报告，两套计数。

## 这一轮的主题

三轮下来是一条递进的线：

| 轮次 | 病灶 |
|---|---|
| 第五轮 | 门修好了，但生产路径没接 |
| 第六轮 | 测试跑了生产代码，但喂的输入是生产端永远不会产生的形状 |
| **第七轮** | **门接上了、形状也对了，但门读的那个字段是调用方能写的** |

`metadata["aggregate_only"] is True` 是「一个 dict 里的字符串」。任何能往 EvidenceStore 写记录的路径都能写这个字符串——最直接的就是 MCP `bind_evidence`，它接受调用方自带的 metadata / producer / generation_mode。所以上一轮新建的 host-owned 审计**产出了正确的结论，而最终门禁并不验证结论的出处**。

## 逐条结论（全部属实，无一条被驳回）

| id | 审查描述 | 核实（读源码） | 处置 |
|---|---|---|---|
| P0-1 | 最终门禁只查 `aggregate_only == True`，不验证出处；MCP `bind_evidence` 可自带 metadata | 属实：`figure_egress.py:165` 只做真值判断；`mcp_server.py:691-705` 原样透传 metadata / producer | 新增 `_verify_host_privacy_authorization()`：basis / producer / audit version / 审计记录存在 / 审计文件 SHA 复算 / 收据 schema / `aggregate_only is True` / figure_id 一致 / 每个 source 的 SHA 三方一致。MCP 侧拒收 8 个保留 metadata key + 拒绝冒充 host producer |
| P0-2 | Parquet 只读 schema；CSV/TSV 只查列名与 count 列；JSON 只查 key；读文件前不复算 SHA | 属实：`_inspect_parquet` 用 `pq.read_schema`；`_inspect_delimited` 只对 count 列建 pairs；`_inspect_json` 的列表元素从不进入 pairs | 新增流式 `_ValueScanner`，三种格式一律逐格扫描；审计前复算 SHA 并与注册摘要比对；审计版本 1.0.0 → 1.1.0，收据 schema → `/2` |
| P0-3 | footnote 解析到 step 但该 step 无此指标时返回 `([], None)`，调用方回落全局 | 属实：`return scoped, (step_id if scoped else None)` | 改为 `_ScopedMetric(values, step_id, cited_step)`；`cited_step_lacks_metric` 直接报错并 `continue`，不再回落。AUROC CI 的 `] or list(summaries)` 回落一并删除 |
| P0-4 | 只能同进程恢复，应标注 single-process | 属实 | `HUMAN_REVIEW_RESUME_SCOPE` + `HumanReviewPending.resume_scope` / `resume_pid` / `resumable_here`（schema → `/2`）；`resume_human_review()` 的错误信息点名该边界 |
| P0-5 | `evidence.records()` 异常被吞，用空 authority 继续建审批 | 属实：`except Exception: digests = {}` | 新增 `HumanReviewAuthorityError`，读不到证据就不发审批请求 |
| P0-6 | 只检查第一张表的 `dur_var`，异名 duration 列绕过单位检查 | 属实：`_combined_dur_unit(tables, dfs, tables[0].dur_var)` | 新增 `WindowContractError` + `assert_window_contract()`：`dur_var` / `index_var` / `id_vars` 三项必须一致，在四个合并函数的唯一咽喉点 `_combined_dur_unit` 里执行 |
| P1-a | `record_upload()` 在网络调用之前执行，收据无法区分「已授权 / 已尝试 / 成功 / 失败 / 未知」 | 属实：`authorize_figure_upload` 内部即写 `policy.uploaded`，实际发送在 `visual_qa.py:676` | 每条 entry 带 `transport` 字段（authorized / transport_completed / transport_failed / transport_unknown）；`record_transport_outcome()` 按 SHA 回填；`completed` 阶段把仍是 authorized 的一律降级为 unknown。收据 schema → `/3` |
| P1-b | finding 的 key 白名单不校验值内容 | 属实 | `_sanitize_detail_value()`：路径只留文件名、≥6 位数字 token 替换为 `<id>`、小 `duplicate_count` 报上界、嵌套结构一律 `<withheld>`、字符串截断 300 字 |

## 三处需要说明的设计判断

### 图外发：flag 从「授权」降为「索引」

新门禁把 `aggregate_only` 当作**指向审计的索引**，然后端到端验证那份审计：

1. `aggregate_only_basis == "host_privacy_audit"`；
2. `producer` 属于 host 自己的图渲染器（`publication_figure_skill`）；
3. `aggregate_only_audit_version` 在 `TRUSTED_AUDIT_VERSIONS` 里；
4. `figure_privacy_audit_evidence_id` 能解析到记录，文件存在，**复算 SHA == 注册摘要**；
5. 收据 schema 正确、`audit_version` 与 metadata 一致、`aggregate_only is True`；
6. 收据里的 `figure_id` == 图的 `figure_id`；
7. 图声明的每个 source：审计确实检查过它、图的 metadata 与审计的摘要一致、EvidenceStore 现在记录的摘要仍与之相等。

**这里没有做的**：外发时不重新哈希 source 文件本身。外发的是已经渲染好的图片字节，source 之后在磁盘上被改写并不会改变那张图；真正相关的风险是「同一个 id 下换了一份 artefact」，那由「注册摘要 == 审计摘要」这一条覆盖。

MCP 侧同步堵住入口：`RESERVED_PRIVACY_METADATA_KEYS`（8 个 key）拒收，`producer` 若落在 `TRUSTED_FIGURE_PRODUCERS` 里直接报错。

### 隐私审计：值扫描的假阳性边界

逐格扫描必然带来假阳性风险，取舍是明确的：

- 「标识符形状」定义为 **≥6 位连续数字**，且前面不是数字或小数点——所以 `0.000001`（p 值）不匹配，`30042318`（stay_id）匹配；
- 名字像量级的列/键（`n`、`*_count`、`total`、`rows`…）豁免标识符扫描，因为 4200 例、120 万条测量是摘要表的正常内容；这些列反过来仍然接受**小单元格**检查；
- float 值不扫描——标识符以整数或字符串存储，扫 float 只会把长求和与高精度统计误报掉。

有一条回归专门守这个边界（`test_a_genuine_aggregate_source_is_not_flagged`）：一张带 4200/5221 分层计数和 `0.000001` p 值的正常汇总表必须放行。门禁被误伤到没人敢开，和门禁形同虚设，是同一个失败。

**另外修掉一处审计自身的泄漏**：原来的 reason 串会把命中的 token 原样写进去（`f"...identifier-shaped token: {token}"`），而 reason 会进入图的 evidence metadata 和收据。现在一律脱敏成 `<8-digit token>`，并有回归断言收据里不含原值。

### human review：把边界写成字段，而不是写在文档里

跨进程恢复仍然不支持，原因和上一轮实测的一样：phase handoff 里有活的 `EvidenceStore`（含 `RLock`），任何 checkpointer 都序列化不了，所以它们被移出 checkpoint state；新进程拿不到那些对象。

这一轮做的是**把这个事实变成机器可读的**：`HumanReviewPending.resume_scope == "same_process"`（`Literal`，调用方改不了）、`resume_pid`、`resumable_here`。一个审批 UI 现在可以在**弹给医生之前**就判断这个 pause 能不能在自己这儿被回答，而不是等医生签完字才发现送不回去。要真正跨进程需要 phase handoff 可重建——那是 pipeline 结构改造，不在本轮范围，也不在本轮声称的能力范围内。

## 测试

`tests/research_agent/test_review_20260728_agent_fixes.py`，41 项。原则同前几轮，并针对本轮病灶再收紧一层：**图相关的测试全部建真实 `EvidenceStore`、注册真实制品、跑真实审计**，不用带便利属性的假对象。

- 图外发：host 审计过的图仍能上传（生产路径不能被收紧收死）／自签 `aggregate_only` 被拒／审计链三个环节各缺一个都被拒／不受信 producer 顶着真审计的 metadata 被拒／审计收据被改写被拒／metadata 与审计对不上被拒／声明了审计没查过的 source 被拒／MCP 拒收保留 key 与冒充 producer
- 隐私审计：CSV 值级标识符（列名完全干净）／JSON 列表元素里的标识符／Parquet 小单元格（schema 看不出）／Parquet 值级标识符／正常汇总表不误报／注册后被改写的 source 被拒／收据不回显原值
- 数值 scope：审查方描述的确切场景（主模型 step 无 Brier、敏感性 step 有且数值吻合）／未加脚注仍回落／脚注 step 拥有指标时正常收窄／CI 回落已删除（源码断言）
- human review：`resume_scope` / `resume_pid` / `resumable_here`／异进程 pause 被识别／新实例 resume 报错里点名边界／证据读不出时 fail-close
- 窗口契约：异名 `dur_var`（审查方原例：`duration_hours` × `duration_minutes`）／`index_var` 不一致／`id_vars` 不一致／一致时照常合并／协变量表仍不参与
- 外发传输：真适配器上失败被记为 failed／两条分支都关环（源码断言）／policy 级 completed／收据区分状态／未知 outcome 被拒
- MCP 值校验：路径只留文件名／标识符 token 替换／小 `duplicate_count` 报上界／大 count 原样／嵌套结构 withheld

### 上一轮测试的一处自我修正

`test_review_20260727_agent_fixes.py` 的 `_record()` 辅助函数用 `"e"*64` 当摘要——一个真实 EvidenceStore 永远不会产生的值。这一轮加了「读文件前复算 SHA」之后它全线失败，暴露的正是审查方第六轮批评的同一个毛病（测试喂了生产端不会产生的形状）。已改为按文件真实哈希，`sha` 参数保留以便**故意**测试摘要不符的路径。

`test_review_20260725_agent_fixes.py` 的 `_registered_figure()` 同理：原来手写 `{aggregate_only: True}`，现在跑真实审计并注册收据。

## 已知未做

1. MCP：JSON-RPC batch 上限、每客户端并发、pipeline 取消、单工具超时、每用户配额（第五轮起累计）。
2. 跨进程 / 新 Pipeline 实例恢复 human review —— 需要 phase handoff 可重建。本轮把不支持这件事**声明化**了，没有实现它。
3. `pipeline.py` / `execution/phase.py` 拆分。
4. 完整 Agent suite、主 suite、Python 3.10–3.12 CI、packaging CI —— 需要 PR 才能跑，本轮未推送。
5. 渲染位图里的 mark 计数仍未做，收据继续如实写 `mark_count_verified: false`。

## 测试结果

```
tests/research_agent/test_review_20260728_agent_fixes.py            41 passed
第四/五/六/七轮 review suite + MCP + graph 合跑              202 passed, 2 skipped
聚焦回归子集（figure/egress/mcp/graph/visual/manuscript/privacy）  1310 passed, 2 skipped
tests/research_agent 全量（4 chunk × 90 文件，固定顺序）
  chunk_aa  1851 passed,  3 skipped   21:00
  chunk_ab  1455 passed              05:40
  chunk_ac  1781 passed,  1 skipped   16:18
  chunk_ad  1935 passed,  3 skipped   05:27
  ────────────────────────────────────────
  合计      7022 passed,  7 skipped,  0 failed
tests/（不含 research_agent 与 benchmarks）        1502 passed, 54 skipped, 3 failed*
ruff check src tests                                         All checks passed
```

**关于全量跑法的诚实口径**：分 4 个 chunk（每个 90 个文件，各自独立进程）+ `-p no:randomly`。这么跑有两个原因：单进程全量在本机会 OOM 被杀（本轮开头实测在 8% 处被杀），以及 `pytest-randomly` 的随机顺序会触发下面那条既有的 ContextVar 泄漏。**结果是 0 failed、`grep -c "usr/bin/docker"` 也是 0**——但这与「那条泄漏已修」不是一回事：固定顺序 + 分进程本来就会规避它，本轮没有修它，它仍是独立跟进项。

\* 那 3 项在**改动前的 HEAD `6b5815b` 上就是红的**（干净树 `git stash -u` 实测同样 3 failed）：

- `test_arch_measure.py::test_checked_in_architecture_baseline_has_no_regression`
- `test_full_concept_dictionary_audit.py::test_public_database_source_mappings_have_no_structural_errors`
- `test_research_agent_resource_baseline.py::test_checked_in_resource_context_baseline_has_no_drift`

前两项与本轮无关；第三项是 checked-in 基线随 src 变动而漂移的设计使然，与 figure2 scorer-tree digest 同类——重新 lock 应该发生在这一系列外审修复定稿之后，而不是每改一次刷一次。

## 已知红（与本轮无关，沿自上一轮）

- `tests/benchmarks/figure2_canonical9/evaluator/` 的 scorer-tree digest：覆盖 `evaluator/*.py` 加核心 src，任何 src 改动都要重新 lock，且没有 CI leg 在跑它。
- `tests/research_agent` 在**单进程 + 随机顺序**下可能出现约 44 项 `FileNotFoundError: '/usr/bin/docker'` 失败：`execution/method_capabilities.py` 的 `_RUNTIME_SNAPSHOT_PROVIDER` 是个 ContextVar，`set()` 后丢弃 token 从不复原，加上 `pytest-randomly` 导致顺序随机。本轮的 4-chunk 固定顺序跑法没有触发它（0 项），但那是规避不是修复；已单独开出跟进项，不在本轮范围。判别方法：`grep -c "usr/bin/docker" <log>` 等于失败数即是它。
