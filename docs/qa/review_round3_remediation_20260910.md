# 第三轮独立复核修复回执（2026-09-10）

本轮在 `/Volumes/外置硬盘/GitHub/EASYICU`、`codex/dev9-web-acceptance-20260906` 上实施；基线为 `aa4f3a95f66b9fa0805e111e26c865c04e48253a`。本回执记录提交前的修复与验证，随本轮补丁保存。用户授权直接修复确认的问题，并在独立复核接受后授权提交。没有切换分支、stash、合并 main、重启 8765 宿主或执行真实患者/Provider 任务。

此前的逐项裁定在工作区 `task_logs/20260910_review3.md`。本轮修复 10 项确认的问题；P2-3 冲突体重降级建议不采纳：不同有效体重是冲突证据，不能当作缺测吞掉。这里的关闭范围是列出的代码路径与反例，不能解释为全项目再无缺陷。

## 改动和验收

| 原编号 | 最终行为与证据 |
| --- | --- |
| P1-A resolve-source | 路由与 mining 后端均调用既有 connector 检查。关闭时即使请求 `allow_network=true` 也不取远端元数据，并返回关闭原因；开启路径和本地手填元数据保留。路由/后端 × 开/关矩阵验证，联网函数用 mock 计数。 |
| P1-B writer 回调 | 第四个 STRICT repair 回调绑定本次 `per_step_records`。测试提取 write phase 实际传入的 callback 表达式，用真实 EvidenceStore 和 ManuscriptRepairPass 重放历史证据：旧裸回调认为通过的句子，现在出现在 residual drop 回执，最终当前账本校验通过。该缺陷原本是晚失败和诊断失真，未证明最终发表门可被绕过。 |
| P1-C Analyzer | 删除异常转成功占位文本；异常经现有 coordinator 记录 `execution_raised` 和 error finding，并刷新部分 manifest。硬止损类异常继续向上抛出；串行停止后续步骤，并行阻止尚未开始的排队工作。空白/失败占位响应也拒绝；resume 对旧占位解释验摘要、读内容并拒绝复用。失败封存、恢复和正常完整 Pipeline 分别验证。 |
| P1-D Cross-DB 空白 | 布尔映射中的空白变为缺测，不再触发 KeyError。真实 FastAPI TestClient 请求验证 0/1、yes/no、true/false 混空白/缺测仍返回完整比较；缺测不当作 0。 |
| P2-1 少尿证据对齐 | `oliguria_gt6h` 使用现有严格 Series/index 校验；错误索引或非 Series 显式失败。缺失和未知字符串不转换为阳性。未修改临床阈值、持续时长算法或 KDIGO。 |
| P2-2 缓存索引 | 同一个持久索引按 root 分存回执，兼容旧 schema 1，写出 schema 2。A→A→B→A 只在首次读 A/B 时算文件哈希；未变索引不重写。保留目录遍历和 stat 新鲜度检查；同大小、同 mtime 的替换文件仍使缓存失效。未引入无失效条件的内存缓存，也不声称消除了全部 I/O 或锁开销。 |
| P2-4 图件文案 | 修正降级消息：最终图件/来源绑定审计仍适用；display-suite 的面板数和角色多样性仅是设计建议。没有修改评分或放宽图件门。 |
| P2-5 区间标签 | 检测显式 alpha、formula/discrete 子模块与别名。确认默认 95% 的普通 Logit 家族才允许既有字面标签修复；非默认/未知 alpha、OLS、显式 use_t 等仅报错，不臆造 Wald 95%。既有坐标绑定修复与幂等性测试保留。 |
| P2-6 Cross-DB 读取 | CSV/Parquet/XLSX 投影分批读取，附加行/单元格/字节预算；超限明确失败。来源读取失败返回来源和错误类型，不泄漏原异常路径，也不跳过坏源返回完整成功。同步摘要有 deadline 和并发槽限制；异步既有 checkpoint 贯穿特征批次。Excel 仍取第一个工作表，独立于保存时的活动页。 |
| P2-7 table 包装 | 修正不存在的相对导入；公开 `new_src_tbl`、列/源配置、时间取整均实际调用验证。table 局部 ID-map helper 委托既有数据加载 owner，要求可读取的 ICUDataSource，保留一对多关系和指定伴生列。顶层已明确退役的 `easyicu.id_map_helper` 不恢复为另一个竞争入口。 |
| P2-3 保持 | `assess_urine_windows` 遇冲突有效 keyed weight 仍抛 `ConflictingKeyedWeightError`。独立合成探针验证；拒绝把此错误当作“缺少体重”吞掉。 |

## 资源和停止边界

Cross-DB 每文件预算为最多 1,000,000 行、2,000,000 个投影单元格，文件大小及累计 DataFrame 深层内存各不超过 128 MiB，批次最多 10,000 行。读取完整才返回分布；这不是抽取前 N 行的近似统计。同步摘要默认 120 秒，可在既有 0.05–600 秒范围指定期限，最多两个同步后台工作槽。超大导出现在会收到明确的预算错误，不能把这个行为理解为支持任意规模导出。

期限约束请求等待时间，不能强制打断操作系统中已经阻塞的文件读取；后台任务在下一个检查点退出，其槽位在实际退出前不释放。预算是投影数据的检查，不是操作系统级 RSS 硬限额；批次解码和 concat 可有额外瞬时开销。

Provider 全局持久账本原本已有支出限制，本修复不声称发现或修补了无限费用漏洞。硬止损传播后不再继续排队，但已经运行的并行工作仍需到自己的停止边界。普通 Analyzer 异常按既有失败 owner 封存，不假造成功解释。

## 验证记录

级别为 **combined engineering validation**，不是 full exact-head CI。命令使用项目 `.venv/bin/python`，显式 `PYTHONPATH=/Volumes/外置硬盘/GitHub/EASYICU/src:/tmp/easyicu-review-plotly`。后一个路径仅补本机缺少的 Plotly 依赖；未重装或改写虚拟环境。测试使用临时合成源和模拟 LLM。

原始输出保存在工作区 `outputs/easyicu-review-independent-20260910/`：

| 记录 | 结果 |
| --- | --- |
| `round3_repair_core.txt` | `tests/core`：2238 passed、54 skipped、1 deselected，83.30 秒。默认 slow/真实数据选择规则仍适用。 |
| `round3_repair_agent.txt` | Analyzer、调度、Provider、writer、区间门、恢复等 15 个文件：367 passed、1 failed、6 deselected，40.82 秒。失败对照见下。 |
| `round3_repair_web.txt` | Web/Cross-DB/文献相关 9 个文件：308 passed，26.42 秒。 |
| `round3_repair_synthetic_pipeline.txt` | 显式 `-m ''` 执行 `test_pipeline_end_to_end_synthetic_cohort`：1 passed，158.61 秒。不是完整慢集成套件。 |
| `round3_repair_final_adjacent.txt` | Excel 首表修正、Analyzer 空白响应新增测试及相邻路径：197 passed，29.52 秒。与上方有重叠，不相加充当独立总数。 |
| `round3_repair_crossdb_final.txt` | 最后确认超时测试会等待后台线程释放槽位，并复验修改函数格式化后行为：10 passed，4.38 秒。 |
| `round3_repair_governance.txt` | 167 passed、7 failed、2 skipped、2 deselected。首次包含本轮调度器新增 LOC；经限定基线登记后，架构检查见下一行。 |
| `round3_repair_arch_final.txt` | 34 passed、1 failed；恢复为原有 17 项架构漂移，本轮新增 LOC 没有混入历史欠账。 |

本轮增加的断言覆盖：双层出网门、当前账本 repair 残留、Analyzer 普通失败/硬停止/空响应/旧检查点、严格索引、未知布尔证据、root 回执迁移与内容替换、区间 alpha/别名/参考分布、6 类 table 包装调用、Cross-DB 实际 HTTP、三种文件格式预算、来源错误、同步期限、批次取消和 Excel 首表。

自审逐文件检查了异常传播、历史证据复用、完整源统计、时间/ID 契约及当前进程状态。修改 Python 文件 Ruff 和 `git diff --check` 通过；最后只对 Cross-DB 改动函数格式化，前后 AST 一致。

## 既有失败与基线登记

Agent 唯一失败是 `test_cohort_translation_budget_owner_is_structural_not_prose_routed`：旧 `SimpleNamespace` 测试替身缺少 `icu_rule_refs`，在既有 `has_scientific_runtime_owner` 中报 AttributeError。用 `round3_head_source_overlay.py` 加载 HEAD 版全部已改生产模块后，同一测试、同一调用栈与异常复现，见 `round3_repair_preexisting_agent_head.txt`。没有为了变绿而放宽产品类型契约。

治理 7 项仍为架构漂移、desktop 断言、未登记顶层模块、模块图、资源上下文基线、大测试文件增长、gitignore 贡献文件断言。用 HEAD 的导入源码及 `Path.read_text/read_bytes` 文件读取覆盖重放这 7 项，原始结果见 `round3_repair_preexisting_governance_head.txt`。该对照不修改工作树，也不是新 checkout 的完整 CI。

机器对照 `round3_repair_failure_comparison.json` 确认 7 个治理失败 nodeid 和最终 17 个架构超限差值与 HEAD 一致；没有仅凭失败数量相同就认定细节不变。

`execution_phase.json` 仅登记本轮 coordinator +22 LOC、Analyzer +3 LOC；历史中的前后源码 SHA 和变更理由一并保留。Analyzer 原有 81 行超限及其他 17 项架构问题不被整体刷新掩盖；基线里旧 SHA 仍属原历史快照，本轮精确源码 SHA 在追加历史与交付清单中。

## 交付状态

用户转述 DeepSeek 独立复核接受全部 10 项修复、认可保留冲突体重错误。之后再次核对原交付 26 个文件 SHA 全部一致，未发现新增阻断。复核报告的 `2236 core passed` 与本方 `2238 passed` 尚缺原始 nodeid 完全对账：裸环境 import 失败已单独复现，另一项差异保留为未核销记录，不笼统归因于 slow 规则。

用户已明确授权“没有问题就提交，有问题修复后提交”。提交前仅更新本回执的复核和提交状态，生产源码与已验证版本一致；按当前开发分支执行本地提交，不推送或部署。原验证时的文件 SHA、补丁、检查命令和自审摘要在 `round3_repair_delivery.json` 与 `round3_repair.patch`；提交范围检查及最终 SHA 另记 `round3_commit_scope.json`、`round3_commit_receipt.json`。精确版本以包含本回执的提交为准。

既有 E1/E2、论文结果和全量真实数据未重新验证，研究/发表状态不升级。剩余治理与旧测试契约问题有明确记录，不能把本回执当作全库 CI 已全绿。
