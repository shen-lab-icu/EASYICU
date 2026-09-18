# 2026-09-10 审阅问题修复回执

用户授权在当前 EasyICU 开发目录直接接手修复已确认问题，并保留无问题的行为，随后授权本地提交。基线为 `codex/dev9-web-acceptance-20260906@efa6a4c6d23c51a458c307291a9249be33662b9c`。保留并复验接手前已有的时间对齐补丁；未切分支、stash、合并 main、部署或启动研究运行。提交用于保存本轮开发修复，独立复核仍待执行。

本回执是源码与合成测试的工程证据。它不证明历史 E1/E2、现有导出或论文产物正确，也不授予重新提取患者数据、Provider、临床发布或投稿权限。

## 逐项处理

| 审阅项 | 已安装的行为与证据 |
|---|---|
| 时间对齐缓存 | 保留前两轮补丁：冷/热缓存一致；必须对齐的 datetime 在前置映射与主块失败时抛出 ConceptError；numeric/特定 timedelta 错误回退保留原契约。回归位于 `tests/core/test_cross_layer_safety_regressions.py`。 |
| P2-17 timedelta 正常加载 | 已是相对时长的轴显式换算为小时，避免通过 datetime 解析而变为 NaN；覆盖三种 ID 布局。 |
| SOFA-2 时长 | 新 `scores/urine_windows.py` 发布与同一窗口绑定的评估率、覆盖与原因；描述性旧 uo 列保留。两处 4 分入口消费同一严格 >6 h 区间证据。详见下节。 |
| DuckDB ID SQL | 平面与分片读取两条路径均引用标识符、编码值；字符串单值不拆字符。引号、字符串列表和注入样式输入测试通过。仍保留加载层二次过滤。 |
| PubMed 门控 | 路由与后端发现、查重、文献详情复用 connector 开关检查；关闭时拒绝联网并记录工具事件。测试确认关闭详情入口时零 HTTP 请求。 |
| 图件面板静默失败 | 选源 normalise、分层与缺失面板失败均显式阻止完整图件成功交付；无可选记录仍可合法省略。 |
| 并行崩溃 | Future 绑定实际 step；与串行复用终态记录、error finding、部分 manifest 刷新。已提交的独立任务可完成，不声称其他 worker 从未执行。 |
| P2-1 WBC | 复用入院时间/单位 owner，按同患者同小时匹配；删除数值大小猜单位、无界最近匹配与患者均值回退。远期 WBC 不作分母；未匹配保留 NaN 与原因。 |
| P2-3 持久缓存身份 | 键加入源位置、配置与文件内容指纹；换目录或内容重写失效；无法证明稳定内容的 callable loader 不复用磁盘结果。共享指纹下沉到既有 `content_identity.py`，API 原导出保持。 |
| P2-4 跨站空 POST 风险 | loopback 检查后校验 Origin/Sec-Fetch-Site；跨站写请求 403。同源浏览器与无 Origin 的本地程序仍允许。 |
| P2-5 摘要缓存竞态 | 查询复制、写入复制和淘汰均受同一 RLock 保护；真实聚合入口并发 12 来源与返回对象隔离测试通过。 |
| P2-7 offline 名称 | 只有注册的完整 mock/offline 名称可免网络 opt-in；不再以子串认定任意模型为离线。 |
| P2-8 AST 失败 | 无法解析的源码产生 error finding；保留既有 Python 3.11/PEP 701 专用修复提示。机械检查与 plausibility 检查不能把解析失败当作通过。 |
| P2-9 STRICT 账本 | manuscript 预检查接收当前步骤账本；数字引用、科学声明及方法事实与最终 binder 使用相同当前证据范围。历史记录仍存档，不等于当前授权。 |
| P2-11 robustness 截断 | 展示全部有效变体，删除固定前 10 行截断；12 变体加主结果的真实渲染/CSV测试覆盖。 |
| P2-12 图件 CSV | 必需源 CSV 写失败抛出错误；不继续交付表面完整的图件。register_file 原本会传播失败，未按误报改写。 |
| P2-13 caption 隐私 | reader_caption 纳入图件文字隐私扫描。 |
| P2-14 方案一致率 | 每个字段显式包含全部后端；缺失字段不再因分母遗漏而得到虚高一致率。 |
| P2-15 discovery gate 读取 | 已存在的 gate CSV 或 step_summary JSON 损坏时明确失败，不当作未发现阻断条件。 |
| P2-16 患者报告导出 | `render_patient_report` 默认先选择患者再移除标识符，匿名标题/随机文件名；显式 `include_identifiers=True` 才保留 ID。Plotly 真正生成 HTML 并核验单患者选择及无 ID。局部工作台显示仍有合法识别用途。 |

以下三项保持：AUMC LOS 小时除 24 正确；显式选择的本地 run 根未证明越权，不强制改为唯一全局根；SOFT 默认本身不是 bug，受约束 profiles 的 STRICT 配置继续保留。KDIGO 原有完整窗口实现未改动。

## SOFA-2 的可评估范围

临床阈值仍采用 [SOFA-2 原文 Table 2 与脚注 p](https://jamanetwork.com/journals/jama/fullarticle/2840822)。源码修复没有把合成 bin 假设写成数据库实测覆盖事实。

- 事件量路径采用明确的工程估计：每个有值的记录代表以其时间结尾的请求 interval bin。覆盖按实际时间交集计算，重复时间只计一次；缺口、显式 NaN 不获覆盖，真实 0 保留。原因列为 `complete_estimated_bins`。这不证明所有数据库的真实观测时长都恰好等于请求 interval，仍需源级临床核验。
- HiRID rate 按既有“当前率对应前一观测间隔”的契约积分，第一条不凭空获时长；稀疏评估点不等于时长中断。
- 评分值只在完整覆盖、有效体重且体积归属可确定时发布。非整除窗口若切到事件量 bin 中间，标为 `partial_volume_bin` 并不给评估率，不能假设均匀分配。该组合的评分声明为不可评估，不宣称已正确计算任意部分 bin。
- >6 h 判据使用截至当前记录、最短且严格超过 6 h 的完整源区间后缀；尿量、平均率、持续时长来自同一区间。它不是任意连续端点区间搜索，也不是两相邻 6 h 移动平均的代理；恰好 6 h 不满足该判据。
- 正常 `sofa2_renal` 字典并不包含 `rrt_criteria`。因此 >6 h 证据随 `uo_6h` 传递，直接供 scorer 和 RRT callback 共用，避免上游计算后根本到不了评分器。
- `_merge_tables` 明确登记伴生列。`change_interval(row_evidence_columns=...)` 作为完整患者/时间/值记录保留证据；要求重新分箱、补值或存在冲突时抛错，resolver 不再吞掉该错误。覆盖列不能 sum、值 median 后拼成伪证据。
- 直接调用 `sofa2_renal` 的窗口输入也需同窗 covered_h；缺证据仅禁用尿量分支。肌酐、实际 RRT、合法 episode 与原 rate+duration API 保留。

回归包含：稀疏历史行数足够但目标窗不满；旧行跨长缺口；重复记录；NaN 与零；非整除 interval；两个低移动均值但并集率不低；HiRID 0/6/8 h 区间；两处 4 分入口；真实 resolver 两患者的冷/热缓存、回调、重采样、合并和评分。

## 验证与边界

验证等级：**combined**，开发默认排除的 slow tests 另行说明，不称 full exact-head CI。原始命令、输出及文件 SHA 在工作区 `outputs/easyicu-review-independent-20260910/repair-*`。核心测试使用显式绝对 `PYTHONPATH`，未依赖迁移前 editable 路径。Plotly 未装在项目 venv；仅为导出验证安装到 `/tmp/easyicu-review-plotly`，未改运行服务环境。

- `repair-core-final.txt`：**2222 passed / 54 skipped / 1 deselected**，155.08 秒；含 KDIGO 与临床 golden fixture。尿量原因文案明确为 estimated 后单独复验 `repair-urine-final.txt`：**16 passed**（含追加的 nullable/Arrow 缺测输入）。
- `repair-agent-web-final.txt`：**542 passed / 279 deselected**，38.77 秒。默认排除 slow pipeline；另见下面专项结果。
- `repair-governance-final.txt`：**167 passed / 7 failed / 2 skipped / 2 deselected**。七类依次为架构旧超额、desktop 静态约定、bibliographic_metadata 顶层登记、module graph、resource/context digest、三个旧超长 Web 测试、gitignore 模板例外。均在接手前 HEAD 对照中已有；架构仍为原 17 个超额，未新增超额条目。
- `repair-pipeline-final.txt`：显式开启 slow 后，完整模拟流水线、STRICT 失败诊断、Writer 失败、clinical skill、hypothesis gate、literature gate、异步入口及取消入口 **8 passed**。整文件扩散执行 605.74 秒后主动收束，第九项执行中被 KeyboardInterrupt 中断；其余未执行，不记为通过。
- `repair-writer-integration-final.txt`：与改动直接相关的 writer digest、证据修复、占位符、失败及 STRICT readiness 专项 **17 passed / 262 deselected**。
- 46 个修改 Python 文件编译和 Ruff 通过；git diff --check 通过；6 个 CURRENT.md lint 通过。

治理检查不会用“全量刷新基线”消除未审阅的历史漂移。并行异常回调已复用同一 owner，解析诊断复用已有 support；不新增大型函数回归。`authority/evidence_store.py` 的必要账本参数/检查增加 8 行，在架构 baseline_history 中只增加这 8 行预算，仍保留此前 52 行超额及其他全部旧阈值。该调整不是接受旧仓库的全部增长。

本轮不会声称“同名失败即细节完全相同”：源码 digest 的变化应当如实记录；已有历史对照输出保留。历史运行/论文结果影响追溯仍未执行。
