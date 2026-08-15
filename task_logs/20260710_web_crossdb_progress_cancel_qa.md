# Cross-DB 逐库进度、协作取消与来源动线 QA

- 日期：2026-07-10
- 模块 / phase：`web` / `WEBAPP-FASTAPI-NATIVE-QA` · `PATIENT-CROSSDB-VISUAL-PARITY`
- 分支：`fix/easyicu-concept-bounds-enforcement`
- 提交：`b27fd74`（后端）+ `e220715`（前端）
- 基线：`f5e37dd`

## 目标

修复真实 Cross-DB raw job 长时间停在 `0/6`、取消只能在整段读取结束后生效、断线重放可能倒退、来源选择需要跳回其他页面，以及取消失败无可见反馈的问题。保持 local-only、aggregate-only 和 bounded-read 边界，不扩大首次运行的 12 个核心概念范围。

## 已实现

### 后端

- `Job` 的事件、取消与终态统一在单实例 `RLock` 下原子更新；`accepted=true` 的取消不会再竞争成 `done`。
- SSE 通过同一锁内的 `events_since()` 取得事件切片与状态；终态后不接受迟到 progress，`end` 保持唯一且为最后事件。
- raw loader 在数据库、24-concept chunk、fallback concept 前后检查取消；当前有界读取返回后停止，不强杀 Python 线程。
- progress 只发送数据库、chunk、计数和状态等聚合字段，不发送路径、患者标识符、DataFrame 行或 value 数组。
- 显式请求的任一数据库目录缺失、或真实读取出现 operational error 时 fail closed；root scan 的 `runnable` 与运行端契约一致。

### 前端

- 新增 route-owned `screens-viz-crossdb-progress.js`，持有 job/progress/cancel/database-row 状态和可访问渲染；`crossdb.css` 持有对应样式。
- 每库/每块进度、原生 `<progress>`、取消中状态和数据库 ledger 可见；取消按钮重绘后保持焦点。
- 取消 API 缺失/拒绝时恢复原进度，并在 loading card 内显示 `role=alert` 的错误，而不是静默恢复按钮。
- continuity owner 增加 seq watermark、cancel fence、same-job stale stream fence 和 `500/1000/2000/5000 ms` 退避；历史 replay 不再误重置退避。
- Source A 在 Cross-DB 页面内直接列出、选择、添加和刷新 registered exports；Source B 的数据库卡改为真实按钮并暴露 `aria-pressed`。
- raw folder scan、selected database identities 与 terminal result 都按当前 source identity fail closed；raw 成功载入后不再提供无效的同配置 rerun。

## 自动化验证

- 修改域集成门：`256 passed, 1 warning in 75.12s`。
- Ruff：修改的 Python owner/tests 全过。
- Node：4 个前端 owner/shell syntax 全过；progress、continuity、source-choice 三个可执行行为合约全过。
- owner 回归：预期 selector 只存在于 `crossdb.css` / progress owner；Patient、Cohort、Agent、Extraction、Guided、Ideas、Settings owner 中不存在 Cross-DB progress selector。
- `git diff --check` 通过；CSS/JS brace 与 route-purity 回归通过。
- 未重复全仓 3,000+ 测试；按修改依赖图执行了 jobs/security/routes/static/workspace/Cross-DB 共 256 项。

## 真实桌面浏览器 QA

- 视口：1280×720（桌面/笔记本范围）。
- 数据根：`/Volumes/外置硬盘/databases`；选择 MIMIC-IV + eICU，Quick preview，12 个核心概念。
- folder scan：6 个目录识别，2/2 selected recognized，运行按钮只在契约满足时启用。
- active progress：可见 `MIMIC-IV · chunk 1/1`、`0/2`、eICU pending；数据库状态按真实事件推进。
- cancel：按钮点击后仍为 active element，文本 `Cancel requested`，`aria-disabled=true`；MIMIC-IV complete、eICU stopping；随后终态明确显示 cancelled，没有写入成功 workspace。
- overflow/clipping：document `1280 == 1280`；active card `990 == 990`，database rows `313 == 313`，均无横向或内部溢出。
- console：只有 EasyICU hydration info，0 error / 0 warning。

## 截图证据

- 旧来源动线：`task_logs/2026-07-10_web_crossdb_progress_qa/assets/00-before-real-setup.png`
- Source A 页面内来源选择：`task_logs/2026-07-10_web_crossdb_progress_qa/assets/01-after-source-choice.png`
- 逐库/逐块 active progress：`task_logs/2026-07-10_web_crossdb_progress_qa/assets/02-active-progress.png`
- cancel requested + focus-preserving state：`task_logs/2026-07-10_web_crossdb_progress_qa/assets/03-cancel-requested.png`
- cancelled terminal：`task_logs/2026-07-10_web_crossdb_progress_qa/assets/04-cancelled-terminal.png`

## 边界与下一结构债务

- 协作取消不能中断正在执行的一次底层 DuckDB/CSV read；它保证该次有界读取返回后不再进入下一 chunk/数据库。
- `screens-viz.js` 仍为 5,826 行。progress、continuity、source-choice 已有显式 owner；下一次结构性改动应把 Cross-DB setup/scan/render orchestration 拆到 `screens-viz-crossdb-setup.js`，并继续通过显式 namespace 共享状态，不能复制闭包函数。
- 本轮未 push，且未触碰 Claude 正在运行的 benchmark 或 `research_output/`。
