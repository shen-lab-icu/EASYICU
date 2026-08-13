# 20260712 Web 端 UX 动线修复（13 项路线图逐项落地）

> 依据：`task_logs/20260711_web_ux_flow_review.md`（8 维度 86 条评审）。全部在主会话内小步完成，未使用多 agent workflow。

## 修复清单（编号对应评审报告路线图）

### 快修组
1. **dock CTX 缺口 + 后端重复 chip** — `copilot-dock.js` CTX 补 `ideas`/`dictionary`/`guided` 条目（含 tiers/ideanext/concept 三个新 @say 答案）；`copilot_sessions.py::_chips_for_route` 修复未知路由 `common[:1] + common` 造成的「开始研究引导 ×2」重复 chip，补 ideas/dictionary 的 intro、chips 和 `open_ideas`/`open_dictionary` navigate 动作。实测 #ideas 上下文显示「当前页面：想法挖掘」、chips 无重复。
2. **Start demo 不切模式** — `screens-help.js` 两个按钮改 `data-help-startdemo`，经 `setDataMode('demo', {onapply})` 切换后才导航（`i18n.js` setDataMode 新增 onapply 回调；confirm 被取消则不导航）。实测 Real→点击→demo+#extraction。
3. **装饰性开关** — 删除高级导出里三个只翻样式的 switch（按患者列表筛选/行索引/时间戳文件名）及两个 display-only 处理器；保留真实的 Merge mode。
4. **取消 ≠ 失败** — `screens-extraction.js` 新增 `exportCancelled` 终态 + `cancelledState()` 中性卡片，如实列出已写入的 N/M 个模块文件与 out_dir（后端 cancelled result 本就返回 files）；不再渲染红色「抽取失败」。
5. **错误信息人话优先** — `api.js` postJSON/postBlob 改用 `apiError()`：4xx 优先展示后端 `detail.reason`（人话），机器码与 `path -> HTTP n` 移到 `err.technical`/`err.code`；5xx 保留技术前缀。全局生效（ideas PDF 错误等）。
6. **Agent guide 重复按钮** — 删除 `screens-agent.js` actionHtml 里与顶栏 Page guide 打开同一 dock 的「代理指南」按钮；更新 `test_webserver_static_routes.py`/`test_webserver_ux_reliability.py` 契约为「不得再有 per-screen data-cpopen」。

### 中修组
7. **命名统一** — EN 一律 "Guided Copilot"、zh 一律「研究引导」：侧边栏/首页/共享提示/extraction与agent交叉链接/help 的 "Guided study" 全部改名；guided 屏 gd-name 改 `t('Guided Copilot','研究引导')`，副标题改「对话式研究规划」，「引导式 Copilot」第三变体清除；dock 全部 zh chips/文案、app.js 交接横幅 zh 改「研究引导」。测试断言同步（6 处）。
8. **Real 采样明示** — SSE `start` 事件的 cohort_report 存入 `exportCohortReport`；`cohortScaleNote()` 在运行卡+完成卡显示「已从 M 条匹配住院中采样 N 条（采样上限）」/「完整匹配队列」；推荐卡 Real 模式第 4 格从 "Source · local" 改为 `≤N stays (sample cap)`。
9. **Real 空态 zh + 出口按钮** — `screens-viz.js` patientSourceReadyCard 全部 t() 化并加「打开数据抽取」按钮；`screens-agent-render.js` 新增 `gateCheckLabel()` 双语映射 6 个 evidence-gate check id，签署清单不再英文裸奔。
10. **review_blocked 逐条 check** — `screens-guided.js` 新增 `guidedGateCheckLabel/Rows/FailedNames`：阻断时分析卡渲染逐条 check（通过=绿/待签署=灰/失败=红标「未通过」，guided.css 加 `.gd-task.failed`），聊天消息列出失败 check 名称；空 checks 数组按 fail-closed 显示。

### 结构组
11. **后台重渲染守卫 + IME + 诚实确认** — `screens-extraction.js` 三个 continuity 回调改 `backgroundRepaint()`（仅 #extraction/#icd 时重渲染）；`screens-viz.js` `repaintScreen(id)` 增加 activeVizRoute() 守卫（含 #audit/#sofareclass/#icd 别名）；`screens-dict.js` 搜索输入节点保持稳定、只重绘 `#dictResults` 区域（实测 sameNode=true、焦点保留、结果更新——修复中文 IME 被打断）；`i18n.js` setDataMode confirm 在有运行中任务时如实追加「跨库扫描会被取消/抽取任务将不再被跟踪」，且有运行任务时即使 EU_HASWORK=false 也弹确认。
12. **Copilot→classic 交接带 config** — `screens-guided.js` data-open 派发器在目标为 extraction 时设置 EU_GUIDED_HANDOFF（prefill hints + 机器可读 config：cohort_preset/modules/format/export_dir/max_patients/window）；`screens-extraction.js` `applyGuidedPrefill()` 把此前被 take() 后丢弃的 handoff 真正消费进表单状态（exPath 按其声明约定仍不预填，扫描仍是权威绑定）。
13. **Agent Plan 卡去 fixture** — `screens-agent.js` planList()：seed 计划优先；Real 模式无 seed 渲染真实 preflight 步骤（导出快照/队列质量摘要/有界产物/证据核验与签署）；demo fixture 标注「演示示例计划」；状态 pill 从计划数组计算，删除硬编码 "5 ready · 1 needs review"。实测 demo=「计划 · 6 步 · 演示示例计划」+「5 就绪 · 1 待核验」（计算值恰同），seed 项目=「7 就绪」。

14. **demo 完成页谎称写盘**（评审 onboarding P1，路线图外补做）— `doneState()` 在无真实 result（demo 种子路径）时改为如实声明「演示种子预览 —— 没有向磁盘写入任何文件」，真实 run 文案不变。实测 demo 运行后完成卡显示新文案。

## 验证
- 前端契约/行为测试：`pytest -k webserver` **369 passed, 1 skipped**（含更新后的命名/按钮/版本锁断言）；page-guide 后端 11 passed。
- 全部 40 个 static/js 文件 `node --check` 通过；`ruff check copilot_sessions.py` 通过；`git diff --check` 干净。
- 浏览器实测（1280×720，demo+real、EN+zh）：dock 上下文/chips、guided ⌘K 聚焦 composer、字典搜索节点稳定、Start demo 切模式、Agent 无重复按钮、Plan 卡三种来源、10 条路由循环 0 console error、0 横向溢出。
- `index.html` 12 个改动文件缓存版本 bump 至 `?v=20260712-ux-fixes`，版本锁测试同步。

## 未做（评审中的其余项）
- P2/P3 长尾（约 60 条）仍在 `task_logs/20260711_web_ux_flow_review.md` 待排期；较大的未做项：shell 级运行中任务 chip、Ideas 可行性 tier 术语统一渲染器、Cross-DB 来源层级调整、guided demo 演练双语化、frontier 空态/失败态区分、entity 导航临床筛选。
