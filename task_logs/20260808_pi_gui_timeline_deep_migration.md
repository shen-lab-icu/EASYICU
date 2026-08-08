# Guided Pi · pi-gui/Codex 时间线深度迁移

日期：2026-08-08

分支：`feat/pi-copilot-shell`

任务：把手写的 Agent 活动卡片替换成与 Codex / pi-gui 同构的真实对话时间线，并用真实模型回合做同屏对比。

## 来源与许可

- 源码参考：`minghinmatthewlam/pi-gui@eb9a7380705dffad36db3efa771ee825aafbef6f`。
- 深读文件：`apps/desktop/src/timeline-item.tsx`、`apps/desktop/src/styles/timeline.css`、`apps/desktop/src/conversation-timeline.tsx`、`apps/desktop/src/icons.tsx`。
- 迁移的是组件结构与交互模式：768px 阅读列、无卡片 activity/tool 行、旋转 disclosure、状态 pip、耗时 divider、仅展开体有边框。
- EasyICU 保留自己的 Pi event projection、PHI-safe receipt、stable code、owner、一次性授权和 scientific authority；没有迁移通用文件/bash 工具。
- MIT notice 已写入 `src/easyicu/webserver/static/THIRD_PARTY_NOTICES.md` 并纳入 wheel package data。

## 生产修改

- `static/js/screens-guided-pi.js`
  - 使用既有 EasyICU 共享图标目录映射不同工具与生命周期事件；没有新增 emoji、CSS 图形或散落 SVG。
  - activity 从大卡片改成 Codex 风格耗时/工作状态行；折叠详情只列生命周期事实与工具回执，不展示 raw reasoning。
  - tool 从独立卡片改成内联 header；展开后显示摘要、真实 tool 名、stable code、duration 和 owner。
  - 用户消息改为右侧气泡；助手正文改为无卡片阅读列。
  - 历史 transcript 按 user timestamp、tool call/result、final assistant message 重建每轮 activity，刷新或应用重启后耗时轨迹仍存在。
- `static/css/guided-pi.css`
  - owner 内完成全部样式；未污染 app/redesign/tweaks/guided catch-all。
  - reading measure `768px`，正文/输入 `15px`；running/success/error 同时有文本和状态 pip，避免只靠颜色。
  - `:focus-visible`、reduced-motion、receipt max-height/overflow 和长字段换行均保留。
- `static/index.html`
  - cache key 更新为 `20260808-pi-timeline3`。

## 真实浏览器 canary

测试项目：`New local study`；模型：`gpt-5.6-luna`；未勾选 Configure / Run / Cancel。

用户消息：

> 请只使用 EasyICU 的只读工具检查当前研究配置和工作区状态，然后用两句话总结；不要修改配置，也不要启动运行。

真实可见结果：

- 耗时轨迹：`耗时 7.0 秒`（重新从 transcript 恢复后的口径）。
- 工具 1：`easyicu_workspace_status` → `easyicu_workspace_status_ready`，owner `easyicu.webserver.study_contexts`。
- 工具 2：`easyicu_inspect_context` → `study_context_projected`。
- 最终回复正确描述当前 StudyContext revision 1、缺失槽位和无活动任务/运行，并明确未修改或启动。
- 浏览器 refresh 后恢复完整用户消息、耗时、两条工具与回复。
- Uvicorn + Pi sidecar 完整重启后再次恢复同一会话与时间线，证明不是进程内假状态。

## 视觉与交互证据

- Codex 完整对话参考：`task_logs/browser_audit_20260808_pi_timeline/reference_codex_thread.png`
- Codex 工具轨迹参考：`task_logs/browser_audit_20260808_pi_timeline/reference_codex_tool_timeline.png`
- EasyICU 完成态：`task_logs/browser_audit_20260808_pi_timeline/easyicu_pi_timeline_complete.png`
- EasyICU 工具展开态：`task_logs/browser_audit_20260808_pi_timeline/easyicu_pi_tool_expanded.png`
- 同屏归一化比较：`task_logs/browser_audit_20260808_pi_timeline/comparison.png`
- QA 报告：`design-qa.md`，`final result: passed`。

默认桌面视口 `1572 × 1354`：

- document `scrollWidth = clientWidth = 1572`；横向 overflow false。
- 三条 activity/tool summary 均 `scrollWidth = clientWidth = 768`。
- composer bottom = viewport bottom = `1354`；transcript 是唯一纵向滚动 owner。
- 完成态与展开态均无 console error。

## 回归门

```text
node --check src/easyicu/webserver/static/js/screens-guided-pi.js
python -m pytest -q tests/test_pi_copilot_static.py tests/test_webserver_static_routes.py
79 passed, 1 warning
```

静态合同同时锁定：owner presence/foreign absence、CSS brace/comment、bottom composer、15px copy、pi-gui attribution/package data、历史 activity 重建和 JS syntax。
