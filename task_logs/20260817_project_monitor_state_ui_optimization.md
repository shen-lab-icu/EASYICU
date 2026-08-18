# Project Monitor 状态 UI 优化

- 日期：2026-08-17
- 模块：web
- 任务：WEBAPP-FASTAPI-NATIVE-QA
- 工作树：`/Users/haibo/Documents/GitHub/EASYICU-figure2-integration`
- 分支 / 当前基线：`integration/figure2-e1-h3-20260816@0d2434f`（本轮开始时为 `c79bb20`；并行 Planner 任务仅提交了 `progressive_compiler.py` 及其测试，未触碰本轮 Web owner）
- 状态：源码与浏览器 QA 完成；本日志随 Copilot / Project Monitor 边界提交归档

## 问题

Project Monitor 在本地项目索引请求失败时会同时显示：

1. 左侧“Local research projects unavailable”；
2. 左侧“No monitored projects yet”；
3. 右侧“No project selected”；
4. 无条件渲染的项目标题、Plan 流程条与 Tabs。

因此同一页面同时表达“加载失败”“成功但为空”和“已有项目正在 Plan”三种互斥状态。流程条还使用固定最小宽度加 `overflow-x:auto`，在笔记本内容宽度下出现内部横向滚动条。

## 修复

- 新增单一 `monitorViewState`，状态只允许 `loading / error / empty / ready` 四选一。
- `loading / error / empty` 不再渲染项目详情标题、流程条和 Tabs。
- 错误态只保留一个 Retry 主操作和一个 Open Guided Copilot 次操作。
- 成功空态只保留一个 Open Guided Copilot 主操作和一个 View completed example 次操作，不再重复 Refresh。
- 左侧项目索引只显示紧凑状态摘要，不再重复主操作。
- 应用侧栏在失败时显示未知计数 `—`，成功空列表才显示 `0`；不再把索引不可用误报成“确定无项目”。
- 已选项目的五步流程条在桌面端自适应收缩；1280px 下隐藏次要描述，标题按词换行，不再出现内部滚动条。
- 样式放在 Project Monitor 的 `agent-layout.css` owner；没有写入 catch-all CSS/JS。

## 直接验证

```text
89 passed, 5 warnings
```

命令范围：

- `tests/test_webserver_static_routes.py`
- `tests/test_static_frontend_ownership.py`
- `node --check src/easyicu/webserver/static/js/screens-agent.js`
- `git diff --check`

CSS owner / 结构扫描：

```text
agent.css braces=103/103 comments=7/7
agent-layout.css braces=41/41 comments=3/3
agent-layout.css foreign-route selectors=0
```

## 浏览器 QA

隔离服务器：`127.0.0.1:8898`，隔离 `EASYICU_HOME=/tmp/easyicu-ui-qa.uXqrAM`；QA 后服务器和 Playwright session 均已停止。

- populated，1280×720：`pageOverflow=0`，`pipelineOverflow=0`，正常页面 console 0 warning / 0 error。
- forced error，1280×720：`monitorState=error`，pipeline/header/tabs 均为 0，Retry=1，页面溢出 0；侧栏不再显示假 `0` 或假 active project。模拟 503 产生的浏览器网络 error 是测试注入的预期信号。
- successful empty，1280×720 / 1600×900：pipeline/header/tabs 均为 0，Guided 主操作=1，Refresh=0，内部项目栏按钮=0，页面溢出 0。
- 中文 successful empty，1280×720：页面溢出 0，文案和按钮未挤压。

截图：

- `output/playwright/project-monitor-populated-1280x720.png`
- `output/playwright/project-monitor-error-1280x720.png`
- `output/playwright/project-monitor-empty-final-1600x900.png`
- `output/playwright/project-monitor-empty-zh-1280x720.png`

## 边界

本轮只优化 `#agent` Project Monitor 的状态展示与布局；没有把需求收集、provider/model 配置或运行发起移回 Project Monitor，也没有启动 Web E1 或科学分析。
