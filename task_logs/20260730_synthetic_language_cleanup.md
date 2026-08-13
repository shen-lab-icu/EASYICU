# WebApp QA — 合成数据用户文案清理

- 日期：2026-07-30
- 分支：`codex/web-copilot-cockpit-lite-20260729`
- 范围：Patient Review、Cohort Statistics、Cross-DB 的合成数据入口与提示文案

## 变更

- 将用户可见的“合成兜底 / 离线兜底 / synthetic fallback”统一改为“合成演示 / 合成数据”。
- 保留必要边界：合成数据不含真实记录，不得用于临床推断或稿件结果。
- 仅修改既有页面所有者文件；未新增 CSS、未改内部 `fallback` 控制契约。
- 更新静态资源缓存版本，并补充中英文旧词缺席回归。

## 验证

- `node --check`：4 个变更 JS owner 均通过。
- 聚焦 Python/Node 回归：两组共 14 个 pytest 用例通过；两个可执行 owner contract 通过。
- 浏览器 Patient Review：
  - 入口显示“合成演示 / 加载合成演示”；
  - 加载后显示“合成审阅已就绪 / 临床约束合成数据”；
  - Data Quality 说明显示“覆盖率和质量指标只描述临床约束合成数据”；
  - 页面可见文本不含“兜底”。
- 浏览器 Cohort Statistics：显示“合成数据对照 · 仅用于界面演练”，页面可见文本不含“兜底”。
- 1280×720 桌面视口：document/main 横向溢出均为 `0`；控制台 error 为 `0`。

## 证据

- 截图：`output/ui-qa/20260730_synthetic_copy/patient_synthetic_copy.png`
- 代码 owner：
  - `src/easyicu/webserver/static/js/screens-viz.js`
  - `src/easyicu/webserver/static/js/screens-viz-patient-demo-sources.js`
  - `src/easyicu/webserver/static/js/screens-viz-patient-quality.js`
  - `src/easyicu/webserver/static/js/screens-viz-crossdb-source.js`
