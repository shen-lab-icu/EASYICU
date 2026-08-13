# Cohort Statistics 信息架构与生存口径收敛

- 日期：2026-07-30
- 任务：`COHORT-INFORMATION-ARCHITECTURE-COMPACTION`
- 分支：`codex/web-copilot-cockpit-lite-20260729`
- 状态：已由 `a55b3f3` 完整撤回，未合并、未 push

## 撤回记录

- 2026-07-31：按用户要求，提交 `2edbf54` 的 12 个文件变更已通过 `git revert` 完整撤回；本日志仅保留历史审计证据，不代表当前页面状态。
- 恢复后的 Cohort focused 测试为 `14 passed`，Node survival contract、JS syntax、删除 owner 检查与 `git diff --check` 均通过。

## 结论

队列统计默认进入“队列概览”，顶层由五个同权标签收敛为“队列概览 / 分析 / 数据可用性”；特征对比、生存分析和 SOFA 重分层成为“分析”下的二级方法。重复的 Agent 预检、来源状态板、模块 chips、跳转卡、页尾下一步和拦截说明已移除。

队列概览、特征对比、生存曲线和 SOFA 转移矩阵继续复用各自 ECharts owner。数据可用性只保留三个摘要指标和默认折叠的 19 模块明细，不再复制 Patient Review 的逐特征缺失审计。

## 数据边界

- 28 天死亡卡仍显示当前导出的事件率摘要，但不再用 `los_hosp` 冒充 28 天事件/删失时间。
- 生存 owner 只允许同时具有可审计 curve payload 和合格时间合同的结局被选中。
- 当前官方 MIMIC Demo 因而默认选择可审计的院内死亡全队列 KM；28 天 KM 以明确原因 disabled。
- ICU 死亡仅表述为“当前注册导出没有 ICU 专用死亡事件列”，没有扩写成“数据源没有 ICU 死亡”。

## Owner

- 路由信息架构：`src/easyicu/webserver/static/js/screens-viz-cohort-shell.js`
- 路由布局：`src/easyicu/webserver/static/css/cohort.css`
- 生存合同与交互：`src/easyicu/webserver/static/js/screens-viz-cohort-survival.js`
- 生存布局：`src/easyicu/webserver/static/css/cohort-survival.css`
- 路由装配/API state：`src/easyicu/webserver/static/js/screens-viz.js`

## 验证

- Cohort owner/static focused：`17 passed`
- Cohort survival Node contract：`{"comparisonSeries":2,"defaultSeries":1,"repaints":1}`
- 扩展静态路由：`81 passed, 1 deselected`；deselect 为已知 worktree basename 环境断言
- JS syntax：shell / survival / route assembly 全通过
- CSS owner presence/absence、brace/comment scan：通过
- 浏览器：官方 MIMIC-IV Demo 逐页打开队列概览、特征对比、生存分析、SOFA 重分层、数据可用性
- 1280px 桌面视口：document/body 横向 overflow 为 0；展开 19 模块明细后仍无页面级横向溢出
- 生存页面实测：院内死亡为唯一 enabled 结局；28 天死亡显示口径原因且不可选；无失效结局假选中

## 截图

- `output/ui-audit/20260730_cohort_information_architecture/01_overview.png`
- `output/ui-audit/20260730_cohort_information_architecture/02_group_analysis.png`
- `output/ui-audit/20260730_cohort_information_architecture/03_survival.png`
- `output/ui-audit/20260730_cohort_information_architecture/04_data_availability.png`
- `output/ui-audit/20260730_cohort_information_architecture/05_sofa_reclassification.png`

## 保留的下一步

自定义生存分组不能只在前端添加阈值控件。需要后端 row-level owner 提供 `module → feature → rule/threshold → time-zero/baseline/landmark → receipt` 合同后，现有预设才可升级为可审计的自定义分组。
