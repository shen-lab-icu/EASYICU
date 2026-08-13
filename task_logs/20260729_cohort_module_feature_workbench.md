# COHORT-MODULE-FEATURE-WORKBENCH — Cohort 模块化特征选择与有界分组汇总

> 日期：2026-07-29
> 分支：`codex/web-copilot-cockpit-lite-20260729`
> 基线：`4a86f0a`
> 状态：实现与真实浏览器 QA 完成；未 merge、未 push

## 用户问题

1. 279 个可比较特征一次性铺成 chip，模块、已选项和可选项堆在同一块，难以扫描。
2. 分组图把内置汇总项和全部已选特征一次画完，没有清晰的“按模块动态添加”路径。
3. 图后表格重复所有数值，并给每行放同样的“描述性”状态，信息量低且容易误解为统计结论。

## 实现

- 新增专属前端 owner：
  - `src/easyicu/webserver/static/js/screens-viz-cohort-groups.js`
  - `src/easyicu/webserver/static/css/cohort-groups.css`
- 特征选择改为“模块下拉 → 模块内特征下拉 → 添加”；当前选择单独显示为可移除紧凑 chip，不再渲染 279 个按钮。
- 分组图只接收当前已选 feature rows；每页最多 6 个，上一页/下一页切换，新增特征后自动定位到包含新特征的末页。
- `screens-viz-cohort-charts.js` 增加共享 ECharts grouped-bar contract。条形长度只在同一特征内部归一化，图中直接标出原始中位数/百分比，并明确禁止跨特征比较条长。
- 精确聚合值改为默认折叠的 `<details>`，只保留特征与各分组数值，删除重复“状态/描述性”列。
- 中文环境将动态 `Median X` 显示为 `X 中位数`，避免中英文混排。
- `screens-viz.js` 只保留 route state、API reload 与 owner 装配；旧 picker、CSS bar renderer 和事件绑定已删除。
- 修复 Cohort summary cache key：明确的 `selected_features: []` 与未提供选择（默认特征）现在是两个不同缓存键，清空后不会回弹为默认 10 项。

## 数据与统计边界

- 数据仍来自既有 `/api/cohort-review/summary` aggregate-only 合同，没有返回患者行。
- 页面是分组中位数/百分比的汇总预览，不新增或伪造 p 值、置信区间、SMD、匹配队列或因果结论。
- 精确聚合表仍完整保留当前选择的全部特征，只是默认折叠。

## 验证

### 自动化

- Cohort owner / route / cache 聚焦门：`12 passed`
- Web 静态与相关路径扩展门：`82 passed, 1 failed`
  - 唯一失败为隔离 worktree 名称导致既有 callback `project_ref.hint == "EASYICU"` 断言看到 `easyicu-copilot-cockpit-lite`；与本次 Cohort 改动无关。
- `node --check`：`screens-viz.js`、`screens-viz-cohort-charts.js`、`screens-viz-cohort-groups.js` 全过。
- CSS owner presence/absence、brace/comment scan、Python compile、`git diff --check` 全过。

### 真实浏览器（official MIMIC-IV Demo，1280×720）

- 后端真实范围：19 modules、279 comparable features；默认选择 10 项。
- 新 picker：19 个模块选项；Blood Gas 当前有 8 个未选特征；legacy 279-chip toggle 数为 0。
- ECharts：每页 6 个已选特征；中文动态标签为 `LACT 中位数`、`HR 中位数` 等。
- Chemistry → ALB：加入后 10→11，自动显示 `7–11 / 11`；移除后恢复 10。
- 清空：selected 10→0、chart 1→0、exact details 1→0；恢复默认后回到 10。
- 精确值：默认关闭；11 项选择时展开得到 11 行，状态列为 0。
- 页面高度从旧实现约 3,721px 降到 2,371px；picker 约 327px，profile 约 653px。
- `document.scrollWidth - innerWidth == 0`；console error/warning 0。

## 后续边界

- 若要加入正式 Table One、P 值、SMD 或调整后模型，必须进入 Agent Projects 的数值证据审计流程；不要在 Cohort 汇总预览里静默开放。
- 当前最多 48 个选择仍由后端合同限制；可视层继续每页 6 个，不把大量特征重新堆回一个长图。
