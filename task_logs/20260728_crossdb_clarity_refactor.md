# Cross-DB clarity refactor

> 时间：2026-07-28 04:48 EDT  
> 任务：`WEBAPP-FASTAPI-NATIVE-QA · PATIENT-CROSSDB-VISUAL-PARITY`

## 结果

- 产品名称统一为“跨库对比 / Cross-database comparison”，避免把数据审阅页误解为模型 benchmark。
- 设置态改为渐进式单路径流程：演示模式固定使用已就绪的 MIMIC-IV Demo + eICU Demo；真实模式先选择“已准备的 EasyICU 导出”或“原始 ICU 数据库目录”，仅展开所选路径，原始目录完成扫描后才显示数据库、采样和运行控制。
- 结果态拆为“概览 / 覆盖 / 分布 / 质量与溯源”四个明确任务区。页面顶部统一持有“更换数据 / 导出 / 重新运行”，正文不再重复动作。
- 分布页默认显示 12 个核心特征，同时支持按 19 个模块浏览和搜索本次真实聚合返回的全部 300 个 feature profiles；任何时刻只绘制当前选中特征的一张主图，并附精确统计表。
- 官方双库结果继续走 prepared export 和真实聚合 API，不用 seeded synthetic profile 冒充真实 Demo。浏览器实测为 2 个来源、19 个共享模块、300 个可用 feature profiles，其中 240 个在两个来源均存在。

## Owner 与模块化

- 新增 `src/easyicu/webserver/static/js/screens-viz-crossdb-results.js`（462 行），专门持有 Cross-DB loaded-result renderer 和交互；共享 `screens-viz.js` 降至 4,811 行。
- Cross-DB 样式仍只在 `src/easyicu/webserver/static/css/crossdb.css`（317 行）中维护，没有向 `app.css`、`tweaks.css` 或其他 catch-all 文件追加 route-specific 规则。
- 新增 `tests/js/crossdb_results_owner.test.js`，锁定完整目录筛选、单主图、结果 tabs 与无重复动作；更新 setup/source owner、静态 wiring 和 CSS presence/absence 合同。

## 自动化验证

```text
pytest -q \
  tests/test_webserver_static_routes.py \
  tests/test_webserver_workspace_summary.py \
  tests/test_webserver_patient_feature_coverage.py \
  tests/test_webserver_patient_demo_data.py \
  tests/test_webserver_demo_sources.py \
  tests/test_webserver_route_contracts.py \
  tests/test_webserver_crossdb_setup_frontend.py \
  tests/test_webserver_cohort_profile_ui.py \
  tests/test_repository_contract.py

274 passed, 1 warning in 16.06s
```

- 4 个受影响 JS owner 的 `node --check` 全过。
- setup、source pair、results owner 三组 Node 行为合同全过。
- `git diff --check` 全过。

## 浏览器验收

- 官方 Demo 对在 catalog 刷新后保持 2/2 ready，不再出现一个来源“已就绪”、另一个仍要求准备的矛盾状态。
- 加载态明确显示“2 个官方 Demo · 仅聚合”；结果页显示 2 sources / 19 modules / 300 feature profiles。
- “全部映射特征”可显示 300 项；搜索 `vent` 返回 22 项，页面始终只有 1 张主图。
- 1280 px 桌面视口无 document 横向 overflow；真实模式选择 raw 路径后，未扫描前不渲染采样和运行控件。
- 截图：
  - `output/ui-audit/20260728_crossdb_clarity_refactor/01-distributions-all-features.jpg`
  - `output/ui-audit/20260728_crossdb_clarity_refactor/02-demo-source-pair.jpg`
  - `output/ui-audit/20260728_crossdb_clarity_refactor/03-real-progressive-source.jpg`

## 仍未扩大的范围

- registered 多导出长请求的 async job / timeout / cancel 产品决策仍是独立后续任务；本轮没有把该问题藏进前端 fallback。
- 旧 synthetic profile 只保留为显式离线兜底，不参与默认官方 Demo 结果。

## 后续口径纠正（2026-07-28 05:08 EDT）

本日志记录的 300 个 profiles 中包含 `patientunitstayid` 在 19 个模块里的重复实体标识。后续修复已把标识列排除，官方双 Demo 的真实临床目录口径为 19 模块、281 个特征；同时 raw 快速结果与完整目录范围已明确分开。详见 `EASYICU/task_logs/20260728_crossdb_full_catalog_scope.md`。
