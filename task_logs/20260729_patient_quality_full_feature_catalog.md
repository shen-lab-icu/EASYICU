# Patient Review：完整特征质量目录

> 日期：2026-07-29 13:31 EDT  
> 分支：`codex/web-copilot-cockpit-lite-20260729`  
> 基线：`b38bfb3`  
> 状态：已实现、已验证，未合并、未 push

## 用户问题

Patient Review 的“数据质量”页只展示目录就绪摘要和高缺失特征排行，281 个目录特征中的绝大多数只出现在总数里。用户无法判断某个特征属于哪个模块、是否已有观测、是否进入质量计算，也无法按模块查看全部特征。

## 实现

- `screens-viz-patient-quality.js` 以 `feature_coverage.modules[].features` 为完整目录主表，按 feature id 左连接 `quality_metrics.features`。
- 页面新增“完整特征质量目录”，完整呈现 19 个模块、281 个特征；默认只展开首个模块，并提供：
  - 特征名称 / ID / 单位搜索；
  - 19 模块筛选；
  - 需要关注 / 已计算 / 未计算 / 未观测筛选；
  - 全部展开 / 全部收起。
- 每个特征明确区分两个口径：
  - 数据状态：已有观测、已物化待核验、全空、未物化、不支持；
  - 质量范围：质量已计算、建议复核、需要关注、质量未计算、未进入质量计算。
- 缺失率、异常率、重复率仅在后端真实计算过时显示；未计算项显示 `—`，不伪造 0%。
- 后端移除 `quality_metrics.features` 的 80 项截断，继续只返回聚合特征质量行，不返回患者明细行。
- 样式保留在 Patient insights owner；未向 shared/catch-all 或其他 route CSS 泄漏。

## 来源与口径核验

对 active MIMIC-IV Demo prepared export 直接调用 Patient Review API：

```json
{
  "modules": 19,
  "features": 281,
  "status_counts": {
    "observed": 257,
    "materialized_unknown": 19,
    "all_null": 4,
    "structurally_unavailable": 1
  },
  "quality_features": 35,
  "quality_concept_count": 35,
  "payload_scope": "aggregate_quality_metrics_no_row_payload"
}
```

所以页面同时显示“281 个完整目录特征”和“35 个已有界质量计算”是两个真实但不同的口径；其余特征不能被误写成质量良好或缺失率为 0。

## 浏览器验收

来源：official MIMIC-IV Demo，140 个实体、19 个模块、151,373 行。

- DOM：19 个模块、281 个特征行，初始只展开 1 个模块。
- 模块筛选：`blood_gas` 精确显示 9 个特征。
- 搜索：中文“中心静脉压”精确显示 1 个特征。
- 质量筛选：
  - 已计算：35；
  - 未计算：246；
  - 未观测：24。
- 全部展开后 19 组打开；全部收起后 0 组打开。
- 1280×720：document、main、catalog 均无横向 overflow；console 0 error / 0 warning。

## 自动化验证

- JS syntax：`screens-viz-patient-quality.js`、`screens-viz.js` 通过。
- Node owner contract：完整目录合并、未计算不伪造比例、既有图表和审计表合同通过。
- 聚焦回归：10 passed。
- 扩展回归：217 passed，1 个既有失败；失败为隔离 worktree 名称 `easyicu-copilot-cockpit-lite` 与旧断言硬编码 `EASYICU` 不一致，不涉及本次代码。
- CSS owner presence/absence、brace/comment scan、`git diff --check` 通过。

## 关键文件

- `src/easyicu/webserver/patient_drilldown/__init__.py`
- `src/easyicu/webserver/static/js/screens-viz-patient-quality.js`
- `src/easyicu/webserver/static/js/screens-viz.js`
- `src/easyicu/webserver/static/css/patient-insights.css`
- `tests/js/patient_quality_owner.test.js`
- `tests/test_webserver_patient_browse_frontend.py`
- `tests/test_webserver_patient_demo_data.py`
- `tests/test_webserver_static_routes.py`
- `tests/test_webserver_workspace_summary.py`
