# PATIENT-MODULE-FEATURE-STATIC-COMPARISON

## Outcome

Patient Review 的顶层「多患者对比」已从单个扁平特征下拉框改为两级目录：

1. 先选择完整 19 个临床模块；
2. 再选择该模块的全部目录特征，并标明时间序列、静态数值、静态分类或当前导出不可用。

选择仍复用 `/api/patient-review/feature`，最多读取当前页前 5 个伪匿名实体，不新增全量患者载荷。

## Visualization contract

- 时间索引数值且至少两个患者有轨迹：共享 ICU 相对时间轴折线图。
- 静态数值：患者点图；不虚构时间轴。
- 静态分类：患者 × 观测类别矩阵。
- 少于两个可比较观测：fail closed，并给出明确状态。

ECharts option 和 lifecycle 仍由 `screens-viz-patient-charts.js` 持有；选择/API/cache 状态由 `screens-viz-patient-comparison.js` 持有；布局只修改 `patient-comparison.css`。

## Verification

- Node owner contracts:
  - 5-entity bound and page-scoped cache
  - module → feature selection reset
  - trajectory / static numeric / static categorical projection
  - ECharts scatter options and renderer fallback
- Affected Web suite: `85 passed, 2 deselected, 1 warning`
  - 两个 deselect 均为隔离 worktree 的既有路径/版本断言：
    - seeded demo 旧 `screens-viz.js` cache token
    - callback project hint 假定 worktree 名为 `EASYICU`
- `git diff --check`: pass
- JS syntax checks: pass
- CSS owner/foreign-selector/brace/comment scan: pass
- Desktop browser QA at 1280px:
  - module options: 19
  - demographics feature options: 6
  - `age` → `static-numeric-comparison`, SVG mounted
  - `sex` → `static-categorical-comparison`, SVG mounted
  - `hr` → `comparison`, SVG mounted
  - document horizontal overflow: 0
  - dynamic x-axis title and zoom navigator no longer overlap

## Browser evidence

- `output/ui-qa/20260730_patient_module_feature_static_comparison/01c-two-level-dynamic-spacing-final.jpg`
- `output/ui-qa/20260730_patient_module_feature_static_comparison/02b-static-numeric-dot-plot-chart.jpg`
- `output/ui-qa/20260730_patient_module_feature_static_comparison/03-static-categorical-matrix.jpg`

## Scope

- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Worktree: `.worktrees/easyicu-copilot-cockpit-lite`
- No merge and no push.
