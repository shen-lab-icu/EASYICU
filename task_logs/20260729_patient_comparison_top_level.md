# Patient Review multi-patient comparison top-level restoration

- Task ID: `WEB-PATIENT-COMPARE-TOPLEVEL-20260729`
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Scope: native Patient Review information architecture only; no backend/API or data-contract changes.

## Outcome

Patient Review now exposes `Multi-patient comparison / 多患者对比` as a top-level tab in place of the hidden single-patient overview entry. The existing same-feature comparison owner remains the sole state/API owner and still caps projections to five pseudonymous entities.

Time Series now contains only the two single-entity views it owns:

1. Feature catalog
2. Trajectory gallery

The duplicated inner comparison mode was removed. Shell navigation and Page Guide copy now use the same `tables · trends · comparison` model.

## Ownership

- Top-level route wiring: `src/easyicu/webserver/static/js/screens-viz.js`
- Time-series/comparison rendering boundary: `src/easyicu/webserver/static/js/screens-viz-patient-series.js`
- Comparison state and bounded API fan-out: `src/easyicu/webserver/static/js/screens-viz-patient-comparison.js` (reused unchanged)
- Comparison-only CSS: `src/easyicu/webserver/static/css/patient-comparison.css`
- Shared time-series CSS: `src/easyicu/webserver/static/css/patient-series.css`
- Shell summary: `src/easyicu/webserver/static/js/app.js`
- Page Guide copy: `src/easyicu/webserver/static/js/copilot-dock.js`

Comparison selectors were migrated out of `patient-series.css` into `patient-comparison.css`; the ownership regression checks presence in the owner and absence from the shared series file.

## Verification

- JavaScript syntax checks passed for `app.js`, `copilot-dock.js`, `screens-viz.js`, and `screens-viz-patient-series.js`.
- CSS owner/brace/comment scan passed for the comparison owner.
- Focused Patient Review regression: `4 passed`.
- Broader native frontend regression: `85 passed, 1 deselected`.
- The deselected pre-existing test is worktree-name-sensitive:
  `test_native_extraction_feature_definition_manifest_records_callback_provenance`
  expects project hint `EASYICU`, while this isolated worktree is named
  `easyicu-copilot-cockpit-lite`.

## Browser QA

Viewport: `1134px`, local native UI at `http://localhost:8876/?qa=density5#patient`.

- Top tabs: `数据表 / 时间序列 / 多患者对比 / 数据质量`
- Active top-level comparison rendered five bounded pseudonymous traces.
- Feature selector changed from heart rate to lactate; label, unit (`mmol/L`), selected option, and five traces updated together.
- Sidebar summary updated to `表格 · 趋势 · 对比`.
- Document horizontal overflow: `false`
- Patient content horizontal overflow: `false`
- Comparison workspace clipping: `false`
