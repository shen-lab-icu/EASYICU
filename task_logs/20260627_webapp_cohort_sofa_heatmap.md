# 2026-06-27 WebApp Cohort SOFA Heatmap

## Scope

- Replace the plain SOFA-1 to SOFA-2 transition table with a Cohort-owned heatmap matrix.
- Keep the backend payload unchanged: the UI still consumes `sofa_reclassification.transition_matrix`.
- Avoid mixed visible units by adding a matrix value toggle:
  - `Percent` / `百分比`
  - `N`

## Changed Files

- `src/easyicu/webserver/static/js/screens-viz.js`
  - Added `cohortSofaMatrixMode`.
  - Added `cohortSofaHeatmap`.
  - Added `[data-cohort-sofa-matrix-mode]` click handling.
- `src/easyicu/webserver/static/css/cohort.css`
  - Added `.sofa-heatmap`, `.sofa-heat-cell`, `.sofa-matrix-toggle`, and legend styles.
- `src/easyicu/webserver/static/index.html`
  - Bumped `cohort.css` and `screens-viz.js` cache keys.
- `tests/test_webserver_static_routes.py`
  - Added owner-file assertions for the heatmap UI.

## Verification

- `/Users/haibo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node --check src/easyicu/webserver/static/js/screens-viz.js`
- `.venv/bin/python -m compileall -q src/easyicu/webserver`
- `.venv/bin/pytest -q tests/test_webserver_static_routes.py -k 'cohort_comparison_radios'`
- `git diff --check`
- `python3 EASYICU/tools/lint_main_plan.py`

## Browser Evidence

Opened `http://127.0.0.1:8786/?_v=sofa-heatmap-20260627#cohort`, loaded the active real export, switched to SOFA reclassification, and verified:

- `.sofa-heatmap`: `1`
- `.sofa-heat-cell`: `16`
- `#cohbody table.mini-table`: `0`
- Default active toggle: `百分比`
- After selecting `N`, active toggle changed to `N` and sample cell values were counts such as `47,037`, `8,374`, `639`.
- Browser console errors: `[]`

## Notes

This is a presentation-layer fix only. The matrix still reports bounded aggregate SOFA-1/SOFA-2 transitions and does not return paired patient rows.
