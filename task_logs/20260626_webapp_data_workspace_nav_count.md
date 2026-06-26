# 2026-06-26 WebApp Data Workspace Navigation Count

## Issue

Manual browser QA found the Data Workspace sidebar showing `2 / 3` while the group contains four direct entries:

- Extract Data
- Patient Drilldown
- Cohort Review
- Cross-DB Compare

This mixed the global 3-step research pipeline count with the 4-item Data Workspace group, making the active position misleading.

## Fix

Changed the Data Workspace group counter in `src/easyicu/webserver/static/js/app.js` to use the active item index within `CLASSIC`:

- `#extraction` -> `1 / 4`
- `#patient` -> `2 / 4`
- `#cohort` -> `3 / 4`
- `#crossdb` -> `4 / 4`

Updated `src/easyicu/webserver/static/index.html` to bump the `app.js` cache key and added static regression assertions in `tests/test_webserver_static_routes.py`.

## Browser Evidence

Browser QA report:

- `output/playwright/data_workspace_count_20260626/data_workspace_count.json`
- `output/playwright/data_workspace_count_20260626/crossdb_sidebar_count.png`

The DOM check confirmed all four routes show the expected count, with `overflowX=0` and no console errors.

## Verification

- `node --check src/easyicu/webserver/static/js/app.js`
- Native JS `node --check` for all webserver static JS
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py` -> 124 passed, 1 warning
- `python -m compileall -q src/easyicu/webserver`
- `git diff --check`
- Provider remained dormant: `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`
