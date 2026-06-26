# 2026-06-27 WebApp Cross-DB Raw Selection State Fix

## Issue

In real Cross-DB raw database mode, the folder scan correctly detected six
database folders under `/Volumes/外置硬盘/databases`. However, clicking one
database card to exclude it invalidated the entire scan result. The UI then
looked as if every database had become unchecked or unrecognized, even though
the root folder and the other database folders had not changed.

## Fix

- Updated `src/easyicu/webserver/static/js/screens-viz.js`.
- Added `crossRawSelectionStatusFor(path)` so scan recognition and current
  user selection are separate states.
- `crossRawScanReadyFor(path)` now derives readiness from the current selected
  database keys intersected with the last successful scan's detected keys.
- Real-mode database-card toggles no longer call `invalidateCrossRawRootScan()`.
  Scan invalidation is still kept for real root-path edits.
- The scan panel now recomputes selected/detected/missing counts from the
  current UI selection. A detected but deselected database is shown as
  `not selected` / `未选择`, not `not found` / `未识别`.
- Bumped `screens-viz.js` cache key in `src/easyicu/webserver/static/index.html`.
- Added static regression assertions in `tests/test_webserver_static_routes.py`.

## Browser Evidence

Verified in the in-app browser at:

`http://127.0.0.1:8785/?_v=crossdb-selection-state-20260627#crossdb`

With root path `/Volumes/外置硬盘/databases`:

- Before deselection: six cards showed `已识别` with folders
  `mimiciv`, `eicu`, `aumc`, `hirid`, `mimiciii`, and `sic`.
- After clicking `MIMIC-III`: only `MIMIC-III` changed to `未选择 · 文件夹 mimiciii`.
- The other five cards remained `已识别`.
- The scan panel remained valid and reported:
  `已识别数据库文件夹: 6 · 已选且识别: 5/5 · 至少需要 2 个`.
- Run hint reported: `5 / 6 · 文件夹检查通过`.

## Validation

- `/Users/haibo/.nvm/versions/node/v24.11.0/bin/node --check EASYICU/src/easyicu/webserver/static/js/screens-viz.js`
- `EASYICU/.venv/bin/python -m compileall -q EASYICU/src/easyicu/webserver`
- `EASYICU/.venv/bin/pytest -q EASYICU/tests/test_webserver_static_routes.py -k 'crossdb_restores_distribution_visuals or crossdb_availability_matrix'`
  - `2 passed`
- `EASYICU/.venv/bin/pytest -q EASYICU/tests/test_webserver_workspace_summary.py -k 'crossdb_raw_root_scan'`
  - `2 passed`
