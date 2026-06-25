# 2026-06-26 WebApp Guided Copilot inline extraction slice

## Scope

User clarified that Guided Copilot should complete as much of the Classic workflow as possible inside the guided conversation instead of only deep-linking to Classic pages. This slice implements the first concrete path: data extraction setup and local concept-definition Q&A inside `#guided`.

## Changes

- Added a Guided Copilot extraction card in `src/easyicu/webserver/static/js/screens-guided.js`.
- The card keeps the user in `#guided` and supports local folder scan, cohort preset choice, module selection, Parquet default, 500 safety cap vs all stays, and real extraction job launch through `/api/jobs/extract`.
- The card does not prefill author-machine paths. Users must paste or choose a local ICU data folder, then run the existing scan API first.
- Existing EasyICU module exports are fail-closed into “register export” instead of being re-extracted.
- Added local concept-definition Q&A for questions such as “SOFA-2 是怎么定义的？”, backed by `window.EU_CATALOG` from `/api/catalog`.
- Added route-owned Guided CSS in `src/easyicu/webserver/static/css/guided.css`; no new catch-all CSS file.
- Added `startExtractionJob` API wrapper in `src/easyicu/webserver/static/js/api.js`.

## Browser evidence

- Opened `http://127.0.0.1:8782/?_v=guided-inline-extraction-scan-20260626#guided`.
- Clicked “Prepare Data”: stayed on `#guided`, showed an inline extraction setup card, did not prefill a local path, selected all 19 modules by default, and used Parquet by default.
- Filled `/Volumes/外置硬盘/databases/mimiciv`, clicked “Analyze folder”: scan recognized `MIMIC-IV · prepared · 31 tables · 19 modules`, and the local extraction button became enabled.
- Asked `SOFA-2 是怎么定义的？`: response used local catalog metadata for `sofa2`, including code, group, unit, definition, and current export coverage note.
- I did not click the final extraction run button during manual browser QA to avoid writing a new export while the user is debugging.

## Verification

- `python -m compileall -q src/easyicu/webserver`
- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py` -> `122 passed, 1 warning`
- `git diff --check`
- Provider dormant smoke: `ai_enabled=False`, `ready=False`, `client_constructed=False`, `network_calls=0`, `secrets_returned=False`

## Remaining work

- This is the first inline Guided Copilot workflow slice, not the full replacement for Classic.
- Next slices should bring review/cohort/KM/agent handoff into the same conversation-first pattern, while keeping Classic pages as direct expert views.
- A native folder picker modal would improve the path input; current implementation accepts typed/pasted local paths.
