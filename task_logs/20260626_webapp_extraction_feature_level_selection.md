# 2026-06-26 WebApp Extraction Feature-Level Selection

## Scope

User reported that Data Extraction only exposed module-level switches and did not allow choosing concrete features inside each module.

## Changes

- `src/easyicu/webserver/static/js/screens-extraction.js`
  - Added module detail expansion.
  - Added per-concept checkboxes using `window.EU_CATALOG.groupConcepts` and `window.EU_CATALOG.dict`.
  - Kept default behavior as all 19 modules and all 247 concepts selected.
  - Added module-level all/clear actions.
  - Sends `concepts` with real extraction job payload when the catalog selection is available.
- `src/easyicu/webserver/static/css/extraction.css`
  - Added owner-scoped styles for expanded module details and concept toggles.
  - No new route-specific CSS was added to catch-all files.
- `src/easyicu/webserver/app.py`
  - Passes `concepts` from `/api/jobs/extract` into the export runner.
- `src/easyicu/webserver/dataio.py`
  - `make_export_runner` now supports module-specific concept selections.
  - Invalid concepts fail closed before writing export files.
  - `_manifest.json` records `concept_selection`.
  - `README.md` records selected concept count.
- Tests updated:
  - `tests/test_webserver_static_routes.py`
  - `tests/test_webserver_workspace_summary.py`

## Verification

- `python -m compileall -q src/easyicu/webserver`
- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
- `ruff check src/easyicu/webserver/dataio.py src/easyicu/webserver/app.py tests/test_webserver_workspace_summary.py tests/test_webserver_static_routes.py`
- `pytest -q tests/test_webserver_static_routes.py -k extraction`: 8 passed.
- `pytest -q tests/test_webserver_workspace_summary.py -k 'export_runner_honors_module_specific_concept_selection or export_runner_rejects_unknown_selected_concepts'`: 2 passed.
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py`: 126 passed.
- `git diff --check`

## Browser QA

Temporary server: `127.0.0.1:8784`, stopped after QA.

Report:

- `output/playwright/extraction_concepts_20260626055044/concept_selection_qa.json`
- Desktop screenshot: `output/playwright/extraction_concepts_20260626055044/desktop.png`
- Mobile screenshot: `output/playwright/extraction_concepts_20260626055044/mobile.png`

Observed behavior:

- Initial custom extraction: `19 模块 · 247 概念`.
- Expanded default module shows 6 feature toggles.
- Unchecking the first demographics feature changes count to `19 模块 · 246 概念` and header `5/6`.
- Clearing demographics changes count to `18 模块 · 241 概念` and header `0/6`.
- Re-selecting demographics restores `19 模块 · 247 概念` and header `6/6`.
- Mobile viewport `393x852`: `overflowX=0`.
- Browser console errors: none.

## Notes

The browser QA was run in demo mode, so it did not submit a live extraction job. The real job payload and backend behavior are covered by unit tests: selected `{"demographics": ["age"], "vitals": ["hr", "map"]}` results in `load_concepts(["age"])` and `load_concepts(["hr", "map"])`, and unknown concepts fail closed before export files are created.
