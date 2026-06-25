# 2026-06-26 Remove Seeded Audit Bundle Residual

## Scope

Static audit found `screens-audit.js` still loaded in the native FastAPI bundle. The router already maps `#audit` to the Cohort coverage panel, so this file was no longer an active route owner. It contained old seeded coverage and SOFA reclassification panels that could be mistaken for real Cohort logic if reused.

## Changes

- Removed `src/easyicu/webserver/static/js/screens-audit.js`.
- Removed the script tag from `src/easyicu/webserver/static/index.html`.
- Updated Cohort coverage/SOFA fallback in `src/easyicu/webserver/static/js/screens-viz.js` to fail closed with an explicit unavailable message until a real cohort-review payload is loaded.
- Added static regressions in `tests/test_webserver_static_routes.py` to keep `screens-audit.js`, `window.EUAudit`, and `window.EUSofa` out of the active bundle.

## Evidence

- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check` passed.
- `python -m compileall -q src/easyicu/webserver` passed.
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py` passed: `120 passed, 1 warning`.
- `python tools/qa_native_fastapi_routes.py --base-url http://127.0.0.1:8799/ --out-dir output/playwright --no-screenshots --strict-offscreen` passed.
  - Report: `output/playwright/native_fastapi_route_qa_20260626_004920/route_qa.json`
  - All desktop/mobile routes reported `overflowX=0`, `offscreen=0`, `clipped=0`, `consoleErrors=0`.
- `git diff --check` passed.

## Notes

Demo mode remains allowed for explicitly labelled demos, but inactive seeded result scripts should not remain loaded in the production native bundle.
