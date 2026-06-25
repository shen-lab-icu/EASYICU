# 2026-06-26 WebApp Native Strict Layout And Button QA

## Scope

Follow-up after the native FastAPI WebApp full workflow batches. This pass tightened route-level visual QA and reran exhaustive content-button checks against the current server on `127.0.0.1:8799`.

## Changes

- Added `src/easyicu/webserver/static/css/agent.css` as the Agent Projects route owner for mobile/compact layout fixes.
- Wired `agent.css` in `src/easyicu/webserver/static/index.html`.
- Changed mobile `.table-scroll` from hidden overflow to explicit horizontal scrolling in `src/easyicu/webserver/static/css/app.css`, avoiding silent column clipping.
- Tightened `tools/qa_native_fastapi_routes.py` so strict route QA ignores only explicit ellipsis, shell sidebar viewport clipping, and children inside declared horizontal scroll containers.
- Added static regression checks in `tests/test_webserver_static_routes.py`.

## Evidence

- `python -m py_compile tools/qa_native_fastapi_routes.py` passed.
- `python -m compileall -q src/easyicu/webserver` passed.
- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check` passed.
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py` passed: `120 passed, 1 warning`.
- `python tools/qa_native_fastapi_routes.py --base-url http://127.0.0.1:8799/ --out-dir output/playwright --no-screenshots --strict-offscreen` passed.
  - Report: `output/playwright/native_fastapi_route_qa_20260626_003907/route_qa.json`
  - Desktop/mobile route matrix: all visible routes had `overflowX=0`, `offscreen=0`, `clipped=0`, `consoleErrors=0`.
  - Unknown hash fallback rewrote to `#entry` and rendered the Chinese Entry title when language state was Chinese.
- `python tools/qa_native_fastapi_button_audit.py --base-url http://127.0.0.1:8799/ --scope content --progress --fail-on-noop --out-dir output/playwright --after-click-wait-ms 120 --networkidle-ms 120` passed.
  - Report: `output/playwright/native_fastapi_button_audit_20260626_004028/button_audit.json`
  - Totals: `192` candidates, `181` clicked, `181` changed, `0` no-op, `0` click errors, `0` console errors, `11` skipped (`8` already selected, `3` disabled).
- Provider status stayed dormant through `/api/agent-runs/provider-status?provider=openai`: `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`.
- `git diff --check` passed.

## Notes

The strict route QA now distinguishes real layout failures from explicit text truncation and explicit horizontal scroll regions. It still fails on body horizontal overflow, console errors, unhandled clipped content, and unknown-hash routing regressions.
