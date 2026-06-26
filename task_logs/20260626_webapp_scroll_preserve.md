# 2026-06-26 WebApp Scroll Preservation

## Issue

Manual browser QA found that clicking controls in lower parts of pages often jumped the viewport back to the top.

Root cause: the app shell treated every render as a route navigation. `app.js` always reset both `.content.scrollTop` and `window.scrollTo(0, 0)`, while many route controls call `window.__euRender()` for same-page updates.

## Fix

Updated `src/easyicu/webserver/static/js/app.js` so render has explicit scroll semantics:

- `render({ resetScroll: true })` is used for initial load, real route navigation, hash fallback/alias navigation, and keyboard route jumps.
- Same-page `window.__euRender()` calls preserve `window.scrollX/window.scrollY` and `.content.scrollTop`.
- Sidebar expand/collapse and route-local controls no longer force the browser to the top.

Updated `src/easyicu/webserver/static/index.html` to bump the `app.js` cache key and added static assertions in `tests/test_webserver_static_routes.py`.

## Browser Evidence

Report:

- `output/playwright/scroll_preserve_20260626/scroll_preserve.json`
- `output/playwright/scroll_preserve_20260626/scroll_preserve_dictionary.png`

Checked behavior:

- Data Extraction: scrolled to export format controls, clicked CSV, scroll stayed at `1026 -> 1026`.
- Settings: scrolled to density controls, clicked Compact, scroll stayed below fold at `1560 -> 1560`.
- Route navigation still resets to top: Data Extraction bottom -> Data Dictionary resulted in `1175 -> 0`.
- Console errors: none.

## Verification

- Native JS `node --check` for all webserver static JS
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py` -> 124 passed, 1 warning
- `python -m compileall -q src/easyicu/webserver`
- `git diff --check`
- Provider dormant: `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`
