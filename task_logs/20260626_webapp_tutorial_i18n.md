# 2026-06-26 WebApp Tutorial i18n Fix

## Scope

Fix the native FastAPI `#tutorial` / Get Started page language split. The page was showing mixed copy in Chinese mode, including `GET STARTED · 快速上手`, English headings, English body text, and an English `Start demo` topbar action.

## Changed Files

- `src/easyicu/webserver/static/js/screens-help.js`
  - Converted tutorial page copy, rail labels, stage descriptions, FAQ copy, and the `Start demo` action to runtime `t(en, zh)` strings.
- `src/easyicu/webserver/static/js/app.js`
  - Added breadcrumb label localization for shared shell breadcrumbs.
  - Allowed `screen.actionHtml` to be a function so topbar/mobile actions react to the active language.
- `src/easyicu/webserver/static/index.html`
  - Bumped `app.js` and `screens-help.js` cache versions.
- `tests/test_webserver_static_routes.py`
  - Added a static regression that blocks hard-coded mixed tutorial copy and asserts dynamic localized action/breadcrumb support.

## Browser Evidence

Current-code QA ran on `http://127.0.0.1:8783/?_v=tutorial-i18n-qa#tutorial`.

- Report: `output/playwright/tutorial_i18n_20260626/tutorial_i18n.json`
- Screenshots:
  - `output/playwright/tutorial_i18n_20260626/tutorial_zh.png`
  - `output/playwright/tutorial_i18n_20260626/tutorial_en.png`

The browser check confirmed:

- Chinese mode shows Chinese headline, Chinese `开始演示`, and `首页 / 快速上手`.
- Chinese mode no longer shows the English headline or mixed `Get started · 快速上手`.
- English mode shows English headline, English `Start demo`, and `Home / Get Started`.
- English mode no longer shows the Chinese headline.
- `overflowX=0`.

## Verification

- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py` -> `124 passed, 1 warning`
- `python -m compileall -q src/easyicu/webserver` -> passed
- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check` -> passed
- `git diff --check` -> passed
- Provider readiness smoke: `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`

## Notes

`127.0.0.1:8782` is still an older uvicorn process. The current-code preview for this fix is on `127.0.0.1:8783`.
