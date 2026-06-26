# 2026-06-26 WebApp Guided Folder Memory Button

## Scope

- Fixed the Guided Copilot **Open folder memory** button inside the local study folder setup panel.
- Kept the change inside route owner files:
  - `src/easyicu/webserver/static/js/screens-guided.js`
  - `src/easyicu/webserver/static/css/guided.css`
  - `src/easyicu/webserver/static/index.html`
  - `tests/test_webserver_static_routes.py`
- No FastAPI backend contract change was required. `/api/guided/project/open` already accepted existing local project folders and failed closed outside the EasyICU projects root.

## Behavior

- Empty path now shows a visible inline status in the setup card and focuses the path field.
- Opening a folder shows an inline loading status while the backend restores scoped project memory.
- Backend failures remain fail-closed and show the reason inline and in the conversation.
- Successful open binds the Guided conversation to that local project folder and restores the project-scoped memory.

## Evidence

- Browser QA against current server `127.0.0.1:8783`:
  - Report: `output/playwright/guided_folder_memory_20260626_115923/guided_folder_memory_qa.json`
  - Screenshot: `output/playwright/guided_folder_memory_20260626_115923/guided_folder_memory_open.png`
  - Empty path status was visible.
  - A real metadata-only draft folder under `~/.easyicu/projects/...` opened successfully.
  - Console errors: `[]`
- Focused backend memory tests:
  - `pytest -q tests/test_webserver_workspace_summary.py -k 'guided_project_memory or guided_project_open'`
  - `3 passed`
- Static/owner contract:
  - `pytest -q tests/test_webserver_static_routes.py -k guided`
  - `4 passed`
- Focused WebServer regression:
  - `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py`
  - `124 passed, 1 warning`
- Syntax and hygiene:
  - `python -m compileall -q src/easyicu/webserver`
  - `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
  - `ruff check tests/test_webserver_static_routes.py`
  - `git diff --check`
- Provider stayed dormant:
  - `ai_enabled=false`
  - `ready=false`
  - `client_constructed=false`
  - `network_calls=0`
  - `secrets_returned=false`

## Notes

- The user was viewing stale port `8782`. The verified current-code server is `8783`.
- This patch does not create or read patient rows; the browser QA creates only a metadata-only Guided draft folder.
