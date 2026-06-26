# 2026-06-26 Guided Folder Menu

## Scope

Changed Guided Copilot's local project binding entry from an inline chat form into a rail-owned project menu and modal dialog.

## Files

- `src/easyicu/webserver/static/js/screens-guided.js`
- `src/easyicu/webserver/static/css/guided.css`
- `src/easyicu/webserver/static/index.html`
- `tests/test_webserver_static_routes.py`

## Behavior

- `New / open study folder` now opens a two-option menu: new blank study folder or use existing folder.
- The open/create forms render in a modal dialog, not as a bot message inside the conversation thread.
- Existing backend paths are reused: `createGuidedDraft(payload)` and `/api/guided/project/open`.
- On successful open/create, the modal closes and Guided restores the folder-scoped conversation memory.

## Evidence

- JS syntax: `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
- Static tests: `pytest -q tests/test_webserver_static_routes.py -k guided` -> 4 passed
- Guided backend focused tests: `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py -k 'guided'` -> 9 passed
- Compile: `python -m compileall -q src/easyicu/webserver`
- Browser QA: `output/playwright/guided_folder_menu_20260626/guided_folder_menu_qa.json`
  - menu items: 2
  - chat bubbles before/after menu click unchanged
  - old inline `.gd-thread .gd-draft-setup`: 0
  - desktop and 393px mobile screenshots captured
  - mobile `overflowX=0`
- Diff check: `git diff --check`
- Provider readiness: `/api/agent-runs/provider-status` returned `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`.

## Notes

The browser cannot reliably expose arbitrary absolute folder paths from a native directory picker. The existing safe path remains explicit local path paste plus backend validation under the EasyICU projects root.
