# 2026-06-26 Guided Folder Picker

## Scope

Improved the Guided Copilot project-folder dialog so users can choose from detected local project folders instead of copying an absolute path by hand.

## Files

- `src/easyicu/webserver/static/js/screens-guided.js`
- `src/easyicu/webserver/static/css/guided.css`
- `src/easyicu/webserver/static/index.html`
- `tests/test_webserver_static_routes.py`

## Behavior

- The `Use existing folder` dialog now shows `Detected local project folders`.
- Detected choices are real local folders from Guided draft registry and Agent run history.
- Clicking a detected row opens that folder through `/api/guided/project/open`.
- Manual path paste remains as an advanced fallback for folders not yet detected.
- The dialog re-renders when local draft/run scans finish, so the list appears without reopening the dialog.

## Evidence

- JS syntax: `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
- Static guided tests: `pytest -q tests/test_webserver_static_routes.py -k guided` -> 4 passed
- Guided backend focused tests: `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py -k 'guided'` -> 9 passed
- Compile: `python -m compileall -q src/easyicu/webserver`
- Browser QA: `output/playwright/guided_folder_picker_20260626/guided_folder_picker_qa.json`
  - detected known project rows: 12
  - manual fallback input: present
  - click detected row opened project memory directly
  - mobile dialog `overflowX=0`
- Diff check: `git diff --check`
- Provider readiness: `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`.

## Note

Native browsers do not expose arbitrary absolute directory paths to JavaScript. The user-facing default is now direct selection from known local EasyICU folders, while path paste remains available for uncommon folders that are not yet indexed.
