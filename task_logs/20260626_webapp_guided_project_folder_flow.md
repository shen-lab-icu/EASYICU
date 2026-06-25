# 2026-06-26 WebApp Guided Project Folder Flow

## Scope

Fixed the Guided Copilot left-rail "new study" entry so it no longer feels like a silent registry-only draft. The entry now asks the user to either open an existing local EasyICU project folder or create a new local study folder.

## Changes

- `src/easyicu/webserver/static/js/screens-guided.js`
  - Renamed the rail action to `New / open study folder`.
  - Replaced the one-path draft form with two explicit paths:
    - Open existing project folder via `/api/guided/project/open`.
    - Create new metadata-only local study folder via `/api/guided/drafts`.
  - Added `openExistingGuidedProject()` so existing folders restore folder-scoped Guided memory.
  - Removed visible old "New study draft" entry copy from Guided UI strings.
- `src/easyicu/webserver/static/css/guided.css`
  - Added owner-scoped `.gds-choice` styles for the two folder actions.
- `src/easyicu/webserver/static/index.html`
  - Bumped Guided JS/CSS cache keys to `20260626-folder-flow`.
- `tests/test_webserver_static_routes.py`
  - Locked the folder-first UI contract and cache-bust versions.
- `tests/test_webserver_workspace_summary.py`
  - Added a backend regression: existing local project folders under the EasyICU projects root can be opened as scoped Guided memory even without `guided_draft.json`.

## Verification

- Browser DOM on `127.0.0.1:8782#guided`:
  - `data-newstudy` text: `New / open study folder`
  - setup panel contains `Open existing project folder`
  - setup panel contains `data-existing-project-dir`
  - setup panel contains `data-openprojectfolder`
  - setup panel contains `Create new local study folder`
- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
- `python -m compileall -q src/easyicu/webserver`
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py -k 'guided or static_routes'`
  - `32 passed, 85 deselected, 1 warning`
- `git diff --check`

## Boundary

No Streamlit code, no catch-all redesign CSS, no FastAPI backend route rewrite. The backend path validation remains fail-closed: folders must live under the local EasyICU projects root.
