# 2026-06-26 Guided Copilot Local Folder Browser

## Scope

Guided Copilot's "New / open study folder" flow previously required a user to paste a local project path or rely on auto-detected project folders. The user explicitly requested a folder picker. This pass adds an in-Guided local folder browser while keeping project setup inside Copilot.

## Implementation

- Added a Guided-owned folder browser in `src/easyicu/webserver/static/js/screens-guided.js`.
- Added route-owned styles in `src/easyicu/webserver/static/css/guided.css`.
- Reused the existing local FastAPI filesystem endpoint through `window.EU_API.listDir`.
- Kept manual path paste as an advanced fallback, not the main path.
- Updated `src/easyicu/webserver/static/index.html` cache versions to `20260626-guided-folder-browser`.
- Added static route assertions in `tests/test_webserver_static_routes.py`.

This is not a browser-native Finder/Explorer dialog because a web page cannot safely expose arbitrary absolute filesystem paths. The picker is still a real local folder browser: it lists folders via the local EasyICU server, stays on the user's machine, and opens the selected folder through the existing Guided project API.

## Validation

- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
- `python -m py_compile src/easyicu/webserver/app.py src/easyicu/webserver/dataio.py`
- `pytest -q tests/test_webserver_static_routes.py -k guided` -> 4 passed
- `pytest -q tests/test_webserver_static_routes.py` -> 30 passed, 1 warning
- `git diff --check`

## Browser QA

URL: `http://127.0.0.1:8785/?_v=guided-folder-browser#guided`

Flow:

1. Open Guided Copilot.
2. Click `New / open study folder`.
3. Choose `Use existing folder`.
4. Click `Browse...`.
5. Navigate `Home -> easyicu -> projects -> guided-new-local-study-5ef2c7`.
6. Click `Use this folder`.

Observed requests:

- `GET /api/fs/list`
- `GET /api/fs/list?path=/Users/haibo/easyicu`
- `GET /api/fs/list?path=/Users/haibo/easyicu/projects`
- `GET /api/fs/list?path=/Users/haibo/easyicu/projects/guided-new-local-study-5ef2c7`
- `POST /api/guided/project/open`

Browser assertions:

- `overflowX = 0`
- project memory card appears
- selected local folder path appears
- folder browser closes after selection

Artifacts:

- `output/playwright/guided_folder_browser_20260626/guided_folder_browser_after.png`
- `output/playwright/guided_folder_browser_20260626/guided_folder_browser_qa.json`
