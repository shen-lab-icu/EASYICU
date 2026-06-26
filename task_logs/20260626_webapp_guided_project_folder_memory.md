# 2026-06-26 WebApp Guided Copilot Project Folder Memory

## Scope

Guided Copilot now requires a local study folder before starting required setup flows. The change keeps the user's intended Copilot workflow inside Guided Copilot, but binds every conversation to a project folder so drafts, local runs, Idea Mining handoffs, and Agent Projects do not share anonymous state.

Changed files:

- `src/easyicu/webserver/static/js/screens-guided.js`
- `src/easyicu/webserver/static/css/guided.css`
- `src/easyicu/webserver/static/index.html`
- `tests/test_webserver_static_routes.py`

## Behavior

- Opening `#guided` no longer auto-creates an anonymous guided session.
- Selecting a goal such as Idea Mining, Prepare Data, Review Data, or Run Agent first asks the user to create or open a local study folder.
- The selected goal is remembered while the folder is being created/opened, then resumes inside that project-folder memory context.
- New local guided drafts immediately open `guided_copilot_session.json` through `/api/guided/project/open`, so the folder owns the conversation state.
- The frontdoor shows an explicit memory banner: unbound folders prompt for "New / open folder"; bound folders show the compact local project path.

## Verification

- `python EASYICU/tools/lint_main_plan.py` passed before the plan update.
- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check` passed.
- `python -m compileall -q src/easyicu/webserver` passed.
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py -k 'guided or native_guided'`: 8 passed, 110 deselected.
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py`: 122 passed, 1 warning.
- Browser QA on `http://127.0.0.1:8782/?_v=guided-project-memory-qa#guided` verified:
  - initial state shows the local-folder memory gate;
  - clicking Data Extraction opens folder setup instead of starting an anonymous session;
  - creating a QA draft resumes the selected extraction flow in the new project context;
  - `overflowX=0` and no console errors.
- Browser QA artifacts:
  - `output/playwright/guided_project_memory_20260626/guided_project_memory.json`
  - `output/playwright/guided_project_memory_20260626/guided_project_memory.png`
- Button audit: `output/playwright/native_fastapi_button_audit_20260626_101659/button_audit.json`
  - 198 candidates, 186 clicked, 186 changed, 0 no-op, 0 click errors, 0 console errors.
- Provider dormant smoke:
  - `ai_enabled=False`
  - `ready=False`
  - `client_constructed=False`
  - `network_calls=0`
  - `secrets_returned=False`
- `git diff --check` passed.

## Notes

The QA-created temporary local folder and registry row were removed after browser verification. Required study setup remains inside Guided Copilot; Classic Workspace stays an optional expert/review surface.
