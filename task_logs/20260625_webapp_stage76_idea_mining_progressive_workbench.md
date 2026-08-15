# Stage76 - Idea Mining Progressive Workbench

Date: 2026-06-25

## Scope

Refactored the native FastAPI Idea Mining UI so the user sees one core action at a time instead of a dense mixed form/result page.

Touched files:

- `src/easyicu/webserver/static/js/screens-ideas.js`
- `src/easyicu/webserver/static/css/ideas.css`
- `tools/qa_native_fastapi_idea_mining.py`
- `tests/test_webserver_static_routes.py`

## UX Changes

- Added a four-step workbench flow: Source -> Idea ledger -> Feasibility / pre-experiment -> Plan handoff.
- Kept local history on the left, but made the active work area progressive and stateful.
- Collapsed citation metadata and network/provider opt-in into `details` sections.
- Moved prior-art opt-in into the feasibility step so the user can enable it where the action happens.
- Reworked the left contextual rail to reuse the existing shell pattern (`rail-block`, `rail-head`, `setup-row`) instead of unstyled custom `rail-title` / `rail-kv` / `rail-note` markup.
- Preserved existing backend buttons and API contracts: resolve source, mine, prior-art, handoff, create Agent project, history reload.
- Kept route-owned CSS in `ideas.css`; did not add route-specific rules to `redesign.css`, `app.css`, or another catch-all file.

## Evidence

Commands passed:

- `node --check src/easyicu/webserver/static/js/screens-ideas.js`
- `python -m py_compile tools/qa_native_fastapi_idea_mining.py`
- `pytest -q tests/test_webserver_static_routes.py::test_native_idea_mining_is_first_class_route_and_backend_wired tests/test_webserver_workspace_summary.py::test_idea_mining_web_run_creates_ledger_preexperiment_and_handoff`
- `python tools/qa_native_fastapi_idea_mining.py --port 8795 --no-screenshots`
- `git diff --check -- src/easyicu/webserver/static/js/screens-ideas.js src/easyicu/webserver/static/css/ideas.css tools/qa_native_fastapi_idea_mining.py tests/test_webserver_static_routes.py`
- `python EASYICU/tools/lint_main_plan.py`

Browser QA report:

- `output/playwright/native_fastapi_idea_mining_20260625_151355/idea_mining_qa.json`

Key QA results:

- `/api/ideas/resolve-source`, `/api/ideas/mine`, `/api/ideas/handoff`, `/api/ideas/create-agent-project`, and `/api/ideas/run` all returned 200.
- History card restored ledger, pre-experiment, handoff, and project seed.
- Agent page showed the idea-derived project seed.
- `overflowX=0`, console errors `[]`, raw row markers absent.
- Provider dormant: `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`.

## Next

Continue full-page WebApp QA, but keep Idea Mining changes in owner files. The next Idea Mining backend step is still real literature/prior-art opt-in and seed-to-Agent run creation; this pass only reduced UI complexity and preserved the existing local flow.

## Stage77 Addendum - Local Project Binding

Date: 2026-06-25

The Idea Mining and Agent sidebars were tightened so Real mode no longer mixes in fabricated project rows.

Changes:

- `src/easyicu/webserver/static/js/screens-agent.js`
  - Renamed hard-coded studies to `DEMO_STUDIES`.
  - Real mode now lists only local Agent project seeds returned by `/api/ideas/agent-projects`.
  - If no local seed exists, the Agent page shows an explicit empty state with actions to open Idea Mining or refresh local projects.
  - `Run analysis` now fails closed in Real mode when no active registered export is selected instead of falling back to a demo animation.
- `src/easyicu/webserver/static/js/screens-ideas.js`
  - Left rail now says `Local idea runs` and labels entries as local runs, not research projects.
- `src/easyicu/webserver/ideas/mining.py`
  - `/api/ideas/history` filters out rows whose `idea_mining_run.json` no longer exists.
  - `/api/ideas/agent-projects` filters out rows whose `project_seed.json` no longer exists.
- `tests/test_webserver_static_routes.py` and `tests/test_webserver_workspace_summary.py`
  - Added regression checks for Real-mode local-only Agent projects and missing-file filtering.

Evidence:

- `node --check src/easyicu/webserver/static/js/screens-agent.js && node --check src/easyicu/webserver/static/js/screens-ideas.js`
- `python -m py_compile src/easyicu/webserver/ideas/mining.py`
- `pytest -q tests/test_webserver_static_routes.py::test_native_idea_mining_is_first_class_route_and_backend_wired tests/test_webserver_workspace_summary.py::test_idea_mining_lists_only_existing_local_runs_and_projects tests/test_webserver_workspace_summary.py::test_idea_mining_web_run_creates_ledger_preexperiment_and_handoff`
- `python tools/qa_native_fastapi_idea_mining.py --port 8797 --no-screenshots`

Browser QA report:

- `output/playwright/native_fastapi_idea_mining_20260625_152125/idea_mining_qa.json`

Key QA result:

- Idea Mining created a real local `project_seed.json`; Agent Projects then showed only that local idea-derived project, with no Sepsis/Cross-DB/Lactate/AKI demo studies mixed into Real mode.
