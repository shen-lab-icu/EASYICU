# 2026-06-26 Guided Copilot Project Slot Persistence

## Scope

Guided Copilot now persists required setup configuration inside the selected local project-folder session instead of keeping it only in browser memory. This preserves the intended Copilot behavior: required study setup can be completed inside the Guided conversation, while Idea Mining, Data Extraction, Review, and Agent Projects still own their execution backends and artifacts.

## Changed files

- `src/easyicu/webserver/guided_sessions.py`
- `src/easyicu/webserver/static/js/api.js`
- `src/easyicu/webserver/static/js/screens-guided.js`
- `src/easyicu/webserver/static/index.html`
- `tests/test_webserver_static_routes.py`
- `tests/test_webserver_workspace_summary.py`

## Behavior

- Added a metadata-only `update_slots` Guided action.
- Slot updates require an existing `project_folder` session and fail closed without one.
- Persisted setup includes bounded configuration for:
  - active flow
  - study params
  - active export summary
  - extraction path, cohort, modules, format, scan, and result metadata
  - patient/review source selection summary
  - agent question and run summary metadata
  - idea source, plan edits, local run id, handoff id, and linked Agent project path
- Row-level keys such as `tableRows`, `series`, `patient`, `stay_id`, `subject_id`, and `hadm_id` are rejected from persisted slot data.
- Reopening a local Guided project restores the saved setup card in the conversation.

## Browser QA

Report:

- `output/playwright/guided_project_slots_20260626/guided_project_slots_verified.json`

Verified checks:

- Guided page loaded.
- Data extraction goal asked for local study folder first.
- Local draft folder was created under `~/easyicu/projects`.
- Extraction setup card appeared after folder binding.
- Path, module selection, and CSV format were saved through `/api/guided/action`.
- `guided_copilot_session.json` contained the current bounded slots.
- Reloading and reopening the draft restored the extraction card, path, CSV format, and single selected module.
- Disk session did not contain row-level markers.
- `overflowX=0`.
- Temporary QA project folder and registry row were removed after the run.

## Verification

- `python -m compileall -q src/easyicu/webserver`
- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
- `pytest -q tests/test_webserver_workspace_summary.py -k 'guided_project_memory_persists_bounded_setup_slots or guided_project_memory_restores_conversation_per_local_folder'`
- `pytest -q tests/test_webserver_static_routes.py -k 'guided or shell_language_icon'`
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py`
- `git diff --check`
- Provider dormant smoke: `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`.

## Note

An initial browser QA attempt against the old `127.0.0.1:8782` server saw stale Python backend behavior. The verified browser run used a fresh controlled server on `127.0.0.1:8783`.
