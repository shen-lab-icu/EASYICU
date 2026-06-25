# 2026-06-26 WebApp Full-Flow QA And Idea Form Fix

Task: continue `WEBAPP-FASTAPI-NATIVE-QA` by exercising the native FastAPI WebApp from entry routes through extraction, visual review, cross-db comparison, idea mining, agent runs, settings, and reference routes. Fix real blockers rather than accepting fake/no-op UI.

## Change

- Fixed Idea Mining source metadata inputs in `src/easyicu/webserver/static/js/screens-ideas.js`.
- Moved source metadata from a collapsed `details` panel into a visible route-owned section. These fields are part of the primary mining input, so they should not be hidden behind an optional disclosure.
- Added matching owner CSS in `src/easyicu/webserver/static/css/ideas.css`.

## Failure Found

Initial Idea Mining browser QA failed on `#ideaTitle`:

```text
Page.fill: Timeout 30000ms exceeded
locator("#ideaTitle") resolved ... element is not visible
```

This was a real usability/testability issue: article metadata was required by the workflow but hidden inside a collapsed section, so the browser automation and a normal user path could both miss it.

## End-to-End Evidence

Passed after the fix:

- `python tools/qa_native_fastapi_extraction_filters.py --port 8821 --no-screenshots`
- `python tools/qa_native_fastapi_patient_drilldown.py --port 8822 --no-screenshots`
- `python tools/qa_native_fastapi_cohort_parity.py --port 8823 --no-screenshots`
- `python tools/qa_native_fastapi_crossdb_parity.py --port 8824 --no-screenshots`
- `python tools/qa_native_fastapi_idea_mining.py --port 8825 --no-screenshots`
- `python tools/qa_native_fastapi_agent_workflow.py --port 8826 --no-screenshots`
- `python tools/qa_native_fastapi_settings_workflow.py --port 8827`
- `python tools/qa_native_fastapi_reference_routes.py --port 8828`
- `python tools/qa_native_fastapi_routes.py --base-url http://127.0.0.1:8782/ --no-screenshots`

Key report paths:

- `output/playwright/native_fastapi_idea_mining_20260626_013129/idea_mining_qa.json`
- `output/playwright/native_fastapi_agent_workflow_20260626_013145/agent_workflow_qa.json`
- `output/playwright/native_fastapi_route_qa_20260626_013210/route_qa.json`
- `output/playwright/native_fastapi_crossdb_parity_20260626_012824/crossdb_parity_qa.json`
- `output/playwright/native_fastapi_patient_drilldown_20260626_012824/patient_drilldown_qa.json`

Observed guarantees:

- Extraction, Patient, Cohort, Cross-DB, Idea Mining, Agent, Settings, Dictionary, States, Help, and Guided routes have no desktop/mobile overflow or console errors in route QA.
- Idea Mining now completes source resolution, local mining, Agent handoff, Agent project creation, local history restore, and Agent navigation with HTTP 200 responses.
- Agent workflow creates real local artifacts: `run_context.json`, `cohort_summary.json`, `table1_summary.json`, `missingness_audit.json`, `roc_curve.json`, `calibration_curve.json`, `quality_gate.json`, and `evidence_ledger.json`; downloads and bundle export work.
- Agent signoff stays `signed_analysis_only`; `reportable=false`, `draft_unlocked=false`, `external_calls=0`.
- Settings controls persist through the backend and diagnostics export excludes secrets.
- Dictionary no longer shows the old `not audited` text on the current 8782 bundle; current labels distinguish active-export coverage, not-extracted concepts, and catalog audit scope.

Additional checks:

- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
- `python -m compileall -q src/easyicu/webserver`
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py` -> `120 passed, 1 warning`
- `pytest -q tests/research_agent/test_idea_mining.py tests/research_agent/test_idea_mining_funnel.py tests/research_agent/test_idea_mining_extended_feasibility.py` -> `87 passed`
- `git diff --check`

## Remaining Product Work

- Continue replacing any remaining seeded/demo visuals with backend-backed data or explicit unavailable/fail-closed panels.
- Cross-DB real six-database density/n x n parity still depends on additional real exports; current real two-export and fixture paths are covered.
- Conversation-first Copilot should continue as a separate implementation line, with project-folder-backed memory and hard handoff into Idea Mining / Agent Projects.
