# 2026-06-26 WebApp Guided Inline Review + Agent Preflight

## Scope

Continue the conversation-first Guided Copilot path. This slice keeps Classic Workspace as the expert view, but lets a user stay inside `#guided` for common review and Agent preflight steps after preparing/registering data.

Changed files:

- `src/easyicu/webserver/static/js/screens-guided.js`
- `src/easyicu/webserver/static/css/guided.css`
- `src/easyicu/webserver/static/index.html`
- `tests/test_webserver_static_routes.py`

## Implemented

- Added inline `Review Data` workflow in Guided Copilot:
  - Calls `window.EU_API.loadPatientReviewDrilldown`.
  - Calls `window.EU_API.loadCohortReviewSummary`.
  - Shows patient drilldown, cohort aggregate, feature coverage, and KM/log-rank status in the conversation.
  - KM/log-rank is fail-closed when the active export lacks event and time-to-event fields.
- Added inline `Run a Research Project` preflight workflow:
  - Calls `window.EU_API.startAgentRun`.
  - Uses `run_type: "preflight"`, `llm_provider: "mock"`, and `external_llm_opt_in: false`.
  - Streams the local job through `/api/jobs/{job_id}/events`.
  - Keeps manuscript output locked and reportable false until normal evidence/signoff gates pass.
- Updated Guided Copilot goal copy so it no longer claims it only routes/prefills.
- Added route-owned CSS in `guided.css`; no new catch-all CSS or JS files were created.
- Bumped static cache keys for `guided.css`, `api.js`, and `screens-guided.js`.

## Browser QA

Playwright report:

- `output/playwright/guided_inline_workflows_20260626/guided_inline_workflows_qa.json`
- `output/playwright/guided_inline_workflows_20260626/guided_review.png`
- `output/playwright/guided_inline_workflows_20260626/guided_agent.png`

Result:

```json
{
  "console_errors": [],
  "checks": {
    "shell": true,
    "review_card": true,
    "km_status_present": true,
    "agent_card": true,
    "agent_completed": true,
    "draft_locked_or_reportable_false": true,
    "overflowX": 0,
    "hash": "#guided"
  }
}
```

## Command Evidence

Passed:

```bash
find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check
python -m compileall -q src/easyicu/webserver
pytest -q tests/test_webserver_static_routes.py
pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py
git diff --check
```

Focused pytest result:

- `122 passed, 1 warning`

Provider dormant smoke:

```json
{
  "ai_enabled": false,
  "ready": false,
  "client_constructed": false,
  "network_calls": 0,
  "secrets_returned": false
}
```

## Honest Limits

- The current active export on the test server contains limited modules, so KM/log-rank correctly shows schema-blocked until outcome/time-to-event fields are present.
- Idea Mining and the full Agent Projects workspace still own their artifacts; this slice adds an inline Guided preflight, not a full replacement of every Agent tab.
- Remaining product work is to keep moving additional no-op or seeded UI paths into true backend-backed workflows or explicit fail-closed states.
