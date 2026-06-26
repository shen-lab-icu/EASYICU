# 2026-06-26 Guided Copilot Inline Idea Mining QA

## Scope

This pass closes the biggest remaining Guided Copilot workflow gap: Idea Mining can now be started, configured, mined, reviewed, handed off, and converted into an Agent Project inside the Guided Copilot conversation surface. It does not force the user to jump into Classic Workspace for required setup.

Changed owner files:

- `src/easyicu/webserver/static/js/screens-guided.js`
- `src/easyicu/webserver/static/css/guided.css`
- `src/easyicu/webserver/static/index.html`
- `tests/test_webserver_static_routes.py`
- `tools/qa_native_fastapi_button_audit.py`

No catch-all CSS/JS file was extended. The new UI styles live in the Guided route owner CSS.

## Behavior Added

- `#guided` now detects idea-mining intent from goal cards or user text.
- Guided Copilot renders an inline Idea Mining card with source mode, source fields, network opt-in, local mining, feasibility, pre-experiment summary, prior-art check, handoff, and Agent Project creation.
- The workflow calls existing backend APIs instead of duplicating execution logic:
  - `resolveIdeaSource`
  - `mineIdeas`
  - `checkIdeaPriorArt`
  - `handoffIdea`
  - `createIdeaAgentProject`
- Prior-art network calls remain opt-in. Local mining and pre-experiment checks run without external calls.
- Empty chat send buttons are no longer classified as clickable no-op failures by button audit; text-send has separate positive QA.

## Browser Evidence

- Route QA passed:
  - `output/playwright/native_fastapi_route_qa_20260626_092225/route_qa.json`
  - desktop/mobile routes, `overflowX=0`, `offscreen=0`, `clipped=0`, `consoleErrors=0`
- Button audit passed:
  - `output/playwright/native_fastapi_button_audit_20260626_092417/button_audit.json`
  - 110 candidates, 102 clicked, 102 changed, 0 no-op, 0 click errors, 0 console errors
- Guided text-send QA passed:
  - `output/playwright/guided_inline_ideas_20260626/guided_text_send_qa.json`
  - typed "How is SOFA-2 defined?", received `.gd-concept-answer`, `overflowX=0`, no console errors
- Guided inline Idea Mining E2E passed:
  - `output/playwright/guided_inline_ideas_20260626/guided_inline_ideas_qa.json`
  - screenshot: `output/playwright/guided_inline_ideas_20260626/guided_inline_ideas.png`
  - selected Idea Mining, filled source, mined locally, produced ledger/pre-experiment, wrote handoff, created Agent Project, no console errors, `overflowX=0`

## Command Evidence

- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
- `python -m compileall -q src/easyicu/webserver`
- `pytest -q tests/test_webserver_static_routes.py`
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py`
- `pytest -q`
  - `2321 passed, 41 skipped, 950 warnings`
- `python tools/qa_native_fastapi_routes.py --base-url http://127.0.0.1:8782/`
- `python tools/qa_native_fastapi_button_audit.py --base-url http://127.0.0.1:8782/ --routes guided ideas agent extraction patient cohort crossdb settings dictionary --viewports desktop --max-clicks 180 --progress --fail-on-noop`
- `ruff check tools/qa_native_fastapi_button_audit.py tests/test_webserver_static_routes.py`
- `git diff --check`

Provider status stayed dormant: `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`.

## Honest Boundary

This completes the Guided Copilot inline Idea Mining slice and validates the current native WebApp button/route surface. It does not mean every advanced Classic Workspace option has been cloned into Guided Copilot. The intended architecture remains: Copilot owns conversation state and required setup; Idea Mining, Data Extraction, Patient/Cohort/Cross-DB, and Agent Projects own backend execution and evidence artifacts.
