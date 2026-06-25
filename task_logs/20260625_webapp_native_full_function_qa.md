# 2026-06-25 FastAPI Native WebApp Full Function QA

Scope: clicked and exercised the native FastAPI WebApp route surface on a live local server, focusing on safe controls, real backend workflows, route state, settings persistence, provider safety, and Agent resume/review behavior.

Server under test:

- `PYTHONPATH=src .venv/bin/python -m uvicorn easyicu.webserver.app:app --host 127.0.0.1 --port 8791`

Changes made during QA:

- `src/easyicu/webserver/static/js/screens-agent.js`
  - Ensured opened or restored Agent run reviews populate `window.EU_AGENT_RUN_REVIEW`.
  - Cleared the debug review state on review-load failure so stale review state cannot masquerade as restored.
- `tools/qa_native_fastapi_agent_workflow.py`
  - Waits for the restored run review API to complete before asserting reload/resume success.
- `tools/qa_native_fastapi_button_audit.py`
  - Counts generic `status=error` results as failures.
  - Forces safe demo mode for non-destructive button audits; real data behavior is covered by dedicated fixture/real-path QA.
  - Emits the original candidate when a candidate disappears after reload.
- `tools/qa_native_fastapi_extraction_filters.py`
  - Opens the current manual/custom extraction panel before clicking module controls.
- `tools/qa_native_fastapi_settings_workflow.py`
  - Updated assertions to match current Settings design: global workspace/token/evidence controls are locked or run-owned; active controls are data mode, language, density, AI opt-in, reduce motion, export path, docs, diagnostics, reset.
- `tools/qa_native_fastapi_reference_routes.py`
  - Dictionary lactate assertion is bilingual (`Lactate`, `乳酸`, or `lact` code).
- Route QA scripts now use `domcontentloaded` instead of `networkidle` to avoid false hangs on routes with live metadata polling.

Browser QA evidence:

- Route QA passed: `output/playwright/native_fastapi_route_qa_20260625_230048/route_qa.json`.
- Per-route safe content button audits passed for desktop/mobile routes, with latest Cross-DB mobile fix at `output/playwright/native_fastapi_button_audit_20260625_232048/button_audit.json`.
- Extraction advanced filters passed: `output/playwright/native_fastapi_extraction_filters_20260625_231351/extraction_filter_qa.json`.
- Patient drilldown passed: `output/playwright/native_fastapi_patient_drilldown_20260625_231409/patient_drilldown_qa.json`.
- Cohort parity passed: `output/playwright/native_fastapi_cohort_parity_20260625_231413/cohort_parity_qa.json`.
- Cross-DB parity and density distribution passed: `output/playwright/native_fastapi_crossdb_parity_20260625_231456/crossdb_parity_qa.json`.
- Idea Mining workflow passed: `output/playwright/native_fastapi_idea_mining_20260625_231504/idea_mining_qa.json`.
- Agent workflow passed after reload-review stabilization: `output/playwright/native_fastapi_agent_workflow_20260625_231835/agent_workflow_qa.json`.
- Settings workflow passed: `output/playwright/native_fastapi_settings_workflow_20260625_232455/settings_workflow.png` and JSON in the same run folder.
- Dictionary/States reference workflow passed: `output/playwright/native_fastapi_reference_routes_20260625_232455/reference_routes.png` and JSON in the same run folder.
- Real extraction job flow passed against `/Volumes/外置硬盘/databases/mimiciv`, one module only:
  - Report screenshot: `output/playwright/native_fastapi_extraction_job_flow_20260625_232526/extraction_job_flow.png`.
  - Generated bounded QA export: `/Users/haibo/.easyicu/exports/easyicu_export_20260625_232530_miiv_parquet`.
  - Verified default Parquet, default all modules, clear/select-all/core controls, sticky export summary, tagged output folder, `_manifest.json`, and `README.md`.

Command evidence:

- `python -m py_compile tools/qa_native_fastapi_button_audit.py tools/qa_native_fastapi_agent_workflow.py tools/qa_native_fastapi_reference_routes.py tools/qa_native_fastapi_settings_workflow.py tools/qa_native_fastapi_extraction_filters.py`
- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
- `python -m compileall -q src/easyicu/webserver`
- `git diff --check`
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py`
  - Result: `119 passed, 1 warning`.

Safety observations:

- Provider status after Settings reset: `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`.
- Agent workflow stayed local for the tested preflight path and restored run review after refresh.
- Human signoff remained `signed_analysis_only`; `reportable=false`, `draft_unlocked=false`.
- Patient row markers were absent in Agent browser QA.

Residual issues / next actions:

- The all-routes single-shot button audit is too slow because it opens a fresh page per candidate; use per-route audits for now or refactor the tool to batch candidates per route.
- Route QA still reports visual overflow/clipping counts on some complex desktop/mobile routes, especially Agent and Cohort mobile. These are polish/layout follow-ups, not backend wiring failures.
- Extraction job QA created a small real QA export under `~/.easyicu/exports`; keep it as evidence or remove manually after review.
- Continue from `WEBAPP-FASTAPI-NATIVE-QA`: implement real Table 1 / missingness / ROC / calibration artifact producers before treating Agent Outputs as complete.

## Agent Outputs real artifact producer follow-up

Additional changes:

- Added `src/easyicu/webserver/agent_outputs.py`.
  - Computes bounded local `table1_summary.json`, `missingness_audit.json`, `roc_curve.json`, and `calibration_curve.json` from the active export.
  - Artifacts are aggregate-only and avoid row-level identifier keys.
  - ROC/calibration fail closed with `status: not_available` when required outcome/predictor conditions are not met.
- Updated `src/easyicu/webserver/agent_runs.py`.
  - The new output artifacts are whitelisted in Agent run review, download, bundle, evidence ledger, privacy scan, signoff hash, and signoff stale detection.
  - Preflight runs now produce 8 local artifacts instead of 4.
- Updated `src/easyicu/webserver/static/js/screens-agent.js`.
  - Output cards label the real Table 1, missingness, ROC, and calibration artifacts.
- Updated `tests/test_webserver_workspace_summary.py`.
  - Tests assert the new artifacts exist, are scan-clean, are signed/bundled, and contain real computed payloads rather than placeholders.

Follow-up verification:

- `python -m py_compile src/easyicu/webserver/agent_outputs.py src/easyicu/webserver/agent_runs.py`
- `python -m compileall -q src/easyicu/webserver`
- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check`
- `ruff check src/easyicu/webserver/agent_outputs.py src/easyicu/webserver/agent_runs.py tests/test_webserver_workspace_summary.py`
- `pytest -q tests/test_webserver_workspace_summary.py -k 'agent_run_job_uses_active_registry or agent_run_review_and_local_signoff or full_agent_mock_run_writes_locked_strict_evidence_artifacts or agent_artifact_privacy_scan'`
  - Result: `4 passed, 84 deselected, 1 warning`.
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py`
  - Result: `119 passed, 1 warning`.
- `python tools/qa_native_fastapi_agent_workflow.py --port 8797 --no-screenshots`
  - Result: passed.
  - Report: `output/playwright/native_fastapi_agent_workflow_20260625_235351/agent_workflow_qa.json`.
  - Browser evidence: Agent Outputs showed 8 real artifact cards: `run_context.json`, `cohort_summary.json`, `table1_summary.json`, `missingness_audit.json`, `roc_curve.json`, `calibration_curve.json`, `quality_gate.json`, `evidence_ledger.json`.
  - Bundle contained the same 8 artifacts.
  - Signoff integrity checked all 8 artifacts.
  - Privacy scan: `row_level_markers=[]`.

Safety observations:

- Provider remained dormant during QA: `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`.
- The generated ROC artifact is real and derived from the fixture export. The calibration artifact honestly returned `not_available` on the 3-stay fixture because at least 5 entities are required for bounded calibration bins.
- The temporary QA server on port `8797` exited; no listener remained afterward.

Residual issues / next actions:

- Refactor `qa_native_fastapi_button_audit.py` to batch candidates per route instead of reopening a page for every candidate.
- Continue Agent/Cohort mobile clipping and spacing polish.
- Continue replacing any remaining fake/seeded result paths with real backend results or explicit fail-closed unavailable states.

## Button audit follow-up and stale-server finding

Additional changes:

- Updated `tools/qa_native_fastapi_button_audit.py`.
  - Added `--collect-only`, `--progress`, and configurable wait arguments so the audit can run in short route/viewport slices instead of silently spending minutes reopening pages.
  - The tool now prints per-route candidate/click/skip counts while it runs.

Debug finding:

- Running the audit against the existing user-visible `127.0.0.1:8782` server showed `POST /api/guided/session -> 405`.
- The current code has this POST route, so the failure was traced to a stale uvicorn process rather than browser cache or a current-code frontend bug.
- A fresh current-code server on `127.0.0.1:8799` reproduced the same `Start study -> #guided` click with no 4xx responses.

Verification:

- `python -m py_compile tools/qa_native_fastapi_button_audit.py`
- `ruff check tools/qa_native_fastapi_button_audit.py`
- Collect-only baseline on current visible routes:
  - `python tools/qa_native_fastapi_button_audit.py --base-url http://127.0.0.1:8782/ --scope content --collect-only --progress --out-dir output/playwright`
  - Report: `output/playwright/native_fastapi_button_audit_20260626_000229/button_audit.json`
  - Result: 194 visible controls classified, no clicking.
- Current-code click audit on fresh port 8799:
  - `python tools/qa_native_fastapi_button_audit.py --base-url http://127.0.0.1:8799/ --scope content --max-clicks 4 --progress --out-dir output/playwright --fail-on-noop`
  - Report: `output/playwright/native_fastapi_button_audit_20260626_000658/button_audit.json`
  - Result: 196 candidates, 69 safe sampled clicks, 69 changed, 0 no-op, 0 click errors, 0 console errors.

Residual issues / next actions:

- The button audit is now usable, but it is still a sampled audit (`--max-clicks 4`) rather than exhaustive click coverage.
- For exhaustive all-controls QA, continue improving the tool to reuse page state or run per-route batches so it does not take several minutes.
- The stale 8782 process should be restarted before user-facing visual review, otherwise newly implemented backend routes may look broken even when current code is correct.
