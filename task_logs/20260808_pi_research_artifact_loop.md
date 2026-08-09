# Pi research workflow and governed artifact preview — 2026-08-08

## Scope

- Task: `PI-COPILOT-RESEARCH-CLOSURE`
- Module: `web`
- Branch: `feat/pi-copilot-shell`
- Production commit: `102ee52` (`feat(web): close Pi research artifact loop`)
- Goal: connect a scientific question to conversational StudyContext setup, an authorized EasyICU preflight, real run artifacts, and a clickable right-side preview without moving scientific execution or evidence authority into Pi.

## Implemented contracts

- New Pi conversations default to Research workflow; Workspace mode remains an explicit artifact-authoring option.
- The research system prompt follows the existing owner sequence: inspect typed context, ask for missing slots, save only with the one-turn `configure` grant, start only an authorized deterministic preflight, then inspect run/validation/evidence/artifacts on a later bound turn.
- Successful StudyContext updates and run submissions mark the turn authority stale as before, but the browser host now rebinds automatically after the Pi turn settles. The user is not left behind a manual stale-state banner.
- `easyicu_inspect_plan`, `easyicu_inspect_validation`, `easyicu_inspect_evidence`, and `easyicu_list_artifacts` emit path-free browser resources. A resource contains only `project_id` at the route boundary plus `run_id`, a whitelisted JSON artifact name, label, and media type.
- The new research-artifact route resolves `project_id -> StudyContext -> run_id -> host project_dir` inside the FastAPI owner. `project_dir` and absolute source paths are removed before the payload reaches the browser. Cross-project run IDs, non-whitelisted names, privacy-scan failures, and oversized previews fail closed with stable codes.
- The Guided right panel now accepts project files, sandboxed webpages, and Research Agent artifacts. Research artifacts open in a readable structured/table view with an audit JSON tab, reusing the pure `AGENT_RENDER` renderer contract.

## Real model and browser canary

The live route `http://127.0.0.1:8765/?ui=20260808-research-flow1#guided` was exercised in Chinese with the configured local `gpt-5.6-luna` endpoint and the existing demo project `New local study`.

1. Read-only request: the model called `easyicu_workspace_status`, `easyicu_inspect_context`, and `easyicu_inspect_capability`; it reported the missing typed slots and correctly stated that no run meant no real artifacts.
2. Authorized setup request: the model called `easyicu_update_study_context`; revision advanced from 1 to 2. The turn ended with no stale banner and the composer remained enabled, proving browser-host auto-rebind.
3. Authorized preflight request: the model inspected context/status/capabilities and submitted one deterministic job `cc107553bdda`. A same-turn post-mutation inspection was correctly blocked by `pi_session_authority_stale`; the host rebound after settlement.
4. Follow-up request: the model read the completed job, validation, evidence, and artifacts. The persisted run was `run_cc107553bdda`; job status was `done`; the quality state remained `analysis_only`, 5/6 checks passed, and `human_signoff` remained required. No full run or external scientific provider call occurred.
5. The artifact tool projected 8 real run artifacts: `run_context.json`, `cohort_summary.json`, `table1_summary.json`, `missingness_audit.json`, `roc_curve.json`, `calibration_curve.json`, `quality_gate.json`, and `evidence_ledger.json`.
6. Clicking `Table 1 摘要` replaced the study-progress aside with the governed preview. The readable view showed denominator 140 and rows `overall=140`, `survived=125`, `deceased=15`; the JSON tab remained available for audit.

Desktop/laptop overflow check at `1141 x 994`:

- document `scrollWidth/clientWidth = 1141/1141`
- conversation `525/525`
- transcript `525/525` (vertical scrolling only)
- preview `615/615`
- readable artifact `613/613`

## Verification

- `ruff check ...`: passed
- Node syntax / Pi package check: passed
- `pytest -q tests/test_pi_copilot_contract.py tests/test_pi_copilot_routes.py tests/test_pi_copilot_gateway.py tests/test_pi_copilot_static.py tests/test_webserver_static_routes.py`: `133 passed`
- `git diff --check`: passed
- Browser: real model, real EasyICU preflight, tool timeline, artifact clicks, structured preview, close/restore layout, and desktop overflow all verified.

## Honest remaining boundary

- Pi still starts only the deterministic local preflight. Full/provider research execution retains its existing dedicated confirmation and provider gate.
- A submitted background preflight is inspected on a subsequent user turn after authority rebind; the product does not silently spend another model call when the background job finishes.
- This canary proved a real table and structured ROC/calibration artifacts. Rendered image files appear only when their existing Agent/figure owner actually produces a whitelisted figure-gallery payload; the Copilot does not invent figures.
