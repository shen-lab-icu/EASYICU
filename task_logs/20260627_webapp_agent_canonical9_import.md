# 2026-06-27 WebApp Agent canonical9 import

## Scope

Imported the completed Fig 2 canonical9 aware benchmark runs into the native WebApp Agent Projects screen as read-only reviewable projects. This is for group-meeting demonstration and framework inspection; it does not mark Fig 2 manuscript results as frozen or reportable.

## Changed files

- `tools/import_canonical9_to_agent_projects.py`
- `src/easyicu/webserver/agent_runs.py`
- `src/easyicu/webserver/static/js/screens-agent.js`
- `src/easyicu/webserver/static/css/agent.css`
- `src/easyicu/webserver/static/index.html`
- `tests/test_webserver_static_routes.py`
- `tests/test_webserver_workspace_summary.py`

## Import result

Source: `research_output/_parallel_obs_20260613`

Command:

```bash
python3 tools/import_canonical9_to_agent_projects.py --dry-run
python3 tools/import_canonical9_to_agent_projects.py
```

Imported 9 Agent project facades under `~/easyicu/projects/fig2-*`, each with 10 bounded JSON artifacts:

- `run_context.json`
- `cohort_summary.json`
- `quality_gate.json`
- `agent_plan.json`
- `manuscript_draft.json`
- `benchmark_scorecard.json`
- `workflow_graph.json`
- `figure_gallery.json`
- `source_run_manifest.json`
- `evidence_ledger.json`

Seed index: `~/.easyicu/webserver_agent_project_seeds.json`

Observed project API state:

```text
POST /api/ideas/agent-projects -> 200
projects 12 canonical9 9
E1/E2/E3/H2/H3/M2: gate_reportable source disposition
H1/M3: analysis_only source disposition
M1: diagnostic_only source disposition
```

All imported Web artifacts remain `reportable: false`, `draft_unlocked: false`, and `analysis_only` or `blocked` in the local WebApp gate until the Fig 2 soft-vs-strict strategy is resolved.

## API verification

Checked one imported run:

```text
POST /api/agent-runs/review
project_dir=/Users/haibo/easyicu/projects/fig2-e1-sepsis3-mortality/run_20260613T004906_66dc3b
status 200 ok True
artifact_payloads include benchmark_scorecard.json, workflow_graph.json, figure_gallery.json, source_run_manifest.json
figure_gallery figures: 3
```

## Tests

```bash
python3 -m py_compile tools/import_canonical9_to_agent_projects.py
./.venv/bin/python -m pytest \
  tests/test_webserver_static_routes.py::test_native_assistant_labels_disambiguate_page_guide_guided_copilot_and_agent_guide \
  tests/test_webserver_static_routes.py::test_native_agent_outputs_fail_closed_to_real_artifacts \
  tests/test_webserver_static_routes.py::test_native_agent_research_blocks_are_project_owned \
  tests/test_webserver_static_routes.py::test_native_agent_canonical9_import_is_project_owned \
  tests/test_webserver_workspace_summary.py::test_agent_run_review_and_local_signoff_write_safe_artifact \
  tests/test_webserver_workspace_summary.py::test_agent_run_review_exposes_canonical9_import_artifacts \
  tests/test_webserver_workspace_summary.py::test_idea_mining_lists_only_existing_local_runs_and_projects \
  -q
```

Result: `7 passed, 1 warning`.

CSS owner scan:

```text
agent.css braces 86 86 balanced True
canonical9 marker True
screens.css/redesign.css/ideas.css/guided.css/cohort.css/extraction.css/pages.css foreign_hits []
```

## Browser QA

Started current WebApp code on `http://127.0.0.1:8790/#agent` and inspected with Playwright CLI.

Desktop 1440x1000:

- Project list showed canonical9 imports at top.
- E1 overview showed canonical9 status, cohort, evidence count, missing evidence, errors, and score bars.
- `Open imported run` loaded the real local run through `/api/agent-runs/review -> 200`.
- Outputs tab showed 10 artifacts, including Benchmark scorecard, Workflow graph, Figure gallery, Source run manifest.
- Figure gallery rendered 3 embedded PNGs with natural sizes 1492x1071, 1400x800, and 1600x1000.
- Console: 0 errors, 0 warnings.
- Horizontal overflow: `body.scrollWidth == clientWidth == 1440`.

Desktop 1280x800:

- Real mode showed `研究项目 · 12`.
- Demo project labels were absent.
- Canonical9 count was 9.
- Horizontal overflow: `body.scrollWidth == clientWidth == 1280`.

Screenshot:

- `output/playwright/canonical9_agent_import/agent_figure_gallery_1440x1000.png`

Python `tools/qa_native_fastapi_routes.py` was not run because the project `.venv` currently lacks the Python `playwright` package; browser checks above used the bundled Playwright CLI instead.

## Follow-up: gallery diversity fix

After inspecting the page, the first gallery pass over-emphasized shared `publication_figure`, missingness, probe, and audit figures. That made different benchmark questions look visually too similar even though the original run folders contained task-specific figures.

Patch:

- Added task-specific figure priorities in `tools/import_canonical9_to_agent_projects.py`.
- Reimported the 9 read-only WebApp project facades.
- Kept the original benchmark run folders unchanged.

Updated first gallery figures:

```text
E1: sepsis3 prevalence stages
E2: lactate mortality association plot
E3: Publication figure / join semantics audit (only two available source figures)
M1: forest plot primary and complete case estimates
M2: discrimination calibration
M3: clustering embedding or heatmap
H1: adjusted effect plot
H2: propensity score overlap
H3: trajectory cluster profiles
```

Verification:

```bash
python3 -m py_compile tools/import_canonical9_to_agent_projects.py
python3 tools/import_canonical9_to_agent_projects.py
./.venv/bin/python -m pytest \
  tests/test_webserver_workspace_summary.py::test_agent_run_review_exposes_canonical9_import_artifacts \
  tests/test_webserver_static_routes.py::test_native_agent_canonical9_import_is_project_owned \
  -q
```

Result: `2 passed, 1 warning`.

Playwright UI check on `#agent`: selected H1, opened imported run, opened `figure_gallery.json`; gallery order was `adjusted effect plot`, `Publication figure`, `probe exposure distribution`, `probe selected distributions`. All four images completed loading and console remained 0 errors / 0 warnings.

## Presentation recommendation

For group meeting, use Real mode on `#agent`, select E1/E2/M2/H2 as fast successful examples and M1/H1/M3 as honest analysis-only/diagnostic examples. The strongest demo path is:

1. Agent Projects list: show nine `九问运行`.
2. E1 overview: show plan, gate state, score bars, evidence count.
3. Open imported run: show 10 local artifacts.
4. Open Figure gallery: show generated figures and the JSON preview with embedded image payload hidden.
5. Mention that Fig 2 manuscript freeze is still pending soft-vs-strict strategy, so the WebApp is showing auditability rather than claiming final reportability.

## Follow-up: group-meeting Agent UX cleanup

User browser review found three presentation problems: the left rail used awkward `想法种子` / `九问运行` labels, imported question outputs only appeared after opening History -> Review, and the review tab looked like a broken draft/sign-off gate for read-only canonical9 runs.

Patch:

- Replaced visible Agent labels with `已完成分析` for canonical9 imports and `研究想法` for Idea Mining handoffs.
- Added `reviewableRunForStudy()` so imported canonical9 packages behave like reviewable local runs in Outputs and Review without requiring the History tab.
- Changed output cards from file-browser style JSON labels to presentation-first cards: Figure gallery, Benchmark scorecard, Workflow graph, Quality check, Evidence ledger, Plan, Locked claims, Context, Cohort, and Provenance.
- Auto-loads the task-specific `figure_gallery.json` when opening Outputs.
- Changed canonical9 `Draft` tab label to `审阅` and rendered a read-only review path without local sign-off controls.

Focused verification:

```bash
/Users/haibo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node --check src/easyicu/webserver/static/js/screens-agent.js
./.venv/bin/python -m pytest \
  tests/test_webserver_static_routes.py::test_native_agent_outputs_fail_closed_to_real_artifacts \
  tests/test_webserver_static_routes.py::test_native_agent_research_blocks_are_project_owned \
  tests/test_webserver_static_routes.py::test_native_agent_canonical9_import_is_project_owned \
  tests/test_webserver_static_routes.py::test_native_ui_uses_verification_terms_instead_of_gate_literal_translations \
  tests/test_webserver_workspace_summary.py::test_agent_run_review_exposes_canonical9_import_artifacts \
  -q
```

Result: `5 passed, 1 warning`.

In-app browser verification on `http://127.0.0.1:8790/#agent`:

- Left rail no longer contains `想法种子`, `九问运行`, `Idea seed`, or `Canonical run`.
- E1 `产出 10` opens directly without History and shows 10 readable cards plus 3 figure-gallery images.
- E2 output gallery differs from E1 (`lactate mortality association`, `lactate distribution`, `robustness`) confirming task-specific visuals.
- `审阅` tab shows `只读审阅` and `展示路径`; no `不可用`, no `签署前已阻断`, and no local sign-off controls.
- Current in-app viewport check for Outputs and Review: `offscreenCount=0`, `clippedCount=0`, no horizontal overflow.

Full `tests/test_webserver_static_routes.py -q` was also run. Agent-related and UI-terminology tests passed, but five unrelated pre-existing failures remain in Extraction/Sepsis cache-bust and manifest-contract assertions; they were not introduced by this Agent patch.

## Follow-up: presenter-view audit

The in-app browser was re-checked from the standpoint of a group-meeting presenter: can the presenter explain the project, question, result figures, scorecard, and evidence boundary without exposing backend implementation terms?

Patch:

- Added a canonical9 `汇报摘要 / Study brief` card on Overview so one selected question reads as a completed evidence-bound Agent analysis, not a raw project folder.
- Mapped internal run states through `runStatusLabel()` in Overview, Outputs, Benchmark, and Review surfaces.
- Hid the `Open Idea Mining / 打开 Idea 挖掘` action for imported canonical9 projects.
- Renamed canonical9 pipeline ending from draft-oriented wording to `Review / 审阅`.
- Added a featured result-figure block above the output cards when `figure_gallery.json` is selected or auto-loaded.
- Added `readableArtifactText()` for read-only rendering of artifact claim text, so strings such as `gate_reportable` and `awaiting_human_signoff` remain in source artifacts but are not shown as audience-facing labels.

Verification:

```bash
/Users/haibo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node --check src/easyicu/webserver/static/js/screens-agent.js
./.venv/bin/python -m pytest \
  tests/test_webserver_static_routes.py::test_native_agent_outputs_fail_closed_to_real_artifacts \
  tests/test_webserver_static_routes.py::test_native_agent_research_blocks_are_project_owned \
  tests/test_webserver_static_routes.py::test_native_agent_canonical9_import_is_project_owned \
  tests/test_webserver_static_routes.py::test_native_ui_uses_verification_terms_instead_of_gate_literal_translations \
  tests/test_webserver_workspace_summary.py::test_agent_run_review_exposes_canonical9_import_artifacts \
  -q
```

Result: `5 passed, 1 warning`.

CSS / owner scan:

```text
ag-present-brief and ag-featured-results only appear in Agent owner files:
src/easyicu/webserver/static/js/screens-agent.js
src/easyicu/webserver/static/css/agent.css
tests/test_webserver_static_routes.py
git diff --check: clean
```

In-app browser QA on `http://127.0.0.1:8790/#agent`:

- E1 Overview shows `汇报摘要`, `核验通过`, 10 outputs, and read-only boundary; no visible `gate_reportable`, no `打开 Idea 挖掘`, no `不可用`.
- E1 Outputs shows `主要审阅产出` and `结果图件` above the cards; figure gallery contains task-specific Sepsis-3 prevalence, publication forest, and missingness figures.
- E2 Outputs shows task-specific lactate figures: lactate mortality association, lactate distribution, robustness, and probe distribution. This confirmed the questions no longer look visually identical.
- E2 Review shows `待审阅`, read-only route actions, and no sign-off button. The raw source sentence `gate_reportable` is displayed as `核验通过` in the read-only claim view.
- Browser dimensions during QA: `innerWidth=1230`, document `scrollWidth=1215`, no horizontal overflow.

Screenshots:

- `output/product_audit/agent_group_meeting_20260627/08-e1-outputs-viewport.png`
- `output/product_audit/agent_group_meeting_20260627/09-e1-outputs-results-viewport.png`
- `output/product_audit/agent_group_meeting_20260627/10-e2-outputs-results-viewport.png`
- `output/product_audit/agent_group_meeting_20260627/13-e2-review-final-no-raw-status.png`

## Follow-up: evidence-link and cross-data capability surfacing

User review identified two under-emphasized story points in Agent Projects: the evidence-link mechanism should be a visible highlight, and the Agent module should explain how cross-database analysis connects to the Agent workflow.

Patch:

- Added `Evidence Link / 证据链接` capability card to Overview and Outputs.
- The card summarizes claim-to-artifact traceability as `Claim -> Evidence ID -> SHA-256 -> Gate`, with evidence item count, hashed artifact count, locked claim count, and missing-evidence count.
- Added direct actions from the card to `evidence_ledger.json` and `source_run_manifest.json`.
- Added `Cross-data scope / 跨数据范围` card to state the current data context before claims are shown.
- For canonical9 imports, the card truthfully labels the current scope as `MIMIC-IV canonical benchmark universe` and points users to Cross-DB workspace for six-database comparison, instead of overstating each canonical9 run as a cross-database result.
- Moved new capability styles into `src/easyicu/webserver/static/css/agent-capabilities.css` and wired it explicitly in `index.html`, rather than expanding the already-large `agent.css`.

Verification:

```bash
/Users/haibo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node --check src/easyicu/webserver/static/js/screens-agent.js
git diff --check -- src/easyicu/webserver/static/js/screens-agent.js src/easyicu/webserver/static/css/agent-capabilities.css src/easyicu/webserver/static/index.html tests/test_webserver_static_routes.py
./.venv/bin/python -m pytest \
  tests/test_webserver_static_routes.py::test_native_agent_outputs_fail_closed_to_real_artifacts \
  tests/test_webserver_static_routes.py::test_native_agent_canonical9_import_is_project_owned \
  tests/test_webserver_workspace_summary.py::test_agent_run_review_exposes_canonical9_import_artifacts \
  -q
```

Result: `3 passed, 1 warning`.

In-app browser QA on `http://127.0.0.1:8790/#agent`:

- Overview shows 2 capability cards.
- Evidence card displays `154` evidence items, `9` hashed artifacts, `3` locked claims, and `0` missing evidence for E1.
- Cross-data card displays `MIMIC-IV canonical benchmark universe`, denominator `94,458`, and a direct `打开跨库工作台` action.
- `打开证据账本` jumps to Outputs and selects `evidence_ledger.json`; no visible `gate_reportable`, no `不可用`, no horizontal overflow.
- `打开跨库工作台` navigates to `#crossdb`; browser confirmed title `跨库对比` and no unavailable state.

Screenshots:

- `output/product_audit/agent_group_meeting_20260627/14-agent-capabilities-overview.png`
- `output/product_audit/agent_group_meeting_20260627/15-agent-evidence-ledger-jump.png`
- `output/product_audit/agent_group_meeting_20260627/16-agent-crossdb-jump.png`

## Follow-up: structured research-question display

User review found that long benchmark prompts were rendered as one dense paragraph even when they contained numbered requirements. This made the Agent overview hard to explain in group meeting.

Patch:

- Added `questionParts(text)` and `renderStructuredQuestion(s)` in `src/easyicu/webserver/static/js/screens-agent.js`.
- The overview now renders the prompt as `核心问题`, `数据上下文`, and an ordered `任务要求` list.
- The parser handles both newline-preserved numbered prompts and folded single-line prompts such as `...: 1. ... 2. ...`.
- Question tags are now inferred from the actual prompt text, so E1 shows `@sofa`, E2 shows `@lactate`, ventilation tasks show `@ventilation`, etc., instead of defaulting non-AKI tasks to `@lactate`.
- The question card moved to the first overview body block, before summary/capability cards, so the presentation order is question -> evidence/capability -> outputs.
- Added `focusAgentBody()` so selecting another project or tab scrolls the right-side content into view. This fixes the confusing feeling that clicking E1/E2 did not visibly open anything when the page was still at the top.
- New owner CSS file: `src/easyicu/webserver/static/css/agent-question.css`, wired explicitly in `index.html`.

Verification:

```bash
/Users/haibo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node --check src/easyicu/webserver/static/js/screens-agent.js
git diff --check -- src/easyicu/webserver/static/js/screens-agent.js src/easyicu/webserver/static/css/agent-question.css src/easyicu/webserver/static/index.html tests/test_webserver_static_routes.py
./.venv/bin/python -m pytest \
  tests/test_webserver_static_routes.py::test_native_agent_outputs_fail_closed_to_real_artifacts \
  tests/test_webserver_static_routes.py::test_native_agent_canonical9_import_is_project_owned \
  tests/test_webserver_workspace_summary.py::test_agent_run_review_exposes_canonical9_import_artifacts \
  -q
```

Result: `3 passed, 1 warning`.

In-app browser QA on `http://127.0.0.1:8790/#agent`:

- E2 Overview shows the question card at the top of the viewport after selecting the project.
- Sections shown: `核心问题`, `数据上下文`, `任务要求`.
- E2 has 4 independent requirement rows and tags `@MIMIC-IV canonical benchmark universe`, `@first_24h`, `@lactate`.
- No horizontal overflow in the question card; no visible `不可用` / `unavailable`.
- Outputs still show 10 artifact cards; Review still shows read-only review and no unavailable state.

Screenshot:

- `output/product_audit/agent_group_meeting_20260627/17-agent-structured-question-e2.png`
