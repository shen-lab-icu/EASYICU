# 2026-07-03 Agent Science Workbench Iteration Log

## Iteration 20: Real Export Discovery and Large-Export Agent Preflight Fast Path

Problem found:

- A real full MIMIC-IV export was available at `/Volumes/外置硬盘/easyicu_fullexport_miiv_20260622` with 94,458 stays, 19 modules, and 98,596,746 declared rows.
- Idea discovery and prior-art opt-in worked on this export and produced a ready Agent project seed, but the first Agent preflight timed out after resolving the source.
- The cause was not UI state: preflight was using the visualization workspace summarizer and output artifact builder, both of which attempted row-level reads that are inappropriate for a full export.

Changes:

- Added a large-export preflight fast path in `src/easyicu/webserver/agent_runs.py`.
  - Preflight uses registry/manifest metadata when the active export declares more than 1,000,000 rows.
  - The snapshot records `snapshot_basis=registry_metadata`, `artifact_scope=metadata_only_large_export_preflight`, and the row-scan skip reason.
  - The quality check accepts `manifest_file_inventory` for preflight while preserving the existing row-intersection audit for small exports and richer UI views.
- Added metadata-only Agent output artifacts in `src/easyicu/webserver/agent_outputs.py`.
  - `table1_summary.json` and `missingness_audit.json` are explicitly marked `metadata_only`.
  - ROC and calibration artifacts are `not_available` with a reason that a bounded sampled cohort is required.
  - Direct identifier columns remain excluded from persisted artifacts.
- Added a regression test for large-export Agent preflight in `tests/test_webserver_workspace_summary.py`.

Real-export E2E evidence:

- `task_logs/20260703_real_export_discovery_e2e.json`
  - Registered the full export as active and set WebApp data mode to real.
  - Verified discovery blocks without network opt-in, then PubMed opt-in performed 5 metadata calls.
  - Idea Mining found a ready export contract: 94,458 entities, 19 modules, 98,596,746 rows, no missing required concepts.
  - Prior-art review completed with status `searched`.
  - Agent project seed had `agent_run_ready_after_human_confirmation=true` and no blockers.
- `task_logs/20260703_real_export_agent_preflight_fastpath.json`
  - Re-ran the same seed after the fix.
  - Agent preflight completed in 0.87 seconds.
  - Final status: `done`; gate status: `analysis_only`; uploads/tokens: 0.
  - Output artifacts: `table1_summary.json` status `metadata_only`, `missingness_audit.json` status `metadata_only`, `roc_curve.json` status `not_available`.

Verification:

- Syntax:
  - `.venv/bin/python -m py_compile src/easyicu/webserver/agent_runs.py src/easyicu/webserver/agent_outputs.py tests/test_webserver_workspace_summary.py`: passed.
- Lint:
  - `.venv/bin/ruff check src/easyicu/webserver/agent_runs.py src/easyicu/webserver/agent_outputs.py tests/test_webserver_workspace_summary.py`: passed.
- Focused tests:
  - `.venv/bin/pytest -q tests/test_webserver_workspace_summary.py::test_agent_run_job_uses_active_registry_and_writes_bounded_artifacts tests/test_webserver_workspace_summary.py::test_agent_run_large_export_preflight_uses_registry_metadata_fast_path tests/test_webserver_workspace_summary.py::test_idea_mining_real_export_and_prior_art_unlock_agent_run_gate`: 3 passed, 1 warning.
- Wider WebApp regressions:
  - `.venv/bin/pytest -q tests/test_webserver_workspace_summary.py -k 'idea_mining or agent_run_job'`: 9 passed, 109 deselected, 1 warning.
  - `.venv/bin/pytest -q tests/test_webserver_idea_sources.py tests/test_webserver_capabilities.py tests/test_webserver_science_workbench.py tests/test_webserver_static_routes.py`: 75 passed, 1 warning.
- Plan hygiene:
  - `.venv/bin/python tools/lint_main_plan.py`: passed before updating the main plan.

Next:

- Run one additional editorial/review-derived seed on the same full export so Figure 5 has a small candidate ledger rather than a single-example smoke test.
- Decide whether the Figure 5 source-data protocol needs a bounded sample stage after metadata-only preflight, or whether the next step should hand off to the full research-agent discovery pipeline.

## Iteration 21: Second Review Seed and Idea Mining Large-Module Fast Path

Problem found:

- A second ARDS/mechanical-ventilation seed initially timed out in `/api/ideas/mine` on the full MIIV export.
- The bottleneck was analogous to Iteration 20 but one step earlier: Idea Mining feasibility tried to read full respiratory/ventilator feature columns for large modules before creating the candidate ledger.
- This would bias discovery toward ideas that happen to use small modules, rather than clinically useful respiratory questions.

Changes:

- Added a per-feature row-count guard in `src/easyicu/webserver/ideas/mining.py`.
  - If a mapped concept lives in a module with more than 1,000,000 declared rows, Idea Mining records schema/manifest feasibility instead of loading the full column.
  - Metadata-only feature stats are explicit: `status=metadata_only`, `metric_kind=schema_presence`, `coverage_basis=manifest_file_inventory`, and the scan limit are persisted.
  - Small modules and small fixtures still use the existing row-level event-rate/coverage logic.
- Added `test_idea_mining_large_module_uses_metadata_only_feature_stats` in `tests/test_webserver_workspace_summary.py`.
  - The fixture declares a 2,000,000-row ventilator module but writes only a tiny CSV.
  - The test verifies that `peep` is metadata-only, `death` remains row-level event-rate, and no direct identifiers are returned.

Second seed evidence:

- Source: `Driving pressure in mechanical ventilation: A review`, World Journal of Critical Care Medicine, 2024, PMID `38633474`, DOI `10.5492/wjccm.v13.i1.88385`.
  - Source lookup references used for metadata: PubMed `https://pubmed.ncbi.nlm.nih.gov/38633474/` and F6Publishing article record `https://www.f6publishing.com/PublishedArticleProcessDetail/88385`.
- E2E artifact: `task_logs/20260703_second_review_seed_pubmed_source_e2e.json`.
  - Idea title: `Mechanical Ventilation and In-hospital Mortality in adult ICU patients`.
  - Full export feasibility: `ready`.
  - Mapped concepts: `mech_vent`, `death`, `driving_pres`, `fio2`, `peep`.
  - Metadata-only concepts: `mech_vent`, `fio2`; row-level ready concepts: `death`, `driving_pres`, `peep`.
  - Prior-art search: `searched`, 14 metadata hits.
  - Agent preflight: `done`, seed project root, `snapshot_basis=registry_metadata`, gate `analysis_only`.
- Candidate comparison artifact: `task_logs/20260703_discovery_candidate_comparison.json`.
  - Preferred Fig5 protocol candidate: `candidate_2_driving_pressure_ventilation`.
  - Rationale: cleaner review-source provenance and clinically coherent respiratory/ventilator question.
  - Boundary: not Fig5 source data yet; respiratory features need bounded sample or full pipeline confirmation because two concepts are metadata-only and two measured concepts have low row-level coverage.

Verification:

- Syntax:
  - `.venv/bin/python -m py_compile src/easyicu/webserver/ideas/mining.py tests/test_webserver_workspace_summary.py`: passed.
- Lint:
  - `.venv/bin/ruff check src/easyicu/webserver/ideas/mining.py tests/test_webserver_workspace_summary.py`: passed.
- Focused tests:
  - `.venv/bin/pytest -q tests/test_webserver_workspace_summary.py::test_idea_mining_large_module_uses_metadata_only_feature_stats tests/test_webserver_workspace_summary.py::test_idea_mining_real_export_and_prior_art_unlock_agent_run_gate`: 2 passed, 1 warning.
- Wider regressions:
  - `.venv/bin/pytest -q tests/test_webserver_workspace_summary.py -k 'idea_mining or agent_run_job'`: 10 passed, 109 deselected, 1 warning.
  - `.venv/bin/pytest -q tests/test_webserver_idea_sources.py tests/test_webserver_capabilities.py tests/test_webserver_science_workbench.py tests/test_webserver_static_routes.py`: 75 passed, 1 warning.
- Plan hygiene:
  - `.venv/bin/python tools/lint_main_plan.py`: passed before updating the main plan.

Next:

- Implement a bounded sample feasibility stage for candidate 2: compute row-level coverage and denominator checks for `mech_vent`, `fio2`, `driving_pres`, `peep`, and `death` without full-module scans.
- Only after that bounded sample passes should the candidate be promoted into the full research-agent discovery pipeline for Fig5 source-data generation.

## Iteration 22: Bounded Sample Feasibility for Driving-Pressure Candidate

Problem found:

- Iteration 21 correctly prevented full-module scans, but metadata-only feature presence was not enough for Fig5 triage.
- The UI also rendered metadata-only respiratory concepts as `0%` coverage, which was misleading.
- On the first real bounded sample run, `mech_vent` was incorrectly displayed as 0% because the MIMIC-IV export encodes mechanical ventilation as `invasive` / `noninvasive`, not `true` / `1`.

Changes:

- Added `/api/ideas/bounded-feasibility` in `src/easyicu/webserver/app.py`.
- Added `bounded_sample_feasibility()` in `src/easyicu/webserver/ideas/mining.py`.
  - Reads only capped samples per feature, with default 100,000 and hard cap 250,000 records per feature.
  - Uses `pyarrow.parquet.ParquetFile.iter_batches()` for parquet so large respiratory/ventilator modules are not full-scanned.
  - Persists `bounded_sample_feasibility.json` under the idea run directory.
  - Marks the result `claim_level=feasibility_sample_not_reportable`.
- Extended boolean/event parsing to treat `invasive`, `noninvasive`, and `non-invasive` as positive values.
- Wired the UI in `src/easyicu/webserver/static/js/api.js` and `src/easyicu/webserver/static/js/screens-ideas.js`.
  - Added a real "Run bounded sample check" action in the feasibility panel.
  - History-loaded idea runs now show the persisted bounded sample artifact.
  - Metadata-only concepts display as schema present / sample required instead of fake 0% coverage.
  - The sample result text is localized in the current UI language rather than exposing raw backend technical copy.
- Added regression coverage in `tests/test_webserver_workspace_summary.py` and `tests/test_webserver_static_routes.py`.

Real full-export evidence:

- Artifact copied to `task_logs/20260703_driving_pressure_bounded_sample_feasibility.json`.
- Input run: `idea_20260702_232750_931377000_8e954325`, source PMID `38633474`.
- Active export: `/Volumes/外置硬盘/easyicu_fullexport_miiv_20260622`.
- Bounded sample status: `needs_review`, not reportable.
- Required concepts were all sample-checked:
  - `mech_vent`: event rate 34.2% in sample.
  - `death`: event rate 10.0%.
  - `driving_pres`: coverage 81.7%.
  - `fio2`: coverage 48.9%, remaining coverage risk.
  - `peep`: coverage 96.9%.
- Interpretation: candidate remains viable for Fig5 protocol triage, but FiO2 coverage requires denominator/missingness review before full Agent execution.

Browser QA:

- In-app browser route: `http://127.0.0.1:8765/?science_v=20260703-bounded#ideas`.
- Loaded the driving-pressure history run, opened the feasibility panel, and verified:
  - New bounded sample button is present.
  - Persisted sample result is visible.
  - `mech_vent` sample event rate is shown as 34.2%, not 0%.
  - `fio2` sample coverage is shown as 48.9%.
  - Metadata-only concepts show schema-present/sample-required wording.
  - No horizontal overflow at the current 1265 px desktop viewport.

Verification:

- Syntax:
  - `.venv/bin/python -m py_compile src/easyicu/webserver/ideas/mining.py src/easyicu/webserver/app.py tests/test_webserver_workspace_summary.py tests/test_webserver_static_routes.py`: passed.
- Lint:
  - `.venv/bin/ruff check src/easyicu/webserver/ideas/mining.py src/easyicu/webserver/app.py tests/test_webserver_workspace_summary.py tests/test_webserver_static_routes.py`: passed.
- Focused tests:
  - `.venv/bin/pytest -q tests/test_webserver_workspace_summary.py::test_idea_mining_large_module_uses_metadata_only_feature_stats`: 1 passed, 1 warning.
  - `.venv/bin/pytest -q tests/test_webserver_static_routes.py::test_native_idea_mining_is_first_class_route_and_backend_wired tests/test_webserver_static_routes.py::test_native_ui_uses_verification_terms_instead_of_gate_literal_translations`: 2 passed, 1 warning.
- Wider regressions:
  - `.venv/bin/pytest -q tests/test_webserver_workspace_summary.py -k 'idea_mining or agent_run_job'`: 10 passed, 109 deselected, 1 warning.
  - `.venv/bin/pytest -q tests/test_webserver_idea_sources.py tests/test_webserver_capabilities.py tests/test_webserver_science_workbench.py tests/test_webserver_static_routes.py`: 75 passed, 1 warning.
- Server:
  - Restarted local server on `127.0.0.1:8765`; current uvicorn PID was `84620` after restart.

Next:

- Add a small denominator/missingness review for the driving-pressure candidate, focused on FiO2 and the respiratory/ventilator sample windows.
- If that review is acceptable, hand this candidate to the full discovery-to-manuscript pipeline for Fig5 source-data generation.
