# Copilot native Data Extraction flow

Date: 2026-08-23
Branch: `codex/easyicu-desktop-app-v1`
Base checkpoint: `d5a1c29`
Status: implemented and locally verified; not merged or pushed

## Outcome

Formal API-backed EasyICU Copilot can return a typed `native_workspace` resource for Data Extraction. Clicking that resource mounts the existing Data Extraction owner in Copilot's right preview; it does not render a second simplified extraction card. Local paths and patient rows remain in the host UI, while the model receives only the path-free workspace coordinate and governed receipts.

The conversation-to-extraction round trip now covers:

1. Copilot calls `easyicu_start_extraction` after the extraction grant.
2. Incomplete setup still returns an actionable native workspace in `setup` state.
3. The right preview scans local ICU folders, distinguishes prepared database roots from existing EasyICU module exports, and runs the existing extraction pipeline.
4. Sync persists the StudyContext handoff and rebinds the Copilot session/workflow projection.
5. Existing validated exports return `review` state and are reused instead of starting a duplicate extraction.

The embedding adapter is isolated in `screens-extraction-embedded.js`; `screens-extraction.js` exposes the small `EU_EXTRACTION_NATIVE_OWNER` contract. This keeps Copilot preview behavior out of the already-large route owner while preserving one extraction implementation.

## Fail-closed defect found by real extraction

The initial real MIMIC-IV run failed at cohort selection because the recommended preset requested `first_icu_stay`, a patient-global criterion that the current MIMIC extract intentionally does not claim without a history-completeness receipt. A second run exposed that `False` was being passed to `PatientFilter.first_icu_stay` instead of `None`, which still requested the unavailable criterion.

The fix is database-aware and fail-closed:

- MIMIC recommended extraction now states `adult ICU stays (first-stay status unavailable)` and submits `preset=all_icu`, age 18-100, `exclude_readmissions=false`.
- `_resolve_export_cohort` passes `first_icu_stay=None` unless exclusion/`adult_first` explicitly requires `True`.
- The generic patient-filter boundary remains strict; no synthetic first-stay label or silent fallback was introduced.

## Real-data verification

An isolated server used `EASYICU_HOME=/private/tmp/easyicu-api-copilot-qa.U3mylo`, port 8925, the configured local 8317 OpenAI-compatible endpoint, and `gpt-5.6-luna`. Credentials were not logged or returned.

The official MIMIC-IV demo prepared root was recognized as:

- database: MIMIC-IV
- layout: Prepared (Parquet)
- source tables: 32
- mappable modules: 19

The repaired recommended extraction completed in about 6 seconds:

- cohort: 140 adult ICU stays
- modules/files: 6
- total rows: 32,298
- `demographics.parquet`: 140 rows
- `vitals.parquet`: 12,020 rows
- `chemistry.parquet`: 1,946 rows
- `sofa2_score.parquet`: 18,038 rows
- `sepsis3_sofa2.parquet`: 14 rows
- `outcome.parquet`: 140 rows
- reproducibility artifacts: `_manifest.json`, `feature_definitions.json`, `feature_definitions.csv`, `README.md`

Manifest evidence recorded `preset=all_icu`, `exclude_readmissions=false`, and selected cohort size 140. This is official demo-data engineering evidence, not full-database or clinical validation.

## Verification

- Python/JS syntax checks passed for all touched owners.
- Pi gateway `npm run check` passed.
- Copilot/extraction focused tests: `7 passed, 120 deselected`.
- Static/extraction/export focused tests: `19 passed, 204 deselected`.
- Browser: 1440x1000, no horizontal overflow, zero console errors/warnings after the final render.
- Runtime assertions: native owner present, embedded adapter mounted, completion view visible.
- CSS ownership scan: balanced braces/comments; no Patient, Cohort, Cross-DB, Settings, or Ideas selectors in `guided-pi-preview.css`.
- `git diff --check` passed.

Browser evidence: `task_logs/browser_qa_20260823/copilot_native_extraction_complete.png`.

## Boundaries

- No merge or push was performed.
- The Apple Silicon App/DMG was not rebuilt at this checkpoint; Web source and the next exact-head desktop build will share this implementation.
- The full repository CI was intentionally not run during this Web iteration. Per project policy, exact-head full CI is reserved for the later freeze/merge/release checkpoint.
