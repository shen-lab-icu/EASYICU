# 2026-06-27 Cohort Clinical Profile Replacement

## Trigger

Manual browser review found that the Cohort Review "Age x ICU LOS complexity" heatmap was large but clinically weak. The chart used age bands and ICU length of stay as a proxy for "cohort profile", while omitting clinically interpretable dimensions such as treatment exposure, organ support, diagnosis/comorbidity availability, and coverage risk.

## Changes

- Removed the age x ICU LOS complexity payload from the Cohort Review summary contract.
- Added `summary.clinical_profile` in `src/easyicu/webserver/cohort_review.py`.
- Rendered the real Cohort profile as clinical domains in `src/easyicu/webserver/static/js/screens-viz.js`:
  - demographics
  - severity and outcomes
  - treatments and organ support
  - diagnoses and comorbidities
  - data completeness
- Added owner-file CSS in `src/easyicu/webserver/static/css/cohort.css`.
- Bumped `cohort.css` and `screens-viz.js` static asset versions in `src/easyicu/webserver/static/index.html`.
- Updated tests so the old proxy heatmap is explicitly disallowed.

## Data Boundary

The new treatment and diagnosis cards use module-level entity coverage and row counts from the registered export manifest/coverage scan. They do not return patient rows. Missing diagnosis/comorbidity modules are shown as unavailable instead of inferred from unrelated fields.

## Verification

- `python3 -m py_compile src/easyicu/webserver/cohort_review.py`
- `node --check src/easyicu/webserver/static/js/screens-viz.js`
- `pytest -q tests/test_webserver_workspace_summary.py -k 'cohort_review or survival'` -> 10 passed
- `pytest -q tests/test_webserver_static_routes.py -k 'cohort_snapshot or native_cohort'` -> 4 passed
- `python -m ruff check src/easyicu/webserver/cohort_review.py tests/test_webserver_workspace_summary.py`
- `python -m compileall -q src/easyicu/webserver`
- `git diff --check`
- Live source server at `127.0.0.1:8793`:
  - HTML includes `cohort.css?v=20260627-clinical-profile`
  - HTML includes `screens-viz.js?v=20260627-clinical-profile`
  - `/api/cohort-review/summary` returns `clinical_profile.payload_scope=cohort_aggregate_only_no_patient_rows`
  - `/api/cohort-review/summary` no longer returns `summary.complexity`
  - Current registered export reports treatment profile ready for vasopressors, ventilation, respiratory, renal, and medications; diagnosis/comorbidity remains unavailable because the active export has no diagnosis module.

## Next

Run a browser visual pass when Playwright/npm is available, then continue Patient/Cross-DB visual parity checks. The Cohort profile should not reintroduce proxy-only visualizations unless they have clear clinical interpretation and source provenance.
