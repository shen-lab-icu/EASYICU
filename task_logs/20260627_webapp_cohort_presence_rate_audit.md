# 2026-06-27 Cohort Presence-rate Audit Fix

## Issue

The Cohort Statistics coverage audit treated event/exposure modules as missingness coverage. In the real export this made rows such as `sepsis3_sofa1` show `40.3% bad`, even though that percentage is the Sepsis-3 event rate, not a missingness rate.

## Fix

- Expanded shared presence-rate semantics in `src/easyicu/webserver/dataio.py`:
  - event-rate modules: `sepsis3_sofa1`, `sepsis3_sofa2`
  - exposure-rate modules: `vasopressors`, `vasopressor`, `ventilator`, `ventilation`
- Updated Cohort, Extraction, and Patient quality-status helpers to mark these modules as neutral rate rows instead of low coverage rows.
- Added `metric_kind` to module quality payloads so the UI can distinguish `coverage`, `event_rate`, and `exposure_rate`.
- Updated Cohort Coverage Audit UI in `src/easyicu/webserver/static/js/screens-viz.js`:
  - table column now reads `Coverage / rate`
  - event rows display `Event rate / 发生率`
  - exposure rows display `Exposure rate / 暴露率`
  - internal labels such as `bad` are mapped to user-facing interpretations
  - explanatory note says event/exposure rows are incidence or exposure prevalence, not missingness coverage
- Bumped `screens-viz.js` and `cohort.css` cache-bust version in `static/index.html`.

## Evidence

Focused tests:

```text
.venv/bin/python -m pytest -q tests/test_webserver_workspace_summary.py -k 'cohort_review_presence_rate_modules_are_not_low_coverage or cohort_review_summary_uses_active_source_without_row_payload'
2 passed, 106 deselected

.venv/bin/python -m pytest -q tests/test_webserver_static_routes.py -k 'cohort'
6 passed, 37 deselected

/Users/haibo/.nvm/versions/node/v24.11.0/bin/node --check src/easyicu/webserver/static/js/screens-viz.js
.venv/bin/python -m ruff check src/easyicu/webserver/dataio.py src/easyicu/webserver/cohort_review.py src/easyicu/webserver/extraction_filters.py src/easyicu/webserver/patient_drilldown.py tests/test_webserver_workspace_summary.py tests/test_webserver_static_routes.py
.venv/bin/python -m compileall -q src/easyicu/webserver
All checks passed.
```

Live real-export API smoke on `127.0.0.1:8794`:

```text
sepsis3_sofa1 {"metric_kind": "event_rate", "covered_entities": 38055, "coverage_pct": 40.3, "quality_status": "neutral"}
sepsis3_sofa2 {"metric_kind": "event_rate", "covered_entities": 39478, "coverage_pct": 41.8, "quality_status": "neutral"}
vasopressors {"metric_kind": "exposure_rate", "covered_entities": 28882, "coverage_pct": 30.6, "quality_status": "neutral"}
ventilator {"metric_kind": "exposure_rate", "covered_entities": 40694, "coverage_pct": 43.1, "quality_status": "neutral"}
vitals {"metric_kind": "coverage", "covered_entities": 94439, "coverage_pct": 100.0, "quality_status": "ok"}
quality {"modules_ok": 14, "modules_warn": 1, "modules_bad": 0, "modules_neutral": 4, "modules_unknown": 0, "watchlist_count": 1, "median_coverage_pct": 100.0}
```

## Next

Open a fresh browser URL with the new bundle version, e.g. `http://127.0.0.1:8794/?_v=presence-rate-audit-20260627#cohort`, and confirm the Cohort coverage audit now labels Sepsis rows as event rates and treatment rows as exposure rates.
