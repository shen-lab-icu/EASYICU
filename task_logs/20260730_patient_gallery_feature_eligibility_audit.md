# PATIENT-GALLERY-FEATURE-ELIGIBILITY-AUDIT

- Date: 2026-07-30
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Module: `web`
- Phase: Patient Review interaction QA
- Source checked: official MIMIC-IV Clinical Database Demo v2.2
- Patient scope: pseudonymous `Entity 1`

## Outcome

Patient Review now applies one explicit rule to every feature opened in a
trajectory module: a numeric time series requires at least two **distinct,
valid, timed numeric observations** for the current patient.

The feature-detail boundary returns a bounded `eligibility` receipt with:

- total non-null observation count;
- numeric observation count;
- timed numeric observation count;
- distinct valid time count;
- the stable criterion and drawable decision.

The gallery no longer repeats global and module-level “loaded trajectory”
counts. Its selected-module summary is reduced to:

`17 / 17 checked · 10 drawable · main chart 6`

The expandable accounting section states the rule once and gives one reason
for each feature that did not become a chart.

## Official-demo audit: vasopressors, Entity 1

| Feature | Observations | Distinct valid times | Drawable | Result |
|---|---:|---:|---|---|
| `norepi_rate` | 33 | 33 | yes | trajectory |
| `norepi_dur` | 3 | 3 | yes | trajectory |
| `norepi_equiv` | 36 | 36 | yes | trajectory |
| `norepi60` | 30 | 30 | yes | trajectory |
| `epi_rate` | 0 | 0 | no | current patient has no observation |
| `epi_dur` | 0 | 0 | no | current patient has no observation |
| `epi60` | 0 | 0 | no | current patient has no observation |
| `dopa_rate` | 4 | 4 | yes | trajectory |
| `dopa_dur` | 1 | 1 | no | fewer than two distinct timed numeric observations |
| `dopa60` | 3 | 3 | yes | trajectory |
| `dobu_rate` | 71 | 71 | yes | trajectory |
| `dobu_dur` | 4 | 4 | yes | trajectory |
| `dobu60` | 69 | 69 | yes | trajectory |
| `adh_rate` | 0 | 0 | no | current patient has no observation |
| `phn_rate` | 0 | 0 | no | current patient has no observation |
| `vaso_ind` | 0 | 0 | no | current patient has no observation |
| `other_vaso` | 27 | 27 | yes | trajectory |

Total: 17 checked, 10 drawable, 7 not drawable.

## Verification

- Python/API:
  - `106 passed, 1 deselected`
  - includes a negative contract proving two observations at the same time do
    not qualify as a trajectory.
- JavaScript owner contracts:
  - Patient series owner passed.
  - Patient feature-loader owner passed.
  - Patient gallery owner passed.
- Syntax:
  - `python -m py_compile` passed.
  - `node --check` passed.
  - `git diff --check` passed.
- CSS ownership:
  - no CSS was changed;
  - the existing Patient route-purity and balanced-syntax regression passed.
- Browser QA:
  - selected module reported `17 / 17` checked, `10` drawable, `6` in the
    aligned main chart;
  - omission list contained exactly 7 entries;
  - legacy `条已加载轨迹` copy was absent;
  - document, main content, and Patient series panel horizontal overflow were
    all `0`.

Screenshot:

- `output/ui-qa/20260730_patient_gallery_feature_eligibility_audit/vasopressor-feature-audit.png`

## Files changed

- `src/easyicu/webserver/patient_drilldown/feature_detail.py`
- `src/easyicu/webserver/static/js/screens-viz-patient-features.js`
- `src/easyicu/webserver/static/js/screens-viz-patient-series.js`
- `tests/test_webserver_patient_feature_coverage.py`
- `tests/js/patient_series_owner.test.js`

