# 2026-06-27 Cohort SOFA Matrix Granularity

## Scope

User noted that the SOFA-1 to SOFA-2 transition matrix was too coarse for a 0-24 score. This patch keeps the existing clinical-band view but adds user-selectable matrix granularity:

- coarse: 4 clinical bands
- medium: 6 bands
- fine: 12 bands
- exact: 25 exact scores, 0-24

## Implementation

- Backend `cohort_review.py` now returns an aggregate-only `exact_score_matrix` and `exact_score_bins` for paired SOFA-1/SOFA-2 scores.
- Frontend `screens-viz.js` bins the exact matrix on demand and offers 4/6/12/25 score granularity controls.
- Cohort owner CSS `cohort.css` keeps the 25-column exact matrix inside a horizontal matrix scroller, without page-level horizontal overflow.
- No patient rows or identifiers are returned; the matrix is score-pair counts only.

## Verification

- `./.venv/bin/python -m pytest -q tests/test_webserver_workspace_summary.py -k sofa_reclassification`: 1 passed.
- `./.venv/bin/python -m pytest -q tests/test_webserver_static_routes.py -k cohort`: 6 passed.
- Bundled Node `--check` across native webserver JS: passed.
- Browser QA on `http://127.0.0.1:8765/?_v=sofa-granularity-20260627#cohort`:
  - medium columns: 6
  - fine columns: 12
  - exact columns: 25
  - page `overflowX=0`
  - matrix scroller active for exact view
  - console errors: 0
- `git diff --check`: passed.

## Notes

The control is exploratory and descriptive. It does not add inferential statistics or patient-level rows.
