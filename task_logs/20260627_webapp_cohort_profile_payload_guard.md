# 2026-06-27 Cohort Profile Payload Guard

## Context

Manual QA found that the Cohort Statistics `Cohort profile / 队列画像` panel could render a
local workspace summary with all clinical aggregate values shown as `—`.

The bug was a frontend state mismatch: `cohortView === "loaded"` accepted any
`EU_VIZ_WORKSPACE` object in Real mode. A Patient/Extraction workspace summary can contain
file/module metadata without the backend `EU_COHORT_REVIEW.summary`, so the snapshot panel
fell back to an incomplete summary.

## Fix

- `src/easyicu/webserver/static/js/screens-viz.js`
  - Added `cohortLoaded()` so Real mode requires `EU_COHORT_REVIEW.summary`.
  - Added `reloadStaleRealCohortIfNeeded()` to recover old loaded states by reloading
    `/api/cohort-review/summary`.
  - Updated the top action and render loaded checks to use the stricter payload guard.
- `src/easyicu/webserver/static/index.html`
  - Bumped `screens-viz.js` cache key so browsers fetch the fixed script.
- `tests/test_webserver_static_routes.py`
  - Added static assertions for the Real-mode payload guard and stale reload path.

## Evidence

- `pytest -q tests/test_webserver_static_routes.py -k 'cohort_real_page_is_backend_backed_and_bilingual or cohort_comparison_radios_are_stateful_controls or cohort_snapshot_renders_real_clinical_profile'`
  - `3 passed`
- `node --check src/easyicu/webserver/static/js/screens-viz.js`
  - passed
- Browser QA:
  - `output/playwright/cohort_payload_guard_20260627/cohort_payload_guard.json`
  - `EU_COHORT_REVIEW.summary.cohort_size = 10`
  - Snapshot stat values: `10`, `57.2`, `50%`, `20%`, `7`, `10%`
  - `hasDashOnlyStats=false`
  - `viewportOverflow=0`
  - `consoleErrors=[]`

## Boundary

No backend calculations changed. Existing concurrent worktree changes outside Cohort UI were
not staged or modified by this task.
