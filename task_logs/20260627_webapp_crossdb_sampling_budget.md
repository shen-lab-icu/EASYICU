# 2026-06-27 WebApp Cross-DB raw sampling budget

## Scope

The Cross-DB real raw-database density job can touch six local ICU database folders. This patch makes the bounded sampling budget visible before the run and sends the selected budget to the backend job.

## Changes

- `src/easyicu/webserver/static/js/screens-viz.js`
  - Added `Quick preview`, `Standard sample`, and `Deeper sample` profiles.
  - Default is `Quick preview`: max 200 entities per database and max 600 values per feature.
  - Standard is 300 / 1500; deeper is 800 / 3000.
  - The selected budget is shown in the setup card and run gate before starting the job.
  - The job request now uses the selected `max_patients` and `sample_size` instead of fixed hidden values.
- `src/easyicu/webserver/crossdb_review.py`
  - Background progress events now include the bounded `max_patients` and `sample_size` used by the raw job.
- `src/easyicu/webserver/static/index.html`
  - Bumped `screens-viz.js` cache key to `20260627-crossdb-sampling-budget`.
- Tests now lock the frontend budget controls and backend progress payload.

## Verification

- `node --check src/easyicu/webserver/static/js/screens-viz.js`
- `python -m compileall -q src/easyicu/webserver`
- `pytest -q tests/test_webserver_static_routes.py -k 'crossdb_restores_distribution_visuals or crossdb_availability_matrix'`
- `pytest -q tests/test_webserver_workspace_summary.py -k 'crossdb_raw_distribution_job_streams_progress_and_result or crossdb_raw_root_scan'`
- `git diff --check`
- `python3 EASYICU/tools/lint_main_plan.py`

Browser verification on `http://127.0.0.1:8785/?_v=crossdb-sampling-budget-20260627#crossdb`:

- The Cross-DB setup shows the sampling budget card.
- Default selected profile is Quick preview.
- Standard sample toggles correctly and updates the displayed budget.
- The page was restored to Quick preview after verification.
- No real six-database density job was started during this QA pass.

## Notes

The backend already bounded the raw job internally. The user-facing risk was that the UI hid those limits, making a six-database run look like an unbounded full-table operation. The new default is intentionally conservative for the first plot; users can opt into a deeper sample when they want smoother curves.
