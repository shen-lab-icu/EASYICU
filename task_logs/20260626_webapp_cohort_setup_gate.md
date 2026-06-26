# 2026-06-26 Cohort Statistics Setup Gate

## Scope

Fix `#cohort` opening directly into a seeded demo result. The page previously initialized `cohortView` as `loaded`, so users could see `Sepsis vs Non-sepsis` and downstream review panels without explicitly configuring or running anything.

## Changed Files

- `src/easyicu/webserver/static/js/screens-viz.js`
  - Changed `cohortView` default from `loaded` to `idle`.
  - Reset Cohort state to `idle` when global data mode changes.
  - Added an explicit setup gate for Demo mode.
  - Real mode now stays idle on failed export load instead of rendering a loaded page after failure.
  - Added `Edit setup` handling for Cohort.
  - Added bilingual copy for the setup gate.
- `src/easyicu/webserver/static/index.html`
  - Bumped `screens-viz.js` cache key.
- `tests/test_webserver_static_routes.py`
  - Added static regression assertions for the Cohort setup gate and non-loaded default.

## Browser Evidence

QA ran on current code at `http://127.0.0.1:8783/?_v=cohort-setup-gate-4#cohort`.

- Report: `output/playwright/cohort_setup_gate_20260626/cohort_setup_gate.json`
- Screenshots:
  - `output/playwright/cohort_setup_gate_20260626/cohort_initial_gate.png`
  - `output/playwright/cohort_setup_gate_20260626/cohort_after_demo_run.png`

Browser assertions:

- Initial direct open has `[data-cohort-config-required="true"]`.
- Initial direct open does not show `Sepsis vs Non-sepsis`.
- Initial direct open shows the explicit demo review action.
- After clicking the setup-card run button, the setup gate disappears.
- After clicking, seeded demo Cohort review loads and shows the group/KM area.
- `overflowX=0`.
- Console/page errors: none.

## Verification

- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py` -> `124 passed, 1 warning`
- `python -m compileall -q src/easyicu/webserver` -> passed
- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 node --check` -> passed
- `git diff --check` -> passed
- Provider readiness smoke: `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`

## Notes

This does not remove Demo mode. It makes Demo mode explicit: seeded Cohort Statistics are available only after the user intentionally runs the demo review.
