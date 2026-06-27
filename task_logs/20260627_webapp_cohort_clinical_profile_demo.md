# 2026-06-27 Cohort Clinical Profile Demo Shape

## Scope

The Cohort Review `Cohort profile` tab no longer uses a thin demo snapshot made only from age bands and SOFA severity bars. The demo path now shows the clinical domains a real cohort profile should expose, while still labeling the numbers as seeded UI examples.

## Changes

- `src/easyicu/webserver/static/js/screens-viz.js`
  - Added `demoCohortClinicalProfile()` with aggregate-only domains:
    - demographics
    - severity and outcomes
    - treatments and organ support
    - diagnoses and comorbidities
    - vitals and laboratory profile
    - data completeness
  - Replaced the old demo age/SOFA-only profile body with the shared `cohortClinicalProfile()` renderer.
  - Added a compact phenotype-balance overview using seeded aggregate values only.
- `src/easyicu/webserver/static/css/cohort.css`
  - Added route-owned `.cprof-spark-*` styles.
- `src/easyicu/webserver/static/index.html`
  - Bumped the Cohort CSS and visualization JS cache keys for this change.
- `tests/test_webserver_cohort_profile_ui.py`
  - Added static ownership/content regression for the clinical profile demo shape and cache keys.

## Verification

- `find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 /Users/haibo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node --check`
- `./.venv/bin/python -m pytest -q tests/test_webserver_cohort_profile_ui.py tests/test_webserver_static_routes.py`
  - `44 passed, 1 warning`
- `git diff --check`
- Browser smoke with local Chrome channel on `http://127.0.0.1:8765/?_v=cohort-clinical-profile-20260627#cohort`:
  - clicked `运行演示队列审阅`
  - opened `队列画像`
  - confirmed treatment/organ support, diagnoses/comorbidities, vitals/labs, and data completeness sections render
  - confirmed the old age/SOFA-only profile is not the body
  - `overflowX=0`

## Notes

This patch improves demo clinical shape only. It does not claim these seeded values are research findings. Real-mode cohort profile still depends on the active local export summary.
