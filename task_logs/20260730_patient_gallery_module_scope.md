# PATIENT-GALLERY-MODULE-SCOPE

- Date: 2026-07-30
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Module: `web`
- Phase: Patient Review interaction QA
- Task: remove the unvalidated cross-module “clinical focus” preset

## Outcome

The trajectory gallery now exposes only the 19 real catalog modules. The
previous `clinical-focus` option was a frontend-only mixture of SOFA, vital
sign, and respiratory signals; it did not represent a validated clinical
priority definition and has been removed.

The gallery selection owner now:

- defaults to the first available real catalog module;
- keeps the selected module when the patient changes;
- falls back to the first available module if a stale or removed module key is
  encountered;
- preserves catalog-to-gallery feature targeting and bounded auto-loading.

No backend data contract, feature count, API, CSS, or chart renderer changed.

## Verification

- Focused automated tests: `12 passed`.
- Extended Patient/static suite: `85 passed`; four unrelated pre-existing
  failures remain:
  - two stale expected-key assertions for existing Patient quality/ECharts
    owner contracts;
  - one stale `screens-viz.js` cache-version assertion;
  - one isolated-worktree callback hint expecting the directory name
    `EASYICU`.
- JavaScript syntax: both changed Patient owner files passed `node --check`.
- Route ownership:
  - Patient gallery behavior remains in
    `screens-viz-patient-gallery.js`;
  - Patient trajectory markup remains in
    `screens-viz-patient-series.js`;
  - no Cohort, Cross-DB, or Guided route markers entered either owner;
  - no CSS changed.
- `git diff --check`: passed.

## Browser QA

Source: clinically constrained synthetic fallback, matching the reported
screen. Desktop/laptop viewport.

- dropdown contains exactly 19 options;
- `临床重点（跨模块）` is absent from visible page text;
- initial scope is a real module (`SOFA-2 评分`);
- after selecting `生命体征` and switching from Synthetic entity 1 to entity
  2, `生命体征` remains selected and the entity-specific chart rerenders;
- document, main, and Patient gallery horizontal overflow are all `0`;
- the aligned chart remains inside its owning panel.

Screenshot:

- `output/ui-qa/20260730_patient_module_scope/patient-gallery-module-only-entity2.jpg`

## Files changed

- `src/easyicu/webserver/static/index.html`
- `src/easyicu/webserver/static/js/screens-viz-patient-gallery.js`
- `src/easyicu/webserver/static/js/screens-viz-patient-series.js`
- `tests/js/patient_gallery_owner.test.js`
- `tests/js/patient_series_owner.test.js`
- `tests/test_webserver_patient_browse_frontend.py`
- `tests/test_webserver_static_routes.py`
