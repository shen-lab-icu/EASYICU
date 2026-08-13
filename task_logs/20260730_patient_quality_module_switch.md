# Patient Data Quality module switch

- Task ID: `PATIENT-QUALITY-MODULE-SWITCH`
- Owner: Web / Patient Review / Data Quality
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Date: 2026-07-30

## User problem

The quality view showed a fixed 19-module overview but offered no visible way to select one module and inspect its features. The three summary chips described the scope without acting as navigation.

## Implemented contract

- Keep `全部模块（19）` as the default overview.
- Add one visible module selector above the chart.
- Selecting a module replaces the overview with every measurement feature in that module that has a valid entity-level missingness denominator.
- Long modules retain all feature rows and use bounded ECharts scrolling instead of truncation.
- The summary chips update to the selected module scope.
- The complete feature-quality catalog opens and synchronizes its module filter to the same selection.
- Returning to `全部模块` restores the original module-level chart and resets the lower filter.
- Event and exposure prevalence remain excluded from missingness.

## Verification

- `node tests/js/patient_quality_owner.test.js src/easyicu/webserver/static/js/screens-viz-patient-quality.js`
  - passed; includes `module_feature_switch=true`
- `node tests/js/patient_echarts_owner.test.js src/easyicu/webserver/static/js/screens-viz-patient-charts.js`
  - passed; includes `module_feature_missingness=true`
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_patient_browse_frontend.py -k 'not extraction_feature_definition_manifest_records_callback_provenance'`
  - `80 passed, 1 deselected`
  - the deselected extraction test derives a project hint from the worktree directory name and is unrelated to Patient Review.
- `git diff --check`
  - passed.
- Browser QA at `http://localhost:8876/?qa=qualitymoduleswitch1#patient`
  - official MIMIC-IV Demo loaded: 140 entities, 19 modules, 281 catalog features.
  - selected `chemistry`: 49/49 measurement features rendered; chart kind `quality-feature-missingness`; one SVG; no visible fallback.
  - lower catalog opened and module filter synchronized to `chemistry`.
  - switched back to `all`: chart kind `quality-modules`, scope `all`, lower filter `all`.
  - document, main content, quality section and dynamic panel horizontal overflow: all `0px`.

## Evidence

- `output/ui-qa/20260730_patient_quality_module_switch/quality-module-switch.png`

