# Patient Review selection clarity

Date: 2026-07-29  
Task: `PATIENT-REVIEW-SELECTION-CLARITY`  
Branch: `codex/web-copilot-cockpit-lite-20260729`  
Base commit: `f97978f`

## User-visible problem

The three time-series modes were technically different but not self-explanatory:

- “Module overview” sounded like another patient summary and still resembled the gallery.
- The trajectory-gallery module selector was hidden inside a collapsed individual-chart browser.
- Cross-patient comparison automatically chose one backend feature and exposed no selection control.

## Implemented product contract

### Feature catalog

- Renamed the first mode to `Feature catalog / 特征目录`.
- States explicitly that it is a data catalog, not a patient summary.
- Column language now distinguishes export observations, current-patient loaded trajectories, and additional loadable features.
- Synthetic demo now publishes feature-level coverage metadata for all 281 catalog definitions; 100 clinically modelled features are marked observed and unmodelled concepts remain `not_materialized`.
- Catalog observation counts use the greater of declared export coverage and directly observed/loaded feature state, preventing a loaded trajectory from appearing under a false zero-observation module.

### Trajectory gallery

- Added one visible `Choose what to view / 选择查看范围` selector above the main chart.
- Default remains a six-signal cross-module clinical focus.
- Choosing a module updates both the aligned ECharts view and the disclosed individual-feature browser.
- Counts expose the full loaded set (83 synthetic numeric trajectories), the selected scope, and the intentionally bounded 24-card individual browser.
- Selection state is owned by `screens-viz-patient-gallery.js`; presentation remains in the existing Patient series/gallery owners.

### Cross-patient comparison

- Added a grouped selector containing every loaded numeric trajectory feature.
- Synthetic fallback rebuilds the deterministic comparison payload for the chosen feature; browser QA switched from heart rate to lactate across five synthetic entities.
- Real exports reuse the existing `/api/patient-review/feature` projection endpoint for up to the first five pseudonymous entities on the current navigation page.
- The new comparison owner caches by source/page/feature, rejects stale responses after source or page changes, and fails visibly when fewer than two drawable trajectories are available.
- No new backend endpoint, row payload, direct identifier, or duplicate comparison policy was added.

## Ownership

- Gallery state/action owner: `src/easyicu/webserver/static/js/screens-viz-patient-gallery.js`
- Comparison state/API owner: `src/easyicu/webserver/static/js/screens-viz-patient-comparison.js`
- Comparison layout owner: `src/easyicu/webserver/static/css/patient-comparison.css`
- Existing rendering owner extended: `src/easyicu/webserver/static/js/screens-viz-patient-series.js`
- Existing gallery layout owner extended: `src/easyicu/webserver/static/css/patient-gallery.css`
- Existing API transport owner reused unchanged: `src/easyicu/webserver/static/js/api.js`

New owner sizes remain bounded: comparison JS 246 lines, gallery JS 49 lines, comparison CSS 58 lines, gallery CSS 190 lines. The new owners contain no Cohort, Cross-DB, or Guided route markers.

## Verification

- JS syntax: all changed Patient JS files pass `node --check`.
- Executable owner contracts:
  - comparison requests one feature for at most five entities, caches within a page, and rejects stale responses;
  - gallery selection updates and resets independently;
  - demo can rebuild a five-trace `map` comparison;
  - feature catalog keeps all 281 definitions and distinguishes numeric trajectories from categorical observations.
- Focused affected gate: `16 passed, 1 warning`.
- Extended Web static/Patient gate: `85 passed, 1 known worktree-name failure, 1 warning`; the only failure is the pre-existing assertion expecting callback project hint `EASYICU` while an isolated worktree is named `easyicu-copilot-cockpit-lite`.
- CSS owner presence/absence, foreign-route scan, brace/comment balance, and `git diff --check` pass.
- Browser QA on `http://localhost:8876/?qa=density5#patient`:
  - feature catalog, gallery, and comparison controls are visible in Chinese;
  - gallery selection changed from cross-module focus to Chemistry and updated `6 / 9` focus plus `9 / 9` browser scope;
  - comparison selection changed from heart rate to lactate and rendered five bounded traces;
  - at 1119×994, document, workbench, mode bar, gallery scope/focus card, comparison controls, and feature selector all have `scrollWidth == clientWidth`;
  - console error count is zero.

## Deferred

- The individual feature browser remains intentionally capped at 24 rendered cards per selected scope to avoid mounting dozens of hidden ECharts instances. The UI now states the shown/available count instead of implying that only 24 features exist.
- No entity-set editor was added; comparison deliberately uses the first five pseudonymous entities on the current bounded page. Entity selection can be considered separately if researchers need arbitrary matched sets.
