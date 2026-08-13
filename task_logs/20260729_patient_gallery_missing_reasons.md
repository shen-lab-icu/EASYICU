# Patient gallery omitted-trajectory accounting

- Date: 2026-07-29
- Task: `PATIENT-GALLERY-MISSING-REASONS`
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Scope: Patient Review trajectory gallery only

## User-facing problem

The SOFA-2 module has seven catalog features, while the selected eICU Demo patient has only two drawable multi-point trajectories. The gallery previously displayed only those two charts and the phrase “all available trajectories are shown”; the other five features disappeared from the gallery, so users could not tell whether the page was incomplete, still loading, or genuinely lacked patient observations.

## Implemented contract

- The gallery now accounts for every feature in the selected module: catalog total, plotted count, and not-plotted count.
- A compact disclosure lists each omitted feature and its current-patient reason:
  - no observation for this patient;
  - one observed/static value, insufficient for a trajectory;
  - categorical observation;
  - automatic check pending;
  - check failed;
  - otherwise unavailable.
- Modules with six or fewer omitted features open the explanation by default. Larger modules keep it collapsed and bounded by an internal scroll region.
- Empty charts are not fabricated for missing features.
- Gallery rendering remains in `screens-viz-patient-series.js`; layout remains in the dedicated `patient-gallery.css` owner.

## Browser evidence

Official eICU Demo, Entity 1, SOFA-2:

- 7 catalog features;
- 2 drawable trajectories;
- 5 not plotted;
- the five omitted features are `sofa2_resp`, `sofa2_coag`, `sofa2_liver`, `sofa2_cns`, and `sofa2_renal`;
- each is explicitly reported as “当前患者无观测”.

At 1521×1354:

- document `scrollWidth == clientWidth` (1506 px);
- main `scrollWidth == clientWidth` (1258 px);
- explanation panel `scrollWidth == clientWidth` (1186 px);
- all five rows fit without internal vertical scrolling (`175 == 175`).

Screenshots:

- `output/ui-qa/20260729_patient_gallery_missing_reasons/01-before-two-of-seven.jpg`
- `output/ui-qa/20260729_patient_gallery_missing_reasons/02-after-reasons-visible.jpg`

## Verification

- Node syntax: passed.
- Executable Patient series owner contract: passed, including omission reasons for missing and single-value features.
- Focused Patient tests: `7 passed`.
- CSS owner presence/absence, brace/comment scan, and `git diff --check`: passed.
- Static-route suite: `71 passed, 1 known failure`; the only failure is the pre-existing worktree-name callback hint assertion (`easyicu-copilot-cockpit-lite` vs `EASYICU`), unrelated to this patch.
- Chinese and English explanation copy were both exercised in the browser.

## Honest boundary

The explanation reports the selected patient’s bounded feature-detail result. It does not claim that a missing patient trajectory means the feature is absent from the export or source database.
