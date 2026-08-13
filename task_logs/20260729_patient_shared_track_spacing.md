# Patient shared multi-track spacing

- Date: 2026-07-29
- Task: `PATIENT-SHARED-TRACK-SPACING`
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Scope: shared Patient Review aligned clinical timeline

## User-facing problem

The vertically aligned clinical tracks were separated by only 12 px. In dense six-track views such as SOFA-2, adjacent panels visually ran together and read like one continuous grid. The same geometry was shared by other modules, so a SOFA-only style patch would have left the underlying problem in place.

## Implemented contract

- `screens-viz-patient-story-charts.js` now owns a single 20 px gutter for every multi-track clinical timeline.
- Single-track charts retain their original top geometry and do not receive a phantom gutter.
- The overall chart height remains unchanged; panel height is recomputed inside the existing bounded ECharts layout.
- Six-track views receive five identical 20 px gutters and still fit above the shared zoom strip.
- The change applies to SOFA, vital-sign, other module-gallery timelines, and the Patient Overview clinical story because they all consume `clinicalTimelineOption()`.
- No route CSS or broad override file was changed.

## Browser evidence

Official eICU Demo, Entity 2, at 1521×1354:

- SOFA-2: six aligned tracks with a visible gutter between every adjacent pair; total-score and 0–4 component axes remain fully visible.
- Vital signs: five aligned tracks inherit the same gutter without a module-specific rule.
- The last track, shared ICU time axis, and zoom strip remain inside the chart shell; no horizontal clipping or label collision was visible.

Screenshots:

- `output/ui-qa/20260729_patient_track_spacing/01-before-tight-tracks.jpg`
- `output/ui-qa/20260729_patient_track_spacing/02-after-sofa-shared-gap.jpg`
- `output/ui-qa/20260729_patient_track_spacing/03-after-vitals-shared-gap.jpg`

## Verification

- Node syntax check: passed.
- Executable Patient story-chart owner contract: passed, including six-track 20 px gutter geometry, bottom-bound fit, and single-track geometry.
- Focused Patient and static-route tests: `77 passed, 1 known unrelated failure`.
- The only failure is the pre-existing worktree-name callback hint assertion (`easyicu-copilot-cockpit-lite` vs `EASYICU`).
- `git diff --check`: passed.

## Honest boundary

The additional spacing improves visual grouping and scanability; it does not independently constitute a complete accessibility audit or validate every browser zoom level.
