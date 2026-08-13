# Patient multi-track spacing and bottom-axis separation

- Date: 2026-07-30
- Task: `PATIENT-TRACK-SPACING-AXIS-SEPARATION`
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Scope: shared Patient Review aligned clinical timeline

## User-facing problem

The previous 20 px inter-track gutter still made dense six-track views feel stacked. More importantly, the chart reserved only 48 px below the final track, placing the shared ICU-time title directly on the zoom slider. Increasing only the gap would have compressed the six panels further inside the fixed 560 px shell.

## Implemented contract

- Increased the shared inter-track gutter from 20 px to 28 px.
- Added an 82 px bottom-axis reserve for tick labels, the centered ICU-time title, and the zoom slider.
- Increased the axis-title gap to 34 px.
- Positioned the 14 px zoom slider 8 px above the chart bottom.
- Added owner-level adaptive timeline height: six-track timelines requested at 560 px expand to 640 px; fewer-track views keep a bounded 320–760 px height.
- The slot DOM height and ECharts option use the same shared height calculation, so the canvas and geometry cannot diverge.
- The change lives entirely in `screens-viz-patient-story-charts.js`; no route CSS or broad override file was added.

## Browser evidence

Official MIMIC-IV Demo, Entity 1, at 1521×1354:

- Clinical-focus and SOFA-2 views both render six 67 px tracks.
- All five measured inter-track gaps are exactly 28 px.
- The ICU-time title ends 22.5 px above the zoom slider.
- Chart height is 640 px.
- Document and main-content horizontal overflow are both 0.
- SOFA-2 retains the fixed 0–24 total-score domain and 0–4 component domains.

Screenshots:

- `output/ui-qa/20260730_patient_track_spacing2/01-after-six-track-separation.jpg`
- `output/ui-qa/20260730_patient_track_spacing2/02-after-sofa-six-track-separation.jpg`

## Verification

- `node --check src/easyicu/webserver/static/js/screens-viz-patient-story-charts.js`: passed.
- Executable story-chart owner contract: passed, including 28 px gutters, 640 px six-track shell, 82 px bottom reserve, and zoom placement.
- Focused Patient/static route checks: `9 passed`.
- Patient + static route suite: `78 passed, 1 known unrelated failure`.
- The only failure is the pre-existing worktree-name callback hint assertion (`easyicu-copilot-cockpit-lite` vs `EASYICU`).
- `git diff --check`: passed.
