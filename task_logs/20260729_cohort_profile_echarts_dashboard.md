# Cohort profile ECharts dashboard — 2026-07-29

## Scope

- Active module: Web
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- User issue: Cohort Profile repeated static cards/progress bars, mixed clinical phenotype with data coverage, and did not use the available visualization surface meaningfully.

## Source-backed design

The active `POST /api/cohort-review/summary` payload was inspected before implementation. The official MIMIC demo returned:

- 140 aggregate entities
- median age 63 years (21–91)
- median worst SOFA-2 6, observed range 0–22
- median ICU stay 2.16 days
- Sepsis-3 positive 47.1% and hospital mortality 10.7%
- ventilation and vasopressor module record presence of 50.0% and 37.1%
- no admission-source bins and no diagnosis/comorbidity module

The UI therefore does not invent an admission-source chart, a disease spectrum, or patient-level physiology. Treatment-module values are labeled as record presence rather than treatment-effect estimates.

## Implementation

- Added dedicated route owners:
  - `static/js/screens-viz-cohort-profile.js`
  - `static/css/cohort-profile.css`
- Extended the Cohort ECharts owner with three profile-specific visual contracts:
  - recorded-sex donut plus age-band rose chart
  - fixed-domain SOFA-2 clinical bullet chart, always 0–24
  - horizontal event-rate / record-presence comparison with distinct colors
- Reduced the top summary to three compact exact-value metrics.
- Removed the old `.cprof-*` progress-card CSS from broad `cohort.css`.
- Reframed coverage as a separate audit question with a direct `data-cohgo="coverage"` handoff.
- Added aggregate-only provenance disclosure and an explicit “diagnosis unavailable, do not infer disease spectrum” boundary.

## Verification

- JavaScript syntax:
  - `node --check screens-viz-cohort-profile.js`
  - `node --check screens-viz-cohort-charts.js`
- Owner contract:
  - `tests/js/cohort_profile_owner.test.js`
  - verifies rose composition, fixed SOFA max 24, two event rows, two record-presence rows, no legacy progress bars
- Static/ownership suite:
  - `76 passed, 1 deselected`
  - the deselected test is the known isolated-worktree callback hint assertion (`EASYICU` versus worktree directory name)
- Backend/workspace regression:
  - `tests/test_webserver_workspace_summary.py`: `134 passed`
- CSS ownership:
  - owner brace/comment scan balanced
  - `.cohort-profile-dashboard` present only in `cohort-profile.css`
  - Patient/Cross-DB/Guided/Ideas markers absent
- Browser QA on official MIMIC demo:
  - all three ECharts mounted with real SVG output: 14, 6, and 16 paths
  - SOFA chart visibly reads `6 / 24`; observed `0–22` remains separate
  - Chinese and English views both render without mixed headings
  - Coverage Audit handoff opens the coverage panel and returns cleanly
  - desktop viewport has no horizontal overflow (`1280 == document.scrollWidth`)

## Non-goals

- No backend scientific contract changed.
- No diagnosis, admission-source, or patient-level measurements were synthesized.
- No merge or push was performed.
