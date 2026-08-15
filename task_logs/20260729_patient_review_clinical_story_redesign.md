# Patient Review clinical-story redesign

- Date: 2026-07-29
- Task ID: `PATIENT-REVIEW-CLINICAL-STORY`
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Base before this patch: `ef37f7a`
- Scope: native FastAPI Patient Review frontend only; no backend/API/data-contract changes

## Problem

The previous Patient Review exposed more real data, but the interaction still read
as a collection of independent widgets:

- Module Overview repeated the same per-feature charts shown in Trajectory Gallery.
- Patient Overview separated severity and organ information without one clinical
  time cursor.
- Data Quality mixed several denominators into similarly sized cards and made the
  readiness donut visually dominant.
- Trajectory Gallery rendered 15 independent charts with separate axes and no
  focused default, search, or module filter.

## Implemented design

1. Patient Overview now leads with one coordinated ECharts clinical timeline:
   SOFA-2 severity, MAP/SBP perfusion, and heart rate use vertically aligned
   grids, one axis pointer, and one zoom window. Compact facts summarize change,
   peak severity, and non-zero organ components. The six SOFA-2 components remain
   as a compact status strip rather than a second large chart.
2. Module Overview is now a 19-row catalog/readiness matrix. It shows observed
   definitions, loaded trajectories, loadable features, and one action column.
   Exact feature inventories remain collapsed and the view contains no trajectory
   chart. Module/feature search is local and source-safe.
3. Data Quality now labels four distinct scopes: catalog readiness, bounded QC
   concepts, bounded audit records, and actionable missingness. Readiness is a
   direct stacked bar with exact counts; missingness remains a separate ranked
   chart and four-item issue queue. Exact audit tables and browser limits use
   progressive disclosure.
4. Trajectory Gallery defaults to six clinically useful signals on one aligned
   timeline. The complete loaded set remains available in one collapsed individual
   chart browser with feature search and module filter.

## Ownership and contracts

- Added `static/js/screens-viz-patient-story-charts.js` as the explicit owner for
  coordinated timelines and direct catalog-readiness composition.
- Added `static/css/patient-story.css` for overview/quality clinical-story layout.
- Added `static/css/patient-gallery.css` for gallery-only layout, keeping
  `patient-series.css` under the route CSS review budget.
- The existing Patient chart owner now publishes `isStepFeature()` so gallery code
  does not import or duplicate the owner’s private intervention policy.
- Added presence/absence ownership checks, balanced brace/comment checks, and the
  executable `patient_story_charts_owner.test.js` contract.

## Verification

- Focused frontend gate: `15 passed, 1 warning`.
- Node owner contracts: Patient ECharts, clinical-story charts, Patient series,
  Patient quality, demo fidelity, and official demo source all passed.
- `node --check` passed for all modified JavaScript.
- `git diff --check` passed.
- Official MIMIC-IV Demo browser review at 1669×1354:
  - 140 entities, 19 modules, 281 catalog features, 151,373 rows.
  - Overview mounted one coordinated SVG chart; no fallback.
  - Quality mounted readiness + risk charts, four scope cards, and four priority
    issue rows; exact audit sections defaulted closed.
  - Module search reduced 19 rows to 4 for a clinical-term query and restored all
    19.
  - Gallery mounted six aligned focus tracks from 15 loaded trajectories; search
    reduced the individual browser to one matching feature.
  - `body` and `main` both had `scrollWidth == clientWidth`; final console
    warning/error count was zero.

## Browser evidence

- `output/ui-qa/20260729_patient_review_clinical_story/01_patient_overview.png`
- `output/ui-qa/20260729_patient_review_clinical_story/02_module_overview.png`
- `output/ui-qa/20260729_patient_review_clinical_story/03_data_quality.png`
- `output/ui-qa/20260729_patient_review_clinical_story/04_trajectory_gallery.png`

## Boundaries

- The browser remains a bounded review surface; it does not load all 184
  trajectory candidates at once.
- Catalog definitions, materialized columns, non-null observations, drawable
  patient trajectories, and bounded QC concepts remain separate denominators.
- No Copilot, backend execution, export, evidence gate, or Agent behavior changed.
- The branch is not merged and not pushed.
