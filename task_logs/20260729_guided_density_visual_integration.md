# WEB-COPILOT-COCKPIT-LITE — Guided density and visualization integration

Date: 2026-07-29
Branch: `codex/web-copilot-cockpit-lite-20260729`

## Outcome

- Expanded the Guided conversation canvas to 1040 px and made inline review
  cards consume the available message width.
- Replaced the vertically stacked Patient + Cohort + KM review with a
  summary-first workspace that renders exactly one Patient, Cohort, or Survival
  view at a time.
- Reused the existing Patient Review and Cohort Statistics payloads. The
  default Survival view mounts the shared ECharts Kaplan-Meier option from the
  real active export aggregate; it does not introduce a seeded visualization.
- Reduced the left project rail to one-line project names with a state dot.
  Phase, next decision, time, and path remain in the tooltip/accessible label.
- Added route-pure CSS/JS owners:
  - `static/js/screens-guided-review-workspace.js`
  - `static/css/guided-review.css`
  - `static/css/guided-projects.css`

## Verification

- JavaScript syntax checks: pass.
- Pure review workspace contract: 32 cases passed.
- Focused Guided tests: pass.
- Static + UX suite: 82 passed, 1 unrelated worktree-path failure:
  `test_native_extraction_feature_definition_manifest_records_callback_provenance`
  expects repository hint `EASYICU`, while an isolated worktree correctly
  reports `easyicu-copilot-cockpit-lite`.
- CSS ownership scan: balanced braces/comments, no `!important`, no `:has`,
  no review selectors in `guided.css`, and no project selectors in the review
  owner.
- Browser QA at 1669 × 1354:
  - review card width 467.7 px → 945 px
  - conversation width 760 px → 1040 px
  - project rows 34 px, one line, 12 visible
  - Patient/Cohort/Survival switches each showed one live panel
  - one real ECharts survival canvas mounted
  - zero horizontal overflow
  - no console warning/error

QA artifacts:
`output/ui-qa/20260729_guided_density_visuals/`
