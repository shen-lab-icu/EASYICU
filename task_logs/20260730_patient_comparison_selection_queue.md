# Patient comparison selection queue

- Date: 2026-07-30
- Task: `PATIENT-COMPARISON-SELECTION-QUEUE`
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Scope: Native Patient Review → Multi-patient comparison

## User-facing problem

The comparison page exposed a paginated list of patients but silently plotted the
first five entities from that page. Users could not search for, add, or remove
comparison patients, and changing a discovery page implicitly changed chart
membership.

## Implemented contract

- Replaced the generic single-patient navigator with an explicit comparison queue.
- The queue contains 2–5 pseudonymous entities and is the only source of chart
  membership.
- Added explicit add/remove controls and an honest full-queue disabled state.
- Added local exact patient/stay identifier search; results remain pseudonymous,
  the raw identifier clears immediately, and it is not rendered or added to the
  URL.
- Kept patient pagination as discovery only. Changing pages does not replace the
  selected queue or the chart legend.
- Scoped feature projections and caches to the ordered selected-entity refs.
- Kept module-first and feature-second comparison selection for all 19 modules,
  including trajectory, static numeric, and static categorical charts.
- Removed the duplicate ECharts value-axis unit that collided with the scroll
  legend; units remain in the feature header and tooltip.

## Browser evidence

Official MIMIC-IV Demo:

- Initial queue: 5 patients.
- Removed Entity 2: selected queue and chart legend both changed to 4.
- Added Entity 6: selected queue and chart legend both changed to 5.
- Moved from page 1 to page 2: selected queue and chart legend were unchanged.
- Searched an exact local demo stay ID: result returned as Entity 17; input,
  rendered body, and URL contained no raw identifier after the response.
- Added Entity 17 after freeing one slot: selected queue and chart legend both
  became Entity 1, Entity 3, Entity 4, Entity 5, Entity 17.
- Document and main-content horizontal overflow: 0.
- Duplicate `bpm` axis title in the chart SVG: absent.

Screenshots:

- `output/ui-qa/20260730_patient_comparison_selection_queue/01-explicit-selection-queue.png`
- `output/ui-qa/20260730_patient_comparison_selection_queue/02-queue-synced-chart.png`

## Verification

- Node syntax checks: passed for Patient comparison, series, charts, and host.
- Executable comparison-owner contract: passed.
- Patient browse frontend + native static route suite:
  `78 passed, 1 deselected`.
- The deselected test is the known worktree-name provenance assertion
  (`easyicu-copilot-cockpit-lite` vs `EASYICU`), unrelated to this patch.
- Route-specific CSS ownership/presence/absence and balanced-syntax contract:
  passed.
- `git diff --check`: passed.
