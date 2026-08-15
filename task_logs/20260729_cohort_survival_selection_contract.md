# COHORT-SURVIVAL-SELECTION-CONTRACT

- Date: 2026-07-29
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Scope: Cohort Statistics → Survival curves
- User finding: the page silently chose `Sepsis vs Non-sepsis`, then presented a two-curve KM and log-rank result even though the user had not selected or confirmed a grouping definition.

## Contract change

The survival workbench now separates two valid analyses:

1. No comparison group selected:
   - one entire-cohort Kaplan-Meier curve;
   - one number-at-risk row;
   - `logrank.status = not_applicable`;
   - no p-value or effect contrast is rendered.
2. A ready comparison group is explicitly selected:
   - two group curves;
   - exploratory log-rank and absolute effect contrast;
   - the exact grouping basis is visible before the plot.

Outcome selection is also explicit. Outcomes without both an event flag and usable event/censoring time stay disabled and explain the missing input.

## Ownership and implementation

- Backend contract owner: `src/easyicu/webserver/cohort_review.py`
  - adds the `cohort` grouping option;
  - makes it the default;
  - computes one-curve KM without requiring a two-group split;
  - preserves comparative log-rank for ready two-group definitions.
- Frontend interaction owner: `src/easyicu/webserver/static/js/screens-viz-cohort-survival.js`
  - owns outcome/group selection, result framing, risk table and demo fallback;
  - removes the previous silent `sepsis` selection from `screens-viz.js`.
- CSS owner: `src/easyicu/webserver/static/css/cohort-survival.css`
  - survival selectors were removed from broad `cohort.css`;
  - the two-step setup does not force unequal sections to the same height.
- The old demo comparison was removed. Demo fallback now renders one simulated full-cohort curve and does not fabricate a grouping or log-rank result.

## Automated evidence

- `tests/test_webserver_workspace_summary.py`: `134 passed`
  - includes a no-comparison-metadata case where age, sex, SOFA and Sepsis splits are unavailable but overall KM remains ready.
- Static/owner regression:
  - `83 passed, 1 deselected`
  - deselected test is the pre-existing callback `project_ref.hint == EASYICU` assertion that fails solely because this isolated worktree directory is named `easyicu-copilot-cockpit-lite`.
- Node owner contract:
  - default chart series `1`;
  - explicit comparison series `2`;
  - selection triggers one repaint;
  - demo remains one series without p-value.
- `node --check` passed for the new owner and host screen.
- CSS owner presence/absence, foreign-route markers, brace/comment counts and `git diff --check` passed.

## Real browser evidence

Source: official MIMIC-IV prepared demo at `http://localhost:8876/?qa=density5#cohort`.

- Default outcome: `mort_28d`.
- Default group: `cohort`.
- Default UI:
  - heading `Kaplan-Meier 全队列曲线`;
  - selected receipt `全队列单曲线 · 不做组间检验`;
  - one ECharts SVG;
  - one risk-table row;
  - `组间检验：未运行`;
  - no comparative p-value.
- Explicit Sepsis selection:
  - heading changes to `Kaplan-Meier 组间比较`;
  - two risk-table rows;
  - `χ² 1.18 · p = 0.278`;
  - grouping definition states it uses the registered Sepsis-3 (SOFA-2) event module.
- Outcome selection:
  - switching to hospital mortality updates the endpoint, event/time fields and result title while preserving the explicitly selected group;
  - ICU mortality remains disabled with a source-specific reason.
- Reset to full cohort returns to one curve and no log-rank.
- 1280px desktop QA:
  - document width = viewport width = `1280`;
  - main `clientWidth == scrollWidth == 1032`;
  - no horizontal overflow.

## Deliberate boundary

This remains an aggregate exploratory review. It does not turn a selected split into a manuscript-ready claim, and it does not add adjusted survival models, confidence intervals, matching, or row-level cohort building.
