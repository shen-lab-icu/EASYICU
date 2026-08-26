# Generic article-display policy and unseen-role stress test

- Date: 2026-08-25 EDT
- Active task: `FIG2-DEV9-HELDOUT27`
- Preflight base HEAD: `563ed02b3086b936527441abc583b1812d97943e`
- Authority: `analysis_only`; paper authorization remains false

## Problem

The deterministic figure executors produced evidence-bound plots, but article
placement and display-purpose metadata were not compiled by one runtime owner.
The Dev9 packaging layer still contained incomplete table-purpose overrides.
That made a new question vulnerable to an unowned article role or an unresolved
main/supplementary placement during Qualification12 or Held-out27.

## Generic repair

- Added the dependency-neutral `ArticleDisplayPolicy` owner. It maps typed
  article roles to `main|supplementary` and
  `scientific_result|diagnostic|context|audit` without inspecting task ids,
  variable names, database names, titles, or numeric values.
- Every `FigureContract` now receives an immutable placement, display purpose,
  and stable policy reason code from that owner.
- Failed-closed analyses cannot emit scientific-result displays. A terminal
  diagnostic may be shown in the main article only when explicitly declared
  central to the research question.
- Routine measurement/missingness evidence remains supplementary; it becomes a
  main scientific result only when data quality is the typed research question.
- Registered robustness and sensitivity results are scientific-result evidence;
  assumption checks, feasibility, stability, and selection diagnostics remain
  diagnostic evidence.
- Unknown article roles fail at the policy owner with
  `ARTICLE_DISPLAY_ROLE_UNSUPPORTED`. Conflicting hand-declared purposes fail
  with `ARTICLE_DISPLAY_PURPOSE_CONFLICT`.
- Dev9 table roles are now complete and typed. The shared policy itself contains
  no Dev9 case, exposure, outcome, database, or result names.

## Unseen-role evidence

Random labels (`flux_7q`, `ion_zeta`, `split_sigma`, `marker_kappa`, and
`signal_omega`) produced identical decisions from their typed roles alone.
A source scan asserts that every statically declared runtime role is accepted by
the policy and that case-specific tokens are absent from the owner.

The first full provider-free Dev9 replay failed on the previously unowned,
general role `validation_design`. The owner was extended for validation-design
context and the replay was restarted in a new output directory. A subsequent
inventory exposed a second general semantic defect: scientific robustness was
being counted as a diagnostic when combined with a primary estimate. Moving the
generic `robustness` role to scientific-result purpose cleared the defect without
using E3, KDIGO, or any result value in the policy.

## Preflight replay and QA

- Successful package:
  `/Volumes/外置硬盘/easyicu_data/figure2_dev9_article_suite_policy_probe3_20260825`
- Provider calls: `0`; tokens/cost: `0`
- Scientific recomputation: `false`
- Nine task packages generated; unresolved placement: `0`; unresolved display
  purpose: `0`
- Figure contracts/PNG/SVG/PDF/TIFF: `26/26/26/26/26`
- All 26 contracts have source data, a core claim, a statistics note, and a
  display-policy receipt; all SVGs retain text; checked PDFs are single-page.
- Related figure/publication/article-display regression: `1194 passed, 1 skipped`
- Focused policy tests: `11 passed`; Ruff and `git diff --check`: passed.
- Visual review covered E2 main association and supplementary measurement
  context, E3 primary plus sensitivity, M2 calibration/discrimination, and the
  H2/H3 failed-closed diagnostics; no new clipping or semantic-axis defect was
  observed.

## Remaining boundary

E2 honestly retains `single_composite_only` and
`main_figure_count_outside_planning_target`: it has one genuine main result
figure and two supplementary measurement figures. The policy does not invent or
duplicate another scientific result to satisfy a count target.

This preflight proves current typed figure executors and article-role contracts
can traverse one case-neutral placement policy. It does not prove that unknown
Held-out27 questions will have sufficient data, correct scientific designs, or
paper-ready results. Qualification12 remains the required development gate
before any one-shot Held-out27 run.
