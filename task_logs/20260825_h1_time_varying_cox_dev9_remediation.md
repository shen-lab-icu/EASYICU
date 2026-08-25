# H1 deterministic time-varying Cox and Dev9 article-package closure

Date: 2026-08-25 EDT

Authority: development evidence only (`analysis_only`); paper authorization
remains 0/9.

## Problem

The frozen H1 landmark Cox analysis had a severe global proportional-hazards
violation (`p=7.889e-35`). The constant hazard ratio was correctly withheld, but
the article package lacked a deterministic time-varying adjusted sensitivity.

## Owner-scoped remediation

- Added the generic pure method owner
  `methods/time_varying_cox.py::fit_piecewise_time_varying_cox`.
- Expanded sealed complete-case landmark data into start-stop intervals and fit
  all registered covariates with interval interactions. The case protocol fixes
  the post-landmark intervals at 0-7, 7-14 and 14-27 days.
- The landmark-survival executor now writes
  `landmark_time_varying_cox_summary.csv` and displays interval-specific
  adjusted contrasts when global PH is rejected. It does not restore the
  invalid constant-HR claim.
- The article renderer separates H1 into four main figures: survival with risk
  table, RMST, time-varying associations, and risk-set/PH diagnostics.
- The method implementation is case-neutral. H1-specific cut points and the
  published comparator citation stay in the H1 protocol rather than in shared
  prompts or global logic.

## Exact evidence

- Scientific commit: `de6403a246ef995fa4ab9e72b7d7e387b8e47281`.
- H1 exact deterministic run:
  `/Volumes/外置硬盘/easyicu_data/figure2_dev9_h1_timevarying_de6403a_20260825/`.
- H1 manifest:
  `/Volumes/外置硬盘/easyicu_data/figure2_dev9_h1_timevarying_de6403a_20260825/exact_h1_remediation_manifest.json`.
- Full Dev9 article package:
  `/Volumes/外置硬盘/easyicu_data/figure2_dev9_article_display_full_de6403a_20260825/`.
- Full package manifest:
  `/Volumes/外置硬盘/easyicu_data/figure2_dev9_article_display_full_de6403a_20260825/article_display_remediation_manifest.json`.

The H1 analysis retained 78,600 landmark-eligible patients and 78,580 complete
cases. The exact run took 7.69 seconds and used zero Provider, Planner or Coder
calls, zero model tokens and zero model cost. Interval-specific estimates were
stable across the three registered intervals, but they remain observational
development results and are not interpreted causally or clinically here.

## Verification

- New/adjacent focused tests: 37 passed.
- Broader H1/M3/H3 owner and boundary suite: 79 passed.
- Black, Ruff, `py_compile` and `git diff --check`: passed.
- Full article package: 24 FigureContracts = 18 main + 6 supplementary.
- Export QA: 24/24 PNG, SVG, PDF and TIFF present and non-empty; 24/24 SVG XML
  valid; 24/24 PDFs single-page; all raster geometry above the minimum gate.
- Every contract has a core claim, statistics note and existing source-data
  file. Contact-sheet review found no clipping, missing axes or promotion of
  routine missingness into the primary clinical-result figures.

## Remaining boundary

H2 still lacks a verified non-use comparison group and H3 still has no interior
BIC optimum; both therefore remain fail closed with zero main result figures.
M3 retains stability and alternative-algorithm agreement, but low agreement
prevents phenotype naming. These are correct non-results, not gaps to fill with
question-specific logic. Dev9 remains a development split and cannot enter the
paper denominator.

## Exact-head architecture closure

The first remote freeze attempt exposed six architecture-ratchet failures that
already existed at the branch's incoming `04e5a60` baseline. They were closed
without increasing the LOC ratchet: reviewed literature projection, required
plan obligations, deterministic panel finding propagation, and planner-only
runtime handling now live behind their existing owner contracts. Final
architecture measurements are `pipeline.py` 8,414 versus baseline 8,419;
`run` 778 versus 782; `_generate_or_resume_plan` 449 versus 449; and
`_run_plan_phase` 618 versus 618. The module graph explicitly seals the three
new typed owners (`external_benchmark_authority`, `time_varying_cox`, and
`sensitivity_plan_shaping`) with 594 modules, 2,425 directed edges and zero
cycles. The combined focused freeze suite is 182 passed; the independent M3/H3
fail-closed adjacency suite is 66 passed.
