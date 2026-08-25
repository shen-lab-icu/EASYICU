# Dev9 article-level figure package remediation

Date: 2026-08-25 EDT  
Renderer commit: `1203aca4821f10aabe2906d0a2f98f359b859c34`  
Branch: `codex/literature-to-design-contract`

## Decision

One four-panel composite is one figure, not a complete ICU article package.
The accessible comparator pack contained 14 published papers. Excluding one
JAMA extraction failure from the count summary, 13 full texts contained 2–10
main figures (median 3) and 1–4 main tables (median 2). Eleven of the 14 papers
had accessible supplementary material; complex phenotype/trajectory papers had
substantially larger eFigure/eTable packages.

The framework now uses a non-binding planning target of 2–4 complementary main
figures plus 2–3 main tables, followed by question-specific supplementary
material. This is not a count-based publication gate.

Routine missingness and measurement-process evidence remains mandatory but its
visual placement is supplementary. It can be promoted into a main figure only
when missingness/measurement is explicitly the scientific question or changes
the estimand, denominator, or interpretation of the primary result.

## Generic contract repair

- `FigureRoleStrategy` now records `placement=main|supplementary`.
- Association, prediction, phenotyping and descriptive data-quality visual
  roles default to supplementary.
- Explicit missingness/measurement questions deterministically promote the
  data-quality role to main.
- The article-figure audit rejects a supplementary-only role placed in a
  primary publication figure.
- The Planner blueprint now requests an article-level suite rather than one
  primary composite and explicitly separates routine audit detail.
- The article-analysis contract still requires typed data-quality evidence even
  when its visual destination is supplementary.

## Frozen-source redraw

Output root:

`/Volumes/外置硬盘/easyicu_data/figure2_dev9_article_display_remediation_1203aca_20260825/`

E3 was split into:

1. `e3_main_figure_1_outcomes_by_kdigo_stage`: mortality risk with source Wilson
   95% CI and ICU length of stay median/IQR across all four ordered stages.
2. `e3_main_figure_2_adjusted_and_sensitivity`: all fitted primary contrasts and
   all five converged scientific sensitivity analyses.
3. `e3_supplementary_figure_s1_missingness`: all 17 routine measurement/
   missingness rows, explicitly labelled supplementary.

M2 was split into:

1. `m2_main_figure_prediction_performance`: calibration, ROC, precision-recall,
   and mean ± SD across 10 patient-level repeated splits.
2. `m2_main_figure_prediction_performance_supplementary_decision_curve`:
   exploratory decision-curve evidence, explicitly labelled supplementary
   because no clinical threshold or external utility was authorized.

Every figure has PNG, SVG, PDF, TIFF, a FigureContract and source CSV. The
package contains 36 files and five FigureContracts. Source rows consumed are
E3 4/3/5/17 and M2 94,458/1/1/11/10, matching the frozen parents.

## Verification

- Focused planning/rendering/article-contract regressions: `52 passed`.
- Ruff format/check and `git diff --check`: passed.
- Exact package manifest records code HEAD `1203aca4821f10aabe2906d0a2f98f359b859c34`.
- Provider/Planner/Coder calls for the redraw: `0`; tokens/cost: `0`.
- Scientific recomputation: `false`; all plots consume registered frozen rows.
- Five SVG files passed `xmllint`; five PDFs are single-page 183 mm exports;
  PNG dimensions range from 2,160×884 to 2,220×1,299.
- All five exact-output PNG hashes match the visually reviewed preflight render.
- FigureContract audit produced only two expected duplicate-role warnings:
  mortality and LOS are complementary `descriptive_result` panels, while ROC
  and PR are complementary `model_performance` panels. No error finding exists.

## Authority boundary

This repairs article structure and deterministic display quality only. It does
not change cohorts, estimates, models, task scores, or scientific authority.
Dev9 remains `analysis_only`, paper-authorized 0/9. H2 remains correctly without
an effect figure, and M2 still lacks external validation.
