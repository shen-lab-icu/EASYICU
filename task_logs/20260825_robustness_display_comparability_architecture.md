# Dev9 robustness-display comparability architecture repair

- Date: 2026-08-25
- Active task: `FIG2-DEV9-HELDOUT27`
- Exact code HEAD: `890305701fe29ba292dabe643882937a11419e8f`
- Authority: `analysis_only`; paper authorization remains false

## Root cause

The shared renderer treated any rows containing an estimate and confidence interval as if they were comparable effects. It did not require a common estimand, contrast, effect unit, scale, reference, or statistical independence. E2 therefore placed a prespecified 5.0-vs-2.1 mmol/L odds ratio, a per-1-mmol/L functional-form estimate, and a duplicated missing-data estimate on one odds-ratio axis. The resulting figure was visually valid but scientifically uninterpretable.

A second placement defect made routine exposure-measurement context a main figure even when measurement was not the study question.

## Generic owner repair

- Added `src/easyicu/research_agent/figures/robustness.py` as the dependency-neutral owner of robustness-display comparability.
- Common-effect display now requires explicit `estimand_id`, `contrast_id`, `effect_unit`, scale, convergence, and independence metadata.
- Unresolved comparability fails closed with stable reason code `ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED`.
- Non-comparable robustness rows render as a registered/converged/independent coverage matrix, not as a forest plot.
- The shared owner contains no E1, E2, M1, sepsis, lactate, bilirubin, or hard-coded result logic.
- Comparable article contrasts remain evidence-bound: the Dev9 adapter reads frozen contrast CSVs and validates one shared reference and confidence-interval containment before rendering.
- Routine measurement context is supplementary by default. A task may opt into main placement only when the measurement process is itself the research question; M1 does so, E2 does not.

## Exact regeneration

- Package: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_article_suite_8903057_20260825`
- Manifest SHA-256: `9f709849b3a81e295fc4245a61d7b206fb3c59d78156460d9294d107b59167f8`
- Inventory SHA-256: `2bd485437b4f2035bca34b25f9db238ff7ef45050d74cd2ae1f0a4e530191a7c`
- Provider calls: 0
- Scientific recomputation: false
- Aggregate inventory: 18 main figures, 7 supplementary figures, 17 main tables, 17 supplementary tables
- Contract placement unresolved: 0 for all nine tasks
- E2: 1 main association figure, 2 supplementary measurement figures, 2 main tables, 2 supplementary tables
- M1: 3 main figures because measurement/source process is the prespecified study subject

E2 retains an explicit planning warning (`single_composite_only`, `main_figure_count_outside_planning_target`). This is an honest evidence-coverage gap, not permission to promote missingness or manufacture another result figure.

## Verification

- Focused tests: 31 passed
- Ruff: passed
- `git diff --check`: passed
- Figure-contract JSON parsed: 25/25
- SVG XML parsed: 25/25
- PDF single-page check: 25/25
- PNG positive dimensions: 25/25
- Misleading display phrases absent from final figure contracts/SVGs: `Registered odds-ratio range`, `Robustness ranges`, `No recorded source`, `Outcome risk by source state`
- Manual visual review: E2 main continuous curve plus same-reference contrasts, both E2 supplementary measurement figures, and M1 main measurement figure; no clipping or shared-axis comparability error observed

## Claim boundary

This repair validates rendering semantics and evidence placement only. It does not upgrade any Dev9 item to paper-ready, does not add external validation, and does not authorize Qualification12 or Held-out27 execution.
