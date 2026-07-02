# 2026-06-27 Fig2 Canonical9 scorecard heatmap

Task: generate a group-meeting-ready figure showing how the 9 canonical benchmark questions score across the Agent evaluation dimensions.

Active phase/task: `FIG2-CANONICAL9-GATE` and `WEBAPP-FASTAPI-NATIVE-QA`.

## Inputs

Source scorecards:

- `/Users/haibo/easyicu/projects/fig2-e1-sepsis3-mortality/run_20260613T004906_66dc3b/benchmark_scorecard.json`
- `/Users/haibo/easyicu/projects/fig2-e2-lactate-mortality/run_20260613T004906_f281a7/benchmark_scorecard.json`
- `/Users/haibo/easyicu/projects/fig2-e3-kdigo-gradient/run_20260613T004906_f38135/benchmark_scorecard.json`
- `/Users/haibo/easyicu/projects/fig2-m1-hepatobiliary-missingness/run_20260613T014227_75a79f/benchmark_scorecard.json`
- `/Users/haibo/easyicu/projects/fig2-m2-mortality-prediction/run_20260613T015642_eeb0fc/benchmark_scorecard.json`
- `/Users/haibo/easyicu/projects/fig2-m3-sepsis-subphenotype/run_20260613T015846_4a42a8/benchmark_scorecard.json`
- `/Users/haibo/easyicu/projects/fig2-h1-ventilation-survival/run_20260613T004906_073471/benchmark_scorecard.json`
- `/Users/haibo/easyicu/projects/fig2-h2-vasopressor-causal/run_20260613T013103_96fbb9/benchmark_scorecard.json`
- `/Users/haibo/easyicu/projects/fig2-h3-trajectory-clustering/run_20260613T013708_34719e/benchmark_scorecard.json`

The exported full-audit dimensions are:

- Plan completion
- Code execution
- Result validity
- Evidence binding
- Audit / conclusion safety
- Reporting completeness
- Fairness / subgroup

## Imagegen storyboard

Used the built-in `image_gen` tool as a non-data storyboard only, per project figure rule. The generated image is not used as the result figure and does not contain trusted values.

Saved storyboard copy:

- `output/fig2_canonical9_scorecard/canonical9_scorecard_storyboard_imagegen.png`

Prompt summary: landscape scientific score-matrix layout, 9 questions x evaluation dimensions, right-side total-score strip, colorblind-safe heatmap style, no fabricated numeric values.

## Code-backed final figure

Script:

- `tools/plot_canonical9_scorecard.py`

Generated artifacts:

- Group-meeting main figure:
  - `output/fig2_canonical9_scorecard/canonical9_scorecard_core_heatmap.png`
  - `output/fig2_canonical9_scorecard/canonical9_scorecard_core_heatmap.pdf`
  - `output/fig2_canonical9_scorecard/canonical9_scorecard_core_heatmap.svg`
  - `output/fig2_canonical9_scorecard/canonical9_scorecard_core_matrix.csv`
- Chinese talk track:
  - `output/fig2_canonical9_scorecard/canonical9_scorecard_talk_track_zh.md`
- Dimension inclusion audit:
  - `output/fig2_canonical9_scorecard/canonical9_scorecard_dimension_audit.csv`
- Full 7-dimension audit backup:
  - `output/fig2_canonical9_scorecard/canonical9_scorecard_heatmap.png`
  - `output/fig2_canonical9_scorecard/canonical9_scorecard_heatmap.pdf`
  - `output/fig2_canonical9_scorecard/canonical9_scorecard_heatmap.svg`
  - `output/fig2_canonical9_scorecard/canonical9_scorecard_matrix.csv`

Design decisions:

- The main group-meeting figure keeps five presentation columns: plan completion, code execution, result sanity, evidence link, and reporting checklist.
- The left-side task labels are short question-style descriptions rather than keyword titles, so a viewer can immediately tell whether the row is an association, prediction, causal/confounding-aware, subphenotype, or trajectory task.
- `result_sanity` is a derived deterministic presentation column. It uses hard `result_validity` failures when present; otherwise it reads `source_run_manifest.json` and scores whether execution completed and both numeric audit and analysis validation passed. This makes the fifth column explainable without requiring a frozen gold reference.
- `result_validity` is excluded from the main figure because only 1/9 tasks is scored; the other tasks are unscored because locked numeric references are not frozen.
- `audit_conclusion_safety` is excluded from the main figure because all imported scorecards show 1.0 while the notes say no per-task hazard key was available; this is a fail-closed floor check, not a comparable full hazard-handling score.
- `fairness_subgroup` is excluded from the main figure because it is scored in only 6/9 tasks and every scored value is 1.0, making it non-discriminative and task-applicability dependent.
- Gray `n/a` cells mean unscored/not applicable, not zero.
- Right-side bar shows mean over scored dimensions only.
- Parentheses on the right show scored dimensions out of the plotted dimension count when any dimension is unscored.
- Output order is E1, E2, E3, M1, M2, M3, H1, H2, H3.
- 2026-06-27 palette refresh: score gradients now use a Nature-style muted red -> neutral -> blue-green/blue scale; all source scores and row/column definitions are unchanged.

Key readout for group meeting:

- E1/E2/E3/H1/H2 share the same core pattern: plan partial, code/result sanity/evidence link full, reporting partial.
- M1 is visibly weaker because the execution gate did not produce a reportable result, result sanity is 0, evidence link is 0, and reporting completeness is not scored for the emitted checklist type.
- M3 is weaker because the hard result-validity gate caught a near-empty/degenerate cluster, so result sanity is 0 despite completed execution and full evidence binding.
- M2 and H3 have the strongest core means among the nine because both have full plan/code/result sanity/evidence link scores and slightly better reporting checklist completion.

## Verification

```bash
./.venv/bin/python tools/plot_canonical9_scorecard.py
./.venv/bin/python -m py_compile tools/plot_canonical9_scorecard.py
git diff --check -- tools/plot_canonical9_scorecard.py
./.venv/bin/python tools/lint_main_plan.py
```

Results:

- Plot script completed and wrote PNG/PDF/SVG plus source CSV.
- The core PNG was visually inspected after regeneration.
- Python compile passed.
- Git whitespace check passed for the plotting script.
- Main plan lint passed with 8 status rows.
- 2026-06-27 Nature palette refresh regenerated PNG/PDF/SVG/TIFF and was visually checked in `output/fig_refresh_qa/fig2_fig3_nature_palette_contact_sheet.png`.
