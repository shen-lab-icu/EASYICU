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

The plotted dimensions are:

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

- `output/fig2_canonical9_scorecard/canonical9_scorecard_heatmap.png`
- `output/fig2_canonical9_scorecard/canonical9_scorecard_heatmap.pdf`
- `output/fig2_canonical9_scorecard/canonical9_scorecard_heatmap.svg`
- `output/fig2_canonical9_scorecard/canonical9_scorecard_matrix.csv`

Design decisions:

- Gray `n/a` cells mean unscored/not applicable, not zero.
- Right-side bar shows mean over scored dimensions only.
- Parentheses on the right show scored dimensions out of 7 when any dimension is unscored.
- Output order is E1, E2, E3, M1, M2, M3, H1, H2, H3.

Key readout for group meeting:

- E1/E2/E3/H1/H2 show the same profile: plan partial, code/evidence/audit/fairness full, reporting partial, result-validity currently unscored.
- M1 is visibly weaker because evidence binding is 0 and three dimensions are unscored.
- M3 is visibly weaker because result validity is 0 and reporting completeness is 0.60.
- M2 and H3 have the strongest scored means among the nine, but H3 has only 5/7 scored dimensions.

## Verification

```bash
./.venv/bin/python tools/plot_canonical9_scorecard.py
./.venv/bin/python -m py_compile tools/plot_canonical9_scorecard.py
git diff --check -- tools/plot_canonical9_scorecard.py
./.venv/bin/python tools/lint_main_plan.py
```

Results:

- Plot script completed and wrote PNG/PDF/SVG plus source CSV.
- Python compile passed.
- Git whitespace check passed for the plotting script.
- Main plan lint passed with 8 status rows.
