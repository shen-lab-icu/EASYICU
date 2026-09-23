# Fixed-landmark categorical association

Adjusted association between an ordered or categorical exposure classified
before a fixed landmark and a binary outcome observed after it, in ICU-stay
data with repeated stays per patient. Reference implementation:
`easyicu.research_agent.skill_packages.landmark_categorical_association`.

## Scope

This skill runs one prespecified design end to end:

- eligibility = alive and under observation at the landmark (the host's own
  `landmark_eligibility_mask`, so the pipeline and this package agree row for row);
- exposure = a closed level set fixed in the specification; a row whose exposure
  cannot be evaluated is kept as `unknown`, described in every denominator and
  excluded from every model — it is **never** recoded to the reference level;
- primary model = logistic regression with treatment contrasts against the
  declared reference level, the declared covariate roster, and patient-level
  cluster-robust covariance when repeated stays are declared (the host kernel
  `run_adjusted_association_from_env`);
- absolute risk per level with Wilson intervals; adjusted per-level trend
  (declared level index); Cochran–Armitage trend for the outcome and
  Jonckheere–Terpstra trend for continuous secondary outcomes, Holm-adjusted
  within the declared family;
- prespecified sensitivity refits: alternate exposure definitions,
  first-ICU-stay restriction, restricted-cubic-spline forms for continuous
  covariates (same kernel, same covariance);
- Table 1 by exposure level under the repeated-units schema (no p-values).

**What this skill does NOT do**

- Time-to-event modelling, competing risks or RMST — use the survival workflows.
- Causal identification (IPTW, matching, g-methods) — the claim ceiling is
  `analysis_only`; nothing here estimates an effect.
- Multiple imputation — the primary model is complete-case and says so.
- Exposure or covariate selection — every variable arrives in the specification.
- Landmarks other than the one declared; a moving or time-updated exposure
  belongs to the time-varying exposure workflow.
- Novelty or transportability claims — see `references/comparators_kdigo_mortality.md`
  for how a comparator table is filled in instead.

## When to use this skill

- The exposure is determined inside a fixed window after ICU admission
  (e.g. strict KDIGO stage at 0–24 h) and the outcome is a later binary event
  (in-hospital death) — the landmark removes immortal-time leakage.
- Several definitions of the same exposure exist and must be compared without
  choosing one after seeing results.
- Repeated ICU stays per patient are present and must be handled by design.

Do not use it when the outcome is a time and censoring matters, when the
exposure changes after the landmark, or when the question is causal.

## Inputs

One row per ICU stay with, at minimum:

| Column role | Spec field | Requirement |
|---|---|---|
| identity | `identity_column` | unique per row; `p<patient>:s<stay>` when the dependence contract derives the patient by prefix |
| exposure | `exposure` | values in `exposure_levels` or missing (= unknown) |
| outcome | `outcome` | 0/1, no missing |
| event time | `event_time_column` | hours from ICU admission; required when outcome = 1 |
| observation | `observation_duration_column` | hours or days of follow-up from ICU admission, ≥ 0 |
| covariates | `covariates[]` | continuous, binary or categorical with declared levels and reference |
| optional | `alternate_exposures[]`, `first_stay_column`, `secondary_outcomes[]`, `measurement_audit_columns[]` | as declared |

Formats: `.parquet`, `.csv`, `.tsv` or an in-memory `pandas.DataFrame`. The
specification is a `LandmarkCategoricalSpec` (JSON or Python); the E3-shaped
example is `example_spec()`.

## Outputs (all in the output directory)

Tables (CSV): `cohort_flow`, `exposure_level_counts`, `measurement_audit`,
`table_one`, `absolute_risk`, `adjusted_association_estimates`,
`adjusted_association_coefficients`, `adjusted_trend`, `ordinal_trend_tests`,
`secondary_outcome_summary`, `association_sensitivity_grid`,
`functional_form_sensitivity`, `robustness_summary`.

Headline: `key_metrics.csv` — one row; **the only source for every number a
report quotes**. `caveat_flags.json` — the machine-readable flags behind the
mandatory caveat sentences.

Figures (PNG + SVG, each with a `reader_caption` in `figures.json`):
`figure_cohort_flow`, `figure_exposure_ascertainment`,
`figure_absolute_risk_by_level`, `figure_adjusted_association_forest`,
`figure_robustness_forest`, `figure_secondary_<outcome>_by_level`.

Documents: `report.md` (numbers copied from the tables), `spec.json`,
`manifest.json` (SHA-256 of every file, plus `data_provenance`). Kernel outputs
per refit stay under `_kernel/<variant>/` for audit.

Provenance is declared, never inferred: pass `provenance="..."` to
`load_cohort`/`run_all` (CLI `--provenance`), or let a DataFrame carry
`attrs["provenance"]` as the synthetic example does. A cohort with neither is
recorded as `undeclared_by_caller` in the manifest and the report header, so a
missing declaration is visible instead of a blank field. Official demo data is
spelled `official_demo:<source_id>` (e.g. `official_demo:eicu_demo_v2_0_1`);
its results stay engineering material, never a clinical claim.

## Standard workflow

Use the entry functions exactly as shown. Do not write inline
`statsmodels`/`matplotlib` code for anything this skill already produces; if a
step needs something the specification cannot express, stop and extend the
specification (or the package) rather than the run.

```python
from easyicu.research_agent.skill_packages.landmark_categorical_association import (
    LandmarkCategoricalSpec, load_cohort, run_analysis, generate_all_plots, export_all,
)

spec = LandmarkCategoricalSpec.model_validate(spec_json)   # or example_spec()
cohort = load_cohort("cohort.parquet", spec)               # ✓ Cohort loaded and landmark-restricted successfully!
result = run_analysis(cohort, work_dir="results")          # ✓ Analysis completed successfully!
figures = generate_all_plots(result, "results")            # ✓ All plots generated successfully!
export_all(result, "results", figures=figures)             # === Export Complete ===
```

One call does all four in order: `run_all(cohort, spec, "results")`. Command line:

```bash
python -m easyicu.research_agent.skill_packages.landmark_categorical_association --example --out results_example
python -m easyicu.research_agent.skill_packages.landmark_categorical_association --cohort cohort.parquet --spec spec.json --out results --provenance "official_demo:eicu_demo_v2_0_1"
```

**Verification.** A step is complete only when its token has been printed:

1. `✓ Cohort loaded and landmark-restricted successfully!`
2. `✓ Analysis completed successfully!`
3. `✓ All plots generated successfully!`
4. `=== Export Complete ===` — printed only after the export consistency gate
   (`Consistency check: PASSED (k/k hard checks)`).

If a token is missing, the step failed; read the exception. Do not write the
missing artefact by hand.

**If the scripts fail — failure hierarchy**

1. Fix the input or the specification and rerun (most cases: an undeclared level,
   a missing column, a non-binary outcome).
2. Modify the package script, add a test, record the change.
3. Use the script as a reference and cite it.
4. Write from scratch — only when the design is genuinely outside this family,
   and say so in the report.

## Numbers and caveats in any downstream report

- Copy every N, OR, CI, p-value and proportion from `key_metrics.csv` or the
  exported tables. Never recompute or round from memory.
- Every raised flag has a mandatory sentence (`caveat_sentences()`;
  `references/caveat_flags.md`): `epv_below_minimum`, `separation_detected`,
  `complete_case_warning`, `unknown_exposure_share_high`,
  `small_reference_group`, `sparse_exposure_levels`, `empty_exposure_levels`,
  `sensitivity_direction_consistent = false`, failed refits, non-estimable
  functional forms. The closing sentence on the `analysis_only` ceiling is
  always present.
- Report the unknown-exposure row wherever the known levels are reported.
- Name the landmark and state that the estimand conditions on surviving to it.

## Related workflows

| Need | Use |
|---|---|
| Continuous exposure with dose–response | landmark spline dose-response workflow |
| Time-to-event outcome | landmark / time-varying survival workflows |
| Causal contrast | adjusted exposure–outcome study with the causal method suite |
| Ordered exposure, descriptive only | ordinal dose-response module |

## References

Methods and thresholds: `references/methods.md`. Flags → sentences:
`references/caveat_flags.md`. Reporting checklist mapping (STROBE / RECORD):
`references/reporting_checklist.md`. Comparator table template for KDIGO
stage–mortality studies: `references/comparators_kdigo_mortality.md`.
