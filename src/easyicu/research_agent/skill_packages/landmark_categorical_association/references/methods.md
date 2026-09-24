# Methods used by the fixed-landmark categorical association skill

Every calculation is delegated to an existing EasyICU owner; this note records
which owner, what it assumes and which threshold the report applies.

## Landmark eligibility

`execution.runners.landmark_categorical_association_executor.landmark_eligibility_mask`
keeps rows with a non-negative event time, alive at the landmark (`outcome == 0`
or `event_time > landmark_hours`) and observed at the landmark
(`observation_duration >= landmark`). Classifying the exposure only from
information before the landmark and starting follow-up at the landmark is the
standard remedy for immortal-time bias (Anderson et al. 1983; Suissa 2008). The
estimand therefore conditions on surviving to the landmark; deaths before it are
counted in `cohort_flow.csv`, not analysed.

## Exposure states

Observed values are matched to the declared level set with the host's
`level_spelling` (numeric `3.0` and text `"3"` are the same level). Missing
values form the `unknown` state: reported in `absolute_risk.csv`,
`exposure_level_counts.csv`, `measurement_audit.csv` and Table 1's
`group_missing_excluded_n`; excluded from every model. When the exposure
definition needs complete evidence for its reference level (a strictly
ascertained grade whose lowest level requires every contributing domain to be
observed), this state is scientifically meaningful, so the share of unknown rows
is itself a headline number (`unknown_exposure_share`, flagged at
`unknown_exposure_warning_share`, default 10 %).

## Primary model

`execution.runners.adjusted_association_executor.run_adjusted_association_from_env`
fits `statsmodels` logistic MLE with the declared treatment contrasts and
covariates and, when a dependence contract is declared, cluster-robust
covariance with patients derived by `contracts.dependence.resolve_patient_groups`.
The primary contrast row is the declared `primary_contrast_level` versus the
reference. The kernel refuses to write a null estimate; a model that cannot be
fitted as declared raises and the run stops (`AnalysisContractError`).

Events per parameter (EPV) = events among fitted rows / number of non-constant
design columns; flagged below `epv_minimum` (default 10; Peduzzi et al. 1996 as
the conventional rule of thumb, not a proof of adequacy). Separation is the
kernel's own `separation_detected` verdict.

## Adjusted trend

The exposure is recoded as the declared level index (`ordinal_linear`,
`declared_level_index`) in the same logistic model; the coefficient is reported
as an odds ratio per one-level increment. This is a test of a linear trend in
the log-odds across the declared ordering, not a claim that levels are
equidistant.

## Ordered trend tests

`methods.ordered_trends.cochran_armitage_trend` (binary outcome, consecutive
scores) and `methods.ordered_trends.jonckheere_terpstra_trend` (tie-corrected
asymptotic JT for continuous secondary outcomes). The declared family for Holm
adjustment (`statsmodels.stats.multitest.multipletests`) is the outcome trend
plus one JT test per secondary outcome on all exposure-known rows; the
survivors-only JT is descriptive and outside the family because deaths shorten
a length of stay and the two denominators answer different questions.

## Absolute risk

`methods.ordered_trends.wilson_interval` per level and for the unknown state.

## Sensitivity refits

Each refit runs the identical kernel: alternate exposure definitions on the rows
where that definition is evaluable; first-ICU-stay restriction on rows whose
flag is `1`/`true`. `log_or_delta_vs_primary` and
`direction_consistent_with_primary` compare the primary-contrast OR with the
primary model; a refit that cannot be estimated stays in the grid with
`fit_status = failed` and its reason.

## Functional form

For each declared continuous covariate the primary design is rebuilt with the
covariate replaced by a restricted cubic spline basis from
`methods.rcs_dose_response.rcs_basis` (Harrell knot quantiles; default 4 knots)
and refitted with `robustness.estimators.fit_estimator` under the same
covariance contract; the primary-contrast OR is compared with the linear form.
The joint Wald test of the nonlinear spline terms comes from
`methods.rcs_dose_response.rcs_fit` + `nonlinearity_wald_test` and uses a
model-based covariance (no clustering) — it is labelled as such in the table.
Knot placement can fail on discrete covariates with few distinct values; that
row is then `not_estimable` with the reason and is listed in the flags.

## Table 1

`methods.table_one.build_grouped_table_one` under schema
`easyicu.table_one/2` (repeated units: no p-values, SMD only where two groups),
`missing_group_policy = exclude_and_report` so the unknown-exposure count
travels on every row.

## Thresholds (declared in the specification, quoted by the report)

| Field | Default | Meaning |
|---|---|---|
| `epv_minimum` | 10 | events per parameter below which the model is flagged |
| `complete_case_warning_share` | 0.20 | share of exposure-known rows dropped for missing covariates that triggers the selection-bias caveat |
| `unknown_exposure_warning_share` | 0.10 | share of landmark rows with unknown exposure that triggers the ascertainment caveat |
| `small_reference_group_n` | 50 | reference-level size below which contrasts are flagged |
| `sparse_level_events` | 5 | events per level below which the level is flagged |

## References

- Anderson JR, Cain KC, Gelber RD. Analysis of survival by tumor response. *J Clin Oncol* 1983;1:710–719.
- Suissa S. Immortal time bias in pharmaco-epidemiology. *Am J Epidemiol* 2008;167:492–499.
- Peduzzi P, et al. A simulation study of the number of events per variable in logistic regression analysis. *J Clin Epidemiol* 1996;49:1373–1379.
- Harrell FE. *Regression Modeling Strategies*, 2nd ed. Springer 2015 (restricted cubic splines, knot quantiles).
- Jonckheere AR. *Biometrika* 1954;41:133–145; Terpstra TJ. *Indag Math* 1952;14:327–333.
- von Elm E, et al. STROBE statement. *PLoS Med* 2007;4:e296. Benchimol EI, et al. RECORD statement. *PLoS Med* 2015;12:e1001885.
