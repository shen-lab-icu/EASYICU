# Host method

Call chain: association family (`association_adjusted_v1`) →
`association.adjusted_association` (`analysis_method_suite.py`,
`method_adapter_catalog.py`) → `adjusted_association_executor_code` →
`run_adjusted_association_from_env` → `execution.model_matrix.primary_model_rows`
→ `robustness.estimators.fit_estimator` for binary/continuous outcomes, or
`methods.ordered_trends.fit_outcome_model` for ordered/free-days/count outcomes.
The scripts call this same runtime owner.

The explicit `PlannedModelRequirement` selects statsmodels Logit MLE (logit link,
Newton optimizer, maxiter=100, OR), OLS (identity link, default pseudoinverse fit,
coefficient), or modified Poisson GLM
(Poisson/log link, default IRLS, maxiter=100, binary outcome, RR). Logit and OLS default to
model-based covariance. Modified Poisson uses HC0 sandwich covariance. A declared
`PlannedDependenceRequirement` selects statsmodels `cov_type="cluster"` with the
host's resolved patient groups and library-default correction/inference options;
no cluster is inferred from a column name. In the validated statsmodels 0.15.0
environment, cluster covariance has use_correction=True, adjust_df=True and
use_t=False (normal Wald inference); the host passes no overrides.
OLS model-based intervals use its
library-default t inference; binary model intervals use normal Wald inference.

New outcome types require `outcome_definition.estimand`. An ordinal definition
lists its ordered levels; `statsmodels_ordered_logit` estimates a common OR
toward higher levels and reports a proportional-odds score test. Free days use
that model over complete 24-hour free days: the plan declares the horizon,
support concept (or union of support concepts) and death score -1 or 0. Death
at or before the horizon overrides support duration. The callback unions
duration-bearing support intervals clipped to [0, horizon], subtracts from
the horizon and counts complete free days. All coordinates are hours from ICU
admission. The concept owner requires support coverage and survival follow-up
through the horizon; an admission flag, isolated samples or discharge alone
cannot establish ventilator-free days. Missing source capability refuses with
a typed reason. Derivation audits contain death-status counts, frequencies and
median/IQR without identifiers or patient rows.

Count definitions prespecify `statsmodels_count_poisson` or
`statsmodels_negative_binomial` (NB2), and a positive exposure-time column or
finite log offset; null exposure declares counts per observation. The offset
coefficient is fixed at one. Both report a rate ratio and 95% normal Wald CI
and an NB2 dispersion score under a Poisson null. Ordered and count models
honor declared cluster covariance. Diagnostics never switch the model. A PO
violation limits common-OR interpretation to analysis only. Changing the
estimand, levels, death score, horizon or offset invalidates prior plan approval.
Hurdle, zero-inflated and win-ratio sensitivities are not implemented.

Ordered/free-days/count fits also publish an aggregate outcome distribution;
free days include median/IQR, and ordered/free-days outcomes include an ordered
bar figure. Table 1 uses the existing ordinal `median_iqr` or `count_percent`
row contracts on callback-derived values.

The host compiles every `ModelTermSpec` using the declared transform, closed
levels, and reference. Categorical exposures yield every non-reference contrast
and retain the declared headline contrast. Rows with missing outcome or predictors
are excluded by default; an explicit host-owned `baseline_missing_handling`
retains the declared covariates' unmeasured state through the same host compiler.
For a retained continuous/ordinal covariate the compiler fills the observed
median over fitting rows and adds an unmeasured indicator; a retained discrete
covariate gains an unmeasured level. An unmeasured state with fewer than 20 rows,
or only one outcome class in a binary model, is demoted and those rows are
dropped; the host repeats this check until the fitting rows stop changing.
No policy or reference is selected by these scripts. Rank-deficient, unestimable,
or nonconverged models fail rather than drop declared terms.

Outputs are `adjusted_association_estimates.csv`,
`adjusted_association_coefficients.csv`, `step_summary.json`, and a forest plot of
the reported contrasts. The CSVs are unrounded; SEs stay on the fitted scale.
The host exports transformed coefficients/CI (OR/RR) or raw OLS coefficients,
and does not export coefficient p values. The scripts do not synthesize absent
p values. GLM-binomial, mixed models, and signed subgroup variants are not owned
by this generic executor and are refused.

Dependencies: easyicu, numpy, pandas, pyarrow, pydantic, scipy, statsmodels, matplotlib.
The installed EasyICU package also supplies `skill_packages.reference_io`.
