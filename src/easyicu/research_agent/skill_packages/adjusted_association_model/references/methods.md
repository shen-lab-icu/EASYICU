# Host method

Call chain: association family (`association_adjusted_v1`) →
`association.adjusted_association` (`analysis_method_suite.py`,
`method_adapter_catalog.py`) → `adjusted_association_executor_code` →
`run_adjusted_association_from_env` → `execution.model_matrix.primary_model_rows`
→ `robustness.estimators.fit_estimator`. The scripts call this same runtime owner.

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
