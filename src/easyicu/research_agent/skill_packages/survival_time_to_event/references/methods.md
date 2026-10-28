# Host method

Call chain: time-to-event family (`survival_time_to_event_v1`) →
`time_to_event.cox_hr` (`analysis_method_suite.py`, `method_adapter_catalog.py`) →
`survival_primary_executor_code` → `run_survival_primary` →
`_fit_declared_cox` / lifelines `CoxPHFitter()`, plus `methods.ph_schoenfeld.ph_test`.
The scripts invoke that same runtime owner in a child process so STEP_OUT_DIR
cannot leak into the caller's environment.

Cox uses the lifelines default semiparametric fit: Efron ties, Breslow baseline,
no penalty, no strata, no weights, no clusters and model-based covariance
(`robust=False`). The log-hazard coefficient, SE, normal Wald 95% CI, p and HR
come from the fitter's summary. There is no substitute ties method or optimizer
seed. Exposure and covariate coding/reference levels come from `ModelTermSpec`;
the primary product supports one exposure coefficient, not a multi-contrast
categorical exposure.

Only complete cases of time, event and compiled model terms are fitted; no
imputation. Time must be finite and >0, and event codes must match the declaration.
Follow-up strictly beyond the declared numeric horizon is clipped to the horizon
and censored; an event exactly at the horizon remains an event. Code 0 means
censored and `event_value` means event. Time origin, censoring description and
population must be supplied; the input must already implement those definitions.
The scripts do not derive them from raw event records.

The Schoenfeld diagnostic uses the host's KM time transform and global Bonferroni
p value. Exposure OR global p below the declared alpha triggers the declared PH
policy; a violated assumption never grants paper authorization. Discrete exposure
groups get the host's log-rank comparison on the same complete-case horizon cohort;
continuous exposures are not implicitly grouped. Declared competing events use
the same cause-specific Cox, Aalen–Johansen CIF, Gray and Fine–Gray owners, requiring
local R with package survival for Fine–Gray; missing dependencies fail closed.
No competing-risk substitute is used by the example.

Outputs: unrounded `survival_effect_estimates.csv`, `survival_ph_diagnostic.csv`,
log-rank (or declared competing-risk) tables, the host receipt, prepared cohort,
step summary and an HR forest plot. The plot uses the reported HR/CI without a
refit; it does not claim the governed pipeline's full publication figure bundle.

Dependencies: easyicu, numpy, pandas, pyarrow, pydantic, scipy, lifelines, matplotlib.
Competing-risk mode additionally requires local R and R package survival.
The installed EasyICU package also supplies `skill_packages.reference_io`.
