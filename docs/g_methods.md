# Longitudinal MSM and point-intervention g-formula

`causal_emulation.g_methods` is a deterministic, analysis-only method component.
The standard executor registry routes it through the existing time-varying runner.
The target-trial methods owner delegates its MSM fit to `methods.msm`; point
standardization uses `methods.gformula.gformula_ate`.

## Reviewed specification

Classic and progressive plans carry the same closed `GMethodSpec` as
`method` / `custom_method = g_methods(<JSON>)`. Use `g_method_serialization`
from `contracts.time_varying_exposure`, rather than constructing the string.
Bind exactly one digest-verified `cohort:person_period` input, its declared
raw columns, and these products, in order:

- `table:g_method_estimates`
- `table:g_method_weight_diagnostics`
- `log:g_method_receipt`
- `report:g_method_checklist`

The specification names the selected database, person identifier, period,
binary treatment and event columns, baseline stabilizers, confounder history,
treatment and confounder lags, interval count and width in hours, ICU-admission
time zero, censoring policy, positivity window, truncation quantiles and seed.
Consistency, exchangeability and correct specification must be explicitly
acknowledged. Lags refer to the declared grid, never to row offsets across people.

For `longitudinal_msm`, treatment lags must be positive, confounder lags include
zero, and `n_bootstrap` is null. The outcome design is current treatment,
declared treatment lags, baseline stabilizers and interval intercepts. Evolving
confounders enter the weight denominator, not the outcome adjustment set.
The estimand is the per-interval treatment odds ratio conditional on the declared
history and baseline stabilizers; it is not an RR or a Cox hazard ratio.

For `point_gformula`, use one interval, no treatment lags, confounder lags `[0]`,
complete follow-up, no truncation, and at least 20 bootstrap replicates. Logistic
standardization reports treated/untreated risks, their RD and RR, percentile
bootstrap intervals, and attempted/successful replicate counts. The existing
linear g-formula API remains available to its callers without claiming an RR.

## Input-owner boundary

`prepare_g_method_execution_input` in `acquisition.time_varying_execution_input`
validates an **already constructed** person-period table and preserves its
source receipt in Parquet metadata (`easyicu.g_methods_source`). It does not
extract histories, turn baseline rows into trajectories, fill missing values,
or reinterpret the existing running-maximum exposure model as a treatment.

The target-trial or time-varying input owner supplies the receipt: schema
`easyicu.g_methods_source/1`, producer `target_trial` or `time_varying`, selected
database, `structure: person_period`, `identity_unit: person`, the identity
column, available columns, history columns, pretreatment columns, interval
width/time zero, upstream source SHA-256 and reviewed specification SHA-256.
The host context carries this same receipt in `cohort.provenance` under the
metadata key. It is source metadata, never a Planner-authored declaration.

Each of the six databases must supply its own receipt and actual history.
Dictionary coverage alone is insufficient. Baseline-only target-trial exports
and exposure-only running-maximum panels are refused with typed reasons until
their owners provide the required treatment/confounder histories. No real-data
availability or acceptance is asserted by these synthetic tests.

Rows start at period zero, are consecutive and unique per person, and end at
the first event, declared censoring or administrative horizon. All covariates
are numeric and complete; baseline stabilizers are constant within a person.
One treatment stream per person is supported. Re-entry or multiple episodes
sharing a person and period require an upstream design change.

`stabilized_ipcw` requires a binary end-of-interval censoring flag: one for an
event-free dropout before the horizon, zero for events and administrative end.
Censoring weights affect subsequent intervals only. An uncensored interval
with no dropout contributes one. Undeclared dropout is refused.

## Diagnostics and claim ceiling

The MSM calls the sequential IPTW kernel on each risk set, accumulating weights
along each person's observed path. The numerator includes treatment history and
baseline stabilizers; the denominator additionally includes the declared
confounder history. IPCW also conditions on current treatment. The positivity
screen is fail closed, with the failing interval on the typed exception.

Truncation clips the **combined cumulative** treatment/censoring weight at its
quantiles within each interval. The table records bounds, numbers clipped,
raw and analysis ESS, weight distribution and propensity ranges, including
censoring propensities. Internal person weights are not published.

Weighted pooled-logistic intervals use a person-cluster sandwich and treat
fitted weights as fixed. Full propensity-estimation uncertainty is not estimated.
All products disclose observational identification assumptions, unmeasured
confounding, model dependence and the `analysis_only` ceiling. The workflow hub
shows an optional history-intake/MSM branch; its metadata is not executable.
Multi-interval g-formula, a weighted Cox MSM and dynamic regime optimization are
not offered. STROBE items 9, 12a and 16 recognize the actual products; no
sensitivity-analysis credit is granted merely for running an MSM.
