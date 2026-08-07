# EasyICU research-agent analysis-method suite

_Generated from `easyicu.research_agent.planning.analysis_method_suite`. Do not edit by hand — edit the registry and regenerate._

`capability_registry` declares the ONE primary estimand per family. This matrix declares the **full standard method suite** a reviewer expects for each family, and how each method is produced:

- **Tier** — `primary` (the reported estimand) · `standard_supporting` (diagnostics/robustness a reviewer routinely expects) · `exploratory` (optional deeper add-on, labelled as such) · `planned` (recognised, not yet implemented — fails closed, never approximated).
- **Implementation** — deterministic (source-data-backed runner/panel) · LLM-coded (value-provenance verified) · planned.

A `planned` method carries no runner. It must fail closed if requested as a primary estimand — e.g. competing-risks CIF is never answered with a Cox HR.

## Survival / time-to-event

| Method | Tier | Implementation | Produces | Runner |
| --- | --- | --- | --- | --- |
| Cox proportional-hazards hazard ratio | primary | deterministic ✅ | host-bound Cox result + PH diagnostic + digest receipt | `survival_primary_cox` |
| Kaplan-Meier curves + log-rank by exposure | standard_supporting | deterministic ✅ | kaplan_meier curve data + log-rank; KM panel | `time_to_event` |
| Proportional-hazards check (Schoenfeld residuals / PH test) | standard_supporting | deterministic ✅ | schoenfeld_test.csv (covariate, chi2, p) + schoenfeld_plot (diagnostics panel slot) | `survival_primary_cox` |
| Subgroup hazard ratios / interaction forest | standard_supporting | LLM-coded ⚠️ | subgroup HR forest | — |
| Restricted mean survival time (RMST) | standard_supporting | LLM-coded ⚠️ | rmst.csv (group, rmst, ci) + difference | — |
| Competing-risks cumulative incidence (Fine-Gray / CIF) | planned | planned ⛔ | cause-specific CIF | — |
| Time-varying coefficients / landmark survival | planned | planned ⛔ | time-varying HR / landmark curves | — |

## Causal inference / target-trial emulation

| Method | Tier | Implementation | Produces | Runner |
| --- | --- | --- | --- | --- |
| Stabilised-IPTW marginal odds ratio | primary | LLM-coded ⚠️ | agent-declared causal effect table + target-trial/identification protocol | `causal_emulation` |
| Covariate balance (SMD love plot, before vs after weighting) | standard_supporting | deterministic ✅ | covariate_balance.csv (covariate, smd_before, smd_after) + love-plot panel | `causal_emulation` |
| Positivity / overlap (propensity distribution + trimming) | standard_supporting | LLM-coded ⚠️ | propensity distribution + trimming report | — |
| E-value (sensitivity to unmeasured confounding) | standard_supporting | deterministic ✅ | e_values.csv (term, odds_ratio, ci, e_value, e_value_lower_bound) | `sensitivity` |
| Negative-control outcome / exposure | standard_supporting | LLM-coded ⚠️ | negative-control contrast | — |
| Doubly-robust / matching sensitivity (AIPW, TMLE, PS matching) | exploratory | planned ⛔ | alternative-estimator effect table | — |
| g-formula / marginal structural models (time-varying treatment) | planned | planned ⛔ | g-computation / MSM effect | — |

## Association (general + graded ordinal / dose-response)

| Method | Tier | Implementation | Produces | Runner |
| --- | --- | --- | --- | --- |
| Ordinal dose-response (OR per +1 stage, per-stage forest, monotonicity) | primary | LLM-coded ⚠️ | agent-declared ordered trend + dose_response.csv; deterministic forest | — |
| Adjusted association (logistic / linear) | primary | deterministic ✅ | host-bound adjusted effect + typed model/coefficient contract + deterministic forest | `adjusted_association_estimates` |
| Multiple adjustment sets (crude / minimal / full) | standard_supporting | LLM-coded ⚠️ | nested-model effect table | — |
| Effect modification / interaction + subgroup forest | standard_supporting | LLM-coded ⚠️ | interaction test + subgroup forest | — |
| Missing-data audit + complete-case vs imputation sensitivity | standard_supporting | deterministic ✅ | missingness_summary.csv + sensitivity contrast | `missing_data` |
| Multiple-testing correction (FDR / Bonferroni) | standard_supporting | deterministic ✅ | adjusted p-value table | `multiple_testing` |
| Restricted cubic spline dose-response (continuous exposure) | standard_supporting | planned ⛔ | spline_dose_response.csv (x, log_or, ci) + spline panel | — |
| E-value (sensitivity to unmeasured confounding) | standard_supporting | deterministic ✅ | e_values.csv | `sensitivity` |
| Robustness panel (alternative specifications) | exploratory | deterministic ✅ | robustness_panel.csv | `robustness_sensitivity` |
| Mediation / quantitative bias analysis | planned | planned ⛔ | mediation / bias-analysis table | — |

## Prediction / risk modelling

| Method | Tier | Implementation | Produces | Runner |
| --- | --- | --- | --- | --- |
| Discrimination + calibration (AUROC + calibration curve) | primary | LLM-coded ⚠️ | AUROC + calibration curve + ROC panels (figure deterministic) | `prediction` |
| Calibration slope/intercept + Brier score | standard_supporting | LLM-coded ⚠️ | calibration metrics row; Brier annotated on the calibration panel when present | — |
| DeLong CI / test on AUROC | standard_supporting | LLM-coded ⚠️ | auroc_delong.csv (auroc, ci_low, ci_high [, comparison p]) | — |
| Decision-curve analysis / net benefit (DCA) | standard_supporting | LLM-coded ⚠️ | decision_curve.csv (threshold, net_benefit, net_benefit_all) | — |
| Sensitivity/specificity/PPV/NPV at clinical thresholds | standard_supporting | LLM-coded ⚠️ | threshold_metrics.csv; confusion matrix at chosen thresholds | — |
| Feature attribution (SHAP / permutation importance) | standard_supporting | LLM-coded ⚠️ | feature_importance.csv; beeswarm/bar summary | — |
| Subgroup / fairness performance | standard_supporting | deterministic ✅ | per-subgroup performance table | `fairness` |
| Internal validation (bootstrap optimism / cross-validation) | standard_supporting | LLM-coded ⚠️ | internal-validation performance table | — |
| Conformal prediction intervals | exploratory | LLM-coded ⚠️ | conformal coverage table | — |
| External / cross-database validation | planned | planned ⛔ | external-cohort performance + recalibration | — |
| Net reclassification / IDI vs a baseline model | planned | planned ⛔ | NRI / IDI table | — |
| Dynamic / landmark prediction (time-updated risk) | planned | planned ⛔ | landmark performance over time | — |

## Phenotyping / clustering (cross-sectional + longitudinal trajectory)

| Method | Tier | Implementation | Produces | Runner |
| --- | --- | --- | --- | --- |
| Cluster solution + stability (cross-sectional subphenotypes) | primary | LLM-coded ⚠️ | cluster assignments + stability; heatmap + stability + outcome-by-cluster figure (deterministic) | `phenotyping` |
| Number-of-clusters selection (silhouette / gap / BIC) | standard_supporting | LLM-coded ⚠️ | k-selection criterion curve | — |
| Stability / reproducibility (bootstrap, consensus, adjusted Rand) | standard_supporting | LLM-coded ⚠️ | stability report | — |
| Typed trajectory-cluster stability refits | standard_supporting | deterministic ✅ | digest-bound stability specification, refit ledger, adjusted-Rand table, aligned assignments, and freeze/report decision | `trajectory_cluster_stability` |
| Cluster sizes + degenerate-cluster flag | standard_supporting | deterministic ✅ | cluster_sizes.csv; figure panel | `phenotyping` |
| Outcome-by-cluster descriptive comparison | standard_supporting | deterministic ✅ | outcome_by_cluster.csv; figure panel | `phenotyping` |
| Trajectory-feature clustering | primary | LLM-coded ⚠️ | agent-declared feature representation + cluster assignments + silhouette/stability/size QC + outcome-by-trajectory | `phenotyping` |
| LCGA / group-based trajectory modelling (GBTM) | planned | planned ⛔ | latent-class trajectory model | — |
| Mixed-effects / growth-mixture trajectory models | planned | planned ⛔ | mixed-effects / GMM trajectory model | — |
| DTW / time-series distance clustering | planned | planned ⛔ | DTW distance clustering | — |
| Landmark trajectory prediction | planned | planned ⛔ | landmark trajectory-prediction model | — |

## Descriptive / measurement audit

| Method | Tier | Implementation | Produces | Runner |
| --- | --- | --- | --- | --- |
| Descriptive summaries / Table 1 / measurement-process audit | primary | LLM-coded ⚠️ | descriptive summary table (bound); base figure deterministic | — |
| Baseline-characteristics table (Table 1) | standard_supporting | LLM-coded ⚠️ | table_one.csv | — |
| Missing-data / completeness audit | standard_supporting | deterministic ✅ | missingness_summary.csv | `missing_data` |

## Planned methods (declared, not implemented — fail closed)

| Family | Method | Why it matters |
| --- | --- | --- |
| time_to_event | Competing-risks cumulative incidence (Fine-Gray / CIF) | Cause-specific cumulative incidence when a competing event (e.g. death) precludes the outcome. |
| time_to_event | Time-varying coefficients / landmark survival | Relax the constant-HR assumption over follow-up. |
| causal_emulation | Doubly-robust / matching sensitivity (AIPW, TMLE, PS matching) | Re-estimate the effect under an alternative identification to test IPTW robustness. |
| causal_emulation | g-formula / marginal structural models (time-varying treatment) | Handle time-varying treatment and confounding affected by prior treatment. |
| association | Restricted cubic spline dose-response (continuous exposure) | Non-linear exposure-response without imposing linearity or arbitrary cut-points. |
| association | Mediation / quantitative bias analysis | Decompose direct/indirect effects or quantify plausible bias. |
| prediction | External / cross-database validation | Transportability of discrimination + calibration to a second ICU database. |
| prediction | Net reclassification / IDI vs a baseline model | Incremental value of new predictors over an established score. |
| prediction | Dynamic / landmark prediction (time-updated risk) | Re-estimated risk from time-updated features (landmarking). |
| phenotyping | LCGA / group-based trajectory modelling (GBTM) | Latent-class growth / group-based trajectory model — a model-based longitudinal method. |
| phenotyping | Mixed-effects / growth-mixture trajectory models | Random-effects longitudinal models with latent classes. |
| phenotyping | DTW / time-series distance clustering | Cluster raw trajectories under a shape-aware (dynamic time warping) distance. |
| phenotyping | Landmark trajectory prediction | Predict outcome from trajectory shape up to a landmark time. |
