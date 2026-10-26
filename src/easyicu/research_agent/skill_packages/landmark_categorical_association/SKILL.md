---
name: "fixed-landmark-association-study"
description: "A complete landmark study: compare outcomes across exposure groups among patients still in hospital at a set time point, without immortal-time bias."
category: "Association"
---

# Fixed-landmark association study

## Purpose

A complete landmark study: compare outcomes across exposure groups among patients still in hospital at a set time point, without immortal-time bias.

## When to use

Use this for questions such as:
Among patients still in hospital after the first ICU day, is early lactate group associated with later hospital death?

## When not to use

- Exposure keeps changing after the selected time point → Time-varying exposure survival study
- The question asks for a treatment effect → Target-trial emulation

## How it works

Start the comparison among patients alive and still observed at the same landmark. Use only measurements obtained before that point to classify exposure. Report unknown exposure separately, and interpret the estimate only for patients eligible at the landmark.

## Methods

EasyICU computes every estimate itself; this package's reference scripts call the same code.

### Estimators
- Eligibility at the landmark: alive and still observed at the landmark; exposure is classified only from measurements before it, and follow-up starts at the landmark.
- Primary model: logistic regression with the declared contrasts and covariates, cluster-robust standard errors by patient when repeated stays are declared dependent.
- Absolute risk per exposure level; unknown exposure is reported as its own state and excluded from the models. Ordered levels add an adjusted per-level trend and Cochran–Armitage / Jonckheere–Terpstra tests with Holm adjustment.

### Reading the results
- Estimates apply only to patients alive and observed at the landmark; changing the landmark changes the population.
- An odds ratio overstates the risk ratio when the outcome is common (roughly above 10 %, as ICU mortality often is); read it with the absolute risks and do not report it as a relative risk.
- Fewer than about 10 outcome events per model parameter makes the estimates unstable; the plan counts parameters, not variables.
- A large unknown-exposure share is itself a finding; it is flagged above 10 % by default.

### Methodological literature
- Anderson JR, Cain KC, Gelber RD. Analysis of survival by tumor response. J Clin Oncol. 1983;1:710-719.
- Dafni U. Landmark analysis at the 25-year landmark point. Circ Cardiovasc Qual Outcomes. 2011;4:363-371.
- Liang KY, Zeger SL. Longitudinal data analysis using generalized linear models. Biometrika. 1986;73:13-22.
- von Elm E, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement. Lancet. 2007;370:1453-1457.
- Benchimol EI, et al. The REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) statement. PLoS Med. 2015;12:e1001885.

## What the plan will propose for your review

The plan proposes time zero, landmark and eligibility, exposure window and levels, reference group, outcome horizon, covariates and repeated stays; you review them together before anything runs. Copilot resolves routine configuration and presents a complete plan.
Inputs: an approved cohort with patient/stay identifiers, the specified measurements, units and timestamps, and outcome/follow-up information when used.

## Steps and their checks

1. **Cohort characterization and Table 1**. Host checks: Counts used as denominators; Which measurements are available; Agreement between counts and frequencies.
   - Baseline-characteristics table (Table 1)
2. **Missingness and measurement audit**. Host checks: Counts used as denominators; Which measurements are available; Agreement between counts and frequencies.
   - Missing-data / completeness audit
   - Missing-data audit + complete-case vs imputation sensitivity
3. **Exposure and outcome distribution**. Host checks: Exposure and outcome groups account for all patients; Uncertainty calculation matches repeated observations; Interpretation stays descriptive.
4. **Fixed-landmark association study**. Host checks: Eligibility and observation at the landmark; Recorded variable coding; Fitted model matches the reviewed model; Agreement of estimates and uncertainty intervals.
   - Adjusted association (logistic / linear)
### Reference scripts (executable)
The reference implementation loads the cohort, runs the analysis, draws figures and exports a checked report. The existing completion messages include `✓ Analysis completed successfully!` and `=== Export Complete ===`; the latter appears only after export consistency checks pass.

## Outputs

- `table_one.csv`
- `cohort_flow.csv`
- `measurement_audit.csv`
- `absolute_risk.csv`
- `adjusted_association_estimates.csv`
- `key_metrics.csv`
- `caveat_flags.json`
- `report.md`
- `manifest.json`
These are expected files, not recorded results. Outputs are produced only after a successful reviewed run.

## Per-database notes

For the public-demo plan illustrated in examples/public_demo.md; the actual plan is reassessed. Exact means the requested definitions are mapped, not that the study has run.
- **MIMIC-IV — Exact**. Real-data acceptance not verified by the source profile.
- **eICU-CRD — Exact**. Real-data acceptance not verified by the source profile.
- **AmsterdamUMCdb — Substitution proposed**. death before hospital discharge → death within 72 hours after ICU discharge; Real-data acceptance not verified by the source profile.
- **HiRID — Substitution proposed**. death before hospital discharge → death before ICU discharge; Real-data acceptance not verified by the source profile.
- **MIMIC-III — Exact**. Real-data acceptance not verified by the source profile.
- **SICdb — Exact**. Real-data acceptance not verified by the source profile.
Any substitution changes the definition and requires review of a revised plan. Mapping does not establish observed coverage or adequate sample size.
These source facts do not verify local data access, landmark eligibility or whether a model can be fitted.

## Common stops

- **The analysis data differ from the reviewed copy.** Ask Copilot to refresh the data record and prepare the updated plan for review.
- **The supplied table does not contain the agreed columns.** Review the proposed extraction correction or revised variable list before rerunning.
- **The plan leaves part of your question unanswered.** Review the revised plan that includes that comparison or explicitly narrows the question.
- **The requested analysis is not supported for this plan.** Review the stated limitation and the proposed sibling workflow or data change before proceeding.
- **The record of the reviewed question cannot be read.** Ask Copilot to prepare a fresh complete plan for review.

## References

- STROBE 16: Adjusted association (logistic / linear).
- STROBE 12c: Missing-data audit + complete-case vs imputation sensitivity.
- STROBE 14a: Baseline-characteristics table (Table 1).
- STROBE 12c: Missing-data / completeness audit.
- RECORD: source and variable definitions must accompany the descriptive and association outputs; this package alone does not complete the checklist.
- STROBE 13 / RECORD 13.1: `cohort_flow.csv`; STROBE 16: `adjusted_association_estimates.csv`.

## Boundary

EasyICU owns data access, execution, evidence records and the limits on conclusions. Review the full plan before running. Do not change definitions or fill gaps silently; preserve failed checks and revise the plan. Public demos support learning, not clinical conclusions.
Results can enter a report only after the host's required reviews pass.
