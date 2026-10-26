# Worked example: MIMIC-IV demo

Public source: MIMIC-IV demo. This is a proposed teaching plan, not a recorded run; no result values or successful execution are claimed.

Among patients still in hospital after the first ICU day, is early lactate group associated with later hospital death?

## Plan proposed for your review

Adults; ICU admission as time zero; a 24-hour landmark; highest lactate in hours 0–24 grouped below 2, 2 to below 4, and at least 4 mmol/L; lowest group as reference; death after the landmark before hospital discharge; age and sex adjustment; patient-clustered uncertainty for repeated stays; unknown exposure described and excluded from the model.

## Expected files

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

If the demo lacks the required observations, events or patient separation, the host records the limitation. Review a revised plan; do not fabricate outputs or treat a demo as validation on real data.
