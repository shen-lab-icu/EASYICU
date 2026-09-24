# Reporting checklist mapping (STROBE / RECORD → exported files)

Where each reporting item is answered by this skill's output, and what still
needs the enclosing study (source description, literature, interpretation).

| Item | STROBE / RECORD | Covered by | Still owed by the study |
|---|---|---|---|
| Study design | STROBE 4 | `report.md › Design`; `spec.json` | rationale for the landmark and level set |
| Setting, dates | STROBE 5; RECORD 6.1–6.3 | — | database, version, extraction receipts (host provenance) |
| Participants, eligibility | STROBE 6; RECORD 6.1 | `cohort_flow.csv`, `figure_cohort_flow` | upstream cohort criteria (age, admission type) |
| Variables | STROBE 7; RECORD 7.1 (codes/algorithms) | `spec.json`, `measurement_audit.csv` | exposure algorithm text and code digest (the host module that derives it), covariate derivation |
| Data sources / measurement | STROBE 8 | `measurement_audit.csv`, `figure_exposure_ascertainment` | validation of the algorithm against the source |
| Bias | STROBE 9 | landmark design (`report.md › Fixed limitations`), `caveat_flags.json` | unmeasured confounding discussion |
| Study size | STROBE 10 | `key_metrics.csv` (`n_landmark`, `n_fit`, `epv`) | — |
| Quantitative variables | STROBE 11 | `functional_form_sensitivity.csv` | choice of knots if changed |
| Statistical methods | STROBE 12 | `references/methods.md`, `adjusted_association_coefficients.csv` | — |
| Participants (flow) | STROBE 13; RECORD 13.1 | `cohort_flow.csv` with the exposure-known/unknown split | — |
| Descriptive data | STROBE 14 | `table_one.csv`, `secondary_outcome_summary.csv` | — |
| Outcome data | STROBE 15 | `absolute_risk.csv` (incl. unknown row) | — |
| Main results | STROBE 16 | `adjusted_association_estimates.csv`, `adjusted_trend.csv`, `key_metrics.csv` | — |
| Other analyses | STROBE 17 | `association_sensitivity_grid.csv`, `functional_form_sensitivity.csv`, `ordinal_trend_tests.csv`, `robustness_summary.csv` | — |
| Key results / limitations / interpretation | STROBE 18–20 | `report.md › Caveats`, `› Fixed limitations` | literature-anchored interpretation; comparator table |
| Generalisability | STROBE 21 | closing caveat sentence | — |
| Data access / cleaning | RECORD 12.1–12.3, 19.1 | `manifest.json` (SHA-256), `_kernel/` receipts | linkage and cleaning description |
| Funding, code availability | STROBE 22; RECORD 22.1 | package version in `manifest.json` | — |
