# Caveat flags and the sentences they require

`caveat_flags.json` is written by `export_all`; `caveat_sentences()` turns the
raised flags into the sentences below. A report that quotes this skill's numbers
must carry the sentence for every raised flag. The wording may be translated but
not weakened; the numbers in it come from `key_metrics.csv`.

| Flag | Raised when | Required sentence (English template) |
|---|---|---|
| `epv_below_minimum` | `epv < epv_minimum` | "The primary model has {epv} events per parameter, below the declared minimum of {epv_minimum}; the adjusted odds ratios are potentially overfitted and their intervals should be read as unstable." |
| `separation_detected` | kernel reports (quasi-)separation | "The primary logistic fit showed (quasi-)separation; the affected odds ratio and its interval are not reliable estimates." |
| `complete_case_warning` | share of exposure-known rows dropped for missing covariates > `complete_case_warning_share` | "{share} of landmark rows with a known exposure were excluded from the primary model because of missing covariates (complete-case analysis…); the estimate is subject to selection bias and no imputation was performed." |
| `unknown_exposure_share_high` | unknown share of landmark rows ≥ `unknown_exposure_warning_share` | "{share} of the landmark population had a non-evaluable exposure (reported as 'unknown'…); these rows are described but not modelled, so the adjusted estimates apply to the evaluable subset only." |
| `small_reference_group` | reference level n < `small_reference_group_n` | "The reference level {ref} contains only {n} rows…; contrasts against it are unstable." |
| `sparse_exposure_levels` | any modelled level with events < `sparse_level_events` | "Exposure level(s) {levels} have fewer than {k} events; their odds ratios are imprecise." |
| `empty_exposure_levels` | a declared level has no rows | "Declared exposure level(s) {levels} were not observed…; the corresponding contrasts are absent, not zero." |
| `sensitivity_direction_consistent = false` | any fitted refit reverses the primary direction | "At least one prespecified sensitivity refit reversed the direction of the primary contrast; see robustness_summary.csv before interpreting the primary estimate." |
| `failed_sensitivity_variants` non-empty | a refit could not be estimated | "Sensitivity refit(s) that could not be estimated and remain in the denominator: {ids}." |
| `functional_form_not_estimable` non-empty | a spline refit failed | "Functional-form check(s) not estimable: {covariates}." |
| always | — | "All results are observational associations in a fixed {landmark} h landmark population with claim ceiling 'analysis_only': no causal effect, no transportability beyond the source, and no novelty claim is made." |

Flags are computed once, in `run_analysis`, from the same tables the report
prints. They are not re-derived by the writer, and a downstream reader that
needs a new caveat adds a flag here rather than a sentence there.
