# Muse/Spark 14-item code audit (2026-09-20)

## Release decision

Eight findings are correctness or fail-closed defects and are fixed in the
current checkout. One finding is a source-limited safety gap and is fixed to
the strongest bound HiRID can support. Three AKI findings describe deliberate
source-native phenotypes and must not be used as the cross-database KDIGO
endpoint. The hospital-to-ICU finding mixes an intentional ricu-compatible
assignment rule with a real provenance-receipt gap. The outcomes-test claim is
overstated because synthetic outcome regressions already run in CI.

The thesis-wide harmonized renal endpoint remains `aki_stage_reference`.
`aki_stage_source_native` and its components are sensitivity/provenance fields;
their definitions differ by database and are not poolable.

## Finding disposition

| # | Finding | Verdict | Action / release consequence |
|---|---|---|---|
| 1 | Plain-DataFrame `change_interval` grouped only by time | Confirmed | Fixed: canonical or explicit patient/stay IDs are grouping keys and cannot be averaged. Legacy callers now pass the resolved ID explicitly. |
| 2 | SIC native urine stage divides by 70 kg | Correct observation, wrong target | Retained in the frozen SIC source-native profile. The harmonized reference profile uses the uniform reference implementation and real weight evidence. Never pool the native field. |
| 3 | MIMIC-III native urine windows use 5/11/23 h | Correct observation, wrong target | Retained to reproduce the source-native implementation. It does not define `aki_stage_reference`. |
| 4 | AUMC native output is stage-3-like binary | Correct observation, wrong target | Registry already declares `CASE_LEVEL_STAGE3_LIKE_BINARY`. It is not a 0--3 KDIGO stage and is not poolable. |
| 5 | Fahrenheit conversion differed between DuckDB and Python | Confirmed | Fixed: blank-unit rows from Fahrenheit-only source items are converted in the Python fallback too. |
| 6 | Two WinTbl expansion routes used different endpoints | Confirmed | Fixed: both routes compute the endpoint from raw start plus duration before grid placement, matching the documented ricu path. Medication `dex` is therefore added to the v6 refresh scope via the `medications` module. |
| 7 | HiRID numeric time had no outlier isolation | Confirmed, source-limited | Fixed: enforce the requested pre-ICU allowance and a 366-day upper sanity bound. HiRID has no declared discharge timestamp, so an exact episode upper bound cannot be inferred. |
| 8 | SIC used only a 366-day coarse bound | Confirmed | Fixed: enforce `TimeOfStay - ICUOffset` plus the shared 24 h post-discharge allowance and the requested pre-ICU lookback. |
| 9 | `load_blood_gas` swallowed errors and guessed merge keys | Confirmed | Fixed: call the canonical multi-concept loader once and propagate extraction failures. |
| 10 | `load_demographics` dropped interval/window arguments | Confirmed | Fixed: both parameters are forwarded. |
| 11 | AUMC single `numericitems.parquet` was not detected | Confirmed | Fixed: file and directory layouts are recognized. |
| 12 | Data-path resolution used lexical versions, substring aliases, and silent fallback | Confirmed | Fixed: numeric-aware version ordering, exact normalized aliases, and `FileNotFoundError` on unresolved roots. |
| 13 | Hospital rows are inner-joined and forward-assigned without a receipt | Mixed | Excluding admissions with no ICU stay is correct for ICU-stay exports; forward assignment by ICU outtime with `rollends=TRUE` intentionally follows ricu, and later episode bounds quarantine out-of-window rows. Persistent assignment-count receipts remain a governance improvement, not evidence that current ICU rows are misassigned. |
| 14 | `test_outcomes.py` is real-data-only, so CI tests nothing | Overstated | That file is real-data-gated, but synthetic mortality, censoring, alias, malformed-date, and fail-closed tests run in `test_data_correctness_regressions.py`, `test_death_inhospital_callback.py`, and `test_concept_outcome_fail_closed.py`. |

## Verification contract

The focused regression set must cover patient-isolated interval aggregation,
blank-unit Fahrenheit conversion, raw-start window endpoints, SIC episode
bounds, HiRID sanity bounds, convenience-loader failure propagation, numeric
version ordering, exact aliases, and AUMC single-file detection. The complete
core suite must pass before the v6 benchmark/rebuild resumes.

For v6, do not change the three source-native AKI algorithms. Refresh and
validate `renal` so every database publishes a usable `aki_stage_reference`,
then bind downstream studies only to that reference field. Any source-native
comparison must retain its profile ID and output-kind metadata.
