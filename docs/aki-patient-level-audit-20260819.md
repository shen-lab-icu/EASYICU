# AKI Patient-Level Audit — 2026-08-19

This audit uses deidentified ICU stay identifiers from the full six-database,
19-module candidate
`full6_native_v2_kdigo_7d_baseline_9913f31c_20260819`. Its renal module was
re-read from all six raw databases at clean commit `9913f31c`; the other
modules were byte-reused from the validated parent candidate.
Thresholds and official implementation references are documented in
[`aki-source-contract.md`](aki-source-contract.md).

## Positive examples

| Database | Stay | Weight (kg) | Creatinine trigger hour | Creatinine (current / 48 h low / 7 d low, mg/dL) | Creatinine stage | UO trigger hour | UO rate (6 h / 12 h / 24 h, mL/kg/h) | UO stage |
|---|---:|---:|---:|---|---:|---:|---|---:|
| AUMC | 22 | 85.0 | 2 | 5.802 / 4.275 / 4.275 | 3 | 13 | 0.052 / 0.075 / NA | 2 |
| eICU | 142173 | 122.7 | 202 | 0.950 / 0.730 / 0.460 | 2 | 180 | 0.472 / 0.256 / 0.295 | 3 |
| HiRID | 107 | 70.0 | 71 | 3.020 / 0.950 / 0.950 | 3 | 59 | 0.168 / 0.306 / 0.282 | 3 |
| MIMIC-III | 200053 | 121.8 | 32 | 4.100 / 1.800 / 1.300 | 3 | 42 | 0.102 / 0.162 / 0.279 | 3 |
| MIMIC-IV | 30003306 | 79.0 | 39 | 4.200 / 2.600 / 2.400 | 3 | 23 | 0.076 / 0.066 / 0.087 | 3 |
| SICdb | 100340 | 55.0 | 176 | 1.200 / 0.400 / 0.400 | 3 | 49 | 0.390 / 0.469 / 1.222 | 2 |

Each component stage matches the KDIGO threshold, and the final stage equals
the maximum available creatinine, urine-output, and RRT component stage.

## Complete negative examples

Stays AUMC 1, eICU 141415, HiRID 4, MIMIC-III 200003, MIMIC-IV 30000213,
and SICdb 100097 each contain a row where all three component ascertainments
are negative, all component stages are 0, `aki` is false, and
`aki_ascertainment` is `negative_complete`.

## Whole-export rule recomputation

Across the six refreshed renal exports, 2,477,387 creatinine-stage rows with a
published current creatinine were independently recomputed from the exported
48-hour and 7-day minima. Mismatches were zero in every database. Every renal
file contained the five new baseline-provenance columns, had zero duplicate
`(stay_id, charttime)` keys, and retained the public pre-ICU boundary (AUMC,
eICU, MIMIC-III, MIMIC-IV and SICdb: -24 h; HiRID: -1 h).

## Seven-day baseline correction

| Database | First non-negative creatinine rows | Assessable before | Assessable after | Baseline source after |
|---|---:|---:|---:|---|
| AUMC | 22,496 | 11,005 (48.92%) | 19,222 (85.45%) | 19,222 observed pre-ICU; 3,272 unavailable |
| HiRID | 29,182 | 74 (0.25%) | 74 (0.25%) | 74 observed at -1 h; 29,108 unavailable |

The AUMC gain is therefore attributable to observed `-168` to `-24 h`
measurements, not imputation. HiRID did not gain fabricated admission
baselines. For example, AUMC stay 166 changed from indeterminate to creatinine
stage 1 at 0 h (current 1.199, prior 7-day low 0.826 mg/dL). HiRID stay 107
remained unknown at its first creatinine, then used observed ICU history: at
48 h, current 1.753 versus prior low 0.950 mg/dL produced creatinine stage 1;
its urine-output component independently reached stage 3 at 59 h.
