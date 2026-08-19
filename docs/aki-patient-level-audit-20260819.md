# AKI Patient-Level Audit — 2026-08-19

This audit uses deidentified ICU stay identifiers from the full six-database,
19-module candidate `full6_native_v2_aki_sic_timeaxis_82abec4b_20260819`.
Thresholds and official implementation references are documented in
[`aki-source-contract.md`](aki-source-contract.md).

## Positive examples

| Database | Stay | Weight (kg) | Creatinine trigger (current / 48 h low / 7 d low, mg/dL) | Creatinine stage | UO trigger (6 h / 12 h / 24 h, mL/kg/h) | UO stage |
|---|---:|---:|---|---:|---|---:|
| AUMC | 22 | 85.0 | 5.360 / 4.275 / 4.275 | 3 | 0.052 / 0.075 / NA | 2 |
| eICU | 142173 | 122.7 | 0.950 / 0.730 / 0.460 | 2 | 0.472 / 0.256 / 0.295 | 3 |
| HiRID | 107 | 70.0 | 3.020 / 0.950 / 0.950 | 3 | 0.168 / 0.306 / 0.282 | 3 |
| MIMIC-III | 200053 | 121.8 | 4.100 / 1.800 / 1.300 | 3 | 0.102 / 0.162 / 0.279 | 3 |
| MIMIC-IV | 30003306 | 79.0 | 4.200 / 2.600 / 2.400 | 3 | 0.076 / 0.066 / 0.087 | 3 |
| SICdb | 100340 | 55.0 | 1.200 / 0.400 / 0.400 | 3 | 0.390 / 0.469 / 1.222 | 2 |

Each component stage matches the KDIGO threshold, and the final stage equals
the maximum available creatinine, urine-output, and RRT component stage.

## Complete negative examples

Stays AUMC 1, eICU 141415, HiRID 4, MIMIC-III 200003, MIMIC-IV 30000213,
and SICdb 100097 each contain a row where all three component ascertainments
are negative, all component stages are 0, `aki` is false, and
`aki_ascertainment` is `negative_complete`.

## Whole-export rule recomputation

Across the six renal exports, 2,121,721 creatinine-stage rows, 12,772,292
urine-stage rows, 14,667,524 final-stage rows, and 504,706 complete-negative
rows were independently recomputed from the exported inputs. Mismatches were
zero in every database and every check.
