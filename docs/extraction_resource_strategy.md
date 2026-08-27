# Extraction resource strategy

EasyICU automatically chooses the fastest evidence-supported extraction mode.
Patient batching is a fallback for insufficient memory, not the normal path.

## Stable user contract

1. EasyICU checks **currently available memory**, including an effective
   container limit when present. Total installed RAM is not used as a promise.
2. If every selected module has a full-cohort measurement and the largest
   measured process-tree peak plus 10% launch headroom fits, EasyICU runs the
   entire selected cohort in one shot.
3. Multiple modules run sequentially in isolated workers, so the requirement is
   the **maximum** selected-module threshold, not their sum.
4. If available memory is below the threshold, EasyICU falls back to patient
   batches and tells the user the available memory, required threshold, expected
   speed penalty, and how to restore the fastest mode. For a measured module it
   chooses the fewest supported partitions and never exceeds the three-scan
   extraction contract; a small threshold miss therefore becomes two large
   batches, not a long sequence of tiny batches.
5. If memory is sufficient, no cleanup warning is shown.
6. An explicit user `batch_size` remains authoritative.
7. A measured light-module profile never authorises an unmeasured module. Mixed
   or unmeasured MIMIC-III, MIMIC-IV, and AUMC requests retain the conservative
   24 GiB full-cohort guard until their own measurements are recorded.

The machine-readable owner is `easyicu.api.extraction.plan_extraction_resources`.
Its stable reason codes are:

- `measured_profile_fast_path`
- `measured_profile_insufficient_memory`
- `calibrated_fast_path`
- `unmeasured_profile_memory_guard`
- `explicit_batch_size`

## MIMIC-IV v3.1 full-cohort measurements

Scope: 94,458 ICU stays, one module per isolated process, native-v2 output on
external storage. The required available-memory threshold is the measured peak
process-tree RSS multiplied by 1.10. Times are extraction/package measurements,
not promises for different disks, CPUs, source layouts, or EasyICU revisions.

| Module | One-shot time | Measured peak RSS | Automatic one-shot threshold |
|---|---:|---:|---:|
| demographics | 2.4 s | 808.5 MiB | 0.87 GiB |
| outcome | 2.5 s | 1,344.3 MiB | 1.44 GiB |
| blood_gas | 5.4 s | 1,658.2 MiB | 1.78 GiB |
| vasopressors | 13.2 s | 2,060.8 MiB | 2.21 GiB |
| hematology | 27.3 s | 2,620.9 MiB | 2.82 GiB |
| chemistry | 60.3 s | 2,651.5 MiB | 2.85 GiB |
| ventilator | 78.7 s | 1,633.8 MiB | 1.76 GiB |
| vitals | 107.3 s | 3,544.4 MiB | 3.81 GiB |

Therefore, a user extracting only full-cohort MIMIC-IV `blood_gas` with 2 GiB
currently available should receive the one-shot fast path and no memory warning.

The eight-module production-shaped one-shot run took 359.7 seconds versus
522.4 seconds for the corresponding streamed run (1.45x faster), with a maximum
process-tree RSS of 3,544.4 MiB. The exact-clean ventilator verification at
commit `59f775e` took 78.7 seconds one-shot versus 197.5 seconds in fixed 5,000
stay batches; both published 963,266 rows and matched bidirectionally under
`EXCEPT ALL` (0/0).

## Evidence and limits

- Eight-module benchmark:
  `/Volumes/外置硬盘/tmp/easyicu-light-native-all-c35zNOsy/benchmark_result.json`
- Exact-clean ventilator one-shot/stream invariance:
  `/Volumes/外置硬盘/tmp/easyicu-ventilator-invariance-fix-59f775e/verification_result.json`

These measurements support resource selection only for the listed MIMIC-IV
modules and cohort size. They do not establish a safe one-shot threshold for
the remaining modules or other databases. Add a new production profile only
after a clean full-cohort run records commit identity, cohort size, elapsed time,
process-tree peak RSS, output validity, and partition-invariance evidence where
batching can affect semantics.
