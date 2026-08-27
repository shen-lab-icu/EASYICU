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
   speed penalty, and how to restore the fastest mode. A measured batch profile
   overrides the historical three-scan heuristic when real evidence proves that
   larger batches cross the memory contract. The selected batch is the largest
   successful measured size, not an arbitrary small default.
5. If memory is sufficient, no cleanup warning is shown.
6. An explicit user `batch_size` remains authoritative.
7. A measured light-module profile never authorises an unmeasured module. Mixed
   or unmeasured MIMIC-III, MIMIC-IV, and AUMC requests retain the conservative
   24 GiB full-cohort guard until their own measurements are recorded.

The machine-readable owner is `easyicu.api.extraction.plan_extraction_resources`.
Its stable reason codes are:

- `measured_profile_fast_path`
- `measured_profile_fastest_safe_batch`
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

## eICU full-cohort measurements under the 8 GiB contract

Scope: 200,859 ICU stays, one module per isolated process, two DuckDB threads,
2 GiB DuckDB memory limit, native-v2 output and temporary spill on external
storage. The one-shot admission threshold is measured process-tree RSS times
1.10. The hard profiling stop was 7,447 MiB, which is the largest observed RSS
that can retain 10% headroom inside 8,192 MiB currently available.

Fourteen modules completed one-shot:

| Module | One-shot time | Peak RSS | Automatic threshold |
|---|---:|---:|---:|
| demographics | 3.0 s | 1,194.6 MiB | 1.28 GiB |
| outcome | 2.6 s | 1,114.3 MiB | 1.20 GiB |
| blood_gas | 7.3 s | 1,104.6 MiB | 1.19 GiB |
| hematology | 18.5 s | 3,024.5 MiB | 3.25 GiB |
| chemistry | 38.1 s | 5,381.2 MiB | 5.78 GiB |
| vasopressors | 21.0 s | 5,517.3 MiB | 5.93 GiB |
| ventilator | 105.4 s | 7,435.9 MiB | 7.99 GiB |
| vitals | 158.1 s | 5,362.4 MiB | 5.76 GiB |
| renal | 438.1 s | 6,354.4 MiB | 6.83 GiB |
| medications | 175.9 s | 7,320.6 MiB | 7.86 GiB |
| neurological | 88.8 s | 5,073.9 MiB | 5.45 GiB |
| sepsis_shared | 11.8 s | 4,977.7 MiB | 5.35 GiB |
| sofa2_score | 558.1 s | 6,926.6 MiB | 7.44 GiB |
| sepsis3_sofa2 | 372.8 s | 6,268.4 MiB | 6.73 GiB |

Five modules crossed the one-shot contract and therefore use the fastest
successful measured batch instead:

| Module | One-shot evidence | Fastest verified batch | Batched time | Batched peak RSS |
|---|---:|---:|---:|---:|
| respiratory | stopped at 7,576.3 MiB | 50,000 stays (5 batches) | 251.7 s | 6,252.8 MiB |
| circulatory | stopped at 7,941.3 MiB | 50,000 stays (5 batches) | 494.7 s | 6,172.8 MiB |
| other_scores | stopped at 7,797.5 MiB | 67,000 stays (3 batches) | 436.3 s | 6,512.3 MiB |
| sofa1_score | stopped at 7,631.4 MiB | 67,000 stays (3 batches) | 512.5 s | 6,316.5 MiB |
| sepsis3_sofa1 | stopped at 7,475.6 MiB | 67,000 stays (3 batches) | 586.9 s | 6,294.5 MiB |

At 8 GiB currently available, selecting any subset of the 14 one-shot modules
runs each selected module one-shot. Selecting `respiratory` or `circulatory`
uses 50,000-stay batches; selecting any of the other three batch-only modules
uses 67,000-stay batches. A full 19-module request uses the strictest measured
50,000-stay batch. These measured batch paths show no cleanup warning because
8 GiB already fits their fastest verified peak plus headroom. A warning appears
only when currently available memory is below the relevant measured batch
threshold.

The physical-layout A/B found that contiguous 25,000-stay respiratory batches
took 234.2 seconds at 6,061.4 MiB versus 251.7 seconds at 6,252.8 MiB for the
production interleaved 50,000-stay path: only 6.9% faster despite more effective
Parquet pruning. Increasing the SOFA resolver cache from 512 MiB to 1 GiB was
safe and reduced the observed run from the production-default 512.5 seconds to
490.6 seconds (4.3%); 1.5 GiB showed no additional first-batch reuse and was
stopped. The product does not automatically enable the 1 GiB candidate. These
are optimisation diagnostics, not a new semantic output contract.

## Evidence and limits

- Eight-module benchmark:
  `/Volumes/外置硬盘/tmp/easyicu-light-native-all-c35zNOsy/benchmark_result.json`
- Exact-clean ventilator one-shot/stream invariance:
  `/Volumes/外置硬盘/tmp/easyicu-ventilator-invariance-fix-59f775e/verification_result.json`
- eICU 19-module process-tree receipts and A/B outputs:
  `/Volumes/外置硬盘/tmp/easyicu-6db-resource-profile-30f5228-ak9lbI/eicu/`
- Persistent eICU audit and limitations:
  `task_logs/20260827_eicu_19_module_resource_standard.md`

These measurements support resource selection only for the listed MIMIC-IV and
eICU modules and cohort ceilings. They do not establish a safe threshold for
the remaining MIMIC-IV modules or the other four databases. Add a new production
profile only after a clean full-cohort run records commit identity, cohort size,
elapsed time, process-tree peak RSS, output validity, and partition-invariance
evidence where batching can affect semantics.
