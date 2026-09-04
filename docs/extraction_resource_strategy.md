# Extraction resource strategy

EasyICU automatically chooses the fastest evidence-supported extraction mode.
Patient batching is a fallback for insufficient memory, not the normal path.

## Stable user contract

1. EasyICU checks **currently available memory**, including an effective
   container limit when present. Total installed RAM is not used as a promise.
2. If every selected module has a full-cohort measurement and the largest
   measured process-tree peak plus 10% launch headroom fits, EasyICU runs the
   entire selected cohort in one shot.
3. Multiple modules run sequentially in isolated workers. Each module therefore
   follows its own measured strategy: a batch-only module must not force a
   measured one-shot module in the same request to repeat source scans. The
   aggregate request plan remains a conservative summary, not the execution
   batch applied to every module.
4. If available memory is below the threshold, EasyICU falls back to patient
   batches and tells the user the available memory, required threshold, expected
   speed penalty, and how to restore the fastest mode. A measured batch profile
   overrides the historical three-scan heuristic when real evidence proves that
   larger batches cross the memory contract. The selected batch is the largest
   successful measured size, not an arbitrary small default.
5. If memory is sufficient, no cleanup warning is shown.
6. An explicit user `batch_size` remains authoritative in the public API. The
   formal selected-module release launcher blocks this override unless the
   operator supplies both an explicit acknowledgement and an audit reason.
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
| renal | 281.7 s | 7,362.0 MiB | 7.91 GiB |
| respiratory | 73.1 s | 5,077.9 MiB | 5.46 GiB |
| medications | 88.7 s | 6,749.9 MiB | 7.25 GiB |
| neurological | 70.9 s | 4,604.9 MiB | 4.95 GiB |
| circulatory | 334.4 s | 4,721.5 MiB | 5.07 GiB |
| sepsis_shared | 21.7 s | 4,627.9 MiB | 4.97 GiB |
| other_scores | 190.6 s | 4,821.9 MiB | 5.18 GiB |
| sofa1_score | 165.5 s | 6,259.9 MiB | 6.72 GiB |
| sofa2_score | 437.7 s | 6,315.5 MiB | 6.78 GiB |
| sepsis3_sofa1 | 267.9 s | 5,749.6 MiB | 6.18 GiB |
| sepsis3_sofa2 | 428.7 s | 7,286.7 MiB | 7.83 GiB |

All 19 public modules now have full-cohort evidence. At 8 GiB currently
available, any subset including the complete 19-module selection receives the
one-shot fast path and no cleanup warning. The strictest module is `renal`,
which requires 8,098.2 MiB available after applying the 10% headroom rule. A
user extracting only full-cohort `blood_gas` still needs just 1,824.0 MiB and
therefore receives one-shot with 2 GiB available.

The eight-module production-shaped one-shot run took 359.7 seconds versus
522.4 seconds for the corresponding streamed run (1.45x faster), with a maximum
process-tree RSS of 3,544.4 MiB. The exact-clean ventilator verification at
commit `59f775e` took 78.7 seconds one-shot versus 197.5 seconds in fixed 5,000
stay batches; both published 963,266 rows and matched bidirectionally under
`EXCEPT ALL` (0/0).

The new 8 GiB profiling pass used a 7,447 MiB process-tree hard stop, two
DuckDB threads and a 2 GiB DuckDB memory limit. All 11 previously unmeasured
modules completed without triggering the stop. Logs show stable repeated
bucket scans inside `other_scores`, SOFA-2 and Sepsis-3 dependency graphs.
Those are code-level I/O/cache optimisation candidates; patient batching would
repeat the scans and is not the preferred speed fix while one-shot fits.

## MIMIC-III partial full-cohort measurements under the 8 GiB contract

Scope: 61,532 ICU stays with the same 7,447 MiB hard stop, two DuckDB threads,
2 GiB DuckDB limit and external output/spill. Thirteen modules have successful
one-shot evidence and medications has a measured batch profile; the remaining
five score/Sepsis modules retain the unmeasured guard.

| Module | Time | Peak RSS | Fastest verified mode |
|---|---:|---:|---|
| demographics | 12.2 s | 1,008.5 MiB | one-shot |
| outcome | 1.8 s | 612.3 MiB | one-shot |
| blood_gas | 8.2 s | 1,936.3 MiB | one-shot |
| hematology | 14.1 s | 2,637.5 MiB | one-shot |
| chemistry | 24.6 s | 2,823.9 MiB | one-shot |
| vasopressors | 111.9 s | 7,415.4 MiB | one-shot |
| ventilator | 91.9 s | 2,220.9 MiB | one-shot |
| vitals | 68.3 s | 6,577.7 MiB | one-shot |
| renal | 210.3 s | 6,231.2 MiB | one-shot |
| respiratory | 96.1 s | 7,182.0 MiB | one-shot |
| neurological | 39.0 s | 6,044.1 MiB | one-shot |
| circulatory | 447.1 s | 6,159.8 MiB | one-shot |
| sepsis_shared | 11.1 s | 5,659.3 MiB | one-shot |
| medications | 384.0 s | 7,236.8 MiB | 31,000 stays (2 batches) |

MIMIC-III medications one-shot ended without a worker manifest after an
observed 6,972.3 MiB lower-bound peak. A 40,000-stay candidate was then stopped
at 7,458.9 MiB, while 31,000 completed at 7,236.8 MiB. The streamed outcome
partition defect found during this search was repaired at the public concept
boundary; the fresh 31,000 + 30,532 package has 61,532 outcome rows and matches
the one-shot output under bidirectional `EXCEPT ALL=0/0`.

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

At an 8 GiB planning budget, selecting any subset of the 14 one-shot modules
runs each selected module one-shot. `respiratory` and `circulatory` use
50,000-stay batches; the other three batch-only modules use 67,000-stay
batches. Mixed and full requests preserve those per-module decisions rather
than applying the strictest 50,000-stay batch to every module. These measured
batch paths show no cleanup warning because 8 GiB already fits their fastest
verified peak plus headroom. A warning appears only when the planning budget is
below the relevant measured batch threshold.

The formal selected-module release launcher fixes its default planning budget
at 8,192 MiB for reproducibility even on a very large server. Use
`EX-A03_refresh_selected_modules.py --plan-only` to inspect the complete
database-by-module plan without cloning a candidate or opening raw data.

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
- MIMIC-IV remaining 11-module 8 GiB receipts:
  `/Volumes/外置硬盘/tmp/easyicu-6db-resource-profile-30f5228-ak9lbI/mimiciv/`
- Persistent MIMIC-IV audit and limitations (local workspace evidence, not
  repository source): workspace-root
  `task_logs/20260827_mimiciv_19_module_resource_standard.md`
- MIMIC-III partial receipts and partition repair:
  `/Volumes/外置硬盘/tmp/easyicu-6db-resource-profile-30f5228-ak9lbI/mimiciii/`
  and workspace-root
  `task_logs/20260827_mimiciii_partition_boundary_repair.md`
- eICU 19-module process-tree receipts and A/B outputs:
  `/Volumes/外置硬盘/tmp/easyicu-6db-resource-profile-30f5228-ak9lbI/eicu/`
- Persistent eICU audit and limitations (local workspace evidence, not
  repository source): workspace-root
  `task_logs/20260827_eicu_19_module_resource_standard.md`

These measurements support resource selection only for the listed MIMIC-IV and
eICU modules and cohort ceilings. They do not establish a safe threshold for
the other four databases. Add a new production
profile only after a clean full-cohort run records commit identity, cohort size,
elapsed time, process-tree peak RSS, output validity, and partition-invariance
evidence where batching can affect semantics.
