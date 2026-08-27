# eICU 19-module extraction resource standard

Date: 2026-08-27 EDT  
Profiling baseline: `30f5228`  
Planner/profile integration baseline: `98e950b` plus the current owner patch  
Dataset: `/Volumes/外置硬盘/databases/eicu`  
Evidence root: `/Volumes/外置硬盘/tmp/easyicu-6db-resource-profile-30f5228-ak9lbI/eicu/`

## Question

For each of the 19 public extraction modules on all 200,859 eICU stays, what is
the fastest path supported by real process-tree RSS evidence under an 8 GiB
currently-available-memory contract? Patient batching is allowed only when a
full-cohort one-shot does not fit.

## Method

- One database/module at a time in an isolated process.
- Native-v2 publishing; source, output, temporary files and DuckDB spill on the
  external disk.
- `EASYICU_DUCKDB_THREADS=2`, `EASYICU_DUCKDB_MEMORY_LIMIT=2GB`,
  `EASYICU_OVERRIDE_MEMORY_GB=8`.
- Process-tree RSS sampled every 0.05 seconds.
- 7,447 MiB hard stop, corresponding to the largest measured peak that retains
  10% headroom inside 8,192 MiB.
- A successful one-shot is authoritative only for the exact database, cohort
  ceiling and module. A killed run is a lower bound, not a completed peak.
- For batch-only modules, larger candidate batches were tested first. A smaller
  batch became authoritative only after the larger candidate crossed the same
  hard stop.

## Result

Fourteen modules completed one-shot:

| Module | Seconds | Peak RSS MiB | Receipt directory |
|---|---:|---:|---|
| demographics | 2.951 | 1,194.6 | `demographics` |
| outcome | 2.624 | 1,114.3 | `outcome` |
| blood_gas | 7.293 | 1,104.6 | `blood_gas` |
| hematology | 18.537 | 3,024.5 | `hematology` |
| chemistry | 38.103 | 5,381.2 | `chemistry` |
| vasopressors | 21.048 | 5,517.3 | `vasopressors` |
| ventilator | 105.388 | 7,435.9 | `ventilator_threshold8g` |
| vitals | 158.082 | 5,362.4 | `vitals` |
| renal | 438.054 | 6,354.4 | `renal` |
| medications | 175.857 | 7,320.6 | `medications_threshold8g` |
| neurological | 88.829 | 5,073.9 | `neurological` |
| sepsis_shared | 11.817 | 4,977.7 | `sepsis_shared` |
| sofa2_score | 558.108 | 6,926.6 | `sofa2_score_threshold8g` |
| sepsis3_sofa2 | 372.807 | 6,268.4 | `sepsis3_sofa2_threshold8g` |

Five modules require batching at this contract:

| Module | One-shot lower-bound RSS MiB | Fastest successful batch | Seconds | Peak RSS MiB | Receipt directory |
|---|---:|---:|---:|---:|---|
| respiratory | 7,576.3 | 50,000 | 251.704 | 6,252.8 | `respiratory_batch50k` |
| circulatory | 7,941.3 | 50,000 | 494.721 | 6,172.8 | `circulatory_batch50k` |
| other_scores | 7,797.5 | 67,000 | 436.285 | 6,512.3 | `other_scores_batch3_retry` |
| sofa1_score | 7,631.4 | 67,000 | 490.554 | 6,316.5 | `sofa1_score_batch67k_cache1024_98e950b` |
| sepsis3_sofa1 | 7,475.6 | 67,000 | 586.933 | 6,294.5 | `sepsis3_sofa1_batch67k_dependencies` |

`respiratory` and `circulatory` both crossed the hard stop at 67,000 stays and
completed at 50,000. `other_scores` and `sofa1_score` crossed it at 101,000 and
completed at 67,000. The first `sepsis3_sofa1` batch attempt was invalid because
the test package omitted `sepsis_shared` and `sofa1_score`; the authoritative
run includes `outcome`, `sepsis_shared`, `sofa1_score` and `sepsis3_sofa1`.

## Output invariance and optimisation diagnostics

- The pre/post dependency-overlap SOFA outputs both contain 18,004,152 rows,
  198,770 stays, charttime -24.0 to 12,176.0, and the same order-independent
  all-column row-hash sum `166069046187761024218910109`.
- Respiratory contiguous 25,000-stay A/B: 234.235 seconds / 6,061.4 MiB versus
  production interleaved 50,000-stay 251.704 seconds / 6,252.8 MiB. Parquet
  pruning helps, but repeated item-bucket scans and recursive concepts remain
  the larger bottleneck.
- SOFA cache A/B: 1 GiB completed at 490.554 seconds / 6,316.5 MiB. The 1.5 GiB
  candidate showed the same remaining first-batch dependency reloads and was
  deliberately terminated at 166.607 seconds / 5,617.0 MiB; it is not a full
  performance result.
- The recurring “31 CSV files need conversion” message does not mean extraction
  used CSV. This eICU root already contains Parquet/sharded layouts; its legacy
  conversion status lacks the newer source-content receipts. Blind reconversion
  would waste external-disk I/O and is not part of this result.

## Product decision

- Register the 14 successful one-shot profiles in the resource owner.
- Register measured 50,000/67,000 batch profiles for the five one-shot failures.
- Remove the legacy 150,000-stay ceiling only for fully measured one-shot
  profiles; an unmeasured large module cannot borrow this authority.
- Measured batch evidence may exceed the historical three-scan heuristic.
- At sufficient memory for the fastest verified path, show no cleanup warning.
  Warn only below the relevant measured one-shot or batch threshold.

## Supports / does not support

This supports automatic resource selection for the listed eICU modules up to
200,859 stays on the profiled source layout. It does not prove identical elapsed
time on another disk/CPU, one-shot safety for the five killed modules above
8 GiB, or semantic invariance for every failed one-shot (there is no completed
one-shot output to compare). It is extraction-performance evidence, not
Qualification12, Held-out27, clinical, causal or publication authority.
