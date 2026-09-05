# Extraction resource strategy

EasyICU prefers measured one-shot execution, then a recorded batched plan.
"Largest tested batch" is not proof of globally fastest execution or the
minimum feasible partition count. Timings and failure bounds are specific to
the implementation, source layout and runtime envelope that produced them.
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
   requests retain the conservative full-cohort guard for every module without
   its own measurement; the AUMC SOFA-2 measurement below therefore does not
   authorise one-shot or 5,000-stay batching for another AUMC module.
8. An explicit resource budget owns the lower-layer worker configuration as
   well as the stay batch. At 8,192 MiB this means two internal workers, two
   Arrow/DuckDB threads, a 2,048 MiB DuckDB limit and a 512 MiB resolver cache;
   these values are recorded in the output manifest.
9. The API now executes the recorded plan with **fixed batches by default**.
   It does not grow later batches using the host's available RAM. Explicit
   adaptive growth is experimental and rejected with a fixed resource budget.
   A budgeted batched request requires an output directory and streaming;
   leaving `stream_output_batches=None` automatically enables streaming for
   batched disk exports. An in-memory full-module return is not covered by the
   measured streaming contract. These settings are not an OS memory hard cap:
   profiling still requires process-tree monitoring with an enforced stop.
10. Deferred merging is restricted to `aumc/respiratory`. `eicu/sofa2_score`
    retains its established append-after-each-child schedule. Both isolation
    and deferred-merge flags are included in module manifests.

The machine-readable owner is `easyicu.api.extraction.plan_extraction_resources`.
Its stable reason codes are:

- `measured_profile_fast_path`
- `measured_profile_fastest_safe_batch`
- `measured_profile_insufficient_memory`
- `calibrated_fast_path`
- `unmeasured_profile_memory_guard`
- `explicit_batch_size`

The legacy reason code `measured_profile_fastest_safe_batch` remains stable
for callers; it denotes the registered measured plan, not an optimality proof.

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

Twelve currently unaffected modules retain one-shot evidence:

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

Five modules crossed the one-shot contract and therefore use the fastest
successful measured batch instead:

| Module | One-shot evidence | Fastest verified batch | Batched time | Batched peak RSS |
|---|---:|---:|---:|---:|
| respiratory | stopped at 7,576.3 MiB | 50,000 stays (5 batches) | 251.7 s | 6,252.8 MiB |
| circulatory | stopped at 7,941.3 MiB | 50,000 stays (5 batches) | 494.7 s | 6,172.8 MiB |
| other_scores | stopped at 7,797.5 MiB | 67,000 stays (3 batches) | 436.3 s | 6,512.3 MiB |
| sofa1_score | stopped at 7,631.4 MiB | 67,000 stays (3 batches) | 512.5 s | 6,316.5 MiB |
| sepsis3_sofa1 | stopped at 7,475.6 MiB | 67,000 stays (3 batches) | 586.9 s | 6,294.5 MiB |
| sofa2_score | old one-shot invalidated after IMV update | remeasurement required | — | — |
| sepsis3_sofa2 | old one-shot invalidated after IMV update | remeasurement required | — | — |

The pre-2026-09-04 measurements for `sofa2_score` (6,926.6 MiB) and
`sepsis3_sofa2` (6,268.4 MiB) were quarantined because the eICU IMV
ascertainment update changed their respiratory dependency. After per-partition
process isolation and dependency fixes, 30k, 29k and 28k candidates still
crossed the 7,447 MiB hard stop on event-dense batches. The complete 25k
benchmark-only closure passed: external process-tree RSS was 6,750.1 MiB,
while the more conservative internal module sampler recorded 6,800.3 MiB.
That run nevertheless inherited host-sized lower-layer defaults despite its
8,192-MiB planning input. It is now diagnostic evidence only; the registry
keeps both modules invalidated until the corrected execution envelope has a
complete replacement receipt.

At an 8 GiB planning budget, selecting any subset of the 12 unaffected
one-shot modules runs each selected module one-shot. `respiratory` and
`circulatory` use
50,000-stay batches; the other three batch-only modules use 67,000-stay
batches. Mixed requests preserve those per-module decisions rather
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

## AUMC status under the 8 GiB contract

The sealed AUMC extraction was not batched: it processed all 23,106 stays in one
shot on a server with 385,972.7 MiB (about 377 GiB) assigned memory. Its
process-tree peak was
28,439.5 MiB; the in-process module sampler recorded 14,300.9 MiB for SOFA-1
and 26,363.6 MiB for SOFA-2. It therefore proves that one-shot works on that
large server, but it does not prove that one-shot fits a 16 GiB or 8 GiB
machine.

That release also recorded each isolated module's peak. Twelve modules remain
outside the later IMV/SOFA semantic repair and already fit the 8-GiB contract,
including 10% launch headroom. They therefore retain the measured full-cohort
one-shot path instead of inheriting the generic 5,000-stay guard.

| Module | One-shot time | Measured peak RSS | Threshold with 10% headroom |
|---|---:|---:|---:|
| demographics | 0.4 s | 210.1 MiB | 231.1 MiB |
| outcome | 0.3 s | 193.5 MiB | 212.9 MiB |
| blood_gas | 11.7 s | 2,207.2 MiB | 2,427.9 MiB |
| hematology | 78.9 s | 2,060.8 MiB | 2,266.9 MiB |
| chemistry | 74.9 s | 1,665.7 MiB | 1,832.3 MiB |
| vasopressors | 7.4 s | 2,253.5 MiB | 2,478.9 MiB |
| vitals | 73.5 s | 4,101.1 MiB | 4,511.2 MiB |
| renal | 67.2 s | 4,101.8 MiB | 4,512.0 MiB |
| medications | 29.5 s | 3,782.0 MiB | 4,160.2 MiB |
| neurological | 13.1 s | 2,035.7 MiB | 2,239.3 MiB |
| circulatory | 29.6 s | 3,677.6 MiB | 4,045.4 MiB |
| sepsis_shared | 1.3 s | 1,531.7 MiB | 1,684.9 MiB |

Three non-SOFA owners cannot use those measurements as an 8-GiB one-shot
authority: `respiratory` peaked at 28,893.8 MiB, `ventilator` at 14,020.3 MiB,
and `other_scores` at 15,553.3 MiB. `respiratory` and `ventilator` now use the
current measured batch profiles below; `other_scores` is also measured below.
None of these three now relies on the generic 5,000-stay safety guard.

The full AUMC `respiratory` owner was measured separately because its 15
concepts are materially heavier than the respiratory subset consumed by SOFA.
The first implementation retained Arrow/native writer pages while the next
patient batch was running, so 8,000, 7,000 and 6,000 candidates all crossed the
7,447-MiB hard stop late in extraction. The corrected path writes each batch in
a fresh process, defers bounded Parquet merging until all extraction children
have exited, and then runs native-v2 publication separately. At commit
`0e2b2dd0`, 5,000 completed all five partitions in 836.0 seconds. The external
whole-run peak was 6,996.6 MiB; the conservative largest internal batch sample
was 7,254.6 MiB, so the registry threshold is 7,980.1 MiB after 10% headroom.
Those larger failures predate the final deferred-merge implementation. Five
partitions are verified for `0e2b2dd0`; fewer partitions have **not** been ruled
out on that implementation. Re-test larger balanced partitions before making
a minimum-batch claim.

The final native-v2 table retained all 2,537,113 sealed row keys. Sixteen of 17
physical columns had identical logical multisets. The only change was
`ecmo=True` at stay 16292, hour 23: raw `procedureorderitems` contains the
explicit `ECMO - aPTT controle` source at 23.8 hours, and an independent
single-stay extraction reproduced the same value. This is a corrected old
source-evidence omission, not a partition-boundary artifact.

The AUMC `ventilator` boundary was then measured without batch-process
isolation because its same-process writer remained stable. The rounded
two-partition candidate of 12,000 stays crossed the 7,447-MiB hard stop at
7,473.0 MiB. The 8,000-stay candidate completed three partitions in 395.9
seconds (392.6 seconds in module extraction), with a 6,025.4-MiB external
process-tree peak and a lower 5,704.1-MiB internal module peak. Its admission
threshold was therefore 6,627.9 MiB after 10% headroom. Three partitions were
verified; failure at 12,000 does not exclude two balanced partitions of 11,553.

All 1,445,236 native row keys, schema fields and 12 raw/base concept columns
matched the sealed release. The only logical differences were 43 `vent_mode`,
67 `vent_breath_seq`, and 36 `driving_pres_controlled` values. These are the
effect of commit `095159ef`, which made hourly categorical `first`
deterministic. That earlier acceptance was insufficient: two separate axes
could select different native records at the same time and invent a hybrid
mode. Repeating the production algorithm was not an independent semantic
check. The source-selection repair and its independent raw oracle are
described in `extraction_review_fixes_20260905.md`. A separately reproduced
AUMC tidal-volume pre-resampling defect also requires a fresh module receipt;
the old timing and equality statement above are historical, not acceptance
of the corrected implementation.

The corrected `a86b4fd6` implementation subsequently completed the full cohort
with fixed 8,000/8,000/7,106 batches: 402.636 seconds including publication,
6,281.5 MiB external RSS, and a 6,909.65 MiB admission threshold. The full raw
mode and tidal-volume oracles pass, and all 11 unaffected base fields are
unchanged. See `extraction_review_fixes_20260905.md` for the intentional
tidal-volume corrections, precision contract and non-sealable candidate paths.

Finally, the AUMC `other_scores` rounded two-partition candidate of 12,000
stays crossed the same hard stop at 7,576.7 MiB after 68.1 seconds. The 8,000-
stay candidate completed three partitions in 439.1 seconds (436.3 seconds in
module extraction), with a 7,069.1-MiB external process-tree peak and a lower
6,738.9-MiB internal module peak. Its admission threshold is 7,776.0 MiB after
10% headroom. The published table contained 2,580,685 unique stay-hour rows
from all 23,106 stays; schema, keys, qSOFA, SIRS, MEWS and NEWS were exactly
equal to the sealed release in both multiset directions. Three is a verified
partition count, not a minimum proof: 11,553/11,553 was not tested. No additional
process-isolation mechanism was needed for the tested 8,000-stay plan.

Under the deterministic 8,192-MiB worker envelope, a first combined benchmark
showed that AUMC SOFA-1 could finish at 8,000 stays per batch, but its later
SOFA-2 step crossed the hard stop, so that partial run was not registered. A
dedicated boundary test then rejected the smallest rounded two-partition
candidate, 12,000 stays, at 7,472.3 MiB after 31.7 seconds. The dedicated
8,000-stay closure completed in three partitions (8,000/8,000/7,106): the
external process-tree peak was 6,535.4 MiB and `sofa1_score` took 574.9
seconds. This verifies three partitions under the 8-GiB contract, not the
minimum possible count. Two balanced 11,553-stay partitions remain untested;
even candidates with the same partition count can have different runtimes.

The SOFA-1 candidate preserves exactly the sealed 2,871,000 row keys and all
six component values. Its total is recomputed from those components on every
row; the sealed total differed from that formula on 2,349,193 rows because it
had previously been consolidated independently. The corrected standard
SOFA1-based Sepsis-3 output contains 6,900 positive event rows versus 6,707 in
the sealed package after first-event times are recomputed. Only
`sofa1_score.parquet` and `sepsis3_sofa1.parquet` changed; all other AUMC
module Parquets were SHA-identical. The 8,000-stay policy is therefore
registered only for these two modules.

The exact post-semantics SOFA-2 boundary search rejected both 7,000 stays
(7,714.9 MiB external process-tree RSS) and 6,000 stays (7,607.4 MiB). The
5,000-stay run at commit `2964e85a` completed all five partitions
(5,000/5,000/5,000/5,000/3,106) and the downstream Sepsis-SOFA2 closure. The
external monitor recorded 6,434.0 MiB over 942.9 seconds; the more conservative
module sampler recorded 6,583.5 MiB for `sofa2_score` over 927.2 seconds.
The registry uses that higher peak plus 10% headroom, yielding a 7,241.9-MiB
admission threshold. Thus 5,000 is the largest verified safe batch among
the tested 1,000-stay candidates, not a generic AUMC default.

The first normal-imputation implementation retained 124,393 hours created
only by the dense gap grid. A source-assessment marker now prevents those
synthetic empty hours from becoming scores: 33,018 were removed. The final
candidate has 2,679,032 SOFA-2 rows over 23,104 stays. Its 91,375 keys not
present in the old complete-case-oriented package are real component-owner
assessment times at which all six domains were unavailable; their primary
score is zero and their availability receipts remain false. This keeps them
distinguishable from the 251 fully available normal zero-score rows.

A read-only comparison against the sealed package found byte changes in only
`sofa2_score` and its experimental downstream `sepsis3_sofa2`; the other 17
AUMC module Parquets were SHA-identical. On common keys, coagulation, liver,
cardiovascular, CNS and renal values and receipts were identical. Respiratory
value/observed/available differed at one key, as expected from the corrected
worst-state/IMV publication rule. All 2,679,032 totals equal the receipt-aware
sum of the six components, range from 0 to 19, and no receipt-disclaimed row
has a non-zero total. The measured 5,000-stay policy is registered only for
`sofa2_score` and `sepsis3_sofa2`; no evidence here changes the strategy or
content of another AUMC module.

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

These measurements support resource selection only for the listed MIMIC-IV,
MIMIC-III, eICU and AUMC modules and cohort ceilings. They do not establish a
safe threshold for an unlisted module or database. Add a new production
profile only after a clean full-cohort run records commit identity, cohort size,
elapsed time, process-tree peak RSS, output validity, and partition-invariance
evidence where batching can affect semantics.
