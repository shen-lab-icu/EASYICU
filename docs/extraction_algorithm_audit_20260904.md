# Extraction algorithm and resource audit (2026-09-04)

## Scope and conclusion

This audit covers the EasyICU selected-module refresh used to repair invasive
ventilation evidence in eICU and MIMIC-III and its downstream SOFA/Sepsis
closure. Extraction is paused: the post-IMV eICU SOFA-2 path disproved the old
resource profile, and the audit found a pre-existing SOFA publication defect
that affects all six databases. No new release is authorised by this document
yet.

No result in this audit authorises changing an AKI definition, cohort, time
window, or analysis endpoint. It changes extraction scheduling and closes one
respiratory publication invariant only.

## Corrected execution path

1. The release launcher obtains cohort sizes from the sealed `outcome` module
   receipt; `--plan-only` does not open raw database tables.
2. The planner selects a mode for every database and module under 8,192 MiB.
3. Modules run sequentially in isolated spawned processes. A measured one-shot
   module remains one-shot even when another module in the same refresh needs
   batching.
4. Streamed modules append Arrow batches to one temporary Parquet and publish
   atomically. The release launcher disables adaptive growth so the reviewed
   50k/67k boundary cannot silently change at runtime.
5. Native-v2 publication validates time bounds, physical bounds, row grain,
   semantic receipts, schema, hashes and provenance before replacement.

For the eICU respiratory dependency closure at 8 GiB, the executable plan is:

| Module | Mode | Stay batch | Planned batches | Evidence |
|---|---|---:|---:|---|
| respiratory | measured batch | 50,000 | 5 | 67k crossed the 8 GiB contract |
| sofa1_score | measured batch | 67,000 | 3 | measured full cohort batches |
| sofa2_score | quarantined fallback | 25,000 | 9 | old profile invalid after IMV dependency change; not yet a verified final batch |
| sepsis3_sofa1 | measured batch | 67,000 | 3 | measured dependency-complete run |
| sepsis3_sofa2 | quarantined fallback | 25,000 | 9 | inherits the invalidated SOFA-2 dependency; not yet a verified final batch |

`sepsis_shared` is no longer a refreshed module in this closure. Its sealed
Parquet is copied and SHA-verified in staging as a read-only Sepsis dependency,
then removed with staging. This avoids a raw reread and prevents an independent
infection module from being mislabeled as changed by an IMV/SOFA repair.

The previous launcher passed one request-wide 5,000-stay override. That bypassed
the resource owner, caused avoidable repeated scans and made the manifest say
`explicit_batch_size`. The request-wide automatic path also applied the
strictest module batch to light modules. Both paths are now blocked or removed
from formal release execution.

The 25,000-stay entries above are conservative planner fallbacks, not a new
fixed production standard. The final value must be the largest candidate that
completes under the 7,447 MiB process-tree hard stop after the algorithm fixes
below. Until then, the executable reason code is
`invalidated_profile_memory_guard` and the candidate must remain unsealed.

## SOFA-2 algorithm review after the failed run

The old eICU measurements (`sofa2_score` 6,926.6 MiB and
`sepsis3_sofa2` 6,268.4 MiB) predated the IMV ascertainment change. They are no
longer present in the measured-profile registry. Controlled post-update runs
at 67,000, 50,000 and 40,000 stays crossed the 7,447 MiB hard stop. A pre-fix
31,000 run was stopped at the user's request before completion. After the
duplicate-materialisation fixes, 40,000 still crossed the limit during its
first batch (7,449.9 MiB), while 31,000 completed its first batch and crossed
the limit only as the second batch began (7,465.4 MiB). The latter localises
the remaining defect to cross-batch allocator retention rather than a
31,000-stay working set. None of these failed runs authorises a production
batch size.

After exact per-batch process isolation was added, a full 31,000-stay-batch
run completed all seven batches and all three selected eICU modules. Its
high-frequency internal sampler nevertheless measured a 7,479.3 MiB peak,
above the 7,447 MiB admission limit; the slower external sampler missed this
brief peak and reported 7,404.8 MiB. The 31,000 profile therefore remains
rejected. No lower candidate is to be measured until the semantic repairs in
the next section pass their full test gate.

The code review found three algorithmic peak owners:

1. `sofa2_score` requests the aggregate and its six component outputs. The
   aggregate recursively computes those components, but the 512 MiB reusable
   cache can correctly reject the large frames. The export loop then computed
   them again. A top-level request-owned, zero-copy result map now reuses only
   components that must already remain alive for final output; it does not
   enlarge the reusable cache.
2. The aggregate callback built a six-score wide frame, independently rebuilt
   a 12-receipt wide frame and then outer-joined both. It now carries each
   component's two owner receipts through one indexed concat.
3. The dense hourly gap grid represented ordinal scores and binary receipts as
   float64. The SOFA-2 owner now requests exact float32 storage for this grid.
4. Python, Arrow and DuckDB cleanup calls did not return all native allocator
   arenas after a completed batch. The eICU `sofa2_score` stream path now runs
   every patient batch in a fresh spawned process. The bounded parent appends
   the child's temporary Parquet in 64k-row Arrow record batches; process exit
   is therefore the memory-release contract. This exception is registered by
   the exact `(database, module)` pair and does not change another database or
   module without its own evidence.

Batch patient identifiers are passed through the recursive component loader
and into the DuckDB/Arrow source predicates. The partition boundary is not
silently dropped. The near-flat peaks across the failed 67k/50k/40k attempts
therefore pointed to redundant materialisation and allocator lifecycle, not a
justification for an arbitrary 5,000-stay split.

The isolation changes do not alter thresholds, time windows, missing-data
policy, public columns or row-grain semantics. The isolation writer has
regression coverage for partition order, frozen-schema alignment, atomic
failure cleanup and exact target scoping. Focused regression tests must pass
before another hard-limited performance run is allowed.

## SOFA publication and Sepsis dependency finding

The native-v2 publisher previously resolved duplicate `(stay_id, charttime)`
rows by independently taking the median of every numeric column. That rule is
valid for ordinary continuous measurements, but not for an ordinal organ
severity score and its derived total. Across the sealed six-database release,
SOFA-1 total/component mismatches were found in every database; SOFA-2 had the
same defect wherever the total was available. M1 is directly affected because
it derives `sofa_nonrenal = sofa - sofa_renal`.

The repaired algorithm is:

1. For a duplicate ICU-hour, consolidate each SOFA organ component with
   `max(non-null)` (the worst observed state), matching the score callback's
   trailing-window maximum.
2. For SOFA-2, only values with an owner-issued availability receipt can enter
   that maximum.
3. Recompute SOFA-1 as the sum of the six consolidated components, retaining
   the established SOFA-1 missing-as-zero policy.
4. Recompute SOFA-2 only when all six consolidated components are available;
   rebuild aggregate observed/available receipts from the component receipts.
5. Before deriving Sepsis-3, perform the same component-first consolidation on
   producer artifacts. Never compute a delta over arbitrary duplicate-row
   order or over an independently aggregated old total.

Both the pandas fallback and bounded DuckDB/Arrow publisher paths implement
the same rule, and the rule is recorded in each module's row-grain manifest as
`score_component=max_non_null_worst_state` and
`derived_score_total=recomputed_after_component_consolidation`.

The release tool accepts an audited per-database module scope so the final
repair can refresh respiratory only in eICU and MIMIC-III while refreshing only
SOFA/Sepsis derivatives in AUMC, HiRID, MIMIC-IV and SIC. Reused modules in
every selected database receive the same bounded logical-multiset audit as a
fully publication-only database; selecting a database no longer exempts its
unrelated modules from invariance proof.

The selected-refresh publisher also had an independent recovery defect: it
could finish all expensive modules and then fail because the selected staging
directory did not contain `outcome.los_icu`, which is required only as the
time-bound authority. Recovery now copies and SHA-verifies the sealed outcome
artifact, recognises complete pre-native producer staging, and republishes it
without rereading raw data. The copied outcome remains a dependency and is not
declared as refreshed.

The already completed 31,000-stay candidate was recovered this way without raw
reread, but it was produced before the SOFA consistency repair and remains an
audit-only, unsealable candidate.

## Clinical-semantics finding (not silently changed)

The component threshold implementations have source-bound golden tests, but
the aggregate is explicitly registered as a non-canonical database
operationalisation pending independent clinical review. It currently requires
all six component availability receipts before publishing a total; official
SOFA-2 day-1 scoring instead treats missing components as normal/zero and its
later longitudinal policy carries the last value forward. The production
callback also emits hourly trailing-24-hour values, whereas the official
longitudinal description is daily.

This is a scientific-policy difference, not a performance bug. It is preserved
for output invariance during the resource repair and must be reviewed as a
separate versioned semantic decision before SOFA-2 is used as a canonical
endpoint.

## Time and I/O review

### Sound choices retained

- Concepts are loaded by module, not one concept at a time, so concepts sharing
  a raw table can reuse the loader scan.
- Each process exits after its execution unit, returning Python, Arrow and
  DuckDB allocator arenas to the operating system.
- Streamed output is appended directly to Parquet; full module DataFrames are
  not accumulated in the parent process.
- Existing source releases are cloned by hard link and only refreshed module
  files are atomically detached, so unchanged modules incur no raw reread.
- Parquet uses Snappy and bounded Arrow batches; DuckDB semantic operations use
  one thread, a 512 MiB publication limit and spill beside the candidate.

### Remaining non-optimal work

1. **Score dependency scans.** Isolated `sofa1`, `sofa2` and Sepsis-3 execution
   favors memory safety but can reread overlapping dependency Parquets. The
   Sepsis pass now projects only suspected infection, six component scores and
   the necessary SOFA-2 availability receipts rather than reading every score
   sidecar. Grouping whole modules could reduce more I/O, but the combined peak
   has not been validated under the 8 GiB contract.
2. **Source rescans in patient batches.** A batch-only module may re-enter the
   same raw table for each patient partition. Predicate pruning helps when the
   physical layout is favorable, but eICU profiling showed that more contiguous
   25k partitions were only 6.9% faster than the verified 50k interleaved path.
   A durable item-level source cache or a one-pass partitioned scan is the
   relevant future optimisation, not smaller batches.
3. **Respiratory semantic publication pass.** Canonical `supp_o2` support is
   checked after time filtering because a removed ventilation point can orphan
   an hour-bucketed oxygen claim. The repair performs an additional bounded
   DuckDB scan and, only when orphans exist, one Arrow schema-restoration pass.
   Folding this rule into the primary row-grain consolidation could remove one
   pass, but must first reproduce the current logical multiset and metadata
   exactly.
4. **MIMIC-III score profiles.** Five score/Sepsis modules remain unmeasured
   under 8 GiB, so the planner conservatively chooses 20,000 stays (four
   batches). Benchmarking larger candidates under a hard process-tree stop is
   the only defensible way to reduce those batches.
5. **Other database profiles.** AUMC, HiRID and SIC do not yet have the same
   module-level 8 GiB evidence. Their conservative plans are not proof of a
   fastest batch.
6. **Hash and publication reads.** SHA-256, logical multiset QC and metadata
   binding necessarily reread published files. These passes are audit costs;
   removing them would weaken release integrity. They can be scheduled once per
   immutable artifact but should not be skipped.

### SICDB SOFA-2 structural availability

SICDB is an explicit availability exception, not an extraction failure. Its
canonical CNS owner has no fully ascertained rows, so a six-component SOFA-2
total is structurally unavailable even though all six component columns
contain scored rows and the other five component owners contain available
evidence. The refresh gate permits an all-null SICDB SOFA-2 total only when
those exact conditions hold and the aggregate availability receipts remain
coherent; the same output still fails for every other database.

## Memory review

- The formal launcher uses a fixed 8,192 MiB planning contract rather than the
  host's installed memory. This prevents a large shared server from silently
  selecting an execution mode that could not be reproduced on the profiled
  machine.
- Measured admission requires process-tree peak RSS plus 10% headroom.
- Known eICU batch-only modules use the largest successful measured batch, not
  a generic 5,000-stay fallback.
- Explicit `--batch-size` is rejected by the release CLI unless accompanied by
  an override acknowledgement and a recorded reason.
- A formal selected-module release now fails before cloning or rereading raw
  data when any database/module would use a calibrated, invalidated or other
  unmeasured fallback. Explicit benchmark overrides remain possible but are
  marked non-admissible and the release sealer rejects them.
- The planner's output is copied into per-module producer manifests and
  selected-refresh provenance so the executed policy can be compared with the
  preflight plan.

## Acceptance gates before promotion

- Resource-plan unit tests and selected-refresh tests pass.
- SOFA-1 and SOFA-2 totals exactly match their post-consolidation component
  contracts in all six databases, and Sepsis labels are derived from those
  coherent timelines.
- Respiratory native-v2 tests prove unsupported `supp_o2` is cleared while
  FiO2- or ventilation-supported rows remain unchanged.
- The cross-database QC recomputes semantic-row accounting and fails on a
  tampered receipt.
- Refreshed eICU and MIMIC-III modules must pass row count, schema, row-grain,
  time-axis, semantic and SHA/provenance checks.
- Unselected database/module logical multisets must match the sealed source.
- The candidate remains unsealed and no global `current` pointer is changed
  until downstream review is complete.

## Decision

The targeted rebuild remains paused. First complete the focused and full
relevant test suites, then perform a publication-only six-database SOFA impact
audit and determine whether all Sepsis-3 derivatives require regeneration.
Only after that semantic closure may a lower eICU batch candidate be measured;
31,000 is already rejected, so 30,000 is the highest sensible next candidate.
It is admitted only if process-tree RSS plus the required headroom fits the
8,192 MiB contract. MIMIC-III profiling and the broader release rebuild come
afterwards. No candidate may change `current` before downstream review.
