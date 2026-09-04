# Extraction algorithm and resource audit (2026-09-04)

## Scope and conclusion

This audit covers the EasyICU selected-module refresh used to repair invasive
ventilation evidence in eICU and MIMIC-III and its downstream SOFA/Sepsis
closure. Extraction is paused: the post-IMV eICU SOFA-2 path disproved the old
resource profile, so no new release is authorised by this document yet.

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
| sepsis_shared | one-shot | 200,859 | 1 | measured full cohort |
| sofa1_score | measured batch | 67,000 | 3 | measured full cohort batches |
| sofa2_score | quarantined fallback | 25,000 | 9 | old profile invalid after IMV dependency change; not yet a verified final batch |
| sepsis3_sofa1 | measured batch | 67,000 | 3 | measured dependency-complete run |
| sepsis3_sofa2 | quarantined fallback | 25,000 | 9 | inherits the invalidated SOFA-2 dependency; not yet a verified final batch |

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

These changes deliberately do not alter thresholds, time windows, missing-data
policy, public columns or row-grain semantics. The isolation writer has
regression coverage for partition order, frozen-schema alignment, atomic
failure cleanup and exact target scoping. Focused regression tests must pass
before a hard-limited performance run is allowed.

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
   favors memory safety but can reread overlapping dependency Parquets. Grouping
   them could reduce I/O, but the combined peak has not been validated under the
   8 GiB contract.
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
- The planner's output is copied into per-module producer manifests and
  selected-refresh provenance so the executed policy can be compared with the
  preflight plan.

## Acceptance gates before promotion

- Resource-plan unit tests and selected-refresh tests pass.
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

The targeted rebuild remains paused. First run the focused and full relevant
test suites and prove output invariance. The next hard-limited experiment is
31,000 stays, because one such batch already completed before accumulation in
the former long-lived process; 40,000 already failed within its first batch and
must not be treated as the first candidate. If 31,000 completes the full cohort,
larger candidates below 40,000 may be tested to find the fewest safe partitions.
Only the largest successful candidate may enter the measured profile.
MIMIC-III profiling and any broader release rebuild come afterwards.
