# Extraction algorithm and resource audit (2026-09-04)

## Scope and conclusion

This audit covers the EasyICU selected-module refresh used to repair invasive
ventilation evidence in eICU and MIMIC-III and its downstream SOFA/Sepsis
closure. The implementation is now evidence-directed and bounded under a fixed
8 GiB release contract. It is not globally time-optimal: several remaining
full-table scans can be removed only after output-invariance benchmarks.

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
| sofa2_score | one-shot | 200,859 | 1 | measured full cohort |
| sepsis3_sofa1 | measured batch | 67,000 | 3 | measured dependency-complete run |
| sepsis3_sofa2 | one-shot | 200,859 | 1 | measured full cohort |

The previous launcher passed one request-wide 5,000-stay override. That bypassed
the resource owner, caused avoidable repeated scans and made the manifest say
`explicit_batch_size`. The request-wide automatic path also applied the
strictest module batch to light modules. Both paths are now blocked or removed
from formal release execution.

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

The current algorithm is suitable for the targeted rebuild after the above
tests, but “optimal” is claimed only where a measured fastest-safe profile
exists. The next performance experiment should profile the five MIMIC-III
score modules, then test a fused read-only SOFA/Sepsis dependency worker against
the isolated reference using bidirectional `EXCEPT ALL` and process-tree RSS.
