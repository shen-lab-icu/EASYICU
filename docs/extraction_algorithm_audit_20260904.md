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
2. The planner selects a mode for every database and module under 8,192 MiB,
   and installs that same contract inside every spawned worker: two internal
   workers, two Arrow threads, two DuckDB threads, a 2,048 MiB DuckDB limit and
   a 512 MiB resolver cache.
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
| sofa2_score | invalidated guarded pilot | 25,000 | 9 | old closure used host-sized lower-layer defaults; formal run blocked |
| sepsis3_sofa1 | measured batch | 67,000 | 3 | measured dependency-complete run |
| sepsis3_sofa2 | invalidated guarded pilot | 25,000 | 9 | inherits the invalidated SOFA-2 dependency; formal run blocked |

`sepsis_shared` is no longer a refreshed module in this closure. Its sealed
Parquet is copied and SHA-verified in staging as a read-only Sepsis dependency,
then removed with staging. This avoids a raw reread and prevents an independent
infection module from being mislabeled as changed by an IMV/SOFA repair.

The previous launcher passed one request-wide 5,000-stay override. That bypassed
the resource owner, caused avoidable repeated scans and made the manifest say
`explicit_batch_size`. The request-wide automatic path also applied the
strictest module batch to light modules. Both paths are now blocked or removed
from formal release execution.

The previous 25,000-stay entries are no longer production strategies. A
complete benchmark-only closure at EasyICU commit `f061bcc0` finished in 619.2 seconds:
external process-tree RSS peaked at 6,750.1 MiB and the higher internal sampler
peak was 6,800.3 MiB, but its child still detected the 1.5-TiB/384-core host
and selected host-sized parallel/cache defaults. The resource planner had used
8,192 MiB only to choose the stay batch; it had not constrained the lower
layers. The two SOFA-2 profiles are therefore invalidated until remeasured
under the corrected worker envelope. The benchmark artifact remains
non-sealable and diagnostic only.

## Execution-envelope finding

The release API accepted `resource_budget_mb=8192` but previously used it only
in `plan_extraction_resources`. On the shared server, a child still reported
64 workers, eight Arrow/DuckDB threads, a 4-GiB DuckDB limit and a default
resolver-cache ceiling derived as 25% of 1.5 TiB. This made the declared 8-GiB
run host-dependent and allowed abrupt allocation spikes during an AUMC
SOFA-2 boundary test.

The corrected worker setup converts the available-memory contract into one
deterministic lower-layer envelope before any loader or DuckDB connection is
constructed. The envelope is written to producer and native manifests and to
benchmark provenance. An existing profile without this envelope can inform
the next candidate batch but cannot authorise a formal release. The aggregate
request planner was also corrected to include one-shot modules when reporting
the maximum sequential peak; per-module execution itself was already correct.

The memory-evidence runner now starts the measured command in a separate
session and records `status=interrupted`, exit code 130, after stopping its
complete process tree. This closes the prior failure mode in which Ctrl-C left
a checkpoint incorrectly marked `running`.

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

After those repairs, complete hard-stop runs rejected 30,000 stays at 7,459.7
MiB, 29,000 at 7,447.9 MiB and 28,000 at 7,854.0 MiB. The non-monotonic peak
confirms that per-batch event density matters in addition to stay count. The
25,000-stay run completed all nine SOFA-2 partitions and all nine downstream
Sepsis partitions without crossing the limit. It produced 17,393,757 SOFA-2
rows and 30 positive Sepsis-3/SOFA-2 rows with error-free producer manifests.

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
2. For SOFA-2, consolidate the recorded component score and its owner-issued
   availability receipt separately; a receipt-disclaimed non-zero score is an
   error, while a missing patient-level domain contributes the normal score of
   zero to the primary aggregate.
3. Recompute SOFA-1 as the sum of the six consolidated components, retaining
   the established SOFA-1 missing-as-zero policy.
4. Recompute the primary SOFA-2 total with normal-value imputation, matching
   the JAMA analysis; rebuild aggregate observed/available receipts as the
   separate all-six-component complete-case indicator.
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

## Clinical-semantics finding and versioned correction

The component threshold implementations have source-bound golden tests, but
the aggregate remains a non-canonical database operationalisation pending
independent clinical review. The earlier resource-repair branch temporarily
required all six component receipts before emitting any total. A full AUMC
comparison showed that this reduced 1,682,831 historical non-null totals to
2,940 and therefore could not be accepted as an output-invariance repair.

The versioned correction follows the published primary analysis: a
patient-level missing domain contributes zero, while `sofa2_available` and
`sofa2_observed` retain the all-six-domain complete-case status. This separates
the primary score from the complete-case sensitivity analysis. It does not
permit an entirely absent database domain: SICDB still publishes no aggregate
total because its canonical CNS owner is structurally unavailable. The
production callback remains an hourly trailing-24-hour operationalisation,
whereas the official longitudinal description is daily; that limitation stays
explicit.

The correction was then exercised in a hard-limited AUMC full-cohort closure.
The old strict total was non-null for only 2,940 of 2,587,657 rows. A first
normal-imputation pass exposed a distinct row-domain defect: 124,393 hours
created solely by `fill_gaps` acquired zero scores. The aggregate now carries
an internal source-assessment time marker through the rolling fast path. A
patient-level missing domain contributes zero only at a component-owner
assessment or where at least one component remains available in the rolling
window; a completely synthetic empty grid hour remains missing and cannot add
a module row.

At commit `2964e85a`, the corrected AUMC package contains 2,679,032 SOFA-2
rows. The 91,375 new keys are owner assessment times with all six domains
unavailable, not synthetic gap rows; their total is zero and their aggregate
availability receipt is false. Exactly 2,940 rows remain in the all-six-domain
complete-case receipt. Every total equals the receipt-aware component sum,
ranges from 0 to 19, and no unavailable component contributes a non-zero
value. The experimental Sepsis-SOFA2 sensitivity output changes from 1,466 to
6,907 positive event rows because the newly defined incomplete trajectories
can supply baseline and change scores; it does not replace standard
SOFA1-based Sepsis-3.

The AUMC SOFA-1 closure was measured independently at commit `8f72b901`.
A 12,000-stay candidate, the smallest rounded size capable of covering 23,106
stays in two partitions, crossed the 7,447-MiB hard stop at 7,472.3 MiB. The
8,000-stay candidate completed 8,000/8,000/7,106 stays with a 6,535.4-MiB
process-tree peak. Its 2,871,000 published keys and six component columns are
identical to the sealed package, while all totals equal the component sum.
The sealed total violated that formula on 2,349,193 rows because totals and
components had been consolidated independently. The corrected standard
SOFA1-based Sepsis-3 sensitivity therefore moves from 6,707 to 6,900 first
positive event rows. No non-SOFA1 module other than its declared Sepsis
consumer changed SHA.

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
5. **Other database profiles.** AUMC now has exact 8-GiB evidence for both
   SOFA score modules and their two Sepsis consumers; its other 15 modules,
   together with HiRID and SIC modules, do not yet have equivalent evidence.
   Their conservative plans are not proof of a fastest batch.
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
- `--benchmark-only` runs exactly one database closure, writes a dedicated
  non-sealable resource receipt and skips six-database republication. This
  prevents a batch-boundary experiment from reprocessing unrelated databases.
- The 8-GiB planning budget now constrains the child worker's parallelism,
  DuckDB buffers and resolver cache; the exact envelope is part of provenance.
- The planner's output is copied into per-module producer manifests and
  selected-refresh provenance so the executed policy can be compared with the
  preflight plan.
- Cumulative selected-refresh provenance now retains the exact lower-layer
  execution limits as well as the high-level memory budget and batch plan;
  inherited lineage can no longer drop the DuckDB/thread/cache contract.

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

The targeted rebuild remains paused while eICU, AUMC, HiRID, MIMIC-III and SIC
score closures receive measurements under the exact lower-layer 8-GiB
envelope. The prior eICU 25,000-stay receipt is diagnostic but no longer
admissible. A publication-only impact
audit confirmed that current native-v2 republication adds only
catalog-declared all-null placeholders to unaffected modules until it reaches
the deliberately corrected SOFA files; existing source columns remain
multiset-identical. The final refresh scope therefore regenerates SOFA-1,
SOFA-2 and their Sepsis consumers in all six databases, plus respiratory only
in eICU and MIMIC-III. No candidate may change `current` before downstream
review.
