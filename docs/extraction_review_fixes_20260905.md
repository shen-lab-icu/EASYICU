# Extraction review fixes — 2026-09-05

Scope: follow-up to the review of `b3226e8d` through `2659d109`.
No sealed release, global `current` pointer or downstream study is modified.

## Corrected defects

1. Ventilator control and breath-sequence axes previously broke exact-time
   ties independently on the derived labels. Both now select the same native
   label before hourly `first` aggregation. Exact source time remains primary;
   lexical native label, then source item, resolve ties deterministically, not
   as a clinical priority. Only conflicting rows are sorted. This shared
   mapper can affect tied mode records in AUMC, MIMIC-III, MIMIC-IV and HiRID;
   it is not restricted to the original eICU/MIMIC-III IMV request.
2. A fixed budget previously allowed default adaptive streaming to expand
   5,000 stays to the entire remainder using host RAM. Default execution now
   preserves the plan, rejects explicit adaptive growth with a fixed budget,
   and automatically streams batched disk exports. Budgeted patient batches
   cannot silently collect the full module in memory.
   The older full-six launcher `tools/reextract_native_export_v2.py` also
   explicitly enabled adaptive growth and omitted the API budget. It now
   passes the smaller of assigned and planning memory, disables growth and
   preserves per-module automatic plans rather than applying the aggregate
   batch to every light module. Explicit benchmark overrides remain explicit.
3. Deferred merging had also changed eICU SOFA-2. It is now explicitly scoped
   to AUMC respiratory; eICU appends after each child as before. Regression
   tests check the extraction/append event ordering and the output values.
4. Independent numerical checking exposed a pre-existing AUMC `tidal_vol`
   defect, reproduced with the original mapper as well. The raw fallback
   pre-resampled large source frames before bounds and cross-source pooling.
   An original [0, 2849] mL pair became 1424.5 despite the existing 2000-mL
   upper bound; a small extraction correctly returned 0. Pre-resampling is
   now disabled specifically for this AUMC concept. Unit conversion, raw
   bounds and pooled hourly aggregation retain their existing definitions.
   Other high-frequency fallback concepts have not been changed or certified
   by this fix and deserve a separate scope-controlled review.

## Independent verification design

- Synthetic tests cover source-order reversal, exact-time conflicts, separate
  patient partitions and unknown modes. No test should accept two individually
  valid categories if their combination was never recorded.
- Synthetic raw-source tests add 1001 unrelated rows and assert that AUMC
  tidal-volume values do not change. Unequal source multiplicities exercise
  pooled medians as well as pre-aggregation bounds.
- An independent AUMC raw-mode oracle joins `listitems` to `admissions`, uses
  `(measuredat - admittedat)` for ICU-relative hours, and chooses native records
  directly from source time/label/item. All 108 previously changed keys across
  68 stays have raw support. The corrected mapper matched all 108; the former
  mapper disagreed with this selection rule at 100. Agreement is with an
  explicit deterministic rule, not proof that a tied charted mode is uniquely
  clinically correct.
- The small extraction revealed 639 `tidal_vol` differences against the old
  full-cohort candidate; the other 11 base columns matched. The same 639 were
  present with the original mapper. These must be validated against raw
  pooled values, not silently required to reproduce the flawed full export.
- Full-module verification must use a fresh candidate, the fixed worker
  envelope and external process-tree stop, then compare keys/schema, the
  unaffected base columns, raw mode pairs and corrected tidal-volume values.
  No patient identifiers or ICU records belong in Git.

## What is not established

The older respiratory 6k/7k failures predate deferred merging and cannot prove
that five batches are minimal now. A 12k failure does not exclude balanced
11,553/11,553 partitions for a 23,106-stay cohort. Same-count partitions can
still differ in speed. Existing profiles are recorded feasible plans, not
proofs of globally fastest execution, fewest batches, or an optimal algorithm.
The public budget constrains workers/buffers/cache, not the OS address space;
process-tree monitoring remains necessary for boundary measurements.

## Full-cohort result on corrected code

Extraction implementation: `a86b4fd6`. Only AUMC `ventilator` was re-extracted,
in a fresh non-sealable benchmark candidate. The sealed `outcome` dependency
was reused without modification for native publication.

- Cohort: 23,106 stays; fixed partitions: 8,000 / 8,000 / 7,106. The manifest
  confirms no adaptive growth, no batch-process isolation and no deferred merge.
- Full execution including publication: 402.636 seconds; module: 399.5 seconds.
  External RSS peak: 6,281.5 MiB; PSS: 6,207.2 MiB. The 7,447-MiB monitor stop
  was not triggered. Updated admission threshold: 6,909.65 MiB including 10%
  headroom. Compared with the old 395.9-second run this is about 1.7% longer;
  these unreplicated runs do not establish a stable speed difference.
- Published rows: 1,441,932 with no duplicate stay-hour keys and unchanged
  schema. All 11 unaffected base numeric columns match in a full outer join.
- All 1,436,509 non-null mode hours match the independently selected native
  record; neither missing raw support nor missing expected output was found.
- All 1,432,754 non-null tidal-volume hours match the independently transformed,
  bounded, pooled raw median **after its existing float32 representation**.
  Unrounded double medians differ at 56,938 keys, by at most 0.00005 mL;
  explicit float32 comparison has zero differences. No tolerance was widened
  to conceal an algorithm error, and expected raw hours are not missing.
- Against the prior full-cohort candidate, `tidal_vol` changes at 175,348
  stay-hour keys. There are 3,305 removed keys and one added key; complete raw
  coverage and unaffected-column equality verify that this is a correction of
  the pre-resampling output, not loss of other recorded features. Mode,
  sequence and controlled driving pressure change at 52, 57 and 11 keys.

Reproducible raw checker: `tools/verify_aumc_ventilator_sources.py`. It uses one
DuckDB thread, a 512-MB buffer limit, isolated temporary spill and aggregate-only
JSON receipts; run it separately from extraction. The first diagnostic attempt
had a SQL alias error, and subsequent exact-double checks flagged representation
differences. All failed diagnostic receipts are retained, not overwritten.

Evidence root (outside Git):
`00-data-foundation/easyicu_full6_runs/candidates/` relative to the workspace.

- `benchmark_aumc_ventilator8k_a86b4fd6_20260905/exports/aumc/`
- `.planned_evidence/aumc_ventilator8k_a86b4fd6_20260905_memory.json`
- `.planned_evidence/aumc_ventilator_a86b4fd6_raw_oracle_v4.json`
- `.planned_evidence/aumc_ventilator_a86b4fd6_raw_oracle_memory_v4.json`

The initial core suite plus Web resource-plan contract run produced 1,986
passes, 52 skips, one deselection and one pre-existing failure. A subsequent
120-test targeted run also covered the final profile numbers and float32-oracle
fixture. At that point, the core suite retained the pre-existing
`test_load_circ_failure_uses_levosimendan_and_theophylline` failure (`pd.NA`
comparison); its fixture and circulatory implementation were unchanged from
`2659d109`. This historical failure is retained here; the test-isolation
follow-up below resolves it without changing the production scoring logic.

## Test-isolation follow-up and minimum remaining verification scope

This follow-up changes tests and this document only. It does not change any
production algorithm, concept source, resource profile or runtime limit. No
raw-data extraction, candidate promotion, pointer switch or M1–M4 run is
performed by these checks.

### Circulatory test failure

The drug-recognition fixture supplied lactate, MAP, levosimendan and
theophylline, but omitted seven optional drug streams. `load_circ_failure`
is allowed to supplement missing preloaded concepts from the configured
database; supplying a partial preload does not mean offline execution. As a
result this unit test accidentally depended on the machine's available data.
Sparse supplementary evidence correctly preserved unknown rows, contradicting
the fixture's assumed complete zero-rate evidence.

Both drug-recognition fixtures now explicitly provide all nine optional drug
streams, with zero values for drugs absent in the synthetic scenario. Their
original expected scores are unchanged. A module-local test guard fails on
any unexpected real concept load. Two additional regressions verify that:

- a supplied missing rate remains unknown and is not silently reloaded;
- a controlled, partially matched supplementary stream leaves unmatched
  hours unknown instead of turning them into negative events.

The production circulatory code and its missingness rules are unchanged.
The circulatory test file passes all 11 tests. The combined targeted run with
dispatch and ventilator-source tests passes all 148 tests, with lint passing.
The final core suite plus Web resource-plan contract run reports **2,109
passed, 52 skipped, one deselected, zero failures** in 66.90 seconds (1,887
warnings). This is not a claim that opt-in real-data tests or the entire Web
test suite were executed. Reproduction commands, from this EasyICU worktree:

```bash
python -m pytest tests/core tests/webserver/test_extraction_resource_plan_contract.py -q
ruff check tests/core/test_circ_failure.py tests/core/test_extraction_budget_execution.py tests/core/test_vent_mode_source_consistency.py
git diff --check
```

### Execution and dependency boundaries

The new dispatch matrix covers all six databases and all 19 public modules
(114 combinations). It supplies synthetic IDs, rejects raw concept loads and
captures the worker arguments before any process starts. The checks verify
that each module keeps its own planner-selected batch, automatic streaming
matches that module's plan, the 8,192-MiB budget is forwarded, and adaptive
growth remains disabled. This establishes plan-to-execution consistency,
**not** measured peak memory or optimal batch size for all 114 combinations.

Separate dictionary tests verify identical native source selectors for both
mode axes and walk both `sub_concepts` and `depends_on`. The recent mode and
tidal-volume fixes have only `ventilator` as a declared public-module consumer.
This narrow scope refers to the recent `a86b4fd6` semantic corrections, not
the entire earlier IMV/SOFA migration.

| Remaining concern | Smallest relevant scope | Current evidence / next requirement |
|---|---|---|
| AUMC mode pairing and tidal volume | AUMC `ventilator` | Full-cohort fixed-8k run and independent raw oracle passed above; no repeat is needed for this test-only follow-up. |
| Shared native-mode tie rule in other databases | MIMIC-III, MIMIC-IV and HiRID `ventilator` | Synthetic/source-selector tests pass; current-code real-data verification is still required before publishing changed outputs. |
| Fixed budget and per-module scheduling | Six-database dispatcher | All 114 synthetic dispatch cases pass; no blanket six-database extraction is justified by this check. |
| Earlier eICU IMV/SOFA dependency changes | `sofa2_score`, `sepsis3_sofa2` and their required dependency closure | Their old memory profiles remain invalidated; a complete receipt under the corrected execution envelope is still missing. |
| Other feature definitions | Unchanged by this test-only follow-up | Do not recompute them solely because tests or this document changed. This is not certification of every historical output. |

### Registered evidence is not the same as all historical experiments

At implementation revision `46ce5a91`, the executable resource registry contains:

| Database | Registered one-shot modules | Registered batched modules | Modules without a currently registered measured plan |
|---|---:|---:|---|
| MIMIC-IV | 19 | 0 | None |
| MIMIC-III | 13 | 1 | `other_scores`, both SOFA scores, both Sepsis-3 modules |
| eICU | 12 | 5 | `sofa2_score`, `sepsis3_sofa2` (explicitly invalidated) |
| AUMC | 12 | 7 | None |
| HiRID | 0 | 0 | All 19 |
| SICdb | 0 | 0 | All 19 |

An absent registered plan does **not** mean that a database/module was never
tested, needs new batching, or must be re-extracted now. First recover and
check historical receipts against the code, cohort, source layout and memory
envelope; reuse compatible evidence. Only unresolved, task-relevant gaps
justify new measurements. Existing registered profiles likewise do not prove
that every later source-semantic change has passed raw-data verification.

The outstanding work must not be described as "all six databases verified",
"globally fastest", "fewest possible batches", or "release ready". Sealed
data and global `current` remain separate from the corrected code branch.

## Integration with current main

Merge candidate `872c7b68` combined this branch with `8d3d7f69`. Text merging
was clean, but the wider local test run exposed two genuine binding failures:
the current E1 development profile and offline context baseline still pinned
the pre-correction SOFA-2 dictionary hash. That run was stopped for repair
after 4,474 passes, two failures and 122 skips; it is not a passing checkpoint.

Additive `20260905` E1 development profiles now bind the corrected packaged
SOFA-2 dictionary. All `20260904` profiles remain registered and immutable.
Regression coverage verifies that only the version, lock timestamp and
SOFA-2 dictionary hash differ between each old/new profile pair. Research
options, provider permissions and source-concept bindings are unchanged.
These application profile refs are not the global data-release `current`.

The generated offline context baseline records the intentional hash change in
its append-only history. The measured payload changes only at
`source_sha256["src/easyicu/data/sofa2-dict.json"]`: prompt sizes, selected
resources, provider calls (zero) and patient-data reads (zero) are unchanged.
The 53-test profile/baseline regression run and lint pass after this repair;
the integrated commit must still pass the PR checks before merging.

The full PR run on `3154b5d1` subsequently completed with 16,393 passes,
219 skips and one failure (the separately executed runner gate had 48 passes
and three skips). The remaining failure was `AF_UNIX path too long` while a
Docker security fixture created its socket, before the security check ran.
Real API dispatch tests had not restored process-wide temporary-directory
settings. They now restore both spill environment variables and
`tempfile.tempdir`; the socket fixture additionally uses a private short
directory and covers an intentionally long default temporary path. The
security rejection assertion is unchanged. Both serial and two-worker runs
of the 200 affected tests pass. This follow-up changes no production code.
