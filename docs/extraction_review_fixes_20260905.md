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

The relevant lint checks pass. The core suite plus Web resource-plan contract
run produced 1,986 passes, 52 skips, one deselection and one pre-existing
failure. A subsequent 120-test targeted run also covers the final profile
numbers and float32-oracle fixture. The full suite retains the pre-existing
`test_load_circ_failure_uses_levosimendan_and_theophylline` failure (`pd.NA`
comparison); its fixture and circulatory implementation are unchanged from
`2659d109`. It is not suppressed or presented as a passing full suite.
