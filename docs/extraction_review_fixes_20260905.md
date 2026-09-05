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
