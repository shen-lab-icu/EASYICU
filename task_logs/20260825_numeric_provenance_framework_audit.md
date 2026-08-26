# Numeric provenance framework audit after H1 reader replay

Date: 2026-08-25 EDT

Task: `FIG2-DEV9-HELDOUT27` / framework-level manuscript provenance repair

Disposition: **P0 provenance defects repaired; prior H1 reader v7 withdrawn as validation evidence; analysis-only remains**

## Why this is a framework task

H1 was used only as a probe. The accepted repair criterion was that the same
defect and regression could be stated without reference to ventilation,
MIMIC-IV, or one benchmark item. No H1 value, variable, database, or manuscript
phrase was added to a shared prompt or runtime branch.

## Reproduced framework defects

1. **Near-zero numeric collision.** `_claim_numeric_distance` treated every
   manuscript number with absolute value at most `1e-12` as relatively equal to
   a registered zero. In the H1 v7 reader, `1e-52`, `1e-23`, and `1e-34` all
   reused one footnote whose actual owner was
   `plausibility_audit.age.out_of_range_n=0`.
2. **Unicode scientific notation bypass.** Values rendered as
   `1 × 10⁻⁵²` were not matched by `_NUMERIC_IN_PROSE_RE`, so they bypassed the
   numeric provenance gate entirely.
3. **Ambient effect-scale leakage.** A top-level `effect_scale=hazard_ratio`
   propagated through every nested mapping. RMST confidence limits therefore
   acquired false `effect_scale=hazard_ratio` metadata even though RMST
   difference and hazard ratio are different estimands.

Concrete prior-reader evidence:

- `/Volumes/外置硬盘/easyicu_data/figure2_dev9_h1_reader_final_v7_20260825/manuscript_scaffold_bound.md:17`
  reused `claim_12` for three different scientific-notation thresholds.
- The same file's `claim_12` definition pointed to
  `plausibility_audit.age.out_of_range_n`, canonical value `0`.
- Its RMST CI claim definitions carried `effect_scale=hazard_ratio`.

## Owner-level repair

- The manuscript binder now disables relative matching at zero. Exact equality
  remains valid; rounded zero displays may match only through the existing
  display-aware absolute window.
- The shared numeric lexer/parser now recognizes ASCII and Unicode scientific
  notation. Unicode values must bind to an exact registered numeric claim or
  remain visibly untraced.
- Numeric effect scale is now scoped to the mapping that declares it. It may
  flow through homogeneous lists, while arbitrary nested mappings start a new
  estimand scope. A row containing an explicit ratio field such as
  `hazard_ratio` provides local scale identity for its adjacent generic CI
  fields.

## Negative and preservation regressions

- A scientific-notation value near zero cannot bind to an unrelated zero-count
  claim.
- The exact same ASCII scientific-notation claim still binds.
- Unicode scientific notation is no longer invisible and exact Unicode
  scientific notation still binds.
- Top-level HR confidence limits retain HR identity.
- A nested restricted-mean-difference CI does not inherit HR identity.
- A nested row containing `hazard_ratio` still gives its own CI fields HR
  identity.

Focused validation:

- `tests/research_agent/test_pipeline_authority_regressions.py`: `44 passed, 1 skipped`.
- `tests/research_agent/test_evidence_authority_generation.py` plus
  `tests/research_agent/test_the_binder_reads_the_citation_the_writer_wrote.py`:
  `65 passed`.
- Adjacent numeric-claim/effect-scale selection: `3 passed, 344 deselected`.
- `git diff --check`: pass.

## Real-artifact replay

The fixed walker replayed the frozen H1 step summary with these outcomes:

- `reportable_survival_results.rmst.ci_low/ci_high` -> no hazard-ratio scale;
- each time-varying interval's `ci_low/ci_high` -> local `hazard_ratio` scale;
- registered zero versus manuscript `1e-52` -> no numeric match.

Rebinding the raw H1 scaffold with the fixed shared gate produced 19 untraced
values, including all seven ASCII/Unicode p-value thresholds. No p-value was
bound to a zero claim. This is the intended fail-closed outcome because the
thresholds are not the exact owner-issued p-values and several are numerically
false thresholds.

## Authority consequence

The prior v7 PDF remains a historical artifact but is no longer valid evidence
that “20 numeric claims all bind.” It must not be used as a final reader or
submission artifact. A replacement reader may be generated only after the
generic reportable-result projection contract emits exact owner-issued numeric
literals and the unchanged strict binder passes. No Provider rerun is required
for this framework audit.
