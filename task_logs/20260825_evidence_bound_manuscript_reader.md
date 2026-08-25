# Evidence-bound manuscript reader vertical slice (2026-08-25)

## Scope

- Isolated branch: `codex/writer-quality-gate-20260825`
- Starting HEAD: `306e77aa8723a5badd88374402d8e0b78ea074f2`
- Source run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_writer_resume_84f31fd_20260825/batch/e2/e2_lactate_mortality/aware/run_20260825T024807_5b7759`
- Scope is reporting and reader infrastructure only. No Planner, Coder, statistical analysis, Qualification12, or Held-out27 execution was run.

## Defect reproduced

The original E2 bound manuscript displayed the middle spline-knot quantile as
`0.5`, but `claim_7` had been bound to the rounded lactate missingness value
`0.463888712444`. The correct registered claim is
`scientific_runtime_receipt.spline_knot_quantiles[1] = 0.5`, produced by
`primary_adjusted_association` in
`statistic_step_summary_ea88e43d32198967`.

The cause was generic semantic scoring: the nearby exposure name outweighed
the more specific spline-knot phrase. The repair is case-neutral and contains
no E2, lactate, Sepsis, or benchmark item identifiers.

## Implemented contracts

- Deterministic numeric binding now gives spline-knot semantics priority over
  unrelated missingness candidates.
- `easyicu.manuscript-provenance/1` resolves every displayed numeric marker to
  exactly one registered `NumericClaim` and current EvidenceStore record.
- Every projected source/code/data artifact is digest-checked before release;
  stale, missing, escaped, ambiguous, or mismatched evidence fails closed.
- The public projection is path-free and excludes patient rows and raw data.
- JSON fields are accompanied by RFC 6901 pointers.
- The Copilot/Web reader renders the manuscript, makes every numeric occurrence
  clickable, and opens the exact JSON field, step, evidence SHA, code artifact,
  input artifact, and supporting-artifact lineage.
- LaTeX/PDF output makes bound numbers clickable and includes a numeric
  provenance appendix. Draft and human-review-required markings remain.
- `tools/build_manuscript_reader.py` performs provider-free reporting replay;
  it does not recompute scientific results.

## E2 output and authority

Bundle: `output/pdf/e2_evidence_bound_reader/`

- Corrected manuscript SHA-256: `2672d2f3f7d5c02defaa25b70ad9795c3e723dc8c6bb20d65719d6bfc39dda5d`
- Provenance SHA-256: `3a1f946ebe105ca83bf11b0252bc46e1496c2a85fb9e576f7c6df29daa50ce18`
- PDF SHA-256: `d6ef156fb6c7b388af8e7fbcc1f6e71e2af5319b520c31b6a3190f02ad5b03dc`
- Distinct numeric claims: 8; clickable manuscript occurrences: 22
- Figures: 3; references: 12; PDF pages: 9
- Provider calls/tokens/cost: `0 / 0 / $0`
- Claim ceiling: `analysis_only`
- `publication_authorized=false`; human scientific/authorship review remains required.

## Verification

- Focused reporting/Web/security batch: `119 passed, 2 deselected`.
- Static routes and Copilot research workflow: `196 passed`.
- JavaScript hostile-input cases: `12/12 passed`.
- Ruff format/check, Node syntax checks, and `git diff --check` passed.
- LaTeX log: no overfull boxes, LaTeX errors, undefined citations, or undefined references.
- Browser QA: 22 numeric buttons; opening `0.5` resolved to
  `/scientific_runtime_receipt/spline_knot_quantiles/1`; one detail panel open;
  no horizontal body overflow at the reviewed desktop width.
- Visual PDF review covered pages 1, 7, 8, and 9; no clipping or overlap was observed.

## Boundary

This closes the generic number-to-evidence reader path and one E2 vertical
sample. It does not establish that all nine Dev9 manuscripts meet published
article quality, does not close the published-anchor scientific gaps, and does
not authorize any result for submission. The same reporting-only replay must
be applied to the remaining eligible Dev9 outputs after the Planner branch is
frozen and merged.
