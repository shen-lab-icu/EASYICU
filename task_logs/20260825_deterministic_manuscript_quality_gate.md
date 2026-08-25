# Deterministic manuscript quality gate and Dev9 replay — 2026-08-25

## Scope and authority

This change adds a case-neutral writing-quality layer under the reporting owner.
It does not modify Planner, analysis methods, scientific results, evidence
authority, or publication authority. The implementation branch is
`codex/writer-quality-gate-20260825`, based on exact source HEAD
`e95e72352829c9cb8c05950188e707cbf0114d50`.

The new `manuscript_reader.md` is explicitly non-authoritative. The bound
manuscript and EvidenceStore remain the audit surface. A passing writing audit
does not convert an `analysis_only` run into a paper-ready result.

## What was implemented

1. `reporting/manuscript_quality.py` owns a deterministic contract for required
   manuscript sections, required Abstract labels, required Methods/Results
   subsections, high-confidence Methods/Results adjustment-set consistency, and
   reader-facing leakage of raw runtime identifiers or engineering terms.
2. Evidence filtering may no longer leave an empty section unnoticed. The
   section writer receives one bounded structural retry, then fails closed.
3. Writer prompts now require reader-facing clinical labels and exact reuse of
   the executed adjustment set. These are general reporting rules, not Dev9
   case instructions.
4. The write phase persists `manuscript_quality_audit.json` and the clean reader
   view, registers both with EvidenceStore, and forces the manuscript critique
   to `blocked` when deterministic quality errors remain.
5. Readiness recognizes `manuscript_quality` as a manuscript-blocking validator
   and publishes the two artifact paths when present.
6. `tools/audit_manuscript_quality.py` replays old manuscripts into an isolated
   output directory without changing source runs and without Provider calls.

## Published-anchor relationship

This layer complements rather than replaces the existing 9 x 7 published-paper
comparison in
`task_logs/20260825_dev9_accessible_anchor_writer_quality_remediation.md`.
That comparison used 14/14 accessible full texts and all 11 available
supplement packages, and found 25 scientific/design gaps across 63 dimensions.
Those gaps include external validation, censoring, stability, positivity, and
measurement-by-indication issues that prose cannot repair.

The deterministic writing gate addresses a different failure class: whether a
reader-facing manuscript is structurally complete, internally consistent, and
free of raw engine language. Published effect sizes or directions are not used
as expected answers.

## Read-only replay of the selected Dev9 Writer overlays

Replay artifacts:
`output/manuscript_quality_replay_dev9_20260825/` (gitignored local evidence).

| Case | Writing status | Deterministic errors | Main problem class |
|---|---|---:|---|
| E1 | changes_required | 5 | empty Conclusion, adjustment-set conflict, internal terminology |
| E2 | pass | 0 | none detected by this bounded gate |
| E3 | changes_required | 3 | incomplete Abstract label, internal terminology |
| M1 | changes_required | 3 | incomplete Results subsection/Abstract label, internal terminology |
| M2 | changes_required | 2 | raw patient/runtime identifier in reader prose |
| M3 | changes_required | 1 | internal terminology |
| H1 | changes_required | 2 | internal terminology |
| H2 | changes_required | 2 | internal fail-closed reason code exposed to readers |
| H3 | changes_required | 2 | incomplete Abstract labels |

The replay result is 1/9 pass and 8/9 changes_required. This does not contradict
the earlier numeric/literature/critic pass: those gates check different
contracts. In particular, the E1 Critic pass did not detect its empty Conclusion
or its Methods/Results covariate conflict; the new deterministic owner does.

Provider calls: 0. No frozen Dev9 run directory was modified.

## Verification

- Ruff on changed source/tests/tool: pass.
- Focused reporting, Writer, manuscript, and readiness suite: 209 passed,
  9 deselected, 4 warnings.
- Real selected-overlay replay: 9/9 manuscripts audited; output summary records
  source and reader SHA-256 digests.
- `git diff --check`: pass.

## Next acceptance step

After this branch is merged onto the Planner-frozen exact HEAD, reuse the saved
analysis checkpoints and regenerate only the Writer/reporting suffix once for
the eight failing cases. Accept the regenerated writing layer only when all
nine have `manuscript_quality_audit.status=pass`; keep the published-anchor
science review and paper authority gates separate.
