# Approved executable plan lifecycle — 2026-08-13

## Scope

Close the review finding that the pipeline used one mutable `AnalysisPlan`
object across proposal, host normalization, human review, and execution.  This
is a development/E1 governance fix; no Canonical9 question, shared prompt,
rubric, or formal experiment output was changed.

## Result

- Added typed, digest-bound `ProposedPlan`, `PlanTransformationReceipt`,
  `NormalizedPlan`, and `ApprovedExecutablePlan` authorities.
- The registered lineage records exact changed JSON fields and whether a
  transformation may change scientific semantics.
- The exact normalized plan is persisted before a review is offered.
- Approval persists an immutable executable-plan receipt bound to the shared
  `PlanReviewAuthority` and the exact human decision-set digest before Execute.
- Same-process and cross-process review resume use the same authority owner.
- Paper-facing legacy resumes that would require a new Planner scientific
  decision now fail closed and require a fresh run. Development migration is
  allowed only behind a new human-review pause.
- Existing plan lifecycle evidence is immutable: a resume may reuse the first
  lineage only when the exact normalized plan digest is unchanged.

## Verification

- Core lifecycle, durable review, migration, workflow, drift, and instance
  lifecycle tests: `67 passed`.
- Wider authority/review adjacency before the final owner extraction:
  `135 passed, 1 skipped`.
- Ruff and `git diff --check`: passed.
- `python tools/arch_measure.py --diff
  tools/arch_baselines/execution_phase.json`: passed without refreshing the
  baseline; `pipeline.py` is at or below its guarded LOC baseline.
- Commit: `88510e2`.

## Remaining boundary

This does not make a development Web UAT a formal paper experiment.  E1 must
still be run fresh through the ordinary Web conversation, reviewed, executed,
and assessed against its real artifacts before the exact-head release CI gate.
