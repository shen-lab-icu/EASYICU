# Figure 2 prediction capability integration

- Date: 2026-08-17
- Target: `integration/figure2-e1-h3-20260816`
- Pre-integration target: `5228a04`
- Capability source: `codex/cap-predval-v5-1-persisted-refit-runtime-20260817@1a1fc11`
- Merge commit: `0033f7e`

## Scope

The V1 through V5 branch labels were milestones of one linear prediction-fit and
validation capability chain. Only the final V5.1 tip was merged. The independent
`637bac1` integration-check branch was not used as a merge source.

The capability remains experimental and `analysis_only`. This integration does
not wire it into Planner selection, production authority, or paper-facing claim
publication.

Immediately before the merge, the in-flight E1 association sensitivity-grid
work was characterized and committed as `5228a04`. Its host compiler and
verified-tool executor have focused authority, routing, execution, and tamper
tests; this code commit is not an E1 run receipt.

## Exact-head verification

The shared virtual environment was forced to import this worktree with
`PYTHONPATH=$PWD/src:$PWD`; the editable environment otherwise resolves the
`main` checkout and can produce a false-green result for branch-local code.

- Ruff on all Python files changed since `fcdfb6a`: passed.
- E1 model-grid, current-case authority, all prediction V5.1 owner/evidence/
  persisted-runtime/provenance tests, and capability inventory governance:
  `111 passed`.
- Standalone capability inventory audit: passed.
- Module graph: `547 modules / 2,177 edges / 0 cyclic SCCs`.
- Worktree after verification: clean.

No E1 canary was launched. The last audited scientific outcome therefore remains
unchanged: E1 is not yet complete and formal Held-out27 authority remains 0/27.

## Handoff

Continue from the clean pushed Figure 2 branch. Run the next bounded E1 canary
before entering E2. Prediction capability promotion requires separate real-data
and external-estimator validation; it must remain outside paper authority until
those gates close.
