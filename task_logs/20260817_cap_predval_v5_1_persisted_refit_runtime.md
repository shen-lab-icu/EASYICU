# CAP-PREDVAL-V5.1 persisted refit and runtime authority

Date: 2026-08-17
Task: `CAP-PREDVAL-V5.1`
Capability branch: `codex/cap-predval-v5-1-persisted-refit-runtime-20260817`
Capability worktree: `/Users/haibo/Documents/GitHub/.worktrees/easyicu-cap-predval-v1-20260817`
Base HEAD: `348368e5298e0be380dee782fede169be5c7f4cd`
Implementation HEAD: `9382bbb0948d6bbaa42b736ea66e267620b43408`

## Decision

Close the two explicit V5 gaps without widening authority: capture the current
clean local Git checkout and fitted package environment from the host itself,
persist those records beside one sealed fit, and later re-fit the estimator
from the current EvidenceStore records before accepting the analysis-only
validation receipt.

The public routes accept neither caller-attested Git or environment identity
nor loose persisted bytes, paths, receipts, bindings, claims or aliases. The
fit owner remains the sole numerical owner; the authority layer resolves and
validates persisted records and delegates deterministic re-fitting back to that
owner. The result remains experimental and analysis-only.

## Reproduced failing baseline

The V5.1 test was added before implementation and collected against the V5
base. Collection failed as expected:

```text
ModuleNotFoundError: No module named
'easyicu.research_agent.authority.prediction_model_fit_revalidation'
```

This established that V5 could materialize a sealed fit and recompute its
validation metrics, but had no owner that captured a live clean runtime or
re-fitted the model from persisted evidence after reload.

## Closed behavior

- Runtime capture derives the repository root from the installed owner module,
  requires a clean tracked and untracked Git status, and captures the exact
  commit and tree. Callers cannot submit a repository root, commit, tree,
  dirty flag or source digest.
- The environment lock is observed from the running interpreter and reconciled
  to the package versions sealed in the fit receipt before any runtime record
  is registered.
- Code snapshot, environment lock and runtime receipt are registered as one
  deterministic, closed three-role authority subset. Exact retry is
  idempotent; the route publishes no aliases or claims.
- The persisted source projection contains exactly the columns consumed by the
  fit contract, with the typed-input row identity bound into its receipt.
- Reload validation resolves all seven current EvidenceStore roles, verifies
  their canonical envelopes, digests, sizes and runtime identity, then asks the
  sole fit owner to reconstruct preprocessing, refit the estimator and compare
  the exact model artifact, prediction table and fit receipt.
- Current checkout drift, package drift, source mutation, model or prediction
  drift, stale evidence, dirty Git state and caller-supplied identity all fail
  closed with typed owner reason codes.
- The route writes nothing during revalidation and grants no Planner selection,
  production, publication or manuscript authority.

## Capability-branch verification

- New V5.1 runtime/persisted-refit suite: 9 passed.
- V5.1 + V5 + V4 vertical suite: 41 passed.
- Typed-input SDK, V1-V5.1 owners and bridges, inventory governance, package
  directions, module graph and static architecture: 152 passed in 30.85 s.
- Capability registry, scientific-action catalog, analysis-pattern auditor and
  dynamic prediction: 60 passed with one existing font warning.
- Capability inventory audit: OK.
- Research-agent module graph: 546 modules, 2,151 edges, 0 cyclic modules and
  0 cyclic strongly connected components.
- Targeted Ruff and `git diff --check`: passed.
- Real, unmocked clean-checkout smoke captured commit
  `9382bbb0948d6bbaa42b736ea66e267620b43408`, Git tree
  `09abbfe2fafe4c1d227330b9c05fa11ebc9b9901`, environment evidence SHA-256
  `1cb83c88a0b6487534a4696b7be1f4b275f46f0307ec876c6aa67872d855c414`
  and fit receipt
  `6bb5be450d0ef00edb0f9d5b5206ec77acc89d2de3034f5b7259e593acbe9917`.
  EvidenceStore reload returned the identical validation receipt with 8
  records, 0 aliases, 0 numeric claims and 0 scientific claims.
- The shared virtual environment is editable against the main checkout, so the
  smoke explicitly placed this worktree's `src` first on `PYTHONPATH`. Its
  temporary root was resolved to the canonical `/private/var/...` path because
  the typed-input filesystem authority correctly rejects the macOS `/var`
  symlink path.

## Development-branch integration rehearsal

The live development lane was read at clean, remote-synchronized HEAD
`b5fec0025e0ce0888e3076bd3391f48e95c07630` on
`integration/figure2-e1-h3-20260816`. It was not modified. A separate worktree
and branch, `codex/cap-predval-v5-1-integration-check-20260817`, were created at
that exact commit.

The complete capability chain (`381216f`, `dd7a369`, `f9b2c9d`, `4cc0a79`,
`5d26589`, `6db2cc4`, `348368e`, `9382bbb`) cherry-picked without conflicts.
The code-complete rehearsal HEAD was
`ed528d3aa1aed37524d9f5d04061b43d0ed5c37d`.

- Exact focused integration matrix: 152 passed in 28.34 s.
- Exact adjacent integration matrix: 60 passed in 24.77 s with one existing
  font warning.
- Capability inventory audit, targeted Ruff and `git diff --check`: passed.
- Combined module graph: 546 modules, 2,159 edges, 0 cyclic modules and
  0 cyclic strongly connected components.
- Real integration-checkout smoke captured clean commit
  `ed528d3aa1aed37524d9f5d04061b43d0ed5c37d`, Git tree
  `d1a1f57590acdaf9be550c2ba9ed122eaad6d199`, the same environment evidence
  SHA-256, and fit receipt
  `16fc06bdc4794fbca493ca907101edb006a19b41338767bf5baf60f52fc5a8e2`.
  Reload re-fitting was stable with 8 records and no aliases or claims.

## Merge boundary and remaining gates

The capability chain is code-compatible with the exact current development
branch and is ready for an isolated experimental review/cherry-pick. This task
did not mutate or merge the live development branch, which remains concurrently
owned by the Figure 2 lane.

No full exact-head CI was run. Under the current project policy, focused checks
remain the iteration gate; a full checkpoint belongs after E1 is 11/11 and the
combined HEAD is being frozen, merged, released or prepared for formal
experiments. The current result must not be described as stable or
formal-ready before that checkpoint succeeds.

The evidence remains synthetic and limited to one fixed L2 logistic estimator.
There is no external estimator oracle, real ICU dataset validation,
categorical-feature support, tuning, cross-validation, uncertainty analysis or
transport validation. The capability is not wired into Planner selection,
pipeline execution or manuscript production. Independent review and an
explicit development-lane merge decision remain required before changing the
live branch.
