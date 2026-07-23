# Canonical9 E1 secondary-only model-contract closure

Date: 2026-07-23  
Task: `FIG2-CANONICAL9-REALRUN`  
Scope: MIMIC-IV full0717-v2, aware arm, local Luna Provider

## Honest status

- Paper-facing Canonical9 remains **0/9**. Neither diagnostic batch is paper
  authority.
- Batch
  `batch_20260723_luna_miiv_adaptive_12d158c/e1_sepsis3_prevalence_mortality`
  completed Steps 01--04. Step 05 executed successfully in the immutable
  Docker runtime, then a post-execution model-contract repair was blocked
  before evidence registration. Step 06 was dependency-blocked.
- The batch process was stopped before spending Provider calls on E2.
- Cross-database extraction remained paused to avoid memory/swap contention.

## Reproduced root cause

The archived Step 05 script returned code 0 in 3.559 seconds on the exact
94,458-stay typed cohort and produced the coefficient, estimate, flow, and
summary files. Replaying that script in the same source-bound image reproduced
the result.

The host validators then reported two framework-contract issues:

1. the Planner-owned typed roster intentionally contained one required
   **secondary** operational representation because the context-declared
   primary exposure was absent from the sealed physical columns, while the
   validator still demanded exactly one primary model;
2. the sklearn ridge fallback computed a real convergence boolean from
   `n_iter_`, but did not publish the controlled
   `convergence_method=optimizer_success` and `optimizer_success` aliases.

The subsequent Luna contract repair changed only the host-helper call
signature, adding unsupported `frame=` and `value_column=` keywords. Mechanical
preflight correctly quarantined it. The failure was therefore an internal
contract contradiction plus a missing deterministic metadata projection, not
a failed statistical fit.

## General fix

- A non-empty Planner-owned `model_requirements` roster with no primary entry
  may now remain secondary-only. Such a roster still rejects any generated
  primary model and never relabels an operational proxy as the context primary.
- Contract-error guidance explicitly preserves a secondary-only roster instead
  of asking the Coder to invent a primary exposure.
- Added `penalized_convergence_contract_v1`, an automatic structural repair
  that targets exactly one model named by
  `penalized_convergence_not_verified` and copies an already-computed
  convergence boolean into the controlled contract fields. It does not refit,
  choose a method, alter rows, or introduce a result.
- Updated the stale duplicate wiring assertion to the existing
  `integrity_universe_path` authority introduced by `43c995d`; the production
  wiring itself was not changed.

## Verification

- Focused model/repair/registry matrix: **127 passed**.
- Expanded contract/repair/budget matrix: **397 passed**; five source-bound
  Docker tests failed closed before the new image was built because the old
  image source SHA did not match the modified checkout. This is expected
  release behavior, not counted as a pass.
- Exact archived Step 05, deterministically patched and replayed with
  `--network=none`: return code 0.
- Replayed `step_summary_integrity`: 0 findings.
- Replayed `primary_model_contract`: 0 findings.
- Ruff, `compileall`, `git diff --check`, architecture measurement, resource
  baseline, and zero-cycle module graph pass after the honest baseline refresh.

## Next authority boundary

Build a source-bound immutable image from the clean commit, rerun the five
source-bound tests against it, create a fresh Canonical9 authority directory,
and start a fresh E1. The prior batch cannot be resumed for paper authority
after a framework-code change.
