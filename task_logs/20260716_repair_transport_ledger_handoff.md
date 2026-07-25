# Repair transport ledger — Claude review handoff

Date: 2026-07-16  
Base: `main@4e13edc`  
Scope: Track A repair-control-plane hardening only; no E3 execution and no
scientific routing changes.

## What changed

- Bumped the single provider/repair receipt from schema v4 to v5.
- Each new logical repair entry is reserved as `transport.state=pending` before
  any provider call.
- `RepairCoordinator` now seals the same entry before returning:
  - success: transport mode, exact after-code SHA-256, provider-history length,
    provider-history digest, and actual provider-call count;
  - failure: exception type plus the same provider-history checkpoint (no error
    prose or model output is stored).
- All six pipeline-owned Coder repair routes pass the current logical attempt
  ID. Receipt-persistence errors are re-raised and cannot be demoted to an
  ordinary Coder failure or deterministic fallback.
- Resume behavior is explicit:
  - exact pending reservation with unchanged provider history: reuse the same
    logical attempt without appending another entry;
  - pending reservation after one or more paid provider calls: fail closed
    before another route can pay again or ignore it;
  - completed or failed transport: terminal and content-addressed.
- Legacy v3/v4 logical entries migrate one way to `legacy_untracked`; v1-v4
  receipts remain readable.
- The pre-step cohort-translation entry now restores the complete verified
  receipt state, rather than keeping categories while accidentally dropping
  logical-repair/final-audit state.
- The read-only performance harness validates schema v5 repair transport while
  continuing to reconstruct the old E3 schema-v2 baseline unchanged.

## Deliberately unchanged

- Planner/Coder still own exposure, outcome, cohort, method, model and estimand.
- No benchmark, KDIGO, H2/E2/E3, database or Figure-2-specific rule was added.
- No deterministic primary runner was enabled.
- Final deterministic gates and final-audit reservation semantics were not
  relaxed.
- E3 remains at the same failed Step 02; no API call or experiment rerun was
  made.

## Verification

- Provider/repair/preflight/performance focused suite: `377 passed` in 4.78 s.
- Resume/meta/characterization/post-repair/visual/execute-contract suite:
  `266 passed` in 190.52 s.
- Combined focused evidence for this batch: `643 passed`, 0 failed.
- Ruff: passed.
- Black check (scoped files): passed.
- `py_compile`: passed.
- `git diff --check`: passed.
- Old E3 performance baseline remains exactly:
  - 15 total calls = 12 step-scoped + 3 planner;
  - 7 repair calls;
  - 366,592 tokens;
  - Step 02 active wall 373.5 s;
  - Step 02 sandbox compute 1.719 s.

## Review focus / remaining boundary

Please review in particular:

1. schema-v5 migration and strict transport-field validation;
2. unpaid-pending exact replay versus paid-pending fail-close behavior;
3. whether any of the six `ProviderCallBudgetReceiptError` paths can still be
   swallowed by an outer fallback;
4. the cohort-translation full-state restoration path;
5. whether persisting only the after-code digest (not the script body) is the
   right boundary before StepAuthorityCapsule.

This batch does **not** claim full recovery when a process crashes after the
transport result is sealed but before the returned code is stored in a durable
step capsule. It prevents ambiguous duplicate payment and records the exact
result digest; durable candidate-code recovery is the next StepAuthorityCapsule
batch and should not be folded into this review commit.

