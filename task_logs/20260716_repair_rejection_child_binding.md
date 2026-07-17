# Exact-child binding for repair rejection rollback

Date: 2026-07-16

Branch: `refactor/agent-control-plane`

Base: `6a2eb92`

## Finding

`CheckpointAuthority` bound a completed repair rejection latch to the exact
parent capsule and repaired-code digest, but not to the child capsule selected
by that repair. If another checkpoint advanced `current_capsule_ref` before a
late rejection callback arrived, the stale callback could restore the older
parent and roll back the newer selected authority.

## Fix

`StepAttemptState` now records `last_completed_repair_child_ref` only after the
repair child checkpoint succeeds. Rejection restores the parent only when all
three coordinates still match:

1. the rejected code digest is the completed repair digest;
2. the current selector is exactly that repair child; and
3. the exact parent latch is still present.

A successful parent restoration clears parent, child, and code latches
together. Checkpoint failure preserves the child selector and every latch, so a
later safe retry remains possible. No plan, scientific method, validator,
provider budget, evidence, or alias policy changed.

## Verification

- Red-first adversarial test reproduced the stale rollback (`newer → parent`).
- The same test now preserves the newer capsule and leaves the old latch inert.
- Step-attempt authority, capsule integration, and deferred runtime-repair
  suites: 29 passed in the canonical outer environment.
- Black, Ruff, `py_compile`, and `git diff --check`: required before commit.

E3 was not run or modified.
