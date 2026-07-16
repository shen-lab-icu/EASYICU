# Attempt-owned repair provider accounting — review handoff

Date: 2026-07-16
Base: `main@571690a`
Branch: `codex/agent-step-authority-capsule`
Scope: Claude review findings F1–F3 only; no StepAuthorityCapsule and no E3 run.

## Why this patch exists

Schema-v5 repair receipts originally treated any provider-history growth after
a logical repair reservation as proof that the repair itself had paid for a
provider call. An unrelated `concept_audit` could therefore turn an unpaid
pending repair into a false paid-pending integrity error. The direction stayed
fail-closed, but the error was false and could escape from availability probes
outside the normal structured step-failure path.

The same review found two smaller receipt-validation gaps:

- a `pending` logical repair was not required to be the final ledger entry;
- transport validation ran before the reservation history type was validated,
  producing a misleading transport-history error for a malformed reservation.

## Design

- The top-level provider receipt remains schema v5. No second ledger was
  introduced.
- The nested `RepairAuthorityBinding` is now v2 and binds the exact host-owned
  `provider_category` used by its `RepairCoordinator`.
- Attempt-owned calls are exact matches only:
  - `<provider_category>_patch`
  - `<provider_category>_full_rewrite`
- Pending paid detection and terminal `provider_calls` use the same helper;
  repeated transport retries are counted individually.
- Before any provider call, `RepairCoordinator` checks that its actual category
  equals the category bound to the current pending logical attempt. A mismatch
  fails closed with zero provider calls.
- Legacy/v1 bindings remain readable and deliberately keep the conservative
  rule that any provider call after reservation makes a pending attempt
  ambiguous.
- Binding v2 without `provider_category`, and legacy bindings that claim the v2
  field, are rejected before a receipt is persisted.
- A pending logical repair must be the final ledger entry.
- Core logical-repair history is validated before its transport checkpoint.
- The read-only performance harness mirrors the same v1/v2 accounting rules.

## Explicit compatibility decision

An old unpaid pending v1 binding is not silently upgraded to v2. Reconstructing
a new v2 binding changes the authority digest, so exact resume fails closed
unless the old binding is replayed unchanged. This is conservative and avoids
inventing transport ownership for historical receipts. The current E3 run uses
an older receipt schema and is not blocked by this boundary.

## Verification

- Provider/repair/performance focused suite: **79 passed**.
- Resume/meta/characterization/execute-contract/visual suite: **327 passed**.
- Combined scoped verification: **406 passed, 0 failed**.
- Ruff: passed.
- Black check on changed production/tool files: passed.
- `py_compile`: passed.
- `git diff --check`: passed.
- No LLM/API call and no E3 resume was performed.

## Next batch (not included)

Implement a content-addressed `StepAuthorityCapsule` as recovery glue, not as a
new current-evidence authority. It should persist verified step context, typed
bindings, candidate-code blobs and repair-ticket identity, with the monotonic
run checkpoint as the only current reference. The first acceptance target is
zero repeated initial generation, concept audit and sandbox execution for an
exact failed-step resume.
