# Lossy numeric coercion deterministic continuation

Date: 2026-07-16

Branch: `refactor/agent-control-plane`
Base: `04014d6`

## Goal

Allow a typed `LOSSY_NUMERIC_COERCION` finding to receive one narrow,
science-neutral fail-closed guard without spending another Coder repair call.
This is required for exhausted logical-repair receipts to make safe progress
without resetting or extending their monotonic provider-call ledger.

## Scope

- Added a deterministic repair that applies only when the typed repair reason is
  `LOSSY_NUMERIC_COERCION`.
- The repair requires exactly one unambiguous dict audit assignment containing
  the host-standard `newly_invalid_or_coerced_n` computation shape
  (`notna() & isna()` followed by `sum()`).
- It inserts only a fail-closed `> 0` guard after that existing audit record.
- Human-facing finding text alone cannot route the repair.
- Multiple candidate audit sites fail closed with no mutation.
- Fractional/normalized values, duplicate count keys, mapping unpack, shadowed
  `int`, same-line statements, and guards inside failure-suppressing `try` or
  `with` scopes are ineligible.
- The preflight now binds coercions, loss counts, and guards by lexical scope,
  definition site, statement block, and straight-line dominance. The only
  cross-function proof is a closed direct return/call receipt path whose every
  call is immediately guarded.
- `lossy_ordinal_rounding` now has a distinct typed repair reason and cannot
  authorize this numeric-coercion repair.
- Both pre-execution and post-mutation deterministic repair paths now forward
  typed repair reasons.

## Ownership boundary

The repair does not select or change the exposure, outcome, cohort, method,
estimand, source column, coercion policy, ordinal domain, or missing-data
strategy. It only enforces an audit count that the candidate code already chose
to compute. No benchmark-, database-, or manuscript-item token was added to
production code.

## Exact prior-run verification

The quarantined Step 02 candidate from the existing E3 diagnostic run was loaded
read-only and repaired in memory. The change was exactly three lines; the
`lossy_numeric_coercion` preflight finding disappeared. The existing provider
receipt remains unchanged at 6/7 calls and retains its final-audit allowance.
The historical run directory was not modified.

## Verification

- Focused typed repair, context/preflight, scope/dominance, repair-reason, and
  meta probes: `374 passed`.
- Deferred audit/runtime/capsule integration (outer sandbox): `7 passed`.
- Receipt/quarantine/capsule resume selection (outer sandbox): `27 passed`.
- Evidence/resume characterization baseline: `48 passed`.
- The integration regression sets `max_step_llm_repair_attempts=0` and proves:
  one deterministic repair, zero Coder repair calls, one final concept audit,
  and an `ok` step.
- `git diff --check`: clean.

E3 was deliberately not resumed in this batch. It remains gated on the wider
resume regression and the P0-5 performance hard gate.
