# SOFA-2 owner-receipt LOCF closure

- Date: 2026-08-09
- Review baseline: `2c3d5222b0795e3274e2ccef3c8025ee67efd4f8`
- Implementation: `d864e43`
- Task: `SOFA2-RECEIPT-LOCF-V6`

## Finding

The six component owners correctly issued `observed` and `available` receipts,
but the aggregate still allowed a missing-as-normal synthetic zero to enter the
longitudinal state before LOCF. An auxiliary-only event could therefore replace
a prior valid component score with zero and truncate the intended 24-hour
carry-forward.

## Closure

- When an owner-issued `*_available` receipt exists, the aggregate now masks a
  component value unless that receipt says the value is available. The masking
  happens before gap filling and LOCF.
- A prior valid component state can therefore carry across an auxiliary-only
  event, while `observed` remains false at the later time point.
- A first-day component with no valid history still contributes zero through the
  existing missing-as-normal sum policy.
- Legacy score-only aggregate inputs remain an explicit caller assertion and
  retain their pre-receipt compatibility behavior.

## Regression evidence

- Red phase: the new longitudinal and legacy-compatibility tests both failed on
  the review baseline (`2 failed`).
- New regression tests after the fix: `2 passed`.
- SOFA-2 callback plus clinical-contract files: `61 passed`.
- Focused clinical/API/catalog/static-route gate: `137 passed, 70 deselected`.
- Clinical conformance marker: `57 passed, 12771 deselected`.
- Ruff, `git diff --check`, and static catalog synchronization: passed.

## Non-claims

This closure did not run a Provider, patient data, Canonical9, or a six-database
evidence validation. The six-database evidence state remains `mapping_only`, and
paper authority remains frozen at 4/9.
