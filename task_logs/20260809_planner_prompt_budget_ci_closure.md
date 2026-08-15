# Planner fixed-prompt budget CI closure

- Date: 2026-08-09
- Failing review baseline: `2c3d5222b0795e3274e2ccef3c8025ee67efd4f8`
- Implementation: `6fc003f`
- Task: `PLANNER-FIXED-PROMPT-BUDGET-CI`

## Failure

The completed research-agent Python 3.10 job on the review baseline failed one
ratchet test after 9,776 passes:

```text
the fixed Planner prompt is 51629 bytes, over the 51600 budget
```

The current branch reproduced the same failure locally. This was independent of
the SOFA-2 longitudinal fix.

## Closure

The optional E-value and subgroup-contract guidance was shortened without
removing a field, changing its typed contract, or raising the reviewed budget.
The fixed prompt now measures 51,589 bytes against the 51,600-byte ratchet.

## Verification

- Prompt-budget ratchet and recorded-context cases: `11 passed`.
- All tests directly referencing the Planner prompt builder, prompt metrics, or
  the two affected optional scientific contracts: `523 passed`.
- Focused E-value/parser/prompt subset: `62 passed`.
- Ruff and `git diff --check`: passed.

No Provider call, patient-data read, Canonical9 run, or budget increase was used.
