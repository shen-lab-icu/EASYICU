# E3 primary cohort raw-universe resume

Date: 2026-07-15

## Scope

Fix the case-neutral execution boundary for the plan's unique primary producer of
both `artifact:analysis_cohort` and an attrition/flow table. The Agent continues
to own the cohort predicates and generated code; the host only supplies the raw
run universe to that one producer, replays the Planner-locked predicates, and
validates the resulting cohort and accounting before sealing evidence.

## Implementation

- Commits: `b7c205c fix(agent): bind primary cohort attrition to raw universe`
  and `3095315 fix(agent): fail close cohort authority aliases`.
- The raw-universe binding requires an exact cohort-definition method family,
  closed cohort/attrition products, and a unique analysis-cohort producer.
- The host verifies ordered row identity, all authoritative cohort values,
  sequential or partition-style attrition accounting, summary denominators,
  and the raw/filtered input SHA values before and after execution.
- Input mutation fails closed and suppresses all remaining steps for the run.
- Canonical attrition rule IDs bind ordered rows to Planner predicates; every
  simultaneously declared count, denominator, identity, or partition-role alias
  must agree. Unsafe execution paths verify host input SHA values before return.
- No E2/E3/H2/H3, database, clinical-variable, or nine-task routing was added to
  production code; no deterministic runner selects cohort science.

## Verification

- Focused final suite: `86 passed`.
- Resume/meta suite: `179 passed` in 130.05 seconds.
- Final combined cohort/pipeline/terminal/resume/meta suite after adversarial
  alias hardening: `200 passed` in 130.11 seconds.
- Black, Ruff, `py_compile`, and `git diff --check`: all passed.
- The separate characterization-test files already present in the worktree were
  not included in this commit.

## Real E3 checkpoint replay

Run:

`research_output/_diagnostic_e3_8317_resume_fastfix3_20260715/E3_kdigo_gradient/aware/run_20260715T151048_e98a01`

Only `01_cohort_flow` was resumed, using the prior Agent-generated script and
stopping immediately after that step.

- latest outer step status: `ok`
- generation mode: `resumed_code_reuse`
- code SHA: `3222fdc2c3099b2b3591f7a971b7147a868dca9e2a2299ef02f8c37cf36b7bb1`
- raw universe: 94,458 rows
- final authoritative/produced cohort: 74,708 rows
- sequential remaining counts: 94,458 → 94,458 → 74,829 → 74,708 → 74,708
- exclusions at each row: 0 / 0 / 19,629 / 121 / 0
- output `stay_id` order is exact, unique, and identical to the host-authoritative
  filtered cohort; every host-authoritative value column is unchanged
- the only additional output column is the Agent-derived `kdigo_stage`
- code repair attempts: 0; concept repair attempts: 0; full rewrite: 0
- statistical, clinical, guard, product-contract, and figure-source findings: 0

The one provider call made during this resume was a concept re-audit under the
new raw-universe authority binding, not a code generation or repair call. The
run-level manuscript-skipped ERROR is expected because this diagnostic command
used `--stop-after-step-id 01_cohort_flow`; it is not a Step 01 gate failure.

## Next

Freeze shared engine at `3095315` during development execution. Resume E2 and E3
from their first incomplete steps in separate processes with BLAS threads
limited to one; add H3 only if memory remains safe. Do not edit engine source
while benchmark processes are active.
