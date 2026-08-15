# Agent endpoint contract fail-closed repair

Date: 2026-08-05
Task: `AGENT-ENDPOINT-CONTRACT-FAILCLOSED-20260805`
Status: code repair and focused verification complete; no Provider call or benchmark experiment launched.

## Reproduced failure

For analysis families whose registry requires a typed endpoint, a missing
`AnalysisPlan.endpoint` remained a warning after planning. The pipeline could
therefore continue into scientific steps after both the Planner and Replanner
failed to declare follow-up time, time origin, censoring, and event semantics.

The red-first test set failed at three boundaries before the repair:

- the endpoint validator had no final-gate error mode;
- a Replanner candidate without the required endpoint was accepted;
- the real pipeline did not block scientific execution after the retry miss.

## Repair

- Keep the plan-phase endpoint finding at warning severity so the initial miss
  remains repairable.
- Reuse the same endpoint rule at the Replanner candidate gate with error
  severity; no duplicate endpoint policy was added.
- At execute preflight, send a directed repair instruction that forbids
  inference from question prose, column names, dtypes, or step prose.
- Re-evaluate the contract after that retry. If it still fails, record an
  `endpoint_contract` error with `stage=execute_final` and
  `reason=endpoint_retry_exhausted`, mark the run non-ready, and execute no
  scientific plan step.
- Emit the stable audit reason `endpoint_contract_blocked` so the empty
  execution is attributable to the endpoint owner boundary.

## Verification

- Endpoint contract and real pipeline test file: `29 passed`.
- Typed DAG, cohort contract, replan budget, and empty-execution neighbors:
  `41 passed, 1 deselected` (the deselected test reads a mutable recorded-run
  corpus and fails identically on the untouched branch).
- Targeted pipeline retry/fallback selection: `4 passed`.
- The first three tests of the heavyweight 274-test pipeline file passed in
  140 seconds before the disproportionate full run was stopped; no failure was
  observed.
- Ruff, `git diff --check`, and Python compilation of changed production files:
  passed.

## Scope boundary

The repair is commit `18d7063` on branch
`codex/agent-endpoint-contract-fix-20260805`. It does not alter endpoint
inference, benchmark cases, global prompts, Provider behavior, or Claude's
concurrent Copilot/WebApp changes in the primary worktree.
