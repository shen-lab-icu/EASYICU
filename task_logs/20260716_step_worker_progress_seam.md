# A2 batch 4b — StepWorkerProgress seam

Date: 2026-07-16

Base: `refactor/agent-control-plane@2a148dd`

Scope: behavior-preserving consolidation of per-worker scratch progress only.
E3 was not resumed. No provider call, prompt, scientific routing, validator,
checkpoint, capsule, evidence, cohort, exposure, outcome, method, estimand, or
publication policy changed.

## Outcome

`step_worker_state.py` now owns 15 transient counters and provenance labels for
exactly one `_execute_one_step` worker:

- resumed-code and Critic-repair flags;
- deterministic fallback/standard-executor and runner-repair labels;
- concept-audit and deterministic-concept-repair counters/names;
- LLM mutation provenance; and
- aggregate, contract, visual, and runtime repair attempt counters.

The state is constructed inside `_execute_one_step`, so parallel workers never
share it. It owns no code, output, quarantine, digest, checkpoint, capsule,
provider receipt, EvidenceStore, plan, context, or scientific authority.
`repair_attempts` remains a scratch output-mutation counter; it is not the
provider-call ledger or `StepRepairBudget`.

The old nested `_script_generation_mode` helper moved to the data-only state's
pure `generation_mode()` projection. Its priority remains exactly:

`deterministic standard > LLM repair > deterministic fallback > runner repair
> any code/concept repair > resumed reuse > initial LLM`.

The deterministic-standard terminal branch can project
`llm_repair_used=False` without mutating the stored flag. The two old phase
handoffs are kept in their original positions: Critic repair becomes later LLM
mutation provenance only at the concept-loop boundary, and a pre-execution
runner repair becomes the current runner-repair label only after that loop.

## Structural delta

AST/symbol-table measurements against `2a148dd`:

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `_execute_one_step` stored locals | 374 | 359 | -15 |
| `_execute_one_step` free vars | 62 | 61 | -1 |
| targeted scratch `nonlocal` declarations | 9 | 0 | -9 |
| local generation-mode implementations | 1 | 0 | -1 |

`pipeline_execute.py` is 14,549 lines after formatting; the new data-only owner
is 53 lines. This is batch 4b worker-progress consolidation, not a claim that
the remaining control-plane extraction or Track A performance gate is complete.

## Verification

- New worker-progress unit/static tests: `5 passed`, including exhaustive
  legacy/new generation-mode equivalence over all boolean priority combinations.
- Authority/capsule/characterization/meta shard: `118 passed`.
- Broader execute-contract/deferred-repair/resume/schema/trajectory/visual
  shard: `212 passed`, `1 deselected`, `1 failed` in 151.41 s. The sole failure
  (`test_visual_repair_log_keeps_structured_collision_detail`) was reproduced
  unchanged on clean `2a148dd`; it is existing figure-policy test debt.
- The separately observed
  `test_contract_budget_does_not_consume_visual_layout_budget` failure was also
  reproduced unchanged on clean `2a148dd`; it is not part of this diff.
- Focused pipeline fallback/runtime/trajectory/capsule-resume cases passed;
  nested runner cases were executed in the canonical outer environment.
- No target scratch name remains as a bare local or `nonlocal`; JSON keys and
  call keyword labels were not rewritten.
- Black, Ruff, `py_compile`, and `git diff --check`: passed.
- Independent read-only adversarial review: ACCEPT; AST normalization found no
  control-flow, keyword-label, string-key, or authority-boundary drift.

## Honest boundary / next action

This is a small reviewable seam for Claude to audit. It intentionally leaves
`code`, `run_result`, quarantine state, approved digests, sealed-renderer
authorization, `step_record`, `StepAttemptState`, checkpoint/capsule selectors,
provider receipts, evidence aliases, and run-level state in their existing
owners. Track A still needs the remaining worker-state/control extraction and
the P0-5 performance milestone before the same E3 run may resume from Step 02.
