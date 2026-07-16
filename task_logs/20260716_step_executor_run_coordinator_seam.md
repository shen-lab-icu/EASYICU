# A2 batch 4a — StepExecutor / RunCoordinator seams

Date: 2026-07-16

Base: `refactor/agent-control-plane@7fd8cbd`

Scope: behavior-preserving ownership seams only. No scientific routing,
validator, evidence, repair, prompt, cohort, exposure, outcome, method, or
estimand rule changed. E3 was not resumed.

## Changes

### StepExecutor

`step_execution.py` now owns the smallest mechanical sandbox boundary:

1. if the selected runner does not own output cleanup, clear the exact
   host-selected output directory;
2. call `runner.run(step_id, code, resolved_inputs_path)` exactly once;
3. return the runner result unchanged and propagate cleanup/runner exceptions
   unchanged.

Runner selection, timeout choice, runner identity, execution-context digest,
capsule replay, input-authority verification, output-safety handling, capsule
sealing, validation, repair, and evidence publication remain in their existing
owners. Exact capsule replay bypasses StepExecutor, preserving zero sandbox
execution for unchanged sealed code.

### RunCoordinator

`run_coordination.py` now owns only step-queue mechanics:

- sequential queue advancement and executed-step bookkeeping;
- application of a caller-authorized stop or replan transition;
- removal of the current failed step before a directed retry;
- parallel worker submission and forwarding of worker exceptions to the
  caller-owned finding handler.

The caller remains the only current-plan authority. `RunExecutionState` stores
the queue and executed ids but deliberately does not store a second plan copy.
All directed/ordinary replan decisions, viability checks, typed/trajectory DAG
rules, stop/corruption priority, and progress messages remain in
`pipeline_execute.py`; it returns a typed transition to the coordinator. Plan
replacement is applied through a callback that updates the original `plan`,
`step_order`, and `total_steps` before rebuilding the queue.

## Locked ordering

The sequential path remains:

`execute/checkpoint → mark executed → input-authority corruption → explicit
stop → directed replan → successful-step replan`.

The parallel path still uses `copy_context()` and converts worker exceptions to
the existing `step_executor` finding. Sequential runner exceptions still
propagate. Docker-owned cleanup remains inside DockerRunner, so an unconfirmed
teardown sentinel cannot be bypassed by host cleanup.

## Verification

- new seam + execute-contract tests: `76 passed`;
- capsule/runtime-repair/replan/cohort runner-backed tests in the canonical
  outer environment: `26 passed`;
- Docker runner tests: `32 passed`;
- host runner tests in the canonical outer environment: `25 passed`;
- the broader 143-test control/characterization/meta group produced four known
  nested-`sandbox-exec` false failures; all four passed when rerun outside the
  nested sandbox, while the other `139 passed` in-place;
- meta-generalization and all five characterization files stayed green;
- Ruff, Black, and diff-check are required before commit.

## Honest status

This is **batch 4a**, not completion of A2 batch 4. It establishes real,
independently testable owners but does not yet shrink `_execute_one_step` or
collect its many mutable closure locals into an explicit worker state. The next
sub-batch must move a substantial stateful worker boundary without introducing
a second plan/evidence authority. P0-5 performance hard gates are also still
pending and must not be inferred from this structural commit.
