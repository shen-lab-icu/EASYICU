# Canonical9 + Extension3 parallel coverage plan

> Task ID: `FIG2-EXTENSION3-COVERAGE`
> Date: 2026-07-16
> Scope: benchmark specification and scheduling only; no shared-engine change.

## Decision

The ICU research capability programme now contains twelve development
questions in two explicitly separated layers:

1. **Canonical9**: the existing E1--E3, M1--M3 and H1--H3 suite.  Its current
   development score remains 6/9.  Its questions, runs and paper-facing
   protocol are not changed by this plan.
2. **Extension3**: three newly specified method-coverage probes:
   `M4_crossdb_prediction_validation`,
   `H4_ventilation_competing_multistate`, and
   `H5_dynamic_rrt_target_trial`.

The difficulty labels are M/H/H because external validation adds one standard
design axis, whereas competing/multi-state outcomes and dynamic treatment with
time-varying confounding require interacting temporal and causal contracts.

## Why Extension3 is not called held-out

The user intends to use these questions to expose and fix framework gaps.  Once
their results can influence the engine, they are development probes by
definition.  Calling them held-out would overstate generalisation and create a
review vulnerability.  The existing requirement for a separate, untouched
3--6 item held-out set remains unchanged.

## Schedule

1. Finish and close the current Canonical9 development runs.
2. Freeze the Canonical9 engine/configuration and start the full
   discovery-to-manuscript idea-mining run for one or two exploratory examples.
3. In parallel with idea-mining, materialise Extension3 inputs and execute the
   three tasks in a separate development lane.
4. Extension3 may motivate only case-neutral invariants.  Case-specific
   exposures, outcomes, task IDs, answers and routes stay in the item/rubric,
   never in shared prompts or the shared engine.
5. Preserve an explicit version boundary: an Extension3-driven engine update
   does not silently change or relabel frozen Canonical9 artifacts.

## Paper placement

The primary Figure 2 protocol remains Canonical9 until Extension3 has real,
consistently scored artifacts.  The default publication plan is to report
Extension3 in Extended Data as coverage expansion.  Promoting Figure 2 from 9
to 12 tasks requires a new frozen protocol version and fresh, consistently
scored runs; the task count must not be changed after seeing favourable or
unfavourable results.

## Artifacts

- `benchmarks/extension3/README.md`
- `benchmarks/extension3/extension3_benchmark.jsonl`
- `tests/research_agent/test_extension3_benchmark_spec.py`

No file under `src/easyicu/research_agent/` is changed by this task.
