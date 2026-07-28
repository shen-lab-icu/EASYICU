# E1 scientific closure and architecture handoff — 2026-07-28

## Status

- Branch: `fix/external-review-20260724-p0-p1`
- Code endpoint: `d7fe639`
- Old E1 with 13 resumes remains `diagnostic_only`; it must not be resumed or promoted.
- Figure 2 paper-facing score remains 0/9. The frozen scorer digest is intentionally not refreshed during development.
- No Provider call was made while implementing or verifying this repair series.

## Closed findings

The repair series converts the latest E1 audit findings into owner-module contracts:

1. Event missingness distinguishes measurement absence, event absence and conditional non-applicability.
2. Stay counts are no longer labelled as patient counts when subject identity is unavailable.
3. SOFA first-time derivatives carry time units; representation rules no longer contradict generated means.
4. Statsmodels Wald intervals and underflowed P values are labelled accurately.
5. Critic review receives a bounded structured scientific summary instead of relying on file presence.
6. Provider costs aggregate across resumes and separate reported use from unknown/reserved upper bounds.
7. Durable heartbeats expose phase and step progress.
8. Finalized attempt history is externalized and the manifest points to bounded history records.
9. Evidence aliases and manifests bind the current plan authority.
10. E1 protocol and acceptance enforce Table 1 SMD, typed event semantics, typed cohort consumption, invalid event-time review, landmark handling, readmission sensitivity and nonlinear sensitivity.
11. Planner-owned exposure labels replace generic `Category 0/1` output.
12. Step 06 deterministic robustness execution and its plausibility receipt share one immutable obligation scope and can pass without a Provider call.

## Architecture boundary closure

Commit `d7fe639` moved policy decisions back to owner modules:

- Critic semantics: `research_agent/review/step_semantics.py`
- Plan manifest projection: `research_agent/authority/plan_input_closure.py`
- Runner timeout: `StepExecutor.runner_timeout`
- Repair preflight composition: `research_agent/repairs/preflight.py`
- Cohort display semantics: `research_context/cohort_granularity.py`

No architecture baseline was refreshed. The existing architecture guard stayed green, as did module graph, seven Import Linter contracts, Deptry, Ruff and diff-check.

## Verification

- 158 targeted research-agent tests passed.
- 22 focused owner-contract tests passed.
- 4 exact resume regressions passed.
- A broader resume/provider batch reached 89 passed before being stopped because it had expanded into repetitive coverage; no failure occurred before the stop.
- Earlier focused validation included 201 Step 06 tests, 148 formal safety/acceptance/plausibility tests, 263 Table 1 tests and 35 E1 acceptance/figure tests.

## Fresh E1 policy

1. Use a clean detached worktree at `d7fe639`.
2. Materialize E1 into a new external directory; do not mutate or reuse the earlier `a3d8508` materialization.
3. Run zero-Provider materialization, planner and runtime preflight first.
4. Build an exact-SHA Docker image and require source identity match plus `network=none`.
5. Run only the aware arm with `adaptive_v1` against the local loopback Provider.
6. Treat the first fresh run as a development canary. Do not refresh the formal Figure 2 freeze or start E2–E9 unless E1 passes the complete execution and scientific closure.
7. On a framework failure, stop, assign it to one owner module, repair with focused tests, create a new SHA/image and start a fresh run. Do not spend repair calls repeatedly inside a contaminated run.

## Intentionally deferred

- Exact actual cost for the interrupted historical call cannot be reconstructed; reports must preserve it as unknown and show a conservative upper bound separately.
- Historical duplicate artifacts are not deleted. Storage deduplication is a separate migration and must not mutate evidence-of-record.
- Submission-grade modelling choices and figure polish are validated through the new E1 sensitivity/acceptance contracts; the old E1 figures remain diagnostic.
