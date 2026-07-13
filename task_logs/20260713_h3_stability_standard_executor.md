# FIG2-CANONICAL9-GATE — H3 typed stability standard executor

Date: 2026-07-13  
Module: benchmark实验  
Phase: development stress completion  
Branch: `main`  
Implementation commit: `061f697` (with prerequisite commits `80be2d8`, `82b9f4f`)

## Objective

Replace repeated generation and auditing of a large clustering-refit script with a general standard-method boundary: the Planner owns every scientific choice in a typed `TrajectoryStabilitySpec`; the deterministic executor only computes that frozen workload. Do not add an H3-specific runner, do not select k/threshold/seed in shared code, and do not execute Step 05 until the stability gate passes.

## Architecture outcome

- Planner-owned typed spec with exact method and closed typed inputs/outputs.
- Generic observed-data diagonal-GMM refit implementation; executor cannot receive outcome bindings and cannot change k, resampling design, threshold, or failure policy.
- Standard executions use an independent bounded timeout (default 3600 s); ordinary generated code remains at 300 s.
- Assignment replay is streamed to a private pending file and atomically published only after successful refits; private files are denied at evidence enumeration and registration.
- Docker timeout handling now uses cidfile plus a unique name, confirmed teardown, a read-only run/step/script mount with only `outputs/` writable, hostile symlink/hardlink rejection, and an immediate pipeline terminal when outputs cannot be proven quiescent.
- Large CSV multiple-testing discovery performs a semantics-preserving mmap preflight before row materialisation when no p-value column exists.

## Verification before the real run

- Claude F1/F2/meta redline suite: `94 passed`.
- Docker/runner and hostile-path tests: `30 passed`.
- New mmap, ExperimentSpec/bench timeout, trajectory executor/planning tests: `36 passed`.
- Pipeline standard success/terminal tests: all 10 green (9 passed in the combined run; the one test whose probe initially covered safe upstream steps passed after narrowing the probe to the unsafe target step).
- Independent adversarial review found no remaining blocker after the final patch.
- Ruff, `py_compile`, and `git diff --check`: passed.

## Real H3 resume

Run directory:

`research_output/_diagnostic_h3_fresh_20260712_v2/bench_h3_gpt-56-luna/H3_trajectory_clustering/aware/run_20260712T220834_314aaa`

Execution boundary:

- aware arm only;
- resumed from and stopped after `04_latent_class_stability_freeze`;
- reused saved `analysis_plan_revision_6.json` (no new Planner/Coder call);
- ordinary timeout 300 s; standard-executor timeout 3600 s;
- effective isolation `macos_sandbox_exec`, network denied;
- runtime 379.813 s, return code 0, no timeout.

Planner-owned design:

- selected k = 6;
- 100 resamples, 80% without replacement (`n=72,027` each);
- base seed 20250308 with frozen SeedSequence derivation;
- observed-data EM diagonal Gaussian mixture;
- mean adjusted Rand index threshold = 0.70;
- threshold failure action = fail closed and require a new Planner revision.

## Result

- Refits: 100 successful, 0 failed.
- Mean adjusted Rand index: **0.5357391480440364**.
- Threshold: 0.70; result: **failed**.
- Executor summary: `status=failed_closed`, `freeze_status=not_frozen_stability_threshold_failed`.
- Outer step record: `status=deterministic_standard_blocked`, `standard_executor_terminal_reason=executor_reported_failed_closed`.
- Candidate solution was not changed: k remained 6 and all 90,034 `(stay_id, cluster_label_k6)` assignments exactly matched the published `cluster_assignments.csv` row for row.
- Published replay sizes: 100 rows in `cluster_stability.csv`, 7,202,700 rows in `cluster_stability_assignments.csv`, and 90,034 rows in `cluster_assignment_provenance.csv` (counts exclude headers).
- No private pending file remained. No Step 05 directory or execution record exists.
- The current failed outputs are diagnostic-only. The new summary evidence `statistic_step_summary_c9a06d43` registered zero NumericClaims; five claims from an older superseded failed attempt remain historical in the append-only store and are not current reporting authority.

## Interpretation and next action

The framework defect is closed: the agent-owned scientific protocol executed quickly, reproducibly, and with a truthful fail-closed boundary. H3 itself is not reportable because the selected six-class solution did not meet the Planner's own stability criterion. This must not be "fixed" by hand-changing k, threshold, seed, or allowing outcome characterization. A retry requires a new explicit Planner revision.

Development status therefore remains 6/9. The next real execution target is E2, followed by the remaining E3 steps; H3 remains scientifically blocked at Step 04.
