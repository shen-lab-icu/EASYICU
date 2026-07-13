# Figure 2 agent hardening consolidation — 2026-07-12

## Scope

The user asked to stop managing the Figure 2 work across separate worktrees: commit the stable Claude Web/Copilot changes, merge the active work into local `main`, and continue the H3 development run from the main workspace. No remote branch was pushed or deleted.

## Consolidated commits

- `8cd951c fix(web): preserve truthful guided workflow state`
  - committed the stable Claude Web/Copilot patch;
  - added an explicit extraction-continuity `isRunning()` state so terminal `done`/`failed`/`cancelled` metadata is not presented as a live job;
  - Web verification: 79 focused tests passed, plus JavaScript syntax and diff checks.
- `587ddb3 fix(agent): align mock pipeline with typed products`
  - restored typed `output_files`, render-only figure children, and exact protocol registration in the deterministic MockLLM pipeline.
- `6e71a38 fix(agent): freeze trajectory stability role inputs`
  - made trajectory typed-product ownership unique at plan level;
  - migrated legacy split plans so stability consumes the frozen selection manifest instead of repeating selection/characterization outputs;
  - constrained stability to the frozen population, method, k, coordinates, and assignments.
- `6de9070 merge: integrate Figure 2 agent hardening`
  - local `main` now contains both the Web/Copilot commit and the full Figure 2 agent-hardening branch.
- `a676ace fix(agent): bind legacy trajectory identifiers`
  - future candidate-selection artifacts must carry `id_column` and stable model IDs;
  - legacy stability replay may resolve an identifier only from one complete, unique, full-set-matching shared column and may derive a selected-model provenance ID only from one exact selected-k/method record.
- `5be5c91 fix(agent): bind legacy trajectory method family`
  - future cluster-selection manifests must carry the exact clustering method family;
  - legacy replay may use candidate-model metadata only when the top-level and every fitted record agree on one normalized method family, never from filenames, step ids, or prose.
- `ddce5f8 fix(agent): bind trajectory replay schemas`
  - added digest-verified `trajectory_representation_schema` and `candidate_cluster_solution_schema` products;
  - legacy migration verifies the active step summary, representation, membership, candidate models, assignments, and selection before registering schemas on the original agent-owned producers;
  - `fail_closed` / `failed_closed` now propagate to the outer step as `contract_failed`.
- `8a68f10 fix(agent): preserve trajectory refit semantics`
  - schema v2 carries the selected fit method and covariance structure;
  - stability is explicitly forbidden from applying a schema to the wrong table or replacing an observed-data fit with a complete-data estimator.

After the merge, the temporary worktrees `EASYICU_fig2_dev_20260712` and `EASYICU_e3_run_20260712` were removed. Their scientific outputs remain under the main workspace `research_output/`. The merged local feature branches were deleted; remote refs were left unchanged.

## Integrated verification

- Claude-requested F1/F2/meta gate: `94 passed` across `test_validators_figure_source_trace.py`, `test_trajectory_clustering_routing.py`, and `test_meta_benchmark_spec.py`.
- Trajectory DAG/resume/authority gate: `43 passed`.
- Web/Copilot focused gate including extraction continuity: `79 passed`.
- Full deterministic Mock pipeline end-to-end: `1 passed` after the invalid old H3 process released CPU. Two concurrent attempts had correctly failed closed under resource contention (coder/figure fallback); they were not treated as integration success.
- Post-schema trajectory/fail-close/meta gate: `135 passed`; Ruff and diff checks passed.
- An offline hard-link replay of the real H3 checkpoint generated both schemas without case constants: 90,034 frozen rows, one verified identifier set, 84 ordered coordinates, selected k=6, and the historical observed-data diagonal-GMM fit contract all reconciled.

## H3 development run

Run directory:

`research_output/_diagnostic_h3_fresh_20260712_v2/bench_h3_gpt-56-luna/H3_trajectory_clustering/aware/run_20260712T220834_314aaa`

The old stability attempt remained `contract_failed`: it repeated candidate selection across k and wrote `cluster_sizes` from the stability owner. Its computed numbers are non-reportable. The merged main code generated `analysis_plan_revision_4.json`, which removes those cross-role outputs and binds `manifest:cluster_selection` to current evidence `log_cluster_selection_c1365499`.

The first revision-4 rerun proved that the role boundary was fixed: no selection table, size table, profile, outcome, or figure was written. It then failed closed because the historical candidate artifacts lacked explicit `id_column` and `selected_model_id`. This led to the general `a676ace` schema rule above. The next rerun correctly implemented those structural legacy bindings and then failed closed because the old selection manifest also lacked `clustering_method`; this led to `5be5c91`.

Revision 5 now binds both typed replay schemas into stability's resolved inputs. The first real v1-schema attempt was correctly rejected because generated code applied representation coordinates to the assignment table; no characterization/outcome/figure product was written. Schema v2 then froze `fit_method=observed_data_em_diagonal_gaussian_mixture` and `covariance_type=diag`. The coder request succeeded only after long structured-output retries, then entered concept-audit repair; the repair was interrupted rather than spending another unbounded long request. Therefore H3 stability remains **not passed**, and Step 05 was not run.

## Next action

The next architectural step is to make the agent emit a short typed `cluster_stability_spec` (resampling design, refit count/seeds, and frozen fit contract), then let a standard trajectory-stability method-family executor compute exactly that spec. The executor must not choose method, k, coordinates, population, missing-data handling, resample design, or outcome. Until that agent-owned spec boundary exists—or a full coder repair completes and passes every gate—do not run Step 05 and do not add H3-specific keywords or a benchmark-specific runner.
