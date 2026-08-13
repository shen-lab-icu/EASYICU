# Canonical9 M3 fixed-window trajectory context gate

Date: 2026-08-05  
Status: code fix complete on isolated branch; M3 remains incomplete  
Branch/commit: `codex/canonical9-m3-context-gate-20260805@d439f0a40c76185c06e42fd5b27bbe484580916a`

## Outcome

The M3 first-24-hour stay-level clustering task was being judged as a fixed-window trajectory DAG solely because its plan family was `trajectory_clustering`. The context had no materialized trajectory and no wide fixed-window variables. This made the host demand trajectory-only role ownership and schema products that an ordinary clustering plan neither declared nor needed.

Commit `d439f0a` compiles trajectory applicability once in the trajectory owner and passes the resulting boolean to both Coder prompts and the shared deterministic result gate. It also removes contradictory instructions that asked the Coder to add identity/model fields to the closed `cluster_selection.json` schema.

The fix is verified, but M3 is not complete. In verify53 the cluster step executed successfully in Docker, then the task-level wall-clock budget had already expired, so the post-execution LLM concept auditor could not run. The run correctly remained `diagnostic_only` / `incomplete`.

## Root cause and boundary

- M3 context: `materialized_inputs.trajectory == null`; no `fixed_window_trajectory` variable.
- H3 control context: a typed materialized trajectory is present.
- Before the fix, `trajectory_role_code_contract` and trajectory-specific declared-product findings were applied from structural products/method names without a context applicability decision.
- The owner contract is now `trajectory_context_is_bound(context)`. It recognizes either a typed materialized long trajectory or fixed-window wide variables.
- `trajectory_plan_contract_applies(plan, context)` still requires the `trajectory_clustering` analysis family, then delegates the data-side decision to that owner contract.
- Generic declared-product code receives an immutable `trajectory_role_contract_applies` value; it does not reimplement the predicate.

## Code changes

- `trajectory/plan_contract.py`: publish the context binding predicate; gate role-specific Coder text; correct the closed selection-manifest instructions.
- `agents/core.py` and `agents/agentic_coder.py`: pass the compiled context applicability to generation and repair prompts.
- `gates/contract.py` → `plan_utils.py` → `contracts/declared_product.py`: pass one plan/context applicability decision into the trajectory-specific result checks.
- `test_trajectory_role_scope.py`: red-first coverage for prompt schema, ordinary-clustering prompt silence, result-gate silence, and live shared-gate wiring.

## Verification

Red phase:

- `tests/research_agent/test_trajectory_role_scope.py`: 4 expected failures, 10 passes.

Green phase:

- Focused trajectory-role suite: 14 passed.
- Trajectory DAG/bound-tier/guide/execution/declared-product matrix: 374 passed.
- Agentic Coder + prompt-budget matrix: 95 passed, 2 failed. The same two prompt-budget failures reproduce unchanged at parent commit `18d7063`, so they are pre-existing and were not modified.
- Ruff on all changed files: passed.
- Real recorded context prompt probe: M3 `context_bound=false` and no fixed-window trajectory role contract; H3 `context_bound=true` and contract present.
- Offline replay of verify52 Step 04: `trajectory_contract_applies=false`; all former `trajectory_role_scope` / `trajectory_role_result` errors disappear. Its historical malformed selection manifest still fails the generic closed clustering-summary contract, as intended.
- Local provider protocol probe through the existing protected environment file: model `gpt-5.6-luna`, 6/6 calls, 0 retries, status passed. No credential value was printed or copied into artifacts.

## Exact-SHA runtime

- Image: `easyicu-research-agent:dev-d439f0a-20260805`
- Digest: `sha256:5bb2d1103a1c695655c07e9b50e6fb1c735d95e5ffa51400ffa53e8c1cf19042`
- `check_agent_runtime.py`: `status=ready`, `network=none`.
- Launcher: `项目进度/benchmark实验/verify53_m3_context_gate.sh`

## verify53 development run

- Batch: `/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260805_luna_miiv_FULL_d439f0a_verify53`
- Run: `m3_sepsis_subphenotype/aware/run_20260805T175857_613e90`
- Mode: aware-only, `--development-diagnostic`, not paper-authorized.
- Code identity: exact clean `d439f0a`; Docker network `none`.
- Pipeline result: 5/12 completed, Step 05 `repair_failed`, 3 analysis errors, `execution_complete=false`, `manuscript_ready=false`.
- Step 05 Docker execution: `returncode=0`, `timed_out=false`, duration `3157.961s`.
- No `trajectory_role_scope` or `trajectory_role_result` finding exists in the run.
- Terminal cause: after Step 05 execution, the LLM concept auditor was refused with `TASK_WALL_CLOCK_EXHAUSTED` (`4398.587s > 3600s`); the subsequent repair reservation correctly failed because the task was already `budget_exhausted`.
- Provider accounting: 14 successful calls, 266,975 reported tokens, estimated cost `$3.58081`; two additional failed-usage-unknown attempts are represented only in the conservative ledger upper bound.

## New blocker and next action

The generated Step 05 script used five candidate k values, ten stability seeds per k, `KMeans(n_init=20)`, and full-cohort `silhouette_score` for every seed. This is a general computational-budget defect: a scientifically valid candidate/stability design must declare a bounded, reproducible evaluation strategy that fits the task wall clock. Do not encode M3, Seymour, a specific variable, or a specific k grid into a shared prompt.

Next work should first characterize the general owner boundary for clustering evaluation budgets, write a red test that rejects unbounded full-pairwise silhouette/stability work on large cohorts, then implement the smallest typed budget or deterministic sampling contract. Only after a new exact-SHA aware-only development run passes Step 05 should M3 be rescored; the current Canonical9 score remains 4/9.

## 2026-08-06 shared-branch handoff and verify70 repair authority

The isolated M3 work and the endpoint-contract lineage were merged without textual conflict at `3ce0e1c`; the shared working baseline is now `fix/external-review-20260724-p0-p1`. Claude then closed three merge-adjacent test seams without changing production repair behavior: `68c3634`, `dd7e4a2`, and `478511f`. In particular, the two resume tests had reported zero Coder repairs because their monkeypatch doubles did not accept the production function's new keyword-only `step` argument. The production function was already correct; the stale doubles raised `TypeError` before the repair could run. Commit `478511f` restores the original one-repair assertion and adds a keyword-only seam guard.

Fresh aware-only development batch `batch_20260805_luna_miiv_FULL_0c5a5dc_verify70` reached Step 03 and failed with:

`TypeError: float() argument must be a string or a real number, not 'StrictNumericInput'`

The deterministic runner did produce the unique expected repair `strict_numeric_input_result_projection_v1`, changing code SHA-256 `92048f76…` to `298028b1…`. The candidate was then rejected by the central automatic-repair policy because the repair id was absent from the exact registry. The conservative fallback classified it as `method_substitution`, `introduces_numbers=true`, and `requires_disclosure=true`; the runtime-diagnostic gate therefore correctly denied it under the metadata it had. This was not a missing repair opportunity and did not justify adding another lifecycle round.

Commit `1c51fd3` adds only that exact id to the syntactic repair registry. Its contract is representation-only: append `.values` to one uniquely attributable `strict_numeric_input(...)` result before a known NumPy/pandas numeric constructor. It does not select rows, columns, methods, thresholds, or numeric values. Unknown repair ids remain conservative method substitutions and remain auto-denied.

Verification:

- Red-first registry test reproduced `classification_source=fallback:unknown_method_substitution` before the fix.
- Focused registry/runtime-policy matrix: 47 passed; broader deterministic-runner adjacency matrix: 83 passed.
- Ruff and diff-check passed.
- Real verify70 offline replay used the recorded Step 03 script and run log. Both production authorization predicates now return true for the exact syntactic repair.
- The repaired 94,458-row script executed in a temporary output directory in 7.5 seconds and generated non-empty `cluster_features.parquet`, `feature_quality_scaling.csv`, and `step_summary.json`. No recorded run artifact was modified.

Two historical prompt-budget reds were also adjudicated. They predated the merge and contradicted the intentional `fcea995` contract: an unposable minimal patch falls through to a lossless full rewrite rather than spending a scientific repair attempt without a provider call. Commit `cee7df6` updates only those tests. A 169,329-byte typed ticket is still rejected by the role transport before the provider; a 49,108-byte full rewrite carrying complete host binding remains below the configured 40,000-token transport ceiling and is allowed without truncation. Prompt/repair matrix: 88 passed.

Canonical9 remains 4/9. The next evidence-producing action is an exact-SHA image from the shared branch followed by a fresh aware-only M3 development canary. verify53's later clustering-compute cost remains a known likely downstream blocker, but it should not be changed again until the fresh merged run reaches it under the repaired authorization path.

## 2026-08-06 verify71–73 host-audit adjudication

Three fresh aware-only development runs moved M3 past the strict-numeric seam and exposed three independent host false positives. None changes the Canonical9 score or creates paper evidence.

- verify71 (`cee7df6`) reached the large-cohort clustering step with the previous steps clean, then rejected a wrapper that capped `sample_size` at 5,000 and deterministically seeded the underlying sklearn call. Commit `24a36e6` resolves imported aliases separately from local wrappers and proves conditional caps/seeds through call sites. The exact verify71 script changed from two silhouette violations to zero; unsafe-cap and dynamic-seed wrappers remain blocked.
- verify72 (`a3ce0d8`) proved the wrapper fix in a real run: the clustering script entered Docker. Its generated code built a local `input_bindings` summary with a top-level `input_key`; the mechanical gate conflated a later `for binding in input_bindings` target with an earlier typed-binding variable of the same name and twice rewrote the legal access to nonexistent `identity_row`. Commit `a637086` respects loop-target scope while preserving fail-closed behavior before the shadow and in loop `else`. The exact pre-repair SHA `7790e8f1…` passes both gates and executed against all 94,458 rows in the network-disabled image in about 21 seconds, producing non-empty cluster assignments, phenotype structure, and step summary. The temporary replay output was moved to Trash; the immutable verify72 run was not modified. Mechanical preflight adjacency: 717 passed.
- verify73 (`a637086`) generated `random_state = seed_base + candidate_count` with `candidate_count` from the fixed finite range `range(2, 7)`. The gate accepted the bounded sample but did not recognize that deterministic seed expression. Commit `7c6c179` proves static finite integer ranges and additive derived seeds, while a runtime-derived range remains blocked. The exact verify73 quarantined script changed from one silhouette violation to zero; clustering-gate adjacency: 142 passed.

Current clean development authority is `fix/external-review-20260724-p0-p1@7c6c179`. The next action is a new exact-SHA image and fresh aware-only M3 development run; do not resume terminal verify71, verify72, or verify73.

## 2026-08-06 verify74–75 typed-input boundary closure

Both runs were fresh, aware-only development diagnostics and remain ineligible for paper use. Canonical9 therefore stays 4/9.

verify74 used exact image `easyicu-research-agent:dev-7c6c179-20260806` (`sha256:a49a76a…`). It moved through feature construction, main clustering, stability analysis, profile outputs, and downstream figures. Its automatic figure child failed because the Coder compared a real 54-column parent artifact with the 32-column projection shown in the prompt. The host had labeled that lossy projection `columns`, making it indistinguishable from a complete schema. Commit `ed473dc` now uses `columns_preview`, `schema_preview_complete=false`, and an explicit omitted-column count/location whenever the projection is truncated. Complete schemas retain `columns`. Commit `f5cea7f` separately normalizes an absent legacy primary predictor represented by `""` to operational exposure `null`; explicitly malformed non-empty values remain fail-closed.

verify75 used exact clean `f5cea7f`, image digest `sha256:34c8292e71eaae4d646074c6a2d713bc1210bd333038e88e137c65f46390375f`, Docker network `none`. It completed 8 of 9 required steps. Main clustering, stability, phenotype profiles, cluster visualization, and outcome comparison all completed; all five evidence kinds were present. The sole failed step was `04_cluster_and_characterize_structure_figure` with:

`RuntimeError: Input table:cluster_characteristics does not have the required all_rows contract`

The resolved-input manifest proves `consumption_contract` is a top-level sibling of `product_contract`, with `mode=all_rows` and a verified row count. The generated script instead called `product_contract.get("consumption_contract", {})`. Commit `b85e9c2` adds a closed syntactic repair that activates only when the exact runtime error names a host binding whose top-level contract proves `all_rows`, the nested product contract has no competing contract, and exactly one attributable AST owner access exists. It changes only the lookup owner; the existing fail-closed all-rows test remains intact. Missing, subset, nested, ambiguous, already-correct, and unrelated cases are unchanged.

Verification:

- Red-first test failed at import before implementation; 6 focused tests pass after implementation.
- Repair registry/runtime adjacency: 53 passed.
- Existing deterministic code-repair suite: 117 passed.
- Ruff and diff checks pass.
- The immutable verify75 script and resolved binding were replayed in a temporary output directory. The repair changed exactly one owner access; replay status was `completed`, diagnostics were empty, and PNG/SVG/PDF/TIFF plus the figure contract were produced. The original run was not modified.
- verify75 batch result: 34 successful provider calls, 491,063 reported tokens, estimated cost `$6.68519`; one failed-usage-unknown attempt is represented in the conservative upper bound. Terminal execution status remained incomplete because the recorded run predates `b85e9c2`.

The posthoc Figure 2 evaluation also reports `Figure 2 paper scorer tree digest mismatch`. This reproduces in the existing evaluator adjacency and is independent of these source changes; no scorer or rubric file was modified. Do not refresh a frozen authority digest as incidental cleanup. Resolve that authority mismatch separately after M3 reaches execution completion.

Current clean development authority is `fix/external-review-20260724-p0-p1@b85e9c2`. Next action: build an exact-SHA, network-none image and run a fresh aware-only M3 development canary. Do not resume verify74 or verify75.

## 2026-08-06 verify76 minimal-patch acceptance gap

verify76 used exact clean `b85e9c2`, image `easyicu-research-agent:dev-b85e9c2-20260806`, digest `sha256:e7fae2a36621a28a8baebf9906fcf55ab35bd95edb5ebef60ba6920247ad3659`, and Docker network `none`. It was a fresh aware-only development diagnostic and remains ineligible for paper use.

The new Planner plan had 11 required steps. Cohort definition, availability flow, feature quality/scaling, and measurement-process audit completed. The main clustering script then executed successfully on all 94,458 stays and produced its seven planned products. The final LLM concept audit correctly rejected the script because it refit median imputation and `StandardScaler` on upstream artifacts the Planner had designated as the prespecified scaled representations. This is a real scientific-contract failure and was not relaxed.

The second LLM concept-repair response exposed a separate transport defect. Its minimal patch removed the `SimpleImputer` and `StandardScaler` imports but left both constructor calls. The patch parsed as Python, so `RepairCoordinator` marked the logical transport completed; the post-mutation mechanical gate then found `undefined_helper_call` and `unresolvable_name`. Because this was the second logical scientific repair attempt, the step stopped at 4/11 even though three provider-call slots remained. The original executed script has zero mechanical preflight errors; the quarantined patch has exactly those two errors.

Commit `396c0e2` adds a generic patch-candidate acceptance callback to the repair coordinator. `CoderAgent` runs the existing mechanical preflight owner against an applied minimal patch (restoring the outbound-safe script first for external transports). A mechanically invalid patch is not returned as a completed repair; it falls through to the already-authorized full-rewrite transport within the same logical repair attempt. The callback neither chooses nor changes any scientific method, variable, row, model, or threshold. Existing valid minimal patches remain the default.

Verification:

- Two new tests failed before implementation and passed afterward.
- Repair coordination, Coder preflight, provider budget, and prompt-budget matrix: 537 passed.
- Post-repair concept/deferred-audit/unposable-patch matrix: 27 passed.
- Agent/repair/execution package boundaries: 23 passed.
- Ruff, compile, and diff checks pass.
- One old prompt-budget fixture was exposed as mechanically invalid because it called `strict_numeric_input` without importing it; the fixture now models a genuinely executable successful patch.
- One prior lifecycle test was updated to assert the stronger contract: invalid contract patch → same-attempt full rewrite, one logical repair instead of two.

verify76 used 11 provider calls, 193,800 reported tokens, estimated cost `$2.61308`, and ended `blocked_by_concept_audit`. Canonical9 remains 4/9. Current clean development authority is `fix/external-review-20260724-p0-p1@396c0e2`; next action is a fresh exact-SHA, network-none M3 canary. Do not resume verify76.

## 2026-08-06 verify77 helper-resolved typed-input root repair

verify77 used exact clean `396c0e2`, image `easyicu-research-agent:dev-396c0e2-20260806`, digest `sha256:81d5817228edf7d6a1dd88bb29c236a99035372beefd5a8b1eac099d11cc890d`, and Docker network `none`. It was a fresh aware-only development diagnostic and is not paper evidence.

The run made one important lifecycle advance: Step 04 completed after two repair attempts. This is direct real-run evidence that `396c0e2` rejects a mechanically invalid minimal patch and continues through the authorized same-attempt rewrite path. The run then stopped at Step 05 after completing 4 of 12 required steps.

Both Step 05 LLM repair attempts were consumed by the same mechanical path-root error. The host manifest records product `relative_path` values relative to `EASYICU_RUN_DIR`, including their leading `evidence/` segment. The generated helper accepted a binding entry plus `EASYICU_EVIDENCE_DIR` and evaluated `Path(evidence_dir) / relative_path`. The first execution therefore failed on `artifact:analysis_cohort`; the first LLM rewrite special-cased that input but left the helper unchanged for `dataset:cluster_features`, causing the second execution to fail identically. The second rewrite fixed the root and executed successfully, producing eight expected artifacts. Its final summary used `selected_n_clusters` instead of the required top-level `n_clusters` or `cluster_count`; the final contract correctly failed closed, but no LLM repair budget remained.

Commit `528de21` extends the existing `resolved_input_run_root_v1` syntactic repair rather than adding a parallel repair identity. It claims a one-level helper only when all of the following are statically proven: a narrow local JSON loader reads the exact host-issued `EASYICU_RESOLVED_INPUTS_JSON`; the manifest/input entry descends from that document; the helper's `relative_path` descends from that entry parameter; the root parameter is used only in the attributable path join; and every helper call supplies a proven host binding plus the evidence root. Reassigned, ambiguous, unrelated, or partly unproven shapes remain unclaimed.

Verification:

- Red-first focused test reproduced zero findings before implementation.
- Typed-input, repair, preflight-owner, and module-boundary adjacency: 219 passed.
- Ruff, compile, and diff checks passed.
- The exact immutable verify77 initial Step 05 script produced one `resolved_input_run_root_v1` finding and zero after repair.
- The repaired script ran in the same network-disabled image against the immutable verify77 run inputs and exited 0, producing `cluster_assignments.parquet`, `cluster_characteristics.csv`, `cluster_count.json`, `cluster_selection.json`, `clustering_algorithm_details.json`, `clustering_methodology.json`, `phenotype_structure.csv`, and `step_summary.json` in `/Volumes/外置硬盘/easyicu_data/canonical9_runs/verify77_path_replay.HHfhcC`.
- Replaying the real step contract with verify77's actual trajectory-role applicability switch leaves exactly one finding: the missing top-level `n_clusters`/`cluster_count` alias. No result contract was weakened.

verify77 used 19 provider calls, 313,958 reported tokens, and estimated cost `$4.36364`. Canonical9 remains 4/9. Current development authority is `fix/external-review-20260724-p0-p1@528de21`; next action is a fresh exact-SHA, network-none M3 canary. Do not resume verify77.

## 2026-08-06 verify78 statistic lineage and equivalent sampling control flow

verify78 used exact clean `528de21`, image `easyicu-research-agent:dev-528de21-20260806`, digest `sha256:68c063992296d3064ad8a45ad954a5eeec1bc04133a0c870ff76c3ee6e9e1e01`, and Docker network `none`. It was a fresh aware-only development diagnostic and is not paper evidence.

The run completed 7 of 9 required steps. Step 03 feature construction recovered from a strict-numeric non-finite-value failure in one LLM repair. Step 04 candidate clustering executed without either verify77 path-root failure, reached the expected `selected_n_clusters` alias contract with repair budget remaining, and completed after one contract repair. Cluster profiles, phenotype structure, and the final cluster visualization also completed. This verifies that `528de21` moved repair capacity from mechanical path failures to the actual result contract.

Two steps remained failed:

- `04_cluster_candidate_models_figure` produced all four figure formats plus three source-data CSV files, but the statistic source CSV truthfully declared the exact hash-bound `statistic_step_artifact_cf7c323d18a82b81__cluster_count.json`. The validator treated every `source_table` declaration as tabular and searched only upstream tables, so it rejected the JSON provenance even though the CSV value matched the verified typed statistic.
- `05_cluster_stability_audit` was blocked pre-execution for supposedly unbounded silhouette evaluation. The quarantined candidate actually computed `bounded_n=min(5000,n)`, used a statically fixed seed, selected without replacement, and indexed both the feature matrix and labels with the same indices. The prior recognizer accepted only an equivalent branch shape that first materialized sampled arrays.

Commit `76ec2ce` closes both owner-local gaps. A statistic source may name a JSON artifact only when the filename is the exact current digest-verified typed binding from a successful parent and the source values replay against that statistic; foreign names and wrong values remain rejected. The silhouette gate now also proves the index-first control flow, including a static sample upper bound at every helper call, deterministic seed expressions, exact full-population branch safety, no-replacement selection, and shared indices. Dynamic seeds, missing seeds, samples above 5,000, and overwritten subsets remain blocked.

Verification:

- Both new tests failed before implementation and passed afterward.
- Figure-source, repair, method-compatibility, and pairwise-budget adjacency: 228 passed.
- Ruff, compile, and diff checks passed.
- Replaying the real verify78 figure output and resolved typed bindings now returns zero figure-source errors.
- Replaying the real quarantined stability script now returns zero `large_cohort_silhouette_unbounded` violations.
- The final run status remained fail-closed: `execution_complete=false`, failed steps exactly the two above, Canonical9 unchanged at 4/9.

Provider accounting: 42 successful calls, 586,015 provider-reported tokens, estimated cost `$7.95949`; three HTTP 500 attempts are usage-unknown, giving a conservative upper bound of 45 calls, 625,552 tokens, and `$8.47774`.

### Isolated Pandera proof of concept

Pandera 0.32.1 was evaluated with `uv run --no-project`; no package or lockfile was changed. A strict schema validated the real 94,458-row verify77 `cluster_assignments.parquet` in about 0.008 seconds. An in-memory mutation simultaneously detected a missing cluster column, a foreign column, duplicate `stay_id`, an out-of-range cluster label, and the resulting wrong label cardinality. The result supports a later narrow adapter inside the typed-table owner, while keeping EasyICU's stable issue codes, artifact paths, digests, and fail-closed envelope authoritative. Do not add Pandera to production dependencies until one owner-specific integration POC defines that translation contract.

Current development authority is `fix/external-review-20260724-p0-p1@76ec2ce`; next action is a fresh exact-SHA, network-none M3 canary. Do not resume verify78.

## 2026-08-06 verify79 execution closure and publication-finalization seams

verify79 used exact clean `76ec2ce`, image `easyicu-research-agent:dev-76ec2ce-20260806`, digest `sha256:7f8803b8079eac3305f623370ca10381251eab269dcfd75eabe6474a3a94f8f4`, and Docker network `none`. It was a fresh aware-only development diagnostic and is not paper evidence.

All seven planned scientific steps completed. Feature quality recovered through the existing strict-numeric projection repair, candidate clustering completed once, phenotype profiles completed after one LLM repair, the stability audit completed, and visualization completed after contract/concept repair plus `relocate_known_host_helper_import_v1`. The run therefore proves that both `76ec2ce` fixes work in the real pipeline: the sampled large-cohort silhouette implementation was not falsely quarantined, and the digest-bound statistic lineage no longer blocked its figure source.

The final run nevertheless remained `diagnostic_only` for two publication-finalization reasons. The primary publication renderer selected the generic one-panel adjusted-association family from `cluster_visualization_profile_source_data.csv`, treating `hr_mean_z` as an exposure and `death` as an outcome, instead of using the canonical phenotyping products. Separately, Writer emitted `{evidence:03_feature_availability_flow}` although the registered step id was `02_feature_availability_flow`; the unresolved placeholder correctly prevented publication readiness.

Commit `29bc22f` closes only those two owner-local seams:

- The phenotype renderer recognizes exact canonical product names `phenotype_profiles` and `cluster_stability`, accepts the real stability schema `metric + estimate`, and records both exact evidence sources in the figure contract/source data.
- Writer repairs a numbered step reference only when removing the numeric prefix leaves an exact suffix that uniquely matches one registered numbered step. There is no fuzzy, substring, or nearest-neighbor matching; ambiguity remains unresolved and fail-closed.

Verification:

- Red-first tests reproduced both failures before implementation.
- Focused and adjacent figure/writer/evidence matrix: 121 passed; Ruff, compile, and diff checks passed.
- Exact verify79 figure-finalization replay generated three roles (`phenotype_structure`, `phenotype_profile`, `stability`), recorded both phenotype-profile and cluster-stability evidence ids, and returned `audit_errors=[]`.
- Exact writer-finalization replay removed the nonexistent `03_feature_availability_flow`, emitted the registered `02_feature_availability_flow`, and recorded the exact ordinal repair.
- These are offline replays of the two finalization owners, not a retroactive pass for verify79.

Provider accounting: 40 provider-reported calls, 543,170 tokens, estimated cost `$7.31638`. The stability step took roughly 17 minutes; this is performance debt, not a correctness finding, and no performance patch was mixed into the finalization fix.

Canonical9 remains 4/9. Current development authority is `fix/external-review-20260724-p0-p1@29bc22f`; next action is a fresh exact-SHA, network-none M3 canary. Do not resume verify79.

## 2026-08-06 verify82–83 source-bundle and typed-input receipt closure

Both runs were fresh aware-only development diagnostics. Neither is paper evidence, neither changed the Canonical9 score, and neither terminal ledger may be resumed.

verify82 used exact clean `09aa59d`, image `easyicu-research-agent:dev-09aa59d-20260806`, digest `sha256:9636cf52b259d816332dc40b0b6e3aa2c8abe3d271f47cb96849d90504eeb106`, and Docker network `none`. It completed 4 of 11 required steps. The bounded `**kwargs` silhouette wrapper passed the method gate and real clustering executed on 94,458 stays, selecting `cluster_count=2` with silhouette `0.7650912789275442`, sample size 5,000, and fixed seed 20250301. Step 05 then failed its figure-source contract after two LLM repairs. The final bundle directly named an ambiguous same-step table, omitted an upstream parquet parent and another same-step table, and lacked a separately verifiable `same_step:statistic:cluster_count` companion.

Commit `a57cbcb` adds structural repair `complete_bound_figure_source_bundle_v1` in a dedicated figure-source owner. It copies only exact typed frames or already-written same-step CSV/parquet frames, removes only validator-identified invalid sources, and emits a two-column companion only for a side-effect-free scalar already written as a same-step statistic receipt. Missing, duplicate, ambiguous, upstream-statistic, or unresolved shapes decline. Verification: 426 adjacent tests passed; the exact verify82 script matched the repair and a full 94,458-row offline replay returned zero official figure-source errors. No original run artifact was modified. verify82 accounting was 14 provider-reported calls, 262,621 tokens, and `$3.71351`.

verify83 used exact clean `a57cbcb`, image `easyicu-research-agent:dev-a57cbcb-20260806`, digest `sha256:d1d4c71508ce2ff5e3438353ac09b63e9502dfe6a9dd7b4641f0acf197f409c4`, and Docker network `none`. It stopped earlier, at 3 of 11 completed steps. Step 04 generated `cluster_feature_matrix.parquet`, `cluster_feature_missingness.csv`, `cluster_preprocessing.json`, and a completed summary, but the summary omitted `input_bindings`. The host correctly refused to attest for LLM-generated code and blocked Step 05 on dependency evidence. This is distinct from the prior deterministic-fallback receipt fix: the script itself selected `artifact:analysis_cohort`, read its evidence id/relative path/digest, checked the digest, and loaded all 94,458 rows, but failed to project that proof into its summary. verify83 accounting was 12 provider-reported calls, 232,009 tokens, and `$3.04239`.

Commit `238ce3e` adds structural repair `complete_typed_input_receipt_v1` to the typed-input repair owner. It activates only for one exact missing host key and proves the closed dataflow `EASYICU_RESOLVED_INPUTS_JSON -> open/json.load -> manifest -> inputs -> binding`, direct evidence/digest/path extraction, a fail-closed digest guard, one exact pandas table load, and one dumped `step_summary` mapping. It then projects the script-proven key, evidence id, digest, `loaded=true`, and actual loaded row count. Arbitrary config mappings, missing digest guards, multiple inputs, ambiguous assignments, and existing receipts decline. Verification: 97 receipt/validator/host-ownership/registry tests passed after the final tightening; the exact verify83 script replayed all 94,458 rows in a temporary directory and produced a receipt identical to the host manifest. Earlier adjacent regression before the tightening was 484/484; the tightening only reduces eligible shapes. No original run artifact was modified.

The user ended the run window after verify83. Current clean development authority is `fix/external-review-20260724-p0-p1@238ce3e`. Do not build another image or call the Provider in this window. The next work window should begin with Claude review of `a57cbcb` and `238ce3e`; only after review should a new exact-SHA, network-none, fresh aware-only M3 canary be considered. Canonical9 remains 4/9.

## 2026-08-07 verify84 bound-panel measurement-status alias

verify84 used exact clean `238ce3e`, image `easyicu-research-agent:dev-238ce3e-20260806`, digest `sha256:67b0dfe2f2bd3e1c75202c778becd2137fe5d4da5910bb02bbb7918180918d0f`, and Docker network `none`. It was a fresh aware-only development diagnostic and is not paper evidence.

The run stopped at 3 of 11 required steps. Step 03 completed and recorded the typed `analysis_cohort` binding with evidence id, digest, and 94,458 rows. Step 04 failed twice because the generated script derived `alb_first_measured`, `hr_first_measured`, and related names from `FEATURE_NAMES` ending in `_first`, while the exact bound `time_aligned_feature_panel` contract exposes `alb_measured`, `hr_measured`, and the corresponding aliases. The host correctly kept both attempts fail-closed; no deterministic repair was applied in the old ledger.

Commit `2892ede` adds the narrow structural repair `bound_panel_measurement_status_alias_v1` to the typed-input repair owner. It requires the exact measurement-status error family, one literal panel key, one literal feature-name list whose values end in `_first`, an exact missing-name set, a digest-bound resolved panel binding, and matching bound product columns. It patches only the status-column expression to remove the `_first` suffix; arbitrary column inference, ambiguous bindings, and unrelated scripts decline.

Verification:

- Six new negative/positive tests plus typed-input, figure-source, code-repair, and host-receipt adjacency: 148 passed under the canonical `.venv`.
- Ruff, compile, and diff checks passed.
- The immutable verify84 Step 04 script replayed offline against the real 94,458-row bound panel after the repair and exited successfully, producing all ten expected outputs. This is an owner-local replay, not a retroactive pass for verify84.

verify84 used 15 provider-reported calls, 270,186 total tokens, and estimated cost `$3.76808`. Canonical9 remains 4/9. Current development authority is `fix/external-review-20260724-p0-p1@2892ede`; next action is to build a new exact-SHA image and run fresh verify85. Never resume verify82/83/84.

## 2026-08-07 verify85 and label-preserving silhouette gate

verify85 was launched as a fresh aware-only development diagnostic with image `easyicu-research-agent:dev-2892ede-20260807`, image digest `sha256:5e688e09e8a3f5e46fcf15763ea3e78b082d4cef1b420cf1c37f0d027aa8b352`, and Docker network `none`. It must not be treated as a clean source-bound acceptance: while the run was active, the shared branch advanced from `2892ede` to remote merge `31dc0aa`, and the ledger records host `git_sha=31dc0aa` beside the `2892ede` image. No old ledger was resumed.

The run completed 3 of 9 required steps. Steps 02 and 03 completed. Step 04 first failed on a generated typed-input path shape, then the runtime repair executed; its per-cluster silhouette calculation called `silhouette_score` on a subset containing one label and raised `ValueError: Number of labels is 1`. The subsequent safe full rewrite used a deterministic permutation sampler that retained one row per label and filled to a cap of 5,000, but the method-compatibility gate rejected the inner sklearn call because the budget keywords were no longer present. The step therefore ended `blocked_by_concept_audit`; Canonical9 stayed 4/9.

Commit `1d79596` extends the large-cohort silhouette gate with a conservative AST proof for that exact label-preserving sampler. It requires a statically bounded cap and deterministic seed at every helper call, `default_rng(...).permutation(...)`, one selected row per unique label, no-replacement fill, an explicit label-preservation guard, and sampled-array inputs to the metric. A plain random subset or an unproven wrapper remains rejected.

Verification:

- The new gate test is red-first and now has both positive and negative cases; the complete large-cohort budget matrix is 22 passed.
- The prior focused repair/typed-input/figure-source adjacency run passed 214 tests; three unrelated phenotype figure helper tests remain pre-existing API-shape failures (`_silhouette_value` returns a tuple while those tests subtract it as a scalar) and were not changed by this patch.
- Ruff, compile, and diff checks pass.
- The exact quarantined verify85 Step 04 full rewrite replayed offline in the network-disabled `dev-2892ede` image against the real bound inputs and exited 0, producing eight expected clustering outputs. This is an owner-local replay, not a retroactive pass for verify85.

verify85 used 14 provider-reported calls, 224,376 total tokens, and estimated cost `$3.04740`. Current development authority is `fix/external-review-20260724-p0-p1@1d79596`; next action is to build a new exact-SHA image and run fresh verify86. Never resume verify82/83/84/85.
