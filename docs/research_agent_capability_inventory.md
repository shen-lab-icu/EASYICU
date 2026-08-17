# Research-agent capability inventory

**What this is.** The reachability registry for package-local modules that look
unreachable to the static import graph. It distinguishes code that a public
entry point demonstrably reaches from experimental or disabled code. "Already
written" is not the same as "available in the product". Each production row
must name the exact public-API-to-executor route and one integration test that
traverses it; every other row must stay visibly experimental or disabled.

**What this is not.** It is not a redundancy audit. Zero-inbound-import is only
the *most visible* kind of unused code; it says nothing about unreachable
branches inside wired modules, string- or template-dispatched entry points
(`run_feasibility_protocol` is called from a generated code template and looks
orphaned to any import scan), or paths that exist but are never traversed in a
real run. Do not compute "% of the module that is live" from this file.

**How to regenerate the candidate list**

```bash
python tools/research_agent_module_graph.py > /tmp/graph.json
```

Modules with in-degree 0 that are not `__init__.py` are candidates; then check
each leaf name against `src/easyicu` (excluding `research_agent`), `tests/`,
`tools/` and `pyproject.toml` before calling anything unused. Console-script
entry points (`cli`, `replication_cli`) are reached through
`[project.scripts]`, not imports.

Measured 2026-08-14 at `26bdcdc`.

## Status vocabulary

| status | meaning |
| --- | --- |
| `production_reachable` | A public API reaches the real runtime owner, and the row binds an exact integration test proving that route. |
| `experimental` | Implemented or tested, but not advertised as a stable product capability and not relied on by the default runtime. |
| `disabled` | Intentionally unavailable on production paths; retained only for a named compatibility/removal boundary. |

## Inventory

| module | LOC | status | owner | activation precondition | tests | route proof | review |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `methods/rmst.py` | 264 | `experimental` | methods | `time_to_event.rmst` is now published before planning and its exact reviewed kernel is selected into Coder authority. It remains experimental because no deterministic `owns_step` executor/typed result validator exists and no external survRM2 oracle is installed. | 21 | - | 2026-10-01 |
| `methods/decision_curve.py` | 228 | `experimental` | methods | `prediction.decision_curve` now reaches Planner → typed Plan action → Coder resource selection. It remains Coder-generated/analysis-only until a deterministic DCA owner, panel and external oracle are registered. | 11 | - | 2026-10-01 |
| `methods/delong_auc.py` | 285 | `experimental` | methods | `prediction.delong_ci` now reaches Planner → typed Plan action → exact Coder kernel and remains oracle-checked against `pROC::roc.test`; no deterministic host owner currently promotes it. | 4 | - | 2026-10-01 |
| `methods/conformal.py` | 165 | `experimental` | methods | `prediction.conformal_intervals` now reaches Planner → typed Plan action → exact Coder kernel. It remains exploratory/analysis-only until a typed split/coverage contract and deterministic validator are registered. | 5 | - | 2026-10-01 |
| `methods/survival_inputs.py` | 219 | `experimental` | methods | Published to Planner as a reviewed survival supporting primitive and available to the digest-bound Coder selector; it is not a standalone estimand owner and no production executor currently consumes it directly. | 4 | - | 2026-10-01 |
| `methods/temporal_features.py` | 306 | `experimental` | methods | Published to Planner as a reviewed survival/causal timing primitive and available to the digest-bound Coder selector; the trajectory/landmark scientific contract still governs whether it can support a reportable result. | 9 | - | 2026-10-01 |
| `methods/dynamic_prediction.py` | 313 | `experimental` | methods | `dynamic_prediction` → required typed action `prediction.dynamic_prediction` → digest-bound Coder resources (`dynamic_prediction`, `temporal_features`, `sklearn`). The reviewed kernel owns leakage-safe landmark feature/label/evaluation helpers. A separate deterministic binary fit owner now exists experimentally, but it is not registered on this landmark Planner/runtime route, so fitting here remains generated and analysis-only. | `tests/research_agent/test_scientific_action_catalog.py::test_dynamic_action_selects_sklearn_and_both_reviewed_timing_kernels` | - | 2026-10-01 |
| `execution/runners/prediction_validation_executor.py` | 22 | `experimental` | prediction-validation | Compatibility-only incubator adapter re-exporting the dependency-neutral prediction-validation owner. It remains outside Planner/runtime selection and contains no EvidenceStore or paper-authority logic. | 38 | - | 2026-10-01 |
| `prediction_model_fit_owner.py` | 881 | `experimental` | prediction-model-fit | Direct experimental API consumes one host-issued immutable typed input plus a fully fixed subject-level declaration. Numeric median imputation, standardization and L2 logistic regression are fitted only on declared training subjects; exact model JSON, all-row prediction CSV, source/split/model byte digests, package versions and an immutable analysis-only receipt are issued together. Test-only extremes cannot alter train-fitted state, the model artifact independently reconstructs probabilities, and full recomputation rejects payload/model/receipt/source/contract drift. The owner itself remains outside Planner/runtime; a separate experimental authority bridge may consume its sealed bundle without changing its tuning, selection or paper authority. | 22 | - | 2026-10-01 |
| `prediction_validation_owner.py` | 663 | `experimental` | prediction-validation | Shared host owner for the execution adapter and EvidenceStore bridge. It consumes the typed declaration and exact digest-bound UTF-8 CSV bytes, performs the deterministic full receipt recomputation, reconciles the cohort subject set and one subject-disjoint split assignment against every prediction subject, and issues the runtime-bound zero-finding seal internally. Stable cohort/split mismatch reasons cover missing columns, missing or extra subjects, duplicate assignments and split disagreement. Frozen and live base-R oracles still cover AUROC, Brier score, non-identity joint calibration coefficients and threshold counts. It trains no model and grants no claim authority. | 56 | - | 2026-10-01 |
| `authority/prediction_validation_evidence.py` | 549 | `experimental` | prediction-validation | Direct incubator bridge resolves seven exact, digest-verified upstream EvidenceStore records (prediction table, cohort, subject-disjoint split, model, source snapshot, environment lock and runtime receipt), then invokes the shared owner against current stored bytes. The public registration route accepts no caller-supplied receipt or host seal, requires exact cohort membership and split semantics, and registers one deterministic `analysis_only` bundle. Its runtime-subset resolver is reused by the fit-evidence bridge so code/environment/runtime policy is not copied. Registration publishes no aliases, numeric claims or scientific claims; revalidation repeats host recomputation and detects later authority promotion. It remains outside Planner/runtime selection and grants no paper authority. | 18 | - | 2026-10-01 |
| `authority/prediction_model_fit_evidence.py` | 580 | `experimental` | prediction-model-fit | Host-owned composite bridge accepts one V4 sealed fit, the same immutable typed input, a matching validation contract and three pre-registered runtime-authority records. It revalidates the complete fit before writing, materializes canonical source/split/prediction roles plus a model envelope carrying the fit spec, fit receipt, typed-input receipt and model state, builds the closed seven-role lineage internally, then delegates deterministic metrics and analysis-only registration to V3.1. The public route accepts no loose lineage, model, prediction, receipt or seal; retries are idempotent, all four fit-role byte drifts fail current-store validation, and no aliases or claims are published. It remains outside Planner/runtime selection and grants no paper authority. | 10 | - | 2026-10-01 |
| `evaluation_scorecard.py` | 1628 | `experimental` | evaluation | Optional Tier-2 scoring over completed artifacts. Paper-facing scorer authority lives outside the installed engine, under `benchmarks/`. | 7 | - | 2026-11-01 |
| `evaluation/tier2_jury.py` | 649 | `experimental` | evaluation | Optional jury/rubric adapter; not enabled on the default production path. | 8 | - | 2026-11-01 |
| `evaluation/cross_model_panel.py` | 294 | `experimental` | evaluation | Optional cross-model concordance; not enabled on the default production path. | 5 | - | 2026-11-01 |
| `acquisition/foundation.py` | 603 | `experimental` | acquisition | The opt-in Web canary route calls this owner before `ResearchAgentPipeline.run`, but the current integration test replaces both acquisition and Pipeline with fakes. Keep the route experimental until a bounded canary or non-fake integration traverses the real acquisition-to-pipeline boundary. | `tests/test_pi_copilot_research_workflow.py::test_web_runner_delegates_to_research_agent_pipeline` | - | 2026-08-31 |
| `discovery/idea_mining_source_status.py` | 1144 | `experimental` | idea-mining | Idea-mining lane. Needs the differentiated fresh Idea end-to-end run before this is advertised as a stable product capability. | 1 | - | 2026-10-01 |
| `discovery/idea_mining_extended_feasibility.py` | 680 | `experimental` | idea-mining | Same lane. | 5 | - | 2026-10-01 |
| `discovery/idea_mining_data_first_route.py` | 627 | `experimental` | idea-mining | Same lane. | 3 | - | 2026-10-01 |
| `discovery/concept_proposal.py` | 460 | `experimental` | idea-mining | Same lane. | 5 | - | 2026-10-01 |
| `discovery/idea_mining_longitudinal.py` | 349 | `experimental` | idea-mining | Same lane. | 2 | - | 2026-10-01 |
| `discovery/longitudinal_handoff.py` | 263 | `experimental` | idea-mining | Same lane. | 1 | - | 2026-10-01 |
| `authority/source_status_sdk.py` | 240 | `experimental` | authority | Public SDK candidate for source-status receipts; no production consumer is currently registered. | 1 | - | 2026-10-01 |
| `reporting/system_validation_report.py` | 782 | `experimental` | reporting | The governed reviewer demo's completed, privacy-passing Web projection calls this engineering-only report builder, renderer and receipt owner. The exact integration test starts at private `_write_projection`, so no public production route is claimed. | `tests/test_pi_copilot_research_workflow.py::test_pipeline_projection_uses_real_artifacts_and_withholds_identifier_table` | - | 2026-11-01 |
| `reporting/result_card.py` | 281 | `production_reachable` | reporting | Public Pi `easyicu_inspect_interpretation` tool → `build_result_interpretation_card` → bounded aggregate-only projection. | `tests/test_pi_copilot_research_workflow.py::test_interpretation_and_manuscript_tools_bound_large_agent_drafts` | `call:tool_module.execute_tool;trace:easyicu_result_interpretation_projected` | 2026-11-01 |
| `replication/discovery.py` | 698 | `production_reachable` | replication | Public `discover_easyicu_exports` → native/legacy ExportPackage discovery → verified package result. | `tests/research_agent/test_export_package_integration.py::test_replication_discovery_accepts_native_and_legacy_packages` | `call:discover_easyicu_exports` | 2026-11-01 |
| `cli.py` | 426 | `production_reachable` | agent | `[project.scripts] easyicu-research-agent` → `cli.main` → `ResearchAgentPipeline.run`. | `tests/research_agent/test_cli.py::test_public_cli_reaches_the_research_agent_pipeline` | `call:cli.main;trace:run` | 2026-11-01 |
| `script_runtime.py` | 106 | `production_reachable` | agent | Public `script_runtime.write_json` → strict JSON artifact with recursive non-finite normalization. | `tests/research_agent/test_script_io_replaces_the_pasted_helper.py::test_the_imported_writer_emits_null_for_every_non_finite_number` | `call:write_json` | 2026-11-01 |
| `graph.py` | 55 | `disabled` | orchestration | Retired LangGraph builder fails closed; removal target is 2.0 in `docs/deprecation_policy.md`. | 1 | - | 2.0 |
| `scientific_adapters/*` | 52 | `experimental` | scientific-adapters | Optional dowhy / pandera / sksurv adapters are dynamically probed but may not upgrade current scientific capabilities. | 10 | - | 2026-11-01 |
| `replication_cli.py` | 228 | `production_reachable` | replication | `[project.scripts] easyicu-research-replication` → `replication_cli.main` → `ResearchAgentPipeline.reproduce_paper`. | `tests/research_agent/test_replication_cli.py::test_replication_cli_paper_mode_dispatches_to_pipeline` | `call:replication_cli.main;trace:reproduce_paper` | 2026-11-01 |
| `case_plugins/` | 26 | `experimental` | case-plugins | Plugin discovery surface; default production registry remains empty until an explicit plugin activation contract is supplied. | 7 | - | 2026-11-01 |
| `facade.py` | 100 | `production_reachable` | agent | Public `easyicu.research_agent.facade.go` → `ResearchAgentPipeline.run` with fail-closed offline mock; replicate/resume wrappers delegate to the same pipeline. | `tests/research_agent/test_facade.py::test_go_defaults_to_offline_mock_and_delegates_to_pipeline` | `call:facade.go;trace:ResearchAgentPipeline.run` | 2026-11-01 |
| `benchmark_instances.py` | 60 | `production_reachable` | agent | Public `easyicu.research_agent.benchmark_instances.evaluate` → `icu_agent_bench.grade_bench_task` over the frozen checkable subset. | `tests/research_agent/test_benchmark_instances.py::test_evaluate_scores_a_frozen_instance_in_one_call` | `call:evaluate;trace:grade_bench_task` | 2026-11-01 |

## Rules

1. A new module that no production path calls must be added here in the same
   change that introduces it, with a review date.
2. At the review date the owner picks one of: **make it production-reachable**
   with a typed route and integration test, **keep it experimental** with a new
   reason/date, **disable it**, archive it, or delete it.
3. Do not delete an `experimental` row to make a number look better. The
   engineering rule in `CLAUDE.md` stands: zero production references alone
   never justifies deleting a real implementation — a runner, validator or
   authority needs positive evidence it has been superseded.
4. `methods/` kernels wired without exact typed inputs/outputs and a full data
   contract for `owns_step` are a regression even if tests pass. A
   method-string-to-function mapping is not wiring.
5. A `production_reachable` row without an exact
   `tests/...py::test_name` reachability test is invalid and fails the inventory
   audit. The row must also declare `call:<public entrypoint>` route proof; an
    indirect endpoint additionally declares `trace:<key>` that the test asserts
    after invoking the public entrypoint. Unit tests of the isolated kernel are
    not sufficient.
6. The activation route starts at the first public coordinate the bound test
   actually invokes. Do not prepend an upstream UI or tool route that the cited
   test does not exercise.
