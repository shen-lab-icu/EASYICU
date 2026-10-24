The jump from 544 to 600 is an accidental host dependency expansion. All 56 paths first enter in `e529782b1` (mg), through the new `scientific_maturity -> descriptive_report_facts` import. The fixture-only commit and dc3 add zero kernel paths. These are pre-existing source files reached by manuscript reporting, planning, provider retries and orchestration; the jump does not represent 56 new executor owners.

The working-tree patch fixes the shared boundary that exposed this expansion. Two figure renderers now obtain their rendering helpers from the existing `figures/prior_output_support.py` owner. Their old host publication facade retains its compatibility exports. The computed kernel becomes **456 files**, removing the 56 added paths and 88 older host dependencies. The helper logic is unchanged.

Status: done, with one pre-existing manifest-test failure documented below. No commit was made.

## Computed history

`execution_kernel_relative_paths` takes the **easyicu package root**, e.g. `Path("src/easyicu")`, and returns paths relative to that root. I read its implementation and docstring before calling it. Each historical source snapshot contained only tracked `src/easyicu/*.py` files extracted from that revision inside this clone. Each snapshot's own function computed its manifest; source imports were resolved using the same owner logic. No historical application, provider or analysis was run.

| Revision | Kernel files | Added since previous row | Removed |
| --- | ---: | ---: | ---: |
| `e81c40797` | 544 | 0 | 0 |
| `4b7c7c6fd` | 544 | 0 | 0 |
| `e3fe2f16d` | 544 | 0 | 0 |
| `e529782b1` | 600 | 56 | 0 |
| `d07ad011f` | 600 | 0 | 0 |
| `6064eaf6a (worker base)` | 600 | 0 | 0 |

The manifest implementation is unchanged between `4b7c7c6fd` and `d07ad011f`. The earlier e81c-to-4b change caches parsing and produces the same 544-path set. There are no removals during the 544-to-600 expansion, and mg creates no new Python source files.

## Why the host code entered

Both figure files are kernel seeds. At the time of the receipts, their imports led through:

```text
figures/association_prior_outputs.py:20 ─┐
                                      ├─> reporting/publication_bundles.py
figures/sensitivity_prior_outputs.py:17 ┘      -> reporting/readiness.py
                                             -> reporting/scientific_maturity.py
                                                -> reporting/descriptive_report_facts.py
                                                   (new edge in e529782b1, line 48)
```

`descriptive_report_facts` then reaches two broad host branches:

- `orchestration/exposure_group_labels.py:39` imports the grouping record constants from `orchestration/exposure_grouping_phase.py`. That module imports the Planner, configuration, resumed planning compiler and resource infrastructure. The label reader also imports `planning/exposure_group_compile.py` for `planned_exposure_contrasts`.
- Its function-local import at line 642 reaches `reporting/benchmark_report_facts.py`, which imports `planning/question_requirements.py:354` and the population-spec owner. This is production reporting source; no benchmark input directory was inspected.

The closure follows imports anywhere in ordinary modules, including function-local and type-checking imports. A counts-only call need not execute those branches for their source to enter the manifest. In the historical graph, deleting just `scientific_maturity -> descriptive_report_facts` restores **exactly the previous 544-path set**. Deleting only the label-reader-to-grouping-phase edge leaves 558 paths, so moving those two constants alone would leave part of the expansion.

The figure-to-publication-facade dependency predates mg. The sensitivity renderer's import statement is attributed to `f9340065a0`; the association renderer's current import statement and the public helper aliases are attributed to `187cbf95d3`. The mg import exposed that earlier boundary problem. `completion -> writer_stop` is not a cause of this jump: both modules, and the provider-client dependencies of `writer_stop`, were already in the 544-path closure.

## All 56 added kernel paths

Every path below is in the exact set `manifest(d07ad011f) - manifest(4b7c7c6fd)`. Paths use the API's package-relative form; prepend `src/easyicu/` for repository paths. Import locations and symbols are from `e529782b1`. One concrete incoming edge is shown per path; following the incoming rows leads back to the new manuscript-facts import. Package initializers enter by the resolver's package rule or their explicit imports.

The verdict for all rows is **unintended kernel binding through the host reporting facade**. Some rows are valid shared contracts, but this inclusion route is a host workflow dependency, not a new executor import.

| Added kernel path | First kernel inclusion | Incoming import edge |
| --- | --- | --- |
| `research_agent/agents/__init__.py` | `e529782b1` | `research_agent/agents/exposure_grouping_planner.py` requires package initializer |
| `research_agent/agents/coder_generation.py` | `e529782b1` | `research_agent/orchestration/config.py:158` imports `MAX_INITIAL_GENERATION_ATTEMPTS` |
| `research_agent/agents/exposure_grouping_planner.py` | `e529782b1` | `research_agent/orchestration/exposure_grouping_phase.py:61` imports `EXPOSURE_GROUPING_ROLE, ask_exposure_groupings` |
| `research_agent/authority/development_projection.py` | `e529782b1` | `research_agent/authority/typed_binding.py:39` imports `DEVELOPMENT_PRIMARY_COHORT_CONFIRMATION_ROLE, resolve_development_input_projection` |
| `research_agent/authority/evidence_record_resolution.py` | `e529782b1` | `research_agent/authority/typed_binding.py:48` imports `_current_verified_evidence_record, _evidence_record_field, executed_product_evidence_authority` |
| `research_agent/authority/step_runtime.py` | `e529782b1` | `research_agent/authority/evidence_record_resolution.py:10` imports `StepAuthorityRuntimeError, load_explicit_executed_success_step_capsule` |
| `research_agent/authority/typed_binding.py` | `e529782b1` | `research_agent/resources/coder.py:24` imports `_coder_authority_with_typed_parent_schema_receipts` |
| `research_agent/authority/typed_input_receipt.py` | `e529782b1` | `research_agent/authority/development_projection.py:20` imports `typed_input_row_identity_sha256` |
| `research_agent/authority/typed_schema_prompt.py` | `e529782b1` | `research_agent/authority/typed_binding.py:63` imports `typed_parent_schema_context_block` |
| `research_agent/contracts/artifact_consumption.py` | `e529782b1` | `research_agent/authority/typed_binding.py:29` imports `ArtifactConsumptionError, verify_artifact_consumption` |
| `research_agent/contracts/control_signals.py` | `e529782b1` | `research_agent/providers/structured_retry.py:54` imports `ProgressControlSignal` |
| `research_agent/gates/preplan.py` | `e529782b1` | `research_agent/orchestration/exposure_grouping_phase.py:78` imports `preplan_data_findings` |
| `research_agent/know_how/__init__.py` | `e529782b1` | `research_agent/planning/preplan_know_how.py:10` imports `KnowHowHit, KnowHowIntegrityError, KnowHowRegistry` |
| `research_agent/know_how/registry.py` | `e529782b1` | `research_agent/know_how/__init__.py:10` imports `KnowHowCard, KnowHowClaim, KnowHowCitation, KnowHowDesignCandidates, KnowHowHit, KnowHowIntegrityError, KnowHowReviewAttestation, KnowHowRegistry, reviewable_card_content_sha256` |
| `research_agent/orchestration/config.py` | `e529782b1` | `research_agent/orchestration/exposure_grouping_phase.py:113` imports `PipelineConfig` |
| `research_agent/orchestration/exposure_group_labels.py` | `e529782b1` | `research_agent/reporting/descriptive_report_facts.py:32` imports `exposure_group_label_rows` |
| `research_agent/orchestration/exposure_grouping_phase.py` | `e529782b1` | `research_agent/orchestration/exposure_group_labels.py:39` imports `EXPOSURE_GROUPINGS_FILENAME, EXPOSURE_GROUPINGS_RECORD_SCHEMA` |
| `research_agent/orchestration/profiles.py` | `e529782b1` | `research_agent/orchestration/config.py:701` imports `is_paper_facing_profile` |
| `research_agent/orchestration/reviewed_requirements.py` | `e529782b1` | `research_agent/orchestration/exposure_grouping_phase.py:114` imports `bind_reviewed_requirements` |
| `research_agent/planning/accepted_analysis_inputs.py` | `e529782b1` | `research_agent/orchestration/config.py:725` imports `AcceptedAnalysisInputs` |
| `research_agent/planning/cohort_identity.py` | `e529782b1` | `research_agent/planning/exposure_group_compile.py:57` imports `cohort_identity_columns` |
| `research_agent/planning/exposure_group_compile.py` | `e529782b1` | `research_agent/orchestration/exposure_group_labels.py:36` imports `planned_exposure_contrasts` |
| `research_agent/planning/exposure_group_spec.py` | `e529782b1` | `research_agent/orchestration/exposure_grouping_phase.py:104` imports `ExposureGroupings, read_stated_exposure_groupings` |
| `research_agent/planning/ordinal_multi_outcome.py` | `e529782b1` | `research_agent/planning/progressive_compiler.py:96` imports `resolve_ordinal_multi_outcome_contract` |
| `research_agent/planning/population_compile.py` | `e529782b1` | `research_agent/planning/exposure_group_compile.py:68` imports `read_threshold` |
| `research_agent/planning/population_provenance.py` | `e529782b1` | `research_agent/planning/population_compile.py:88` imports `PopulationProvenance, bind_population_provenance` |
| `research_agent/planning/population_spec.py` | `e529782b1` | `research_agent/planning/question_requirements.py:99` imports `quote_written_in` |
| `research_agent/planning/preplan_know_how.py` | `e529782b1` | `research_agent/planning/progressive_compiler.py:121` imports `verify_know_how_decisions` |
| `research_agent/planning/progressive_compiler.py` | `e529782b1` | `research_agent/planning/progressive_resume.py:25` imports `compile_progressive_plan` |
| `research_agent/planning/progressive_host_materialization.py` | `e529782b1` | `research_agent/planning/progressive_resume.py:40` imports `normalize_progressive_action_contract` |
| `research_agent/planning/progressive_resume.py` | `e529782b1` | `research_agent/orchestration/exposure_grouping_phase.py:109` imports `progressive_resume_context_sha256` |
| `research_agent/planning/question_requirements.py` | `e529782b1` | `research_agent/reporting/benchmark_report_facts.py:354` imports `QUESTION_REQUIREMENTS_FILENAME, QuestionRequirementsError, judge_recorded_requirements` |
| `research_agent/planning/target_trial_compile.py` | `e529782b1` | `research_agent/planning/target_trial_configuration.py:40` imports `CompiledTargetTrial, compile_record_sha256, compile_target_trial` |
| `research_agent/planning/target_trial_configuration.py` | `e529782b1` | `research_agent/orchestration/config.py:732` imports `ConfirmedTargetTrial` |
| `research_agent/planning/target_trial_spec.py` | `e529782b1` | `research_agent/planning/target_trial_configuration.py:45` imports `TargetTrialSpec` |
| `research_agent/planning/treatment_capture.py` | `e529782b1` | `research_agent/planning/target_trial_compile.py:103` imports `CaptureEntry, LoadedCaptureRegistry, packaged_treatment_capture_registry` |
| `research_agent/providers/efficiency_budget.py` | `e529782b1` | `research_agent/orchestration/config.py:977` imports `PlannerEfficiencyLimits` |
| `research_agent/providers/strict_json_schema.py` | `e529782b1` | `research_agent/agents/exposure_grouping_planner.py:42` imports `assert_closed_json_schema` |
| `research_agent/providers/structured_diagnostics.py` | `e529782b1` | `research_agent/providers/structured_retry.py:63` imports `infer_validation_stage, json_eof_state, json_truncation_issues, safe_declared_reason_code, safe_json_eof_state, safe_projected_validation_issues, safe_validation_issues, safe_validation_stage, violation_sha256` |
| `research_agent/providers/structured_retry.py` | `e529782b1` | `research_agent/orchestration/exposure_grouping_phase.py:110` imports `StructuredResponseFailure` |
| `research_agent/repairs/patch.py` | `e529782b1` | `research_agent/agents/coder_generation.py:20` imports `looks_like_executable_python` |
| `research_agent/reporting/benchmark_report_facts.py` | `e529782b1` | `research_agent/reporting/descriptive_report_facts.py:642` imports `compile_benchmark_report_facts` |
| `research_agent/reporting/descriptive_report_facts.py` | `e529782b1` | `research_agent/reporting/scientific_maturity.py:48` imports `concise_results_word_floor` |
| `research_agent/reporting/manuscript_method_facts.py` | `e529782b1` | `research_agent/reporting/benchmark_report_facts.py:58` imports `missing_bound_method_facts` |
| `research_agent/reporting/manuscript_provenance.py` | `e529782b1` | `research_agent/reporting/descriptive_report_facts.py:36` imports `verified_evidence_json` |
| `research_agent/reporting/report_fact.py` | `e529782b1` | `research_agent/reporting/descriptive_report_facts.py:35` imports `DescriptiveReportFact` |
| `research_agent/research_context/export_selection.py` | `e529782b1` | `research_agent/planning/population_compile.py:54` imports `SelectionStep, export_applied_selection, export_selection_counts` |
| `research_agent/research_context/prompt_scope.py` | `e529782b1` | `research_agent/resources/coder.py:42` imports `scoped_coder_context` |
| `research_agent/resources/__init__.py` | `e529782b1` | `research_agent/planning/preplan_know_how.py:11` imports `ResourceScheduler, ResourceSelectionQuery` |
| `research_agent/resources/capability.py` | `e529782b1` | `research_agent/resources/__init__.py:4` imports `CapabilityActivation, CapabilityApproval, CapabilityRequest, approved_capability_resource, build_capability_activation, build_capability_request, verify_capability_activation, runtime_snapshot_sha256, write_capability_activation, write_capability_approval, write_capability_request` |
| `research_agent/resources/capability_runtime.py` | `e529782b1` | `research_agent/resources/__init__.py:29` imports `CapabilityWorkflowRuntime` |
| `research_agent/resources/catalog.py` | `e529782b1` | `research_agent/resources/__init__.py:3` imports `ResourceCatalog, protocol_catalog_from_know_how` |
| `research_agent/resources/coder.py` | `e529782b1` | `research_agent/resources/__init__.py:17` imports `CODER_RESOURCE_BUNDLE_SCHEMA, CODER_RESOURCE_PROMPT_LIMIT_BYTES, CoderResourceBundle, CoderResourceIntegrityError, attach_coder_resources, attach_step_coder_input_authority, bind_materialized_coder_authority, bind_primary_cohort_role, build_coder_resource_bundle, persist_coder_resource_bundle` |
| `research_agent/resources/context.py` | `e529782b1` | `research_agent/resources/__init__.py:30` imports `AssembledContext, BoundedContextAssembler, ContextAssemblyReceipt, ContextBudgetExceeded, ContextSegment, bounded_request_metrics` |
| `research_agent/resources/scheduler.py` | `e529782b1` | `research_agent/resources/__init__.py:38` imports `ProtocolResourceSelection, ResourceScheduler, ResourceSelection` |
| `research_agent/resources/schema.py` | `e529782b1` | `research_agent/resources/__init__.py:39` imports `ResourceDescriptor, ResourceSelectionPolicy, ResourceSelectionQuery, ResourceSelectionReceipt` |

## Patch and ownership

- `figures/association_prior_outputs.py` and `figures/sensitivity_prior_outputs.py`: redirect the five helper imports to `figures/prior_output_support.py`.
- `figures/prior_output_support.py`: receive those helpers, their eight table-reading helpers and the existing binary-column exclusion tuple. The 13 function ASTs match their original bodies after symbol renaming; no statistical, labeling or source-selection rule changes.
- `reporting/publication_bundles.py`: import the relocated helpers and preserve all existing private aliases and public wrappers, including the aliases re-exported by `pipeline.py`.
- `tests/research_agent/authority/test_execution_kernel_identity.py`: add a repository-closure regression retaining both figure seeds and their support owner while excluding the host publication/readiness, grouping Planner, progressive compiler, provider retry and coder-resource workflows.

This uses an existing ungated sibling owner, adds no source module, and changes no Planner-visible field, finding code or schema version. It leaves manuscript-fact computation in its existing host owners. The kernel identity resolver and architecture baselines are untouched. The fix addresses both seed-to-host edges; changing just one leaves the host closure reachable through the other.

After both mutants were restored, the actual owner API reports **456 paths**, **zero newly added members**, **144 removed members**, and **zero of the 56 jump paths remaining**. Applying the lead's exact changed-path intersection command produces:

```text
src/easyicu/research_agent/figures/association_prior_outputs.py
src/easyicu/research_agent/figures/prior_output_support.py
src/easyicu/research_agent/figures/sensitivity_prior_outputs.py
```

`reporting/publication_bundles.py` was kernel-bound before this patch and is excluded afterward. Kernel files above were computed from the owner API, not inferred from their directory.

## Validation

All pytest calls used `PYTHONPATH=$PWD/src`, `$PY`, `-q -p no:cacheprovider` and small explicit file batches. Memory (`free -g` and cgroup `memory.current`) was checked before each batch. No xdist workers were used. Fixtures were synthetic; no patient data, network, Docker, browser or research run was used.

| Check / explicit test files | Result |
| --- | --- |
| `tests/research_agent/authority/test_execution_kernel_identity.py` + `tests/research_agent/figures/test_prior_output_figure_support.py` | **9 passed, 1 baseline failure** |
| `tests/research_agent/figures/test_association_figure_rescue.py` | **38 passed** |
| `tests/research_agent/core/test_deterministic_descriptive_absolute_risk.py` + `test_p2_b_fixes.py` + `test_capability_inventory_governance.py` (same directory) | **26 passed** |
| `tests/research_agent/reporting/test_publication_bundle_path_safety.py` + `test_a_concise_result_reports_every_executed_fact.py` (same directory) | **18 passed** |
| `tests/governance/test_research_agent_module_graph.py` + `test_arch_measure.py` + `test_static_architecture_policy.py` (same directory) | **52 passed, 1 deselected by repository configuration** |
| Selected post-fix tests, counting each executed test once | **143 passed, 1 baseline failure, 1 deselected** |
| New boundary regression, before the fix | **1 failed, 6 deselected**, reproducing the host closure |
| New boundary regression, unmutated before the mutants | **1 passed** (a repeat of a test counted above) |
| Restore the association renderer's host-facade import | **1 expected failure: mutant killed** |
| Restore the sensitivity renderer's host-facade import | **1 expected failure: mutant killed** |

**Mutants: 2 killed / 2 total.** The two incoming seed-to-host edges were restored separately, and the boundary regression killed each. Exact original file bytes were restored in `finally` blocks and checked by SHA-256. All 13 moved helper bodies were independently compared as ASTs, allowing only the symbol renames required by their new owner.

**Baseline attribution:** the unchanged `test_execution_kernel_manifest_tracks_runtime_not_control_plane` fails in the supplied clean base checkout too: **5 passed, 1 failed** for its original six-test file. The baseline first fails because `providers/structured_diagnostics.py` is present. The patch excludes that host module; the same test then reaches its later, still-failing expectation that `planning/progressive_contract.py` is absent. The computed manifests show that contract present in **every historical revision in the table**, before this patch. No new test failure was introduced. The first prescribed read-only baseline invocation encountered **6 setup errors** because Numba could not cache beside read-only source; rerunning with `NUMBA_CACHE_DIR` pointed inside this clone's `.codex_lane/tmp` enabled the baseline comparison without editing the base.

Other landing checks:

- `$RUFF check src tests`: **All checks passed**.
- `$PY tools/audit_capability_inventory.py`: **research-agent capability inventory: OK**.
- `$PY tools/arch_measure.py --diff .codex_lane/tmp/execution_phase.base.json`: **0 regressions**. That JSON was copied byte for byte with `git show <.codex_lane/base>:tools/arch_baselines/execution_phase.json`; the tracked baseline is unchanged. No emit or ratchet increase was used.
- `PATH="$(dirname "$PY"):$PATH" PYTHONDONTWRITEBYTECODE=1 bash .codex_lane/sealed_paths_check.sh HEAD`: **exit 0**, six changed paths, none sealed. The venv on `PATH` supplies Pydantic, which the system `python3` lacks.
- `git diff --check`: **exit 0**.
- Proposed commit message: `.codex_lane/commit-message.txt`.

Lead follow-up: run the two ASGI files listed below and resolve/review the pre-existing `progressive_contract.py` manifest expectation before requiring this owner file to be entirely green. The patch changes kernel bytes, so its normal Runner image identity must be rebuilt when it lands; no image build was started here.

## ASGI files for the lead to run on the Mac

The server sandbox blocks TestClient/ASGI socket operations. These adjacent route/artifact checks were deliberately not run here:

tests/webserver/test_agent_run_artifact_privacy.py
tests/webserver/test_run_files_name_runs_by_ref.py
