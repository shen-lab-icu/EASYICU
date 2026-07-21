from __future__ import annotations

import json
import re
from pathlib import Path

from easyicu.research_agent.repairs import source as code_repair
from easyicu.research_agent.repair_registry import (
    InvariantStatus,
    RepairClass,
    RepairExecutionPolicy,
    RepairLedger,
    RepairObservedState,
    automatic_repair_allowed,
    assert_registry_invariants,
    evaluate_invariants,
    is_sealed_renderer_repair,
    make_repair_provenance,
    repair_metadata_for,
)
from easyicu.research_agent.repairs import coordination as repair_coordination


def test_repair_registry_invariants_hold() -> None:
    assert_registry_invariants()


def test_literal_code_repair_ids_are_classified() -> None:
    source = Path(code_repair.__file__).read_text(encoding="utf-8")
    repair_ids = set(re.findall(r"repair_name\s*=\s*['\"]([^'\"]+)['\"]", source))
    assert repair_ids
    unclassified = [
        repair_id
        for repair_id in sorted(repair_ids)
        if repair_metadata_for(repair_id).classification_source.startswith("fallback:")
    ]
    assert unclassified == []


def test_dynamic_repair_id_patterns_are_classified() -> None:
    assert (
        repair_metadata_for("strip_fake_easyicu_import_easyicu_foo_v1").repair_class
        is RepairClass.SYNTACTIC
    )
    assert (
        repair_metadata_for(
            "undefined_helper_stub_to_json_serializable_v1"
        ).repair_class
        is RepairClass.METHOD_SUBSTITUTION
    )
    assert not automatic_repair_allowed("undefined_helper_stub_to_json_serializable_v1")


def test_retired_case_specific_repair_ids_are_not_registered() -> None:
    retired = [
        "generic_v15_table_one_fallback_v1",
        "age_stratified_mortality_dependency_free_v1",
        "formula_dummy_name_fallback_v1",
        "norepinephrine_dose_response_dependency_free_v1",
        "ordinal_primary_association_fallback_v1",
        "robustness_complete_case_or_fallback_v1",
        "shock_primary_assoc_sklearn_v1",
    ]

    for repair_id in retired:
        metadata = repair_metadata_for(repair_id)
        assert metadata.repair_class is RepairClass.METHOD_SUBSTITUTION
        assert metadata.classification_source == "fallback:unknown_method_substitution"
        assert metadata.requires_disclosure is True


def test_nonconvergence_fallback_is_method_substitution() -> None:
    metadata = repair_metadata_for("validation_nonconvergence_fallback_v1")
    assert metadata.repair_class is RepairClass.METHOD_SUBSTITUTION
    assert metadata.introduces_numbers is True
    assert metadata.requires_disclosure is True


def test_primary_row_selection_is_method_substitution_and_auto_denied() -> None:
    metadata = repair_metadata_for("categorical_primary_association_selection_v1")
    assert metadata.repair_class is RepairClass.METHOD_SUBSTITUTION
    assert not automatic_repair_allowed("categorical_primary_association_selection_v1")


def test_repair_ledger_writes_provenance_json(tmp_path: Path) -> None:
    ledger = RepairLedger(tmp_path / "repairs_applied.json")
    provenance = ledger.append_application(
        repair_id="dedupe_required_cols_outcome_v1",
        step_id="04_model",
        trigger={"error_type": "ValueError"},
        transformation="Removed a duplicate outcome column reference.",
        before_text="before",
        after_text="after",
    )
    assert provenance.repair_class == RepairClass.STRUCTURAL.value

    payload = json.loads((tmp_path / "repairs_applied.json").read_text())
    assert payload["schema_version"] == "easyicu.repair_ledger/1"
    assert payload["repairs"][0]["repair_id"] == "dedupe_required_cols_outcome_v1"
    assert payload["repairs"][0]["before_hash"].startswith("sha256:")
    assert payload["repairs"][0]["after_hash"].startswith("sha256:")


def test_make_repair_provenance_conservatively_classifies_unknown() -> None:
    provenance = make_repair_provenance(
        repair_id="future_unreviewed_repair_v1",
        step_id="01_step",
    )
    assert provenance.repair_class == RepairClass.METHOD_SUBSTITUTION.value
    assert provenance.classification_source == "fallback:unknown_method_substitution"
    assert provenance.requires_disclosure is True


# --- P1: runtime invariant verification -----------------------------------


def test_structural_invariant_unverified_without_state_is_not_a_pass() -> None:
    """The P0 honesty hole: a STRUCTURAL repair must not report a pass when no
    state was supplied to check ``row_set_unchanged`` / ``n_unchanged``."""

    provenance = make_repair_provenance(
        repair_id="dedupe_required_cols_outcome_v1",
        step_id="04_model",
    )
    assert provenance.invariant_status == InvariantStatus.UNVERIFIED.value
    assert provenance.invariants_passed is None


def test_structural_n_unchanged_verified_pass_and_fail() -> None:
    metadata = repair_metadata_for("dedupe_required_cols_outcome_v1")

    ok = evaluate_invariants(
        metadata,
        before_state=RepairObservedState(row_count=500, id_values=(1, 2, 3)),
        after_state=RepairObservedState(row_count=500, id_values=(1, 2, 3)),
    )
    assert ok.status == InvariantStatus.VERIFIED_PASS.value
    assert ok.passed is True
    assert ok.failures == ()

    dropped = evaluate_invariants(
        metadata,
        before_state=RepairObservedState(row_count=500, id_values=(1, 2, 3)),
        after_state=RepairObservedState(row_count=498, id_values=(1, 2)),
    )
    assert dropped.status == InvariantStatus.VERIFIED_FAIL.value
    assert dropped.passed is False
    assert "n_unchanged" in dropped.failures
    assert "row_set_unchanged" in dropped.failures


def test_syntactic_repair_has_no_invariants_and_passes_vacuously() -> None:
    metadata = repair_metadata_for("missing_os_import_v1")
    evaluation = evaluate_invariants(metadata)
    assert evaluation.status == InvariantStatus.VERIFIED_PASS.value
    assert evaluation.passed is True


def test_closed_counts_stable_keyword_repair_is_syntactic_and_automatic() -> None:
    metadata = repair_metadata_for("closed_counts_stable_keywords_v1")

    assert metadata.classification_source == "exact"
    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.invariants == ()
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)


def test_preflight_only_helper_repairs_are_syntactic_and_automatic() -> None:
    for repair_id in (
        "publication_export_audit_paths_v1",
        "boolean_reduction_identity_v1",
    ):
        metadata = repair_metadata_for(repair_id)

        assert metadata.classification_source == "exact"
        assert metadata.repair_class is RepairClass.SYNTACTIC
        assert metadata.invariants == ()
        assert metadata.introduces_numbers is False
        assert metadata.requires_disclosure is False
        assert automatic_repair_allowed(repair_id)


def test_local_read_hoist_is_syntactic_and_automatic() -> None:
    metadata = repair_metadata_for("local_read_before_assignment_hoist_v1")

    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.invariants == ()
    assert metadata.introduces_numbers is False
    assert automatic_repair_allowed(metadata.repair_id)


def test_merge_collision_guard_is_syntactic_and_automatic() -> None:
    metadata = repair_metadata_for("pandas_merge_dynamic_column_collision_guard_v1")

    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.invariants == ()
    assert metadata.introduces_numbers is False
    assert automatic_repair_allowed(metadata.repair_id)


def test_near_match_mapping_alias_is_syntactic_and_automatic() -> None:
    metadata = repair_metadata_for("undefined_mapping_near_match_alias_v1")

    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.invariants == ()
    assert metadata.introduces_numbers is False
    assert automatic_repair_allowed(metadata.repair_id)


def test_attrition_rule_id_canonicalization_is_syntactic_and_automatic() -> None:
    metadata = repair_metadata_for("attrition_rule_id_canonicalization_v1")

    assert metadata.classification_source == "exact"
    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.invariants == ()
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)


def test_measurement_provenance_summary_mapping_is_structural_and_automatic() -> None:
    for repair_id in (
        "measurement_provenance_summary_mapping_v1",
        "measurement_provenance_summary_mapping_v2",
        "measurement_provenance_host_receipts_v1",
    ):
        metadata = repair_metadata_for(repair_id)

        assert metadata.repair_class is RepairClass.STRUCTURAL
        assert metadata.introduces_numbers is False
        assert metadata.requires_disclosure is False
        assert automatic_repair_allowed(metadata.repair_id)


def test_observed_binary_domain_guard_is_structural_and_automatic() -> None:
    metadata = repair_metadata_for("observed_binary_primary_exposure_guard_v1")

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)


def test_llm_proven_numeric_domain_guards_are_structural_and_automatic() -> None:
    metadata = repair_metadata_for("llm_proven_numeric_domain_guards_v1")

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)


def test_authored_binary_feasibility_repair_is_structural_and_automatic() -> None:
    metadata = repair_metadata_for("binary_domain_authored_feasibility_v1")

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.invariants
    assert metadata.introduces_numbers is False
    assert automatic_repair_allowed(metadata.repair_id)


def test_provenance_checked_status_contract_is_structural_and_automatic() -> None:
    metadata = repair_metadata_for("provenance_checked_status_contract_v1")

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)


def test_sklearn_runtime_object_diagnostics_is_structural_and_automatic() -> None:
    metadata = repair_metadata_for("sklearn_runtime_object_diagnostics_v1")

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)


def test_all_method_substitutions_are_auto_denied() -> None:
    for repair_id in (
        "drop_overadjustment_covariates_v1",
        "dtype_coerce_v1",
        "primary_predictor_omitted_from_design_v1",
        "rank_safe_statsmodels_design_v1",
        "logit_regularized_fit_v1",
        "logreg_impute_v1",
        "derived_analysis_cohort_materialization_v1",
        "filter_x_cols_after_dummy_encoding_v1",
        "strip_unknown_cols_from_list_literals_v1",
        "sex_binary_encode_for_logit_v1",
        "statsmodels_endog_exog_index_align_v1",
        "publication_contract_optional_v1",
        "undefined_helper_stub_fit_cox_v1",
        "summary_salvage_minimal_contract_v1",
        "table_one_descriptive_repair_v1",
        "outcome_incidence_descriptive_repair_v1",
        "future_unreviewed_repair_v1",
    ):
        assert (
            repair_metadata_for(repair_id).repair_class
            is RepairClass.METHOD_SUBSTITUTION
        )
        assert not automatic_repair_allowed(repair_id)


def test_every_generic_repair_entrypoint_crosses_central_authorization_gate() -> None:
    research_agent_root = Path(code_repair.__file__).resolve().parents[1]
    source = (research_agent_root / "execution/phase.py").read_text(encoding="utf-8")
    publication_figure_source = (
        research_agent_root / "execution" / "publication_figure.py"
    ).read_text(encoding="utf-8")

    assert source.count("_deterministic_summary_repair(") == 3
    assert source.count("deterministic_contract_repair(") == 1
    assert source.count("_deterministic_runner_repair(") == 1
    # A2 batch-1 moved the concept-audit repair behind
    # repair_coordination.authorized_deterministic_concept_repair, which
    # enforces the all-or-nothing central authorization via its mandatory
    # ``authorize`` callback. pipeline_execute must never call the raw
    # repair directly again.
    assert source.count("deterministic_concept_audit_repair(") == 0
    coordination_source = Path(repair_coordination.__file__).read_text(encoding="utf-8")
    assert coordination_source.count("deterministic_concept_audit_repair(") == 1
    assert "authorize(" in coordination_source
    assert "authorize=_authorize_automatic_repair" in source
    # Six historical code-candidate boundaries minus the extracted concept
    # helper, the rendering-only adapter, plus the local helper definition.
    # Case-plugin candidates share the runner boundary and therefore cannot
    # bypass it.
    assert source.count("_authorize_automatic_repair(") == 6
    assert publication_figure_source.count("_authorize_automatic_repair(") == 1
    assert (
        source.count("_authorize_automatic_repair(")
        + publication_figure_source.count("_authorize_automatic_repair(")
        == 7
    )
    assert "authorizer=lambda repair_id: _automatic_repair_authorized(" in source


def test_only_closed_source_figure_renderers_are_structural_and_automatic() -> None:
    for repair_id in (
        "ordered_category_distribution_publication_bundle_v1",
        "distribution_availability_publication_bundle_from_parent_outputs_v1",
        "absolute_risk_incidence_prevalence_publication_bundle_v1",
        "association_publication_bundle_from_planned_model_contract_v1",
        "cohort_flow_publication_bundle_from_parent_outputs_v1",
        "sensitivity_publication_bundle_from_locked_summary_v1",
    ):
        metadata = repair_metadata_for(repair_id)
        assert metadata.repair_class is RepairClass.STRUCTURAL, repair_id
        assert (
            metadata.execution_policy is RepairExecutionPolicy.SEALED_RENDERER
        ), repair_id
        assert is_sealed_renderer_repair(repair_id), repair_id
        assert metadata.figure_product_slots, repair_id
        assert metadata.planner_methods, repair_id
        assert metadata.planner_parent_output_role_groups, repair_id
        assert metadata.implementation_modules, repair_id
        assert "easyicu.research_agent.repair_registry" in (
            metadata.implementation_modules
        ), repair_id
        assert not automatic_repair_allowed(repair_id), repair_id
        assert automatic_repair_allowed(
            repair_id,
            sealed_renderer_wrapper=True,
        ), repair_id

    assert not is_sealed_renderer_repair("publication_bundle_promote_v1")
    assert (
        repair_metadata_for("publication_bundle_promote_v1").execution_policy
        is RepairExecutionPolicy.MUTABLE
    )

    for repair_id in (
        "publication_figure_renderer_from_parent_outputs_v1",
        "prediction_publication_bundle_from_parent_outputs_v1",
        "association_publication_bundle_from_parent_outputs_v2",
        "association_publication_bundle_from_parent_outputs_v3",
        "sensitivity_publication_bundle_from_parent_outputs_v2",
        "survival_publication_bundle_from_parent_outputs_v1",
        "cohort_overlap_publication_bundle_from_parent_outputs_v1",
        "missingness_publication_bundle_from_parent_outputs_v1",
        "phenotype_publication_bundle_from_parent_outputs_v1",
        "descriptive_publication_bundle_from_parent_outputs_v1",
        "absolute_risk_publication_bundle_from_parent_outputs_v1",
    ):
        metadata = repair_metadata_for(repair_id)
        assert metadata.repair_class is RepairClass.METHOD_SUBSTITUTION, repair_id
        assert not automatic_repair_allowed(repair_id), repair_id


def test_ledger_records_invariant_status(tmp_path: Path) -> None:
    ledger = RepairLedger(tmp_path / "repairs_applied.json")
    ledger.append_application(
        repair_id="dedupe_required_cols_outcome_v1",
        step_id="04_model",
        before_state=RepairObservedState(row_count=500),
        after_state=RepairObservedState(row_count=480),
    )
    payload = json.loads((tmp_path / "repairs_applied.json").read_text())
    record = payload["repairs"][0]
    assert record["invariant_status"] == InvariantStatus.VERIFIED_FAIL.value
    assert "n_unchanged" in record["invariant_failures"]


# --- P1.5: step-summary salvage is inside the provenance net ----------------


def test_summary_salvage_repairs_are_classified() -> None:
    assert (
        repair_metadata_for("summary_salvage_stdout_json_v1").repair_class
        is RepairClass.STRUCTURAL
    )
    assert (
        repair_metadata_for("summary_salvage_named_json_v1").repair_class
        is RepairClass.STRUCTURAL
    )
    minimal = repair_metadata_for("summary_salvage_minimal_contract_v1")
    assert minimal.repair_class is RepairClass.METHOD_SUBSTITUTION
    # None of the salvage ids may fall through to the conservative
    # unknown -> METHOD_SUBSTITUTION classification.
    for repair_id in (
        "summary_salvage_stdout_json_v1",
        "summary_salvage_named_json_v1",
        "summary_salvage_minimal_contract_v1",
    ):
        assert not repair_metadata_for(repair_id).classification_source.startswith(
            "fallback:"
        )


def test_summary_salvage_minimal_contract_is_not_auto_authorized() -> None:
    provenance = make_repair_provenance(
        repair_id="summary_salvage_minimal_contract_v1",
        step_id="03_model",
        selection_rule="first non-const association row; mean of perf rows",
    )
    assert provenance.repair_class == RepairClass.METHOD_SUBSTITUTION.value
    assert provenance.selection_rule
    assert provenance.requires_disclosure is True
    assert not automatic_repair_allowed(provenance.repair_id)


# --- P1.5 integration: real salvage -> classified outcome -> ledger entry ---


def test_salvage_step_summary_records_stdout_salvage_end_to_end(tmp_path: Path) -> None:
    from easyicu.research_agent.contracts.runtime import RunResult
    from easyicu.research_agent.schema import AnalysisStep
    from easyicu.research_agent.repairs.summary import salvage_step_summary

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    run_result = RunResult(
        step_id="01_assoc",
        script_path=tmp_path / "analysis.py",
        cwd=tmp_path,
        out_dir=out_dir,
        stdout='noise\n{"primary_or": 1.4, "sample_size": 500}\n',
        stderr="",
        returncode=0,
        duration_seconds=0.1,
    )
    step = AnalysisStep(
        step_id="01_assoc", intent="assoc", expected_outputs=["statistic:primary_or"]
    )

    outcome = salvage_step_summary(run_result, step=step)
    assert outcome is not None
    assert outcome.repair_id == "summary_salvage_stdout_json_v1"
    assert outcome.reset_artefacts is True
    assert repair_metadata_for(outcome.repair_id).repair_class is RepairClass.STRUCTURAL
    # Salvage actually wrote the summary the registration step will read.
    assert (out_dir / "step_summary.json").exists()

    # Feeding the outcome to the ledger (what the execute-phase closure does)
    # produces a real provenance entry — closing the wiring gap.
    ledger = RepairLedger(tmp_path / "repairs_applied.json")
    ledger.append_application(
        repair_id=outcome.repair_id,
        step_id=step.step_id,
        trigger={"source": "summary_salvage", "reason": outcome.trigger_reason},
        transformation=outcome.transformation,
        selection_rule=outcome.selection_rule,
    )
    payload = json.loads((tmp_path / "repairs_applied.json").read_text())
    assert payload["repairs"][0]["repair_id"] == "summary_salvage_stdout_json_v1"
    assert payload["repairs"][0]["repair_class"] == RepairClass.STRUCTURAL.value


def test_salvage_step_summary_does_not_select_from_result_tables(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.contracts.runtime import RunResult
    from easyicu.research_agent.schema import AnalysisStep
    from easyicu.research_agent.repairs.summary import salvage_step_summary

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "step_summary.json").write_text("{}", encoding="utf-8")
    (out_dir / "table_one.csv").write_text(
        "variable,median\nage,65.0\nsofa2,7.0\n", encoding="utf-8"
    )
    run_result = RunResult(
        step_id="01_table_one",
        script_path=tmp_path / "analysis.py",
        cwd=tmp_path,
        out_dir=out_dir,
        stdout="",
        stderr="",
        returncode=0,
        duration_seconds=0.1,
    )
    step = AnalysisStep(
        step_id="01_table_one", intent="t1", expected_outputs=["table:table_one"]
    )

    outcome = salvage_step_summary(run_result, step=step)
    assert outcome is None
    assert (out_dir / "step_summary.json").read_text(encoding="utf-8") == "{}"


def test_salvage_step_summary_returns_none_when_summary_present(tmp_path: Path) -> None:
    from easyicu.research_agent.contracts.runtime import RunResult
    from easyicu.research_agent.schema import AnalysisStep
    from easyicu.research_agent.repairs.summary import salvage_step_summary

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "step_summary.json").write_text('{"primary_or": 1.2}', encoding="utf-8")
    run_result = RunResult(
        step_id="01_assoc",
        script_path=tmp_path / "analysis.py",
        cwd=tmp_path,
        out_dir=out_dir,
        stdout="",
        stderr="",
        returncode=0,
        duration_seconds=0.1,
    )
    step = AnalysisStep(step_id="01_assoc", intent="assoc", expected_outputs=[])
    assert salvage_step_summary(run_result, step=step) is None
