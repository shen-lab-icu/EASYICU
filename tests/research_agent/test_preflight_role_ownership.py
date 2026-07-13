"""Regression tests for deterministic preflight role ownership."""

import inspect
import json
from pathlib import Path

import pytest

from easyicu.research_agent import pipeline_execute
from easyicu.research_agent.pipeline import _sealed_renderer_figure_step_matches_parent
from easyicu.research_agent.schema import AnalysisStep

from easyicu.research_agent.pipeline_execute import (
    _absolute_risk_context_runner_owns_step,
    _cohort_definition_overlap_runner_owns_step,
    _cohort_definition_sensitivity_runner_owns_step,
    _detached_figure_repair_binding,
    _is_terminal_publication_figure_repair_step,
    _primary_cohort_flow_runner_owns_step,
    _repair_publication_figure_in_staging,
    _robustness_sensitivity_runner_owns_step,
    _sealed_typed_figure_products,
    _should_attempt_detached_figure_binding,
    _simple_missingness_audit_runner_owns_step,
    _step_has_figure_only_output_contract,
    _terminal_publication_repair_replan_skip_detail,
    _unowned_sealed_authority_markers,
)


@pytest.mark.parametrize(
    ("repair_id", "planner_method", "parent_outputs"),
    (
        (
            "ordered_category_distribution_publication_bundle_v1",
            "ordinal_exposure_derivation_and_quality_control",
            ["table:marker_distribution"],
        ),
        (
            "distribution_availability_publication_bundle_from_parent_outputs_v1",
            "exposure_distribution_and_missingness_audit",
            ["table:marker_distribution", "table:marker_measurement_audit"],
        ),
        (
            "cohort_flow_publication_bundle_from_parent_outputs_v1",
            "cohort_definition",
            ["table:cohort_flow", "table:attrition"],
        ),
        (
            "sensitivity_publication_bundle_from_locked_summary_v1",
            "cohort_definition_sensitivity",
            ["table:robustness_summary"],
        ),
    ),
)
def test_every_sealed_renderer_requires_a_planner_owned_child_edge(
    monkeypatch,
    repair_id,
    planner_method,
    parent_outputs,
):
    import easyicu.research_agent.pipeline as pipeline_module

    parent_inputs = ["artifact:locked_cohort", "marker_value"]
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_manifest_step",
        lambda run_dir, step_id: {
            "method": planner_method,
            "inputs": parent_inputs,
            "expected_outputs": parent_outputs,
        },
    )
    modern = AnalysisStep(
        step_id="02_parent_figure",
        intent="Render only the Planner-owned parent products.",
        inputs=parent_outputs,
        expected_outputs=["figure:planned_display"],
        method="publication_figure_generation",
    )
    unrelated = modern.model_copy(update={"inputs": ["table:unrelated_result"]})
    legacy = modern.model_copy(
        update={"inputs": parent_inputs, "method": planner_method}
    )

    assert _sealed_renderer_figure_step_matches_parent(
        Path("/unused"), modern, repair_id
    )
    assert _sealed_renderer_figure_step_matches_parent(
        Path("/unused"), legacy, repair_id
    )
    assert not _sealed_renderer_figure_step_matches_parent(
        Path("/unused"), unrelated, repair_id
    )


def test_failed_staged_figure_repair_preserves_agent_exports(tmp_path: Path):
    out_dir = tmp_path / "steps" / "05_result_figure" / "outputs"
    out_dir.mkdir(parents=True)
    sentinel = out_dir / "agent_figure.png"
    source = out_dir / "agent_figure_source_data.csv"
    contract = out_dir / "agent.figure_contract.json"
    sentinel.write_bytes(b"agent-png")
    source.write_text("x,y\n1,2\n", encoding="utf-8")
    contract.write_text('{"source_data":"agent_figure_source_data.csv"}')

    def _declines(**kwargs):
        staging = Path(kwargs["out_dir"])
        (staging / "partial.json").write_text("{}", encoding="utf-8")
        return None

    repaired = _repair_publication_figure_in_staging(
        run_dir=tmp_path,
        current_step_id="05_result_figure",
        out_dir=out_dir,
        authorizer=lambda _repair_id: True,
        renderer=_declines,
    )

    assert repaired is None
    assert sentinel.read_bytes() == b"agent-png"
    assert source.exists() and contract.exists()
    assert not (out_dir / "partial.json").exists()


def test_successful_staged_figure_repair_replaces_bundle_and_rewrites_paths(
    tmp_path: Path,
):
    out_dir = tmp_path / "steps" / "05_result_figure" / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "agent_figure.png").write_bytes(b"old")

    def _renders(**kwargs):
        staging = Path(kwargs["out_dir"])
        (staging / "publication_figure.png").write_bytes(b"new")
        (staging / "step_summary.json").write_text(
            json.dumps({"figure_path": str(staging / "publication_figure.png")}),
            encoding="utf-8",
        )
        return "source_backed_figure_v1"

    repaired = _repair_publication_figure_in_staging(
        run_dir=tmp_path,
        current_step_id="05_result_figure",
        out_dir=out_dir,
        authorizer=lambda _repair_id: True,
        renderer=_renders,
    )

    assert repaired == "source_backed_figure_v1"
    assert not (out_dir / "agent_figure.png").exists()
    assert (out_dir / "publication_figure.png").read_bytes() == b"new"
    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["figure_path"] == str(out_dir / "publication_figure.png")


def test_staged_figure_repair_needs_authorization_before_install(tmp_path: Path):
    out_dir = tmp_path / "steps" / "05_result_figure" / "outputs"
    out_dir.mkdir(parents=True)
    sentinel = out_dir / "agent_figure.png"
    sentinel.write_bytes(b"agent")

    def _renders(**kwargs):
        staging = Path(kwargs["out_dir"])
        (staging / "publication_figure.png").write_bytes(b"generated")
        return "unreviewed_figure_transform_v1"

    repaired = _repair_publication_figure_in_staging(
        run_dir=tmp_path,
        current_step_id="05_result_figure",
        out_dir=out_dir,
        renderer=_renders,
        authorizer=lambda _repair_id: False,
    )

    assert repaired is None
    assert sentinel.read_bytes() == b"agent"
    assert not (out_dir / "publication_figure.png").exists()


def test_staging_cannot_install_a_sealed_renderer_even_if_authorizer_allows_it(
    tmp_path: Path,
):
    out_dir = tmp_path / "steps" / "05_result_figure" / "outputs"
    out_dir.mkdir(parents=True)
    sentinel = out_dir / "agent_figure.png"
    sentinel.write_bytes(b"agent")

    def _renders(**kwargs):
        staging = Path(kwargs["out_dir"])
        (staging / "publication_figure.png").write_bytes(b"generated")
        return "distribution_availability_publication_bundle_from_parent_outputs_v1"

    repaired = _repair_publication_figure_in_staging(
        run_dir=tmp_path,
        current_step_id="05_result_figure",
        out_dir=out_dir,
        renderer=_renders,
        authorizer=lambda _repair_id: True,
    )

    assert repaired is None
    assert sentinel.read_bytes() == b"agent"
    assert not (out_dir / "publication_figure.png").exists()


def test_sealed_renderer_requires_typed_logical_products_not_bare_exports():
    assert _sealed_typed_figure_products(
        ["figure:marker_distribution", "figure:marker_availability"]
    ) == ["figure:marker_distribution", "figure:marker_availability"]
    assert _sealed_typed_figure_products(["marker.png", "marker.svg"]) is None
    assert (
        _sealed_typed_figure_products(["figure:marker_distribution", "marker.svg"])
        is None
    )


def test_generated_code_cannot_self_declare_sealed_authority():
    summary = {
        "sealed_renderer_repair": (
            "distribution_availability_publication_bundle_from_parent_outputs_v1"
        ),
        "sealed_renderer_implementation_sha256": "0" * 64,
        "sealed_renderer_parent_digests": {"step_summary.json": "1" * 64},
        "planner_product_slot_bindings": {},
    }

    assert set(
        _unowned_sealed_authority_markers(
            summary,
            authorized_code_sha256=None,
        )
    ) == set(summary)
    assert (
        _unowned_sealed_authority_markers(
            summary,
            authorized_code_sha256="a" * 64,
        )
        == []
    )


def test_sealed_renderer_never_enters_detached_binding(tmp_path: Path):
    (tmp_path / "figure.png").write_bytes(b"rendered")

    assert _should_attempt_detached_figure_binding(
        out_dir=tmp_path,
        sealed_renderer_authorized_code_sha256=None,
    )
    assert not _should_attempt_detached_figure_binding(
        out_dir=tmp_path,
        sealed_renderer_authorized_code_sha256="a" * 64,
    )


def test_locked_primary_cohort_flow_is_owned_by_deterministic_runner():
    assert _primary_cohort_flow_runner_owns_step(
        "cohort_definition",
        "01_primary_cohort_flow",
        "Define and freeze the primary cohort with explicit attrition.",
        ["table:cohort_attrition", "table:cohort_denominators"],
    )


def test_alternative_cohort_comparison_is_not_primary_flow():
    assert not _primary_cohort_flow_runner_owns_step(
        "cohort_definition_sensitivity",
        "07_cohort_definition_sensitivity_comparison",
        "Compare alternative cohort definitions and overlap.",
        ["table:cohort_attrition", "table:cohort_overlap"],
    )


def test_cohort_flow_output_name_must_match_the_closed_contract():
    assert not _primary_cohort_flow_runner_owns_step(
        "cohort_definition",
        "01_primary_cohort_flow",
        "Build prediction features after cohort setup.",
        ["table:cohort_flow_prediction_features"],
    )
    assert not _primary_cohort_flow_runner_owns_step(
        "cohort_definition",
        "01_primary_cohort_flow",
        "Define the cohort.",
        ["table:cohort_flow", "patient_level_dataset.parquet"],
    )


def test_simple_missingness_audit_is_owned_by_compact_runner():
    assert _simple_missingness_audit_runner_owns_step(
        "data_quality_audit",
        "02_measurement_audit",
        "Audit measurement-process missingness by concept.",
        ["table:missingness_audit"],
    )


def test_missingness_audit_with_analytic_denominator_is_owned():
    assert _simple_missingness_audit_runner_owns_step(
        "data_quality_audit",
        "03_data_quality_and_missingness",
        "Audit measurement availability and report complete-case denominators.",
        [
            "table:missingness_measurement_audit",
            "table:analytic_denominators",
        ],
    )


def test_missingness_runner_rejects_unowned_data_or_test_contracts():
    assert not _simple_missingness_audit_runner_owns_step(
        "data_quality_audit",
        "03_data_quality_and_missingness",
        "Audit missingness and validate preprocessing.",
        [
            "table:missingness_measurement_audit",
            "data:preprocessing_manifest",
        ],
    )
    assert not _simple_missingness_audit_runner_owns_step(
        "data_quality_audit",
        "03_data_quality_and_missingness",
        "Audit missingness and run a mechanism test.",
        ["table:missingness_measurement_audit", "test:missingness_mechanism"],
    )


def test_rich_exposure_repair_is_not_reduced_to_missingness_counts():
    assert not _simple_missingness_audit_runner_owns_step(
        "evidence_repair",
        "04_exposure_evidence_repair",
        "Repair the exposure and missingness evidence contract.",
        [
            "table:missingness_audit",
            "table:joint_availability_audit",
            "table:invalid_range_audit",
            "table:model_availability_audit",
            "table:source_reconciliation",
        ],
    )


def test_compact_missingness_runner_rejects_foreign_reconciliation_contract():
    assert not _simple_missingness_audit_runner_owns_step(
        "data_quality_audit",
        "04_absolute_risk_context_reconciliation",
        "Reconcile exposure representation and source-aware risks.",
        [
            "table:absolute_risk_representation_reconciliation",
            "table:reconciled_absolute_risk",
            "log:representation_gap_notes",
        ],
    )


def test_missingness_prose_does_not_claim_primary_effect_step():
    assert not _simple_missingness_audit_runner_owns_step(
        "logistic_regression",
        "05_primary_adjusted_model",
        "Fit the adjusted association with a complete-case missingness check.",
        ["table:adjusted_odds_ratio"],
    )


def test_missingness_output_name_must_match_the_closed_contract():
    assert not _simple_missingness_audit_runner_owns_step(
        "missingness_audit",
        "02_missingness",
        "Fit an imputation model.",
        ["table:missingness_imputation_model"],
    )
    assert not _simple_missingness_audit_runner_owns_step(
        "missingness_audit",
        "02_missingness",
        "Audit missingness.",
        ["table:missingness_audit", "representation_gap_notes.csv"],
    )


def test_cohort_attrition_owner_is_not_claimed_by_downstream_comparator():
    assert not _cohort_definition_sensitivity_runner_owns_step(
        "descriptive_comparison",
        "05_overlap_and_attrition",
        "Create overlap and attrition evidence for alternative definitions.",
        ["table:cohort_overlap", "table:cohort_definition_attrition"],
    )


def test_cohort_sensitivity_science_scripts_are_never_preflight_owners():
    assert not _cohort_definition_sensitivity_runner_owns_step(
        "cohort_definition_sensitivity",
        "08_definition_sensitivity",
        "Compare estimates across already registered cohort definitions.",
        ["table:sensitivity_grid", "table:outcome_by_definition"],
    )
    assert not _cohort_definition_overlap_runner_owns_step(
        "cohort_definition_overlap",
        ["table:alternative_cohort_attrition", "table:cohort_overlap_matrix"],
    )


def test_absolute_risk_owner_matches_structured_output_contract():
    assert _absolute_risk_context_runner_owns_step(
        "descriptive_context",
        "06_context",
        ["table:exposure_outcome_summary"],
    )


def test_absolute_risk_runner_rejects_figure_and_primary_effect_contracts():
    assert not _absolute_risk_context_runner_owns_step(
        "absolute_risk_context",
        "06_absolute_risk_context_figure",
        ["figure:absolute_risk_context"],
    )
    assert not _absolute_risk_context_runner_owns_step(
        "absolute_risk_context",
        "07_primary_model",
        ["table:adjusted_odds_ratio"],
    )
    assert not _absolute_risk_context_runner_owns_step(
        "data_quality_audit",
        "06_absolute_risk_context_reconciliation",
        [
            "table:absolute_risk_representation_reconciliation",
            "table:reconciled_absolute_risk",
            "log:representation_gap_notes",
        ],
    )
    assert not _absolute_risk_context_runner_owns_step(
        "absolute_risk_context",
        "06_context_with_foreign_product",
        ["table:absolute_risk", "table:subgroup_interactions"],
    )
    assert not _absolute_risk_context_runner_owns_step(
        "absolute_risk_context",
        "06_context_with_bare_foreign_product",
        ["table:absolute_risk", "negative_control_outcomes.csv"],
    )


def test_robustness_runner_matches_separate_structured_comparison():
    assert _robustness_sensitivity_runner_owns_step(
        "prespecified_robustness",
        "08_sensitivity_comparison",
        [
            "table:robustness_matrix",
            "table:robustness_summary",
            "statistic:complete_case_n",
        ],
    )
    assert not _robustness_sensitivity_runner_owns_step(
        "cohort_definition_sensitivity",
        "06_cohort_definition_sensitivity",
        [
            "table:outcome_by_definition",
            "table:sensitivity_grid",
            "table:robustness_matrix",
        ],
    )
    assert not _cohort_definition_sensitivity_runner_owns_step(
        "cohort_definition_sensitivity",
        "06_cohort_definition_sensitivity",
        "Compare outcomes across registered cohort definitions.",
        [
            "table:outcome_by_definition",
            "table:sensitivity_grid",
            "table:robustness_matrix",
        ],
    )


def test_alias_rich_cohort_sensitivity_contract_remains_agent_owned():
    outputs = [
        "table:cohort_definition_overlap_attrition",
        "table:sensitivity_comparison",
        "table:sensitivity_specification_matrix",
        "statistic:primary_or",
        "statistic:complete_case_n",
        "table:robustness_summary",
        "log:missingness_strategy_notes",
    ]

    assert not _robustness_sensitivity_runner_owns_step(
        "cohort_definition_sensitivity",
        "07_cohort_definition_sensitivity_comparison",
        outputs,
    )
    assert not _cohort_definition_sensitivity_runner_owns_step(
        "cohort_definition_sensitivity",
        "07_cohort_definition_sensitivity_comparison",
        "Execute locked cohort, missingness, and outcome variants.",
        outputs,
    )


def test_robustness_runner_rejects_primary_and_figure_owners():
    assert not _robustness_sensitivity_runner_owns_step(
        "logistic_regression",
        "07_primary_adjusted_model",
        ["table:robustness_summary", "statistic:complete_case_n"],
    )
    assert not _robustness_sensitivity_runner_owns_step(
        "prespecified_robustness",
        "08_sensitivity_comparison_figure",
        ["figure:robustness_forest"],
    )
    assert not _robustness_sensitivity_runner_owns_step(
        "prespecified_robustness",
        "08_forecast",
        ["table:sensitivity_grid_forecast"],
    )
    assert not _robustness_sensitivity_runner_owns_step(
        "prespecified_robustness",
        "08_mixed_contract",
        ["table:robustness_matrix", "table:negative_control_outcomes"],
    )
    assert not _robustness_sensitivity_runner_owns_step(
        "prespecified_robustness",
        "08_bare_mixed_contract",
        ["table:robustness_matrix", "negative_control_outcomes.csv"],
    )


def test_primary_estimands_and_cohort_selection_are_not_preflight_dispatched():
    source = inspect.getsource(pipeline_execute.run_execute_phase)
    for forbidden in (
        "primary_cohort_flow_preflight",
        "cohort_definition_overlap_preflight",
        "cohort_definition_sensitivity_preflight",
        "survival_primary_analysis_preflight",
        "causal_primary_analysis_preflight",
        "ordinal_dose_response_preflight",
        "trajectory_clustering_coder_failed",
    ):
        assert forbidden not in source


def test_terminal_rendering_skip_requires_exact_method_and_figure_only_outputs():
    from easyicu.research_agent.schema import AnalysisStep

    rendering = AnalysisStep(
        step_id="08_publication_figure_repair",
        intent="Render the registered primary result bundle.",
        method="rendering_only_repair_from_primary_results",
        expected_outputs=["figure:publication_figure"],
    )
    assert _is_terminal_publication_figure_repair_step(rendering)

    scientific_repair = rendering.model_copy(
        update={
            "intent": "Repair the adjusted model; this is not rendering-only.",
            "method": "mixed_effects_regression",
            "expected_outputs": [
                "table:association_estimates",
                "figure:publication_figure",
            ],
        }
    )
    assert not _is_terminal_publication_figure_repair_step(scientific_repair)


def test_publication_renderer_cannot_preflight_or_replace_mixed_scientific_contract():
    from easyicu.research_agent.schema import AnalysisStep

    figure_only = AnalysisStep(
        step_id="02_model_figure",
        intent="Render the registered model result.",
        method="publication_figure_generation",
        expected_outputs=["figure:publication_figure"],
    )
    mixed = figure_only.model_copy(
        update={
            "expected_outputs": [
                "table:primary_association",
                "figure:publication_figure",
            ]
        }
    )

    assert _step_has_figure_only_output_contract(figure_only)
    assert not _step_has_figure_only_output_contract(mixed)
    production_source = inspect.getsource(pipeline_execute.run_execute_phase)
    assert "if not _step_has_figure_only_output_contract(step):" in production_source
    assert "and _step_has_figure_only_output_contract(step)" in production_source


def test_figure_like_product_names_do_not_launder_typed_scientific_outputs():
    from easyicu.research_agent.schema import AnalysisStep

    base = AnalysisStep(
        step_id="02_model_figure",
        intent="Render the registered model result.",
        method="publication_figure_generation",
        expected_outputs=["figure:publication_figure"],
    )
    for scientific_output in (
        "table:figure_model_estimates",
        "model:forest_plot_model",
        "statistic:chart_effect_estimate",
    ):
        mixed = base.model_copy(
            update={
                "expected_outputs": [
                    "figure:publication_figure",
                    scientific_output,
                ]
            }
        )
        assert not _step_has_figure_only_output_contract(mixed), scientific_output


def test_terminal_repair_skip_uses_latest_status_for_each_step(tmp_path: Path):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    model = AnalysisStep(
        step_id="01_model",
        intent="Fit the agent-selected primary model.",
        method="agent_selected_model",
        expected_outputs=["table:primary_estimate"],
    )
    rendering = AnalysisStep(
        step_id="01_model_figure",
        intent="Render the registered parent result.",
        method="rendering_only_repair_from_primary_results",
        expected_outputs=["figure:publication_figure"],
    )
    outputs = tmp_path / "steps" / model.step_id / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "publication_figure.png").write_bytes(b"old")
    (outputs / "publication_figure.figure_contract.json").write_text(
        json.dumps(
            {
                "panels": [
                    {"role": "descriptive_result"},
                    {"role": "primary_estimand"},
                ],
                "export_formats": ["png"],
            }
        ),
        encoding="utf-8",
    )
    plan = AnalysisPlan(
        research_question="A neutral primary analysis.",
        steps=[model, rendering],
    )

    detail = _terminal_publication_repair_replan_skip_detail(
        plan=plan,
        completed_records=[
            {"step_id": model.step_id, "status": "ok"},
            {"step_id": model.step_id, "status": "contract_failed"},
        ],
        run_dir=tmp_path,
    )

    assert detail is None


def test_detached_repair_binding_comes_from_plan_and_current_outer_ledger():
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    source = AnalysisStep(
        step_id="01_model",
        intent="Fit the agent-selected model.",
        method="agent_selected_model",
        expected_outputs=["table:primary_estimate"],
    )
    target = AnalysisStep(
        step_id="01_model_figure",
        intent="Render the model result.",
        method="publication_figure_generation",
        expected_outputs=["figure:publication_figure"],
    )
    repair = AnalysisStep(
        step_id="09_detached_figure_repair",
        intent="Render registered results only.",
        inputs=[target.step_id],
        method="rendering_only_repair_from_primary_results",
        expected_outputs=["figure:publication_figure"],
    )
    plan = AnalysisPlan(
        research_question="A neutral source-backed result.",
        steps=[source, target, repair],
    )
    records = [
        {"step_id": source.step_id, "status": "ok", "evidence_ids": ["ev_source"]},
        {"step_id": target.step_id, "status": "execution_failed"},
    ]

    assert _detached_figure_repair_binding(
        step=repair, plan=plan, completed_records=records
    ) == (target.step_id, source.step_id, ["ev_source"])
    # A later failed source checkpoint supersedes its historical success.
    assert (
        _detached_figure_repair_binding(
            step=repair,
            plan=plan,
            completed_records=[
                *records,
                {"step_id": source.step_id, "status": "contract_failed"},
            ],
        )
        is None
    )

    production_source = inspect.getsource(pipeline_execute.run_execute_phase)
    compact_source = "".join(production_source.split())
    assert 'step_record["repair_target_step_id"]' in production_source
    assert (
        "lineage_input_evidence_ids=list(dict.fromkeys("
        "[*resolved_input_evidence_ids,*repair_source_evidence_ids]))" in compact_source
    )
    assert "inputs=lineage_input_evidence_idsorNone" in compact_source
    assert (
        '"source_evidence_ids": list(repair_source_evidence_ids)' in production_source
    )


def test_detached_render_repair_cannot_bind_nonfigure_science_target() -> None:
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    source = AnalysisStep(
        step_id="01_source",
        intent="Create the exposure table.",
        method="descriptive",
        expected_outputs=["table:exposure"],
    )
    model = AnalysisStep(
        step_id="02_primary_model",
        intent="Fit using the exposure declared by step '01_source'.",
        method="mixed_effects_regression",
        expected_outputs=["table:association_estimates"],
    )
    repair = AnalysisStep(
        step_id="09_detached_figure_repair",
        intent="Render registered results only.",
        inputs=[model.step_id],
        method="rendering_only_repair_from_primary_results",
        expected_outputs=["figure:publication_figure"],
    )
    plan = AnalysisPlan(
        research_question="A neutral source-backed result.",
        steps=[source, model, repair],
    )

    assert (
        _detached_figure_repair_binding(
            step=repair,
            plan=plan,
            completed_records=[
                {
                    "step_id": source.step_id,
                    "status": "ok",
                    "evidence_ids": ["ev_source"],
                },
                {"step_id": model.step_id, "status": "contract_failed"},
            ],
        )
        is None
    )
