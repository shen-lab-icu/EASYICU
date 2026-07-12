"""Regression tests for deterministic preflight role ownership."""

import inspect
import json
from pathlib import Path

from easyicu.research_agent import pipeline_execute

from easyicu.research_agent.pipeline_execute import (
    _absolute_risk_context_runner_owns_step,
    _cohort_definition_overlap_runner_owns_step,
    _cohort_definition_sensitivity_runner_owns_step,
    _is_terminal_publication_figure_repair_step,
    _primary_cohort_flow_runner_owns_step,
    _repair_publication_figure_in_staging,
    _robustness_sensitivity_runner_owns_step,
    _simple_missingness_audit_runner_owns_step,
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
        renderer=_renders,
    )

    assert repaired == "source_backed_figure_v1"
    assert not (out_dir / "agent_figure.png").exists()
    assert (out_dir / "publication_figure.png").read_bytes() == b"new"
    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["figure_path"] == str(out_dir / "publication_figure.png")


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
