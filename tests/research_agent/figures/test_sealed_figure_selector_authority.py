"""Sealed figure selection requires host-authorized parent methods and products."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_sensitivity_exact_method_authorizes_only_verified_summary(monkeypatch):
    import easyicu.research_agent.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "_verified_direct_parent_table_names",
        lambda run_dir, step_id: {"robustness_summary.csv"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_manifest_analysis_request",
        lambda run_dir, step_id: {
            "step": {
                "method": "cohort_definition_sensitivity",
                "expected_outputs": ["table:robustness_summary"],
            },
            "analysis_family": "association_study",
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_analysis_method",
        lambda run_dir, step_id: "cohort_definition_sensitivity",
    )
    assert (
        pipeline_module.deterministic_figure_repair_id_for_upstream(
            Path("/unused"), "07_sensitivity_figure"
        )
        == "sensitivity_publication_bundle_from_locked_summary_v1"
    )


@pytest.mark.parametrize(
    (
        "planner_method",
        "planner_outputs",
        "reported_method",
        "reported_family",
        "verified_tables",
    ),
    (
        (
            "kaplan_meier_estimation",
            ["table:kaplan_meier_curve_distribution"],
            "ordinal_exposure_derivation_and_quality_control",
            "survival",
            {"severity_distribution.csv"},
        ),
        (
            "mixed_effects_regression",
            ["table:robustness_summary"],
            "cohort_definition_sensitivity",
            "association_study",
            {"robustness_summary.csv"},
        ),
        (
            "survival_analysis",
            ["table:cohort_flow", "table:attrition"],
            "survival_analysis",
            "cohort_definition",
            {"cohort_flow.csv", "attrition.csv"},
        ),
    ),
)
def test_sealed_selector_cannot_be_overridden_by_coder_summary(
    monkeypatch,
    planner_method,
    planner_outputs,
    reported_method,
    reported_family,
    verified_tables,
):
    import easyicu.research_agent.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "_verified_direct_parent_table_names",
        lambda run_dir, step_id: verified_tables,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_manifest_analysis_request",
        lambda run_dir, step_id: {
            "step": {
                "method": planner_method,
                "expected_outputs": planner_outputs,
            },
            "analysis_family": "association_study",
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_analysis_method",
        lambda run_dir, step_id: reported_method,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_analysis_family",
        lambda run_dir, step_id: reported_family,
    )

    assert (
        pipeline_module.deterministic_figure_repair_id_for_upstream(
            Path("/unused"), "07_spoofed_figure"
        )
        is None
    )


def test_sealed_selector_requires_host_recorded_parent_request(monkeypatch):
    import easyicu.research_agent.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "_verified_direct_parent_table_names",
        lambda run_dir, step_id: {"robustness_summary.csv"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_manifest_analysis_request",
        lambda run_dir, step_id: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_upstream_analysis_method",
        lambda run_dir, step_id: "cohort_definition_sensitivity",
    )

    assert (
        pipeline_module.deterministic_figure_repair_id_for_upstream(
            Path("/unused"), "07_legacy_summary_only_figure"
        )
        is None
    )
