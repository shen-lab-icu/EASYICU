"""Fresh Planner measurement-audit authority boundary."""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.planner_measurement_audit import (
    PlannerMeasurementAuditContractError,
    missing_planner_measurement_audit_specs,
    validate_planner_measurement_audit_specs,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _plan(step: AnalysisStep) -> AnalysisPlan:
    return AnalysisPlan(research_question="Audit the cohort.", steps=[step])


def test_count_only_audit_requires_typed_product_meaning() -> None:
    plan = _plan(
        AnalysisStep(
            step_id="audit",
            planned_analysis_role="auxiliary",
            intent="Count measurement missingness.",
            method="measurement_audit",
            expected_outputs=["table:reader_selected_name"],
        )
    )

    assert missing_planner_measurement_audit_specs(plan) == ("audit",)
    with pytest.raises(
        PlannerMeasurementAuditContractError,
        match="Product labels and figure prose do not establish",
    ):
        validate_planner_measurement_audit_specs(plan)


def test_typed_audit_alias_is_accepted() -> None:
    plan = _plan(
        AnalysisStep(
            step_id="audit",
            planned_analysis_role="auxiliary",
            intent="Count measurement missingness.",
            method="measurement_audit",
            expected_outputs=["table:reader_selected_name"],
            measurement_audit_spec={
                "products": [
                    {
                        "product_id": "reader_selected_name",
                        "audit": "measurement_missingness",
                    }
                ]
            },
        )
    )

    validate_planner_measurement_audit_specs(plan)
    assert missing_planner_measurement_audit_specs(plan) == ()


def test_richer_measurement_bias_analysis_is_not_relabelled_as_count_audit() -> None:
    plan = _plan(
        AnalysisStep(
            step_id="bias_analysis",
            planned_analysis_role="secondary",
            intent="Estimate a measurement-bias contrast.",
            method="measurement_bias_audit_and_estimation",
            expected_outputs=["table:measurement_bias_estimates"],
        )
    )

    validate_planner_measurement_audit_specs(plan)
    assert missing_planner_measurement_audit_specs(plan) == ()
