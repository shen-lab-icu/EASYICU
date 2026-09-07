"""Counts-only study authority excludes inferential products and uncertainty."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.planning.dependence_authority import (
    DependenceAuthorityError,
    bind_context_dependence_authority,
)
from easyicu.research_agent.planning.scientific_review import repeated_unit_design_closed
from easyicu.research_agent.reporting.article_contract import build_article_analysis_contract
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, UserPreferences

from .scientific_review_fixtures import (
    _absolute_risk_distribution_step,
    _context,
    _traditional_table_one_step,
)


def test_counts_only_authority_removes_all_uncertainty_before_review() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    distribution_step = _absolute_risk_distribution_step()
    distribution_spec = distribution_step.exposure_outcome_distribution_spec
    assert distribution_spec is not None
    distribution_step = distribution_step.model_copy(
        update={
            "scientific_capability": ("descriptive_exposure_outcome_distribution_v1"),
            "exposure_outcome_distribution_spec": distribution_spec.model_copy(
                update={"risk_difference_contrast": None}
            ),
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[distribution_step],
    )

    bound = bind_context_dependence_authority(plan=plan, context=context)

    distribution = bound.steps[0].exposure_outcome_distribution_spec
    assert distribution is not None
    assert distribution.schema_version == "easyicu.exposure_outcome_distribution/3"
    assert distribution.interval_method == "none_counts_only"
    assert distribution.repeated_unit_interval_method is None
    assert distribution.confidence_level is None
    assert distribution.dependence is None
    assert repeated_unit_design_closed(context, bound) is True


def test_counts_only_article_contract_does_not_require_forbidden_table_one() -> None:
    ordinary = _context()
    context = ordinary.model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )

    contract = build_article_analysis_contract(
        context,
        analysis_type="descriptive_epidemiology",
    )
    ordinary_contract = build_article_analysis_contract(
        ordinary,
        analysis_type="descriptive_epidemiology",
    )

    assert "baseline_context" not in contract.required_roles
    assert all(item.role != "baseline_context" for item in contract.requirements)
    assert all(
        item.module_id != "distribution_prevalence" for item in contract.requirements
    )
    expected_roles = {
        item.role
        for item in ordinary_contract.requirements
        if item.required
        and item.role != "baseline_context"
        and item.module_id != "distribution_prevalence"
    }
    assert set(contract.required_roles) == expected_roles


def test_counts_only_typed_primary_subsumes_generic_distribution_module() -> None:
    context = _context().model_copy(
        update={
            "research_question": (
                "Estimate exposure prevalence and observed outcome event rates "
                "by exposure among ICU stays."
            ),
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            ),
        }
    )

    contract = build_article_analysis_contract(
        context,
        analysis_type="descriptive_epidemiology",
    )

    assert "descriptive_result" in contract.required_roles
    assert "distribution" not in contract.required_roles
    assert "baseline_context" not in contract.required_roles


def test_counts_only_authority_rejects_inferential_table_one_tests() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[_traditional_table_one_step()],
    )

    with pytest.raises(
        DependenceAuthorityError,
        match="forbids inferential Table One",
    ):
        bind_context_dependence_authority(plan=plan, context=context)


def test_counts_only_authority_accepts_descriptive_smd_table_one_and_report() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    table_step = _traditional_table_one_step()
    table_spec = table_step.table_one_spec
    assert table_spec is not None
    table_payload = table_spec.model_dump(mode="python")
    table_payload.update(
        schema_version="easyicu.table_one/2",
        p_values_required=False,
        p_value_adjustment="not_applicable_repeated_units",
    )
    for variable in table_payload["variables"]:
        variable["test"] = "none_descriptive_smd_only"
    table_step = table_step.model_copy(
        update={
            "table_one_spec": type(table_spec).model_validate(table_payload),
        }
    )
    report_step = AnalysisStep(
        step_id="report",
        planned_analysis_role="auxiliary",
        intent="Render the counts-only report.",
        inputs=["table:table_one"],
        expected_outputs=["report:strobe_style_report"],
        method="feasibility_protocol",
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[table_step, report_step],
    )

    bound = bind_context_dependence_authority(plan=plan, context=context)

    assert bound.steps[0].table_one_spec == table_step.table_one_spec
    assert bound.steps[1] == report_step


def test_counts_only_authority_rejects_untyped_descriptive_summaries() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[
            AnalysisStep(
                step_id="age_distribution",
                planned_analysis_role="auxiliary",
                intent="Summarize age by exposure.",
                method="descriptive_distribution",
                inputs=["artifact:analysis_cohort", "exposure", "age"],
                expected_outputs=["table:distribution_prevalence"],
            )
        ],
    )

    with pytest.raises(DependenceAuthorityError, match="permits only"):
        bind_context_dependence_authority(plan=plan, context=context)


def test_counts_only_audit_cannot_launder_a_prevalence_product() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[
            AnalysisStep(
                step_id="laundered_prevalence",
                planned_analysis_role="secondary",
                intent="Mislabel a measurement-process audit as prevalence.",
                method="measurement_audit",
                inputs=["artifact:analysis_cohort", "exposure"],
                expected_outputs=["table:distribution_prevalence"],
                measurement_audit_spec={
                    "products": [
                        {
                            "product_id": "distribution_prevalence",
                            "audit": "measurement_process",
                        }
                    ]
                },
            )
        ],
    )

    with pytest.raises(DependenceAuthorityError, match="audit product names"):
        bind_context_dependence_authority(plan=plan, context=context)


def test_counts_only_spec_cannot_launder_a_p_value_output() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    distribution = _absolute_risk_distribution_step().exposure_outcome_distribution_spec
    assert distribution is not None
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[
            AnalysisStep(
                step_id="laundered_test",
                planned_analysis_role="auxiliary",
                intent="Run a prohibited hypothesis test.",
                method="chi_square_test",
                inputs=["artifact:analysis_cohort", "exposure", "death"],
                expected_outputs=[
                    "table:exposure_outcome_distribution",
                    "statistic:p_value",
                ],
                exposure_outcome_distribution_spec=distribution.model_copy(
                    update={"risk_difference_contrast": None}
                ),
            )
        ],
    )

    with pytest.raises(DependenceAuthorityError, match="permits only"):
        bind_context_dependence_authority(plan=plan, context=context)


def test_counts_only_authority_rejects_a_risk_difference() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[_absolute_risk_distribution_step()],
    )

    with pytest.raises(DependenceAuthorityError, match="forbids risk-difference"):
        bind_context_dependence_authority(plan=plan, context=context)
