"""Causal/survival headline contracts are typed and reconciled to source data."""

from __future__ import annotations

import csv
import json

import pytest

from easyicu.research_agent.gates.family_primary_result import (
    family_primary_result_reconciliation_findings,
)
from easyicu.research_agent.planning.primary_result_contract import (
    validate_required_primary_result,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    FamilyPrimaryResultRequirement,
    EndpointSpec,
    ResearchContext,
    SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
    VariableRole,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Does treatment change 28-day mortality?",
        cohort=CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=8, n_stays=8
        ),
        variables=[
            ConceptDescriptor(
                name="treatment", role=VariableRole.INTERVENTION, dtype="int64"
            ),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
        ],
        primary_exposure="treatment",
        target_outcome="death",
    )


def _causal_requirement() -> FamilyPrimaryResultRequirement:
    return FamilyPrimaryResultRequirement(
        analysis_family="causal_inference",
        exposure_source="treatment",
        outcome="death",
        expected_result_product="table:primary_causal_contrast",
        estimator="iptw",
        effect_scale="risk_difference",
        uncertainty_method="bootstrap_95ci",
        population="eligible target-trial cohort",
        estimand="ATE",
        treatment="initiate treatment",
        comparator="no treatment",
        adjustment_strategy="baseline IPTW",
        overlap_diagnostic="weight distribution and positivity",
    )


def _plan(requirement: FamilyPrimaryResultRequirement | None) -> AnalysisPlan:
    outputs = ["table:primary_causal_contrast"]
    return AnalysisPlan(
        research_question="Does treatment change 28-day mortality?",
        analysis_type="causal_inference",
        steps=[
            AnalysisStep(
                step_id="04_primary",
                intent="Estimate the causal contrast.",
                inputs=["artifact:analysis_cohort"],
                expected_outputs=outputs,
                method="causal_effect_estimation_iptw",
                planned_analysis_role="primary",
                family_primary_result_requirement=requirement,
            )
        ],
    )


def test_causal_primary_plan_requires_its_family_specific_contract() -> None:
    with pytest.raises(ValueError, match="family_primary_result_requirement"):
        validate_required_primary_result(plan=_plan(None), context=_context())


def test_causal_primary_result_contract_is_reconciled_to_registered_csv(
    tmp_path,
) -> None:
    requirement = _causal_requirement()
    plan = _plan(requirement)
    validate_required_primary_result(plan=plan, context=_context())
    result_path = tmp_path / "primary_causal_contrast.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "exposure_source",
                "outcome",
                "effect_scale",
                "effect_estimate",
                "ci_low",
                "ci_high",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "exposure_source": "treatment",
                "outcome": "death",
                "effect_scale": "risk_difference",
                "effect_estimate": "-0.08",
                "ci_low": "-0.13",
                "ci_high": "-0.03",
            }
        )

    findings = family_primary_result_reconciliation_findings(
        step=plan.steps[0],
        plan=plan,
        context=_context(),
        step_summary={
            "output_files": {
                "table:primary_causal_contrast": result_path.name,
            }
        },
        out_dir=tmp_path,
    )

    assert findings == []


def test_family_primary_result_rejects_an_unmatched_or_uncertain_effect(tmp_path) -> None:
    requirement = _causal_requirement()
    plan = _plan(requirement)
    result_path = tmp_path / "primary_causal_contrast.csv"
    result_path.write_text(
        "exposure_source,outcome,effect_scale,effect_estimate\n"
        "treatment,death,odds_ratio,1.2\n",
        encoding="utf-8",
    )

    findings = family_primary_result_reconciliation_findings(
        step=plan.steps[0],
        plan=plan,
        context=_context(),
        step_summary={
            "output_files": {"table:primary_causal_contrast": result_path.name}
        },
        out_dir=tmp_path,
    )

    assert [finding.detail["issue"] for finding in findings] == [
        "result_table_has_no_matching_primary_effect"
    ]


def test_cox_contract_requires_the_ph_diagnostic() -> None:
    with pytest.raises(ValueError, match="proportional_hazards_diagnostic"):
        FamilyPrimaryResultRequirement(
            analysis_family="survival",
            exposure_source="treatment",
            outcome="death",
            expected_result_product="table:survival_effect_estimates",
            estimator="cox_proportional_hazards",
            effect_scale="hazard_ratio",
            uncertainty_method="wald_95ci",
            population="eligible cohort",
            time_origin="ICU admission",
            time_column="follow_up_days",
            event_column="death",
            event_definition="death",
            censoring_strategy="administrative at 28 days",
            competing_risk_strategy="none",
            time_horizon="28 days",
            effect_measure="hazard ratio",
        )


def _survival_context() -> ResearchContext:
    return ResearchContext(
        research_question="Does treatment change 28-day mortality?",
        cohort=CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=8, n_stays=8
        ),
        variables=[
            ConceptDescriptor(
                name="treatment", role=VariableRole.INTERVENTION, dtype="int64"
            ),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
            ConceptDescriptor(
                name="follow_up_days", role=VariableRole.TIME, dtype="float64"
            ),
        ],
        primary_exposure="treatment",
        target_outcome="death",
        endpoint=EndpointSpec(
            name="death",
            kind="time_to_event",
            absence_semantics="no_absent_rows",
            levels=[0, 1],
            event_column="death",
            time_column="follow_up_days",
            time_origin="ICU admission",
            censoring_rule="administrative at 28 days",
        ),
    )


def _survival_requirement() -> FamilyPrimaryResultRequirement:
    return FamilyPrimaryResultRequirement(
        analysis_family="survival",
        exposure_source="treatment",
        outcome="death",
        expected_result_product="table:survival_effect_estimates",
        estimator="cox_proportional_hazards",
        effect_scale="hazard_ratio",
        uncertainty_method="wald_95ci",
        population="eligible cohort",
        time_origin="ICU admission",
        time_column="follow_up_days",
        event_column="death",
        event_definition="death",
        censoring_strategy="administrative at 28 days",
        competing_risk_strategy="none",
        time_horizon="28 days",
        effect_measure="hazard ratio",
        proportional_hazards_diagnostic="global Schoenfeld residual test",
    )


def _survival_plan(requirement: FamilyPrimaryResultRequirement) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Does treatment change 28-day mortality?",
        analysis_type="survival",
        steps=[
            AnalysisStep(
                step_id="04_primary",
                intent="Estimate the survival contrast.",
                inputs=["treatment", "death", "follow_up_days"],
                expected_outputs=[
                    "table:survival_effect_estimates",
                    SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
                ],
                method="cox_proportional_hazards",
                planned_analysis_role="primary",
                family_primary_result_requirement=requirement,
            )
        ],
    )


def _write_survival_receipt(tmp_path, requirement: FamilyPrimaryResultRequirement) -> str:
    receipt_path = tmp_path / "survival_analysis_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "result_product": requirement.expected_result_product,
                "exposure_source": requirement.exposure_source,
                "outcome": requirement.outcome,
                "effect_scale": requirement.effect_scale,
                "analysis_population": requirement.population,
                "n_analysis_rows": 8,
                "n_events": 3,
                "time_origin": requirement.time_origin,
                "time_column": requirement.time_column,
                "event_column": requirement.event_column,
                "event_definition": requirement.event_definition,
                "censoring_strategy": requirement.censoring_strategy,
                "competing_risk_strategy": requirement.competing_risk_strategy,
                "time_horizon": requirement.time_horizon,
                "estimator": requirement.estimator,
                "effect_measure": requirement.effect_measure,
                "proportional_hazards_diagnostic": requirement.proportional_hazards_diagnostic,
                "proportional_hazards_tested": True,
                "proportional_hazards_p_value": 0.42,
            }
        ),
        encoding="utf-8",
    )
    return receipt_path.name


def test_survival_headline_requires_an_execution_receipt(tmp_path) -> None:
    requirement = _survival_requirement()
    plan = _survival_plan(requirement)
    context = _survival_context()
    validate_required_primary_result(plan=plan, context=context)
    result_path = tmp_path / "survival_effect_estimates.csv"
    result_path.write_text(
        "exposure_source,outcome,effect_scale,hazard_ratio,ci_low,ci_high\n"
        "treatment,death,hazard_ratio,0.81,0.66,0.98\n",
        encoding="utf-8",
    )

    missing_receipt = family_primary_result_reconciliation_findings(
        step=plan.steps[0],
        plan=plan,
        context=context,
        step_summary={
            "output_files": {requirement.expected_result_product: result_path.name}
        },
        out_dir=tmp_path,
    )
    assert [finding.detail["issue"] for finding in missing_receipt] == [
        "survival_execution_receipt_unregistered"
    ]

    receipt_name = _write_survival_receipt(tmp_path, requirement)
    findings = family_primary_result_reconciliation_findings(
        step=plan.steps[0],
        plan=plan,
        context=context,
        step_summary={
            "output_files": {
                requirement.expected_result_product: result_path.name,
                SURVIVAL_ANALYSIS_RECEIPT_PRODUCT: receipt_name,
            }
        },
        out_dir=tmp_path,
    )
    assert findings == []


def test_survival_receipt_cannot_relabel_the_endpoint_or_skip_ph(tmp_path) -> None:
    requirement = _survival_requirement()
    plan = _survival_plan(requirement)
    context = _survival_context()
    result_path = tmp_path / "survival_effect_estimates.csv"
    result_path.write_text(
        "exposure_source,outcome,effect_scale,hazard_ratio,ci_low,ci_high\n"
        "treatment,death,hazard_ratio,0.81,0.66,0.98\n",
        encoding="utf-8",
    )
    receipt_name = _write_survival_receipt(tmp_path, requirement)
    receipt_path = tmp_path / receipt_name
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["time_column"] = "guessed_time"
    receipt.pop("proportional_hazards_p_value")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    findings = family_primary_result_reconciliation_findings(
        step=plan.steps[0],
        plan=plan,
        context=context,
        step_summary={
            "output_files": {
                requirement.expected_result_product: result_path.name,
                SURVIVAL_ANALYSIS_RECEIPT_PRODUCT: receipt_name,
            }
        },
        out_dir=tmp_path,
    )
    assert [finding.detail["issue"] for finding in findings] == [
        "survival_execution_receipt_invalid"
    ]
