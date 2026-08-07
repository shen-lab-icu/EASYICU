"""Causal/survival headline contracts are typed and reconciled to source data."""

from __future__ import annotations

import csv

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
    ResearchContext,
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
            event_definition="death",
            censoring_strategy="administrative at 28 days",
            competing_risk_strategy="none",
            time_horizon="28 days",
            effect_measure="hazard ratio",
        )
