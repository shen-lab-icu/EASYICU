"""Owner-contract tests for a compiled binary association sensitivity grid."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyicu.research_agent.contracts.association_execution import (
    ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID,
    association_binary_sensitivity_plan_verdict,
)
from easyicu.research_agent.contracts.declared_product import (
    declared_product_contract_findings,
)
from easyicu.research_agent.contracts.model_terms import ModelTermSpec
from easyicu.research_agent.plan_utils import effect_output_authorized
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    PlannedModelRequirement,
)


def _primary() -> AnalysisStep:
    return AnalysisStep(
        step_id="primary",
        planned_analysis_role="primary",
        intent="Fit the exact primary adjusted association.",
        inputs=["exposure", "outcome", "age", "artifact:analysis_cohort"],
        expected_outputs=["table:adjusted_association_estimates"],
        method="adjusted_association_models",
        scientific_capability="association_adjusted_v1",
        model_requirements=[
            PlannedModelRequirement(
                requirement_id="primary_binary",
                outcome="outcome",
                outcome_type="binary",
                method_family="statsmodels_logit_mle",
                exposure_source="exposure",
                analysis_role="primary",
                analysis_set="source_aware",
                required_for_step_success=True,
                covariates=["age"],
                model_terms=[
                    ModelTermSpec(
                        name="exposure",
                        role="exposure",
                        coding="binary",
                        levels=["0", "1"],
                        reference_level="0",
                        transform="treatment_contrast",
                    ),
                    ModelTermSpec(
                        name="age",
                        role="covariate",
                        coding="continuous",
                        transform="identity",
                    ),
                ],
                exposure_levels=["0", "1"],
                exposure_reference_level="0",
                primary_contrast_level="1",
            )
        ],
    )


def _sensitivity(**updates: object) -> AnalysisStep:
    payload: dict[str, object] = {
        "step_id": "sensitivity",
        "planned_analysis_role": "sensitivity",
        "intent": "Fit the two prespecified binary association variants.",
        "inputs": [
            "exposure",
            "outcome",
            "age",
            "artifact:analysis_cohort",
            "table:adjusted_association_estimates",
        ],
        "expected_outputs": ["table:scientific_sensitivity"],
        "method": "prespecified_binary_association_sensitivity",
        "scientific_capability": ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID,
        "sensitivity_spec_ids": ["full_cohort", "landmark"],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _summary() -> dict[str, object]:
    return {
        "output_files": {"table:scientific_sensitivity": "scientific_sensitivity.csv"},
        "analysis_rows": [
            {
                "analysis_id": "full_cohort",
                "n_stays": 100,
                "n_deaths": 20,
                "odds_ratio": 1.5,
                "ci_low": 1.2,
                "ci_high": 1.9,
            },
            {
                "analysis_id": "landmark",
                "n_stays": 90,
                "n_deaths": 12,
                "odds_ratio": 1.4,
                "ci_low": 1.1,
                "ci_high": 1.8,
            },
        ],
    }


def _kinds(findings: list[object]) -> set[str]:
    return {
        str(finding.detail["kind"])
        for finding in findings
        if getattr(finding, "detail", None)
    }


def test_closed_plan_grants_only_the_compiled_sensitivity_effect_authority() -> None:
    parent = _primary()
    child = _sensitivity()
    plan = AnalysisPlan(
        research_question="Does exposure associate with outcome?",
        analysis_type="association_study",
        steps=[parent, child],
    )

    verdict = association_binary_sensitivity_plan_verdict(
        plan.steps[1],
        plan_steps=plan.steps,
    )
    assert verdict.claimed is True
    assert verdict.contract is not None
    assert verdict.contract.sensitivity_ids == ("full_cohort", "landmark")
    assert effect_output_authorized(plan.steps[1]) is True


def test_sensitivity_can_inherit_the_signed_landmark_categorical_parent() -> None:
    parent = _primary().model_copy(
        update={"method": "signed_landmark_categorical_association"}
    )
    child = _sensitivity()

    plan = AnalysisPlan.model_validate(
        {
            "research_question": "Does ordinal exposure associate with outcome?",
            "analysis_type": "association_study",
            "steps": [
                parent.model_dump(mode="json"),
                child.model_dump(mode="json"),
            ],
        }
    )

    verdict = association_binary_sensitivity_plan_verdict(
        plan.steps[1],
        plan_steps=plan.steps,
    )
    assert verdict.claimed is True
    assert verdict.contract is not None


def test_capability_fails_closed_without_its_exact_parent_or_output_shape() -> None:
    with pytest.raises(ValidationError, match="parent_ambiguous"):
        AnalysisPlan(
            research_question="Does exposure associate with outcome?",
            analysis_type="association_study",
            steps=[_sensitivity()],
        )

    malformed = _sensitivity(
        expected_outputs=["table:scientific_sensitivity", "table:extra"]
    )
    assert effect_output_authorized(malformed) is False
    with pytest.raises(ValidationError, match="shape_invalid"):
        AnalysisPlan(
            research_question="Does exposure associate with outcome?",
            analysis_type="association_study",
            steps=[_primary(), malformed],
        )


def test_result_gate_closes_ids_counts_and_effect_intervals() -> None:
    step = _sensitivity()
    valid = declared_product_contract_findings(
        step=step,
        step_summary=_summary(),
        effect_method_authorized=effect_output_authorized(step),
    )
    assert valid == []

    invalid_summary = _summary()
    invalid_summary["analysis_rows"] = [
        {
            "analysis_id": "full_cohort",
            "n_stays": 10,
            "n_deaths": 11,
            "odds_ratio": 1.5,
            "ci_low": 1.6,
            "ci_high": 1.7,
        }
    ]
    findings = declared_product_contract_findings(
        step=step,
        step_summary=invalid_summary,
        effect_method_authorized=effect_output_authorized(step),
    )

    assert "association_binary_sensitivity_ids_mismatch" in _kinds(findings)
    assert "association_binary_sensitivity_row_invalid" in _kinds(findings)
    assert "unauthorized_effect_product" not in _kinds(findings)
