"""A typed choice survives transport, signing and host code generation."""

import json

import pytest

from easyicu.research_agent.agents.plan_payload import decode_planner_transport_payload
from easyicu.research_agent.contracts.model_terms import (
    DataPreparationPlan,
    preparation_for_step,
    preparation_method,
)
from easyicu.research_agent.schema import AnalysisStep


def test_progressive_contract_exposes_typed_preparation_and_transport_compiles_signed_choice():
    from easyicu.research_agent.planning.progressive_contract import (
        ProgressiveSkeletonStep,
    )

    schema = ProgressiveSkeletonStep.model_json_schema()
    assert "data_preparation" in schema["properties"]
    choice = {
        "missing_data": {
            "strategy": "multiple_imputation",
            "m": 4,
            "variables": [{"name": "x", "method": "pmm"}],
        }
    }
    payload = {
        "steps": [
            {
                "step_id": "fit",
                "intent": "Fit the declared model",
                "method": "custom_analysis",
                "data_preparation": choice,
            }
        ]
    }
    decoded = decode_planner_transport_payload(payload)
    step = AnalysisStep.model_validate(decoded["steps"][0])
    restored = AnalysisStep.model_validate_json(step.model_dump_json())
    plan = preparation_for_step(restored)
    assert plan.missing_data.m == 4
    assert plan.missing_data.sensitivity_strategy == "complete_case"
    assert plan.missing_data.variables[0].method == "pmm"
    assert "data_preparation" in payload["steps"][0]


def test_materialization_schema_offers_preparation_with_configured_imputation_controls():
    from easyicu.research_agent.agents.progressive_payload import (
        progressive_step_materialization_request,
    )
    from easyicu.research_agent.contracts.model_terms import DEFAULT_IMPUTATION_CONFIG
    from easyicu.research_agent.planning.progressive_contract import (
        ProgressiveOutlineStep,
    )

    step = ProgressiveOutlineStep(
        step_id="fit",
        planned_analysis_role="primary",
        module_id="adjusted_association",
        objective="Fit the prespecified synthetic model",
        variable_names=["exposure", "age", "death"],
        scientific_action_id="association.adjusted_association",
    )
    schema = json.loads(
        progressive_step_materialization_request(
            outline_step=step,
            outline_step_sha256="a" * 64,
            variable_names=step.variable_names,
            scientific_action_ids=[step.scientific_action_id],
        ).schema_json
    )
    assert "DataPreparationPlan" in schema["$defs"]
    for field in ("m", "seed", "iterations", "donors"):
        assert schema["$defs"]["MissingDataPlan"]["properties"][field][
            "const"
        ] == getattr(DEFAULT_IMPUTATION_CONFIG, field)


def test_method_declaration_roundtrips_and_default_imputation_policy_is_runtime_config():
    from easyicu.research_agent.contracts.model_terms import DEFAULT_IMPUTATION_CONFIG

    plan = DataPreparationPlan(missing_data={"strategy": "complete_case"})
    step = AnalysisStep(
        step_id="fit",
        intent="Fit a model",
        method=preparation_method("custom_analysis", plan),
    )
    assert preparation_for_step(step) == plan
    assert plan.missing_data.m == DEFAULT_IMPUTATION_CONFIG.m
    assert plan.missing_data.seed == DEFAULT_IMPUTATION_CONFIG.seed
    assert plan.missing_data.sensitivity_strategy == "multiple_imputation"


def test_a_causal_contract_cannot_claim_unimplemented_multiple_imputation():
    from pydantic import ValidationError

    from easyicu.research_agent.contracts.family_primary import (
        FamilyPrimaryResultRequirement,
    )
    from easyicu.research_agent.contracts.model_terms import PreparationRefusal

    declared = {
        "analysis_family": "causal_inference",
        "exposure_source": "synthetic_treatment",
        "outcome": "synthetic_outcome",
        "expected_result_product": "table:synthetic_causal_effect",
        "estimator": "stabilized_iptw",
        "effect_scale": "risk_difference",
        "uncertainty_method": "robust_sandwich",
        "population": "synthetic eligible cohort",
        "estimand": "average treatment effect",
        "treatment": "treated",
        "comparator": "untreated",
        "adjustment_strategy": "baseline confounders",
        "overlap_diagnostic": "propensity overlap",
    }
    assert FamilyPrimaryResultRequirement(**declared).missing_data_policy == "complete_case"
    with pytest.raises(ValidationError) as caught:
        FamilyPrimaryResultRequirement(**declared, missing_data_policy="multiple_imputation")
    refusal = caught.value.errors()[0]["ctx"]["error"]
    assert isinstance(refusal, PreparationRefusal)
    assert refusal.reason_code == "mi_causal_model_unsupported"


def test_preparation_diagnostic_fields_survive_without_private_coordinates():
    from easyicu.research_agent.providers.structured_diagnostics import (
        safe_projected_validation_issues,
    )

    fields = {"data_preparation"}

    def collect(node):
        if isinstance(node, dict):
            fields.update(node.get("properties", {}))
            for child in node.values():
                collect(child)
        elif isinstance(node, list):
            for child in node:
                collect(child)

    collect(DataPreparationPlan.model_json_schema())
    for field in sorted(fields):
        projected = safe_projected_validation_issues([
            {"location": ["data_preparation", field, "private-coordinate"], "issue_type": "missing"},
        ])
        assert projected[0]["location"] == ["data_preparation", field, "<other>"]
        assert "private" not in json.dumps(projected)


def test_the_step_contract_refuses_preparation_outside_its_execution_coordinates():
    from easyicu.research_agent.planning.progressive_contract import ProgressiveSkeletonStep

    payload = {
        "step_id": "synthetic_fit", "module_id": "custom_analysis",
        "planned_analysis_role": "primary", "objective": "Fit a synthetic model",
        "custom_method": "synthetic_model",
        "raw_inputs": ["x", "y"],
        "outputs": [{"product_id": "table:synthetic_fit", "semantic_role": "custom"}],
        "data_preparation": {"missing_data": {"strategy": "complete_case"}},
    }
    for action in ("prediction.discrimination_calibration", "time_to_event.cox_hr"):
        assert ProgressiveSkeletonStep(**payload, scientific_action_id=action).data_preparation is not None
    with pytest.raises(ValueError, match="data_preparation"):
        ProgressiveSkeletonStep(**payload)


def test_review_refuses_preparation_without_an_execution_owner():
    from easyicu.research_agent.planning.primary_result_contract import (
        preparation_step_issue,
    )
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    choice = DataPreparationPlan(missing_data={"strategy": "multiple_imputation"})
    step = AnalysisStep(
        step_id="fit",
        intent="Fit a custom unsupported model",
        method=preparation_method("custom_analysis", choice),
    )
    context = ResearchContext(
        research_question="Synthetic study",
        cohort=CohortDescriptor(
            cohort_name="synthetic", database="synthetic", n_patients=100, n_stays=100
        ),
        variables=[],
    )
    assert preparation_step_issue(step, context) == "preparation_owner_unsupported"
    from easyicu.research_agent.planning.scientific_review import (
        build_plan_scientific_review,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    review = build_plan_scientific_review(
        context=context,
        plan=AnalysisPlan(research_question=context.research_question, steps=[step]),
    )
    finding = next(
        row for row in review.findings if row.code == "DATA_PREPARATION_UNSUPPORTED"
    )
    assert finding.severity == "blocker"
    assert "preparation_owner_unsupported" in finding.message


def test_review_cannot_borrow_raw_authority_from_another_database():
    from easyicu.concept.provenance import ExportDataProvenance, capture_data_provenance
    from easyicu.research_agent.planning.primary_result_contract import (
        preparation_step_issue,
    )
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
    )
    from tests.support.data_provenance import synthetic_dictionary

    provenance = ExportDataProvenance(
        snapshots=(
            capture_data_provenance(
                synthetic_dictionary(), concepts=["lab_signal"], source_database="miiv"
            ),
        )
    )
    context = ResearchContext(
        research_question="Synthetic association",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="eicu",
            n_patients=100,
            n_stays=100,
            provenance={"data_provenance": provenance.model_dump(mode="json")},
        ),
        variables=[
            ConceptDescriptor(
                name="lab_signal", dtype="float64", source_concept="lab_signal"
            )
        ],
    )
    choice = DataPreparationPlan(
        outliers=[
            {"variable": "lab_signal", "rule": "winsorize", "lower": 0, "upper": 5}
        ]
    )
    step = AnalysisStep(
        step_id="fit",
        intent="Fit a synthetic association",
        planned_analysis_role="primary",
        method=preparation_method("adjusted_association_models", choice),
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:adjusted_association_estimates"],
        model_requirements=[
            {
                "requirement_id": "fit",
                "outcome": "death",
                "outcome_type": "binary",
                "exposure_source": "exposure",
                "method_family": "statsmodels_logit_mle",
                "analysis_role": "primary",
                "analysis_set": "source_aware",
                "covariates": ["lab_signal"],
                "model_terms": [
                    {
                        "name": "exposure",
                        "role": "exposure",
                        "coding": "continuous",
                        "transform": "identity",
                    },
                    {
                        "name": "lab_signal",
                        "role": "covariate",
                        "coding": "continuous",
                        "transform": "identity",
                    },
                ],
            }
        ],
    )
    assert preparation_step_issue(step, context) == "preparation_raw_values_unavailable"
    context.cohort.database = "MIMIC-IV"
    assert preparation_step_issue(step, context) is None


def test_result_cannot_substitute_complete_case_for_a_signed_mi_primary():
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.contracts.model_terms import preparation_result_issues

    choice = DataPreparationPlan(
        missing_data={"strategy": "multiple_imputation", "m": 4}
    )
    step = AnalysisStep(
        step_id="fit",
        intent="Fit the declared model",
        method=preparation_method("custom_analysis", choice),
    )
    summary = {
        "model_contracts": [{"baseline_missing_policy": "drop_missing_baseline"}],
        "data_preparation": {
            "plan": choice.model_dump(mode="json"),
            "plan_sha256": canonical_sha256(choice.model_dump(mode="json")),
        },
    }
    assert "preparation_primary_policy_mismatch" in {
        row["issue"] for row in preparation_result_issues(step, summary)
    }


def test_complete_case_sensitivity_does_not_restate_a_multiple_imputation_primary():
    from easyicu.research_agent.schema import AnalysisPlan
    from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
    from easyicu.research_agent.planning.scientific_review import (
        _complete_case_specs_restating_primary,
    )

    choice = DataPreparationPlan(
        missing_data={
            "strategy": "multiple_imputation",
            "variables": [{"name": "age", "method": "pmm"}],
        }
    )
    primary = AnalysisStep(
        step_id="fit",
        intent="Fit a synthetic association",
        planned_analysis_role="primary",
        method=preparation_method("adjusted_association_models", choice),
        expected_outputs=["table:adjusted_association_estimates"],
        model_requirements=[
            {
                "requirement_id": "fit",
                "outcome": "death",
                "outcome_type": "binary",
                "exposure_source": "exposure",
                "method_family": "statsmodels_logit_mle",
                "analysis_role": "primary",
                "analysis_set": "source_aware",
                "covariates": ["age"],
            }
        ],
    )
    plan = AnalysisPlan(
        research_question="Synthetic association",
        steps=[primary],
        robustness_specs=[
            RobustnessSpec(
                spec_id="cc",
                axis="missing",
                description="Prespecified complete-case comparison",
                missing_override={
                    "strategy": "complete_case",
                    "variables": ["exposure", "death", "age"],
                },
            )
        ],
    )
    assert _complete_case_specs_restating_primary(plan) == ()


def _valid_signed_receipt():
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.contracts.model_terms import preparation_result_issues

    choice = DataPreparationPlan(
        missing_data={"strategy": "multiple_imputation", "m": 4}
    )
    step = AnalysisStep(
        step_id="fit",
        intent="Fit the signed model",
        method=preparation_method("custom_analysis", choice),
    )
    summary = {
        "model_contracts": [{"baseline_missing_policy": "multiple_imputation"}],
        "data_preparation": {
            "plan": choice.model_dump(mode="json"),
            "plan_sha256": canonical_sha256(choice.model_dump(mode="json")),
            "missing_data": {
                key: getattr(choice.missing_data, key)
                for key in ("m", "seed", "iterations", "donors")
            },
            "sensitivity": {"strategy": "complete_case", "status": "fitted"},
        },
    }
    assert preparation_result_issues(step, summary) == []
    return step, summary


@pytest.mark.parametrize("field", ["m", "seed", "iterations", "donors"])
def test_executed_imputation_controls_must_match_the_signed_plan(field):
    from easyicu.research_agent.contracts.model_terms import preparation_result_issues

    step, summary = _valid_signed_receipt()
    summary["data_preparation"]["missing_data"][field] += 1
    assert "preparation_imputation_policy_mismatch" in {
        row["issue"] for row in preparation_result_issues(step, summary)
    }


@pytest.mark.parametrize("field", ["plan", "plan_sha256"])
def test_executed_preparation_requires_both_the_exact_plan_and_its_digest(field):
    from easyicu.research_agent.contracts.model_terms import preparation_result_issues

    step, summary = _valid_signed_receipt()
    summary["data_preparation"][field] = {} if field == "plan" else "0" * 64
    assert preparation_result_issues(step, summary) == [
        {"issue": "preparation_signed_plan_mismatch"}
    ]


@pytest.mark.parametrize(
    "field,value", [("strategy", "multiple_imputation"), ("status", "omitted")]
)
def test_the_complementary_sensitivity_must_be_reported(field, value):
    from easyicu.research_agent.contracts.model_terms import preparation_result_issues

    step, summary = _valid_signed_receipt()
    summary["data_preparation"]["sensitivity"][field] = value
    assert "preparation_complementary_sensitivity_missing" in {
        row["issue"] for row in preparation_result_issues(step, summary)
    }
