"""The append refit inherits the exact signed primary model without mutation."""

from types import SimpleNamespace

import pytest

from easyicu.research_agent.contracts.association_execution import (
    baseline_sensitivity_parent_step,
)
from easyicu.research_agent.planning.family_spec.baseline_association_template import (
    SensitivityAppend,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


@pytest.fixture
def source():
    primary = AnalysisStep(
        step_id="signed_primary", intent="Estimate the synthetic marker association",
        method="adjusted_association_models", scientific_capability="association_baseline_adjusted_v1",
        planned_analysis_role="primary", inputs=["marker", "response", "age"],
        expected_outputs=["table:adjusted_association_estimates"], icu_rule_refs=["signed:model"],
        model_requirements=[{
            "requirement_id": "signed_model", "outcome": "response", "outcome_type": "binary",
            "method_family": "statsmodels_logit_mle", "exposure_source": "marker",
            "analysis_role": "primary", "analysis_set": "source_aware",
            "covariates": ["age"], "model_terms": [
                {"name": "marker", "role": "exposure", "coding": "binary",
                 "transform": "treatment_contrast", "levels": ["0", "1"], "reference_level": "0"},
                {"name": "age", "role": "covariate", "coding": "continuous", "transform": "identity"},
            ],
        }],
    )
    return AnalysisPlan(research_question="Synthetic baseline association", steps=[
        AnalysisStep(step_id="description", intent="Describe the synthetic cohort"), primary,
    ])


def _authority(source, **changes):
    def governed(plan):
        assert plan is source
        return plan.steps[1]

    return SimpleNamespace(
        **{"governed_primary_step": governed, "exposure_encoding": "treatment_contrast",
           "subgroup_terms": (), **changes},
    )


def test_parent_keeps_the_signed_model_bytes_and_does_not_alias_source(source):
    before = source.model_dump_json()
    expected = source.steps[1].model_dump(mode="json")
    expected["scientific_capability"] = "association_adjusted_v1"

    parent = baseline_sensitivity_parent_step(source, _authority(source))

    assert parent.model_dump(mode="json") == expected
    assert source.model_dump_json() == before
    parent.model_requirements[0].model_terms[0].levels.append("2")
    parent.inputs.append("unapproved_variable")
    assert source.model_dump_json() == before


def test_planner_and_executor_share_one_parent_projection():
    assert SensitivityAppend.parent_step is baseline_sensitivity_parent_step


@pytest.mark.parametrize("changes", [
    {"exposure_encoding": "restricted_cubic_spline"},
    {"subgroup_terms": ("stratum",)},
    {"exposure_encoding": "restricted_cubic_spline", "subgroup_terms": ("stratum",)},
])
def test_parent_refuses_a_different_exposure_model_or_existing_interactions(source, changes):
    before = source.model_dump_json()
    with pytest.raises(ValueError, match="categorical model without existing interactions"):
        baseline_sensitivity_parent_step(source, _authority(source, **changes))
    assert source.model_dump_json() == before


def test_parent_preserves_the_signed_authority_refusal(source):
    refusal = ValueError("Synthetic signed model drift")

    def refuse(plan):
        assert plan is source
        raise refusal

    with pytest.raises(ValueError) as stopped:
        baseline_sensitivity_parent_step(source, _authority(source, governed_primary_step=refuse))
    assert stopped.value is refusal
