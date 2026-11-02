"""A model audit compares what ran and does not invent missing fields."""

from easyicu.research_agent.contracts.model_tokens import reported_model_requirement_fields


def test_reported_model_contract_keeps_fallback_and_eight_comparison_fields():
    projected = reported_model_requirement_fields({
        "outcome": "mortality", "model_family": "logit", "method_family": "",
        "analysis_role": "primary", "unrelated": "not a comparison field",
    })
    assert projected == {
        "outcome": "mortality", "outcome_type": None, "outcome_definition": None,
        "method_family": "logit", "exposure_source": None, "analysis_role": "primary",
        "analysis_set": None, "dependence": None,
    }
    assert reported_model_requirement_fields({
        "method_family": "cox", "model_family": "logit",
    })["method_family"] == "cox"
