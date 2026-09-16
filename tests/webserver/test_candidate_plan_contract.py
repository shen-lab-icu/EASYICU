"""A reviewed zero-row candidate must reach package binding without lost science."""

import json
from types import SimpleNamespace

import pytest

from easyicu.webserver.agent_pipeline_runs import (
    ResearchPipelineRunError,
    _candidate_plan_contract,
)


_REVIEW = SimpleNamespace(plan_sha256="a" * 64, context_sha256="b" * 64)


def _seed(contract):
    return json.loads(contract.split("- candidate_plan_seed_json: ", 1)[1])


def test_package_bound_seed_retains_reviewed_specs_and_full_step_roster():
    steps = [
        {
            "step_id": f"step_{index}",
            "planned_analysis_role": "sensitivity",
            "method": "robustness_sensitivity",
            "inputs": ["strict_stage", "reference_stage"],
            "expected_outputs": [f"table:result_{index}"],
        }
        for index in range(34)
    ]
    steps[-1].update({
        "measurement_audit_spec": {"columns": ["strict_stage", "reference_stage"]},
        "robustness_replay_spec": {"products": [{"product_id": "definition", "output": "definition_table"}]},
        "sensitivity_spec_ids": ["definition_sensitivity"],
        "functional_form_spec": {"target_column": "charlson"},
        "literature_design_bindings": [{"citation_key": "prior_study", "design_elements": ["follow_up"]}],
        "patient_rows": [{"stay_id": 1}],
    })
    plan = {
        "research_question": "Compare an exposure with hospital death.",
        "analysis_type": "association_study",
        "endpoint": {"outcome": "death"},
        "robustness_specs": [{"spec_id": "definition_sensitivity"}],
        "design_selection": {"candidates": [{
            "design_id": "landmark", "analysis_type": "association_study",
            "observation_window": "24h to hospital discharge",
            "figure_role": "Display the primary adjusted association.",
            "reviewable_plan": ["population", "exposure", "outcome", "model", "missing", "sensitivity"],
            "decision_reason": "This design matches the prespecified estimand.",
        }]},
        "steps": steps,
    }
    seed = _seed(_candidate_plan_contract(review=_REVIEW, plan=plan))
    assert len(seed["steps"]) == 34
    assert seed["design_selection"][0]["observation_window"] == "24h to hospital discharge"
    assert seed["design_selection"][0]["analysis_type"] == "association_study"
    assert seed["design_selection"][0]["reviewable_plan"][-1] == "sensitivity"
    assert seed["design_selection"][0]["decision_reason"].startswith("This design")
    assert seed["steps"][-1]["functional_form_spec"]["target_column"] == "charlson"
    assert seed["steps"][-1]["sensitivity_spec_ids"] == ["definition_sensitivity"]
    assert seed["steps"][-1]["literature_design_bindings"][0]["citation_key"] == "prior_study"
    assert "patient_rows" not in seed["steps"][-1]


def test_package_bound_seed_rejects_oversize_instead_of_truncating():
    plan = {"steps": [{"step_id": "large", "intent": "x" * 64_000}]}
    with pytest.raises(ResearchPipelineRunError) as caught:
        _candidate_plan_contract(review=_REVIEW, plan=plan)
    assert caught.value.code == "candidate_plan_seed_too_large"
