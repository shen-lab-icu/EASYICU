"""An outcome-by-cluster roster refusal tells the Planner what to fix.

The Planner repairs a refused step from the refusal message alone. A bare
reason code gave it nothing to act on, so the message names the rule and the
offending sealed variables, and the step template states the rule up front.
"""

import json

import pytest

from easyicu.research_agent.agents.progressive_prompt_contracts import (
    step_materialization_shape_contract,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveOutlineStep,
    ProgressiveSkeletonStep,
)

_RAW = ["event_flag", "stay_length", "age_years"]


def _comparison_step(names):
    return ProgressiveSkeletonStep(
        step_id="outcome_comparison", planned_analysis_role="secondary",
        module_id="custom_analysis", objective="Compare outcomes across the frozen clusters.",
        raw_inputs=_RAW, outputs=[{"product_id": "table:outcome_by_cluster", "semantic_role": "custom"}],
        scientific_action_id="phenotyping.outcome_by_cluster", custom_method="frozen_cluster_outcome_description",
        phenotyping_comparison_variables=[{"name": name, "summary": "count_percent"} for name in names],
    )


def test_a_roster_of_listed_inputs_is_accepted():
    step = _comparison_step(["event_flag", "age_years"])

    assert [item.name for item in step.phenotyping_comparison_variables] == ["event_flag", "age_years"]


def test_a_name_missing_from_raw_inputs_is_named():
    with pytest.raises(ValueError) as caught:
        _comparison_step(["event_flag", "lactate_peak", "sofa_peak"])

    message = str(caught.value)
    assert "phenotype_comparison_roster_invalid" in message
    assert "must appear once and also be listed in raw_inputs" in message
    assert "not in raw_inputs: lactate_peak, sofa_peak" in message
    assert "repeated" not in message


def test_a_repeated_name_is_named():
    with pytest.raises(ValueError) as caught:
        _comparison_step(["event_flag", "age_years", "event_flag"])

    message = str(caught.value)
    assert "repeated: event_flag" in message
    assert "not in raw_inputs" not in message


def test_the_outcome_step_template_states_the_rule():
    outline = ProgressiveOutlineStep(
        step_id="outcome_comparison", module_id="custom_analysis", planned_analysis_role="secondary",
        objective="Compare outcomes across the frozen clusters.", variable_names=_RAW,
        scientific_action_id="phenotyping.outcome_by_cluster",
    )

    text = step_materialization_shape_contract(
        outline_step=outline, outline_step_sha256=canonical_sha256(outline.model_dump(mode="json")),
    )

    head, _, guidance = text.partition("\nCopy schema_version")
    assert json.loads(head.split("\n", 1)[1])["step"]["phenotyping_comparison_variables"] is None
    assert "Each name appears once and is also listed in raw_inputs." in guidance
