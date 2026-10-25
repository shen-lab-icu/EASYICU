"""A planned secondary effect action owns its declared table, whatever its name."""

import pytest

from easyicu.research_agent.agents.coder import _declared_output_scope_contract
from easyicu.research_agent.agents._support import _CODER_GUIDE
from easyicu.research_agent.contracts.step_families import effect_output_authorized
from easyicu.research_agent.research_context.prompt_scope import (
    coder_guide_for_step,
    coder_rewrite_guide_for_step,
    compact_initial_coder_guide_for_step,
)
from easyicu.research_agent.schema import AnalysisStep


def secondary(**updates):
    return AnalysisStep(
        step_id="modifier", method="planned_modifier_regression",
        intent="Compare the marker association within ward strata.",
        scientific_action_id="association.effect_modification",
        planned_analysis_role="secondary", inputs=["marker", "event", "ward"],
        expected_outputs=["table:ward_comparison"],
    ).model_copy(update=updates)


@pytest.mark.parametrize("action", ["association.effect_modification", "time_to_event.subgroup_hr"])
@pytest.mark.parametrize("product", ["table:ward_comparison", "table:effect_modification_estimates"])
def test_secondary_action_authorizes_the_exact_declared_table(action, product):
    step = secondary(scientific_action_id=action, expected_outputs=[product])
    assert effect_output_authorized(step)
    assert "effect_output_authorized: true" in _declared_output_scope_contract(step)


@pytest.mark.parametrize("updates", [
    {"scientific_action_id": None},
    {"scientific_action_id": "association.descriptive_context"},
    {"planned_analysis_role": "diagnostic"},
    {"planned_analysis_role": "primary"},
    {"expected_outputs": ["log:ward_comparison"]},
    {"expected_outputs": ["figure:ward_comparison"]},
    {"expected_outputs": []},
])
def test_intent_or_names_alone_do_not_authorize_secondary_effects(updates):
    assert not effect_output_authorized(secondary(**updates))


@pytest.mark.parametrize("guide", [coder_guide_for_step, coder_rewrite_guide_for_step,
                                  compact_initial_coder_guide_for_step])
def test_generation_and_rewrite_guides_preserve_the_stratum_output_shape(guide):
    prompt = guide(_CODER_GUIDE, secondary())
    for column in ("subgroup_column", "subgroup_value", "estimate", "ci_low", "ci_high",
                   "effect_scale", "interaction_p_value", "fit_status", "contrast"):
        assert column in prompt
    assert "95%" in prompt
    assert "omnibus" in prompt
    assert "subgroup_column" not in guide(_CODER_GUIDE, secondary(scientific_action_id=None))
