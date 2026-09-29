"""A custom sensitivity without an action inherits the primary adjusted model.

Without an action, the compiler binds a custom sensitivity only to the
adjusted-association contract, and step materialization cannot change the
action an outline sealed. An outline outside that lineage must therefore name
the action that owns its sensitivity before any step is materialized.
"""

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.progressive_prompt_contracts import (
    outline_shape_contract,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanOutline,
)


def _step(step_id, role, module_id, depends_on=(), action=None):
    return {
        "step_id": step_id, "planned_analysis_role": role, "module_id": module_id,
        "objective": f"Run the prespecified {step_id}.", "depends_on": list(depends_on),
        "variable_names": ["stay_key", "marker_a"], "scientific_action_id": action,
    }


def _outline(analysis_type, steps):
    return ProgressivePlanOutline.model_validate({
        "analysis_type": analysis_type, "cohort_objective": "Describe the sealed cohort.",
        "steps": steps, "rationale": "A prespecified synthetic outline.",
    })


def _clustering_with_stability(action):
    return [
        _step("cohort", "auxiliary", "cohort_definition"),
        _step(
            "trajectory_fit", "primary", "custom_analysis", ["cohort"],
            "phenotyping.trajectory_feature_clustering",
        ),
        _step("fit_stability", "sensitivity", "custom_analysis", ["trajectory_fit"], action),
    ]


def _adjusted_model_with_sensitivity(parent, action=None):
    return [
        _step("cohort", "auxiliary", "cohort_definition"),
        _step("adjusted_model", "primary", "adjusted_association", ["cohort"]),
        _step("marker_form", "sensitivity", "custom_analysis", [parent], action),
    ]


def test_a_clustering_stability_without_its_action_is_refused_at_the_outline():
    with pytest.raises(ValidationError, match="'fit_stability' names no scientific_action_id"):
        _outline("trajectory_clustering", _clustering_with_stability(None))


def test_the_family_action_that_owns_the_stability_is_accepted():
    outline = _outline(
        "trajectory_clustering",
        _clustering_with_stability("phenotyping.trajectory_cluster_stability"),
    )

    assert outline.steps[-1].scientific_action_id == "phenotyping.trajectory_cluster_stability"


@pytest.mark.parametrize("analysis_type", ["association_study", "descriptive_epidemiology"])
def test_an_action_less_sensitivity_on_the_primary_adjusted_model_is_accepted(analysis_type):
    outline = _outline(analysis_type, _adjusted_model_with_sensitivity("adjusted_model"))

    assert outline.steps[-1].scientific_action_id is None


def test_an_action_less_sensitivity_off_the_primary_adjusted_model_is_refused():
    with pytest.raises(ValidationError, match="'marker_form' names no scientific_action_id"):
        _outline("descriptive_epidemiology", _adjusted_model_with_sensitivity("cohort"))


def test_an_association_sensitivity_still_depends_directly_on_the_primary_model():
    with pytest.raises(ValidationError, match="association custom sensitivity steps must follow"):
        _outline(
            "association_study",
            _adjusted_model_with_sensitivity("cohort", "association.rcs_spline"),
        )


def test_the_outline_prompt_states_the_rule_before_the_planner_chooses():
    text = outline_shape_contract(
        analysis_types=["trajectory_clustering"],
        module_ids_by_analysis_type={"trajectory_clustering": ["custom_analysis"]},
    )

    assert (
        "A custom_analysis sensitivity step with scientific_action_id=null is an "
        "adjusted-association sensitivity and must depend directly on the primary "
        "adjusted_association step"
    ) in text
