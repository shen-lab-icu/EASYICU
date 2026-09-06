"""A missing method-layer repair must not revoke already valid source uses."""

from copy import deepcopy

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    _preserve_non_targeted_coordinates_across_literature_repair,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
    ProgressiveStepMaterialization,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _materialization_payloads,
    _context,
    _outline_payload,
    _payload,
)


def _attempt(elements, application, *, divergence=None):
    payload = deepcopy(_materialization_payloads(_payload())[2])
    payload["step"]["literature_bindings"] = [{
        "citation_key": "strobe_2007", "design_elements": elements,
        "application": application, "divergence": divergence,
    }]
    return ProgressiveStepMaterialization.model_validate(payload)


def _repair(previous, current, reason="progressive_step_required_method_layer_unbound"):
    return _preserve_non_targeted_coordinates_across_literature_repair(
        previous=previous, current=current,
        compiler_observation={"path": "literature_bindings", "reason_code": reason},
    )


@pytest.mark.parametrize("reason", [
    "progressive_step_required_method_layer_unbound",
    "progressive_final_method_layer_unbound",
])
def test_adding_dependence_preserves_reporting_and_both_model_authored_applications(reason):
    previous = _attempt(["reporting"], "Report denominators and missingness.")
    current = _attempt(["dependence"], "State the limitation from repeated ICU stays.")
    merged = _repair(previous, current, reason)
    binding = merged.step.literature_bindings[0]
    assert binding.design_elements == ["reporting", "dependence"]
    assert previous.step.literature_bindings[0].application in binding.application
    assert current.step.literature_bindings[0].application in binding.application
    assert merged.step.raw_inputs == previous.step.raw_inputs
    assert current.step.literature_bindings[0].design_elements == ["dependence"]


def test_source_scope_correction_can_remove_an_invalid_old_design_element():
    previous = _attempt(["exposure"], "Use this source for exposure definition.")
    current = _attempt(["reporting"], "Use this source only for transparent reporting.")
    merged = _repair(previous, current, "progressive_step_method_binding_scope_unsupported")
    assert merged.step.literature_bindings == current.step.literature_bindings


def test_coverage_repair_keeps_both_explicit_divergence_notes():
    previous = _attempt(["reporting"], "Report counts only.", divergence="No inferential interval is available.")
    current = _attempt(["dependence"], "Describe the repeated-stay limitation.", divergence="Patient-level independence is unverified.")
    binding = _repair(previous, current).step.literature_bindings[0]
    assert previous.step.literature_bindings[0].divergence in binding.divergence
    assert current.step.literature_bindings[0].divergence in binding.divergence


def test_different_outline_coordinate_does_not_inherit_previous_binding():
    previous = _attempt(["reporting"], "Report denominators and missingness.")
    current = _attempt(["dependence"], "Describe the repeated-stay limitation.").model_copy(
        update={"outline_step_sha256": "f" * 64},
    )
    assert _repair(previous, current) == current


def test_oversized_merged_application_fails_with_precise_current_step_feedback():
    previous = _attempt(["reporting"], "Reporting rationale: " + "a" * 900)
    current = _attempt(["dependence"], "Dependence rationale: " + "b" * 900)
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _repair(previous, current)
    assert caught.value.reason_code == "progressive_literature_repair_scope_conflict"
    assert caught.value.details["path"] == "literature_bindings"


def test_unchanged_binding_is_not_duplicated_on_another_repair():
    previous = _attempt(["reporting", "dependence"], "Report the descriptive repeated-stay population.")
    assert _repair(previous, previous) == previous


def test_expanded_use_keeps_existing_caveat_even_when_no_design_element_is_lost():
    previous = _attempt(["reporting"], "Report counts only.", divergence="No inferential interval is available.")
    current = _attempt(["reporting", "dependence"], "Report the repeated-stay population.")
    assert previous.step.literature_bindings[0].divergence in _repair(previous, current).step.literature_bindings[0].divergence


def test_repair_prompt_receives_the_previous_source_uses_not_only_the_last_error():
    previous = _attempt(["reporting"], "Report denominators and missingness.")
    outline = ProgressivePlanOutline.model_validate(_outline_payload(_payload()))
    prompt = ProgressivePlannerAgent._materialization_prompt(
        context=_context(), outline=outline, outline_step=outline.steps[2],
        outline_step_sha256=previous.outline_step_sha256,
        variables=[], action_rows=[], allowed_literature_citation_keys=["strobe_2007"],
        know_how_context="", planning_contract_context="", prefix_summary=[],
        available_product_refs=[],
        compiler_observation={"path": "literature_bindings", "reason_code": "progressive_step_required_method_layer_unbound"},
        prior_literature_bindings=[b.model_dump(mode="json") for b in previous.step.literature_bindings],
    )
    assert "Report denominators and missingness." in prompt
    assert '"design_elements":["reporting"]' in prompt
    assert "does not revoke other valid uses or caveats" in prompt
