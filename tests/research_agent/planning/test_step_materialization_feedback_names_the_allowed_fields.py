"""Step retries repair existing fields without inventing explanatory keys."""

import copy
import hashlib
import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.progressive_payload import (
    parse_progressive_step_materialization,
)
from easyicu.research_agent.agents.progressive_prompt_contracts import (
    step_materialization_shape_contract,
)
from easyicu.research_agent.planning.progressive_contract import ProgressiveOutlineStep
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_diagnostics import safe_validation_issues
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
    call_llm_with_structured_retry,
    render_parse_failure,
)


_DIGEST = "a" * 64
_OUTPUT = {"product_id": "figure:summary_plot", "semantic_role": "figure"}


def _outline(module="visualization"):
    return ProgressiveOutlineStep(
        step_id="display_summary", module_id=module,
        planned_analysis_role="auxiliary",
        objective="Display the prespecified upstream summaries.",
        depends_on=["preceding_model"], variable_names=["metric_a"],
    )


def _shape_contract(outline):
    return step_materialization_shape_contract(
        outline_step=outline, outline_step_sha256=_DIGEST,
    )


def _payload():
    # Key names and counts only mirror the diagnostic projection. All values
    # are synthetic; neither missing model prose nor redacted keys are inferred.
    return {
        "schema_version": "easyicu.progressive_step_materialization/1",
        "outline_step_sha256": _DIGEST,
        "foundation": None,
        "step": {
            "step_id": "display_summary", "planned_analysis_role": "auxiliary",
            "module_id": "visualization",
            "objective": "Display the prespecified upstream summaries.",
            "depends_on": ["preceding_model"], "raw_inputs": [],
            "product_inputs": [{"producer_step_id": "preceding_model", "product_id": "table:summary"}],
            "outputs": [], "scientific_action_id": None, "custom_method": None,
            "table_one_group_by": None, "table_one_mode": None, "table_one_variables": [],
            "primary_exposure": None, "outcome": None, "outcome_type": None,
            "model_terms": [], "event_level_index": None,
            "reference_exposure_level_index": None, "comparison_exposure_level_index": None,
            "primary_contrast_level_index": None, "denominator_policy": None,
            "missing_exposure_policy": None, "missing_outcome_policy": None,
            "confidence_level": None, "sensitivity_spec_ids": [], "literature_bindings": [],
        },
    }


def _corrected():
    payload = _payload()
    payload["step"]["outputs"] = [dict(_OUTPUT)]
    return payload


def _recorded_shapes():
    # Both forbidden keys and these closed enum values also reconstruct the
    # recorded violation hashes. No free-form model output is used.
    second, third = _corrected(), _corrected()
    second["step"]["outline_population_scope"] = "analysis_cohort"
    third["step"]["outline_population_scope"] = "primary_model"
    return [_payload(), second, third]


def _run(responses, outline=None):
    outline = outline or _outline()
    llm = ScriptedMockLLMClient([json.dumps(row) for row in responses])
    return llm, lambda: call_llm_with_structured_retry(
        llm, [],
        parser=lambda raw: parse_progressive_step_materialization(
            raw, outline_step=outline, outline_step_sha256=_DIGEST,
        ),
        role="progressive_planner_step_materialization",
        max_retries=len(responses) - 1, include_failed_response_on_retry=False,
        format_reminder=_shape_contract(outline),
    )


def test_first_rejection_has_the_recorded_mapping_shape():
    with pytest.raises(ValidationError) as caught:
        parse_progressive_step_materialization(json.dumps(_payload()))
    issue = safe_validation_issues(caught.value)[0]
    assert issue["location"] == ["step"]
    assert issue["issue_type"] == "value_error"
    assert issue["input_shape"] == {
        "kind": "mapping", "key_count": 27,
        "keys": [
            "comparison_exposure_level_index", "confidence_level", "custom_method",
            "denominator_policy", "depends_on", "event_level_index", "literature_bindings",
            "missing_exposure_policy", "missing_outcome_policy", "model_terms", "module_id",
            "objective", "outcome", "outcome_type", "outputs", "planned_analysis_role",
        ],
    }


@pytest.mark.parametrize(("index", "digest"), [
    (1, "c34bb466c56acb11560ab3d0f7bc3e2e2b2b9e1d52a653e4f4aa37f57da54986"),
    (2, "348d513e85124abb1c8a9bfe4a31b46cc0ca253f198f669e02f933d460284b5f"),
])
@pytest.mark.parametrize("outputs", [[], [_OUTPUT]])
def test_forbidden_scope_shapes_keep_their_recorded_diagnostics(index, digest, outputs):
    payload = _recorded_shapes()[index]
    payload["step"]["outputs"] = outputs
    with pytest.raises(ValidationError) as caught:
        parse_progressive_step_materialization(json.dumps(payload))
    assert hashlib.sha256(render_parse_failure(caught.value).encode()).hexdigest() == digest
    assert caught.value.errors()[0]["loc"] == ("step", "outline_population_scope")


@pytest.mark.parametrize("module", ["measurement_audit", "custom_analysis", "visualization", "report"])
def test_missing_output_feedback_names_the_existing_field_and_item_shape(module):
    payload = _payload()
    payload["step"]["module_id"] = module
    if module == "custom_analysis":
        payload["step"]["custom_method"] = "synthetic_method"
    before = copy.deepcopy(payload)
    with pytest.raises(ValidationError) as caught:
        parse_progressive_step_materialization(json.dumps(payload))
    feedback = render_parse_failure(caught.value)
    assert "step.outputs" in feedback
    assert "non-empty array" in feedback
    assert "product_id" in feedback and "semantic_role" in feedback
    assert "Do not add" in feedback
    assert payload == before


@pytest.mark.parametrize("module", ["visualization", "report", "absolute_risk_context"])
def test_repair_template_names_a_closed_roster_and_only_applicable_population_fields(module):
    contract = _shape_contract(_outline(module))
    assert "Allowed step keys are only those shown in this template" in contract
    assert "edit the existing field" in contract
    assert "remove the extra key" in contract
    assert "Do not add explanatory keys" in contract
    assert "step.outputs" in contract
    assert "explicit outline population_scope" not in contract
    population_guidance = contract.split("\n", 2)[2]
    if module == "absolute_risk_context":
        assert "step.population_scope" in population_guidance
        assert "step.population_scope_change_reason" in population_guidance
    else:
        assert "population_scope" not in population_guidance


@pytest.mark.parametrize("index", range(3))
def test_each_recorded_rejection_shape_can_be_repaired_within_the_attempt_budget(index):
    invalid = _recorded_shapes()[index]
    before = copy.deepcopy(invalid)
    llm, run = _run([invalid, _corrected()])
    result = run()
    assert len(llm.calls) == 2
    feedback = llm.calls[1][0][-1].content
    assert "edit the existing field" in feedback
    if index == 0:
        assert "step.outputs" in feedback and "non-empty array" in feedback
    else:
        assert "Extra inputs are not permitted" in feedback
        assert "remove the extra key" in feedback
    assert result.step.outputs[0].model_dump() == _OUTPUT
    assert result.step.population_scope is None
    assert result.step.depends_on == ["preceding_model"]
    assert result.step.product_inputs[0].product_id == "table:summary"
    assert invalid == before


def test_three_recorded_shapes_end_in_an_explained_strict_refusal():
    llm, run = _run(_recorded_shapes())
    with pytest.raises(StructuredResponseFailure) as caught:
        run()
    assert len(llm.calls) == 3
    failure = caught.value
    assert failure.role == "progressive_planner_step_materialization"
    assert "step.outputs" in failure.attempts[0].error_message
    assert [row.validation_issues[0]["issue_type"] for row in failure.attempts] == [
        "value_error", "extra_forbidden", "extra_forbidden",
    ]
    assert failure.__cause__.errors()[0]["loc"] == ("step", "outline_population_scope")
    feedback = llm.calls[2][0][-1].content
    assert "step.outputs" in feedback  # Earlier rule remains visible.
    assert "step.outline_population_scope" in feedback
    assert "remove the extra key" in feedback
