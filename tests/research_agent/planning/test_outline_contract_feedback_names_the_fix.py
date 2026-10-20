"""Outline refusals name the repair and keep only safe rule coordinates."""

import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.progressive_payload import (
    parse_progressive_model,
    progressive_outline_structured_output_request,
)
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.agents.progressive_prompt_contracts import DESCRIPTIVE_SUMMARY_CONTRACT
from easyicu.research_agent.orchestration.progress import (
    ResumableProgressChannel,
    planner_retry_progress_callback,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveOutlineStep,
    ProgressivePlanOutline,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_diagnostics import (
    safe_projected_validation_issues,
    safe_validation_issues,
)
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
    _safe_declared_reason_code,
    call_llm_with_structured_retry,
    render_parse_failure,
    safe_structured_attempt_metadata,
)
from tests.research_agent.planning.progressive_planner_fixtures import _context


_PRIVATE = "private narrative that must never appear in diagnostic feedback"
_CODE = "outline_continuous_summary_role_action"


def _step(**changes):
    return {
        "step_id": "summary", "module_id": "continuous_outcome_summary",
        "planned_analysis_role": "secondary", "scientific_action_id": None,
        "objective": _PRIVATE, "variable_names": ["group_flag", "duration_a"],
        **changes,
    }


def _outline(steps=None, **changes):
    return {
        "analysis_type": "descriptive_epidemiology", "cohort_objective": _PRIVATE,
        "rationale": _PRIVATE, "steps": steps if steps is not None else [_step()],
        **changes,
    }


def _parse(raw):
    return parse_progressive_model(raw, ProgressivePlanOutline)


def _failure(payload):
    with pytest.raises(ValidationError) as caught:
        ProgressivePlanOutline.model_validate(payload)
    return caught.value


@pytest.mark.parametrize(("role", "action"), [
    ("primary", None),
    ("secondary", "descriptive.descriptive_summary"),
    ("primary", "descriptive.descriptive_summary"),
])
def test_continuous_summary_feedback_names_both_values_without_changing_the_design(role, action):
    payload = _outline([_step(planned_analysis_role=role, scientific_action_id=action)])
    before = copy.deepcopy(payload)
    error = _failure(payload)
    text = render_parse_failure(error)
    assert "set planned_analysis_role='secondary' and scientific_action_id=null" in text
    assert f"planned_analysis_role={role!r}" in text
    received_action = repr(action) if action is not None else "null"
    assert f"scientific_action_id={received_action}" in text
    assert "module_id='continuous_outcome_summary'" in text
    assert "population_scope=null" in text
    assert _PRIVATE not in text
    assert payload == before
    assert safe_validation_issues(error)[0]["reason_code"] == _CODE


@pytest.mark.parametrize("change", [
    {"planned_analysis_role": "primary"},
    {"scientific_action_id": "descriptive.descriptive_summary"},
    {"planned_analysis_role": "primary", "scientific_action_id": "descriptive.descriptive_summary"},
])
def test_redacted_attempt_failure_record_and_progress_name_the_same_rule(change, tmp_path):
    from easyicu.webserver.agent_pipeline_runs import _write_pipeline_failure_diagnostic

    audit, progress = [], []
    channel = ResumableProgressChannel(progress.append)
    channel.bind_audit_logger(SimpleNamespace(emit=lambda **event: audit.append(event)))
    llm = ScriptedMockLLMClient([json.dumps(_outline([_step(**change)]))])
    with pytest.raises(StructuredResponseFailure) as stopped:
        call_llm_with_structured_retry(
            llm, [], parser=_parse,
            role="progressive_planner_outline", max_retries=0,
            progress_callback=planner_retry_progress_callback(channel.emit, run_id="synthetic"),
        )
    assert stopped.value.attempts[0].reason_code == _CODE
    records = safe_structured_attempt_metadata(stopped.value.attempts)
    assert records == stopped.value.easyicu_structured_attempt_metadata
    relative = _write_pipeline_failure_diagnostic(
        wrapper_dir=tmp_path, exc=stopped.value, code="research_pipeline_plan_contract_exhausted",
    )
    persisted = json.loads((tmp_path / relative).read_text())
    assert persisted["structured_attempts"] == records
    assert _PRIVATE not in json.dumps(persisted)
    rejected = [row for row in progress if row.get("retry_phase") == "rejected"]
    rejected_audit = [row["detail"] for row in audit if row["detail"]["retry_phase"] == "rejected"]
    assert len(rejected) == len(rejected_audit) == 1
    for row in [*records, *rejected, *rejected_audit]:
        assert row["reason_code"] == _CODE
        assert row["validation_stage"] == "schema_validation"
        assert row["validation_issues"][0]["reason_code"] == _CODE
        assert row["validation_issues"][0]["location"] == ["steps", 0]
        assert row["validation_issues"][0]["issue_type"] == "value_error"
        assert _PRIVATE not in json.dumps(row)
        assert "descriptive.descriptive_summary" not in json.dumps(row)


def test_retry_accepts_only_the_planners_corrected_outline():
    invalid = _outline([_step(planned_analysis_role="primary")])
    corrected = _outline()
    llm = ScriptedMockLLMClient([json.dumps(invalid), json.dumps(corrected)])
    result = call_llm_with_structured_retry(
        llm, [], parser=_parse, max_retries=1,
        role="progressive_planner_outline", include_failed_response_on_retry=False,
    )
    assert len(llm.calls) == 2
    feedback = llm.calls[1][0][-1].content
    assert "set planned_analysis_role='secondary' and scientific_action_id=null" in feedback
    assert _PRIVATE not in feedback
    assert result.steps[0].planned_analysis_role == "secondary"
    assert result.steps[0].scientific_action_id is None
    assert invalid["steps"][0]["planned_analysis_role"] == "primary"


@pytest.mark.parametrize(("payload", "code", "fixes"), [
    (_outline([_step(population_scope="study_cohort")]),
     "outline_study_population_owner", ("population_scope=null", "cohort_definition", "exposure_outcome_distribution")),
    (_outline([_step(module_id="cohort_definition", population_scope="study_cohort",
                     population_scope_change_reason=_PRIVATE)]),
     "outline_study_population_amendment", ("population_scope_change_reason=null",)),
    (_outline([_step(population_scope="analysis_cohort", population_scope_change_reason=_PRIVATE)]),
     "outline_population_scope_owner", ("population_scope=null", "population_scope_change_reason=null", "absolute_risk_context")),
    (_outline([_step(module_id="absolute_risk_context", population_scope_change_reason=_PRIVATE)]),
     "outline_population_scope_required", ("population_scope='analysis_cohort'", "'primary_model'")),
    (_outline([], capability_gap=None),
     "outline_steps_or_capability_gap", ("steps", "capability_gap=null", "steps=[]")),
    (_outline(capability_gap="estimand_unsupported"),
     "outline_steps_or_capability_gap", ("steps", "capability_gap=null", "steps=[]")),
    (_outline([], capability_gap="levels_from_thresholds_unavailable"),
     "outline_capability_gap_concept", ("capability_gap_concept", "levels_from_thresholds_unavailable", "non-empty")),
    (_outline(capability_gap_concept=_PRIVATE),
     "outline_capability_gap_concept", ("capability_gap_concept=null",)),
    (_outline([_step(), _step()]),
     "outline_step_ids_unique", ("steps[1].step_id", "unique")),
    (_outline([_step(depends_on=[_PRIVATE])]),
     "outline_dependency_unknown", ("steps[0].depends_on", "step_id", "preceding")),
    (_outline([_step(depends_on=["later"]), _step(step_id="later")]),
     "outline_dependency_order", ("steps[0].depends_on", "steps[1]", "before")),
    (_outline([_step(depends_on=["summary"])]),
     "outline_dependency_order", ("steps[0].depends_on", "self", "remove")),
    (_outline([_step(module_id="custom_analysis", planned_analysis_role="primary"),
               _step(step_id="other", module_id="custom_analysis", planned_analysis_role="primary")]),
     "outline_primary_step_unique", ("planned_analysis_role", "'secondary'", "'sensitivity'", "'auxiliary'", "at most one")),
    (_outline([_step(module_id="custom_analysis", planned_analysis_role="sensitivity")],
              analysis_type="association_study"),
     "outline_association_sensitivity_parent", ("depends_on", "step_id", "module_id='adjusted_association'", "planned_analysis_role='primary'")),
    (_outline([_step(module_id="custom_analysis", planned_analysis_role="sensitivity")]),
     "outline_custom_sensitivity_action", ("scientific_action_id", "retrieved action", "depends_on")),
])
def test_each_outline_model_rule_names_its_fix_without_free_text(payload, code, fixes):
    error = _failure(payload)
    text = render_parse_failure(error)
    for fix in fixes:
        assert fix in text
    assert _PRIVATE not in text
    assert safe_validation_issues(error)[0]["reason_code"] == code
    if payload["steps"]:
        assert "module_id=" in text
        assert "planned_analysis_role=" in text
        assert "scientific_action_id=" in text
        assert "population_scope=" in text


def test_multiple_refused_steps_keep_each_reason_in_the_redacted_record():
    payload = _outline([
        _step(planned_analysis_role="primary"),
        _step(step_id="other", population_scope="analysis_cohort"),
    ])
    llm = ScriptedMockLLMClient([json.dumps(payload)])
    with pytest.raises(StructuredResponseFailure) as stopped:
        call_llm_with_structured_retry(
            llm, [], parser=_parse, max_retries=0,
        )
    row = stopped.value.easyicu_structured_attempt_metadata[0]
    assert row["reason_code"] == _CODE
    assert [(issue["location"], issue["reason_code"]) for issue in row["validation_issues"]] == [
        (["steps", 0], _CODE),
        (["steps", 1], "outline_population_scope_owner"),
    ]


def test_explicit_schema_wrappers_preserve_codes_but_prose_and_input_cannot_supply_them():
    underlying = _failure(_outline([_step(planned_analysis_role="primary")]))
    wrapper = ValueError(_PRIVATE)
    wrapper.__cause__ = underlying
    assert _safe_declared_reason_code(wrapper) == _CODE
    assert safe_validation_issues(wrapper)[0]["reason_code"] == _CODE
    wrapper.__cause__ = None
    wrapper.__context__ = underlying
    assert _safe_declared_reason_code(wrapper) is None

    class Untrusted(ValueError):
        def errors(self):
            return [{
                "loc": ("steps", 0), "type": "value_error", "msg": _CODE,
                "input": {"reason_code": _CODE},
                "ctx": {"reason_code": _CODE, "error": {"easyicu_safe_diagnostic": {
                    "owner": "easyicu.planning.progressive_outline_v1", "reason_code": _CODE,
                }}},
            }]

    assert _safe_declared_reason_code(Untrusted()) is None
    assert "reason_code" not in safe_validation_issues(Untrusted())[0]


@pytest.mark.parametrize(("owner", "code"), [
    ("untrusted_owner", _CODE),
    ("easyicu.planning.progressive_outline_v1", _PRIVATE),
])
def test_unknown_owners_and_unbounded_codes_are_not_diagnostics(owner, code):
    error = ValueError(_CODE)
    error.easyicu_safe_diagnostic = {"owner": owner, "reason_code": code}
    assert _safe_declared_reason_code(error) is None
    row = {"location": ["steps", 0], "issue_type": "value_error", "reason_code": _PRIVATE}
    assert "reason_code" not in safe_projected_validation_issues([row])[0]


@pytest.mark.parametrize("step", [
    _step(),
    _step(module_id="cohort_definition", population_scope="study_cohort"),
    _step(module_id="exposure_outcome_distribution", population_scope="study_cohort"),
    _step(module_id="absolute_risk_context", population_scope="analysis_cohort"),
    _step(module_id="absolute_risk_context", population_scope="primary_model",
          population_scope_change_reason=_PRIVATE),
])
def test_accepted_step_coordinates_remain_unchanged(step):
    result = ProgressivePlanOutline.model_validate(_outline([step])).steps[0]
    assert all(getattr(result, key) == value for key, value in step.items())


def test_prompt_and_outline_schema_state_the_continuous_summary_coordinate():
    text = "\n".join(message.content for message in ProgressivePlannerAgent.request_messages(_context()))
    block = text.split("Continuous secondary outcomes:", 1)[1].split("\n\n", 1)[0]
    assert 'planned_analysis_role="secondary"' in block
    assert "scientific_action_id=null" in block
    assert "variable_names" in block and "grouping" in block and "every selected continuous outcome" in block
    assert "descriptive.descriptive_summary does not apply to continuous_outcome_summary" in DESCRIPTIVE_SUMMARY_CONTRACT
    properties = ProgressiveOutlineStep.model_json_schema()["properties"]
    assert "continuous_outcome_summary" in properties["planned_analysis_role"]["description"]
    assert "secondary" in properties["planned_analysis_role"]["description"]
    assert "continuous_outcome_summary" in properties["scientific_action_id"]["description"]
    assert "null" in properties["scientific_action_id"]["description"]
    assert "grouping and every selected continuous outcome" in properties["variable_names"]["description"]
    request = progressive_outline_structured_output_request(
        analysis_types=["descriptive_epidemiology"],
        variable_names=["group_flag", "duration_a"],
        scientific_action_ids=["descriptive.descriptive_summary"],
    )
    # Metadata is stripped from the strict transport; do not grow it with prose.
    transport = json.loads(request.schema_json)["$defs"]["ProgressiveOutlineStep"]["properties"]
    assert all("description" not in field for field in transport.values())
