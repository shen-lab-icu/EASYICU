"""A result extension is an exact addition, never permission to change its source."""
from copy import deepcopy
from tests.support.result_append import addition, approvable_review, source_plan, tier_plan

import pytest

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.services import PipelineServices
from easyicu.research_agent.schema import AnalysisStep
from easyicu.webserver.research_plan_revision import PlanRevisionGuard
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError






def test_repair_guard_can_admit_an_exact_host_bound_result_addition():
    source = source_plan()
    addition = AnalysisStep(step_id="append_cases", intent="Check complete cases",
                           method="complete_case_sensitivity", inputs=["marker", "outcome"],
                           planned_analysis_role="sensitivity", expected_outputs=["table:append_cases"])
    expected = source.model_copy(deep=True, update={"steps": [*source.steps, addition], "revision": source.revision + 1})
    original = deepcopy(source.model_dump(mode="json"))
    guard = PlanRevisionGuard.for_append(
        plan=source.model_dump(mode="json"), review=approvable_review(source),
        expected_plan=expected.model_dump(mode="json"), action="sensitivity",
    )
    services = PipelineServices(plan_revision_guard=guard)
    services.validate_plan_revision(plan=expected, context=None)
    extra = expected.model_copy(deep=True)
    extra.steps.append(addition.model_copy(update={"step_id": "unrequested", "expected_outputs": ["table:unrequested"]}))
    with pytest.raises(ResearchPipelineRunError, match="append"):
        services.validate_plan_revision(plan=extra, context=None)
    assert source.model_dump(mode="json") == original




def revision(source, steps=None):
    return source.model_copy(deep=True, update={
        "steps": deepcopy([*source.steps, *(steps or [addition()])]), "revision": source.revision + 1,
    })


def bind(source, expected, action="sensitivity"):
    return PlanRevisionGuard.for_append(
        plan=source.model_dump(mode="json"), review=approvable_review(source),
        expected_plan=expected.model_dump(mode="json"), action=action,
    )


@pytest.mark.parametrize("action,method,scientific_action", [
    ("sensitivity", "complete_case_sensitivity", None),
    ("exposure", "adjusted_association_models", "association.adjusted_association"),
    ("subgroup", "adjusted_association_models", "association.effect_modification"),
])
def test_each_host_compiled_addition_is_exact(action, method, scientific_action):
    source = source_plan()
    expected = revision(source, [addition(method, action=scientific_action)])
    guard = bind(source, expected, action)
    PipelineServices(plan_revision_guard=guard).validate_plan_revision(plan=expected, context=None)
    changed = expected.model_copy(deep=True)
    changed.steps[-1].inputs.append("unrequested_covariate")
    with pytest.raises(ResearchPipelineRunError) as error:
        guard.validate_plan(plan=changed, context=None)
    assert error.value.code == "plan_append_scope_changed"


@pytest.mark.parametrize("edit", [
    "remove_original", "remove_addition", "reorder", "rename", "method", "inputs", "outputs", "role",
    "dependency", "question", "revision", "duplicate_addition",
])
def test_no_edit_to_the_bound_append_is_admitted(edit):
    source = source_plan()
    expected = revision(source)
    guard = bind(source, expected)
    changed = expected.model_copy(deep=True)
    if edit == "remove_original":
        changed.steps.pop(0)
    elif edit == "remove_addition":
        changed.steps.pop()
    elif edit == "reorder":
        changed.steps.reverse()
    elif edit == "rename":
        changed.steps[0].step_id = "different"
    elif edit == "method":
        changed.steps[-1].method = "e_value"
    elif edit == "inputs":
        changed.steps[0].inputs.append("another_marker")
    elif edit == "outputs":
        changed.steps[0].expected_outputs = ["table:replacement"]
    elif edit == "role":
        changed.steps[-1].planned_analysis_role = "primary"
    elif edit == "dependency":
        changed.steps[-1].inputs.remove("table:original")
    elif edit == "question":
        changed.research_question = "A different synthetic research question"
    elif edit == "revision":
        changed.revision += 1
    else:
        changed.steps.append(addition().model_copy(update={"step_id": "another", "expected_outputs": ["table:another"]}))
    with pytest.raises(ResearchPipelineRunError) as error:
        guard.validate_plan(plan=changed, context=None)
    assert error.value.code == "plan_append_scope_changed"
    assert guard.previous_plan == source.model_dump(mode="json")


@pytest.mark.parametrize("edit", ["prefix", "empty", "question", "revision", "id_collision", "output_collision", "forward_dependency"])
def test_host_cannot_bind_a_structurally_unsafe_append(edit):
    source = source_plan()
    expected = revision(source)
    if edit == "prefix":
        expected.steps[0].inputs.append("another_marker")
    elif edit == "empty":
        expected.steps.pop()
    elif edit == "question":
        expected.research_question = "A different synthetic research question"
    elif edit == "revision":
        expected.revision += 1
    elif edit == "id_collision":
        expected.steps[-1].step_id = expected.steps[0].step_id
    elif edit == "output_collision":
        expected.steps[-1].expected_outputs = expected.steps[0].expected_outputs
    else:
        expected.steps[-1].inputs.append("table:missing_producer")
    with pytest.raises((ResearchPipelineRunError, ValueError)):
        bind(source, expected)




def test_article_addition_changes_only_the_new_run_tier():
    source = tier_plan("first_result")
    before = source.model_dump_json()
    expected = revision(source)
    expected.design_selection = expected.design_selection.model_copy(update={"execution_tier": "article"})
    guard = bind(source, expected, "article")
    guard.validate_plan(plan=expected, context=None)
    assert source.model_dump_json() == before
    assert guard.previous_plan["design_selection"]["execution_tier"] == "first_result"
    assert expected.design_selection.execution_tier == "article"
    assert "new approval required" in guard.prompt_contract()
    assert approvable_review(source).plan_sha256 != canonical_sha256(expected.model_dump(mode="json"))


@pytest.mark.parametrize("action,old_tier,new_tier", [
    ("sensitivity", "first_result", "article"), ("article", "article", "article"),
    ("article", "first_result", "first_result"), ("exposure", "article", "first_result"),
])
def test_tier_changes_require_the_exact_article_append(action, old_tier, new_tier):
    source = tier_plan(old_tier)
    expected = revision(source)
    expected.design_selection = expected.design_selection.model_copy(update={"execution_tier": new_tier})
    with pytest.raises(ResearchPipelineRunError):
        bind(source, expected, action)


@pytest.mark.parametrize("change", ["plan_digest", "not_approvable", "unknown_action"])
def test_invalid_append_binding_fails_closed(change):
    source = source_plan()
    review = approvable_review(source)
    if change == "plan_digest":
        review = review.model_copy(update={"plan_sha256": "f" * 64})
    elif change == "not_approvable":
        review = review.model_copy(update={"approval_allowed": False})
    with pytest.raises(ResearchPipelineRunError):
        PlanRevisionGuard.for_append(
            plan=source.model_dump(mode="json"), review=review,
            expected_plan=revision(source).model_dump(mode="json"),
            action="replace" if change == "unknown_action" else "sensitivity",
        )


def test_bound_source_and_addition_are_detached_from_mutable_callers():
    source = source_plan()
    expected = revision(source)
    accepted = expected.model_copy(deep=True)
    guard = bind(source, expected)
    source.steps[0].inputs.append("mutated_source")
    expected.steps[-1].inputs.append("mutated_addition")
    guard.validate_plan(plan=accepted, context=None)
    assert "mutated_source" not in guard.source_plan_json
    assert "mutated_addition" not in guard.append_plan_json


def test_an_empty_plan_is_not_a_result_to_extend():
    source = source_plan().model_copy(update={"steps": []})
    expected = revision(source, [addition().model_copy(update={"inputs": ["marker", "outcome"]})])
    with pytest.raises(ResearchPipelineRunError):
        bind(source, expected)


def test_append_preserves_serialized_steps_even_when_a_default_could_be_reinferred():
    source = source_plan()
    payload = revision(source).model_dump(mode="json")
    payload["steps"][0].pop("sensitivity_spec_ids")
    with pytest.raises(ResearchPipelineRunError):
        PlanRevisionGuard.for_append(
            plan=source.model_dump(mode="json"), review=approvable_review(source),
            expected_plan=payload, action="sensitivity",
        )
