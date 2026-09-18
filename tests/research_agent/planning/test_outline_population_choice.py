"""A proposed population correction cannot disappear during step generation."""

import json

import pytest

from easyicu.research_agent.agents.progressive_payload import (
    parse_progressive_model,
    progressive_step_materialization_request,
)
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.population_requirements import bind_population_requirements
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveOutlineStep,
    ProgressivePlanOutline,
    ProgressiveSkeletonStep,
    ProgressiveStepMaterialization,
)
from easyicu.research_agent.planning.progressive_resume import validate_progressive_materialization_coordinate
from easyicu.research_agent.planning.progressive_compiler import ProgressivePlanCompileError

from .test_population_requirements import _requirement
from .scientific_review_fixtures import _context


def _outline(scope=None, reason=None):
    return ProgressiveOutlineStep(
        step_id="risk_summary", module_id="absolute_risk_context",
        planned_analysis_role="secondary", objective="Describe the declared analysis population.",
        variable_names=["exposure", "outcome"], population_scope=scope,
        population_scope_change_reason=reason,
    )


def _plan(step):
    return ProgressivePlanOutline(
        analysis_type="association_study", cohort_objective="Describe eligible participants.",
        steps=[step], rationale="Preserve the proposed population for review.",
    )


def _materialization(outline, **changes):
    fields = outline.model_dump(mode="json", exclude={"variable_names", "literature_citation_keys"})
    fields.update(primary_exposure="exposure", outcome="outcome", **changes)
    return ProgressiveStepMaterialization(
        outline_step_sha256=canonical_sha256(outline.model_dump(mode="json")),
        foundation=None, step=ProgressiveSkeletonStep.model_validate(fields),
    )


def _validate(outline, materialization):
    validate_progressive_materialization_coordinate(
        materialization, outline_step=outline, step_index=0,
        outline_step_sha256=canonical_sha256(outline.model_dump(mode="json")),
    )


def test_archived_outline_digest_unchanged_but_fresh_choice_required():
    old = _plan(_outline()).model_dump(mode="json")
    assert "population_scope" not in old["steps"][0]
    assert "population_scope_change_reason" not in old["steps"][0]
    restored = ProgressivePlanOutline.model_validate(old)
    assert canonical_sha256(restored.model_dump(mode="json")) == canonical_sha256(old)
    with pytest.raises(ValueError, match="outline requires an explicit population_scope"):
        parse_progressive_model(json.dumps(old), ProgressivePlanOutline)


@pytest.mark.parametrize("scope", ["primary_model", "analysis_cohort"])
@pytest.mark.parametrize("reason", [None, "Correct the requested denominator to the prespecified population."])
def test_selected_population_and_explanation_survive_both_transports(scope, reason):
    outline = _outline(scope, reason)
    parsed = parse_progressive_model(_plan(outline).model_dump_json(), ProgressivePlanOutline)
    assert parsed.steps[0] == outline
    _validate(outline, _materialization(outline))
    request = progressive_step_materialization_request(
        outline_step=outline, outline_step_sha256=canonical_sha256(outline.model_dump(mode="json")),
        variable_names=outline.variable_names, scientific_action_ids=[],
    )
    fields = json.loads(request.schema_json)["$defs"]["ProgressiveSkeletonStep"]["properties"]
    assert fields["population_scope"] == {"type": "string", "const": scope}
    assert fields["population_scope_change_reason"] == (
        {"type": "string", "const": reason} if reason else {"type": "null"}
    )


@pytest.mark.parametrize("changes", [
    {"population_scope": "analysis_cohort"},
    {"population_scope": None},
    {"population_scope_change_reason": None},
    {"population_scope_change_reason": "Use a different scientific justification without revising the outline."},
])
def test_detail_cannot_silently_undo_a_declared_correction(changes):
    outline = _outline("primary_model", "Restore the requested primary model denominator.")
    with pytest.raises(ProgressivePlanCompileError, match="outline-owned fields"):
        _validate(outline, _materialization(outline, **changes))


def test_legacy_materialization_is_not_retrofitted():
    outline = _outline()
    _validate(outline, _materialization(outline, population_scope="analysis_cohort"))


def test_source_change_requires_disclosure_at_outline_before_step_calls():
    context = bind_population_requirements(_context(), _requirement("analysis_cohort").model_dump(mode="json"))
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            _plan(_outline("primary_model")), analysis_types=["association_study"],
            variable_names=["exposure", "outcome"], allowed_literature_citation_keys=[],
            article_context=context,
        )
    assert caught.value.reason_code == "progressive_outline_population_requirement_drift"


@pytest.mark.parametrize("fields", [
    {"module_id": "measurement_audit", "population_scope": "primary_model"},
    {"population_scope_change_reason": "A reason with no specified target population."},
    {"population_scope": "primary_model", "population_scope_change_reason": " " * 20},
])
def test_invalid_outline_population_declarations_are_rejected(fields):
    payload = _outline().model_dump(mode="json")
    with pytest.raises(ValueError):
        ProgressiveOutlineStep.model_validate({**payload, **fields})
