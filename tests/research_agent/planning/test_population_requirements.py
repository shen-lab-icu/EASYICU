"""Population choice survives cosmetic revisions and input preparation."""

import pytest

from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.planning.population_requirements import (
    bind_population_requirements,
    candidate_population_requirements,
    context_population_requirements,
    validate_population_choice,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from .scientific_review_fixtures import _context


def _plan(scope, reason=None, step_id="renamed_result"):
    return AnalysisPlan(
        research_question="Describe an exposure and hospital mortality.",
        steps=[
            AnalysisStep(
                step_id=step_id,
                planned_analysis_role="secondary",
                intent="Describe the prespecified population.",
                method="absolute_risk_context",
                expected_outputs=["table:absolute_risk_context"],
                population_scope=scope,
                population_scope_change_reason=reason,
            )
        ],
    )


def _requirement(scope):
    return candidate_population_requirements(
        _plan(scope, step_id="original_result").model_dump(mode="json"), "a" * 64
    )


@pytest.mark.parametrize("scope", ["primary_model", "analysis_cohort"])
def test_preservation_is_product_bound_and_allows_explicit_scientific_amendment(scope):
    requirement = _requirement(scope)
    context = bind_population_requirements(
        _context(), requirement.model_dump(mode="json")
    )
    validate_population_choice(
        context, product="table:absolute_risk_context", scope=scope, change_reason=None
    )
    same = build_plan_scientific_review(context=context, plan=_plan(scope))
    assert same.facts["population_scope_changes"] == []
    other = "analysis_cohort" if scope == "primary_model" else "primary_model"
    with pytest.raises(ValueError, match="Preserve table:absolute_risk_context"):
        validate_population_choice(
            context,
            product="table:absolute_risk_context",
            scope=other,
            change_reason=None,
        )
    failed = build_plan_scientific_review(context=context, plan=_plan(other))
    assert not failed.approval_allowed
    assert any(f.code == "PLAN_POPULATION_REQUIREMENT_DRIFT" for f in failed.findings)
    reason = "The requested scientific amendment describes the alternative population separately."
    validate_population_choice(
        context,
        product="table:absolute_risk_context",
        scope=other,
        change_reason=reason,
    )
    changed = build_plan_scientific_review(context=context, plan=_plan(other, reason))
    finding = next(
        f for f in changed.findings if f.code == "POPULATION_SCOPE_AMENDMENT_DECLARED"
    )
    assert finding.requires_user_authorization and reason in finding.message
    assert changed.facts["population_scope_changes"][0]["explicit_amendment"]
    assert changed.facts["plan_population_requirements"] == requirement.model_dump(
        mode="json"
    )


def test_absent_legacy_contract_does_not_change_serialization_or_allow_resume_retrofit(
    tmp_path,
):
    plan = _plan(None)
    assert "population_scope_change_reason" not in plan.steps[0].model_dump(mode="json")
    assert (
        candidate_population_requirements(plan.model_dump(mode="json"), "a" * 64)
        is None
    )
    original = _context()
    assert bind_population_requirements(original, None) == original
    plain = PipelineConfig(workdir=tmp_path)
    assert "bound_population_requirements" not in plain.canonical_payload()
    payload = _requirement("primary_model").model_dump(mode="json")
    with pytest.raises(ValueError, match="requires require_human_plan_review"):
        PipelineConfig(workdir=tmp_path, bound_population_requirements=payload)
    config = PipelineConfig(
        workdir=tmp_path,
        bound_population_requirements=payload,
        require_human_plan_review=True,
    )
    assert config.canonical_payload()["bound_population_requirements"] == payload
    bound = bind_population_requirements(_context(), payload)
    assert context_population_requirements(bound).source_plan_sha256 == "a" * 64
    assert bind_population_requirements(bound, payload, restoring=True) == bound
    with pytest.raises(ValueError, match="binding_drift"):
        bind_population_requirements(_context(), payload, restoring=True)
    with pytest.raises(ValueError, match="binding_drift"):
        bind_population_requirements(bound, None, restoring=True)
    missing = build_plan_scientific_review(
        context=bound, plan=AnalysisPlan(research_question="Describe risk", steps=[])
    )
    assert any(f.code == "PLAN_POPULATION_REQUIREMENT_DRIFT" for f in missing.findings)


def test_blank_scope_amendment_is_not_a_change_declaration():
    with pytest.raises(ValueError):
        _plan("analysis_cohort", " " * 20)
