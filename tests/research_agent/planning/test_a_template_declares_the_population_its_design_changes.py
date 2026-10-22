"""A family template that changes a source-bound population says so.

The landmark templates read absolute risk in the primary model's population.
A revision whose source plan read it in the broader analysis cohort changes
that population by the template's design, not by a model's choice.  Hosted
pilot: after the host applied a 24 h landmark setting, every fresh plan
stopped at progressive_outline_population_requirement_drift -- the template
has no way to state an amendment, so it could never pass.
"""

from __future__ import annotations

import json

from easyicu.research_agent.planning.population_requirements import (
    bind_population_requirements,
)

from .family_spec_fixtures import _context, _request, _run, _spec_payload


def _bound(scope: str):
    return bind_population_requirements(_context(), {
        "source_plan_sha256": "a" * 64,
        "populations": [{
            "source_step_id": "risk_summary",
            "output_product": "table:absolute_risk_context",
            "population_scope": scope,
        }],
    })


def _risk_step(scope: str):
    context = _bound(scope)
    _llm, result = _run(context, [json.dumps(_spec_payload(_request(context)))])
    assert result.facts is not None, result
    [outline_step] = [s for s in result.facts.outline.steps if s.module_id == "absolute_risk_context"]
    [materialized] = [
        m.step for m in result.facts.materializations if m.step.module_id == "absolute_risk_context"
    ]
    return outline_step, materialized


def test_a_broader_source_population_is_declared_as_an_amendment() -> None:
    outline_step, materialized = _risk_step("analysis_cohort")

    assert outline_step.population_scope == "primary_model"
    assert "landmark" in (outline_step.population_scope_change_reason or "")
    assert materialized.population_scope_change_reason == outline_step.population_scope_change_reason


def test_the_same_population_needs_no_amendment() -> None:
    outline_step, materialized = _risk_step("primary_model")

    assert outline_step.population_scope_change_reason is None
    assert materialized.population_scope_change_reason is None
