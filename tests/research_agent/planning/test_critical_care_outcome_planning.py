"""Metadata-only compilation of explicit ordered and rate estimands."""

import pytest

from easyicu.research_agent.planning.progressive_compiler import _compile_adjusted_association
from easyicu.research_agent.planning.progressive_contract import ProgressiveSkeletonStep
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
from tests.research_agent.planning.progressive_planner_fixtures import _context, _payload


@pytest.mark.parametrize("kind,definition,method", [
    ("ordinal", {"estimand": "Common OR toward higher prespecified recovery levels",
                 "ordered_levels": ["poor", "fair", "good"]}, "statsmodels_ordered_logit"),
    ("free_days", {"estimand": "Common OR for more ventilator-free days, death scored zero",
                   "free_days": {"horizon_days": 28, "support_concept": "vent_ind", "death_score": 0}},
     "statsmodels_ordered_logit"),
    ("count", {"estimand": "Episode rate ratio over observed follow-up days",
               "count": {"family": "negative_binomial", "exposure_column": "followup"}},
     "statsmodels_negative_binomial"),
])
def test_compiler_preserves_the_declared_outcome_and_estimator(kind, definition, method):
    payload = _payload()["steps"][4]
    payload.update(outcome_type=kind, outcome_definition=definition)
    step = ProgressiveSkeletonStep.model_validate(payload)
    context = _context()
    for column in step.outcome_definition.input_columns:
        context.variables.append(ConceptDescriptor(name=column, dtype="float64", role=VariableRole.OTHER))
    result = _compile_adjusted_association(
        context=context, variables={item.name: item for item in context.variables}, step=step, step_index=4,
    )[0]
    assert result.outcome_definition == step.outcome_definition
    assert result.outcome_type == kind
    assert result.method_family == method


def test_free_days_missing_support_descriptor_refuses_compilation():
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError

    payload = _payload()["steps"][4]
    payload.update(outcome_type="free_days", outcome_definition={
        "estimand": "Common OR of more ventilator-free days, death scored zero",
        "free_days": {"horizon_days": 28, "support_concept": "vent_ind", "death_score": 0},
    })
    context = _context()
    with pytest.raises(ProgressivePlanCompileError, match="unknown"):
        _compile_adjusted_association(
            context=context, variables={item.name: item for item in context.variables},
            step=ProgressiveSkeletonStep.model_validate(payload), step_index=4,
        )


def test_schema_compression_preserves_every_progressive_constraint():
    import json
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
    from easyicu.research_agent.providers.strict_json_schema import (
        closed_pydantic_json_schema, compact_schema_definition_names, intern_schema_shapes,
    )
    from tests.support.schema_names import authority_schema_json

    source = closed_pydantic_json_schema(ProgressivePlanSkeleton)
    wire = intern_schema_shapes(compact_schema_definition_names(source))
    assert authority_schema_json(json.dumps(wire)) == source
    assert len(json.dumps(wire, separators=(",", ":"))) < 16_000


def test_schema_reference_collision_fails_closed(monkeypatch):
    from easyicu.research_agent.providers import strict_json_schema as owner

    monkeypatch.setattr(owner, "transport_definition_name", lambda _: "collision")
    with pytest.raises(ValueError, match="collision"):
        owner.compact_schema_definition_names({"$defs": {
            "First": {"type": "string"}, "Second": {"type": "integer"},
        }})
