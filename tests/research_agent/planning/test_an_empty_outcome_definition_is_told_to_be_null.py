"""An outcome definition that declares nothing is told how to repair itself.

UX leg 1 (2026-11-19, S1446): for a 28-day mortality step the planner sent
``outcome_definition`` as a mapping with an estimand and none of free_days,
ordered_levels or count. The nested validator answered "declare exactly one
free_days, ordered_levels, or count definition", which steered every repair
towards filling a definition; the step-level rule that such a step takes null
never ran. Three structured attempts and three retries repeated the same
violation and the plan was exhausted. The message now names the repair.
Message only: what validates is unchanged. Synthetic payload.
"""

import hashlib
import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.progressive_payload import (
    parse_progressive_step_materialization,
)
from easyicu.research_agent.contracts.model_terms import OutcomeDefinition
from easyicu.research_agent.providers.structured_diagnostics import safe_validation_issues
from easyicu.research_agent.providers.structured_retry import render_parse_failure
from tests.research_agent.planning.test_step_materialization_feedback_names_the_allowed_fields import (
    _corrected,
)

_EMPTY = {"estimand": "Risk of death by day 28 after ICU admission.", "free_days": None,
          "ordered_levels": [], "count": None}
# The walk's recorded violation (diagnostics/research_pipeline_failure.json).
_WALK_VIOLATION = "7bea4a9d7e8828e1441567f1a68f29ffc1caa4e44d3a1d506df94fa823ef9b5e"


def _with_definition(definition):
    payload = _corrected()
    payload["step"]["outcome_definition"] = definition
    return json.dumps(payload)


def test_an_empty_definition_is_told_to_be_null_at_the_walk_coordinate():
    with pytest.raises(ValidationError) as caught:
        parse_progressive_step_materialization(_with_definition(_EMPTY))
    (issue,) = safe_validation_issues(caught.value)
    assert issue["location"] == ["step", "outcome_definition"]
    assert sorted(issue["input_shape"]["keys"]) == ["count", "estimand", "free_days", "ordered_levels"]
    rendered = render_parse_failure(caught.value)
    assert "set outcome_definition to null unless outcome_type is free_days, ordinal or count" in rendered
    assert hashlib.sha256(rendered.encode()).hexdigest() != _WALK_VIOLATION


def test_the_repair_the_message_names_validates():
    assert parse_progressive_step_materialization(_with_definition(None)).step.outcome_definition is None


def test_declared_definitions_keep_their_rules():
    with pytest.raises(ValidationError, match="declare exactly one"):
        OutcomeDefinition(estimand="Days alive and free of support.", ordered_levels=["a", "b", "c"],
                          count={"family": "poisson"})
    with pytest.raises(ValidationError, match="at least three unique ordered levels"):
        OutcomeDefinition(estimand="Ordered recovery category.", ordered_levels=["a", "b"])
    assert OutcomeDefinition(estimand="Ordered recovery category.", ordered_levels=["a", "b", "c"])
