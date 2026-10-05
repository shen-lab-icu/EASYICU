"""Unresolved candidate selection still needs an exact predicate wire shape."""

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.progressive_prompt_contracts import foundation_shape_contract
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveCohortIntent,
    ProgressiveFoundationMaterialization,
)
from easyicu.research_agent.providers.structured_diagnostics import (
    safe_projected_validation_issues,
    safe_validation_issues,
)
from tests.research_agent.planning.progressive_planner_fixtures import _foundation_payload


def test_unresolved_foundation_displays_nested_predicate_shape_without_choosing_population():
    prompt = foundation_shape_contract(outline_sha256="a" * 64, host_cohort=None)
    assert '"selection_mode":"<all_input_rows|predicate_filtered>"' in prompt
    assert '"inclusion":[],"exclusion":[]' in prompt
    assert '"concept_id":"<copy an allowed cohort concept id>"' in prompt
    assert '"op":"<==|!=|<|<=|>|>=|in|not_in|missing|not_missing>"' in prompt
    assert '"number_value":null' in prompt
    assert "This shape does not require adding a cohort restriction" in prompt


def test_a_free_foundation_says_a_named_population_is_applied_only_by_its_predicates():
    """A Planner named its cohort for the question's population and kept every input row.

    Its outline called the input rows the eligible population, and nothing it
    was shown said they are the whole source cohort.  When the Planner chooses
    the cohort, the contract now says so and keeps the shape neutral; a cohort
    the caller binds is copied, so its contract carries no such choice.
    """

    free = foundation_shape_contract(outline_sha256="a" * 64, host_cohort=None)
    assert "The input rows are every row of the source cohort in the data authority" in free
    assert "only the inclusion and exclusion contracts shown there are already applied" in free
    assert "is applied only by the predicates written here" in free
    assert "name the cohort for the rows it keeps" in free

    bound = foundation_shape_contract(
        outline_sha256="a" * 64,
        host_cohort=ProgressiveCohortIntent(name="source", selection_mode="all_input_rows"),
    )
    required = foundation_shape_contract(
        outline_sha256="a" * 64, host_cohort=None, required_cohort_selection_mode="predicate_filtered"
    )
    for contract in (bound, required):
        assert "The input rows are every row" not in contract


def test_foundation_diagnostics_keep_only_closed_schema_coordinates_not_values():
    payload = _foundation_payload()
    payload["foundation"]["cohort"] = {
        "name": "candidate", "selection_mode": "predicate_filtered", "exclusion": [],
        "inclusion": [{
            "concept_id": "registered_flag", "anchor": "icu_admission",
            "start_offset_hours": 0, "end_offset_hours": 24,
            "aggregation": "max", "op": "==",
            "value": {"mode": "private-mode-token", "number_value": "private-value-token"},
            "private-field-token": "private-extra-token",
        }],
    }
    with pytest.raises(ValidationError) as caught:
        ProgressiveFoundationMaterialization.model_validate(payload)
    issues = safe_validation_issues(caught.value)
    paths = [issue["location"] for issue in issues]
    assert ["foundation", "cohort", "inclusion", 0, "value", "mode"] in paths
    assert ["foundation", "cohort", "inclusion", 0, "value", "number_value"] in paths
    assert ["foundation", "cohort", "inclusion", 0, "<other>"] in paths
    assert "private-" not in json.dumps(issues)
    assert safe_projected_validation_issues(issues) == issues
