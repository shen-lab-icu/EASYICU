"""The baseline contract and progressive retrieval must offer the same columns."""

import pytest

from easyicu.research_agent.agents.progressive_planner import select_progressive_variables
from easyicu.research_agent.planning.baseline_requirements import baseline_requirement_projection
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.schema import ConceptDescriptor

from .test_baseline_requirements import _bound_context


def context_with_alternate_values():
    context = _bound_context("age", "charlson")
    return context.model_copy(update={"variables": [
        *context.variables,
        *[ConceptDescriptor(name=f"score_{suffix}", source_concept="charlson", role="other", dtype="float64")
          for suffix in ("max", "min", "mean", "first")],
        *[ConceptDescriptor(name=f"score_{suffix}", source_concept="charlson", role="meta", dtype="int64")
          for suffix in ("n", "measured", "first_time", "last_time")],
    ]})


def required_columns(context):
    return {name for table in baseline_requirement_projection(context)["tables"]
            for coordinate in [table["group_by"], *table["variables"]]
            for name in coordinate["available_columns"]}


def test_source_family_quota_cannot_prune_an_accepted_clinical_alternative() -> None:
    context = context_with_alternate_values()
    before = context.model_dump(mode="json")
    selected = select_progressive_variables(context)
    assert required_columns(context) <= set(selected)
    assert len(selected) <= 48
    assert context.model_dump(mode="json") == before


def test_relevance_ranking_cannot_displace_baseline_anchors() -> None:
    context = context_with_alternate_values()
    context = context.model_copy(update={"variables": [
        *context.variables,
        *[ConceptDescriptor(name=f"age_extra_{n}", role="demographic", dtype="float64")
          for n in range(20)],
    ]})
    selected = select_progressive_variables(context, max_variables=12)
    assert required_columns(context) <= set(selected)
    assert len(selected) == 12
    assert {context.primary_exposure, context.target_outcome} <= set(selected)


def test_too_small_budget_fails_before_provider_instead_of_hiding_required_columns() -> None:
    with pytest.raises(ProgressivePlanCompileError) as caught:
        select_progressive_variables(context_with_alternate_values(), max_variables=2)
    assert caught.value.reason_code == "progressive_required_variables_exceed_budget"


def test_required_roster_does_not_promote_measurement_metadata() -> None:
    columns = required_columns(context_with_alternate_values())
    assert {"score_first", "score_mean", "score_min", "score_max", "cci_value"} <= columns
    assert not columns & {"score_n", "score_measured", "score_first_time", "charlson_time"}


def test_mandatory_clinical_values_leave_optional_quota_for_audit_inputs() -> None:
    selected = select_progressive_variables(context_with_alternate_values())
    assert {"score_n", "score_measured", "score_first_time"} <= set(selected)
