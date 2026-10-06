"""A phenotype plan keeps every row its caller binds.

A reviewed candidate that kept every input row binds the next pass to an
all-rows cohort.  The phenotyping template still applied the study's typed age
and stay bounds, and offered a membership flag, so its plan filtered rows
against that contract and failed only after the Provider call, when the
pipeline compared modes.  The landmark, descriptive, prediction and survival
templates already apply typed bounds only to a predicate-filtered cohort.  Now
the phenotyping template does the same, a caller-bound all-rows request offers
no membership flag, and the plan text says when typed bounds restrict the rows.
Fixtures are generic.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.family_spec.contract import (
    FamilySpecError,
    spec_from_mapping,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec.phenotyping_template import (
    _cohort_intent,
)
from easyicu.research_agent.planning.progressive_compiler import progressive_cohort_concept_ids
from easyicu.research_agent.schema import ResearchContext

from tests.research_agent.planning.family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _phenotyping_context,
    _phenotyping_payload,
    _run,
)

_FEATURES = ["hr_max", "lactate_max", "map_min"]


def _adults(context: ResearchContext) -> ResearchContext:
    """The study's typed cohort keeps adults, as the Web study's cohort does."""

    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    constraints["cohort"] = {**constraints.get("cohort", {}), "age_min": 18}
    return context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            )
        }
    )


def _request(context: ResearchContext, cohort_mode: str | None):
    variables = select_progressive_variables(context)
    return build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=variables,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode=cohort_mode,
        cohort_concept_ids=progressive_cohort_concept_ids(context, variables),
    )


def _payload(request: Any, membership: str | None = None) -> dict:
    return _phenotyping_payload(
        request, features=_FEATURES, baseline=["age"], membership=membership
    )


def test_a_caller_bound_all_rows_cohort_applies_no_typed_bound_or_flag() -> None:
    request = _request(_adults(_phenotyping_context()), "all_input_rows")

    assert request.cohort_selection_mode == "all_input_rows" and request.age_min == 18.0
    assert request.membership_candidates == []
    spec = spec_from_mapping(_payload(request))
    validate_family_plan_spec(spec, request)
    intent = _cohort_intent(request, spec)
    assert intent.selection_mode == "all_input_rows"
    assert intent.inclusion == [] and intent.exclusion == []
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(
            spec_from_mapping(_payload(request, "phenotype_flag")), request
        )
    assert caught.value.path == "cohort_membership_column"


@pytest.mark.parametrize("cohort_mode", ["predicate_filtered", None])
def test_a_filtered_or_unbound_cohort_still_applies_its_typed_bound(cohort_mode) -> None:
    request = _request(_adults(_phenotyping_context()), cohort_mode)

    assert request.cohort_selection_mode == "predicate_filtered"
    assert "phenotype_flag" in request.membership_candidates
    spec = spec_from_mapping(_payload(request, "phenotype_flag"))
    validate_family_plan_spec(spec, request)
    intent = _cohort_intent(request, spec)
    assert intent.selection_mode == "predicate_filtered"
    assert [(item.concept_id, item.op) for item in intent.inclusion] == [
        ("phenotype_flag", "=="),
        ("age", ">="),
    ]


@pytest.mark.parametrize(
    ("cohort_mode", "population"),
    [
        ("all_input_rows", "All analysis rows of the study cohort;"),
        ("predicate_filtered", "Analysis rows of the study cohort;"),
    ],
)
def test_the_plan_says_whether_typed_bounds_restrict_its_rows(cohort_mode, population) -> None:
    context = _adults(_phenotyping_context())
    request = _request(context, cohort_mode)

    _llm, result = _run(
        context,
        [json.dumps(_payload(request))],
        required_primary_cohort_selection_mode=cohort_mode,
    )

    cohort = result.output.cohort
    assert cohort is not None and cohort.selection_mode == cohort_mode
    reviewable = " ".join(result.output.design_selection.selected.reviewable_plan)
    assert population in reviewable
    if cohort_mode == "predicate_filtered":
        assert "All analysis rows" not in reviewable
