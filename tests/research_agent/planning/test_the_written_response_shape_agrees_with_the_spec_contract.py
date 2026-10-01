"""The written response shape asks for what the spec contract accepts.

A Provider route without strict-schema support reads the response contract
in words, with the first request and again in every retry reminder.  For a
proposed survival suite that text said ``"adjustment_set": [] (this family
fits no adjusted model)`` and asked for no labels of selected variables, while
the contract refuses an empty roster there and requires those labels.  A
real planning run then spent all three attempts on the contradiction.  Both
now read one predicate.  Synthetic contexts only (renal replacement therapy
and 90-day mortality; a categorical stage exposure; a descriptive phenotype
study); zero patient rows.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.family_spec_planner import (
    FAMILY_SPEC_GUIDE,
    family_spec_response_shape,
)
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
)
from easyicu.research_agent.planning.family_spec import FamilySpecError, validate_family_plan_spec
from easyicu.research_agent.planning.family_spec.contract import (
    planner_selects_adjustment,
    spec_from_mapping,
)
from easyicu.research_agent.providers.capabilities import llm_supports_strict_json_schema
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.research_agent.planning.family_spec_fixtures import (
    _context as _landmark_context,
    _descriptive_context,
    _request as _landmark_request,
)
from tests.support.survival_proposal import (
    AGE,
    ALLOWED,
    SEX,
    survival_context,
    survival_request,
    survival_spec,
)

NO_MODEL = '"adjustment_set": [] (this family fits no adjusted model)'
AT_LEAST_ONE = "; at least one entry"
LABEL_SELECTION = "and for every variable you select"


def _requests():
    return {
        "proposed_survival": survival_request(survival_context()),
        "landmark_planner": _landmark_request(_landmark_context(exact=False)),
        "landmark_exact": _landmark_request(_landmark_context(exact=True)),
        "descriptive": _landmark_request(_descriptive_context(), cohort_mode="all_input_rows"),
    }


def _refuses_an_empty_roster(request) -> bool:
    # Only the roster is empty; every family's required labels are present.
    spec = survival_spec(request, [])
    try:
        validate_family_plan_spec(spec_from_mapping(spec), request)
    except FamilySpecError as exc:
        return exc.reason_code == "family_spec_adjustment_set_empty"
    return False


@pytest.mark.parametrize("case", ["proposed_survival", "landmark_planner", "landmark_exact", "descriptive"])
def test_the_shape_demands_a_roster_exactly_where_the_contract_refuses_none(case):
    request = _requests()[case]
    shape = family_spec_response_shape(request)
    planner_roster = case in {"proposed_survival", "landmark_planner"}

    assert planner_selects_adjustment(request) is planner_roster
    assert (AT_LEAST_ONE in shape) is planner_roster is _refuses_an_empty_roster(request)
    assert (NO_MODEL in shape) is (case == "descriptive")
    assert (LABEL_SELECTION in shape) is (case != "landmark_exact")


def test_a_proposed_suite_shape_offers_its_candidates_by_their_field_names():
    request = survival_request(survival_context())
    shape = family_spec_response_shape(request)

    assert '"name" (one of ["age", "sex"])' in shape
    assert '"clinical_rationale"' in shape and '"rationale"' not in shape
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(spec_from_mapping(survival_spec(request, [])), request)
    # The refusal names the field the shape names, not a near miss of it.
    assert "clinical_rationale" in str(caught.value)
    assert "at least one whenever any is selectable" in FAMILY_SPEC_GUIDE


def test_a_route_without_a_strict_schema_reads_the_roster_request_first():
    context = survival_context()
    request = survival_request(context)
    llm = ScriptedMockLLMClient([json.dumps(survival_spec(request, [AGE, SEX]))])
    assert not llm_supports_strict_json_schema(llm)

    plan = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED, direct_comparator_literature_keys=[],
        enforce_article_contract=True, article_contract_context=context,
        planning_contract_context="", required_primary_cohort_selection_mode="all_input_rows",
    ).output

    first = "\n".join(message.content for message in llm.calls[0][0])
    assert "Response shape (no schema is attached on this route):" in first
    assert AT_LEAST_ONE in first and LABEL_SELECTION in first and NO_MODEL not in first
    assert plan.adjustment_proposal.covariates == ["age", "sex"]
