"""An adjusted-association family refuses an empty Planner roster for repair.

When the roster is the Planner's to select, the spec contract accepted an
empty adjustment set although the host offered selectable confounders.  The
landmark templates then built a Table 1 with nothing to describe, and the
untyped schema failure ended the paid planning run instead of reaching the
Planner as a reason it can repair.  The proposed survival suite also showed
its open roster to the Planner as a binding empty list.  Synthetic contexts
only (renal replacement therapy and 90-day mortality; a categorical stage
exposure); zero patient rows.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.family_spec_planner import (
    FAMILY_SPEC_GUIDE,
    family_spec_user_prompt,
)
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
)
from easyicu.research_agent.planning.family_spec import FamilySpecError, validate_family_plan_spec
from easyicu.research_agent.planning.family_spec.contract import spec_from_mapping
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.research_agent.planning.family_spec_fixtures import (
    _context as _landmark_context,
    _request as _landmark_request,
    _spec_payload as _landmark_spec,
)
from tests.support.survival_proposal import (
    AGE,
    ALLOWED,
    SEX,
    survival_context,
    survival_request,
    survival_spec,
)

EMPTY_ROSTER = "family_spec_adjustment_set_empty"


def _refusal(spec: dict, request) -> FamilySpecError:
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(spec_from_mapping(spec), request)
    return caught.value


def test_a_proposed_survival_suite_refuses_an_empty_roster_naming_the_candidates():
    request = survival_request(survival_context())
    assert request.proposed_suite is not None and request.adjustment_selection == "planner_selectable"

    refusal = _refusal(survival_spec(request, []), request)

    assert refusal.reason_code == EMPTY_ROSTER and refusal.path == "adjustment_set"
    assert "age" in str(refusal) and "sex" in str(refusal)
    validate_family_plan_spec(spec_from_mapping(survival_spec(request, [AGE])), request)


def test_the_planner_repairs_an_empty_roster_on_its_next_answer():
    context = survival_context()
    request = survival_request(context)
    llm = ScriptedMockLLMClient(
        [json.dumps(survival_spec(request, [])), json.dumps(survival_spec(request, [AGE, SEX]))]
    )

    plan = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED, direct_comparator_literature_keys=[],
        enforce_article_contract=True, article_contract_context=context,
        planning_contract_context="", required_primary_cohort_selection_mode="all_input_rows",
    ).output

    assert len(llm.calls) == 2
    retry = "\n".join(message.content for message in llm.calls[1][0])
    assert "an empty adjustment set is not an adjusted estimate" in retry
    assert plan.adjustment_proposal.covariates == ["age", "sex"]
    table_one = next(step for step in plan.steps if step.method.startswith("table_one"))
    assert {"age", "sex"} <= set(table_one.inputs)


def test_a_landmark_association_family_refuses_an_empty_planner_roster_only():
    selectable = _landmark_request(_landmark_context(exact=False))
    assert selectable.adjustment_selection == "planner_selectable"
    assert _refusal(_landmark_spec(selectable), selectable).reason_code == EMPTY_ROSTER

    # An exact user-reviewed roster is the host's, so the spec may leave it empty.
    exact = _landmark_request(_landmark_context(exact=True))
    assert exact.adjustment_selection == "exact" and exact.exact_roster
    validate_family_plan_spec(spec_from_mapping(_landmark_spec(exact)), exact)


def test_the_proposal_shows_no_binding_roster_and_the_guide_hands_it_to_the_planner():
    request = survival_request(survival_context())
    prompt = family_spec_user_prompt(request, variable_descriptions={})
    design = json.loads(
        prompt.split("Host-fixed design (binding, not editable):\n", 1)[1].split("\n", 1)[0]
    )

    assert design["proposed_suite"]["exposure_onset_column"] == "rrt_first_time"
    assert "adjustment_columns" not in design["proposed_suite"]
    assert "adjustment_columns" not in prompt
    assert "with a proposed_suite instead" in FAMILY_SPEC_GUIDE
    assert "the adjustment set is yours" in FAMILY_SPEC_GUIDE
