"""A signed trajectory plan states the population its owners analyze.

The signed representation reads the whole staged panel and excludes a stay
only under its SOFA-2 window rule, so every plan the trajectory authority owns
selects all input rows.  ``bind_plan`` replaces a Planner draft with the
authority's projection; the draft's population never survives the binding,
and a plan without that population (or with a filter the owners do not apply)
is plan drift.  Synthetic authority and plans only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
    cohort_definition_has_explicit_selection,
)
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    SIGNED_TRAJECTORY_POPULATION,
    TrajectoryScientificAuthorityError,
    build_trajectory_scientific_runtime_authority,
)

QUESTION = (
    "Which organ-dysfunction trajectory classes emerge over the first 72 h of "
    "an ICU stay from SOFA-2 components, and how does 28-day mortality differ "
    "by class?"
)
OWNER_POPULATION = {
    "name": "primary",
    "inclusion": [],
    "exclusion": [],
    "derived_from_named": None,
    "locked_at": "not_locked",
    "selection_mode": "all_input_rows",
}


def _authority():
    design = load_trajectory_design(
        {"coordinate_concepts": ["sofa2_cardio", "sofa2_renal", "sofa2_resp"]}
    )
    return build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(design, protocol_content_sha256="0" * 64)
    )


def _draft(authority, *, cohort) -> AnalysisPlan:
    """A Planner draft that names the signed owners, with its own population."""

    payload = authority.development_execution_only_plan(
        research_question=QUESTION
    ).model_dump(mode="json")
    payload["cohort"] = cohort
    return AnalysisPlan.model_validate(payload)


def _age_filter() -> ConceptPredicate:
    return ConceptPredicate("age", TimeWindow("icu_admit", 0, 24), "first", ">=", 18)


def test_the_projection_states_every_input_row_as_its_population():
    plan = _authority().development_execution_only_plan(research_question=QUESTION)

    assert plan.model_dump(mode="json")["cohort"] == OWNER_POPULATION
    assert dict(SIGNED_TRAJECTORY_POPULATION)["selection_mode"] == "all_input_rows"
    # The caller-bound population gate of every package-bound run requires an
    # explicit selection on the bound plan.
    assert cohort_definition_has_explicit_selection(plan.cohort)


@pytest.mark.parametrize(
    "draft_cohort",
    [
        pytest.param(None, id="no_population"),
        pytest.param(
            {"name": "web_study_fixture", "selection_mode": "all_input_rows"},
            id="all_rows_under_another_label",
        ),
        pytest.param(
            {
                "name": "Adult ICU stays",
                "inclusion": [_age_filter().to_dict()],
                "exclusion": [],
            },
            id="a_filter_the_owners_do_not_apply",
        ),
    ],
)
def test_binding_states_the_owner_population_whatever_the_draft_said(draft_cohort):
    authority = _authority()
    authorities = ScientificRuntimeAuthorities(trajectory=authority, current_case=None)

    bound, _findings = authorities.bind_plan(_draft(authority, cohort=draft_cohort))
    rebound, _ = authorities.bind_plan(bound)

    assert bound.model_dump(mode="json")["cohort"] == OWNER_POPULATION
    assert rebound.model_dump(mode="json")["cohort"] == OWNER_POPULATION
    authority.validate_plan(bound)


@pytest.mark.parametrize(
    "population",
    [
        pytest.param(None, id="no_population"),
        pytest.param(
            CohortDefinition(name="primary", inclusion=(_age_filter(),)),
            id="predicate_filtered",
        ),
        pytest.param(CohortDefinition(name="primary"), id="empty_predicate_filtered"),
    ],
)
def test_a_signed_plan_without_the_owner_population_is_drift(population):
    authority = _authority()
    plan = authority.development_execution_only_plan(research_question=QUESTION)

    with pytest.raises(TrajectoryScientificAuthorityError, match="population"):
        authority.validate_plan(plan.model_copy(update={"cohort": population}))
