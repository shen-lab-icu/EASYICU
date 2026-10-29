"""A phenotype question that asks for outcome rates by phenotype is planned.

A question asking which phenotypes first-day vital signs and labs form, and
how a mortality rate differs between them, stopped before any plan:

- with no saved goal the family router read it as descriptive;
- its "mortality rate" wording made the article contract require the generic
  exposure-prevalence / outcome-by-exposure display, which no phenotyping
  step owns (a phenotype study has no exposure and no adjusted model);
- a broad vital-sign and laboratory roster exceeded the design's variable
  bound, which the Planner was told only after its first draft;
- the outcome-by-cluster description did not count as covering the outcome
  the question asked about.

Synthetic, case-neutral contexts only.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.family_spec_planner import family_spec_response_shape
from easyicu.research_agent.agents.progressive_planner import (
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.planning.family_spec import (
    PHENOTYPING_FAMILY_ID,
    build_family_spec_request,
    family_template_id_for_context,
)
from easyicu.research_agent.planning.family_spec.contract import (
    FamilySpecError,
    fit_roster_budget,
    spec_from_mapping,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.progressive_compiler import progressive_cohort_concept_ids
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.reporting.article_contract import build_article_analysis_contract
from easyicu.research_agent.schema import ConceptDescriptor, ResearchContext, VariableRole

from tests.research_agent.planning.family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _phenotyping_context,
    _phenotyping_payload,
    _run,
)

_ZH_QUESTION = "入 ICU 首日的生命体征和化验能分出哪几种表型？各型院内死亡率有何不同？"
_EN_QUESTION = (
    "Which phenotypes emerge from first-day vital signs and laboratory values, and "
    "how does the in-hospital mortality rate differ between them?"
)
_MEASUREMENTS = [f"marker_{index:02d}_max" for index in range(24)]


def _study(question: str = _ZH_QUESTION, *, broad: bool = False) -> ResearchContext:
    """The phenotyping study as a first turn leaves it: no saved family reading."""

    context = _phenotyping_context()
    extra = [
        ConceptDescriptor(
            name=name, description=f"first-day maximum of marker {name[7:9]}",
            role=VariableRole.LAB, dtype="float64", source_concept=name[:-4],
            analysis_window="icu_admission[0,24]h",
        )
        for name in _MEASUREMENTS
    ] if broad else []
    return context.model_copy(
        update={
            "research_question": question,
            "variables": [*context.variables, *extra],
            "primary_exposure": None,
            "user_preferences": context.user_preferences.model_copy(
                update={"inferred_analysis_family": None}
            ),
        }
    )


def _request(context: ResearchContext):
    variables = select_progressive_variables(context)
    return build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=variables,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode=None,
        cohort_concept_ids=progressive_cohort_concept_ids(context, variables),
    )


@pytest.mark.parametrize("question", [_ZH_QUESTION, _EN_QUESTION])
def test_the_question_routes_to_the_phenotyping_template_without_a_saved_goal(question) -> None:
    context = _study(question)

    types = candidate_analysis_types(context)

    assert types[0] == "trajectory_clustering"
    assert family_template_id_for_context(context, analysis_types=types) == PHENOTYPING_FAMILY_ID


@pytest.mark.parametrize("question", [_ZH_QUESTION, _EN_QUESTION])
def test_an_outcome_rate_by_phenotype_needs_no_exposure_display(question) -> None:
    phenotyping = build_article_analysis_contract(_study(question), analysis_type="trajectory_clustering")
    assert "descriptive_result" not in phenotyping.required_roles
    assert "exposure_outcome_distribution" not in {item.module_id for item in phenotyping.requirements}
    # Another family that compares an exposure still shows its event rates.
    association = build_article_analysis_contract(_study(question), analysis_type="association_study")
    assert "descriptive_result" in association.required_roles


def test_a_broad_roster_is_told_its_bound_before_the_first_draft() -> None:
    request = _request(_study(broad=True))
    budget = fit_roster_budget(request)
    offered = [item.name for item in request.feature_candidates if item.selectable]
    assert len(offered) > budget

    shape = family_spec_response_shape(request)

    assert (
        "The design names at most 24 variables, the row identity and the outcome included: "
        f"feature_variables (at most 22), baseline_variables and any cohort_membership_column "
        f"together name at most {budget}; keep the most informative."
    ) in shape
    over = _phenotyping_payload(request, features=offered[: budget - 1], baseline=["age", "sex"], membership=None)
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(spec_from_mapping(over), request)
    assert caught.value.reason_code == "family_spec_roster_exceeds_design"


def test_a_narrow_roster_is_not_told_a_bound_it_cannot_reach() -> None:
    assert "The design names at most" not in family_spec_response_shape(_request(_study()))


def test_the_phenotype_plan_covers_the_outcome_it_describes_by_cluster() -> None:
    context = _study(broad=True)
    request = _request(context)
    features = [item.name for item in request.feature_candidates if item.selectable]
    payload = _phenotyping_payload(
        request, features=features[: fit_roster_budget(request) - 2], baseline=["age", "sex"], membership=None,
    )

    llm, result = _run(context, [json.dumps(payload)], required_primary_cohort_selection_mode=None)

    assert len(llm.calls) == 1
    plan = result.output
    comparison = next(step for step in plan.steps if step.step_id == "cluster_characterization")
    assert comparison.scientific_action_id == "phenotyping.outcome_by_cluster"
    assert comparison.planned_analysis_role == "secondary"
    assert [step.step_id for step in plan.steps if step.planned_analysis_role == "primary"] == [
        "primary_cluster_solution",
    ]
    figure = next(step for step in plan.steps if step.step_id == "family_cohort_flow_figure")
    assert figure.planned_analysis_role == "auxiliary"
    review = build_plan_scientific_review(
        context=context, plan=plan, literature=None,
        figure_strategy=build_article_figure_strategy(context), runtime_authority=None,
    )
    codes = {finding.code for finding in review.findings}
    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" not in codes
    assert "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE" not in codes
    assert review.facts["missing_model_outcomes"] == []


def test_a_phenotype_plan_without_the_comparison_still_misses_the_outcome() -> None:
    context = _study()
    request = _request(context)
    payload = _phenotyping_payload(request, features=["hr_max", "lactate_max", "map_min"], baseline=[], membership=None)
    _llm, result = _run(context, [json.dumps(payload)], required_primary_cohort_selection_mode=None)
    plan = result.output
    without = plan.model_copy(
        update={"steps": [step for step in plan.steps if step.step_id != "cluster_characterization"]}
    )

    review = build_plan_scientific_review(
        context=context, plan=without, literature=None,
        figure_strategy=build_article_figure_strategy(context), runtime_authority=None,
    )

    codes = {finding.code for finding in review.findings}
    assert "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE" in codes
    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" in codes
    assert not review.approval_allowed
