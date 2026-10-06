"""An outcome-free phenotype study is planned from its features and reviewed.

Cross-sectional phenotypes are discovered from window-bound features; an
outcome the study has is only described by cluster.  The family router still
asked every phenotyping study for a binary outcome and a population flag, so
a question that asks only which phenotypes the features form went to
free-form planning, and the family request refused it.  The template now
describes no outcome when the study names none: the outcome-by-cluster
comparison, which reads an outcome, is not projected, and no baseline roster
is offered for it.  Synthetic, case-neutral contexts only.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

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
    FamilySpecRequest,
    spec_from_mapping,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.progressive_compiler import progressive_cohort_concept_ids
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import ResearchContext

from tests.research_agent.planning.family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _phenotyping_context,
    _phenotyping_payload,
    _run,
)

_FEATURES = ["hr_max", "lactate_max", "map_min"]
_OUTCOME_FINDINGS = {
    "OUTCOME_DEFINITION_UNRESOLVED",
    "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE",
    "PHENOTYPING_COMPARISON_CONTRACT_INVALID",
}


def _study(*, outcome: str | None = None, flag: str | None = None) -> ResearchContext:
    """The phenotyping study, asking only which phenotypes its features form."""

    context = _phenotyping_context()
    outcomes = [outcome] if outcome else []
    return context.model_copy(
        update={
            "research_question": (
                "Among ICU stays, which candidate subphenotypes emerge from first-24-hour "
                "vitals and labs by unsupervised clustering?"
            ),
            "cohort": context.cohort.model_copy(
                update={"outcome_columns": outcomes, "requested_outcome_columns": outcomes}
            ),
            "target_outcome": outcome,
            "endpoint": context.endpoint if outcome else None,
            "primary_exposure": flag,
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


def _plan(context: ResearchContext, *, baseline: list[str]):
    request = _request(context)
    llm, result = _run(
        context,
        [json.dumps(_phenotyping_payload(request, features=_FEATURES, baseline=baseline, membership=None))],
        required_primary_cohort_selection_mode=None,
    )
    assert len(llm.calls) == 1
    return result.output


def _findings(context: ResearchContext, plan) -> set[str]:
    return {
        finding.code
        for finding in build_plan_scientific_review(
            context=context,
            plan=plan,
            literature=None,
            figure_strategy=build_article_figure_strategy(context),
            runtime_authority=None,
        ).findings
    }


@pytest.mark.parametrize(
    ("outcome", "flag"),
    [(None, None), (None, "phenotype_flag"), ("death", None), ("death", "phenotype_flag")],
)
def test_the_router_templates_a_phenotype_study_with_or_without_an_outcome(outcome, flag) -> None:
    context = _study(outcome=outcome, flag=flag)

    types = candidate_analysis_types(context)

    assert types[0] == "trajectory_clustering"
    assert family_template_id_for_context(context, analysis_types=types) == PHENOTYPING_FAMILY_ID


def test_a_study_whose_flag_is_not_closed_is_still_not_templated() -> None:
    context = _study(flag="score_first")

    types = candidate_analysis_types(context)

    assert family_template_id_for_context(context, analysis_types=types) is None


def test_the_request_names_no_outcome_no_exposure_and_offers_no_baseline_roster() -> None:
    request = _request(_study())

    assert (request.outcome, request.outcome_levels, request.event_level_index) == ("", [], 0)
    assert (request.primary_exposure, request.exposure_kind, request.exposure_levels) == ("", "none", [])
    assert request.adjustment_candidates == []
    assert set(_FEATURES) <= {item.name for item in request.feature_candidates if item.selectable}
    with pytest.raises(ValidationError, match="without an outcome has no outcome levels"):
        FamilySpecRequest.model_validate(
            {**request.model_dump(mode="json"), "outcome_levels": ["0", "1"], "event_level_index": 1}
        )
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(
            spec_from_mapping(
                _phenotyping_payload(request, features=_FEATURES, baseline=["age"], membership=None)
            ),
            request,
        )
    assert caught.value.reason_code == "family_spec_baseline_variable_unavailable"


def test_the_outcome_free_plan_describes_its_features_and_needs_no_outcome() -> None:
    context = _study()

    plan = _plan(context, baseline=[])

    assert [step.step_id for step in plan.steps] == [
        "cohort_accounting",
        "feature_quality_audit",
        "primary_cluster_solution",
        "cluster_number_selection",
        "cluster_stability",
    ]
    assert all(name for step in plan.steps for name in step.inputs)
    selected = plan.design_selection.selected
    assert "No outcome; clusters are described by their features only." in selected.reviewable_plan
    design_text = " ".join(
        [selected.estimand, selected.observation_window, selected.figure_role, selected.supports]
    )
    assert "distribution" not in design_text and "None" not in design_text
    assert not _OUTCOME_FINDINGS & _findings(context, plan)


def test_a_chinese_question_reads_the_same_outcome_slot_in_chinese() -> None:
    context = _study().model_copy(
        update={"research_question": "ICU 入住首 24 小时的生命体征与化验能聚成哪些候选亚表型？"}
    )

    plan = _plan(context, baseline=[])

    assert "问题不含结局；聚类只按其特征描述。" in plan.design_selection.selected.reviewable_plan


def test_a_study_with_an_outcome_still_describes_it_by_cluster() -> None:
    context = _study(outcome="death")

    plan = _plan(context, baseline=["age"])

    characterization = plan.steps[-1]
    assert characterization.step_id == "cluster_characterization"
    spec = characterization.phenotype_comparison_spec
    assert spec is not None and spec.outcome_columns == ["death"]
    assert [variable.name for variable in spec.variables][:2] == ["death", "age"]
    assert not _OUTCOME_FINDINGS - {"OUTCOME_DEFINITION_UNRESOLVED"} & _findings(context, plan)
