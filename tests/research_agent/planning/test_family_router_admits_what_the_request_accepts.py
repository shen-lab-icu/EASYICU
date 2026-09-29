"""The family router offers a template only when its request can be sealed.

- A closed exposure domain larger than the categorical request admits (a total
  score's integer range) fits no template.  The attempt falls back to
  Progressive v2 instead of failing before any Planner call.
- Trajectory clustering and cross-sectional phenotype discovery share one
  family.  The cross-sectional template claims neither a question that asks
  for trajectories nor a context carrying a bound fixed-window trajectory; the
  static prediction template does not claim the latter either.

Synthetic, case-neutral contexts only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
)
from easyicu.research_agent.planning.analysis_types import (
    longitudinal_trajectory_requested,
)
from easyicu.research_agent.planning.family_spec import (
    PHENOTYPING_FAMILY_ID,
    PREDICTION_FAMILY_ID,
    family_template_id_for_context,
)
from easyicu.research_agent.planning.family_spec.contract import (
    MAX_EXPOSURE_LEVELS,
    FamilySpecRequest,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import ConceptDescriptor, ResearchContext, VariableRole
from easyicu.research_agent.trajectory.contract import (
    infer_fixed_window_trajectory_metadata,
)
from easyicu.research_agent.trajectory.plan_contract import trajectory_context_is_bound

from .family_spec_fixtures import (
    ALLOWED_CITATIONS,
    _context,
    _descriptive_context,
    _phenotyping_context,
    _prediction_context,
    _request,
)

#: Each categorical headline's synthetic context and the cohort mode its request uses.
_CATEGORICAL_FAMILIES = {
    "association_study": (_context, "predicate_filtered"),
    "descriptive_epidemiology": (_descriptive_context, None),
    "trajectory_clustering": (_phenotyping_context, None),
}

_LONGITUDINAL_QUESTIONS = (
    "Among ICU stays, which longitudinal trajectories of heart rate and lactate over "
    "the first 72 hours emerge by clustering, and how does in-hospital mortality "
    "differ between them?",
    "Cluster ICU stays by the time-varying course of heart rate and lactate over the "
    "first three days and compare in-hospital mortality across the clusters.",
    "Identify subphenotypes from repeated measurements of vital signs during the "
    "first 72 h and compare their in-hospital mortality.",
    "基于入 ICU 后 72 小时的纵向心率和乳酸轨迹进行聚类，并比较各类别的院内死亡。",
    "识别心率和乳酸随时间动态变化的患者亚型，并描述各亚型的院内死亡。",
)

_CROSS_SECTIONAL_QUESTIONS = (
    "Which subphenotypes emerge from first-24-hour haemodynamic and laboratory values "
    "by unsupervised clustering, and how does in-hospital mortality differ between them?",
    "基于入 ICU 24 小时内的生命体征和实验室指标，通过无监督聚类识别患者亚型，"
    "并比较各亚型的院内死亡率。",
)


def _with_score_exposure(context: ResearchContext, level_count: int) -> ResearchContext:
    """Make the exposure a total score whose integer range has ``level_count`` levels."""

    name = str(context.primary_exposure)
    score = ConceptDescriptor(
        name=name,
        description="total severity score",
        role=VariableRole.COMPOSITE_SCORE,
        dtype="float64",
        valid_range=[0.0, float(level_count - 1)],
        is_ordinal=True,
        source_concept=name,
        analysis_window="icu_admission[0,24]h",
    )
    return context.model_copy(
        update={
            "variables": [score if item.name == name else item for item in context.variables]
        }
    )


def _with_bound_trajectory(context: ResearchContext) -> ResearchContext:
    """Add one fixed-window ordinal trajectory (two windows) to the context."""

    windows = [
        ConceptDescriptor(
            name=f"severity_state_h{start}_{end}",
            role=VariableRole.ORDINAL_SCORE,
            dtype="int64",
            is_ordinal=True,
            fixed_window_trajectory=infer_fixed_window_trajectory_metadata(
                column_name=f"severity_state_h{start}_{end}",
                values=pd.Series([0, 1, 2], dtype="int64"),
                source_scale="ordinal",
            ),
        )
        for start, end in ((0, 6), (6, 12))
    ]
    return context.model_copy(update={"variables": [*context.variables, *windows]})


def _with_question(context: ResearchContext, question: str) -> ResearchContext:
    return context.model_copy(update={"research_question": question})


def _planner_calls_of_one_attempt(context: ResearchContext) -> int:
    """Run one family-spec attempt with no scripted reply; count the Planner calls."""

    llm = ScriptedMockLLMClient([])
    with pytest.raises(Exception, match="exhausted|attempts failed"):
        ProgressivePlannerAgent(llm).run_attempt(
            context,
            planner_strategy=FAMILY_SPEC_STRATEGY,
            allowed_literature_citation_keys=ALLOWED_CITATIONS,
            enforce_article_contract=True,
            article_contract_context=context,
            planning_contract_context="",
        )
    return len(llm.calls)


@pytest.mark.parametrize("headline", sorted(_CATEGORICAL_FAMILIES))
@pytest.mark.parametrize(
    "level_count", [2, MAX_EXPOSURE_LEVELS, MAX_EXPOSURE_LEVELS + 1, 72]
)
def test_a_categorical_template_is_offered_only_when_its_request_seals(
    headline: str, level_count: int
) -> None:
    make, cohort_mode = _CATEGORICAL_FAMILIES[headline]
    context = _with_score_exposure(make(), level_count)
    types = candidate_analysis_types(context)
    assert types[0] == headline

    family = family_template_id_for_context(context, analysis_types=types)

    if level_count > MAX_EXPOSURE_LEVELS:
        assert family is None
        return
    request = _request(context, cohort_mode=cohort_mode)
    assert family is not None and request.family_id == family
    assert len(request.exposure_levels) == level_count


def test_a_declared_level_list_beyond_the_bound_is_not_read_as_continuous() -> None:
    context = _context()
    name = str(context.primary_exposure)
    codes = ConceptDescriptor(
        name=name,
        description="source code of the injury",
        role=VariableRole.OTHER,
        dtype="float64",
        ordinal_levels=list(range(MAX_EXPOSURE_LEVELS + 6)),
        source_concept="injury_code",
        analysis_window="icu_admission[0,24]h",
    )
    context = context.model_copy(
        update={
            "variables": [codes if item.name == name else item for item in context.variables]
        }
    )

    assert (
        family_template_id_for_context(context, analysis_types=candidate_analysis_types(context))
        is None
    )


def test_the_request_contract_and_the_router_share_one_level_bound() -> None:
    field = FamilySpecRequest.model_fields["exposure_levels"]
    bounds = [item.max_length for item in field.metadata if hasattr(item, "max_length")]

    assert bounds == [MAX_EXPOSURE_LEVELS]


def test_a_score_exposure_beyond_the_bound_reaches_the_planner_through_progressive_v2() -> None:
    context = _with_score_exposure(_phenotyping_context(), MAX_EXPOSURE_LEVELS + 1)

    assert _planner_calls_of_one_attempt(context) >= 1


@pytest.mark.parametrize("question", _LONGITUDINAL_QUESTIONS)
def test_a_question_asking_for_trajectories_is_left_to_the_general_planner(
    question: str,
) -> None:
    context = _with_question(_phenotyping_context(), question)
    types = candidate_analysis_types(context)
    assert types[0] == "trajectory_clustering"

    assert longitudinal_trajectory_requested(context)
    assert family_template_id_for_context(context, analysis_types=types) is None
    assert _planner_calls_of_one_attempt(context) >= 1


@pytest.mark.parametrize(
    "question", [_phenotyping_context().research_question, *_CROSS_SECTIONAL_QUESTIONS]
)
def test_a_cross_sectional_phenotype_question_keeps_its_template(question: str) -> None:
    context = _with_question(_phenotyping_context(), question)
    types = candidate_analysis_types(context)

    assert not longitudinal_trajectory_requested(context)
    assert family_template_id_for_context(context, analysis_types=types) == PHENOTYPING_FAMILY_ID


def test_the_longitudinal_cue_reads_the_question_only() -> None:
    base = _phenotyping_context()
    context = base.model_copy(
        update={
            "user_preferences": base.user_preferences.model_copy(
                update={
                    "extra_notes": "Describe trajectories over time.",
                    "timing_and_design": "longitudinal follow-up",
                }
            )
        }
    )

    assert not longitudinal_trajectory_requested(context)


@pytest.mark.parametrize(
    ("make", "static_template"),
    [(_phenotyping_context, PHENOTYPING_FAMILY_ID), (_prediction_context, PREDICTION_FAMILY_ID)],
)
def test_a_bound_fixed_window_trajectory_is_not_templated_one_row_per_stay(
    make, static_template: str
) -> None:
    context = make()
    assert (
        family_template_id_for_context(context, analysis_types=candidate_analysis_types(context))
        == static_template
    )

    bound = _with_bound_trajectory(context)

    assert trajectory_context_is_bound(bound)
    assert (
        family_template_id_for_context(bound, analysis_types=candidate_analysis_types(bound))
        is None
    )
