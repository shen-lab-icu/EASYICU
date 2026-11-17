"""Unclaimed longitudinal methods stop before a planner/provider turn."""

import pytest

from easyicu.research_agent.planning.family_spec import (
    FamilySpecError,
    build_family_spec_request,
)
from tests.support.survival_proposal import survival_context
from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    FAMILY_SPEC_STRATEGY,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import UserPreferences
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)


@pytest.mark.parametrize(
    "question",
    [
        "Fit mixed-effects trajectories of creatinine with random slopes.",
        "用混合效应模型拟合肌酐轨迹并估计随机斜率。",
        "Use a joint longitudinal-survival model for lactate trajectories and death.",
        "Fit a random-slope model to lactate trajectories.",
        "Fit a joint-model for lactate trajectories and death.",
    ],
)
def test_unclaimed_trajectory_methods_name_supported_alternatives(question):
    context = survival_context(research_question=question)
    with pytest.raises(FamilySpecError) as caught:
        build_family_spec_request(
            context,
            analysis_types=["trajectory_clustering"],
            variable_roster=[v.name for v in context.variables],
            allowed_literature_citation_keys=[],
        )
    assert caught.value.reason_code == "trajectory_design_unsupported"
    assert "fixed-window trajectory phenotyping" in str(caught.value)
    assert "per-stay phenotyping" in str(caught.value)


def test_the_planner_refuses_before_calling_its_scripted_provider():
    context = survival_context(
        research_question="Fit mixed-effects lactate trajectories with random slopes.",
        user_preferences=UserPreferences(
            inferred_analysis_family="trajectory_clustering"
        ),
    )
    llm = ScriptedMockLLMClient([])
    with pytest.raises(
        ProgressivePlanCompileError, match="fixed-window trajectory phenotyping"
    ) as caught:
        ProgressivePlannerAgent(llm).run_attempt(
            context,
            planner_strategy=FAMILY_SPEC_STRATEGY,
            allowed_literature_citation_keys=[],
            enforce_article_contract=False,
        )
    assert llm.calls == []
    assert caught.value.reason_code == "progressive_trajectory_design_unsupported"


@pytest.mark.parametrize(
    "question",
    [
        "Use fixed-window trajectory phenotyping without a mixed-effects model.",
        "Do not fit random-slope models; use fixed-window trajectories.",
        "不用混合效应模型，使用固定窗口轨迹表型。",
    ],
)
def test_a_negated_unsupported_method_does_not_reject_fixed_window_phenotyping(
    question,
):
    from easyicu.research_agent.planning.trajectory_design_support import (
        unsupported_trajectory_design,
    )

    assert unsupported_trajectory_design(question) is None
