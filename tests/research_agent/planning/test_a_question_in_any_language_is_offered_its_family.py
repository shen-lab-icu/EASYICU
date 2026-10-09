"""A question is offered the family it asks for, whatever its wording.

The keyword scorer reads only some phrasings in some languages: a Chinese
association question asked as "X 是否与 Y 相关" matched no association cue, and
since a pair of columns stopped implying association, no family offered it.
Descriptive, association and prediction families are now always offered when
executable; scoring only orders them, so a misreading cannot lock a Planner out
of the family the question asks for.  The user's own family choice still
narrows the list to itself, and a family nothing executes is never offered.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    candidate_analysis_types,
)
from easyicu.research_agent.planning.analysis_types import (
    infer_analysis_type,
    list_analysis_types,
)
from easyicu.research_agent.schema import UserPreferences
from tests.research_agent.planning.progressive_planner_fixtures import _context

_ALWAYS_OFFERED = {"descriptive_epidemiology", "association_study", "prediction_model"}
_NOTHING_EXECUTES = frozenset(
    spec.key for spec in list_analysis_types() if spec.capability_id is None
)


def _asked(question: str, **preferences: str):
    return _context().model_copy(
        update={
            "research_question": question,
            "user_preferences": UserPreferences(**preferences),
        }
    )


@pytest.mark.parametrize(
    ("family", "zh", "en"),
    [
        (
            "association_study",
            "成人 ICU 患者入科时的血乳酸水平是否与院内死亡相关？",
            "Is the blood lactate level at ICU admission associated with in-hospital death in adult ICU patients?",
        ),
        (
            "association_study",
            "校正年龄和疾病严重度后，入科血肌酐是否与院内死亡相关？",
            "After adjusting for age and illness severity, is admission creatinine associated with in-hospital death?",
        ),
        (
            "prediction_model",
            "利用入 ICU 后前 24 小时的数据，建立并内部验证一个预测院内死亡的模型。",
            "Develop and internally validate a model that predicts in-hospital death from data in the first 24 hours after ICU admission.",
        ),
        (
            "survival",
            "入 ICU 后 90 天的生存是否随入科乳酸水平而不同？",
            "Does 90-day survival after ICU admission differ by admission lactate level?",
        ),
    ],
)
def test_a_question_and_its_translation_are_offered_its_family(family, zh, en):
    for question in (zh, en):
        context = _asked(question)
        # Without a typed exposure the wording is all the scorer reads.
        for asked in (context, context.model_copy(update={"primary_exposure": None})):
            offered = candidate_analysis_types(asked)
            assert family in offered
            assert _ALWAYS_OFFERED <= set(offered)


def test_a_question_the_scorer_misreads_is_still_offered_its_family():
    # The scorer reads no association cue here and ranks description first.
    context = _asked("成人 ICU 患者入科时的血乳酸水平是否与院内死亡相关？").model_copy(
        update={"primary_exposure": None}
    )
    offered = candidate_analysis_types(context)

    assert offered[0] != "association_study"
    assert _ALWAYS_OFFERED <= set(offered)


def test_the_scorers_reading_ranks_first_and_the_always_offered_follow():
    context = _asked(
        "Develop and internally validate a model that predicts in-hospital death "
        "from data in the first 24 hours after ICU admission."
    ).model_copy(update={"primary_exposure": None})
    offered = candidate_analysis_types(context)

    # Nothing in the wording reads as association, so it is offered last.
    assert offered.index("prediction_model") < offered.index("association_study")
    assert offered[-1] == "association_study"


def test_the_planner_is_told_the_order_is_a_reading_not_a_ruling():
    prompt = ProgressivePlannerAgent.request_messages(_context())[1].content

    assert (
        "the order is the host's reading of the question's wording, not a ruling"
        in prompt
    )


def test_the_users_own_family_still_narrows_the_list_to_itself():
    context = _asked(
        "利用入 ICU 后前 24 小时的数据，建立并内部验证一个预测院内死亡的模型。",
        inferred_analysis_family="association_study",
    )

    assert candidate_analysis_types(context) == ("association_study",)
    assert infer_analysis_type(context).key == "association_study"


@pytest.mark.parametrize(
    "question",
    [
        "在 MIMIC-IV 和 eICU 两个数据库中复现乳酸与院内死亡的关联。",
        "Validate an existing mortality score in an external cohort.",
        "Compare cohort definitions for sepsis and how they change mortality.",
    ],
)
def test_a_family_nothing_executes_is_never_offered(question):
    offered = candidate_analysis_types(_asked(question))

    assert offered
    assert not set(offered) & _NOTHING_EXECUTES
