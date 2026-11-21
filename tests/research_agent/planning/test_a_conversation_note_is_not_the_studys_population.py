"""A note the conversation wrote about the study never states its population.

UX rc_150 age leg (S1780/S1782): the study's cohort ``review`` read
「纳入成人 ICU 患者；首次 ICU 住院与重复 ICU 住院的处理留待候选研究计划提出。」
-- the choice left to the plan. The Planner quoted 「首次 ICU 住院」 from it as
the study's own wording, the plan restricted the population to the first stay
of each hospitalization, and the export could not prove it. Only the
researcher's words state a population: the question, the study's own
inclusion statements and its exclusion statement.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from easyicu.research_agent.planning.family_spec.request import _study_cohort_wording
from easyicu.research_agent.planning.progressive_compiler import (
    ProgressivePlanCompileError,
    validate_progressive_foundation,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanFoundation,
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.schema import ResearchContext, UserPreferences
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _planner_context,
    _payload,
)

UX_REVIEW = "纳入成人 ICU 患者；首次 ICU 住院与重复 ICU 住院的处理留待候选研究计划提出。"
UX_QUESTION = "成人 ICU 患者中，ICU 入科时年龄（每增加 10 岁）与院内死亡的关联，校正性别。请用官方 MIMIC-IV 演示数据。"
FIRST_STAY = {"quote": "首次 ICU 住院", "kind": "first_icu_stay_per_hospitalization"}


def _context(question: str, **cohort: Any) -> ResearchContext:
    base = _planner_context()
    constraints = json.loads(base.user_preferences.data_constraints or "{}")
    constraints["cohort"] = {"label": "成人 ICU 患者", "review": UX_REVIEW, **cohort}
    return base.model_copy(update={
        "research_question": question,
        "user_preferences": UserPreferences(data_constraints=json.dumps(constraints, ensure_ascii=False)),
    })


def _foundation(context: ResearchContext, *criteria: dict[str, Any], source: str) -> None:
    skeleton = ProgressivePlanSkeleton.model_validate(_payload())
    cohort = skeleton.cohort.model_copy(update={"population_spec": {"criteria": [
        {"id": f"c{index}", "source": source, "role": "include", **item}
        for index, item in enumerate(criteria, start=1)
    ]}})
    validate_progressive_foundation(
        ProgressivePlanFoundation(
            cohort=cohort,
            display_labels=skeleton.display_labels,
            robustness_intents=skeleton.robustness_intents,
            know_how_decisions=skeleton.know_how_decisions,
        ),
        context=context,
        analysis_type=skeleton.analysis_type,
    )


def test_the_ux_review_note_is_not_quoted_as_the_studys_first_stay() -> None:
    with pytest.raises(ProgressivePlanCompileError) as refused:
        _foundation(_context(UX_QUESTION), FIRST_STAY, source="study_wording")

    assert refused.value.reason_code == "progressive_population_quote_not_verbatim"
    assert refused.value.path == "cohort.population_spec.criteria[0].quote"


@pytest.mark.parametrize("where", ["question", "exclusion_statement"])
def test_the_researchers_own_first_stay_wording_states_it(where) -> None:
    if where == "question":
        context = _context("首次 ICU 住院的患者中，ICU 入科时年龄与院内死亡的关联。")
        source = "question"
    else:
        context = _context(UX_QUESTION, exclusion_statement="仅纳入首次 ICU 住院，排除同次住院的再次入科")
        source = "study_wording"

    _foundation(context, FIRST_STAY, source=source)


def test_the_family_planner_is_not_offered_the_note_as_the_studys_wording() -> None:
    wording = _study_cohort_wording(_context(UX_QUESTION, exclusion_statement="排除再次入科"))

    assert wording == {"label": "成人 ICU 患者", "exclusion_statement": "排除再次入科"}
