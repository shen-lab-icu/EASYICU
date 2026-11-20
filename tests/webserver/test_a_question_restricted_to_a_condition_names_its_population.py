"""A question restricted to a recorded condition names that population (leg 3, S1733).

Leg 3's frozen direction asked 「在脓毒症 ICU 患者中，ICU 入科后 0–24 小时的乳酸水平或
峰值乳酸与院内死亡有什么关联？」.  The launch named no concept for 「脓毒症」, because a
population is not something the sentence studies, so the requirement list never
had to account for it and the plan studied every input row.  A population
restricted to a condition the dictionary records is now a named concept: the
requirement list must read it, and a plan that neither reads nor selects on it
goes back (``question_requirements._defines_the_population``).  Metadata and
synthetic requirements only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.question_requirements import (
    UNACCOUNTED_REASON,
    NamedQuestionConcept,
    QuestionRequirement,
    bind_named_question_concepts,
    question_requirement_problems,
)
from easyicu.webserver.research_launch_scientific import (
    _metadata_only_planning_coordinates,
)
from easyicu.webserver.study_intent import stated_population_conditions

_LEG3_ZH = "在脓毒症 ICU 患者中，ICU 入科后 0–24 小时的乳酸水平或峰值乳酸与院内死亡有什么关联？"
_LEG3_EN = (
    "Among ICU patients with sepsis, how are lactate level or peak lactate in the first "
    "24 hours after ICU admission associated with in-hospital death?"
)
_SEPSIS = ["sep3", "sep3_sofa1", "sep3_sofa2"]


def _named(question: str) -> list[list[str]]:
    coordinates = _metadata_only_planning_coordinates(question=question, database="miiv")
    return [item["concepts"] for item in coordinates["question_named_concepts"]]


@pytest.mark.parametrize(
    ("question", "words"),
    [
        pytest.param(_LEG3_ZH, "脓毒症", id="leg3-zh"),
        pytest.param(_LEG3_EN, "sepsis", id="leg3-en"),
        pytest.param("研究脓毒症患者前24小时最高乳酸与院内死亡", "脓毒症", id="zh-without-zhong"),
        pytest.param(
            "In septic patients, is the first-day maximum lactate associated with in-hospital death?",
            "septic",
            id="en-adjective",
        ),
    ],
)
def test_a_population_restricted_to_sepsis_is_named(question: str, words: str) -> None:
    assert stated_population_conditions(question) == ((tuple(_SEPSIS), words),)
    assert _SEPSIS in _named(question)


@pytest.mark.parametrize(
    "question",
    [
        pytest.param(
            "在成人 ICU 患者中，ICU 入科后 0–24 小时的乳酸水平或峰值乳酸与院内死亡有什么关联？",
            id="no-condition",
        ),
        pytest.param("ICU 入科后 24 小时最高乳酸与发生脓毒症有什么关联？", id="sepsis-as-outcome"),
        pytest.param(
            "不研究脓毒症患者；在成人 ICU 患者中，前24小时最高乳酸与院内死亡有什么关联？",
            id="negated",
        ),
    ],
)
def test_a_question_without_a_condition_population_names_none(question: str) -> None:
    assert stated_population_conditions(question) == ()
    assert _SEPSIS not in _named(question)


def _requirement(rid: str, kind: str, quote: str, concepts: list[str], coverage: str, note=None):
    return QuestionRequirement.model_validate(
        {"id": rid, "kind": kind, "quote": quote, "concepts": concepts,
         "coverage": coverage, "gap": None, "note": note}
    )


def test_a_requirement_list_that_omits_the_named_population_is_refused() -> None:
    (named,) = [
        NamedQuestionConcept.model_validate(item)
        for item in _metadata_only_planning_coordinates(
            question=_LEG3_ZH, database="miiv"
        )["question_named_concepts"]
        if item["concepts"] == _SEPSIS
    ]
    roster = ["lact", "lact_max", "death", "age", "sex", "sep3_sofa1"]
    bound = bind_named_question_concepts(
        [named], roster=roster, relatives=lambda column: frozenset({column})
    )
    assert [item.concepts for item in bound] == [["sep3_sofa1"]]
    exposure = _requirement(
        "r1", "analysis", "乳酸水平或峰值乳酸与院内死亡有什么关联", ["lact_max", "death"], "plan"
    )

    refused = question_requirement_problems(
        [exposure], question=_LEG3_ZH, roster=roster, named=bound
    )
    population = _requirement(
        "r2", "definition", "在脓毒症 ICU 患者中", ["sep3_sofa1"], "definition_only",
        note="Defines the study population: septic ICU patients.",
    )
    accepted = question_requirement_problems(
        [exposure, population], question=_LEG3_ZH, roster=roster, named=bound
    )

    assert [problem.code for problem in refused] == [UNACCOUNTED_REASON]
    assert "脓毒症" in refused[0].message
    assert accepted == ()
