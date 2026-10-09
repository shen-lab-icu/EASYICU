"""Whom a study includes is not what it studies.

The intent reader's exposure and outcome become the planning coordinates of a
study that configures none.  It read them from eligibility: an inclusion clause
naming a minimum age and a minimum ICU stay made age the exposure and the ICU
length of stay the outcome, and a question comparing groups on three listed
outcomes had its second outcome read as the exposure.  The system chose a
study the researcher did not ask, and said nothing.

Now an age or a stay length stated with a bound, and every concept inside an
inclusion or exclusion clause, is a restriction on the population, never the
exposure or the outcome; and no concept the question names as an endpoint is
the exposure.  A slot with nothing else to read stays unread for the plan to
propose.  An exposure the question does state is still read.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import research_launch_scientific
from easyicu.webserver.study_intent import (
    deterministic_intent,
    explicit_outcome_concepts,
)


def _read(question: str) -> tuple:
    slots = deterministic_intent(question)["slots"]
    return slots["exposure"]["value"], slots["outcome"]["value"]


@pytest.mark.parametrize(
    "question",
    [
        "年龄 ≥ 18 岁的患者中，比较三组的院内死亡率",
        "年龄≥18岁的成人中，比较三组的院内死亡率",
        "年龄 > 18 岁的患者中，比较三组的院内死亡率",
        "年龄大于等于 18 岁的患者中，比较三组的院内死亡率",
        "年龄至少 18 岁的患者中，比较三组的院内死亡率",
        "年龄 18 岁以上的患者中，比较三组的院内死亡率",
        "年龄超过 65 岁的患者中，比较三组的院内死亡率",
        "年龄在 18 至 80 岁之间的患者中，比较三组的院内死亡率",
        "年龄 18-80 岁的患者中，比较三组的院内死亡率",
        "Among patients with age >= 18 years, compare in-hospital mortality by group",
        "Among patients with age of at least 18 years, compare in-hospital mortality by group",
        "Among patients with age over 65, compare in-hospital mortality by group",
        "Among patients with age 18 or older, compare in-hospital mortality by group",
    ],
)
def test_a_bounded_age_is_a_restriction_not_the_exposure(question):
    assert _read(question) == (None, "death")


@pytest.mark.parametrize(
    "question",
    [
        "ICU 住院时长 ≥ 48 小时的成人患者中，比较三组的乳酸",
        "ICU 住院时长至少 24 小时的成人患者中，比较三组的乳酸",
        "ICU 住院时长超过 24 小时的成人患者中，比较三组的乳酸",
        "住院时间大于 3 天的成人患者中，比较三组的乳酸",
        "ICU 住院时长 24 小时以上的成人患者中，比较三组的乳酸",
        "至少 48 小时的 ICU 住院时长的成人患者中，比较三组的乳酸",
        "Among adults with an ICU length of stay of at least 24 hours, compare lactate",
        "Among adults with ICU length of stay > 48 h, compare lactate by group",
        "Among adults with a length of stay more than 2 days, compare lactate by group",
    ],
)
def test_a_bounded_stay_length_is_a_restriction_not_the_outcome(question):
    result = deterministic_intent(question)
    assert _read(question) == ("lact", None)
    assert "outcome" in result["unread"]
    assert "outcome_type" in result["unread"]
    assert explicit_outcome_concepts(question) == ()


@pytest.mark.parametrize(
    ("question", "exposure", "outcome"),
    [
        (
            "纳入年龄 ≥ 16 岁、住院时间至少 3 天且确诊脓毒症的成人患者，描述各组的院内死亡率。",
            None,
            "death",
        ),
        (
            "纳入接受机械通气的脓毒症成人患者，评估乳酸与院内死亡的关联。",
            "lact",
            "death",
        ),
        (
            "Excluding patients with AKI on admission, is lactate associated with "
            "in-hospital mortality?",
            "lact",
            "death",
        ),
        ("排除 24 小时内死亡的患者后，描述乳酸的分布。", "lact", None),
        (
            "在多因素模型中纳入乳酸、年龄和 SOFA，评估液体正平衡与急性肾损伤的关系。",
            "fluid_balance",
            "aki",
        ),
    ],
)
def test_a_concept_inside_an_eligibility_clause_is_a_restriction(
    question, exposure, outcome
):
    assert _read(question) == (exposure, outcome)


@pytest.mark.parametrize(
    ("question", "outcome", "listed"),
    [
        (
            "比较两组患者的院内死亡率、ICU 住院时长和再入院率",
            "death",
            ("death", "los_icu", "icu_readmission"),
        ),
        (
            "Compare in-hospital mortality, ICU length of stay and ICU readmission "
            "across the three groups",
            "death",
            ("death", "los_icu", "icu_readmission"),
        ),
    ],
)
def test_an_outcome_the_question_lists_is_not_the_exposure(question, outcome, listed):
    result = deterministic_intent(question)
    assert _read(question) == (None, outcome)
    assert "exposure" in result["unread"]
    assert explicit_outcome_concepts(question) == listed


@pytest.mark.parametrize(
    ("question", "exposure", "outcome"),
    [
        ("年龄与院内死亡的关系", "age", "death"),
        ("Is age associated with ICU length of stay?", "age", "los_icu"),
        ("年龄 ≥ 18 岁的患者中，乳酸与院内死亡的关系", "lact", "death"),
        (
            "年龄 18 岁以上的患者，去甲肾上腺素剂量与 28 天死亡的关系",
            "norepi_rate",
            "mort_28d",
        ),
        (
            "Among adults with age >= 18 years, are lactate and age associated with "
            "in-hospital mortality?",
            "lact",
            "death",
        ),
        (
            "ICU 住院时长超过 24 小时的患者，乳酸与 ICU 住院时长的关系",
            "lact",
            "los_icu",
        ),
        ("乳酸与院内死亡、ICU 住院时长和再入院的关系", "lact", "death"),
        (
            "Outcomes include in-hospital mortality and ICU length of stay; is lactate "
            "associated with them?",
            "lact",
            "death",
        ),
    ],
)
def test_the_exposure_and_outcome_the_question_studies_are_still_read(
    question, exposure, outcome
):
    assert _read(question) == (exposure, outcome)


def test_an_endpoint_studied_as_a_factor_is_left_for_the_plan():
    # Which of two endpoints is the factor is not guessed from word order.
    result = deterministic_intent(
        "Is ICU length of stay associated with 1-year mortality?"
    )
    assert _read(result["question"]) == (None, "mort_365d")
    assert "exposure" in result["unread"]


def test_planning_receives_no_restriction_as_its_coordinates():
    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question=(
            "纳入年龄 ≥ 16 岁、ICU 住院时长至少 36 小时的成人患者，比较两组患者的"
            "院内死亡率、ICU 住院时长和再入院率。"
        ),
        database="eicu_demo",
    )
    assert coordinates["primary_exposure"] is None
    assert coordinates["target_outcome"] == "death"
    assert coordinates["execution_authorized"] is False
