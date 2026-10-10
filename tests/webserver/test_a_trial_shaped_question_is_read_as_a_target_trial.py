"""A trial-shaped question is read as a target trial, from its own words.

The host reads a question as a target trial's causal question when it states
a numeric time zero, starting a treatment within a grace period against not
starting it (or starting it only after the grace period), and a fixed-horizon
death; or when it names a causal method.  Each element of the reading is a
contiguous piece of the question.  A question that disclaims a causal reading,
two bounded start windows, and an early-against-late comparison without hours
are not read as a trial.  Synthetic questions only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.analysis_types import infer_analysis_type
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext
from easyicu.webserver.causal_trial_reading import (
    CAUSAL_TRIAL_ELEMENTS,
    QUESTION_CAUSAL_METHOD,
    QUESTION_TRIAL_SHAPE,
    causal_trial_reading,
)
from easyicu.webserver.study_intent import deterministic_intent

#: The development probe's question, word for word.
_ALBUMIN = (
    "在 MIMIC-IV 的成人 ICU 住院中，以入 ICU 后第 6 小时为时间零点，纳入当时仍存活、"
    "仍在 ICU 且此前未输注白蛋白的住院。比较在之后 24 小时内开始静脉输注白蛋白，"
    "与这 24 小时内不开始，28 天死亡风险相差多少？"
)


def _evidence(question: str) -> dict[str, str]:
    reading = causal_trial_reading(question)
    assert reading is not None
    return {item.element: item.evidence for item in reading.elements}


def test_the_probe_question_is_read_as_a_trial_in_its_own_words() -> None:
    reading = causal_trial_reading(_ALBUMIN)

    assert reading is not None and reading.source == QUESTION_TRIAL_SHAPE
    assert reading.record() == {
        "source": "question_trial_shape",
        "elements": [
            {"element": "time_zero", "evidence": "入 ICU 后第 6 小时"},
            {"element": "initiate_strategy", "evidence": "24 小时内开始静脉输注白蛋白"},
            {"element": "defer_strategy", "evidence": "这 24 小时内不开始"},
            {"element": "fixed_horizon_outcome", "evidence": "28 天死亡"},
        ],
    }
    assert (reading.time_zero_hours, reading.grace_period_hours) == (6.0, 24.0)
    start, end = reading.treatment
    assert _ALBUMIN[start:end] == "静脉输注白蛋白"
    # "此前未输注白蛋白" is who is eligible, not the comparison strategy.
    assert "此前未输注" not in str(reading.record())


@pytest.mark.parametrize(
    ("question", "treatment", "hours"),
    [
        (
            "以入 ICU 后第 12 小时为时间零点，比较 6 小时内开始去甲肾上腺素与这 6 小时内不开始的 90 天死亡。",
            "去甲肾上腺素", (12.0, 6.0),
        ),
        (
            "时间零点为入 ICU 后第 4 小时；比较 12 小时内开始血管活性药和不开始，1 年死亡率有何差别？",
            "血管活性药", (4.0, 12.0),
        ),
        (
            "With time zero at hour 6 after ICU admission, compare starting norepinephrine "
            "within 24 hours versus not starting it, on 28-day mortality.",
            "norepinephrine", (6.0, 24.0),
        ),
        (
            "Hour 4 after admission as time zero: dobutamine started within 12 hours "
            "compared with deferring it, and the 90-day mortality risk difference.",
            "dobutamine", (4.0, 12.0),
        ),
    ],
)
def test_a_trial_is_read_whatever_its_treatment_times_and_horizon(
    question: str, treatment: str, hours: tuple[float, float]
) -> None:
    reading = causal_trial_reading(question)

    assert reading is not None and reading.source == QUESTION_TRIAL_SHAPE
    assert [item.element for item in reading.elements] == [
        "time_zero", "initiate_strategy", "defer_strategy", "fixed_horizon_outcome",
    ]
    assert (reading.time_zero_hours, reading.grace_period_hours) == hours
    start, end = reading.treatment
    assert question[start:end] == treatment
    for item in reading.elements:
        assert question[item.start : item.end] == item.evidence


def test_starting_only_after_the_grace_period_is_the_defer_strategy() -> None:
    question = "以入 ICU 后第 6 小时为时间零点，比较 24 小时内开始白蛋白与 24 小时后才开始的 28 天死亡。"

    assert _evidence(question)["defer_strategy"] == "24 小时后才开始"


@pytest.mark.parametrize(
    "question",
    [
        # Association, description, a landmark association, a prediction.
        "前 24 小时最高乳酸与 28 天死亡的关联",
        "脓毒症与非脓毒症患者的院内死亡比较，报告风险差",
        "第 24 小时仍在 ICU 的患者中，前 24 小时机械通气与 28 天死亡",
        "用前 24 小时数据预测院内死亡",
        # Only some of the elements.
        "以入 ICU 后第 6 小时为时间零点，描述 28 天死亡。",
        "比较 24 小时内开始白蛋白与不开始的 28 天死亡差异",
        "以入 ICU 后第 6 小时为时间零点，比较 24 小时内开始白蛋白与不开始的院内死亡",
        # A disclaimer wins over the shape.
        "以入 ICU 后第 6 小时为时间零点，比较 24 小时内开始白蛋白与不开始的 28 天死亡，只做关联，不做因果。",
        # Two bounded start windows; early against late without hours.
        "以入 ICU 后第 6 小时为时间零点，比较 6 小时内开始去甲肾上腺素与 6 到 24 小时之间开始的 28 天死亡。",
        # A later start that is deferred but bounded is a second window too.
        "With time zero at hour 6 after ICU admission, compare starting norepinephrine within "
        "6 hours versus starting only after 6 hours, between 6 and 24 hours, on 28-day mortality.",
        "早开始与晚开始去甲肾上腺素对 28 天死亡的影响",
    ],
)
def test_a_question_without_the_trial_shape_is_not_read_as_one(question: str) -> None:
    assert causal_trial_reading(question) is None


def test_two_start_windows_named_as_a_target_trial_are_read_by_the_method() -> None:
    # The trial compile then stops the start rule it cannot type.
    question = (
        "按目标试验模拟：以入 ICU 后第 6 小时为时间零点，比较 6 小时内开始去甲肾上腺素与"
        "6 到 24 小时之间开始的 28 天死亡。"
    )

    reading = causal_trial_reading(question)

    assert reading is not None and reading.source == QUESTION_CAUSAL_METHOD
    assert _evidence(question)["causal_method"] == "目标试验模拟"
    assert "initiate_strategy" not in _evidence(question)


@pytest.mark.parametrize(
    "question",
    [
        "用倾向评分估计早期使用激素对 28 天死亡的影响",
        "Estimate the treatment effect of early vasopressors on hospital mortality.",
        "Use inverse probability weighting to compare early and late dialysis.",
        "Emulate a target trial of early antibiotics.",
    ],
)
def test_a_named_causal_method_is_read_as_the_planner_reads_it(question: str) -> None:
    reading = causal_trial_reading(question)
    context = ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="study", database="miiv", n_stays=0),
        variables=[],
    )

    assert reading is not None and reading.source == QUESTION_CAUSAL_METHOD
    assert infer_analysis_type(context).key == "causal_inference"
    assert set(_evidence(question)) <= set(CAUSAL_TRIAL_ELEMENTS)


def test_a_time_zero_is_no_feature_window() -> None:
    # The hours of the time zero no longer read as the study's window.
    window = deterministic_intent(_ALBUMIN)["slots"]["time_window_hours"]

    assert window["value"] != 6
