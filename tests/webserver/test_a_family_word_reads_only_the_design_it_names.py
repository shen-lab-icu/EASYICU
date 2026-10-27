"""A family word reads the design it names, and no other.

风险比 is a risk ratio, the effect measure of an association study, unless it
stands beside HR ("风险比（HR）"): the keyword reader read every 风险比 as a
survival study, so a binary-outcome association question asking for a risk
ratio was routed to survival.  A competing event, a competing risk, a
Fine-Gray model or a cumulative incidence exists only over follow-up time, so a
question naming one asks a time-to-event analysis: 「…把活着转出 ICU 作为竞争
事件，调整年龄和性别。」 was read by the host as an association (its adjustment),
and the conversation had to ask which family it meant (pilot1 10-26).  "风险"
or "risk" alone names neither, nor does a bare cumulative incidence (a
proportion by a time: 「7 天内 AKI 的累积发生率」) or a competing event the
question declines to model ("not accounting for competing events"; lead
review 10-27: both were routed to survival).  Synthetic questions only.
"""

from __future__ import annotations

import pytest

from easyicu.webserver.question_family_check import check_question_family, host_question_family
from easyicu.webserver.study_intent import deterministic_intent


def _family(question: str):
    return deterministic_intent(question)["slots"]["analysis_family"]["value"]


@pytest.mark.parametrize(
    "question",
    [
        "在 eICU demo 中，入 ICU 第一天乳酸与院内死亡的关联，调整年龄、性别和疾病严重程度后，用风险比（RR）表示。",
        "乳酸与院内死亡的关联，报告风险比",
        "Is lactate associated with in-hospital death? Report a risk ratio.",
    ],
)
def test_a_risk_ratio_is_no_survival_word(question: str) -> None:
    assert _family(question) == "association"


@pytest.mark.parametrize(
    "question",
    [
        "eICU 中，乳酸分组的死亡风险比（HR）是多少？",
        "乳酸分组的 HR（风险比）",
        "What is the hazard ratio of death by lactate group?",
    ],
)
def test_a_hazard_ratio_is_a_survival_word(question: str) -> None:
    assert _family(question) == "survival"


COMPETING = "在 eICU demo 中，入 ICU 第一天的最高肌酐与 ICU 内死亡有什么关系？把活着转出 ICU 作为竞争事件，调整年龄和性别。"


@pytest.mark.parametrize(
    "question",
    [
        COMPETING,
        "乳酸与 ICU 死亡的关系，以出院为竞争风险",
        "Is lactate associated with ICU death, with discharge alive as a competing event?",
        "Lactate and ICU death under competing risks, adjusted for age",
        "用 Fine-Gray 模型分析乳酸与 ICU 死亡",
        "Report the cumulative incidence of ICU death by lactate group, with discharge as a competing event.",
        "按乳酸分组报告 ICU 死亡的累积发生率，出院为竞争事件",
        # A declined competing event leaves the design the question names.
        "Cox regression of lactate and ICU death, not accounting for competing events.",
    ],
)
def test_a_competing_event_asks_a_time_to_event_analysis(question: str) -> None:
    assert _family(question) == "survival"
    assert host_question_family(question) == "survival"


def test_the_conversation_and_the_host_agree_on_the_competing_event_question() -> None:
    # The pilot1 record: the conversation read survival, the host its adjustment.
    record = check_question_family(
        {"family": "survival", "words": "与 ICU 内死亡有什么关系？把活着转出 ICU 作为竞争事件"},
        COMPETING,
    )
    assert (record["host_family"], record["status"]) == ("survival", "agreed")


@pytest.mark.parametrize(
    "question",
    [
        "肌酐与死亡风险的关系，调整年龄和性别",
        "Is creatinine associated with the risk of ICU death?",
        "乳酸与死亡风险比对照组高吗？",
        # A bare cumulative incidence is a proportion by a time.
        "What is the cumulative incidence of AKI in the first 7 ICU days?",
        "入 ICU 7 天内 AKI 的累积发生率是多少？",
        "Develop a model to predict AKI; report the cumulative incidence by risk decile.",
        # A competing event the question declines to model.
        "Is lactate associated with ICU death, not accounting for competing events?",
        "乳酸与 ICU 死亡的关系，不考虑竞争风险",
    ],
)
def test_a_risk_alone_names_no_time_to_event_analysis(question: str) -> None:
    assert _family(question) != "survival"
    assert host_question_family(question) != "survival"
