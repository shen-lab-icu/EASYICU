"""A concept the question adjusts for is not the factor it studies.

Hosted pilot: "65 岁及以上与 65 岁以下患者的院内死亡风险有差别吗？请调整性别。"
was read with sex as the exposure, and the run modelled sex with no
adjustment -- its odds ratio was shown as the study's main estimate.
"""

from __future__ import annotations

import pytest

from easyicu.webserver.study_intent import deterministic_intent


def _exposure(question: str):
    slot = deterministic_intent(question)["slots"]["exposure"]
    return slot.get("value")


@pytest.mark.parametrize(
    "question",
    [
        "在 eICU 全部 ICU 入住中，65 岁及以上与 65 岁以下患者的院内死亡风险有差别吗？请调整性别。",
        "65 岁及以上患者的院内死亡更高吗？校正性别后。",
        "Is being 65 or older associated with hospital mortality, adjusted for sex?",
        "Is being 65 or older associated with hospital mortality after controlling for sex?",
    ],
)
def test_an_adjusted_for_concept_is_left_for_the_plan(question: str) -> None:
    assert _exposure(question) is None


def test_the_studied_factor_is_still_read_beside_its_adjustments() -> None:
    assert _exposure(
        "入 ICU 后 24 小时内的最高乳酸与院内死亡有关联吗？请调整年龄和性别。"
    ) == "lact"
    # The adjustment clause ends at its comma; the factor after it is studied.
    assert _exposure("在调整性别后，年龄各组的院内死亡差别还存在吗？") == "age"
    assert _exposure("性别与院内死亡有关联吗？") == "sex"
