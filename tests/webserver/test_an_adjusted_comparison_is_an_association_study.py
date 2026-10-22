"""A question that asks to adjust for something asks for an association study.

Hosted pilot: "65 岁及以上与 65 岁以下患者的院内死亡风险有差别吗？请调整性别。"
named no relationship, so the planner's routing read it as descriptive -- and
a descriptive plan cannot adjust, so it declared the adjustment it was asked
for undoable.
"""

from __future__ import annotations

import pytest

from easyicu.webserver.study_family_reading import (
    normalize_study_family_reading,
    study_family_reading,
)


@pytest.mark.parametrize(
    ("question", "evidence"),
    [
        ("在 eICU 全部 ICU 入住中，65 岁及以上与 65 岁以下患者的院内死亡风险有差别吗？请调整性别。", "调整性别"),
        ("控制年龄后，两组院内死亡还有差别吗？", "控制年龄后"),
        ("Do older patients die more often in hospital, adjusted for sex?", "adjusted for"),
    ],
)
def test_an_adjusted_comparison_is_read_as_an_association(question: str, evidence: str) -> None:
    reading = study_family_reading(question)

    assert reading is not None and reading.family == "association_study"
    record = reading.record()
    assert record["elements"] == [{"element": "adjustment", "evidence": evidence}]
    assert normalize_study_family_reading(record) == record


@pytest.mark.parametrize(
    "question",
    [
        "65 岁以下和 65 岁及以上患者的院内死亡率",
        "请描述 65 岁以上患者的院内死亡率，不调整任何变量。",
        "无需校正，直接比较两组院内死亡率",
        "Describe hospital mortality by age group without adjusting for sex.",
        # A prediction or causal question also adjusts; its own routing reads it.
        "用年龄和性别预测院内死亡，并调整入院类型",
        "What is the causal effect of early vasopressors on death, adjusted for SOFA?",
    ],
)
def test_no_adjustment_or_another_adjusted_family_is_not_read_here(question: str) -> None:
    reading = study_family_reading(question)

    assert reading is None or all(item.element != "adjustment" for item in reading.elements)
