"""A question naming ICU death reads the native ICU death endpoint (pilot1 10-27).

The intent reader mapped 「ICU 死亡」 and "ICU mortality" onto the dictionary's
hospital ``death``, and the plan card said EasyICU had read hospital death from
the question.  The shared setting reader now names ICU death in either
language; hospital death, a declined ICU death and an unspecified death keep
the hospital endpoint.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import study_intent


def _outcome(question: str) -> str | None:
    return study_intent.deterministic_intent(question)["slots"]["outcome"]["value"]


@pytest.mark.parametrize(
    "question",
    [
        "在 eICU demo 中，入 ICU 首日的生命体征和化验能分出哪几种表型？各型 ICU 死亡率有何不同？",
        "在 eICU demo 中，在入 ICU 第 24、48、72 小时，用到当时为止的生命体征和化验，"
        "预测之后 7 天内 ICU 死亡的效果如何？",
        "用 eICU demo 中入 ICU 前 24 小时的数据建立一个预测 ICU 内死亡的模型，它的区分度和校准怎么样？",
        "乳酸与死于 ICU 的关系",
        "Does ICU mortality differ between the phenotypes?",
        "Predict death in the ICU from first-day vitals.",
        "Is total bilirubin associated with ICU mortality?",
        "ICU 住院死亡率与乳酸的关系",
    ],
)
def test_icu_death_reads_the_native_endpoint(question: str) -> None:
    assert _outcome(question) == "icu_death"
    assert study_intent.explicit_outcome_concepts(question) == ("icu_death",)


@pytest.mark.parametrize(
    "question",
    [
        "用入 ICU 前 24 小时的数据预测院内死亡，不要用 ICU 死亡。",
        "Is lactate associated with in-hospital mortality? Do not use ICU mortality.",
        "糖化血红蛋白与院内死亡",
        "Is lactate associated with death?",
        "Is lactate associated with mortality in intensive care unit patients?",
        "Compare SOFA and APACHE II, an ICU mortality score, for predicting in-hospital death.",
    ],
)
def test_hospital_declined_or_unspecified_death_keeps_hospital_death(question: str) -> None:
    assert _outcome(question) == "death"
    assert "icu_death" not in study_intent.explicit_outcome_concepts(question)
