"""Synthetic wording: a hospital stay is not silently renamed an ICU stay."""

import pytest

from easyicu.webserver.study_intent import explicit_outcome_concepts


@pytest.mark.parametrize("question, expected", [
    ("比较甲组和乙组的 ICU 住院时间和住院时间。", ("los_icu", "los_hosp")),
    ("分别描述住院时长与 ICU 住院时长。", ("los_hosp", "los_icu")),
    ("报告住院时间的中位数。", ("los_hosp",)),
    ("报告 ICU 住院时间的中位数。", ("los_icu",)),
    ("报告重症监护室住院时间。", ("los_icu",)),
    ("报告医院住院时长。", ("los_hosp",)),
    ("不研究住院时间，仅描述 ICU 住院时间。", ("los_icu",)),
    ("分别报告住院时间和死亡率。", ("death", "los_hosp")),
    ("别研究住院时间。", ()),
    ("请别分析住院时间。", ()),
    ("不研究住院时间、死亡率。", ()),
    ("Describe hospital length of stay and ICU length of stay.", ("los_hosp", "los_icu")),
    ("限定住院时间至少 5 天的患者，比较死亡率。", ("death",)),
])
def test_endpoint_scope_follows_its_own_words(question, expected):
    assert explicit_outcome_concepts(question) == expected
