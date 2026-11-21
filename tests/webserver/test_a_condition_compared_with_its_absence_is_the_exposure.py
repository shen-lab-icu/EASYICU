"""A condition the question compares with its absence is the exposure, not the population.

「入 ICU 第一天接受机械通气的患者 … 与未通气的患者相比」 studies ventilation:
both groups are in the study.  Since B31 the population reader took the first
clause as a restriction, so the exposure lost vent_ind; with the S1757 planning
constraint the host would even have selected on it, leaving no comparison
group.  A condition whose absence the question names as a group (「未通气」
「非脓毒症」「无 AKI」, "non-ventilated", "without sepsis") is no population
restriction.  Fixtures are generic across the three recorded conditions.
"""

from __future__ import annotations

import pytest

from easyicu.webserver.research_launch_population import stated_population_constraints
from easyicu.webserver.study_intent import stated_population_conditions


@pytest.mark.parametrize("question", [
    "入 ICU 第一天接受机械通气的患者，28 天死亡率与未通气的患者相比有何不同？",
    "脓毒症患者与非脓毒症患者的院内死亡率比较。",
    "AKI 患者与无 AKI 患者相比，住院时间是否更长？",
    "Among septic patients compared with patients without sepsis, is mortality higher?",
    "Patients on mechanical ventilation versus patients not on mechanical ventilation: 28-day death.",
])
def test_a_condition_compared_with_its_absence_restricts_no_one(question: str) -> None:
    assert stated_population_conditions(question) == ()
    assert stated_population_constraints({"question": question}) == []


@pytest.mark.parametrize(("question", "words"), [
    ("脓毒症 ICU 患者中，ICU 入科后首 24 小时峰值乳酸与院内死亡的关联。", "脓毒症"),
    ("机械通气患者中，乳酸与死亡的关联。", "机械通气"),
    ("Among patients with AKI, is the first lactate associated with death?", "AKI"),
    # Another treatment's absence is not the condition's absence.
    ("在 AKI 患者中，比较无肾替代治疗者的死亡。", "AKI"),
])
def test_a_condition_that_only_qualifies_the_patients_still_restricts_them(question: str, words: str) -> None:
    assert [found for _concepts, found in stated_population_conditions(question)] == [words]
