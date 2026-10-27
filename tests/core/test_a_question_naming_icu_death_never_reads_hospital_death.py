"""Where a question counts death is read in one vocabulary (pilot1 10-27).

「各型 ICU 死亡率」, 「预测之后 7 天内 ICU 死亡」 and 「预测 ICU 内死亡」 were planned on
hospital death; the third was attested as answered.  ICU mortality (death
before ICU discharge) and hospital mortality are different endpoints, so the
shared reader names which one a text asks for, in either language, and says
nothing when it names neither.
"""

from __future__ import annotations

import pytest

from easyicu.outcome_availability import (
    HOSPITAL_DEATH,
    ICU_DEATH,
    stated_mortality_setting_mentions,
    stated_mortality_settings,
)


@pytest.mark.parametrize(
    "text",
    [
        "各型 ICU 死亡率有何不同？",
        "预测之后 7 天内 ICU 死亡的效果如何？",
        "建立一个预测 ICU 内死亡的模型",
        "建立一个预测ICU内死亡的模型",
        "ICU 病死率是多少？",
        "ICU 住院期间死亡的比例",
        "死于 ICU 的患者",
        "出 ICU 前死亡",
        "重症监护病房死亡",
        "Does ICU mortality differ between the phenotypes?",
        "Predict death in the ICU from first-day vitals.",
        "How many patients died before ICU discharge?",
        "intensive care unit mortality",
        "ICU deaths by unit type",
        # A unit's own ICU, a spelled-out kind, other verbs and stays.
        "Patients who die in the ICU",
        "MICU mortality",
        "mortality in the medical ICU",
        "death in the ICU among septic patients",
        "ICU 停留期间死亡",
        "ICU 住院死亡率",
    ],
)
def test_icu_death_is_read_in_either_language(text: str) -> None:
    assert stated_mortality_settings(text) == (ICU_DEATH,)


@pytest.mark.parametrize(
    "text",
    [
        "院内死亡率是多少？",
        "住院期间死亡",
        "出院前死亡",
        "Is lactate associated with in-hospital mortality?",
        "death in hospital",
        "death before hospital discharge",
    ],
)
def test_hospital_death_stays_hospital_death(text: str) -> None:
    assert stated_mortality_settings(text) == (HOSPITAL_DEATH,)


@pytest.mark.parametrize(
    "text",
    [
        # A population, not where death is counted.
        "28-day mortality in ICU patients with sepsis",
        "Is lactate associated with mortality in intensive care unit patients?",
        "28-day mortality in intensive care unit patients",
        "mortality in ICU-admitted patients",
        "mortality in ICU septic patients",
        # An instrument named for ICU mortality.
        "an ICU mortality score",
        "the ICU mortality risk score",
        "ICU 死亡率评分",
        # Deaths after the ICU.
        "出 ICU 后 48 小时内死亡",
        "post-ICU mortality",
        # A time on the ICU clock.
        "入 ICU 后 24 小时内死亡",
        "Is SOFA associated with death?",
        "ICU length of stay",
    ],
)
def test_a_text_naming_no_setting_reads_none(text: str) -> None:
    assert stated_mortality_settings(text) == ()


@pytest.mark.parametrize(
    ("text", "settings"),
    [
        ("用入 ICU 前 24 小时的数据预测院内死亡，不要用 ICU 死亡。", (HOSPITAL_DEATH,)),
        ("Use hospital mortality; do not substitute ICU mortality.", (HOSPITAL_DEATH,)),
        ("Report hospital mortality rather than ICU mortality.", (HOSPITAL_DEATH,)),
        ("报告院内死亡，而不是ICU死亡。", (HOSPITAL_DEATH,)),
        # A later affirmative mention keeps its meaning.
        ("不能用 ICU 死亡替代院内死亡，但还要单独报告 ICU 死亡。", (HOSPITAL_DEATH, ICU_DEATH)),
        ("Not only ICU mortality but also hospital mortality.", (ICU_DEATH, HOSPITAL_DEATH)),
        # An instrument is not the endpoint the question predicts.
        (
            "Compare SOFA and APACHE II, an ICU mortality score, for predicting in-hospital death.",
            (HOSPITAL_DEATH,),
        ),
        # Only a cue right before the mention declines it.
        ("Do not exclude patients with ICU mortality.", (ICU_DEATH,)),
    ],
)
def test_only_a_directly_declined_setting_is_not_requested(text: str, settings) -> None:
    assert stated_mortality_settings(text) == settings


@pytest.mark.parametrize(
    ("text", "settings"),
    [
        # Two settings sharing one death noun name both.
        ("Is lactate associated with ICU and in-hospital mortality?", (ICU_DEATH, HOSPITAL_DEATH)),
        ("Compare ICU or hospital death between the groups.", (ICU_DEATH, HOSPITAL_DEATH)),
        ("intensive care unit and hospital mortality", (ICU_DEATH, HOSPITAL_DEATH)),
        ("Is lactate associated with hospital and ICU mortality?", (HOSPITAL_DEATH, ICU_DEATH)),
        ("乳酸与 ICU 及院内死亡的关系", (ICU_DEATH, HOSPITAL_DEATH)),
        ("ICU 和住院期间死亡", (ICU_DEATH, HOSPITAL_DEATH)),
        ("院内及 ICU 死亡", (HOSPITAL_DEATH, ICU_DEATH)),
        # A suspended hyphen.
        ("Is lactate associated with ICU- and in-hospital mortality?", (ICU_DEATH, HOSPITAL_DEATH)),
        ("ICU- and hospital mortality", (ICU_DEATH, HOSPITAL_DEATH)),
        ("hospital- and ICU mortality", (HOSPITAL_DEATH, ICU_DEATH)),
        # The death noun first, then both places.
        ("Is lactate associated with mortality in the ICU and in hospital?", (ICU_DEATH, HOSPITAL_DEATH)),
        ("death in the ICU or in hospital", (ICU_DEATH, HOSPITAL_DEATH)),
        ("death in hospital or in the ICU", (HOSPITAL_DEATH, ICU_DEATH)),
        ("在 ICU 和医院内死亡", (ICU_DEATH, HOSPITAL_DEATH)),
        ("ICU 或医院内死亡", (ICU_DEATH, HOSPITAL_DEATH)),
        ("在医院和 ICU 内死亡", (HOSPITAL_DEATH, ICU_DEATH)),
        # 「ICU 内/中」 before the join.
        ("ICU 内和院内死亡", (ICU_DEATH, HOSPITAL_DEATH)),
        ("在 ICU 中或医院内死亡", (ICU_DEATH, HOSPITAL_DEATH)),
        ("ICU 内或住院期间死亡", (ICU_DEATH, HOSPITAL_DEATH)),
        ("在 ICU 内和医院内死亡", (ICU_DEATH, HOSPITAL_DEATH)),
        # An ICU population, or a place someone is moved to, is not a setting.
        ("ICU patients and in-hospital mortality", (HOSPITAL_DEATH,)),
        ("Patients admitted to the ICU and in-hospital mortality", (HOSPITAL_DEATH,)),
        ("ICU length of stay and in-hospital mortality", (HOSPITAL_DEATH,)),
        ("ICU 患者及院内死亡", (HOSPITAL_DEATH,)),
        ("moved to an ICU and in-hospital mortality", (HOSPITAL_DEATH,)),
        # A second place that starts another noun phrase, or an instrument.
        ("mortality in the ICU and hospital length of stay", (ICU_DEATH,)),
        ("在 ICU 内死亡和住院时长", (ICU_DEATH,)),
        ("an ICU and hospital mortality score", ()),
    ],
)
def test_settings_sharing_one_noun_are_both_read(text: str, settings) -> None:
    assert stated_mortality_settings(text) == settings


def test_a_hospital_phrase_inside_an_icu_one_is_the_icus() -> None:
    (mention,) = stated_mortality_setting_mentions("统计 ICU 住院期间死亡")
    assert mention.endpoint == ICU_DEATH
    assert "统计 ICU 住院期间死亡"[mention.start:mention.end] == "ICU 住院期间死亡"
