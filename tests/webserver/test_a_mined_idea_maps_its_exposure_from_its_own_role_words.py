"""A mined idea names its exposure only where the text names an exposure role.

UX mining walk (2026-11-19, S1444): the agent's request read 「…是否能支持动态
乳酸暴露、院内死亡或其他脓毒症结局研究」. The Chinese exposure clause took any
bare 「暴露」 as "an exposure is named here" and the text after it as the
exposure; Chinese puts the modifier before the keyword, so the clause was
「、院内死亡或其他脓毒症结局研究」 and Sepsis-3 became the exposure, lactate a
covariate. A keyword now names a clause only with a prefix or a copula, or as
the object of 以…为 / …作为; the fallback never picks the studied population or
a named adjustment covariate. Synthetic phrasings; no benchmark item.
"""

from __future__ import annotations

import pytest

from easyicu.webserver.ideas import mining


def _idea(text: str) -> dict:
    hits = mining._match_concepts(text)
    return mining._idea_from_source({"title": ""}, text, hits, {})


def _roles(idea: dict) -> dict[str, str]:
    return {row["concept_id"]: row["role"] for row in idea["mapped_concepts"]}


WALK = (
    "乳酸复测或早期乳酸变化与脓毒症结局：评估官方 MIMIC-IV Clinical Database Demo v2.2 "
    "是否能支持动态乳酸暴露、院内死亡或其他脓毒症结局研究。"
)


@pytest.mark.parametrize("text", [
    WALK,
    "Whether the data support dynamic lactate exposure, in-hospital mortality or other sepsis outcomes",
    # Carbon mine1444: 「暴露是否」, 「分为暴露组」 and a bare "in/with X" name no role.
    "探讨动态乳酸暴露是否增加脓毒症患者院内死亡",
    "按乳酸水平将脓毒症患者分为暴露组和对照组，结局为院内死亡",
    "Whether the data support early change in lactate, in-hospital mortality or other sepsis outcomes",
    "Sepsis mortality associated with lactate",
])
def test_a_bare_exposure_word_after_its_modifier_names_no_exposure(text: str) -> None:
    idea = _idea(text)
    roles = _roles(idea)
    assert roles.get("sep3_sofa1") not in {"exposure", "predictor"}
    assert roles["lact"] in {"predictor", "covariate_or_subgroup"}
    assert idea["exposure_or_predictor"] == "Lactate"


@pytest.mark.parametrize("text", [
    "以乳酸为暴露，研究院内死亡",
    "乳酸作为暴露因素，结局为院内死亡",
    "暴露因素是乳酸，结局为院内死亡",
    "主要暴露：乳酸；结局：院内死亡",
    "暴露为乳酸，结局为院内死亡",
    "脓毒症患者中以乳酸作为暴露，结局为院内死亡",
    "Primary exposure: lactate; outcome: in-hospital mortality",
    "The main exposure is lactate in septic patients; outcome in-hospital mortality",
    "Using lactate as the primary exposure in sepsis patients, outcome in-hospital mortality",
    "以乳酸作为主要暴露探讨脓毒症患者院内死亡",
    "暴露因素是乳酸对脓毒症患者院内死亡的影响",
])
def test_a_named_exposure_role_maps_that_concept(text: str) -> None:
    idea = _idea(text)
    roles = _roles(idea)
    assert roles["lact"] == "exposure"
    assert roles.get("sep3_sofa1") not in {"exposure", "predictor"}
    assert idea["exposure_or_predictor"] == "Lactate"


@pytest.mark.parametrize(("text", "predictor"), [
    ("脓毒症患者中乳酸与死亡", "lact"),
    ("脓毒症患者中尿量与死亡", "urine"),
    ("Among patients with sepsis, urine output and in-hospital mortality", "urine"),
    ("Sepsis patients: urine output and in-hospital mortality", "urine"),
])
def test_the_studied_population_is_not_the_predictor(text: str, predictor: str) -> None:
    roles = _roles(_idea(text))
    assert roles[predictor] == "predictor"
    assert roles.get("sep3_sofa1") not in {"exposure", "predictor"}


def test_an_adjustment_list_ends_at_its_clause_and_never_becomes_the_predictor() -> None:
    text = "调整年龄、性别和SOFA评分，乳酸与院内死亡"
    hits = mining._match_concepts(text)
    assert set(mining._requested_adjustment_concepts(text, hits)) == {"age", "sex", "sofa"}
    assert _idea(text)["exposure_or_predictor"] == "Lactate"
    assert set(mining._requested_adjustment_concepts("按年龄和性别调整后，乳酸与院内死亡", hits)) == {"age", "sex"}


@pytest.mark.parametrize("text", ["早期乳酸控制、脓毒症结局", "控制组为脓毒症患者，乳酸与院内死亡"])
def test_a_control_noun_phrase_is_not_an_adjustment_clause(text: str) -> None:
    hits = mining._match_concepts(text)
    assert mining._requested_adjustment_concepts(text, hits) == []


@pytest.mark.parametrize(("text", "exposure"), [
    # Carbon mine1444 finding 3: the concept adjacent to the keyword, not the longest alias.
    ("The main exposure is SOFA in sepsis patients; outcome in-hospital mortality", "sofa"),
    # Finding 5: forms that mapped before this fix still do.
    ("Primary exposure urine output; baseline lactate; outcome in-hospital mortality", "urine"),
    ("Exposure: urine output; baseline lactate; outcome: mortality", "urine"),
])
def test_the_concept_next_to_the_keyword_is_the_exposure(text: str, exposure: str) -> None:
    roles = _roles(_idea(text))
    assert roles[exposure] == "exposure"
    assert roles.get("sep3_sofa1") not in {"exposure", "predictor"}


@pytest.mark.parametrize(("text", "adjusted"), [
    # Finding 4: 调整后 is "after adjustment"; a bare 控制 is a noun.
    ("按年龄和性别调整后乳酸与院内死亡的关系", {"age", "sex"}),
    ("早期乳酸控制对脓毒症结局的影响", set()),
    ("控制变量包括年龄和性别；乳酸与死亡", {"age", "sex"}),
])
def test_an_adjustment_needs_its_own_clause(text: str, adjusted: set[str]) -> None:
    hits = mining._match_concepts(text)
    assert set(mining._requested_adjustment_concepts(text, hits)) == adjusted
