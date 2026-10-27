"""ICU death is its own endpoint; hospital death never answers for it (pilot1 10-27).

The eICU demo questions 「各型 ICU 死亡率有何不同」 and 「预测之后 7 天内 ICU 死亡」 were
planned on hospital death, and 「预测 ICU 内死亡的模型」 was attested as answered
by a hospital-death model: the context builder read 「ICU 内死亡」 as an
unspecified mortality, the concept catalog gave hospital death every
ICU-mortality phrase, and the planning catalog did not list the native ICU
death endpoint the capability profile models.  Each owner now keeps the two
endpoints apart: a question naming ICU death bound to hospital death records
an endpoint-definition conflict for review, never a silent substitution.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.concept_availability import normalize_concept_name
from easyicu.research_agent.concept_catalog import load_concept_catalog
from easyicu.research_agent.research_context.builder import (
    _enrich_target_outcome_descriptor,
    _infer_outcome_semantics,
)
from easyicu.research_agent.schema import ConceptDescriptor

_PHENOTYPES = "在 eICU demo 中，入 ICU 首日的生命体征和化验能分出哪几种表型？各型 ICU 死亡率有何不同？"
_LANDMARKS = (
    "在 eICU demo 中，在入 ICU 第 24、48、72 小时，用到当时为止的生命体征和化验，"
    "预测之后 7 天内 ICU 死亡的效果如何？"
)
_STATIC = "用 eICU demo 中入 ICU 前 24 小时的数据建立一个预测 ICU 内死亡的模型，它的区分度和校准怎么样？"


@pytest.mark.parametrize(
    "question",
    [
        _PHENOTYPES,
        _STATIC,
        "预测ICU内死亡",
        "ICU 病死率在两组间有差别吗？",
        "死于 ICU 的比例",
        "Predict death in the ICU from first-day vital signs.",
        "Compare ICU deaths between the groups.",
    ],
)
def test_a_question_naming_icu_death_requests_icu_mortality(question: str) -> None:
    semantics = _infer_outcome_semantics(research_question=question, outcome_name="death")
    assert semantics["source_concept"] == "icu_mortality"


@pytest.mark.parametrize(
    ("question", "semantic"),
    [
        ("用入 ICU 前 24 小时的数据建立一个预测院内死亡的模型", "hospital_mortality"),
        ("用入 ICU 前 24 小时的数据预测院内死亡，不要用 ICU 死亡。", "hospital_mortality"),
        ("Use hospital mortality; do not substitute ICU mortality.", "hospital_mortality"),
        ("Is lactate associated with death?", "mortality_unspecified"),
    ],
)
def test_hospital_and_unspecified_death_keep_their_reading(question: str, semantic: str) -> None:
    assert _infer_outcome_semantics(
        research_question=question, outcome_name="death",
    )["source_concept"] == semantic


def _descriptor(name: str, description: str) -> ConceptDescriptor:
    return ConceptDescriptor(name=name, description=description, dtype="float64", source_concept=name)


def _caveats(question: str, descriptor: ConceptDescriptor) -> list[str]:
    _enrich_target_outcome_descriptor(
        descriptors=[descriptor], research_question=question, target_outcome=descriptor.name,
    )
    return descriptor.clinical_caveats


@pytest.mark.parametrize(
    "question", [_PHENOTYPES, _STATIC, _LANDMARKS, "Is lactate associated with ICU mortality in Sepsis-3?"],
)
def test_icu_death_bound_to_hospital_death_is_a_recorded_conflict(question: str) -> None:
    caveats = _caveats(question, _descriptor("death", "in hospital mortality"))
    assert any("Endpoint-definition conflict" in item for item in caveats)


@pytest.mark.parametrize("question", [_PHENOTYPES, _STATIC, _LANDMARKS])
def test_icu_death_bound_to_the_native_endpoint_agrees(question: str) -> None:
    caveats = _caveats(question, _descriptor("icu_death", "ICU death"))
    assert not any("Endpoint-definition conflict" in item for item in caveats)


@pytest.mark.parametrize(
    "question",
    ["Is the first 24-hour peak lactate associated with hospital death?", "乳酸与院内死亡的关系"],
)
def test_hospital_death_bound_to_icu_death_is_a_recorded_conflict(question: str) -> None:
    # The question's setting decides; the bound column's name only when the
    # question states none.
    assert _infer_outcome_semantics(
        research_question=question, outcome_name="icu_death",
    )["source_concept"] == "hospital_mortality"
    assert any(
        "Endpoint-definition conflict" in item
        for item in _caveats(question, _descriptor("icu_death", "ICU death"))
    )
    unstated = "Is lactate associated with death?"
    assert not any(
        "Endpoint-definition conflict" in item
        for item in _caveats(unstated, _descriptor("icu_death", "ICU death"))
    )


@pytest.mark.parametrize(
    "question",
    [
        "Develop an ICU mortality model and validate it against in-hospital death.",
        "Not only ICU mortality but also hospital mortality.",
        "Is lactate associated with ICU and in-hospital mortality?",
        "乳酸与 ICU 及院内死亡的关系",
        "Is lactate associated with ICU- and in-hospital mortality?",
        "Is lactate associated with mortality in the ICU and in hospital?",
    ],
)
def test_a_question_naming_both_settings_lets_neither_win(question: str) -> None:
    assert _infer_outcome_semantics(
        research_question=question, outcome_name="icu_death",
    )["source_concept"] == "mortality_setting_conflict"
    for descriptor in (_descriptor("icu_death", "ICU death"), _descriptor("death", "in hospital mortality")):
        assert any("Endpoint-definition conflict" in item for item in _caveats(question, descriptor))


def test_a_population_of_icu_patients_keeps_its_horizon_endpoint() -> None:
    question = "Is lactate associated with 28-day mortality in intensive care unit patients?"
    assert _infer_outcome_semantics(
        research_question=question, outcome_name="mort_28d",
    )["source_concept"] == "mortality_28d"


def test_hospital_death_bound_to_hospital_death_agrees() -> None:
    caveats = _caveats("预测院内死亡的模型", _descriptor("death", "in hospital mortality"))
    assert not any("Endpoint-definition conflict" in item for item in caveats)


def test_a_landmark_horizon_is_the_predictions_own_window() -> None:
    # 「之后 7 天内」 follows each prediction time: the endpoint is ICU death,
    # not a 7-day mortality from ICU admission.  One prediction time states
    # a static horizon, which stays the request's.
    landmark = _infer_outcome_semantics(research_question=_LANDMARKS, outcome_name="icu_death")
    assert landmark["source_concept"] == "icu_mortality"
    # The window stays part of the request.
    assert [horizon.noun for horizon in landmark["prediction_window"]] == ["7 days"]
    assert "within 7 days after each prediction time" in landmark["label"]
    caveats = _caveats(_LANDMARKS, _descriptor("icu_death", "ICU death"))
    assert not any("Endpoint-definition conflict" in item for item in caveats)
    assert any(item.startswith("Prediction window:") and "7 days" in item for item in caveats)
    unset = _infer_outcome_semantics(
        research_question=_LANDMARKS.replace("ICU 死亡", "死亡"), outcome_name="death",
    )
    assert unset["source_concept"] == "mortality_unspecified"
    assert unset["label"] == "mortality without a specified setting within 7 days after each prediction time"
    single = "在入 ICU 第 24 小时，用到当时为止的数据预测之后 7 天内 ICU 死亡"
    assert _infer_outcome_semantics(
        research_question=single, outcome_name="icu_death",
    )["source_concept"] == "icu_mortality_7d"


def test_the_catalog_lists_native_icu_death_with_its_own_phrases() -> None:
    catalog = load_concept_catalog()
    assert "icu_death" in catalog.available_concepts
    assert catalog.outcome_determinability["icu_death"]["status"] == "known_0_1"
    icu = {alias.lower() for alias in catalog.concept_aliases["icu_death"]}
    hospital = {alias.lower() for alias in catalog.concept_aliases["death"]}
    assert {"icu mortality", "icu death", "death in the icu"} <= icu
    assert not icu & hospital


def test_the_planning_menu_of_a_source_with_icu_death_offers_it() -> None:
    from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog

    concepts = {item.concept_id for item in build_database_capability_catalog("eicu_demo").concepts}
    assert {"death", "icu_death"} <= concepts


def test_icu_mortality_names_the_native_endpoint() -> None:
    assert normalize_concept_name("icu_mortality") == "icu_death"
    assert normalize_concept_name("hospital_mortality") == "death"
    assert normalize_concept_name("mortality") == "death"


def _idea(outcome: str):
    from easyicu.research_agent.discovery.idea_mining import LiteratureIdeaCandidate

    return LiteratureIdeaCandidate(
        source_snapshot_id="snap1",
        citation_key="cite1",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients",
        exposure_or_predictor="early norepinephrine",
        outcome=outcome,
        exposure_core_concept="norepinephrine",
        outcome_core_concept=outcome,
        rationale="the source describes this direction",
        source_quote="more data are needed",
    )


@pytest.mark.parametrize(
    ("available", "resolved"),
    [(["norepi_rate", "death", "icu_death"], "icu_death"), (["norepi_rate", "death"], None)],
)
def test_an_icu_mortality_idea_never_resolves_to_hospital_death(available, resolved) -> None:
    from easyicu.research_agent.discovery.idea_mining import (
        map_literature_idea_to_executable_candidate,
    )

    catalog = load_concept_catalog()
    candidate = map_literature_idea_to_executable_candidate(
        _idea("ICU mortality"),
        available_concepts=available,
        concept_aliases=catalog.concept_aliases,
        outcome_determinability=catalog.outcome_determinability,
    )
    assert candidate.resolved_outcome_concept == resolved
    hospital = map_literature_idea_to_executable_candidate(
        _idea("in-hospital mortality"),
        available_concepts=available,
        concept_aliases=catalog.concept_aliases,
        outcome_determinability=catalog.outcome_determinability,
    )
    assert hospital.resolved_outcome_concept == "death"


def test_a_source_without_native_icu_death_offers_a_reviewed_substitution() -> None:
    from easyicu.databases.research_capabilities import research_capability_profile
    from easyicu.databases.research_requirements import StudyRequirements, assess_study

    profile = research_capability_profile("eicu")
    requirements = StudyRequirements(family="prediction_model", outcome="icu_death", concepts=("age",))
    assert assess_study(profile, requirements).status == "exact"
    without = profile.model_copy(update={"mortality_outcomes": (profile.mortality,)})
    decision = assess_study(without, requirements)
    assert decision.status == "substitution"
    (change,) = [item for item in decision.substitutions if item.capability == "outcome_meaning"]
    assert (change.original, change.substitute) == ("icu_death", "hospital_death")
