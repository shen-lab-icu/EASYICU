"""The first candidate plan reads the exposure the researcher actually named.

Every Web study used to need at least two planning rounds: the first candidate
was reviewed as ``scientific_capability_data_contract_unresolved`` because the
metadata-only planning coordinates had no exposure, and the typed exposure only
reached the StudyContext after that review.  The coordinate owner reads what
the researcher explicitly named; these tests fix two readings that were
missing, and that the reading stays a reading -- never a guess.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import research_launch_scientific, scientific_runtime_projection
from easyicu.webserver.study_intent import (
    deterministic_intent,
    explicit_exposure_aggregation,
)

_ADMISSION_QUESTION = (
    "在已准备好的 eICU Collaborative Research Database Demo v2.0.1 中，评估 ICU "
    "入院类型（内科、外科、其他）与住院死亡之间的关联。入院类型定义为 ICU 入院时"
    "的基线特征，不采用 landmark 设计。"
)
_KDIGO_QUESTION = (
    "在已准备好的 eICU Collaborative Research Database Demo v2.0.1 中，评估入 ICU "
    "24 小时内最高的 KDIGO AKI 分期（0/1/2/3）与住院死亡之间的关联；分期证据不足"
    "的病例单列为 unknown，不并入 0 级。"
)


@pytest.mark.parametrize(
    "question",
    [
        _ADMISSION_QUESTION,
        "Association of ICU admission type with in-hospital mortality",
    ],
)
def test_admission_type_is_read_as_the_named_exposure(question: str) -> None:
    slots = deterministic_intent(question)["slots"]
    assert slots["exposure"]["value"] == "adm"
    assert slots["exposure"]["provenance"] == "user_text"

    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question=question, database="eicu_demo"
    )
    assert coordinates["primary_exposure"] == "adm"
    assert coordinates["target_outcome"] == "death"
    assert coordinates["execution_authorized"] is False


def test_admission_type_does_not_capture_readmission() -> None:
    slots = deterministic_intent("ICU readmission by sex")["slots"]
    assert (slots["exposure"] or {}).get("value") != "adm"
    assert (slots["outcome"] or {}).get("value") != "adm"


def _receipts(monkeypatch: pytest.MonkeyPatch, available: bool) -> list[str]:
    seen: list[str] = []

    def fake(path: object) -> bool:
        seen.append(str(path))
        return available

    monkeypatch.setattr(
        scientific_runtime_projection, "export_kdigo_strict_derivation_available", fake
    )
    return seen


def test_a_named_kdigo_stage_is_read_as_the_observability_preserving_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _receipts(monkeypatch, True)

    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question=_KDIGO_QUESTION, database="eicu_demo", export_path="/bound/export"
    )

    assert coordinates["primary_exposure"] == "aki_stage_strict"
    # The strict stage is already the window reading; "最高" in the question
    # names an aggregation of the source concept, not of the derived stage.
    assert coordinates["primary_exposure_aggregation"] is None
    assert seen == ["/bound/export"]


def test_without_receipts_the_named_stage_is_left_for_the_runtime_to_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _receipts(monkeypatch, False)
    unbound = research_launch_scientific._metadata_only_planning_coordinates(
        question=_KDIGO_QUESTION, database="eicu_demo", export_path="/july/export"
    )
    assert unbound["primary_exposure"] == "aki_stage"
    # The source-concept reading is exactly what it was before.
    named = explicit_exposure_aggregation(_KDIGO_QUESTION, concept_id="aki_stage")
    assert unbound["primary_exposure_aggregation"] == (
        named.aggregation if named is not None else None
    )

    # Without a bound source the reading is never substituted either.
    seen = _receipts(monkeypatch, True)
    no_source = research_launch_scientific._metadata_only_planning_coordinates(
        question=_KDIGO_QUESTION, database="eicu_demo"
    )
    assert no_source["primary_exposure"] == "aki_stage"
    assert seen == []


def test_a_binary_aki_exposure_is_not_turned_into_a_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swapping a 0-3 stage for a yes/no AKI phenotype would change the question."""

    _receipts(monkeypatch, True)
    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question="急性肾损伤与住院死亡的关联", database="eicu_demo", export_path="/x"
    )
    assert coordinates["primary_exposure"] in {None, "aki"}
