"""The exposure operation the question states is never silently overwritten (B31, S1736).

UX on rc_4cc: the question said 「暴露只取入科时的首次乳酸，不使用 24 小时峰值」, the
assistant answered that it would switch to the first value, and the study stored
``primary_exposure_aggregation = max``.  The launch also read 「Sepsis-3 脓毒症」 as the
exposure.  A write that contradicts the operation the question states is now refused
with both readings; an operation the question has superseded is dropped; and a
population condition is not projected as the exposure.  Isolated stores, metadata only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.research_launch_scientific import (
    _metadata_only_planning_coordinates,
)

_FIRST = (
    "Sepsis-3 脓毒症成人 ICU 患者中，ICU 入科时（基线，入科后首次测量）的乳酸水平与院内死亡的关联，"
    "并校正基线协变量。暴露只取入科时的首次乳酸，不使用 24 小时峰值。"
)
_FIRST_EN = "In adult ICU patients, is the first lactate at ICU admission associated with in-hospital death?"
_PEAK = "研究成人 ICU 患者前24小时最高乳酸与院内死亡"
_PLAIN = "研究成人 ICU 患者的乳酸与院内死亡"


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )


def _study(question: str, aggregation: str | None = None) -> dict:
    execution = {"primary_exposure": "lact"}
    if aggregation:
        execution["primary_exposure_aggregation"] = aggregation
    return context_store.upsert_context(
        {"id": "study_b31", "question": question, "execution_concepts": execution},
        lifecycle_write=False,
    )


@pytest.mark.parametrize("question", [_FIRST, _FIRST_EN], ids=["zh", "en"])
def test_a_write_against_the_stated_first_value_is_refused(question: str) -> None:
    with pytest.raises(context_store.StudyContextError) as refused:
        _study(question, "max")

    detail = refused.value.detail
    assert detail["error"] == "study_exposure_aggregation_conflicts_question"
    assert (detail["written"], detail["stated"]) == ("max", "first")
    assert detail["stated_words"]


@pytest.mark.parametrize("question", [_FIRST, _FIRST_EN], ids=["zh", "en"])
def test_the_stated_first_value_is_stored(question: str) -> None:
    saved = _study(question, "first")

    assert saved["execution_concepts"]["primary_exposure_aggregation"] == "first"


def test_a_question_that_supersedes_the_operation_drops_it() -> None:
    saved = _study(_PEAK, "max")

    changed = context_store.upsert_context(
        {"id": "study_b31", "question": _FIRST},
        expected_revision=saved["revision"], lifecycle_write=False,
    )

    assert changed["execution_concepts"] == {"primary_exposure": "lact"}


def test_a_question_that_states_no_operation_keeps_the_stored_one() -> None:
    assert _study(_PLAIN, "max")["execution_concepts"]["primary_exposure_aggregation"] == "max"


def test_the_population_condition_is_not_the_exposure() -> None:
    coordinates = _metadata_only_planning_coordinates(question=_FIRST, database="miiv")

    assert coordinates["primary_exposure"] not in {"sep3", "sep3_sofa1", "sep3_sofa2"}
    assert ["sep3", "sep3_sofa1", "sep3_sofa2"] in [
        item["concepts"] for item in coordinates["question_named_concepts"]
    ]
