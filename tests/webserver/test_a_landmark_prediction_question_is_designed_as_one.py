"""A prediction question stating several prediction times is designed as a dynamic one.

pilot1 10-27: the launch recorded 「在入 ICU 第 24、48、72 小时…预测之后 7 天内 ICU
死亡」 as a static prediction-model design, so the plan built one model.  The
planner's routing now reads that question as a dynamic (landmark) prediction,
the host's question family reads it as the prediction question it is, and the
design recorded from either reading is the dynamic one.  A prediction from one
time or window keeps its static design.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.webserver import source_identity_authority
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.question_family_check import check_question_family, host_question_family
from easyicu.webserver.study_family_design import checked_family_design, record_unset_family_design
from easyicu.webserver.study_family_reading import planner_stated_family

_LANDMARKS = (
    "在 eICU demo 中，在入 ICU 第 24、48、72 小时，用到当时为止的生命体征和化验，"
    "预测之后 7 天内 ICU 死亡的效果如何？"
)
_LANDMARKS_EN = (
    "In adults still in the ICU at 24, 48 and 72 hours, predict ICU death over the "
    "next 7 days from the vital signs available by then."
)
_STATIC = "用 eICU demo 中入 ICU 前 24 小时的数据建立一个预测 ICU 内死亡的模型，它的区分度和校准怎么样？"


@pytest.fixture(autouse=True)
def _no_patient_grouping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source_identity_authority, "resolve_study_patient_grouping", lambda **_kwargs: None,
    )


def _study(tmp_path, **fields: Any) -> dict[str, Any]:
    return {
        "id": "study-landmark", "revision": 5, "analysis_design": {},
        "data_source": {"path": str(tmp_path / "export"), "database": "eicu_demo"},
        **fields,
    }


@pytest.mark.parametrize("question", [_LANDMARKS, _LANDMARKS_EN])
def test_the_host_reads_a_landmark_question_as_a_prediction_question(question: str) -> None:
    assert planner_stated_family(question) == "dynamic_prediction"
    assert host_question_family(question) == "prediction_model"
    check = check_question_family({"family": "prediction_model", "words": question[:30]}, question)
    assert check["status"] == "agreed"
    # A conversation that reads it as a survival analysis disagrees, and the
    # researcher is asked.
    survival = check_question_family({"family": "survival", "words": question[:30]}, question)
    assert survival["status"] == "host_disagrees"


@pytest.mark.parametrize(
    ("question", "family"),
    [(_LANDMARKS, "dynamic_prediction"), (_LANDMARKS_EN, "dynamic_prediction"), (_STATIC, "prediction_model")],
)
def test_a_checked_prediction_question_is_designed_on_its_stated_times(tmp_path, question, family) -> None:
    check = check_question_family({"family": "prediction_model", "words": question[:30]}, question)
    checked = checked_family_design(_study(tmp_path, question_family_check=check), question)
    assert not checked.disagreement
    assert checked.family == family
    assert checked.design == {
        "analysis_family": family, "analysis_unit": "icu_stay", "variance_estimator": "model_based",
    }


@pytest.mark.parametrize(
    ("question", "family"), [(_LANDMARKS, "dynamic_prediction"), (_STATIC, "prediction_model")],
)
def test_the_launch_records_the_design_the_question_states(tmp_path, monkeypatch, question, family) -> None:
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(
        context_store, "upsert_context",
        lambda raw, **kwargs: writes.append({**raw, **kwargs}) or {**raw, "revision": 6},
    )

    launched = record_unset_family_design(_study(tmp_path), question=question)

    (write,) = writes
    assert write["analysis_design"]["analysis_family"] == family
    assert launched["analysis_design"] == write["analysis_design"]
