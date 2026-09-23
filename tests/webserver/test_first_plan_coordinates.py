"""The first candidate plan reads the exposure the researcher actually named.

Every Web study used to need at least two planning rounds: the first candidate
was reviewed as ``scientific_capability_data_contract_unresolved`` because the
metadata-only planning coordinates had no exposure, and the typed exposure only
reached the StudyContext after that review.  The coordinate owner reads what
the researcher explicitly named; these tests fix a reading that was
missing, and that the reading stays a reading -- never a guess.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import research_launch_scientific
from easyicu.webserver.study_intent import deterministic_intent

_ADMISSION_QUESTION = (
    "在已准备好的 eICU Collaborative Research Database Demo v2.0.1 中，评估 ICU "
    "入院类型（内科、外科、其他）与住院死亡之间的关联。入院类型定义为 ICU 入院时"
    "的基线特征，不采用 landmark 设计。"
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
