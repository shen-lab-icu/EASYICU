"""The conversation states the family a question asks for, and the host checks it.

An offline probe of 25 common phrasings (10-22, 关键词漏判/probe.py) found the
host's word readings wrong on 11: 「乳酸越高，院内死亡越多吗？」 and
「用 logistic 回归分析…」 were planned as descriptions, and 「…与生存出院有关吗？」
「生存率是多少？」 were refused on eICU as survival studies.  The conversation's
model now states the family with the question's own words
(``question_family``), and the host checks it against its own reading: the
model's reading stands where the host's words read nothing, and where the
two differ the plan step asks the researcher, whose choice is recorded.
A question rewritten without the stated words drops the check.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.question_family_check import (
    FAMILY_DISAGREEMENT,
    QUESTION_FAMILY_CHECK_FIELD,
    check_question_family,
    family_choice_texts,
    normalize_question_family_check,
    open_disagreement,
    researcher_choice,
    settled_question_family,
)
from easyicu.webserver.study_family_design import (
    checked_family_design,
    record_unset_family_design,
)
from easyicu.webserver.study_family_scope import (
    STUDY_FAMILY_DATABASE_OUT_OF_SCOPE,
    study_family_database_gap,
)

_DISCHARGE = "eICU 中，入 ICU 首日最高乳酸与生存出院有关吗？"
_GRADIENT = "eICU 中，入 ICU 首日最高乳酸越高，院内死亡越多吗？"
_COX = "eICU 中，用 Cox 回归分析首日最高乳酸与院内死亡。"
_HAZARD = "eICU 中，乳酸分组的死亡风险比（HR）是多少？"


def _check(question: str, family: str, words: str) -> dict[str, Any]:
    return check_question_family({"family": family, "words": words}, question)


# -- the check ---------------------------------------------------------------


def test_the_models_reading_stands_where_the_host_reads_no_family() -> None:
    record = _check(_GRADIENT, "association_study", "越高，院内死亡越多吗")

    assert record == {
        "model_family": "association_study",
        "words": "越高，院内死亡越多吗",
        "host_family": None,
        "status": "model_only",
    }
    assert settled_question_family(record, _GRADIENT) == "association_study"
    assert open_disagreement(record, _GRADIENT) is None


def test_both_readings_agree_on_a_stated_method() -> None:
    record = _check(_COX, "survival", "Cox 回归")

    assert (record["host_family"], record["status"]) == ("survival", "agreed")
    assert settled_question_family(record, _COX) == "survival"


def test_differing_readings_settle_nothing_and_ask() -> None:
    record = _check(_DISCHARGE, "association_study", "与生存出院有关吗")

    assert (record["host_family"], record["status"]) == ("survival", "host_disagrees")
    assert settled_question_family(record, _DISCHARGE) is None
    assert open_disagreement(record, _DISCHARGE) == {
        "code": FAMILY_DISAGREEMENT,
        "model_family": "association_study",
        "host_family": "survival",
    }


def test_the_researcher_chooses_one_offered_reading() -> None:
    record = _check(_DISCHARGE, "association_study", "与生存出院有关吗")
    offered = family_choice_texts(open_disagreement(record, _DISCHARGE), "zh")
    assert [family for family, _ in offered] == ["association_study", "survival"]

    chosen = researcher_choice(record, _DISCHARGE, offered[0][1] + "。")
    assert chosen is not None
    assert (chosen["status"], chosen["chosen_family"]) == ("researcher_chose", "association_study")
    assert normalize_question_family_check(chosen) == chosen
    assert settled_question_family(chosen, _DISCHARGE) == "association_study"
    assert researcher_choice(record, _DISCHARGE, "按关联研究做") is None


@pytest.mark.parametrize(
    "stated",
    [
        {"family": "association_study", "words": "这几个字不在题目里"},
        {"family": "regression", "words": "有关吗"},
        {"family": "association_study"},
    ],
)
def test_a_statement_not_resting_on_the_question_is_not_read(stated: dict) -> None:
    with pytest.raises(ValueError):
        check_question_family(stated, _DISCHARGE)


def test_a_question_rewritten_without_the_words_drops_the_check() -> None:
    record = _check(_DISCHARGE, "association_study", "与生存出院有关吗")

    assert open_disagreement(record, "eICU 中，入 ICU 首日最高乳酸与院内死亡有关吗？") is None


# -- what the study's family decides ------------------------------------------


def _study(question: str, check: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "id": "study-family-check",
        "revision": 1,
        "question": question,
        "data_source": {"path": "/prepared/export", "database": "eicu"},
        QUESTION_FAMILY_CHECK_FIELD: check,
    }


def test_an_in_scope_reading_goes_on_and_a_survival_one_needs_a_censoring_clock() -> None:
    discharge = _check(_DISCHARGE, "association_study", "与生存出院有关吗")
    chosen = researcher_choice(discharge, _DISCHARGE, "按关联研究做（暴露与结局是否相关、相关多强）")
    hazard = _check(_HAZARD, "survival", "死亡风险比（HR）")

    assert study_family_database_gap(_study(_DISCHARGE, chosen), question=_DISCHARGE, database="eicu") is None
    # eICU censors in-hospital death at hospital discharge; HiRID records no
    # complete censoring clock, so the same survival reading stops there.
    assert study_family_database_gap(_study(_HAZARD, hazard), question=_HAZARD, database="eicu") is None
    gap = study_family_database_gap(_study(_HAZARD, hazard), question=_HAZARD, database="hirid")
    assert gap is not None and (gap["code"], gap["family"]) == (STUDY_FAMILY_DATABASE_OUT_OF_SCOPE, "survival")


@pytest.mark.parametrize("database", ["eicu", "miiv"])
def test_a_disagreement_is_asked_on_every_database(database: str) -> None:
    record = _check(_DISCHARGE, "association_study", "与生存出院有关吗")

    gap = study_family_database_gap(_study(_DISCHARGE, record), question=_DISCHARGE, database=database)

    assert gap is not None
    assert (gap["code"], gap["model_family"], gap["host_family"]) == (
        FAMILY_DISAGREEMENT, "association_study", "survival",
    )
    assert [item["family"] for item in gap["choices"]] == ["association_study", "survival"]
    assert gap["choices"][1]["zh"] == "按生存分析做（时间到事件）"


def test_the_models_reading_records_the_design_the_words_missed() -> None:
    study = _study(_GRADIENT, _check(_GRADIENT, "association_study", "越高，院内死亡越多吗"))

    checked = checked_family_design(study, _GRADIENT)

    assert (checked.disagreement, checked.family) == (False, "association_study")
    assert checked.design is not None and checked.design["analysis_family"] == "association_study"


def test_the_launch_records_no_design_while_the_researcher_is_asked() -> None:
    study = _study(_DISCHARGE, _check(_DISCHARGE, "association_study", "与生存出院有关吗"))

    assert record_unset_family_design(study, question=_DISCHARGE) is study


# -- the setup owner ------------------------------------------------------------


_CURRENT = {
    "id": "study-family-check",
    "revision": 1,
    "active_job_id": None,
    "question": _DISCHARGE,
}


def _update(monkeypatch, params: dict[str, Any], *, current: dict, message: str):
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(tool_module, "_workflow_snapshot", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**current, **raw, "revision": 2},
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-family-check",
            binding=AuthorityBinding(study_context_id=current["id"], study_revision=current["revision"]),
        ),
        user_message=message,
        allowed_actions={"configure"},
    )
    result = tool_module.execute_tool("easyicu_update_study_context", params, context)
    return result, writes


def test_a_saved_question_records_the_checked_reading(monkeypatch) -> None:
    params = {
        "question": _DISCHARGE,
        "question_family": {"family": "association_study", "words": "与生存出院有关吗"},
    }
    result, writes = _update(monkeypatch, params, current=dict(_CURRENT), message=_DISCHARGE)

    assert (result["status"], result["code"]) == ("ok", "study_context_updated")
    assert writes[0][QUESTION_FAMILY_CHECK_FIELD]["status"] == "host_disagrees"
    assert "analysis_design" not in writes[0]
    assert result["details"][QUESTION_FAMILY_CHECK_FIELD]["host_family"] == "survival"


def test_an_unread_statement_saves_the_rest(monkeypatch) -> None:
    params = {
        "question": _DISCHARGE,
        "question_family": {"family": "association_study", "words": "不在题目里的话"},
    }
    result, writes = _update(monkeypatch, params, current=dict(_CURRENT), message=_DISCHARGE)

    assert result["status"] == "ok"
    assert QUESTION_FAMILY_CHECK_FIELD not in writes[0]
    assert "question_family_unread" in result["details"]


def test_the_researchers_choice_is_recorded_whatever_the_turn_states(monkeypatch) -> None:
    record = _check(_DISCHARGE, "association_study", "与生存出院有关吗")
    current = {**_CURRENT, QUESTION_FAMILY_CHECK_FIELD: record}
    params = {"question_family": {"family": "survival", "words": "与生存出院有关吗"}}

    result, writes = _update(
        monkeypatch, params, current=current, message="按关联研究做（暴露与结局是否相关、相关多强）",
    )

    assert result["status"] == "ok"
    saved = writes[0][QUESTION_FAMILY_CHECK_FIELD]
    assert (saved["status"], saved["chosen_family"]) == ("researcher_chose", "association_study")
    assert writes[0]["analysis_design"]["analysis_family"] == "association_study"
