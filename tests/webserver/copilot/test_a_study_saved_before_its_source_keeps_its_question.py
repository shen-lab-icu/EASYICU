"""A study saved before its data source is confirmed keeps its question.

Exact data columns are bound against a source's catalog.  On a new hosted
project (pilot1, 10-22) the first update named "请调整性别"; the host
synchronized ``sex`` into the executable columns, and with no source bound it
refused the whole update.  The question, outcome and exposure were lost, and
after the researcher confirmed the source no plan could be generated.  Now the
columns are left for the planner, named as not saved with their reason, and
everything else is saved.  With a source bound the columns are still checked.
"""

from __future__ import annotations

from typing import Any

from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)

MESSAGE = "在 eICU 全部 ICU 入住中，65 岁及以上与 65 岁以下患者的院内死亡风险有差别吗？请调整性别。"
# The pilot's call, verbatim.
CALL = {
    "question": MESSAGE,
    "cohort": {
        "preset": "all_icu",
        "label": "eICU 全部 ICU 入住",
        "review": "比较 65 岁及以上与 65 岁以下患者",
        "source_type": "用户明确指定 eICU 全部 ICU 入住",
    },
    "outcome": "院内死亡",
    "primary_exposure": "年龄分组：65 岁及以上 vs 65 岁以下",
    "covariates": ["sex"],
    "covariate_selection": "planner_selectable",
    "analysis_goal": "比较 65 岁及以上与 65 岁以下患者的院内死亡风险，并进行性别调整。",
}
NEW_PROJECT = {
    "id": "study-new-project",
    "revision": 1,
    "active_job_id": None,
    "question": "",
    "cohort": {},
    "outcome": "",
    "primary_exposure": "",
    "analysis_goal": "",
    "execution_concepts": {},
}


def _update(monkeypatch, params: dict[str, Any], current: dict[str, Any]):
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module,
        "_workflow_snapshot",
        lambda *_args, **_kwargs: {
            "next_action_code": "study_setup_incomplete",
            "missing_setup_fields": ["data_source"],
            "study_setup_receipt": {"configuration": {"data_source": {}}},
        },
    )
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**current, **raw, "revision": 2},
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-source-later",
            binding=AuthorityBinding(study_context_id=current["id"], study_revision=current["revision"]),
        ),
        user_message=MESSAGE,
        allowed_actions={"configure"},
    )
    return tool_module.execute_tool("easyicu_update_study_context", params, context), writes


def test_the_question_is_saved_and_the_columns_wait_for_the_source(monkeypatch):
    result, writes = _update(monkeypatch, CALL, NEW_PROJECT)

    assert (result["status"], result["code"]) == ("ok", "study_context_updated")
    (saved,) = writes
    assert saved["question"] == MESSAGE
    assert saved["outcome"] == "院内死亡"
    assert saved["covariates"] == ["sex"]
    assert not saved.get("execution_concepts")
    assert ("execution_concepts", "study_execution_source_required") in [
        (item["field"], item["code"]) for item in result["details"]["unconfirmed_omissions"]
    ]
    assert "execution_concepts (study_execution_source_required" in result["summary"]


def test_proposed_columns_also_wait_for_the_source(monkeypatch):
    result, writes = _update(
        monkeypatch,
        {"question": MESSAGE, "execution_concepts": {"outcome": "death"}},
        NEW_PROJECT,
    )

    assert result["status"] == "ok"
    assert not writes[0].get("execution_concepts")
    assert "execution_concepts" in result["details"]["omitted_unconfirmed_fields"]
