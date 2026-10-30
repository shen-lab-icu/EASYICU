"""A candidate label is chosen by the researcher, not rewritten by a model."""
from types import SimpleNamespace

import pytest

from easyicu.webserver import research_pipeline_run_preparation as launch
from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot import study_context_update as update
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiSessionRecord, ToolExecutionContext


def test_explicit_candidate_uses_verbatim_label_and_clears_old_machine_outcome(monkeypatch):
    current = {"id": "study-synthetic", "revision": 1, "active_job_id": None,
               "question": "old", "outcome": "ICU stay", "cohort": {},
               "execution_concepts": {"outcome": "los_icu", "primary_exposure": "lact"}}
    writes = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(tool_module.study_contexts, "upsert_context",
                        lambda raw, **kwargs: writes.append(dict(raw)) or {**current, **raw, "revision": 2})
    session = PiSessionRecord(session_id="test-candidate", binding=AuthorityBinding(study_context_id=current["id"], study_revision=1))
    message = ("为生成待审候选计划，明确选择候选结局为从首次转入 ICU 或中间监护至病例关闭的时长；"
               "它可能包含普通病房时间，不等同于 ICU 住院时长。请将它写入候选研究配置，"
               "保留原问题与更改原因；这只是候选配置，不批准最终研究计划，也不授权提取数据或执行分析。")
    result = tool_module.execute_tool("easyicu_update_study_context", {
        "outcome": "首次进入 ICU 后的纯 ICU 住院时长",
        "execution_concepts": {"outcome": "los_icu"},
    }, ToolExecutionContext(session=session, user_message=message, allowed_actions={"configure"}))
    assert result["code"] == "study_context_updated"
    assert writes[-1]["outcome"] == "从首次转入 ICU 或中间监护至病例关闭的时长"
    assert "outcome" not in writes[-1]["execution_concepts"]
    assert writes[-1]["execution_concepts"]["primary_exposure"] == "lact"


@pytest.mark.parametrize("message,proposed", [
    ("主要结局采用 ICU 住院天数", "院内死亡"),
    ("不要使用 ICU 住院天数", "ICU 住院天数"),
    ("主要结局是否采用病例时长？", "病例时长"),
    ("候选结局：病例时长和院内死亡", "病例时长"),
    ("主要结局：住院天数；候选结局：病例时长", "病例时长"),
    ("研究乳酸与病例时长的关系。", "病例时长"),
    ("Do not use mortality as the primary outcome.", "mortality"),
    ("Should the primary endpoint be mortality?", "mortality"),
])
def test_denial_question_multiple_choice_and_model_rewriting_are_not_selection(message, proposed):
    assert not update._message_explicitly_selects_primary_outcome(message, proposed)


def test_unresolved_outcome_is_host_owned_digest_bound_and_blocks_launch():
    base = {"question": "candidate", "outcome": "old", "data_source": {"database": "sic"}}
    with pytest.raises(study_contexts.StudyContextError):
        study_contexts.validate_context_update({"unconfirmed_outcome": "new"})
    patch = study_contexts.validate_context_update({"unconfirmed_outcome": "new"}, _server_question_wording_write=True)
    assert study_contexts.scientific_configuration_sha256(base) != study_contexts.scientific_configuration_sha256({**base, **patch})
    for resume in ("", "earlier-run"):
        request = SimpleNamespace(study_context={**base, **patch}, execution_resume_source_run_id=resume)
        with pytest.raises(launch.ResearchPipelineRunError, match="not saved") as error:
            launch._prepare_scientific_launch(request)
        assert error.value.code == "study_primary_outcome_confirmation_required"


def test_pending_outcome_survives_store_and_cannot_be_cleared_by_client(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    created = study_contexts.upsert_context({"question": "synthetic question", "outcome": "old"})
    saved = study_contexts.upsert_context(
        {"id": created["id"], "purpose": "candidate", "unconfirmed_outcome": "new"},
        expected_revision=created["revision"], require_revision=True, _server_question_wording_write=True,
    )
    loaded = study_contexts.get_context(saved["id"])
    assert loaded["unconfirmed_outcome"] == "new" and loaded["outcome"] == "old"
    with pytest.raises(study_contexts.StudyContextError):
        study_contexts.upsert_context({"id": saved["id"], "unconfirmed_outcome": ""},
                                      expected_revision=saved["revision"], require_revision=True)


def test_partial_update_persists_conflict_and_later_explicit_choice_clears_it(monkeypatch):
    current = {"id": "study-synthetic", "revision": 1, "active_job_id": None,
               "question": "old", "outcome": "ICU stay", "cohort": {}}
    writes = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    def save(raw, **kwargs):
        writes.append(dict(raw))
        current.update(raw)
        current["revision"] += 1
        return dict(current)
    monkeypatch.setattr(tool_module.study_contexts, "upsert_context", save)
    session = PiSessionRecord(session_id="test-candidate", binding=AuthorityBinding(study_context_id=current["id"], study_revision=1))
    result = tool_module.execute_tool("easyicu_update_study_context", {"purpose": "探索关联", "outcome": "case duration"},
        ToolExecutionContext(session=session, user_message="研究新的关联。", allowed_actions={"configure"}))
    assert result["code"] == "study_context_updated"
    assert current["outcome"] == "ICU stay" and current["unconfirmed_outcome"] == "case duration"
    with pytest.raises(launch.ResearchPipelineRunError, match="not saved"):
        launch._prepare_scientific_launch(SimpleNamespace(study_context=current))
    result = tool_module.execute_tool("easyicu_update_study_context", {"outcome": "model invented"},
        ToolExecutionContext(session=session, user_message="明确选择候选结局为病例关闭时长。", allowed_actions={"configure"}))
    assert result["code"] == "study_context_updated"
    assert current["outcome"] == "病例关闭时长" and not current["unconfirmed_outcome"]
