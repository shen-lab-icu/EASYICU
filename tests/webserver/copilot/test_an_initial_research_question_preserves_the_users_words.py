"""The first typed research question survives an unsolicited model rewrite."""

import pytest

from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiSessionRecord, ToolExecutionContext
from easyicu.webserver.pi_copilot.study_context_update import update_study_context
from easyicu.webserver.study_question_wording import question_sha256


QUESTION = (
    "我想使用公开的 eICU Demo 做一项描述性研究，研究单位是 ICU 入住。纳入成人，并以入 ICU 后 24 小时作为评估时点："
    "纳入入 ICU 后前 24 小时内首次血清肌酐不超过 4 mg/dL 的入住，肌酐缺失者排除；同时排除同一窗口内最高乳酸"
    "达到或超过 2 mmol/L 的入住，乳酸缺失者也排除。请统计最终纳入人数、各条件排除和缺失人数，并生成队列流程表"
    "和可核查报告。先给我完整研究计划审阅，暂不启动数据提取或分析。"
)
REWRITE = "研究成人 ICU 入住的肌酐、乳酸及其死亡率，制定研究计划。"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    study = study_contexts.upsert_context({"title": "Question wording"})
    session = PiSessionRecord(
        session_id="first-question", binding=AuthorityBinding(
            study_context_id=study["id"], study_revision=study["revision"],
        ),
    )
    return study, session


def _update(session, message, *, turns=None, actions=("configure",), **params):
    context = ToolExecutionContext(session=session, user_message=message, user_turns=turns, allowed_actions=actions)
    result = update_study_context(
        context, {"question": REWRITE, **params},
        project_workflow=lambda *args, **kwargs: {},
    )
    return result, context


def test_original_first_turn_is_saved_alone_with_host_wording_receipt(setup):
    study, session = setup
    result, _ = _update(session, QUESTION, outcome="Invented mortality", purpose="Invented purpose")
    saved = study_contexts.get_context(study["id"])
    assert result["code"] == "study_context_updated"
    assert saved["question"] == QUESTION
    assert saved["question_wording"] == {"source": "user_message", "question_sha256": question_sha256(QUESTION)}
    assert saved["outcome"] == study["outcome"]
    assert saved["purpose"] == study["purpose"]
    assert saved["data_source"] == study["data_source"]


@pytest.mark.parametrize("message, turns", [
    ("好的，继续", [{"text": QUESTION, "message_origin": "typed"}, {"text": "好的，继续", "message_origin": "typed"}]),
    (QUESTION, [{"text": QUESTION, "message_origin": "model_option"}]),
    (QUESTION, [{"text": QUESTION, "message_origin": "typed", "superseded": True}]),
    (QUESTION, [{"text": "different", "message_origin": "typed"}]),
    ("如何使用 EasyICU 生成 ICU 死亡率研究计划？", None),
    ("请生成研究计划", None),
])
def test_clarification_choice_or_ambiguous_message_is_not_saved_as_the_question(setup, message, turns):
    study, session = setup
    result, _ = _update(session, message, turns=turns)
    assert result["code"] == "study_question_not_in_the_researchers_words"
    assert not study_contexts.get_context(study["id"])["question"]


def test_existing_question_is_never_replaced_by_the_fallback(setup):
    study, session = setup
    saved = study_contexts.upsert_context({"id": study["id"], "question": "Existing research question"})
    session.binding = session.binding.model_copy(update={"study_revision": saved["revision"]})
    result, _ = _update(session, QUESTION)
    assert result["code"] == "study_question_not_in_the_researchers_words"
    assert study_contexts.get_context(study["id"])["question"] == "Existing research question"


def test_initial_fallback_cannot_consume_an_ungranted_configuration(setup):
    study, session = setup
    result, _ = _update(session, QUESTION, actions=())
    assert result["code"] == "pi_action_authorization_required"
    assert not study_contexts.get_context(study["id"])["question"]
