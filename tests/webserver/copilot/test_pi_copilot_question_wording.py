"""A Copilot study keeps the researcher's wording and its host provenance."""

from __future__ import annotations

import time
from typing import Any

import pytest

from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding, PiSessionRecord, ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.study_context_update import update_study_context
from easyicu.webserver.study_question_wording import resolve_question_wording
from easyicu.webserver.study_question_wording import question_sha256
from easyicu.webserver.pi_copilot.replay_store import PiConversationReplayStore
from easyicu.webserver.pi_copilot import service as service_module
from easyicu.webserver.pi_copilot.service import PiCopilotService
from tests.webserver.copilot.pi_copilot_contract_fixtures import (
    FakeGateway, allow_unrelated_message_test, study_state as _study_state,
)

study_state = _study_state

ORIGINAL = "Describe platelet counts in adult patients during the first day, and hospital mortality."
REWRITTEN = "Describe day-one platelet counts and hospital mortality in adults."


def _update(monkeypatch: pytest.MonkeyPatch, context: ToolExecutionContext, proposal: str):
    current = {"id": "wording-study", "revision": 1, "question": "Old question"}
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(
        study_contexts, "upsert_context",
        lambda patch, **kwargs: writes.append(dict(patch)) or {**current, **patch, "revision": 2},
    )
    result = update_study_context(
        context, {"question": proposal, "purpose": "Describe the cohort"},
        load_context=lambda _: current, project_workflow=lambda *args, **kwargs: {},
    )
    return result, writes


def _context(message: str, **kwargs: Any) -> ToolExecutionContext:
    return ToolExecutionContext(
        session=PiSessionRecord(
            session_id="wording-session",
            binding=AuthorityBinding(study_context_id="wording-study", study_revision=1),
        ),
        user_message=message, allowed_actions={"configure"}, **kwargs,
    )


def test_a_model_rewrite_is_refused_without_spending_the_configuration_grant(monkeypatch):
    context = _context(ORIGINAL)
    result, writes = _update(monkeypatch, context, REWRITTEN)
    assert result["code"] == "study_question_not_in_the_researchers_words"
    assert writes == []
    assert "configure" in context.allowed_actions

    corrected, writes = _update(monkeypatch, context, ORIGINAL)
    assert corrected["code"] == "study_context_updated"
    assert writes[0]["question"] == ORIGINAL


@pytest.mark.parametrize("question", [
    "成人 ICU 入住 ≥24 小时，血糖 <65 mg/dL 的比例是多少？", ORIGINAL, "如何预测死亡？",
])
def test_spacing_and_punctuation_return_the_original_words(question):
    proposal = question.replace(" ", "").replace("，", "").replace("？", "").replace(",", "")
    resolved = resolve_question_wording(proposal, user_turns=[{"text": question, "message_origin": "typed"}])
    assert resolved is not None and resolved[0] == question


@pytest.mark.parametrize("proposal", [REWRITTEN, ORIGINAL.replace("adult", "young"), ORIGINAL.replace("first", "second")])
def test_rewording_and_scientific_changes_are_not_character_normalization(proposal):
    assert resolve_question_wording(proposal, user_turns=[{"text": ORIGINAL, "message_origin": "typed"}]) is None


def test_a_comparison_symbol_is_part_of_the_researchers_words():
    original = "成人血糖 <65 mg/dL 的比例是多少？"
    assert resolve_question_wording(original.replace("<", ">"), user_turns=[{"text": original, "message_origin": "typed"}]) is None


@pytest.mark.parametrize("original, proposal", [
    ("Report values below 10.5 mg/dL.", "Report values below 105 mg/dL."),
    ("Report a 5% threshold.", "Report a 5 threshold."),
    ("Report the change below -5.", "Report the change below 5."),
    ("Report the ratio 1:2.", "Report the ratio 12."),
    ("Report 1,000 observations.", "Report 1000 observations."),
])
def test_numeric_punctuation_cannot_change_a_quantity(original, proposal):
    assert resolve_question_wording(
        proposal, user_turns=[{"text": original, "message_origin": "typed"}],
    ) is None


def test_consecutive_user_messages_keep_each_original_segment():
    turns = [{"text": "Describe platelets in adults.", "message_origin": "typed"},
             {"text": "Also report hospital mortality.", "message_origin": "typed"}]
    resolved = resolve_question_wording("Describe platelets in adults Also report hospital mortality", user_turns=turns)
    assert resolved is not None
    assert resolved[0] == "Describe platelets in adults.\nAlso report hospital mortality."
    for barrier in [{"text": "Use a registered source.", "message_origin": "host_transition"},
                    {"text": "Describe platelets in adults.", "message_origin": "typed", "superseded": True}]:
        assert resolve_question_wording("Describe platelets in adults Also report hospital mortality", user_turns=[turns[0], barrier, turns[1]]) is None


@pytest.mark.parametrize("origin", ["typed", "model_option"])
def test_explicit_wording_confirmation_uses_only_the_previous_reply(origin):
    turns = [{"text": "确认这个问题的措辞", "message_origin": origin}]
    resolved = resolve_question_wording(REWRITTEN, user_turns=turns, previous_reply=f"Proposed question: {REWRITTEN}")
    assert resolved is not None and resolved[0] == REWRITTEN
    assert resolved[1]["source"] == "confirmed_proposal"
    assert resolve_question_wording(REWRITTEN, user_turns=turns, previous_reply="No proposal was made.") is None


@pytest.mark.parametrize("reply", [
    "好的", "继续", "是否采用这个问题？", "不要采用这个定义", "Do not use this question",
    "Should we use this question", "Explain the use of this question",
    "What does use this question mean", "I will never use this question",
    'I said "use this question" yesterday', "Please confirm this question is clinically appropriate",
])
def test_an_ambiguous_negative_or_question_reply_does_not_confirm_a_proposal(reply):
    assert resolve_question_wording(REWRITTEN, user_turns=[{"text": reply, "message_origin": "typed"}], previous_reply=REWRITTEN) is None


@pytest.mark.parametrize("reply", [
    "确认这个问题的措辞", "就用这个问题", "I confirm this question wording",
    "Please use the proposed wording", "I would like to use this question",
    "不要再问了，确认这个问题的措辞。",
])
def test_a_direct_wording_choice_can_confirm_the_previous_proposal(reply):
    resolved = resolve_question_wording(
        REWRITTEN, user_turns=[{"text": reply, "message_origin": "typed"}],
        previous_reply=REWRITTEN,
    )
    assert resolved is not None and resolved[1]["source"] == "confirmed_proposal"


def test_clicking_the_proposal_is_distinct_from_clicking_a_generic_option():
    resolved = resolve_question_wording(REWRITTEN, user_turns=[{"text": REWRITTEN, "message_origin": "model_option"}])
    assert resolved is not None and resolved[1]["source"] == "clicked_option"
    assert resolve_question_wording(REWRITTEN, user_turns=[{"text": "Keep going", "message_origin": "model_option"}]) is None


def test_wording_receipt_is_server_owned_and_expires_on_a_different_question(tmp_path, monkeypatch):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    receipt = {"source": "user_message", "question_sha256": question_sha256(ORIGINAL)}
    with pytest.raises(study_contexts.StudyContextError, match="study_question_wording_server_owned"):
        study_contexts.upsert_context({"id": "wording-study", "question": ORIGINAL, "question_wording": receipt})
    saved = study_contexts.upsert_context(
        {"id": "wording-study", "question": ORIGINAL, "question_wording": receipt},
        _server_question_wording_write=True,
    )
    assert study_contexts.get_context(saved["id"])["question_wording"] == receipt
    edited = study_contexts.upsert_context({"id": saved["id"], "question": REWRITTEN})
    assert edited["question_wording"] is None


def test_private_turn_words_and_proposals_survive_reopen_without_leaking_to_replay(tmp_path):
    store = PiConversationReplayStore(tmp_path)
    store.start_turn(session_id="s", project_id="p", job_id="a", allowed_actions=[],
                     user_message=ORIGINAL, message_origin="typed")
    store.finish_turn(session_id="s", project_id="p", job_id="a", status="done", assistant_reply=REWRITTEN)
    reopened = PiConversationReplayStore(tmp_path)
    history = reopened.question_wording_context(session_id="s", project_id="p")
    assert history == {"user_turns": [{"text": ORIGINAL, "message_origin": "typed"}], "previous_reply": REWRITTEN}
    public = reopened.snapshot(session_id="s", project_id="p")
    assert not {"user_message", "message_origin", "assistant_reply"}.intersection(public["turns"][0])
    store.supersede_from_turn_index(session_id="s", project_id="p", turn_index=0)
    assert reopened.question_wording_context(session_id="s", project_id="p") == {"user_turns": [], "previous_reply": ""}


def test_service_binds_active_user_words_and_the_previous_visible_proposal(tmp_path, study_state):
    class Gateway(FakeGateway):
        def request(self, method, params, **kwargs):
            response = super().request(method, params, **kwargs)
            if method == "session.prompt":
                response["transcript"] = [{
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "text": "An unseen alternative question."},
                        {"type": "text", "text": f"Proposed question: {REWRITTEN}"},
                    ],
                }]
            return response

    gateway = Gateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    created = service.create_session(project_id="wording-project", external_llm_opt_in=True)
    session_id = created["session"]["session_id"]
    allow_unrelated_message_test(service, session_id)
    for message in [ORIGINAL, "Confirm this question wording"]:
        submitted = service.send_message(
            session_id, project_id="wording-project", message=message,
            message_origin="typed", allowed_actions=[],
        )
        deadline = time.monotonic() + 5
        job = None
        while time.monotonic() < deadline:
            job = service_module.jobs.MANAGER.get(submitted["job_id"])
            if job and job.status != "running":
                break
            time.sleep(0.01)
        assert job is not None and job.status == "done"

    first, second = gateway.tool_contexts
    assert list(first.user_turns) == [{"text": ORIGINAL, "message_origin": "typed"}]
    assert first.previous_reply == ""
    assert list(second.user_turns) == [
        {"text": ORIGINAL, "message_origin": "typed"},
        {"text": "Confirm this question wording", "message_origin": "typed"},
    ]
    assert second.previous_reply == f"Proposed question: {REWRITTEN}"
    resolved = resolve_question_wording(
        REWRITTEN, user_turns=second.user_turns, previous_reply=second.previous_reply,
    )
    assert resolved is not None and resolved[1]["source"] == "confirmed_proposal"
    public = service.replay_store.snapshot(session_id=session_id, project_id="wording-project")
    assert all(not {"user_message", "message_origin", "assistant_reply"}.intersection(turn)
               for turn in public["turns"])
    service.close()


def test_host_captured_wording_cannot_be_mutated_through_context_views():
    context = _context(ORIGINAL)
    with pytest.raises(TypeError):
        context.user_turns[0]["text"] = REWRITTEN
    with pytest.raises(AttributeError):
        context.previous_reply = REWRITTEN


def test_oversized_verbatim_question_is_rejected_instead_of_silently_truncated(monkeypatch):
    question = "Describe ICU outcomes. " * 80
    context = _context(question)
    result, writes = _update(monkeypatch, context, question)
    assert result["code"] == "study_context_field_too_long"
    assert writes == [] and "configure" in context.allowed_actions
