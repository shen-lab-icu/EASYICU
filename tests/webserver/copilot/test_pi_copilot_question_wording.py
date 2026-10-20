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
    "成人 ICU 入住 ≥24 小时，血糖 <65 mg/dL 的比例是多少？", ORIGINAL, "如何预测院内死亡风险？",
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


@pytest.mark.parametrize("origin", [None, "host_prompt", "typed"])
def test_service_requires_an_explicit_typed_origin_for_user_wording(tmp_path, study_state, origin):
    gateway = FakeGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    created = service.create_session(project_id="origin-project", external_llm_opt_in=True)
    session_id = created["session"]["session_id"]
    allow_unrelated_message_test(service, session_id)
    try:
        submitted = service.send_message(
            session_id, project_id="origin-project", message=ORIGINAL,
            message_origin=origin, allowed_actions=[],
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = service_module.jobs.MANAGER.get(submitted["job_id"])
            if job and job.status != "running":
                break
            time.sleep(0.01)
        assert job is not None and job.status == "done"
        turns = gateway.tool_contexts[-1].user_turns
        assert turns[-1]["message_origin"] == (origin or "unknown")
        resolved = resolve_question_wording(ORIGINAL, user_turns=turns)
        assert (resolved is not None) == (origin == "typed")
        params = next(p for method, p, _ in gateway.calls if method == "session.prompt")
        assert params["message_origin"] == (origin or "unknown")
        reopened = PiConversationReplayStore(service.replay_store.root)
        assert reopened.question_wording_context(
            session_id=session_id, project_id="origin-project",
        )["user_turns"] == list(turns)
    finally:
        service.close()


def test_route_accepts_host_prompt_origin_without_claiming_it_was_typed():
    from easyicu.webserver.routes.pi_copilot import PiMessageRequest

    request = PiMessageRequest(project_id="origin-project", message=ORIGINAL, message_origin="host_prompt")
    assert request.message_origin == "host_prompt"
    assert PiMessageRequest(project_id="origin-project", message=ORIGINAL).message_origin is None


@pytest.mark.parametrize("origin", [None, "host_prompt", "unknown"])
def test_non_typed_turns_break_adjacency_after_replay_reopen(tmp_path, origin):
    store = PiConversationReplayStore(tmp_path)
    for index, (text, source) in enumerate([
        ("Describe platelet counts.", "typed"), ("Host-written context.", origin),
        ("Report hospital mortality.", "typed"),
    ]):
        store.start_turn(session_id="s", project_id="p", job_id=str(index),
                         allowed_actions=[], user_message=text, message_origin=source)
    turns = PiConversationReplayStore(tmp_path).question_wording_context(
        session_id="s", project_id="p",
    )["user_turns"]
    assert turns[1]["message_origin"] == (origin or "unknown")
    assert resolve_question_wording("Host-written context", user_turns=turns) is None
    assert resolve_question_wording(
        "Describe platelet counts Report hospital mortality", user_turns=turns,
    ) is None


@pytest.mark.parametrize("cue", [
    "不研究", "没有研究", "无需研究", "不要研究", "我们别研究", "并非研究",
    "而非研究", "除了", "排除",
])
@pytest.mark.parametrize("side", ["before", "after"])
def test_a_match_cannot_discard_a_chinese_negation_or_exclusion(cue, side):
    question = "乳酸与死亡的关系"
    text = f"{cue}{question}，只描述乳酸分布。" if side == "before" else f"{question}{cue}，只描述乳酸分布。"
    assert resolve_question_wording(question, user_turns=[{"text": text, "message_origin": "typed"}]) is None


@pytest.mark.parametrize("cue", [
    "not", "no", "without", "rather than", "instead of", "except", "exclude",
    "rather   than", "instead\t of",
])
@pytest.mark.parametrize("side", ["before", "after"])
def test_a_match_cannot_discard_an_english_negation_or_exclusion(cue, side):
    question = "lactate and hospital mortality"
    text = f"{cue} {question}" if side == "before" else f"{question} {cue}"
    assert resolve_question_wording(question, user_turns=[{"text": text, "message_origin": "typed"}]) is None


@pytest.mark.parametrize("prefix", ["我想研究", "分别研究", "性别", "区别", "类别", "个别", "特别"])
def test_an_affirmative_clause_can_supply_a_question_fragment(prefix):
    question = "乳酸与死亡的关系"
    resolved = resolve_question_wording(question, user_turns=[{"text": prefix + question, "message_origin": "typed"}])
    assert resolved is not None and resolved[0] == question


@pytest.mark.parametrize("separator", ["。", "！", "？", "；", "，", ",", ".", "!", "?", ";", ":", "\n"])
def test_negation_in_a_separate_clause_does_not_block_wording(separator):
    question = "乳酸与死亡的关系"
    text = f"不要预测死亡{separator}我想研究{question}"
    assert resolve_question_wording(question, user_turns=[{"text": text, "message_origin": "typed"}]) is not None


def test_a_later_affirmative_occurrence_can_follow_a_rejected_occurrence():
    question = "乳酸与死亡的关系"
    text = f"不要研究{question}。更正：我想研究{question}。"
    assert resolve_question_wording(question, user_turns=[{"text": text, "message_origin": "typed"}]) is not None


def test_preserved_negation_and_negation_in_an_earlier_message_are_safe():
    question = "不研究乳酸与死亡的关系，只描述乳酸分布。"
    turns = [{"text": question, "message_origin": "typed"}]
    assert resolve_question_wording(question, user_turns=turns)[0] == question
    turns.append({"text": "乳酸与死亡的关系", "message_origin": "typed"})
    assert resolve_question_wording(turns[-1]["text"], user_turns=turns) is not None


@pytest.mark.parametrize("question", ["死亡", "院内死亡率", "研究死亡风险"])
@pytest.mark.parametrize("origin", ["typed", "model_option", "confirmed"])
def test_short_normalized_matches_never_receive_a_wording_record(question, origin):
    turns = [{"text": question, "message_origin": origin}]
    if origin == "confirmed":
        turns = [{"text": "确认这个问题的措辞", "message_origin": "typed"}]
    assert resolve_question_wording(question, user_turns=turns, previous_reply=question) is None


@pytest.mark.parametrize("reply", [
    "We could Describe platelet counts in adults before selecting a plan.",
    "Describe platelet counts in adults before selecting a plan.",
    "We could Describe platelet counts in adults.",
    'You can "choose a question" Describe platelet counts in adults.',
    "Researchers' Describe platelet counts in adults.",
    'We could "Describe platelet counts in adults.',
    'Describe platelet counts in adults "under a different question".',
])
def test_confirming_a_reply_does_not_confirm_an_arbitrary_sentence_fragment(reply):
    assert resolve_question_wording(
        "Describe platelet counts in adults",
        user_turns=[{"text": "Confirm this question wording", "message_origin": "typed"}],
        previous_reply=reply,
    ) is None


@pytest.mark.parametrize("reply", [
    'I propose "Describe platelet counts in adults." for review.',
    "I propose 'Describe platelet counts in adults.' for review.",
    "I propose `Describe platelet counts in adults.` for review.",
    "建议题目：「Describe platelet counts in adults.」请确认。",
    "Proposed question:\nDescribe platelet counts in adults.\nPlease confirm.",
])
def test_a_whole_quoted_question_or_line_can_be_confirmed(reply):
    resolved = resolve_question_wording(
        "Describe platelet counts in adults",
        user_turns=[{"text": "Confirm this question wording", "message_origin": "typed"}],
        previous_reply=reply,
    )
    assert resolved is not None and resolved[1]["source"] == "confirmed_proposal"


@pytest.mark.parametrize("multiple_messages", [False, True])
def test_multiline_wording_survives_the_real_study_context_write(tmp_path, monkeypatch, multiple_messages):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    current = study_contexts.upsert_context({"id": "wording-study", "question": "Old question"})
    lines = ["Describe platelet counts in adults.", "Also report hospital mortality."]
    original = "\n\t".join(lines)
    turns = [{"text": line, "message_origin": "typed"} for line in lines] if multiple_messages else [
        {"text": original, "message_origin": "typed"},
    ]
    context = _context(original, user_turns=turns)
    result = update_study_context(
        context, {"question": " ".join(lines)}, load_context=lambda _: current,
        project_workflow=lambda *args, **kwargs: {},
    )
    assert result["code"] == "study_context_updated"
    saved = study_contexts.get_context(current["id"])
    assert saved["question"] == " ".join(lines)
    assert saved["question_wording"] == {
        "source": "user_message", "question_sha256": question_sha256(original),
    }
    assert result["details"]["question_wording"] == saved["question_wording"]


def test_receipt_does_not_claim_wording_the_store_cleared(monkeypatch):
    current = {"id": "wording-study", "revision": 1, "question": "Old question"}
    monkeypatch.setattr(study_contexts, "upsert_context", lambda patch, **kwargs: {
        **current, **patch, "revision": 2, "question_wording": None,
    })
    result = update_study_context(
        _context(ORIGINAL), {"question": ORIGINAL}, load_context=lambda _: current,
        project_workflow=lambda *args, **kwargs: {},
    )
    assert result["code"] == "study_context_updated"
    assert result["details"].get("question_wording") is None
