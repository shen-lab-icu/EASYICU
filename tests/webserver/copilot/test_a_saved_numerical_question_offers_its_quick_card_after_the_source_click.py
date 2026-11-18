"""A saved numerical question offers its quick card after the source click.

pilot1 (10-26, public eICU demo, the entry page's 「快速回答」 example): a new
project asked 「在 eICU demo 中，入 ICU 第一天最高乳酸的中位数是多少？男女之间有差别吗？」.
The first turn saved the question with no source bound and showed the source
card; the researcher clicked 「使用这份数据」; the data-source transition, which
allowed no tool, only acknowledged the source, and the page started a full
plan automatically.  A quick answer needs a bound source, so no quick card
could ever be offered on this path.

Now the transition offers the saved question to the quick-answer owner when it
is, word for word, a message the researcher typed.  The host binds the turn's
question to it, the conversation's model may submit a reading, and the host
qualifies it into the same digest-bound card the researcher clicks.  A reading
that rewords the question is not qualified.  A study question, a stopped
proposal or no call keep today's acknowledgement and automatic plan: the page
side is in tests/js/guided_pi_quick_answers.test.js and the Node settlement of
the model's decision in tests/js/provider_transient_retry.test.js.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import pytest

from easyicu.webserver import sources, study_contexts
from easyicu.webserver.pi_copilot.contracts import PiCopilotError, ToolExecutionContext
from easyicu.webserver.pi_copilot.tools import execute_tool
from tests.support.node import run_node
from tests.support.quick_answer_conversation import (
    QUESTION,
    TRANSITION,
    confirm_source_and_continue,
    conversation as conversation,
    prompts,
    reading,
    typed_turn,
    wait_for_message,
)

APP_DIR = Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver" / "pi_copilot" / "node_app"
REVIEW_QUESTION = "Could you summarize typical peak lactate in the opening six hours?"


def test_the_transition_offers_the_typed_question_and_its_card_follows_that_turn(conversation):
    service, record, gateway = conversation
    typed_turn(service, record, QUESTION)
    job_id = confirm_source_and_continue(service, record)

    [prompt] = prompts(gateway)
    assert prompt["intent"] == "advance_after_data_source_confirmation"
    assert prompt["quick_question"] == QUESTION
    [result] = gateway.submitted
    assert result["status"] == "ok", result
    public = service.get_session(record.session_id, project_id=record.project_id)["session"]
    [card] = public["quick_answers"]
    assert card["message_job_id"] == job_id == public["last_message_job_id"]
    assert card["proposal"]["status"] == "approval_required", card["proposal"]["reason"]
    assert card["proposal"]["expand"]["reading"]["question"] == QUESTION
    # The same card as a typed quick question: the researcher's click, the
    # displayed digest, numbers from host evidence only.
    answer = service.approve_quick_answer(
        record.session_id, project_id=record.project_id, quick_answer_id=card["quick_answer_id"],
        card_sha256=card["proposal"]["card"]["sha256"],
    )["quick_answer"]["answer"]
    assert answer["status"] == "ok", answer["reason"]
    assert next(row["value"] for row in answer["rows"] if row["statistic"] == "median") == 4


@pytest.mark.parametrize("field,value,replacement", [
    ("measure.concept", "icu_stay_hours", "los_icu"),
    ("grouping[0]", "first_icu_stay", "sex"),
    ("measure.concept", "quartz_metric_zz", "lact"),
])
@pytest.mark.parametrize("repaired", [True, False], ids=["qualified", "full-study"])
def test_unregistered_concepts_get_one_re_read(conversation, field, value, replacement, repaired):
    service, record, gateway = conversation
    question = QUESTION
    measure = reading()["measure"]
    if value == "icu_stay_hours":
        question = "在 eICU demo 中，ICU 住院时长的中位数是多少？"
        measure = {"concept": "los_icu", "statistic": "median"}
        study = study_contexts.upsert_context({"id": record.binding.study_context_id, "question": question})
        record.binding = service._binding_for_context(study)
        service._save_record(record)
    typed_turn(service, record, question)

    def fields_for(concept):
        if field == "grouping[0]":
            return {"grouping": [concept]}
        return {"measure": {**measure, "concept": concept}}

    gateway.initial_fields = fields_for(value)
    gateway.repair_fields = fields_for(replacement if repaired else value)
    job_id = confirm_source_and_continue(service, record)

    first, second = gateway.submitted
    assert first["status"] == "ok" and first["code"] == "quick_reading_concept_unregistered"
    assert first["details"] == {"reading_issues": [{
        "field": field, "received": value, "message": "not a registered concept id", "candidates": [],
    }]}
    [card] = service.get_session(record.session_id, project_id=record.project_id)["session"]["quick_answers"]
    assert card["message_job_id"] == job_id
    proposal = card["proposal"]
    assert second["details"]["quick_answer"]["proposal"] == proposal
    assert proposal["expand"]["reading"]["question"] == question
    assert proposal["expand"]["action"] == "expand_into_full_study"
    if repaired:
        assert proposal["status"] == "approval_required" and second["code"] == "quick_qualified"
        assert proposal["card"]["approval_action"] == "approve_quick_answer"
        plan = proposal["card"]["plan"]
        if field == "grouping[0]":
            assert plan["group_column"] == replacement
        else:
            assert plan["effective_reading"]["measure"]["concept"] == replacement
        if replacement == "los_icu":
            answer = service.approve_quick_answer(
                record.session_id, project_id=record.project_id, quick_answer_id=card["quick_answer_id"],
                card_sha256=proposal["card"]["sha256"],
            )["quick_answer"]["answer"]
            median = next(row for row in answer["rows"] if row["statistic"] == "median")
            assert (median["value"], median["unit"]) == (1.5, "days")
    else:
        assert proposal["status"] == "full_study" and second["code"] == "quick_reading_concept_unregistered"
        assert proposal["reading_issues"] == first["details"]["reading_issues"]
        assert proposal["capability"] is None and proposal["card"] is None
        handoff = service.expand_quick_answer(
            record.session_id, project_id=record.project_id, quick_answer_id=card["quick_answer_id"],
        )
        assert handoff["expand"]["reading"]["question"] == question


@pytest.mark.parametrize("offered", [True, False], ids=["source-confirmation", "direct"])
@pytest.mark.parametrize("repair_fields", [None, {"measure": {"statistic": "typical"}}], ids=["declined", "invalid"])
def test_a_re_read_without_a_valid_submission_leaves_a_handoff(conversation, offered, repair_fields):
    service, record, gateway = conversation
    configure_reading_repair(service, record, gateway, offered=offered)
    gateway.repair_fields = repair_fields
    if offered:
        job_id = confirm_source_and_continue(service, record)
    else:
        job_id = wait_for_message(service.send_message(
            record.session_id, project_id=record.project_id, message=REVIEW_QUESTION, message_origin="typed",
        ))
    codes = ["quick_reading_concept_unregistered"]
    if repair_fields is not None:
        codes.append("quick_reading_invalid")
    assert [result["code"] for result in gateway.submitted] == codes
    [card] = service.get_session(record.session_id, project_id=record.project_id)["session"]["quick_answers"]
    assert card["message_job_id"] == job_id
    proposal = card["proposal"]
    assert (proposal["status"], proposal["reason"]) == ("full_study", "quick_reading_concept_unregistered")
    assert proposal["expand"]["action"] == "expand_into_full_study"
    assert proposal["expand"]["reading"]["question"] == REVIEW_QUESTION


def configure_reading_repair(service, record, gateway, *, offered):
    if offered:
        study = study_contexts.upsert_context({"id": record.binding.study_context_id, "question": REVIEW_QUESTION})
        record.binding = service._binding_for_context(study)
        service._save_record(record)
        typed_turn(service, record, REVIEW_QUESTION)
    else:
        gateway.call_quick_answer_anyway = True
    measure = {**reading()["measure"], "window": {"start_hours": 0, "end_hours": 6}}
    gateway.initial_fields = {"measure": {**measure, "concept": "quartz_metric_zz"}}
    gateway.repair_fields = {"measure": measure}


@pytest.mark.parametrize("offered", [True, False], ids=["source-confirmation", "direct"])
def test_a_re_read_that_stops_does_not_write_the_original_handoff(conversation, monkeypatch, offered):
    from easyicu.webserver.pi_copilot import quick_answers
    from easyicu.webserver.quick_answer import QuickStop

    service, record, gateway = conversation
    configure_reading_repair(service, record, gateway, offered=offered)
    original = quick_answers.prepare_registered_question
    attempts = []

    def prepare(reading, **kwargs):
        attempts.append(reading.measure.concept)
        if reading.measure.concept == "lact":
            raise QuickStop("quick_qualifying_provider_required")
        return original(reading, **kwargs)

    monkeypatch.setattr(quick_answers, "prepare_registered_question", prepare)
    if offered:
        confirm_source_and_continue(service, record)
    else:
        wait_for_message(service.send_message(
            record.session_id, project_id=record.project_id, message=REVIEW_QUESTION, message_origin="typed",
        ))
    assert [result["code"] for result in gateway.submitted] == [
        "quick_reading_concept_unregistered", "quick_qualifying_provider_required",
    ]
    assert service.get_session(record.session_id, project_id=record.project_id)["session"]["quick_answers"] == []
    assert attempts == ["quartz_metric_zz", "lact"]


def invalidate_repair(context, change):
    if change == "study":
        study_contexts.upsert_context({
            "id": context.session.binding.study_context_id, "question": "What was the first recorded lactate?",
        })
    elif change == "authority":
        context.invalidate_authority("authority_changed_in_test")
    elif change == "source":
        sources.load_registry()["sources"][0]["id"] = "src_replaced"


@pytest.mark.parametrize("change", ["study", "authority", "source", "saved-binding"])
@pytest.mark.parametrize("model_submits", [True, False], ids=["resubmitted", "replayed"])
def test_a_stale_re_read_leaves_no_handoff_and_the_message_finishes(conversation, monkeypatch, change, model_submits):
    service, record, gateway = conversation
    configure_reading_repair(service, record, gateway, offered=True)
    submit = service._submit_quick_answer
    attempts = []

    def track_repair(context, reading, *, message_job_id):
        try:
            return submit(context, reading, message_job_id=message_job_id)
        except PiCopilotError as exc:
            attempts.append((reading.measure.concept, exc.code))
            raise

    def after_initial_reading(context):
        if change != "saved-binding":
            invalidate_repair(context, change)
            return

        scoped_record = service._scoped_record

        def rebind_then_load(session_id, *, project_id):
            # Change the saved binding between the turn's freshness check and
            # the owner's scoped read. The new binding itself is current.
            monkeypatch.setattr(service, "_scoped_record", scoped_record)
            study = study_contexts.upsert_context({
                "id": context.session.binding.study_context_id,
                "question": "What was the last recorded lactate?",
            })
            latest = scoped_record(session_id, project_id=project_id)
            latest.binding = service._binding_for_context(study)
            service._save_record(latest)
            refreshed = scoped_record(session_id, project_id=project_id)
            assert refreshed.binding != context.session.binding
            assert not service._stale_details(refreshed)["stale"]
            monkeypatch.setattr(service, "_quick_study", lambda _record: pytest.fail("accepted a changed binding"))
            return refreshed

        monkeypatch.setattr(service, "_scoped_record", rebind_then_load)

    monkeypatch.setattr(service, "_submit_quick_answer", track_repair)
    gateway.after_initial_reading = after_initial_reading
    if not model_submits:
        gateway.repair_fields = None
    confirm_source_and_continue(service, record)

    code = "quick_source_changed" if change == "source" else "pi_session_authority_stale"
    expected_tool_codes = ["quick_reading_concept_unregistered"]
    if model_submits:
        expected_tool_codes.append(code)
    assert [result["code"] for result in gateway.submitted] == expected_tool_codes
    # A stale StudyContext or invalidated turn stops in the tool dispatcher;
    # replacement sources and saved bindings reach the owner on submission.
    owner_received_reread = model_submits and change in {"source", "saved-binding"}
    assert attempts == [("lact" if owner_received_reread else "quartz_metric_zz", code)]
    assert not list(service._quick_store().directory(record.session_id).glob("qa_*/record.json"))
    assert service.get_session(record.session_id, project_id=record.project_id)["session"]["quick_answers"] == []


@pytest.mark.parametrize("change,code", [
    ("source", "quick_source_changed"),
    ("study", "pi_session_authority_stale"),
    ("authority", "pi_session_authority_stale"),
])
def test_a_declined_direct_re_read_with_stale_authority_reports_the_stop(conversation, change, code):
    service, record, gateway = conversation
    configure_reading_repair(service, record, gateway, offered=False)
    gateway.repair_fields = None
    gateway.after_initial_reading = lambda context: invalidate_repair(context, change)
    wait_for_message(service.send_message(
        record.session_id, project_id=record.project_id, message=REVIEW_QUESTION, message_origin="typed",
    ), error_code=code)

    assert [result["code"] for result in gateway.submitted] == ["quick_reading_concept_unregistered"]
    assert service._get_record(record.session_id).last_turn_status == "failed"
    assert not list(service._quick_store().directory(record.session_id).glob("qa_*/record.json"))


def test_a_registered_but_unassessed_concept_does_not_get_a_re_read(conversation, monkeypatch):
    from easyicu.concept.loader import load_concept_dict_cached
    from easyicu.webserver.quick_answer import preparation

    service, record, gateway = conversation
    concept = "feurea"
    assert concept in load_concept_dict_cached()
    gateway.initial_fields = {"measure": {"concept": concept, "statistic": "median"}}
    monkeypatch.setattr(preparation, "acquire_universe_for_question", lambda **_kwargs: pytest.fail("allocated rows"))
    typed_turn(service, record, QUESTION)
    confirm_source_and_continue(service, record)
    [result] = gateway.submitted
    proposal = result["details"]["quick_answer"]["proposal"]
    assert (proposal["status"], proposal["reason"]) == ("stopped", "quick_capability_unassessed")
    assert proposal["capability"]["database"] == "eicu_demo"
    assert proposal["capability"]["unassessed_concepts"] == [concept]
    assert proposal["expand"]["action"] == "expand_into_full_study"


def test_a_study_question_is_offered_alike_and_a_declined_offer_leaves_no_card(conversation):
    # The host classifies no text: a study question is offered the same way,
    # the model submits nothing, and the page keeps its automatic plan.
    service, record, gateway = conversation
    study = "在 eICU demo 中，校正年龄后入 ICU 第一天最高乳酸与院内死亡有关吗？"
    study_contexts.upsert_context({"id": record.binding.study_context_id, "question": study})
    record = service._get_record(record.session_id)
    record.binding = service._binding_for_context(study_contexts.get_context(record.binding.study_context_id))
    service._save_record(record)
    gateway.declines = True
    typed_turn(service, record, study)
    confirm_source_and_continue(service, record)

    [prompt] = prompts(gateway)
    assert prompt["quick_question"] == study
    assert gateway.submitted == []
    assert service.get_session(record.session_id, project_id=record.project_id)["session"]["quick_answers"] == []


@pytest.mark.parametrize("typed,origin", [
    # The setup kept only part of what the researcher typed.
    ("你好，我想问一下：" + QUESTION + "谢谢。", "typed"),
    # The words are a clicked model option, not typed by the researcher.
    (QUESTION, "model_option"),
])
def test_a_saved_question_that_is_not_a_typed_message_is_not_offered(conversation, typed, origin):
    service, record, gateway = conversation
    gateway.call_quick_answer_anyway = True
    typed_turn(service, record, typed, origin=origin)
    confirm_source_and_continue(service, record)

    [prompt] = prompts(gateway)
    assert "quick_question" not in prompt
    [result] = gateway.submitted
    assert (result["status"], result["code"]) == ("blocked", "quick_question_unbound")
    public = service.get_session(record.session_id, project_id=record.project_id)["session"]
    assert public["quick_answers"] == []


def test_the_question_binding_is_the_transition_hosts_alone(conversation):
    service, record, _ = conversation
    typed_turn(service, record, QUESTION)
    context = ToolExecutionContext(session=record, user_message=TRANSITION)
    # Superseded or host-written turns never bind a quick question.
    context.bind_question_wording_context([
        {"text": QUESTION, "message_origin": "typed", "superseded": True},
        {"text": "", "message_origin": "host_transition"},
    ], "")
    assert service._transition_quick_question(context) == ""
    context.bind_question_wording_context([
        {"text": QUESTION, "message_origin": "typed"}, {"text": "", "message_origin": "host_transition"},
    ], "")
    assert service._transition_quick_question(context) == QUESTION
    pending = record.model_copy(deep=True)
    pending.data_source_authorization.status = "pending"
    assert service._transition_quick_question(ToolExecutionContext(
        session=pending, user_turns=context.user_turns,
    )) == ""


def submit(service, record, request: dict[str, Any], *, quick_question: str, job_id: str = "job-transition",
           offered: bool = False):
    service.replay_store.start_turn(session_id=record.session_id, project_id=record.project_id,
                                    job_id=job_id, allowed_actions=[], message_origin="host_transition")
    context = ToolExecutionContext(
        session=record, user_message=TRANSITION,
        data_handling=service._data_handling_for_session(record),
        quick_answer_submit=lambda ctx, read: service._submit_quick_answer(ctx, read, message_job_id=job_id),
        authority_validator=lambda binding: service._binding_stale_details(binding, project_id=record.project_id),
    )
    context.quick_question = quick_question
    context.quick_question_offered = offered
    try:
        return execute_tool("easyicu_quick_answer", {"reading": request}, context)
    finally:
        service.replay_store.finish_turn(session_id=record.session_id, project_id=record.project_id,
                                         job_id=job_id, status="done")


def test_a_reading_that_rewords_the_researchers_question_is_not_qualified(conversation):
    service, record, _ = conversation
    reworded = submit(service, record, reading("eICU demo 首日乳酸峰值的中位数"), quick_question=QUESTION)
    proposal = reworded["details"]["quick_answer"]["proposal"]
    assert (proposal["status"], proposal["reason"]) == ("full_study", "quick_reading_ambiguous")
    assert proposal["card"] is None
    # The handoff keeps the researcher's words, never the model's.
    assert proposal["expand"]["reading"]["question"] == QUESTION
    assert "question_mismatch" in proposal["expand"]["reading"]["ambiguities"]
    with pytest.raises(PiCopilotError, match="quick_question_unbound"):
        submit(service, record, reading(), quick_question="", job_id="job-unbound")


def test_an_offered_question_that_needs_a_full_study_adds_no_card(conversation):
    # 11-03 eICU walk: the adjusted OR question was offered, the host found
    # it needs a full study, and a 「改为完整研究」 card repeated what the
    # conversation was already doing.  The offer now ends without a card.
    service, record, _ = conversation
    adjusted = {**reading(), "adjustments": ["age", "sex"]}
    result = submit(service, record, adjusted, quick_question=QUESTION, offered=True)
    assert (result["status"], result["code"]) == ("ok", "quick_full_study_continues")
    assert "quick_answer" not in result["details"]
    assert service.get_session(record.session_id, project_id=record.project_id)["session"]["quick_answers"] == []
    # A question typed in the conversation keeps its card and remedy.
    typed = submit(service, record, adjusted, quick_question=QUESTION, job_id="job-typed")
    proposal = typed["details"]["quick_answer"]["proposal"]
    assert (proposal["status"], proposal["reason"]) == ("full_study", "quick_requires_adjusted_study")


# The Node side: the transition's host reply and its one tool call.

SOURCE = {"source_id": "src_synthetic", "label": "eICU-CRD Demo", "database": "eicu_demo"}


def _transition(*, offer: bool, question: str = QUESTION) -> str:
    workflow = {"next_action_code": "provider_ready_to_generate_plan",
                "study_setup_receipt": {"configuration": {"data_source": SOURCE, "question": question}}}
    receipts = json.dumps([{"status": "ok", "code": "easyicu_research_workflow_projected",
                            "details": {"workflow": workflow}}], ensure_ascii=False)
    tail = "\\n\\n[EASYICU_HOST_TRANSITION_V1]\\n[EASYICU_DATA_SOURCE_CONFIRMED_V1]\\n" + (
        "[EASYICU_QUICK_ANSWER_OFFER_V1]\\n" if offer else "") + "The host has confirmed the study data source."
    return (f"{{ role: 'user', content: [{{ type: 'text', text: {json.dumps(TRANSITION)}"
            f" + OWNER_CONTEXT_MARKER + {json.dumps(receipts)} + \"{tail}\" }}] }}")


def _settled(messages_js: str) -> dict[str, Optional[str]]:
    node = shutil.which("node")
    if not node or not (APP_DIR / "node_modules").is_dir():
        pytest.skip("Pinned Pi Node runtime is unavailable")
    module = (APP_DIR / "src" / "post-tool-finalization.mjs").as_uri()
    script = f"""
      import {{ hostPostToolFinalization, quickAnswerOfferReply, OWNER_CONTEXT_MARKER }} from {json.dumps(module)};
      const model = {{ api: 'openai-completions', provider: 'test', id: 'test' }};
      const messages = {messages_js};
      const stream = hostPostToolFinalization(model, {{ messages }}, 'zh');
      const host = stream ? (await stream.result()).content[0].text : null;
      console.log(JSON.stringify({{ host, offer: quickAnswerOfferReply({{ messages }}, 'zh') }}));
    """
    completed = run_node(node, script, module=True, cwd=APP_DIR, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


QUESTION_TURNS = f"""
  {{ role: 'user', content: [{{ type: 'text', text: {json.dumps(QUESTION)} }}] }},
  {{ role: 'assistant', content: [{{ type: 'text', text: '研究问题已保存。' }}] }},
"""


def _tool_result(*, error: bool) -> str:
    details = ({"status": "blocked", "code": "quick_bound_source_unavailable"} if error else
               {"status": "ok", "details": {"quick_answer": {"proposal": {"status": "approval_required"}}}})
    return f"""
      {{ role: 'assistant', content: [{{ type: 'text', text: '已确认使用 eICU-CRD Demo 数据。' }},
        {{ type: 'toolCall', id: 'call-quick', name: 'easyicu_quick_answer', arguments: {{}} }}] }},
      {{ role: 'toolResult', toolCallId: 'call-quick', toolName: 'easyicu_quick_answer',
        isError: {json.dumps(error)}, content: [], details: {json.dumps(details)} }},
    """


def test_the_offer_lets_the_model_decide_and_the_host_writes_the_reply():
    # Without an offer the host acknowledges the source before any model call.
    assert _settled(f"[{QUESTION_TURNS}{_transition(offer=False)}]") == {
        "host": "已确认使用 eICU-CRD Demo 数据。", "offer": "",
    }
    # With one, the model's call decides; the acknowledgement replaces its text.
    assert _settled(f"[{QUESTION_TURNS}{_transition(offer=True)}]") == {
        "host": None, "offer": "已确认使用 eICU-CRD Demo 数据。",
    }
    # A setup without a question keeps today's model reply asking for it.
    assert _settled(f"[{_transition(offer=True, question='')}]") == {"host": None, "offer": ""}


@pytest.mark.parametrize("error", [False, True], ids=["card", "refused"])
def test_the_offer_ends_after_its_one_tool_call(error):
    settled = _settled(f"[{QUESTION_TURNS}{_transition(offer=True)},{_tool_result(error=error)}]")
    assert settled == {"host": "", "offer": ""}


def test_only_the_transition_may_offer_and_only_the_quick_answer_tool():
    main = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")
    prompt_session = main.split("async function promptSession", 1)[1].split("function regenerateTarget", 1)[0]
    assert '"quick_question"]), "pi_prompt_invalid")' in prompt_session
    assert 'intent !== "advance_after_data_source_confirmation")) {' in prompt_session
    assert 'if (quickQuestion) record.session.setActiveToolsByName(["easyicu_quick_answer"]);' in prompt_session
    regenerate = main.split("async function regenerateSession", 1)[1].split("async function handleRequest", 1)[0]
    assert "quick_question" not in regenerate
    transition = main.split(': turnIntent === "advance_after_data_source_confirmation"', 1)[1].split(
        ": turnIntent ===", 1)[0]
    assert "${QUICK_ANSWER_OFFER_MARKER}" in transition
    assert "call easyicu_quick_answer with a QuickReading whose question is exactly that text" in transition
    assert "one corrected reading only if the host requests it" in transition
    assert "Never calculate or state a value" in transition


def test_a_source_that_is_not_a_registered_export_is_not_offered(conversation, monkeypatch):
    # A raw folder chosen for the study has no registered export, so every
    # quick answer on it would stop; the transition offers none instead.
    service, record, _ = conversation
    typed_turn(service, record, QUESTION)
    context = ToolExecutionContext(session=record, user_message=TRANSITION)
    context.bind_question_wording_context([
        {"text": QUESTION, "message_origin": "typed"}, {"text": "", "message_origin": "host_transition"},
    ], "")
    assert service._transition_quick_question(context) == QUESTION
    monkeypatch.setattr(sources, "registered_source_identity", lambda path: None)
    assert service._transition_quick_question(context) == ""
