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
question to it, the conversation's model may submit one reading, and the host
qualifies it into the same digest-bound card the researcher clicks.  A reading
that rewords the question is not qualified.  A study question, a stopped
proposal or no call keep today's acknowledgement and automatic plan: the page
side is in tests/js/guided_pi_quick_answers.test.js and the Node settlement of
the model's decision in tests/js/provider_transient_retry.test.js.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest

from easyicu.extensions import ExtensionRegistry
from easyicu.webserver import settings, sources, study_contexts
from easyicu.webserver.pi_copilot import service as service_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding, PiCopilotError, PiSessionDataSourceAuthorization, PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.tools import execute_tool
from tests.support.native_outcome_export import typed_native_export
from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway

APP_DIR = Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver" / "pi_copilot" / "node_app"
QUESTION = "在 eICU demo 中，入 ICU 第一天最高乳酸的中位数是多少？"
TRANSITION = "EasyICU 已确认所选数据来源，请在当前对话中继续。"


def reading(question: str = QUESTION) -> dict[str, Any]:
    return {"question": question, "intent": "describe", "population": {"criteria": []},
            "measure": {"concept": "lact", "summary": "max", "statistic": "median",
                        "window": {"start_hours": 0, "end_hours": 24}}}


class ModelGateway(FakeGateway):
    """The conversation's model: submits a reading of the offered question."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[dict[str, Any]] = []
        self.call_quick_answer_anyway = False
        self.declines = False

    def request(self, method: str, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        response = super().request(method, params, **kwargs)
        context = kwargs.get("tool_context")
        offered = params.get("quick_question")
        if self.declines:
            return response
        if method == "session.prompt" and context is not None and (offered or self.call_quick_answer_anyway):
            # The sidecar hands a host refusal back to the model as a tool error.
            try:
                self.submitted.append(execute_tool(
                    "easyicu_quick_answer", {"reading": reading(offered or QUESTION)}, context,
                ))
            except PiCopilotError as exc:
                self.submitted.append({"status": "blocked", "code": exc.code})
        return response


@pytest.fixture
def conversation(tmp_path, monkeypatch):
    export = typed_native_export(
        tmp_path / "synthetic", database="eicu_demo",
        longitudinal=pd.DataFrame({"stay_id": range(60), "charttime": [1.0] * 60,
                                   "lact": [float(i % 15 + 1) / 2 for i in range(60)]}),
        longitudinal_concepts=["lact"],
        statics=pd.DataFrame({"stay_id": range(60), "age": [45] * 60, "sex": ["Male"] * 30 + ["Female"] * 30}),
        static_concepts=["age", "sex"],
        outcome=pd.DataFrame({"stay_id": range(60), "charttime": [0.0] * 60, "icu_death": [False] * 30 + [True] * 30}),
        outcome_concepts=["icu_death"],
    )
    registry = {"sources": [{"id": "src_synthetic", "database": "eicu_demo", "path": str(export), "ok": True}]}
    monkeypatch.setattr(sources, "load_registry", lambda: registry)
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True, "language": "zh"})
    study = study_contexts.upsert_context({"id": "study-transition", "question": QUESTION,
                                         "data_source": {"database": "eicu_demo", "path": str(export)}})
    gateway = ModelGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway,
                              extension_registry=ExtensionRegistry(tmp_path / "extensions"))
    service.project_store.bind("project-transition", study["id"])
    record = PiSessionRecord(
        session_id="pi-transition", project_id="project-transition", external_llm_opt_in=True,
        credentialed_data_in_history=False,
        binding=AuthorityBinding(study_context_id=study["id"], study_revision=study["revision"]),
        data_source_authorization=PiSessionDataSourceAuthorization(
            status="confirmed", confirmation_mode="select_registered_source",
        ),
    )
    service._save_record(record)
    yield service, record, gateway
    service.close()


def typed_turn(service, record, text: str, *, job_id: str = "job-question", origin: str = "typed") -> None:
    service.replay_store.start_turn(session_id=record.session_id, project_id=record.project_id,
                                    job_id=job_id, allowed_actions=["configure"],
                                    user_message=text, message_origin=origin)
    service.replay_store.finish_turn(session_id=record.session_id, project_id=record.project_id,
                                     job_id=job_id, status="done", assistant_reply="研究问题已保存。")


def confirm_source_and_continue(service, record) -> str:
    submitted = service.send_message(
        record.session_id, project_id=record.project_id, message=TRANSITION,
        message_intent="advance_after_data_source_confirmation", message_origin="host_prompt",
    )
    deadline = time.monotonic() + 30
    job = None
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running" and job.status != "queued":
            break
        time.sleep(0.02)
    assert job is not None and job.status == "done", job and (job.status, getattr(job, "error", None))
    return submitted["job_id"]


def prompts(gateway) -> list[dict[str, Any]]:
    return [params for method, params, _ in gateway.calls if method == "session.prompt"]


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


def submit(service, record, request: dict[str, Any], *, quick_question: str, job_id: str = "job-transition"):
    service.replay_store.start_turn(session_id=record.session_id, project_id=record.project_id,
                                    job_id=job_id, allowed_actions=[], message_origin="host_transition")
    context = ToolExecutionContext(
        session=record, user_message=TRANSITION,
        data_handling=service._data_handling_for_session(record),
        quick_answer_submit=lambda ctx, read: service._submit_quick_answer(ctx, read, message_job_id=job_id),
        authority_validator=lambda binding: service._binding_stale_details(binding, project_id=record.project_id),
    )
    context.quick_question = quick_question
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
    assert "call easyicu_quick_answer exactly once with a QuickReading whose question is exactly that text" in transition
    assert "Never calculate or state a value" in transition
