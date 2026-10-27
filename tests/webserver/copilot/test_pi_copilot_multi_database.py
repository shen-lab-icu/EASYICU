"""Synthetic conversation admission and reload contracts, without ASGI or a model."""
from __future__ import annotations

import json
import time

import pytest

from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import research_multi_actions as actions, sources, state_paths
from easyicu.webserver.pi_copilot import tools
from easyicu.webserver.pi_copilot.contracts import PiCopilotError, ToolExecutionContext
from easyicu.webserver.pi_copilot.replay_store import PiConversationReplayStore
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.research_multi_contracts import DatabaseTarget, MultiDatabaseError, MultiDatabaseRequest
from tests.webserver.copilot.pi_copilot_contract_fixtures import (
    FakeGateway, allow_unrelated_message_test, study_state as _study_state,
)

study_state = _study_state
TOOL = "easyicu_propose_multi_database"


class Gateway(FakeGateway):
    on_prompt = None

    def request(self, method, params, **kwargs):
        response = super().request(method, params, **kwargs)
        if method == "session.prompt" and self.on_prompt:
            self.on_prompt(kwargs["tool_context"])
        return response


@pytest.fixture
def conversation(tmp_path, monkeypatch, study_state):
    study_state["data_source"] = {"database": "miiv_demo"}
    monkeypatch.setattr(state_paths, "state_root", lambda: tmp_path / "state")
    rows = [dict(id="fixture_" + db, database=db, path=str(tmp_path / db), ok=True)
            for db in ("aumc", "sic")]
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": rows})
    monkeypatch.setattr(sources, "registered_source_identity", lambda path: next(
        (SourceIdentity(database=row["database"], registration_id=row["id"])
         for row in rows if row["path"] == path), None))
    gateway = Gateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    session = service.create_session(project_id="project", study_context_id="study-test",
                                     external_llm_opt_in=True)["session"]
    allow_unrelated_message_test(service, session["session_id"])
    dispatched = []

    def dispatch(current, request, batch_id, phase):
        dispatched.append((request, batch_id, phase))
        return dict(job_id="batch-job", kind="agent-run", status="queued", multi_database_batch_id=batch_id)

    monkeypatch.setattr(actions, "_dispatch", dispatch)
    yield service, gateway, session["session_id"], dispatched, rows
    service.close()


def request(session_id):
    return MultiDatabaseRequest(project_id="project", session_id=session_id, study_context_id="study-test",
                                targets=(DatabaseTarget(database="aumc"),))


def send(conversation, text, *, origin="typed"):
    from easyicu.webserver import jobs

    service, _, session_id, _, _ = conversation
    receipt = service.send_message(session_id, project_id="project", message=text, message_origin=origin)
    deadline = time.monotonic() + 5
    job = jobs.MANAGER.get(receipt["job_id"])
    while job.status == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.status == "done", job.snapshot()
    return receipt


def test_proposal_survives_service_reopen_in_scoped_session_replay(conversation):
    service, _, session_id, dispatched, _ = conversation
    receipt = actions.propose(service, request(session_id))
    batch_id = receipt["multi_database_batch_id"]
    assert len(batch_id) == 64 and dispatched[0][2] == "prepare"
    service.close()
    reopened = PiCopilotService(store_path=service.store_path, gateway=Gateway())
    try:
        replay = reopened.get_session(session_id, project_id="project")["session"]["conversation_replay"]
        reference, = replay["multi_database_batches"]
        assert reference["batch_id"] == batch_id
        assert reference["study_context_id"] == "study-test"
        assert reference["message_job_id"] is None
        assert reference["created_at"]
        assert actions.status(reopened, project_id="project", session_id=session_id,
                              batch_id=batch_id) == {"batch_id": batch_id, "status": "preparing"}
        for scope in (dict(project_id="neighbour", session_id=session_id),
                      dict(project_id="project", session_id="other")):
            with pytest.raises(MultiDatabaseError, match="multi_database_study_mismatch"):
                actions.status(reopened, batch_id=batch_id, **scope)
        with pytest.raises(PiCopilotError):
            reopened.get_session(session_id, project_id="neighbour")
        with pytest.raises(PiCopilotError, match="another project or session"):
            reopened.replay_store.snapshot(session_id=session_id, project_id="neighbour")
        assert reopened.replay_store.snapshot(session_id="other", project_id="project")["multi_database_batches"] == []
    finally:
        reopened.close()


def test_typed_model_tool_prepares_a_card_and_never_approves(conversation):
    service, gateway, session_id, dispatched, _ = conversation
    results = []
    gateway.on_prompt = lambda context: results.append(tools.execute_tool(
        TOOL, {"targets": [{"database": "aumc"}, {"database": "sic"}]}, context))
    receipt = send(conversation, "Repeat this study in the other registered databases.")
    assert results[0]["code"] == "multi_database_review_preparing"
    assert results[0]["details"]["approval_required"] is True
    assert len(dispatched) == 1 and dispatched[0][2] == "prepare"
    proposed, batch_id, _ = dispatched[0]
    assert proposed.project_id == "project" and proposed.session_id == session_id
    assert proposed.study_context_id == "study-test"
    assert [row.source_id for row in proposed.targets] == ["fixture_aumc", "fixture_sic"]
    reference, = service.get_session(session_id, project_id="project")["session"]["conversation_replay"]["multi_database_batches"]
    assert reference["batch_id"] == batch_id
    assert reference["message_job_id"] == receipt["job_id"]
    assert not (actions.batch_root(batch_id) / "approval.json").exists()


@pytest.mark.parametrize("text", ["yes", "在 MIMIC-IV 里也跑一下亚组分析"])
def test_ordinary_text_goes_to_the_conversation_without_host_batch_routing(conversation, text):
    _, gateway, _, dispatched, _ = conversation
    send(conversation, text)
    prompts = [params for method, params, _ in gateway.calls if method == "session.prompt"]
    assert [row["message"] for row in prompts] == [text]
    assert dispatched == []


@pytest.mark.parametrize("origin", ["model_option", "host_transition", "unknown", None])
def test_only_the_current_typed_turn_can_request_a_batch(conversation, origin):
    _, gateway, _, dispatched, _ = conversation
    send(conversation, "Earlier typed text.")
    failures = []

    def invoke(context):
        with pytest.raises(PiCopilotError) as error:
            tools.execute_tool(TOOL, {"targets": [{"database": "aumc"}]}, context)
        failures.append(error.value.code)

    gateway.on_prompt = invoke
    send(conversation, "Repeat this study.", origin=origin)
    assert failures == ["multi_database_typed_turn_required"]
    assert dispatched == []


@pytest.mark.parametrize("targets,code", [
    ([], "pi_tool_arguments_required"),
    ([{"database": "unknown"}], "multi_database_request_invalid"),
    ([{"database": "aumc"}, {"database": "aumc", "source_id": "other"}], "multi_database_request_invalid"),
    ([{"database": "aumc"}, {"database": "aumc"}], "multi_database_request_invalid"),
    ([{"database": db} for db in ("miiv", "mimic", "eicu", "hirid", "aumc", "sic", "miiv_demo")], "multi_database_request_invalid"),
    ([{"database": "hirid"}], "multi_database_source_unavailable"),
    ([{"database": "aumc", "source_id": "not_registered"}], "multi_database_source_unavailable"),
    ([{"database": "aumc", "path": "/private/source"}], "multi_database_request_invalid"),
])
def test_invalid_targets_never_dispatch(conversation, targets, code):
    _, gateway, _, dispatched, rows = conversation
    if len(targets) > 6:
        rows.extend(dict(id="fixture_" + item["database"], database=item["database"],
                         path="synthetic-" + item["database"], ok=True)
                    for item in targets if item["database"] not in {row["database"] for row in rows})
    failures = []

    def invoke(context):
        with pytest.raises(PiCopilotError) as error:
            tools.execute_tool(TOOL, {"targets": targets}, context)
        failures.append(error.value.code)

    gateway.on_prompt = invoke
    send(conversation, "Repeat this study in selected databases.")
    assert failures == [code]
    assert dispatched == []


def test_tool_requires_a_bound_study(conversation):
    service, _, session_id, dispatched, _ = conversation
    record = service._get_record(session_id)
    record.binding = record.binding.model_copy(update={"study_context_id": None, "study_context_revision": None})
    context = ToolExecutionContext(session=record, user_message="Repeat this study.")
    with pytest.raises(PiCopilotError) as error:
        tools.execute_tool(TOOL, {"targets": [{"database": "aumc"}]}, context)
    assert error.value.code == "multi_database_study_required"
    assert dispatched == []


def test_batch_reference_outlives_turn_pagination_and_retention(conversation):
    service, _, session_id, _, _ = conversation
    batch_id = actions.propose(service, request(session_id))["multi_database_batch_id"]
    store = service.replay_store
    for index in range(260):
        store.start_turn(session_id=session_id, project_id="project", job_id=f"turn-{index}", allowed_actions=[])
    reopened = PiConversationReplayStore(store.root)
    replay = reopened.snapshot(session_id=session_id, project_id="project", limit=1)
    assert len(replay["turns"]) == 1
    assert [row["batch_id"] for row in replay["multi_database_batches"]] == [batch_id]
    assert "path" not in json.dumps(replay["multi_database_batches"])


def test_batch_reference_is_idempotent_and_regeneration_cannot_erase_it(conversation):
    service, _, session_id, _, _ = conversation
    send(conversation, "Describe the planned comparison.")
    actions.propose(service, request(session_id))
    store = service.replay_store
    before = store.snapshot(session_id=session_id, project_id="project")
    reference, = before["multi_database_batches"]
    store.record_multi_database_batch(session_id=session_id, project_id="project", reference=reference)
    assert store.snapshot(session_id=session_id, project_id="project")["replay_sha256"] == before["replay_sha256"]
    store.supersede_from_turn_index(session_id=session_id, project_id="project", turn_index=0)
    after = store.snapshot(session_id=session_id, project_id="project")
    assert after["turns"] == [] and after["multi_database_batches"] == [reference]


def test_unrecorded_batch_never_starts_preparation(conversation, monkeypatch):
    service, _, session_id, dispatched, _ = conversation

    def unavailable(**kwargs):
        raise OSError("synthetic replay unavailable")

    monkeypatch.setattr(service.replay_store, "record_multi_database_batch", unavailable)
    with pytest.raises(OSError):
        actions.propose(service, request(session_id))
    assert dispatched == []


def test_a_conversation_with_a_batch_can_be_deleted(conversation):
    service, _, session_id, _, _ = conversation
    actions.propose(service, request(session_id))
    public = service.get_session(session_id, project_id="project")["session"]
    assert public["has_history"] is True
    assert service.delete_session(session_id, project_id="project")["deleted"] is True


def test_busy_browser_proposal_and_expired_tool_callback_cannot_dispatch(conversation):
    service, gateway, session_id, dispatched, _ = conversation

    def invoke(context):
        with pytest.raises(MultiDatabaseError, match="pi_session_busy"):
            actions.propose(service, request(session_id))

    gateway.on_prompt = invoke
    send(conversation, "Explain the next step.")
    with pytest.raises(PiCopilotError) as error:
        tools.execute_tool(TOOL, {"targets": [{"database": "aumc"}]}, gateway.tool_contexts[-1])
    assert error.value.code == "multi_database_turn_expired"
    assert dispatched == []


def test_registry_ambiguity_is_refused_and_an_explicit_source_is_preserved(conversation):
    service, gateway, session_id, dispatched, rows = conversation
    rows.append({**rows[0], "id": "second", "path": rows[0]["path"] + "-second"})

    def invoke(context):
        with pytest.raises(PiCopilotError) as error:
            tools.execute_tool(TOOL, {"targets": [{"database": "aumc"}]}, context)
        assert error.value.code == "multi_database_source_ambiguous"
        return tools.execute_tool(TOOL, {"targets": [{"database": "aumc", "source_id": "second"}]}, context)

    gateway.on_prompt = invoke
    send(conversation, "Repeat the study on the second registration.")
    assert len(dispatched) == 1 and dispatched[0][0].targets[0].source_id == "second"
    assert len(service.get_session(session_id, project_id="project")["session"]["conversation_replay"]["multi_database_batches"]) == 1


def test_six_distinct_registered_targets_are_admitted(conversation):
    _, gateway, _, dispatched, rows = conversation
    rows.extend(dict(id="fixture_" + db, database=db, path="synthetic-" + db, ok=True)
                for db in ("miiv", "mimic", "eicu", "hirid"))
    gateway.on_prompt = lambda context: tools.execute_tool(
        TOOL, {"targets": [{"database": row["database"]} for row in rows]}, context)
    send(conversation, "Prepare the study in the six selected databases.")
    assert len(dispatched[0][0].targets) == 6


def test_chat_yes_and_model_supplied_approval_never_approve(conversation, monkeypatch):
    service, gateway, session_id, dispatched, _ = conversation
    batch_id = actions.propose(service, request(session_id))["multi_database_batch_id"]
    monkeypatch.setattr(actions, "approve", lambda *a, **k: pytest.fail("chat approved a batch"))

    def invoke(context):
        with pytest.raises(PiCopilotError) as error:
            tools.execute_tool(TOOL, {"targets": [{"database": "aumc"}], "batch_id": batch_id,
                                     "card_sha256": "b" * 64, "approve": True}, context)
        assert error.value.code == "pi_tool_unknown_arguments"
        # Even if the model interprets yes as a fresh proposal, it can only
        # prepare another card. The first card remains unapproved.
        tools.execute_tool(TOOL, {"targets": [{"database": "aumc"}]}, context)

    gateway.on_prompt = invoke
    send(conversation, "yes")
    assert [phase for _, _, phase in dispatched] == ["prepare", "prepare"]
    assert all(not (actions.batch_root(batch) / "approval.json").exists() for _, batch, _ in dispatched)


def test_other_database_followup_records_the_batch_for_reload(conversation, monkeypatch):
    from easyicu.webserver import research_followups

    service, _, session_id, dispatched, _ = conversation
    monkeypatch.setattr(research_followups, "result_followups", lambda *a, **k: dict(
        source_run_sha256="a" * 64, suggestions=[dict(suggestion_id="b" * 64,
            target_action="other_database", delta={"capability_decision": {"database": "aumc"}})]))
    monkeypatch.setattr(service, "_research_run_row", lambda *a: {"study_id": "study-test"})
    receipt = actions.dispatch_followup(service, project_id="project", session_id=session_id,
        run_id="completed-run", suggestion_id="b" * 64, expected_source_sha256="a" * 64)
    reference, = service.get_session(session_id, project_id="project")["session"]["conversation_replay"]["multi_database_batches"]
    assert reference["batch_id"] == receipt["multi_database_batch_id"]
    assert reference["source_run_id"] == "completed-run"
    assert dispatched[0][2] == "prepare"


def test_a_tool_without_the_host_turn_callback_cannot_submit(conversation):
    service, _, session_id, dispatched, _ = conversation
    context = ToolExecutionContext(session=service._get_record(session_id), user_message="Repeat this study.")
    with pytest.raises(PiCopilotError) as error:
        tools.execute_tool(TOOL, {"targets": [{"database": "aumc"}]}, context)
    assert error.value.code == "multi_database_turn_required"
    assert dispatched == []


def test_empty_typed_turn_cannot_submit(conversation):
    service, _, session_id, dispatched, _ = conversation
    context = ToolExecutionContext(session=service._get_record(session_id), user_turns=[
        {"text": " ", "message_origin": "typed"}], propose_multi_database=lambda req: pytest.fail("empty turn submitted"))
    with pytest.raises(PiCopilotError) as error:
        tools.execute_tool(TOOL, {"targets": [{"database": "aumc"}]}, context)
    assert error.value.code == "multi_database_typed_turn_required"
    assert dispatched == []


def test_proposal_keeps_the_host_bound_source_run(conversation):
    service, _, session_id, _, _ = conversation
    record = service._get_record(session_id)
    record.binding = record.binding.model_copy(update={"run_id": "completed-run"})
    requests = []
    context = ToolExecutionContext(session=record, user_message="Repeat this study.",
        propose_multi_database=lambda req: requests.append(req) or {"multi_database_batch_id": "a" * 64})
    tools.execute_tool(TOOL, {"targets": [{"database": "aumc"}]}, context)
    assert requests[0].source_run_id == "completed-run"


def test_the_tool_is_a_sequential_proposal_without_model_supplied_authority():
    from easyicu.webserver.pi_copilot.tool_catalog import TOOL_CATALOG

    row = next(row for row in TOOL_CATALOG if row.name == TOOL)
    assert (row.surface, row.policy_group, row.execution_mode, row.host_mutating) == ("research", "control", "sequential", True)
    assert row.arguments.model == row.arguments.required == ("targets",)
    assert row.arguments.host == ()


@pytest.mark.parametrize("key", ["project_id", "session_id", "study_context_id", "source_run_id",
                                 "message_origin", "_message_job_id", "plan", "clearance"])
def test_model_cannot_supply_host_scope_or_governance_fields(conversation, key):
    service, _, session_id, dispatched, _ = conversation
    context = ToolExecutionContext(session=service._get_record(session_id), user_message="Repeat this study.")
    with pytest.raises(PiCopilotError) as error:
        tools.execute_tool(TOOL, {"targets": [{"database": "aumc"}], key: "forged"}, context)
    assert error.value.code == "pi_tool_unknown_arguments"
    assert dispatched == []
