"""A primary outcome answered in endpoint wording confirms the outcome slot.

Protocols call the primary outcome the primary endpoint.  The confirmation
gate recognized only outcome wording, so a user who answered the review's
endpoint question in endpoint words, then clicked the Copilot's own
affirmative choice, was asked the same question again after each answer.
A mention inside a research question still records candidate intent only.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import jobs, settings, study_contexts
from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.routes import pi_copilot as route_module
from tests.webserver.copilot.pi_copilot_contract_fixtures import (
    FakeGateway,
    allow_unrelated_message_test,
)

_PROPOSED = "90 天全因死亡（自入 ICU 起算）"


@pytest.fixture
def update(monkeypatch: pytest.MonkeyPatch):
    current = {
        "id": "study-endpoint-wording",
        "revision": 1,
        "question": "",
        "active_job_id": None,
        "cohort": {},
        "outcome": "",
        "primary_exposure": "",
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 2},
    )
    session = PiSessionRecord(
        session_id="pi-endpoint-wording",
        binding=AuthorityBinding(
            study_context_id=current["id"], study_revision=current["revision"],
        ),
    )

    def run(user_message: str, outcome: str = _PROPOSED) -> tuple[dict[str, Any], list]:
        writes.clear()
        result = tool_module.execute_tool(
            "easyicu_update_study_context",
            {"outcome": outcome},
            ToolExecutionContext(
                session=session, user_message=user_message, allowed_actions={"configure"},
            ),
        )
        return result, list(writes)

    return run


@pytest.mark.parametrize(
    "user_message,selected",
    [
        ("主要终点：自入 ICU 起 90 天内全因死亡；ICU 住院天数作为次要描述。", "自入 ICU 起 90 天内全因死亡"),
        ("主要结局：90 天全因死亡", "90 天全因死亡"),
        ("主要终点采用 90 天全因死亡。", "90 天全因死亡"),
        ("请把主要终点改为 90 天全因死亡。", "90 天全因死亡"),
        ("再次确认：主要终点为 90 天全因死亡；次要描述结局为 ICU 住院天数。", "90 天全因死亡"),
        ("The primary endpoint is 90-day all-cause mortality.", "90-day all-cause mortality"),
        ("Primary endpoint: 90-day all-cause mortality from ICU admission.", "90-day all-cause mortality from ICU admission"),
        ("主要结局：ICU 住院时长和院内死亡", "ICU 住院时长和院内死亡"),
        ("主要终点：ICU 住院时长及院内死亡", "ICU 住院时长及院内死亡"),
        ("主要结局：ICU 住院时长以及院内死亡", "ICU 住院时长以及院内死亡"),
        ("主要结局：ICU 住院时长与院内死亡", "ICU 住院时长与院内死亡"),
        ("主要结局：ICU 住院时长、院内死亡", "ICU 住院时长、院内死亡"),
        ("Primary outcome: ICU length of stay and in-hospital mortality", "ICU length of stay and in-hospital mortality"),
        ("Use ICU length of stay and in-hospital mortality as the primary endpoint", "ICU length of stay and in-hospital mortality"),
    ],
)
def test_an_explicit_endpoint_answer_saves_the_primary_outcome(update, user_message, selected):
    result, writes = update(user_message)

    assert result["code"] == "study_context_updated"
    assert writes[-1]["outcome"] == selected


@pytest.mark.parametrize(
    "user_message",
    [
        "研究脓毒症患者的器官功能轨迹与 90 天死亡的关系。",
        "主要终点是什么？",
        "是否确认该主要终点？",
        "确认该主要终点？",
        "是，确认该主要终点和次要描述结局",
        "确认该主要终点",
        "不确认该主要终点",
        "主要终点：待定",
        "主要终点：？",
        "次要终点：ICU 住院天数",
        "What is the primary endpoint?",
        "Primary endpoint: TBD",
        "主要结局：ICU 住院时长或是院内死亡",
        "主要结局：ICU 住院时长和院内死亡？",
        "不确认主要结局：ICU 住院时长和院内死亡",
        "主要结局：ICU 住院时长；主要结局：院内死亡",
        "Primary outcome: ICU length of stay or in-hospital mortality",
    ],
)
def test_a_mention_or_open_question_still_needs_a_choice(update, user_message):
    result, writes = update(user_message)

    assert result["code"] == "study_primary_outcome_confirmation_required"
    assert writes == []


def test_the_refusal_says_how_the_outcome_is_confirmed(update):
    result, _ = update("研究脓毒症患者的器官功能轨迹与 90 天死亡的关系。")

    summary = result["summary"]
    assert "exact label" in summary
    assert "主要结局/主要终点：X" in summary
    assert "offer that exact label as a choice" in summary


def test_the_users_own_label_confirms_itself(update):
    result, writes = update(
        "我想看 90 天全因死亡（自入 ICU 起算），院内死亡只作描述。",
    )

    assert result["code"] == "study_context_updated"
    assert writes[-1]["outcome"] == _PROPOSED


_COMPOUND = "ICU 住院时长和院内死亡"


class _OutcomeGateway(FakeGateway):
    """Script the stored model call; preload the same host receipt as Pi."""

    proposed: str | None = _COMPOUND

    def __init__(self):
        super().__init__()
        self.receipts = []
        self.updates = []

    def request(self, method, params, **kwargs):
        state = super().request(method, params, **kwargs)
        if method == "session.prompt":
            context = kwargs["tool_context"]
            receipt = tool_module.execute_tool("easyicu_inspect_workflow", {}, context)
            self.receipts.append(receipt)
            if self.proposed is not None:
                result = tool_module.execute_tool(
                    "easyicu_update_study_context", {"outcome": self.proposed}, context,
                )
                self.updates.append(result)
        return state


@pytest.fixture
def outcome_route(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    study = study_contexts.upsert_context(
        {"title": "Synthetic outcome confirmation", "question": "研究早期 AKI 与结局的关系？",
         "data_source": {"database": "miiv", "label": "Synthetic demo metadata"},
         "unconfirmed_outcome": _COMPOUND},
        _server_question_wording_write=True,
    )
    gateway = _OutcomeGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    session = service.create_session(
        project_id="synthetic-outcome-project", study_context_id=study["id"],
        external_llm_opt_in=True,
    )["session"]
    allow_unrelated_message_test(service, session["session_id"])
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: service)
    client = TestClient(app)

    def send(message, actions=("configure",), origin="host_prompt"):
        response = client.post(
            f"/api/copilot/pi/sessions/{session['session_id']}/message",
            json={"project_id": session["project_id"], "message": message,
                  "allowed_actions": list(actions), "message_origin": origin},
        )
        assert response.status_code == 200, response.text
        job = jobs.MANAGER.get(response.json()["job_id"])
        deadline = time.monotonic() + 5
        while job.status in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.status == "done", job.error
        rebound = client.post(
            f"/api/copilot/pi/sessions/{session['session_id']}/rebind",
            json={"project_id": session["project_id"]},
        )
        assert rebound.status_code == 200, rebound.text
        return study_contexts.get_context(study["id"])

    return send, gateway, service, study


@pytest.mark.parametrize("message", [f"主要结局：{_COMPOUND}", f"Primary outcome: {_COMPOUND}"])
def test_card_click_persists_compound_and_next_turn_receives_it(outcome_route, message, monkeypatch):
    send, gateway, service, study = outcome_route
    saved = send(message)

    assert gateway.updates[-1]["code"] == "study_context_updated"
    assert saved["outcome"] == _COMPOUND
    assert not saved.get("unconfirmed_outcome")
    updated_workflow = gateway.updates[-1]["details"]["workflow"]
    assert updated_workflow["next_action_code"] != "study_primary_outcome_confirmation_required"
    assert updated_workflow["study_setup_receipt"]["configuration"]["outcome"] == _COMPOUND
    # The click saves the label, without inventing plan approvals or a machine endpoint.
    assert saved["confirmations"] == study["confirmations"]
    assert not saved["execution_concepts"].get("outcome")
    # A repeated click keeps the same confirmed label and cannot resurrect the card.
    repeated = send(message)
    assert repeated["outcome"] == _COMPOUND
    assert not repeated.get("unconfirmed_outcome")
    assert study_contexts.scientific_configuration_sha256(repeated) == study_contexts.scientific_configuration_sha256(saved)
    # Restart the service before the next model turn, using only persisted state.
    restarted = PiCopilotService(store_path=service.store_path, gateway=gateway)
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: restarted)
    record = restarted._get_record(gateway.tool_contexts[-1].session.session_id)
    assert record.binding.study_context_id == study["id"]
    gateway.proposed = None
    send("继续", actions=(), origin="typed")
    workflow = gateway.receipts[-1]["details"]["workflow"]
    assert workflow["next_action_code"] != "study_primary_outcome_confirmation_required"
    assert not workflow.get("unconfirmed_outcome")
    assert "outcome_confirmation" not in workflow["planning_prerequisites_missing"]
    assert workflow["study_setup_receipt"]["configuration"]["outcome"] == _COMPOUND


@pytest.mark.parametrize("message,actions", [
    ("研究早期 AKI 与 ICU 住院时长和院内死亡的关系？", ("configure",)),
    ("继续", ("configure",)),  # A model tool call is not the user's approval.
    ("继续", ()),
])
def test_model_proposal_without_click_keeps_confirmation_pending(outcome_route, message, actions):
    send, gateway, _, _ = outcome_route
    saved = send(message, actions=actions, origin="typed")

    assert gateway.updates[-1]["status"] == "blocked"
    assert saved["outcome"] == ""
    assert saved["unconfirmed_outcome"] == _COMPOUND


def test_confirming_compound_does_not_select_a_model_substitution(outcome_route):
    send, gateway, _, _ = outcome_route
    gateway.proposed = "院内死亡"
    saved = send(f"主要结局：{_COMPOUND}")
    assert saved["outcome"] == _COMPOUND
    assert saved["outcome"] != gateway.proposed

    # A later, different model proposal cannot inherit the earlier approval.
    saved = send("继续", origin="typed")
    assert gateway.updates[-1]["code"] == "study_primary_outcome_confirmation_required"
    assert saved["outcome"] == _COMPOUND
