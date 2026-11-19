"""A run started from a Copilot decision uses the conversation's research model.

"生成候选计划" posts ``/api/jobs/agent-run`` with the conversation's host
decision. The route resolved only the browser's account, so the plan ran on
the account default (here GPT-6.2-Sol) while the researcher had selected
GPT-5.6-Sol for the conversation. The conversation owner's selection now
reaches the run environment; the account owner sets ``EASYICU_CODEX_MODEL``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import host_action_jobs
from easyicu.webserver.app import app
from easyicu.webserver.host_action_contracts import PlanTransitionDecision
from easyicu.webserver.pi_copilot.contracts import ResearchProviderBinding
from easyicu.webserver.routes import agent

ACCOUNT = "c" * 64
DECISION = PlanTransitionDecision(
    next_action_code="provider_ready_to_generate_plan", scientific_configuration_sha256="a" * 64,
)


def _body(*, host_action: bool) -> dict:
    body = {"engine": "research_agent_pipeline", "study_context_id": "study-synthetic",
            "llm_provider": "codex", "credential_source": "codex_user_auth"}
    if host_action:
        body["host_action"] = {"session_id": "pi-synthetic", "project_id": "project-synthetic",
                               "action_code": "generate_plan", "decision": DECISION.model_dump(mode="json")}
    return body


@pytest.fixture
def run(monkeypatch):
    submitted: list = []
    monkeypatch.setattr(agent.codex_account_sessions, "environment_for_request",
                        lambda _request: {"EASYICU_CODEX_SESSION_SHA256": ACCOUNT})
    monkeypatch.setattr(agent.codex_account_sessions, "environment_for_binding",
                        lambda binding, *, model: {"EASYICU_CODEX_SESSION_SHA256": binding,
                                                    "EASYICU_CODEX_MODEL": model})
    monkeypatch.setattr(agent, "submit_agent_run", lambda body, **kw: submitted.append(kw) or {"job_id": "job"})
    monkeypatch.setattr(agent, "_submit_host_decision", lambda action, body, submit, **kw: submit())

    def start(*, provider, host_action=True):
        service = SimpleNamespace(host_action_research_provider=lambda *a, **k: provider)
        monkeypatch.setattr(host_action_jobs, "get_pi_copilot_service", lambda: service)
        response = TestClient(app).post("/api/jobs/agent-run", json=_body(host_action=host_action))
        assert response.status_code == 200, response.text
        return dict(submitted.pop()["account_environment"])

    return start


def _codex(model: str, account: str = ACCOUNT) -> ResearchProviderBinding:
    return ResearchProviderBinding(provider="codex", credential_source="codex_user_auth",
                                   authentication_mode="chatgpt_account", model=model,
                                   account_session_sha256=account)


def test_a_copilot_decision_runs_on_the_conversations_model(run):
    assert run(provider=_codex("gpt-5.6-sol"))["EASYICU_CODEX_MODEL"] == "gpt-5.6-sol"


def test_without_a_conversation_model_the_account_default_stays(run):
    for environment in (run(provider=_codex("gpt-5.6-sol"), host_action=False),
                        run(provider=ResearchProviderBinding()),
                        run(provider=_codex("gpt-5.6-sol", account="d" * 64))):
        assert environment == {"EASYICU_CODEX_SESSION_SHA256": ACCOUNT}
