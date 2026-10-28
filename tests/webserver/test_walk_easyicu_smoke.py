"""Slow ASGI walk contract smoke with the built-in offline provider.

The HTTP routes, package catalogue and request validation are real. Conversation
and execution services are synthetic, so this does not certify scientific
execution or Docker. Run on the lead's Mac: this server blocks TestClient's
AF_UNIX sends. No subprocess, listening server, provider network or real data.
"""

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest


@pytest.mark.slow
def test_two_workflow_walk_over_asgi_with_stubbed_execution_and_offline_provider(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("EASYICU_HOME", str(home))

    from fastapi.testclient import TestClient

    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import guided_sessions, settings, sources
    from easyicu.webserver.app import app
    from easyicu.webserver.pi_copilot.contracts import ResearchProviderBinding
    from easyicu.webserver.routes import agent, jobs, mentions, pi_copilot
    from tools.walk.http import HttpAPI
    from tools.walk.runner import Options, WalkRunner

    projects = home / "projects"
    projects.mkdir()
    monkeypatch.setattr(guided_sessions, "_CONFIG_PATH", home / "guided.json")
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", projects)
    monkeypatch.setattr(guided_sessions, "default_project_parent", lambda: projects)
    monkeypatch.setattr(settings, "load_settings", lambda: {
        "ai_enabled": True, "copilot_enabled": True, "science_skills_enabled": True,
        "connector_pubmed_enabled": False, "connector_zotero_enabled": False,
        "mcp_tools_enabled": False, "tool_audit_enabled": False,
    })
    registry = {"sources": [{"id": "walk-demo", "label": "Synthetic walk source", "database": "sic",
                              "ok": True, "source_kind": "synthetic"}], "active_path": None}
    monkeypatch.setattr(sources, "load_registry", lambda: registry)
    provider = ScriptedMockLLMClient(["Synthetic candidate ready for review."] * 2)

    class OfflineService:
        def __init__(self):
            self.sessions = {}
            self.completed = set()

        def verified_api_research_provider_binding(self):
            return ResearchProviderBinding(model="offline-fixture")

        def initialize_project(self, **kwargs):
            return {"ok": True}

        def create_session(self, project_id, **kwargs):
            sid = f"pi_walk_{len(self.sessions)}"
            session = {"session_id": sid, "project_id": project_id, "last_turn_status": "done",
                       "binding": {"study_context_id": f"study_{len(self.sessions)}", "study_revision": 1}}
            self.sessions[sid] = session
            return {"ok": True, "session": session}

        def get_session(self, session_id, *, project_id, **kwargs):
            session = self.sessions[session_id]
            assert session["project_id"] == project_id
            return {"ok": True, "session": session}

        def authorize_data_source(self, session_id, *, project_id, source_id, **kwargs):
            assert source_id == "walk-demo"
            self.get_session(session_id, project_id=project_id)
            return {"ok": True}

        def resolve_mention(self, session_id, *, project_id, **kwargs):
            self.get_session(session_id, project_id=project_id)
            return {"reference": {"kind": "database", "source": "registered_source", "id": "walk-demo", "digest": "a" * 64}}

        def send_message(self, session_id, *, project_id, message, allowed_actions, **kwargs):
            self.get_session(session_id, project_id=project_id)
            assert "run" not in allowed_actions and "provider_run" not in allowed_actions
            provider.complete([LLMMessage(role="user", content=message)])
            return {"ok": True}

        def get_project_workflow(self, *, project_id):
            rid = "run_" + project_id
            return {"ok": True, "project_id": project_id,
                    "workflow": {"current_stage": "plan", "next_action_code": "operator_plan_approval_required",
                                 "plan_review_summary": {"run_id": rid, "status": "ready_for_approval"},
                                 "plan_execution_ready": True},
                    "host_decisions": {"plan_review": {"family": "plan_review", "run_id": rid,
                        "scientific_configuration_sha256": "a" * 64, "review_authority_sha256": "b" * 64}},
                    "latest_run": {"present": True, "run_id": rid, "gate_status": "pass",
                        "execution_complete": rid in self.completed,
                        "artifact_refs": [{"artifact": "report.md"}, {"artifact": "evidence_ledger.json"}]},
                    "runs": [{"run_id": rid, "plan_available": True}]}

        def approve(self, body):
            assert body["decision"] == "approved"
            assert body["host_action"]["project_id"] in {s["project_id"] for s in self.sessions.values()}
            self.completed.add(body["run_id"])
            return {"job_id": "mock_review_job"}

    service = OfflineService()
    for module in (pi_copilot, mentions):
        monkeypatch.setattr(module, "get_pi_copilot_service", lambda: service)
    monkeypatch.setattr(agent, "submit_agent_run_review", lambda body, **kwargs: service.approve(body))
    monkeypatch.setattr(agent, "_submit_host_decision", lambda action, body, submit, **kwargs: submit())
    monkeypatch.setattr(jobs.job_store, "MANAGER", SimpleNamespace(get=lambda job_id: SimpleNamespace(
        snapshot=lambda: {"status": "done"})))

    # The client is intentionally not entered: application startup reconciles
    # real local jobs, outside this HTTP contract smoke's synthetic scope.
    client = TestClient(app)
    try:
        def exchange(method, url, body, timeout):
            assert urlsplit(url).netloc == "testserver"
            response = client.request(method, url, json=body, timeout=timeout, follow_redirects=False)
            return response.status_code, response.headers, response.content

        options = Options(source="walk-demo", out=tmp_path / "report", base="http://testserver",
                          only=("workflows",), max_workflows=2, approve_test_runs=True)
        report = WalkRunner(options, api=HttpAPI(options.base, exchange=exchange)).run()
    finally:
        client.close()
    results = [row for row in report["items"] if row["phase"] == "workflows" and row["status"] == "passed"]
    assert len(results) == 2, report
    assert len(service.sessions) == len(service.completed) == len(provider.calls) == 2
    assert report["aggregate_only"] is True
    assert not report["status_counts"].get("failed")
