from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from easyicu.webserver import agent_pipeline_runs, raw_preparation_candidate, sources
from easyicu.webserver.routes.agent import raw_preparation_candidate as candidate_route
from tests.webserver.test_sic_raw_preparation_grant import _case


def test_real_review_owner_builds_candidate_without_approving_or_starting(tmp_path, monkeypatch):
    result, source_root, _, checkpoint = _case(tmp_path, monkeypatch, approved=False)
    (tmp_path / "research_context.json").write_text(result.context.model_dump_json())
    pending = SimpleNamespace(run_id="run", run_dir=str(tmp_path), requests=(checkpoint.request,))
    monkeypatch.setattr(agent_pipeline_runs, "raw_preparation_review_owner", lambda run_id, study_id: (pending, {"id": study_id}))
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [{"path": str(source_root)}]})
    outcome = raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")
    assert outcome["status"] == "blocked"
    assert outcome["blocker_codes"]
    assert not (tmp_path / "raw_preparation_checkpoint.json").exists()
    assert not (tmp_path / "approved_executable_plan.json").exists()


def test_route_requires_bound_conversation_and_is_only_a_preview(monkeypatch):
    from easyicu.webserver.pi_copilot import service

    calls = []
    owner = SimpleNamespace(host_action_session=lambda session, **kwargs: calls.append((session, kwargs)))
    monkeypatch.setattr(service, "get_pi_copilot_service", lambda: owner)
    monkeypatch.setattr(raw_preparation_candidate, "create_candidate_for_review",
                        lambda **kwargs: {"status": "blocked", "blocker_codes": ["dependency_partition_roster_unknown"]})
    body = {"run_id": "run", "study_context_id": "study", "session_id": "session", "project_id": "project"}
    assert candidate_route(body)["status"] == "blocked"
    assert calls == [("session", {"project_id": "project", "study_context_id": "study"})]
    with pytest.raises(HTTPException) as error:
        candidate_route({})
    assert error.value.status_code == 400
