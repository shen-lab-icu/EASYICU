"""Synthetic host actions exercise review authority, not patient execution."""
from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.webserver import agent_pipeline_runs, host_action_jobs, raw_preparation_candidate
from easyicu.webserver.host_action_contracts import HostDecisionOffers
from easyicu.webserver.raw_preparation_decisions import verify_host_review
from easyicu.webserver.routes import agent as route
from tests.webserver.test_raw_preparation_decisions import _host_decision


def test_synthetic_host_action_records_decision_and_binds_scope(tmp_path, monkeypatch):
    from easyicu.webserver.pi_copilot import service
    from easyicu.webserver import jobs

    real_get = jobs.MANAGER.get
    pending, args, binding = _host_decision(tmp_path, monkeypatch)
    monkeypatch.setattr(jobs.MANAGER, "get", real_get)
    action = args["host_action"]
    (tmp_path / "raw_preparation_checkpoint.json").write_text(pending.model_dump_json())
    monkeypatch.setattr(raw_preparation_candidate, "available_host_review", lambda **_: (
        action.decision, pending, binding["scope"], tmp_path))
    rows = {}
    def record(_session, **kwargs):
        rows[kwargs["action_key"]] = {"child_job_id": kwargs["child_job_id"]}
    owner = SimpleNamespace(host_action_session=lambda *args, **kwargs: None,
        record_job_host_action=record, host_action_child_job=lambda _session, **kwargs: rows.get(kwargs["action_key"]))
    monkeypatch.setattr(service, "get_pi_copilot_service", lambda: owner)
    original = host_action_jobs.submit_with_host_action
    def submit(host_action, body, callback, **kw):
        return original(host_action, study_context_id=body["study_context_id"], submit=callback,
                        conversations=owner,
                        offered=lambda _: HostDecisionOffers(raw_preparation_review=action.decision), **kw)
    monkeypatch.setattr(route, "_submit_host_decision", submit)
    outcome = route.raw_preparation_review({"study_context_id": "study", "decision": "approved",
        "host_action": action.model_dump(mode="json")})
    assert outcome["decision"] == "approved" and outcome["analysis_authorized"] is False
    checkpoint = RawPreparationCheckpoint.model_validate_json((tmp_path / "raw_preparation_checkpoint.json").read_text())
    verify_host_review(checkpoint=checkpoint, **binding)
    assert checkpoint.decision.reviewer == "actual-host-operator"


def test_unregistered_adapter_refuses_even_a_forged_review_action(tmp_path, monkeypatch):
    from easyicu.webserver import reviewed_snapshot_producer

    pending, args, _ = _host_decision(tmp_path, monkeypatch)
    (tmp_path / "raw_preparation_checkpoint.json").write_text(pending.model_dump_json())
    monkeypatch.setattr(agent_pipeline_runs, "raw_preparation_review_owner", lambda *_: (
        SimpleNamespace(run_dir=str(tmp_path), run_id="run"), {"id": "study"}))
    monkeypatch.setattr(reviewed_snapshot_producer, "_CLOSED_ADAPTERS", {})
    with pytest.raises(HTTPException) as error:
        route.raw_preparation_review({"study_context_id": "study", "decision": "approved",
            "host_action": args["host_action"].model_dump(mode="json")})
    assert error.value.detail["error"] == "raw_preparation_read_closure_unverified"
    assert RawPreparationCheckpoint.model_validate_json(
        (tmp_path / "raw_preparation_checkpoint.json").read_text()).state == "pending"
