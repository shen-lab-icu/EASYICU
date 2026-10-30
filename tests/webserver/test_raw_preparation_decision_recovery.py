"""Recover only a previously host-signed, unconsumed preparation decision."""
import json

import pytest
from fastapi import HTTPException

from easyicu import state_paths
from easyicu.webserver import raw_preparation_decisions as owner, raw_preparation_durable
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding
from tests.webserver.test_raw_preparation_decisions import _host_decision


def _case(tmp_path, monkeypatch):
    pending, args, values = _host_decision(tmp_path, monkeypatch)
    approved = owner.record_host_review(**args)
    binding = RawPreparationHostBinding(**{k: v for k, v in values.items() if k != "scope"})
    monkeypatch.setattr(raw_preparation_durable, "_owner", lambda *_: tmp_path)
    return pending, approved, binding, values["scope"]


def test_original_host_decision_survives_checkpoint_publication_crash(tmp_path, monkeypatch):
    pending, approved, binding, scope = _case(tmp_path, monkeypatch)
    recovered = owner.recover_host_review(pending=pending, binding=binding, scope=scope, job_id="job", run_dir=tmp_path)
    assert recovered == approved
    owner.verify_host_review(checkpoint=recovered, scope=scope, **binding.model_dump(mode="python"))
    owner.consume_host_review(checkpoint=recovered, scope=scope, **binding.model_dump(mode="python"))
    with pytest.raises(ValueError, match="nonce_replayed"):
        owner.recover_host_review(pending=pending, binding=binding, scope=scope, job_id="job", run_dir=tmp_path)


@pytest.mark.parametrize("change", ["job", "scope", "signature", "legacy"])
def test_recovery_rejects_changed_or_incomplete_authority(tmp_path, monkeypatch, change):
    pending, _, binding, scope = _case(tmp_path, monkeypatch)
    job_id = "other" if change == "job" else "job"
    if change == "scope":
        scope = {**scope, "files": ["different.csv"]}
    if change in {"signature", "legacy"}:
        path = state_paths.state_root() / "private-preparation-decisions" / (binding.nonce + ".json")
        value = json.loads(path.read_text())
        if change == "signature":
            value["decided_review"]["reviewer"] = "different"
        else:
            value.pop("decided_review")
            value["signature"] = owner._signature({k: v for k, v in value.items() if k != "signature"})
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="recovery_receipt_invalid"):
        owner.recover_host_review(pending=pending, binding=binding, scope=scope, job_id=job_id, run_dir=tmp_path)


def test_recovery_refuses_changed_run_owner(tmp_path, monkeypatch):
    pending, _, binding, scope = _case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="run_owner_changed"):
        owner.recover_host_review(pending=pending, binding=binding, scope=scope, job_id="job", run_dir=tmp_path / "other")


def test_recovery_refuses_a_replanned_same_run(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
    from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
    from easyicu.research_agent.schema import AnalysisPlan
    from easyicu.webserver import agent_pipeline_runs, jobs
    from easyicu.webserver.host_action_contracts import HostActionTag, decision_key, host_action_id

    _, args, values = _host_decision(tmp_path, monkeypatch)
    original = PlanReviewAuthority.create(plan=AnalysisPlan(research_question="Original", steps=[]))
    payload = {"plan_review_authority": original.model_dump(mode="json"), "synthetic_preparation_scope": values["scope"]}
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Review source and files", payload=payload,
                                        authority_sha256=canonical_sha256(payload))
    pending = RawPreparationCheckpoint.pending(plan_sha256=original.plan_sha256, request=request)
    choice = args["host_action"].decision.model_copy(update={"checkpoint_sha256": pending.checkpoint_sha256,
                                                               "review_authority_sha256": request.authority_sha256})
    action = args["host_action"].model_copy(update={"decision": choice})
    key = decision_key(choice, review_decision="approved")
    tag = HostActionTag(session_id="session", project_id="project", study_context_id="study",
        action_code=action.action_code, decision_key=key, action_id=host_action_id("session", action.action_code, key))
    monkeypatch.setattr(jobs.MANAGER, "get", lambda _: SimpleNamespace(host_action=tag))
    approved = owner.record_host_review(**{**args, "host_action": action, "pending": pending})
    binding = RawPreparationHostBinding(**{k: v for k, v in values.items() if k != "scope"})
    monkeypatch.setattr(raw_preparation_durable, "_owner", lambda *_: tmp_path)
    monkeypatch.setattr(agent_pipeline_runs, "raw_preparation_review_owner", lambda *_: (
        SimpleNamespace(requests=[request]), {}))
    assert owner.recover_host_review(pending=pending, binding=binding, scope=values["scope"],
                                     job_id="job", run_dir=tmp_path) == approved
    different = PlanReviewAuthority.create(plan=AnalysisPlan(research_question="Changed", steps=[]))
    monkeypatch.setattr(agent_pipeline_runs, "raw_preparation_review_owner", lambda *_: (
        SimpleNamespace(requests=[SimpleNamespace(payload={"plan_review_authority": different.model_dump(mode="json")})]), {}))
    with pytest.raises(ValueError, match="plan_owner_changed"):
        owner.recover_host_review(pending=pending, binding=binding, scope=values["scope"], job_id="job", run_dir=tmp_path)


def test_reused_route_restores_exact_checkpoint_without_starting_another_job(tmp_path, monkeypatch):
    from easyicu.webserver import raw_preparation_candidate, jobs
    from easyicu.webserver.routes import agent as route

    pending, args, values = _host_decision(tmp_path, monkeypatch)
    approved = owner.record_host_review(**args)
    path = tmp_path / "raw_preparation_checkpoint.json"
    path.write_text(pending.model_dump_json())
    action = args["host_action"]
    monkeypatch.setattr(raw_preparation_candidate, "available_host_review", lambda **_: (
        action.decision, pending, values["scope"], tmp_path))
    monkeypatch.setattr(route, "_submit_host_decision", lambda *a, **k: {"reused": True, "job_id": "job"})
    monkeypatch.setattr(raw_preparation_durable, "_owner", lambda *_: tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("reused approval must not start another job")
    monkeypatch.setattr(jobs.MANAGER, "run_reserved", forbidden)
    result = route.raw_preparation_review({"study_context_id": "study", "decision": "approved",
                                           "host_action": action.model_dump(mode="json")})
    assert result["reused"] and result["preparation_status"] == "review_recorded"
    assert not result["analysis_authorized"]
    assert type(pending).model_validate_json(path.read_bytes()) == approved


def test_reused_route_does_not_fabricate_missing_or_invalid_receipt(tmp_path, monkeypatch):
    from easyicu.webserver import raw_preparation_candidate
    from easyicu.webserver.routes import agent as route

    pending, args, values = _host_decision(tmp_path, monkeypatch)
    path = tmp_path / "raw_preparation_checkpoint.json"
    path.write_text(pending.model_dump_json())
    action = args["host_action"]
    monkeypatch.setattr(raw_preparation_candidate, "available_host_review", lambda **_: (
        action.decision, pending, values["scope"], tmp_path))
    monkeypatch.setattr(route, "_submit_host_decision", lambda *a, **k: {"reused": True, "job_id": "job"})
    body = {"study_context_id": "study", "decision": "approved", "host_action": action.model_dump(mode="json")}
    assert route.raw_preparation_review(body)["preparation_status"] == "review_recording"
    assert type(pending).model_validate_json(path.read_bytes()) == pending
    owner.record_host_review(**args)
    receipt = state_paths.state_root() / "private-preparation-decisions" / (values["nonce"] + ".json")
    receipt.write_text("{}")
    with pytest.raises(HTTPException):
        route.raw_preparation_review(body)
    assert type(pending).model_validate_json(path.read_bytes()) == pending
