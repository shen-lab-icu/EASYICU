"""Real host decision path, fixed operator, and retained-byte revalidation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request

from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import reviewed_csv_projection as projection
from easyicu.webserver.host_action_contracts import (
    HostActionRequest, HostActionTag, RawPreparationReviewDecision, decision_key, host_action_id,
)
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, record_host_review
from easyicu.webserver.raw_preparation_attestation import verify_attestation


class DifferentImplementation:
    pass


def test_reviewed_fixed_projection_retains_evidence_across_restart(tmp_path, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs, jobs
    from easyicu.webserver.pi_copilot import service

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    source = tmp_path / "registered"
    source.mkdir()
    (source / "table.csv").write_text("id,secret,value\n1,hidden,2\n2,private,3\n")
    identity = SourceIdentity(database="sic", registration_id="source-one")
    monkeypatch.setattr(projection.sources, "load_registry", lambda: {"sources": [{"path": str(source), "ok": True}]})
    monkeypatch.setattr(projection.sources, "registered_source_identity", lambda path: identity if path == str(source) else None)
    scope = {"adapter_id": "registered-csv-projection/1", "source_identity": identity.to_dict(),
             "relative_file": "table.csv", "columns": ["id", "value"], "purpose": "prepare_single_table_only",
             "plan_sha256": "c" * 64, "dependencies_sha256": "d" * 64}
    for invalid in ({**scope, "relative_file": "../table.csv"},
                    {**scope, "columns": ["id", "id"]},
                    {**scope, "source_identity": {"database": "sic", "registration_id": "unregistered"}}):
        with pytest.raises(ValueError):
            projection._validated_scope(invalid)
    payload = {"raw_preparation_scope": scope}
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Fixture review",
                                        payload=payload, authority_sha256=canonical_sha256(payload))
    pending = RawPreparationCheckpoint.pending(plan_sha256="c" * 64, request=request)
    choice = RawPreparationReviewDecision(run_id="run", scientific_configuration_sha256="a" * 64,
        checkpoint_sha256=pending.checkpoint_sha256, scope_sha256=canonical_sha256(scope),
        review_authority_sha256=request.authority_sha256, nonce="b" * 64)
    action = HostActionRequest(session_id="session", project_id="project", action_code="review_raw_preparation", decision=choice)
    key = decision_key(choice, review_decision="approved")
    tag = HostActionTag(session_id="session", project_id="project", study_context_id="study",
        action_code=action.action_code, decision_key=key, action_id=host_action_id("session", action.action_code, key))
    monkeypatch.setattr(jobs.MANAGER, "get", lambda job_id: SimpleNamespace(host_action=tag))
    monkeypatch.setattr(agent_pipeline_runs, "server_reviewer_identity", lambda: "fixture-reviewer")
    monkeypatch.setattr(service, "get_pi_copilot_service", lambda: SimpleNamespace(
        host_action_session=lambda *a, **kw: None, host_action_child_job=lambda *a, **kw: {"child_job_id": "job"}))
    checkpoint = record_host_review(host_action=action, pending=pending, decision="approved", study_context_id="study",
        job_id="job", run_id="run", scientific_configuration_sha256="a" * 64)
    binding = RawPreparationHostBinding(project_id="project", session_id="session", study_context_id="study",
                                        run_id="run", nonce=choice.nonce)
    kwargs = dict(checkpoint=checkpoint, binding=binding, scope=scope, host_work_root=tmp_path)
    with pytest.raises(ValueError, match="host_binding_mismatch"):
        projection.run_reviewed_csv_projection(**{**kwargs, "scope": {**scope, "columns": ["secret"]}})
    output = projection.run_reviewed_csv_projection(**kwargs)
    assert (output / "projection.csv").read_text() == "id,value\n1,2\n2,3\n"
    body = projection.verify_reviewed_csv_projection(root=output.parent, checkpoint_sha256=checkpoint.checkpoint_sha256,
                                                     binding=binding, scope=scope)
    assert body["host_consumption"]["rows"] == 2
    assert body["host_consumption"]["sha256"] == body["snapshot"]["files"][0]["sha256"]
    assert body["host_consumption"]["columns"] == ["id", "value"]
    assert body["analysis_authorized"] is False and body["runtime_integrity_verified"] is False
    verify_args = dict(root=output.parent, checkpoint_sha256=checkpoint.checkpoint_sha256, binding=binding, scope=scope)
    with pytest.raises(ValueError, match="binding_mismatch"):
        projection.verify_reviewed_csv_projection(**{**verify_args, "binding": binding.model_copy(update={"run_id": "another"})})
    with pytest.raises(ValueError, match="implementation_changed"):
        verify_attestation(**verify_args, adapter=DifferentImplementation(), worker=Path(projection.__file__))
    changed_worker = tmp_path / "changed_worker.py"
    changed_worker.write_bytes(Path(projection.__file__).read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="implementation_changed"):
        verify_attestation(**verify_args, adapter=projection._ProjectionImplementation(), worker=changed_worker)
    receipt_path = output.parent / "producer-attestation.json"
    before = receipt_path.read_bytes()
    altered = json.loads(before)
    altered["body"]["host_consumption"]["rows"] = 999
    receipt_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="signature_invalid"):
        projection.verify_reviewed_csv_projection(**verify_args)
    receipt_path.write_bytes(before)
    with pytest.raises(ValueError, match="nonce_replayed"):
        projection.run_reviewed_csv_projection(**kwargs)
    code = '''
import json,sys
from pathlib import Path
from easyicu.webserver import reviewed_csv_projection as p
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding
root,source,scope,sha,nonce=sys.argv[1:]
s=json.loads(scope); i=SourceIdentity.from_record(s["source_identity"])
p.sources.load_registry=lambda:{"sources":[{"path":source,"ok":True}]}
p.sources.registered_source_identity=lambda path:i if path==source else None
b=RawPreparationHostBinding(project_id="project",session_id="session",study_context_id="study",run_id="run",nonce=nonce)
p.verify_reviewed_csv_projection(root=Path(root),checkpoint_sha256=sha,binding=b,scope=s)
'''
    restarted = subprocess.run([sys.executable, "-c", code, str(output.parent), str(source), json.dumps(scope),
                                checkpoint.checkpoint_sha256, choice.nonce], env=os.environ.copy(), capture_output=True, text=True)
    assert restarted.returncode == 0, restarted.stderr


def test_partial_candidate_refuses_other_unknown_dependencies(tmp_path, monkeypatch):
    from easyicu.webserver import raw_preparation_candidate, agent_pipeline_runs
    from easyicu.webserver.raw_preparation_projection import pending_preparation_projection

    _single_table_pending = raw_preparation_candidate._single_table_pending

    source = tmp_path / "registered"
    source.mkdir()
    identity = SourceIdentity(database="sic", registration_id="source-one")
    monkeypatch.setattr(projection.sources, "load_registry", lambda: {"sources": [{"path": str(source), "ok": True}]})
    monkeypatch.setattr(projection.sources, "registered_source_identity", lambda path: identity if path == str(source) else None)
    plan = SimpleNamespace(plan_sha256="c" * 64, model_dump=lambda **kw: {"plan_sha256": "c" * 64})
    body = {"blocker_codes": ["dependency_cohort_inputs_unknown"],
            "declaration": {"plan_sha256": plan.plan_sha256,
                            "source_identity_sha256": canonical_sha256(identity.to_dict())},
            "files": [{"format": "csv", "path": "table.csv", "columns": ["id", "value"]}]}
    dependencies = {**body, "declaration_sha256": canonical_sha256(body)}
    pending = _single_table_pending(plan, dependencies, identity)
    assert pending and "不构成完整队列" in pending.request.summary
    assert pending.request.payload["raw_preparation_scope"]["columns"] == ["id", "value"]
    (tmp_path / "raw_preparation_checkpoint.json").write_text(pending.model_dump_json())
    preview = pending_preparation_projection({"run_id": "run", "project_dir": str(tmp_path)})
    assert preview["approvable"] is False
    assert "全部行和全部字段" in preview["summary"] and "id, value" in preview["summary"]
    monkeypatch.setattr(agent_pipeline_runs, "raw_preparation_review_owner", lambda *_: (
        SimpleNamespace(run_dir=str(tmp_path)), {"id": "study"}))
    with pytest.raises(ValueError, match="partial_review_surface_unavailable"):
        raw_preparation_candidate.available_host_review(run_id="run", study_context_id="study")
    body["blocker_codes"] = ["dependency_cohort_inputs_unknown", "dependency_partition_roster_unknown"]
    dependencies = {**body, "declaration_sha256": canonical_sha256(body)}
    assert _single_table_pending(plan, dependencies, identity) is None


def test_single_table_host_action_runs_private_partial_preparation(tmp_path, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs, host_action_jobs, jobs, raw_preparation_candidate, study_contexts
    from easyicu.webserver.host_action_contracts import HostDecisionOffers
    from easyicu.webserver.pi_copilot import service
    from easyicu.webserver.routes import agent as route

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    source = tmp_path / "registered"
    source.mkdir()
    (source / "table.csv").write_text("id,private,value\n1,secret,2\n")
    identity = SourceIdentity(database="sic", registration_id="source-one")
    monkeypatch.setattr(projection.sources, "load_registry", lambda: {"sources": [{"path": str(source), "ok": True}]})
    monkeypatch.setattr(projection.sources, "registered_source_identity", lambda path: identity if path == str(source) else None)
    scope = {"adapter_id": "registered-csv-projection/1", "source_identity": identity.to_dict(),
             "relative_file": "table.csv", "columns": ["id", "value"], "purpose": "prepare_single_table_only",
             "plan_sha256": "c" * 64, "dependencies_sha256": "d" * 64}
    payload = {"raw_preparation_scope": scope}
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Partial fixture only",
                                        payload=payload, authority_sha256=canonical_sha256(payload))
    pending = RawPreparationCheckpoint.pending(plan_sha256="c" * 64, request=request)
    choice = RawPreparationReviewDecision(run_id="run", scientific_configuration_sha256="a" * 64,
        checkpoint_sha256=pending.checkpoint_sha256, scope_sha256=canonical_sha256(scope),
        review_authority_sha256=request.authority_sha256, nonce="e" * 64)
    action = HostActionRequest(session_id="session", project_id="project", action_code="review_raw_preparation", decision=choice)
    (tmp_path / "raw_preparation_checkpoint.json").write_text(pending.model_dump_json())
    monkeypatch.setattr(raw_preparation_candidate, "available_host_review", lambda **_: (choice, pending, scope, tmp_path))
    monkeypatch.setattr(agent_pipeline_runs, "server_reviewer_identity", lambda: "fixture-reviewer")
    monkeypatch.setattr(study_contexts, "get_context", lambda _: {"id": "study"})
    monkeypatch.setattr(study_contexts, "scientific_configuration_sha256", lambda _: "a" * 64)
    rows = {}
    def record(_session, **kwargs):
        rows[kwargs["action_key"]] = {"child_job_id": kwargs["child_job_id"]}
    owner = SimpleNamespace(host_action_session=lambda *a, **kw: None, record_job_host_action=record,
                            host_action_child_job=lambda _session, **kw: rows.get(kw["action_key"]))
    monkeypatch.setattr(service, "get_pi_copilot_service", lambda: owner)
    original = host_action_jobs.submit_with_host_action
    monkeypatch.setattr(route, "_submit_host_decision", lambda action, body, callback, **kw:
        original(action, study_context_id=body["study_context_id"], submit=callback, conversations=owner,
                 offered=lambda _: HostDecisionOffers(raw_preparation_review=choice), **kw))
    result = route.raw_preparation_review({"study_context_id": "study", "decision": "approved",
                                           "host_action": action.model_dump(mode="json")})
    assert result["preparation_status"] == "single_table_preparing" and result["cohort_complete"] is False
    job = jobs.MANAGER.get(result["preparation_job_id"])
    end = time.monotonic() + 30
    while job.status == "running" and time.monotonic() < end:
        time.sleep(0.05)
    assert job.status == "done", job.snapshot()
    assert job.snapshot()["result"]["cohort_complete"] is False
    outputs = list(tmp_path.glob("reviewed-csv-projection-*/output/projection.csv"))
    assert len(outputs) == 1 and outputs[0].read_text() == "id,value\n1,2\n"
    monkeypatch.setattr(agent_pipeline_runs, "raw_preparation_review_owner", lambda *_: (
        SimpleNamespace(run_dir=str(tmp_path)), {"id": "study"}))
    body = {"project_id": "project", "session_id": "session", "study_context_id": "study",
            "run_id": "run", "job_id": result["preparation_job_id"]}
    def request(origin="http://testserver"):
        return Request({"type": "http", "method": "POST", "scheme": "http", "path": "/", "root_path": "",
                        "server": ("testserver", 80), "headers": [(b"origin", origin.encode())]})
    auth_dir = tmp_path / "home" / ".easyicu" / "private-preparation-decisions"
    auth_before = {item.name: item.read_bytes() for item in auth_dir.iterdir()}
    response = route.raw_preparation_projection_download(body, request())
    assert response.body == b"id,value\n1,2\n"
    assert response.headers["cache-control"] == "private, no-store"
    assert "cohort-incomplete" in response.headers["x-easyicu-artifact-scope"]
    assert {item.name: item.read_bytes() for item in auth_dir.iterdir()} == auth_before
    # Reopen the signed host decision and exact bytes in a fresh interpreter.
    restart_code = '''
import json,sys
from pathlib import Path
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import reviewed_csv_projection as p
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding,verify_consumed_host_review
root=Path(sys.argv[1]); source=sys.argv[2]
c=RawPreparationCheckpoint.model_validate_json((root.parent/"raw_preparation_checkpoint.json").read_text())
s=c.request.payload["raw_preparation_scope"]; i=SourceIdentity.from_record(s["source_identity"])
p.sources.load_registry=lambda:{"sources":[{"path":source,"ok":True}]}
p.sources.registered_source_identity=lambda path:i if path==source else None
b=RawPreparationHostBinding.model_validate_json((root/"delivery-binding.json").read_text())
verify_consumed_host_review(checkpoint=c,binding=b,scope=s)
assert p.read_verified_projection(root=root,checkpoint_sha256=c.checkpoint_sha256,binding=b,scope=s)==b"id,value\\n1,2\\n"
'''
    restarted = subprocess.run([sys.executable, "-c", restart_code, str(outputs[0].parent.parent), str(source)],
                               env=os.environ.copy(), capture_output=True, text=True)
    assert restarted.returncode == 0, restarted.stderr
    for changed in ({**body, "project_id": "different"}, {**body, "session_id": "different"},
                    {**body, "job_id": "different"}, {**body, "path": str(outputs[0])}):
        with pytest.raises(HTTPException):
            route.raw_preparation_projection_download(changed, request())
    with pytest.raises(HTTPException) as error:
        route.raw_preparation_projection_download(body, request("https://evil.example"))
    assert error.value.status_code == 403
    marker = tmp_path / "home" / ".easyicu" / "private-preparation-decisions" / (choice.nonce + ".used")
    old_marker = marker.read_bytes()
    marker.unlink()
    with pytest.raises(HTTPException):
        route.raw_preparation_projection_download(body, request())
    marker.write_bytes(b"{}")
    with pytest.raises(HTTPException):
        route.raw_preparation_projection_download(body, request())
    marker.write_bytes(old_marker)
    host_receipt = marker.with_suffix(".json")
    original_receipt = host_receipt.read_bytes()
    altered_receipt = json.loads(original_receipt)
    altered_receipt["decision"] = "rejected"
    host_receipt.write_text(json.dumps(altered_receipt))
    with pytest.raises(HTTPException):
        route.raw_preparation_projection_download(body, request())
    host_receipt.write_bytes(original_receipt)
    monkeypatch.setattr(projection.sources, "registered_source_identity", lambda _: None)
    with pytest.raises(HTTPException):
        route.raw_preparation_projection_download(body, request())
    monkeypatch.setattr(projection.sources, "registered_source_identity", lambda path: identity if path == str(source) else None)
    source_snapshot = outputs[0].parent.parent / "snapshot" / "table.csv"
    before_snapshot = source_snapshot.read_bytes()
    source_snapshot.write_bytes(before_snapshot + b"changed")
    with pytest.raises(HTTPException):
        route.raw_preparation_projection_download(body, request())
    source_snapshot.write_bytes(before_snapshot)
    output = outputs[0]
    original_output = output.read_bytes()
    output.write_bytes(original_output + b"changed")
    with pytest.raises(HTTPException):
        route.raw_preparation_projection_download(body, request())
