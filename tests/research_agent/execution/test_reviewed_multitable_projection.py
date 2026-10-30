"""Host-bound projections retain the exact input/output mapping, not a cohort."""
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import reviewed_multitable_projection as projection
from easyicu.webserver.host_action_contracts import (
    HostActionRequest, HostActionTag, RawPreparationReviewDecision, decision_key, host_action_id,
)
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, record_host_review


@pytest.fixture
def example(tmp_path, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs, jobs
    from easyicu.webserver.pi_copilot import service

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    source = tmp_path / "source"
    (source / "cases").mkdir(parents=True)
    (source / "lab").mkdir()
    pq.write_table(pa.table({"case": [1, 2], "age": [65, 71], "private": ["x", "y"]}), source / "cases" / "a.parquet")
    table = pa.table({"case": [1, 1, 2], "value": [2., 3., 4.], "private": ["a", "b", "c"]})
    # Arbitrary source metadata can include names of unapproved fields.
    table = table.replace_schema_metadata({b"unapproved": b"private"})
    pq.write_table(table, source / "lab" / "b.parquet", row_group_size=1)
    identity = SourceIdentity(database="sic", registration_id="source-one")
    monkeypatch.setattr(projection.sources, "load_registry", lambda: {"sources": [{"path": str(source), "ok": True}]})
    monkeypatch.setattr(projection.sources, "registered_source_identity", lambda p: identity if p == str(source) else None)
    scope = {"adapter_id": "registered-parquet-projections/1", "source_identity": identity.to_dict(),
             "inputs": [{"table": "cases", "path": "cases/a.parquet", "columns": ["case", "age"]},
                        {"table": "labs", "path": "lab/b.parquet", "columns": ["value", "case"]}],
             "purpose": "prepare_multiple_tables_only", "plan_sha256": "c" * 64,
             "dependencies_sha256": "d" * 64}
    payload = {"raw_preparation_scope": scope}
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Fixture: read two complete files",
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
    approved = record_host_review(host_action=action, pending=pending, decision="approved", study_context_id="study",
        job_id="job", run_id="run", scientific_configuration_sha256="a" * 64)
    binding = RawPreparationHostBinding(project_id="project", session_id="session", study_context_id="study",
                                        run_id="run", nonce=choice.nonce)
    return {"scope": scope, "checkpoint": approved, "binding": binding, "host_work_root": tmp_path}


def test_multiple_tables_retains_exact_source_and_field_lineage(example):
    output = projection.run_reviewed_multitable_projection(**example)
    first, second = [pq.read_table(output / f"part-{i:04d}.parquet") for i in range(2)]
    assert first.to_pydict() == {"case": [1, 2], "age": [65, 71]}
    assert second.to_pydict() == {"value": [2., 3., 4.], "case": [1, 1, 2]}
    assert second.schema.metadata is None
    body = projection.verify_reviewed_multitable_projection(root=output.parent,
        checkpoint_sha256=example["checkpoint"].checkpoint_sha256, binding=example["binding"], scope=example["scope"])
    rows = body["host_consumption"]["inputs"]
    assert [(r["declared_table"], r["source_path"], r["output"], r["rows"], r["columns"]) for r in rows] == [
        ("cases", "cases/a.parquet", "part-0000.parquet", 2, ["case", "age"]),
        ("labs", "lab/b.parquet", "part-0001.parquet", 3, ["value", "case"])]
    assert all(r["transformation"] == "identity" and r["table_identity_verified"] is False for r in rows)
    assert body["analysis_authorized"] is False and body["read_roster_complete"] is False
    assert body["host_consumption"]["cohort_complete"] is False
    with pytest.raises(ValueError, match="nonce_replayed"):
        projection.run_reviewed_multitable_projection(**example)
    (output / "part-0000.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes_changed"):
        projection.verify_reviewed_multitable_projection(root=output.parent,
            checkpoint_sha256=example["checkpoint"].checkpoint_sha256, binding=example["binding"], scope=example["scope"])


def test_scope_refuses_widening_rebinding_and_malicious_paths(example):
    scope = example["scope"]
    for entries in ([scope["inputs"][0], scope["inputs"][0]],
                    [{"table": "x", "path": "../x.parquet", "columns": ["x"]}],
                    [{"table": "x", "path": "x.parquet", "columns": ["x", "x"]}],
                    [{"table": "x", "path": "x.csv", "columns": ["x"]}]):
        with pytest.raises(ValueError):
            projection.require_reviewable_multitable_scope({**scope, "inputs": entries})
    changed = json.loads(json.dumps(scope))
    changed["inputs"][0]["columns"].append("private")
    with pytest.raises(ValueError, match="host_binding_mismatch"):
        projection.run_reviewed_multitable_projection(**{**example, "scope": changed})


def test_missing_columns_fail_closed_after_approved_scope(example, tmp_path):
    # Retained raw source bytes changed without widening the approved scope.
    path = tmp_path / "source" / "cases" / "a.parquet"
    pq.write_table(pa.table({"case": [1]}), path)
    with pytest.raises(ValueError, match="schema_invalid"):
        projection.run_reviewed_multitable_projection(**example)
    assert not list(tmp_path.glob("reviewed-multitable-projection-*/producer-attestation.json"))
