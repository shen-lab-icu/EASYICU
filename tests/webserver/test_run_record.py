from __future__ import annotations

from pathlib import Path
import json

import pytest

from easyicu.webserver.run_record import (
    RunDirectory,
    RunRecord,
    RunRecordReadError,
)


def _record(tmp_path: Path) -> RunRecord:
    directory = RunDirectory.create(tmp_path, "study alpha", "job-1")
    return RunRecord.build(
        directory=directory,
        run_context={"run_id": "run_job-1", "study_id": "study alpha", "mode": "analysis"},
        ledger={"run_type": "full"},
        gate_payload={
            "status": "analysis_only",
            "reason": "human_signoff_required",
            "reportable": False,
            "draft_unlocked": False,
            "checks": [{"id": "human_signoff", "passed": False}],
        },
        readiness_payload={
            "status": "awaiting_human_signoff",
            "signable": True,
            "checks_total": 1,
            "checks_passed": 0,
            "required_confirmations": ["evidence_reviewed"],
        },
        signoff_payload=None,
        signoff_integrity={"status": "unsigned", "signoff_stale": False},
        artifacts=[
            {
                "name": "quality_gate.json",
                "path": str(directory.artifact("quality_gate.json")),
                "relative_path": "quality_gate.json",
                "bytes": 20,
                "sha256": "a" * 64,
                "kind": "json",
            }
        ],
        artifact_payloads={"quality_gate.json": {"gate": {"status": "analysis_only"}}},
    )


def test_run_record_exposes_typed_gate_readiness_artifacts_and_layout(tmp_path: Path) -> None:
    record = _record(tmp_path)

    assert record.ok is True
    assert record.directory.path == tmp_path / "study-alpha" / "run_job-1"
    assert record.directory.pipeline_run("run_inner") == (
        record.directory.path / "pipeline" / "run_inner"
    )
    assert record.gate.status == "analysis_only"
    assert record.readiness.signable is True
    assert record.artifacts[0].name == "quality_gate.json"
    assert record.signoff is None


def test_run_record_public_projection_preserves_review_contract(tmp_path: Path) -> None:
    payload = _record(tmp_path).to_dict()

    assert payload["ok"] is True
    assert payload["gate"]["checks"] == [{"id": "human_signoff", "passed": False}]
    assert payload["readiness"]["required_confirmations"] == ["evidence_reviewed"]
    assert payload["artifacts"][0]["name"] == "quality_gate.json"
    assert payload["signed"] is False


def test_run_record_error_is_an_explicit_discriminated_result(tmp_path: Path) -> None:
    result = RunRecordReadError(
        ok=False,
        error="artifact_json_invalid",
        directory=RunDirectory(tmp_path / "run_bad"),
        artifact="quality_gate.json",
    )

    assert result.ok is False
    assert result.to_dict() == {
        "ok": False,
        "error": "artifact_json_invalid",
        "project_dir": str(tmp_path / "run_bad"),
        "artifact": "quality_gate.json",
    }


def test_run_directory_rejects_paths_disguised_as_artifact_names(tmp_path: Path) -> None:
    directory = RunDirectory(tmp_path / "run_safe")

    with pytest.raises(ValueError, match="run_artifact_name_invalid"):
        directory.artifact("../quality_gate.json")


def test_run_record_nested_state_is_immutable_and_projections_are_detached(tmp_path):
    record = _record(tmp_path)
    with pytest.raises(TypeError):
        record.gate.checks[0]["passed"] = True
    with pytest.raises(TypeError):
        record.artifact_payloads["quality_gate.json"]["gate"]["status"] = "reportable"
    wire = record.to_dict()
    wire["gate"]["checks"][0]["passed"] = True
    wire["artifact_payloads"]["quality_gate.json"]["gate"]["status"] = "reportable"
    assert record.gate.checks[0]["passed"] is False
    assert record.artifact_payloads["quality_gate.json"]["gate"]["status"] == "analysis_only"
    json.dumps(record.to_dict())


def _write_review_files(tmp_path):
    for name, payload in {
        "run_context.json": {"run_id": "run_a"},
        "evidence_ledger.json": {"run_id": "run_a"},
        "quality_gate.json": {"gate": {"status": "analysis_only", "checks": []}},
    }.items():
        (tmp_path / name).write_text(json.dumps(payload))


def test_review_refuses_mixed_run_identities(tmp_path):
    from easyicu.webserver.agent_runs import read_run_record
    _write_review_files(tmp_path)
    (tmp_path / "evidence_ledger.json").write_text(json.dumps({"run_id": "run_b"}))
    record = read_run_record(str(tmp_path))
    assert isinstance(record, RunRecordReadError)
    assert record.error == "run_record_identity_conflict"


def test_review_refuses_content_changed_between_payload_and_inventory(tmp_path, monkeypatch):
    from easyicu.webserver import agent_runs
    _write_review_files(tmp_path)
    inventory = agent_runs._run_artifacts
    def changed_inventory(path):
        (path / "quality_gate.json").write_text(json.dumps({"gate": {"status": "new"}}))
        return inventory(path)
    monkeypatch.setattr(agent_runs, "_run_artifacts", changed_inventory)
    record = agent_runs.read_run_record(str(tmp_path))
    assert isinstance(record, RunRecordReadError)
    assert record.error == "run_record_changed_during_read"


def test_signoff_serializes_nested_frozen_gate_checks(tmp_path):
    from easyicu.webserver import agent_runs

    _write_review_files(tmp_path)
    (tmp_path / "quality_gate.json").write_text(json.dumps({"gate": {
        "status": "analysis_only", "checks": [
            {"id": "source", "passed": True, "details": {"artifacts": ["table"]}},
            {"id": "human_signoff", "passed": False},
        ],
    }}))
    result = agent_runs.create_human_signoff(
        str(tmp_path), reviewer="test reviewer",
        confirmations=sorted(agent_runs._SIGNOFF_CONFIRMATIONS),
    )
    assert result["ok"] is True
    assert result["signoff"]["gate_before_signoff"]["checks"][0]["details"] == {"artifacts": ["table"]}
    json.dumps(result)


def test_pending_plan_projection_is_detached_json():
    from types import SimpleNamespace
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
    from easyicu.webserver.agent_pipeline_runs import _pending_plan_authority

    authority = PlanReviewAuthority.create(plan=AnalysisPlan(
        research_question="A clinical question", steps=[AnalysisStep(
            step_id="summary", intent="Describe", method="descriptive",
            inputs=[], expected_outputs=["table:summary"],
        )],
    ))
    pending = SimpleNamespace(requests=[SimpleNamespace(payload={
        "plan_review_authority": authority.model_dump(mode="json"),
    })])
    wire = _pending_plan_authority(pending)
    json.dumps(wire)
    assert isinstance(wire["steps"], list)
    wire["steps"][0]["method"] = "changed"
    assert authority.plan_payload["steps"][0]["method"] == "descriptive"
