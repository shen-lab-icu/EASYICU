from __future__ import annotations

from pathlib import Path

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
