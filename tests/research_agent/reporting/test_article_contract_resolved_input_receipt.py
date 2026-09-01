from __future__ import annotations

import hashlib
import json
from pathlib import Path

from easyicu.research_agent.reporting.article_contract import (
    _verified_resolved_input_bindings,
)


def _write_receipt(run_dir: Path, *, step_id: str = "calibration") -> tuple[dict, dict]:
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    artifact = evidence_dir / "prediction_scores.csv"
    artifact.write_text("risk,outcome\n0.2,0\n", encoding="utf-8")
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    evidence_id = "table_prediction_scores"
    evidence = {
        "evidence_id": evidence_id,
        "relative_path": "evidence/prediction_scores.csv",
        "sha256": artifact_sha,
        "produced_by_step": "primary_model",
    }
    binding = {
        "declared_kind": "table",
        "product": "prediction_scores",
        "evidence_id": evidence_id,
        "sha256": artifact_sha,
        "produced_by_step": "primary_model",
    }
    binding["identity_row"] = {
        "input_key": "table:prediction_scores",
        **binding,
    }
    payload = {
        "schema_version": "2.1",
        "step_id": step_id,
        "inputs": {"table:prediction_scores": binding},
    }
    receipt = run_dir / "resolved_inputs" / f"{step_id}.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    record = {
        "step_id": step_id,
        "resolved_inputs_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "resolved_input_evidence_ids": [evidence_id],
    }
    return record, evidence


def test_canonical_receipt_is_verified_after_resume_removes_path(tmp_path: Path) -> None:
    record, evidence = _write_receipt(tmp_path)

    bindings = _verified_resolved_input_bindings(
        record=record,
        run_dir=tmp_path,
        evidence_by_id={evidence["evidence_id"]: evidence},
    )

    assert bindings is not None
    assert bindings["table:prediction_scores"]["produced_by_step"] == "primary_model"


def test_canonical_receipt_fallback_still_fails_closed_on_digest_drift(
    tmp_path: Path,
) -> None:
    record, evidence = _write_receipt(tmp_path)
    (tmp_path / "resolved_inputs" / "calibration.json").write_text(
        "{}", encoding="utf-8"
    )

    assert (
        _verified_resolved_input_bindings(
            record=record,
            run_dir=tmp_path,
            evidence_by_id={evidence["evidence_id"]: evidence},
        )
        is None
    )


def test_canonical_receipt_fallback_rejects_unsafe_step_id(tmp_path: Path) -> None:
    record, evidence = _write_receipt(tmp_path)
    record["step_id"] = "../calibration"

    assert (
        _verified_resolved_input_bindings(
            record=record,
            run_dir=tmp_path,
            evidence_by_id={evidence["evidence_id"]: evidence},
        )
        is None
    )
