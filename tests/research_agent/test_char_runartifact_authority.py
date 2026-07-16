"""Freeze-baseline characterization for run-artifact authority (G2)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _write_checkpoint(path: Path, *, sequence: int | None, status: str) -> None:
    payload: dict[str, object] = {
        "per_step_records": [
            {
                "step_id": "01_model",
                "status": status,
                "evidence_ids": [f"result_{status}"],
            }
        ]
    }
    if sequence is not None:
        payload["checkpoint_sequence"] = sequence
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_highest_checkpoint_sequence_is_current_even_with_older_mtime(
    tmp_path: Path,
):
    from easyicu.research_agent.runtime_artifacts import load_run_artifact_authority

    partial = tmp_path / "manifest_partial.json"
    final = tmp_path / "manifest.json"
    _write_checkpoint(partial, sequence=7, status="ok")
    _write_checkpoint(final, sequence=8, status="contract_failed")
    os.utime(final, ns=(1_000_000_000, 1_000_000_000))
    os.utime(partial, ns=(2_000_000_000, 2_000_000_000))

    authority = load_run_artifact_authority(tmp_path)

    assert authority is not None
    assert authority["checkpoint_sequence"] == 8
    assert authority["per_step_records"][-1]["status"] == "contract_failed"


def test_corrupt_newest_checkpoint_fails_closed_without_old_success_fallback(
    tmp_path: Path,
):
    from easyicu.research_agent.runtime_artifacts import (
        RunArtifactAuthorityError,
        load_run_artifact_authority,
    )

    final = tmp_path / "manifest.json"
    partial = tmp_path / "manifest_partial.json"
    _write_checkpoint(final, sequence=1, status="ok")
    partial.write_text("{not-json", encoding="utf-8")
    os.utime(final, ns=(1_000_000_000, 1_000_000_000))
    os.utime(partial, ns=(2_000_000_000, 2_000_000_000))

    with pytest.raises(
        RunArtifactAuthorityError,
        match=r"newest checkpoint manifest_partial\.json is corrupt or unreadable",
    ):
        load_run_artifact_authority(tmp_path)


def test_new_ledgerless_boundary_cannot_replay_an_older_ledger(tmp_path: Path):
    from easyicu.research_agent.runtime_artifacts import (
        RunArtifactAuthorityError,
        load_run_artifact_authority,
    )

    final = tmp_path / "manifest.json"
    partial = tmp_path / "manifest_partial.json"
    _write_checkpoint(final, sequence=None, status="ok")
    partial.write_text(json.dumps({"evidence": []}), encoding="utf-8")
    os.utime(final, ns=(1_000_000_000, 1_000_000_000))
    os.utime(partial, ns=(2_000_000_000, 2_000_000_000))

    with pytest.raises(
        RunArtifactAuthorityError,
        match=r"newest checkpoint manifest_partial\.json does not declare per_step_records",
    ):
        load_run_artifact_authority(tmp_path)


def test_single_valid_preledger_manifest_keeps_legacy_none_signal(tmp_path: Path):
    from easyicu.research_agent.runtime_artifacts import load_run_artifact_authority

    (tmp_path / "manifest.json").write_text(
        json.dumps({"evidence": []}),
        encoding="utf-8",
    )

    assert load_run_artifact_authority(tmp_path) is None


def test_current_evidence_maps_and_run_level_claim_owner_are_exact():
    from easyicu.research_agent.runtime_artifacts import (
        active_step_evidence_ids_by_step,
        run_level_evidence_matches_claim_owner,
    )

    ledger = [
        {
            "step_id": "01_model",
            "status": "ok",
            "evidence_ids": ["model_v1", "table_v1"],
        },
        {
            "step_id": "02_failed",
            "status": "ok",
            "evidence_ids": ["old_result"],
        },
        {
            "step_id": "02_failed",
            "status": "contract_failed",
            "evidence_ids": ["failed_result"],
        },
    ]

    assert active_step_evidence_ids_by_step(ledger) == {
        "01_model": {"model_v1", "table_v1"}
    }
    assert run_level_evidence_matches_claim_owner(
        claim_step_id="research_context",
        evidence_id="research_context",
    )
    assert run_level_evidence_matches_claim_owner(
        claim_step_id="research_context",
        evidence_id="research_context_v2",
    )
    assert not run_level_evidence_matches_claim_owner(
        claim_step_id="research_context",
        evidence_id="research_context_v1",
    )
    assert not run_level_evidence_matches_claim_owner(
        claim_step_id="research_context",
        evidence_id="other_context_v2",
    )
