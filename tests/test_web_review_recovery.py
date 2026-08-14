"""Private Web plan-review recovery contracts."""

from __future__ import annotations

import json

import pytest

from easyicu.webserver.agent_review_recovery import (
    WebReviewRecoveryError,
    WebReviewRecoveryRecord,
    get_record,
    put_record,
    remove_record,
)


def _record(run_id: str = "run_a") -> WebReviewRecoveryRecord:
    return WebReviewRecoveryRecord.create(
        run_id=run_id,
        wrapper_dir="/private/project/run_job",
        study={"id": "study_a", "question": "A question"},
        scientific_configuration_sha256="a" * 64,
        provider_meta={"provider": "openai", "external": True},
        provider_public={"provider": "openai", "model": "model-a"},
        credential_source="pi_verified",
        pipeline_config={"workdir": "/private/project/run_job/pipeline"},
        acquisition_projection={"selected_concepts": ["age"]},
        hard_stop_ledger_path="/private/project/run_job/.runtime/ledger.json",
        hard_stop_task_id="web-job-a",
        hard_stop_declaration_sha256="b" * 64,
        created_at=1.0,
    )


def test_recovery_record_round_trips_and_is_removable(tmp_path) -> None:
    path = tmp_path / "review-index.json"
    put_record(_record(), path=path)

    loaded = get_record("run_a", path=path)
    assert loaded == _record()
    assert path.stat().st_mode & 0o777 == 0o600

    remove_record("run_a", path=path)
    assert get_record("run_a", path=path) is None


def test_recovery_record_tampering_fails_closed(tmp_path) -> None:
    path = tmp_path / "review-index.json"
    put_record(_record(), path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"]["run_a"]["study"]["question"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WebReviewRecoveryError, match="corrupt"):
        get_record("run_a", path=path)


def test_recovery_index_never_silently_evicts_a_pending_pause(tmp_path) -> None:
    path = tmp_path / "review-index.json"
    put_record(_record("old"), path=path, max_records=1)
    newer = _record("new").model_copy(update={"created_at": 2.0})
    newer = WebReviewRecoveryRecord.create(
        **newer.model_dump(exclude={"record_sha256"})
    )
    with pytest.raises(WebReviewRecoveryError, match="capacity"):
        put_record(newer, path=path, max_records=1)

    assert get_record("old", path=path) is not None
    assert get_record("new", path=path) is None
