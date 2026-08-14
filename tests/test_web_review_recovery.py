"""Private Web plan-review recovery contracts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver.agent_review_recovery import (
    WebReviewRecoveryError,
    WebReviewRecoveryRecord,
    WebReviewRecoverySeed,
    get_record,
    put_record,
    put_recovery_seed,
    reconcile_records,
    register_pipeline_work_root,
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
        pipeline_config_sha256="c" * 64,
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


def test_recovery_index_concurrent_updates_do_not_lose_records(tmp_path) -> None:
    path = tmp_path / "review-index.json"
    run_ids = [f"run_{index:03d}" for index in range(64)]

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda run_id: put_record(_record(run_id), path=path), run_ids))

    with ThreadPoolExecutor(max_workers=16) as pool:
        loaded = list(pool.map(lambda run_id: get_record(run_id, path=path), run_ids))

    assert [record.run_id for record in loaded if record is not None] == run_ids
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["records"]) == set(run_ids)


def _durable_seed(
    root: Path,
    *,
    run_id: str = "run_reconciled",
    study_id: str = "study-a",
) -> tuple[Path, str]:
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.human_review_checkpoint import (
        HumanReviewCheckpoint,
        write_checkpoint,
    )
    from easyicu.research_agent.orchestration.workflow import HumanReviewRequest

    wrapper = root / study_id / "run_job-a"
    pipeline_root = wrapper / "pipeline"
    config = PipelineConfig(workdir=pipeline_root)
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review this plan.",
        authority_sha256="a" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    checkpoint = HumanReviewCheckpoint.create(
        run_id=run_id,
        pipeline_config_sha256=config.canonical_digest(),
        environment_identity={},
        llm_signature_sha256="c" * 64,
        run_input_capsule_sha256="d" * 64,
        capability_activation_sha256="e" * 64,
        runtime_capabilities=(),
        runtime_bundle=None,
        requests=(request,),
        plan_handoff={},
        execution_coordinates={},
    )
    run_dir = pipeline_root / run_id
    write_checkpoint(run_dir / "human_review_checkpoint.json", checkpoint)
    seed = WebReviewRecoverySeed.create(
        wrapper_dir=str(wrapper.resolve()),
        study={"id": study_id, "question": "A question"},
        scientific_configuration_sha256="a" * 64,
        provider_meta={"provider": "openai", "external": True},
        provider_public={"provider": "openai", "model": "model-a"},
        credential_source="pi_verified",
        pipeline_config=config.recovery_payload(),
        pipeline_config_sha256=config.canonical_digest(),
        acquisition_projection={"selected_concepts": ["age"]},
        hard_stop_ledger_path=str(wrapper / ".runtime" / "ledger.json"),
        hard_stop_task_id="web-job-a",
        hard_stop_declaration_sha256="b" * 64,
        created_at=1.0,
    )
    put_recovery_seed(seed)
    return wrapper, run_id


def test_missing_global_record_is_reconciled_from_exact_durable_checkpoint(
    tmp_path,
) -> None:
    index = tmp_path / "review-index.json"
    root = tmp_path / "projects"
    _wrapper, run_id = _durable_seed(root)
    register_pipeline_work_root(root, path=index)

    loaded = get_record(run_id, path=index)

    assert loaded is not None
    assert loaded.run_id == run_id
    assert loaded.pipeline_config_sha256


def test_reconciliation_rejects_a_tampered_local_seed(tmp_path) -> None:
    index = tmp_path / "review-index.json"
    root = tmp_path / "projects"
    wrapper, run_id = _durable_seed(root)
    register_pipeline_work_root(root, path=index)
    seed_path = wrapper / ".runtime" / "web_review_recovery_seed.json"
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    payload["study"]["question"] = "tampered"
    seed_path.write_text(json.dumps(payload), encoding="utf-8")

    assert reconcile_records(path=index) == 0
    assert get_record(run_id, path=index) is None


def test_reconciliation_scan_is_bounded(tmp_path) -> None:
    index = tmp_path / "review-index.json"
    root = tmp_path / "projects"
    _durable_seed(root, run_id="run_first", study_id="study-first")
    _durable_seed(root, run_id="run_second", study_id="study-second")
    register_pipeline_work_root(root, path=index)

    reconcile_records(path=index, max_candidates=1)
    payload = json.loads(index.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 1


def test_windows_lock_fallback_serializes_without_fcntl(
    tmp_path, monkeypatch
) -> None:
    import easyicu.webserver.agent_review_recovery as recovery

    calls = []

    class _Msvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(_descriptor, mode, length):
            calls.append((mode, length))

    monkeypatch.setattr(recovery, "fcntl", None)
    monkeypatch.setattr(recovery, "msvcrt", _Msvcrt)
    put_record(_record(), path=tmp_path / "windows-index.json")

    assert calls == [(_Msvcrt.LK_LOCK, 1), (_Msvcrt.LK_UNLCK, 1)]


@pytest.mark.parametrize(
    "checkpoint_state",
    ["pending", "approved_pending_execution", "executing"],
)
def test_web_post_approval_recovery_reuses_exact_stored_decisions(
    tmp_path,
    monkeypatch,
    checkpoint_state,
) -> None:
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.orchestration.human_review_checkpoint import (
        HumanReviewCheckpoint,
        write_checkpoint,
    )
    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewPending,
        HumanReviewRequest,
    )
    from easyicu.webserver import agent_pipeline_runs

    run_id = f"run-{checkpoint_state}"
    run_dir = tmp_path / "pipeline" / run_id
    run_dir.mkdir(parents=True)
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review this plan.",
        authority_sha256="a" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    decisions = [
        {
            "review_id": request.review_id,
            "authority_sha256": request.authority_sha256,
            "decision": "approved",
            "reviewer": "original reviewer",
            "decided_at": "2026-08-14T01:02:03Z",
            "note": "original note",
        }
    ]
    records = [
        {
            "review_id": request.review_id,
            "decision": "approved",
            "server_decided_at": "2026-08-14T01:02:04+00:00",
        }
    ]
    checkpoint = HumanReviewCheckpoint.create(
        run_id=run_id,
        pipeline_config_sha256="b" * 64,
        environment_identity={},
        llm_signature_sha256="c" * 64,
        run_input_capsule_sha256="d" * 64,
        capability_activation_sha256="e" * 64,
        runtime_capabilities=(),
        runtime_bundle=None,
        requests=(request,),
        plan_handoff={},
        execution_coordinates={},
    ).decision_recorded(
        decisions=decisions,
        decision_records=records,
        decision_sha256=canonical_sha256(decisions),
    )
    if checkpoint_state != "pending":
        checkpoint = checkpoint.decision_committed()
    if checkpoint_state == "executing":
        checkpoint = checkpoint.execution_started()
    write_checkpoint(run_dir / "human_review_checkpoint.json", checkpoint)
    recovery_values = _record(run_id).model_dump(exclude={"record_sha256"})
    recovery_values["wrapper_dir"] = str(tmp_path)
    recovered_pending = agent_pipeline_runs._checkpoint_pending_from_record(
        WebReviewRecoveryRecord.create(**recovery_values)
    )
    assert recovered_pending.run_id == run_id

    captured = {}

    class _Pipeline:
        def resume_human_review(self, submitted, **_kwargs):
            captured["decisions"] = submitted
            return SimpleNamespace(manifest_path=run_dir / "manifest.json")

    pending = HumanReviewPending(
        run_id=run_id,
        thread_id=run_id,
        run_dir=str(run_dir),
        requests=(request,),
        resume_scope="durable_checkpoint",
        resume_pid=None,
    )
    entry = agent_pipeline_runs._PendingRun(
        pipeline=_Pipeline(),
        pending=pending,
        wrapper_dir=tmp_path,
        study={"id": "study-a"},
        provider={},
        acquisition=SimpleNamespace(),
        created_at=1.0,
    )
    monkeypatch.setitem(agent_pipeline_runs._PENDING, run_id, entry)
    monkeypatch.setattr(agent_pipeline_runs, "remove_review_recovery_record", lambda *_: None)
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_write_projection",
        lambda **_kwargs: {"status": "complete"},
    )

    result = agent_pipeline_runs.resume_research_pipeline(
        run_id=run_id,
        study_context_id="study-a",
        decision="approved",
        reviewer="different retry reviewer",
        note="different retry note",
        job=SimpleNamespace(emit=lambda _event: None),
    )

    assert result == {"status": "complete"}
    assert captured["decisions"] == decisions
