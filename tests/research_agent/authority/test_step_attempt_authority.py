from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from easyicu.research_agent.authority.step_attempt import (
    CheckpointAuthority,
    StepAttemptState,
    StepAuthorityOperations,
)
from easyicu.research_agent.authority.step_capsule import (
    ContentRef,
    StepAuthorityCapsuleRef,
)


def _ref(marker: str) -> StepAuthorityCapsuleRef:
    return StepAuthorityCapsuleRef(
        step_id="01_summary",
        capsule_sha256=marker * 64,
    )


def _code_ref(code: str) -> ContentRef:
    payload = code.encode("utf-8")
    return ContentRef(
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        media_type="text/x-python",
    )


def _authority(
    tmp_path: Path,
    *,
    state: StepAttemptState,
    step_record: dict,
    records: list[dict],
    history: list[dict],
    flush,
    load_verified,
    persist_candidate=lambda _coordinates, code: _code_ref(code),
    seal_deterministic=lambda *_args, **_kwargs: _ref("c"),
    seal_legacy=lambda *_args, **_kwargs: _ref("d"),
    seal_initial=lambda *_args, **_kwargs: _ref("e"),
    seal_repair=lambda *_args, **_kwargs: _ref("f"),
    receipt=None,
) -> CheckpointAuthority:
    def upsert(target: list[dict], snapshot: dict) -> None:
        target.append(dict(snapshot))

    return CheckpointAuthority(
        run_dir=tmp_path,
        step_id="01_summary",
        state=state,
        step_record=step_record,
        per_step_records=records,
        step_attempt_history=history,
        shared_lock=threading.RLock(),
        flush_partial_manifest=flush,
        upsert_checkpoint=upsert,
        provider_receipt_path=tmp_path / "provider.json",
        reserved_final_category="concept_audit",
        sync_provider_budget=lambda: None,
        operations=StepAuthorityOperations(
            load_verified_capsule=load_verified,
            persist_candidate_code=persist_candidate,
            seal_deterministic_candidate=seal_deterministic,
            seal_legacy_candidate=seal_legacy,
            seal_initial_candidate=seal_initial,
            seal_repair_candidate=seal_repair,
            load_provider_receipt=(lambda *_args, **_kwargs: receipt),
        ),
    )


def _verified(ref: StepAuthorityCapsuleRef, code_ref: ContentRef, stage="candidate"):
    return SimpleNamespace(
        ref=ref,
        capsule=SimpleNamespace(stage=stage, candidate_code=code_ref),
        candidate_code="",
    )


def test_checkpoint_write_failure_restores_all_mutable_authority_state(tmp_path):
    parent = _ref("a")
    child = _ref("b")
    parent_code = _code_ref("value = 1\n")
    state = StepAttemptState(current_capsule_ref=parent)
    step_record = {
        "step_id": "01_summary",
        "attempt_id": "run:01_summary:1",
        "step_authority_capsule_ref": parent.model_dump(mode="json"),
        "step_authority_capsule_stage": "candidate",
        "capsule_pending_repair_attempt_id": 1,
    }
    records = [{"step_id": "00_probe", "status": "ok"}]
    history = [{"step_id": "00_probe", "status": "ok"}]
    expected_record = dict(step_record)
    expected_records = list(records)
    expected_history = list(history)

    def flush() -> None:
        history.extend(dict(item) for item in records)
        raise OSError("simulated checkpoint write failure")

    authority = _authority(
        tmp_path,
        state=state,
        step_record=step_record,
        records=records,
        history=history,
        flush=flush,
        load_verified=lambda *_args, **_kwargs: _verified(child, parent_code),
    )

    with pytest.raises(OSError, match="checkpoint write failure"):
        authority.checkpoint_capsule(
            child,
            status="candidate_checkpointed",
            extra={"new_marker": True},
            delete_fields=("capsule_pending_repair_attempt_id",),
        )

    assert state.current_capsule_ref == parent
    assert step_record == expected_record
    assert records == expected_records
    assert history == expected_history


def test_same_code_candidate_reuses_current_capsule_without_checkpoint(tmp_path):
    code = "value = 1\n"
    code_ref = _code_ref(code)
    current_ref = _ref("a")
    state = StepAttemptState(
        coordinates=SimpleNamespace(),
        current_capsule_ref=current_ref,
    )
    step_record = {"step_id": "01_summary", "attempt_id": "run:01_summary:1"}
    records: list[dict] = []
    history: list[dict] = []
    flush_calls = 0

    def flush() -> None:
        nonlocal flush_calls
        flush_calls += 1

    def unexpected(*_args, **_kwargs):
        raise AssertionError("same-code reuse must not seal a new capsule")

    authority = _authority(
        tmp_path,
        state=state,
        step_record=step_record,
        records=records,
        history=history,
        flush=flush,
        load_verified=lambda *_args, **_kwargs: _verified(current_ref, code_ref),
        seal_deterministic=unexpected,
        seal_legacy=unexpected,
    )

    assert authority.ensure_candidate(code, reason="normalization") == current_ref
    assert state.current_capsule_ref == current_ref
    assert flush_calls == 0
    assert records == []


def test_rejected_repair_restores_exact_parent_then_clears_latch(tmp_path):
    parent = _ref("a")
    child = _ref("b")
    repaired_code = "value = 2\n"
    repaired_ref = _code_ref(repaired_code)
    state = StepAttemptState(
        coordinates=SimpleNamespace(),
        current_capsule_ref=parent,
    )
    step_record = {
        "step_id": "01_summary",
        "attempt_id": "run:01_summary:1",
        "capsule_pending_repair_attempt_id": 1,
        "capsule_pending_repair_binding_sha256": "binding",
        "capsule_pending_repair_failure_status": "runtime_failed",
    }
    records: list[dict] = []
    history: list[dict] = []
    receipt = SimpleNamespace(logical_repairs=[{"transport": {"state": "completed"}}])
    authority = _authority(
        tmp_path,
        state=state,
        step_record=step_record,
        records=records,
        history=history,
        flush=lambda: None,
        load_verified=lambda _run_dir, *, ref, **_kwargs: _verified(ref, repaired_ref),
        seal_repair=lambda *_args, **_kwargs: child,
        receipt=receipt,
    )

    authority.seal_completed_repair_candidate(
        repaired_ref,
        1,
        failure_status="runtime_failed",
    )
    assert state.current_capsule_ref == child
    assert state.last_completed_repair_parent_ref == parent
    assert state.last_completed_repair_child_ref == child
    assert state.last_completed_repair_code_sha256 == repaired_ref.sha256

    authority.reject_completed_repair_candidate(
        repaired_code,
        reason="runtime_repair_semantic_noop",
    )
    assert state.current_capsule_ref == parent
    assert state.last_completed_repair_parent_ref is None
    assert state.last_completed_repair_child_ref is None
    assert state.last_completed_repair_code_sha256 is None
    assert step_record["step_authority_rejected_repair_candidate"] == (
        "runtime_repair_semantic_noop"
    )
    record_count = len(records)

    authority.reject_completed_repair_candidate(
        repaired_code,
        reason="must_not_reapply",
    )
    assert len(records) == record_count
    assert state.current_capsule_ref == parent


def test_repair_rejection_checkpoint_failure_keeps_child_and_latch(tmp_path):
    parent = _ref("a")
    child = _ref("b")
    repaired_code = "value = 2\n"
    repaired_ref = _code_ref(repaired_code)
    state = StepAttemptState(
        coordinates=SimpleNamespace(),
        current_capsule_ref=child,
        last_completed_repair_parent_ref=parent,
        last_completed_repair_child_ref=child,
        last_completed_repair_code_sha256=repaired_ref.sha256,
    )
    step_record = {
        "step_id": "01_summary",
        "attempt_id": "run:01_summary:1",
        "step_authority_capsule_ref": child.model_dump(mode="json"),
        "step_authority_capsule_stage": "candidate",
    }
    records = [dict(step_record, status="candidate_checkpointed")]
    history = [dict(records[0])]
    expected_record = dict(step_record)
    expected_records = list(records)
    expected_history = list(history)
    authority = _authority(
        tmp_path,
        state=state,
        step_record=step_record,
        records=records,
        history=history,
        flush=lambda: (_ for _ in ()).throw(OSError("write failed")),
        load_verified=lambda _run_dir, *, ref, **_kwargs: _verified(ref, repaired_ref),
    )

    with pytest.raises(OSError, match="write failed"):
        authority.reject_completed_repair_candidate(
            repaired_code,
            reason="runtime_repair_semantic_noop",
        )

    assert state.current_capsule_ref == child
    assert state.last_completed_repair_parent_ref == parent
    assert state.last_completed_repair_child_ref == child
    assert state.last_completed_repair_code_sha256 == repaired_ref.sha256
    assert step_record == expected_record
    assert records == expected_records
    assert history == expected_history


def test_stale_repair_rejection_cannot_rollback_newer_current_capsule(tmp_path):
    parent = _ref("a")
    repaired_child = _ref("b")
    newer = _ref("c")
    repaired_code = "value = 2\n"
    repaired_ref = _code_ref(repaired_code)
    state = StepAttemptState(
        coordinates=SimpleNamespace(),
        current_capsule_ref=parent,
    )
    step_record = {
        "step_id": "01_summary",
        "attempt_id": "run:01_summary:1",
        "capsule_pending_repair_attempt_id": 1,
        "capsule_pending_repair_binding_sha256": "binding",
        "capsule_pending_repair_failure_status": "runtime_failed",
    }
    records: list[dict] = []
    history: list[dict] = []
    receipt = SimpleNamespace(logical_repairs=[{"transport": {"state": "completed"}}])
    authority = _authority(
        tmp_path,
        state=state,
        step_record=step_record,
        records=records,
        history=history,
        flush=lambda: None,
        load_verified=lambda _run_dir, *, ref, **_kwargs: _verified(
            ref,
            repaired_ref,
        ),
        seal_repair=lambda *_args, **_kwargs: repaired_child,
        receipt=receipt,
    )

    authority.seal_completed_repair_candidate(
        repaired_ref,
        1,
        failure_status="runtime_failed",
    )
    authority.checkpoint_capsule(newer, status="candidate_checkpointed")
    record_count = len(records)

    authority.reject_completed_repair_candidate(
        repaired_code,
        reason="stale_runtime_repair_rejection",
    )

    assert state.current_capsule_ref == newer
    assert state.last_completed_repair_parent_ref == parent
    assert state.last_completed_repair_code_sha256 == repaired_ref.sha256
    assert len(records) == record_count
    assert "step_authority_rejected_repair_candidate" not in step_record


def test_repair_child_checkpoint_failure_preserves_parent_and_pending(tmp_path):
    parent = _ref("a")
    child = _ref("b")
    repaired_code = "value = 2\n"
    repaired_ref = _code_ref(repaired_code)
    state = StepAttemptState(
        coordinates=SimpleNamespace(),
        current_capsule_ref=parent,
    )
    step_record = {
        "step_id": "01_summary",
        "attempt_id": "run:01_summary:1",
        "capsule_pending_repair_attempt_id": 1,
        "capsule_pending_repair_binding_sha256": "binding",
        "capsule_pending_repair_failure_status": "runtime_failed",
    }
    records: list[dict] = []
    history: list[dict] = []
    expected_record = dict(step_record)
    receipt = SimpleNamespace(logical_repairs=[{"transport": {"state": "completed"}}])
    authority = _authority(
        tmp_path,
        state=state,
        step_record=step_record,
        records=records,
        history=history,
        flush=lambda: (_ for _ in ()).throw(OSError("write failed")),
        load_verified=lambda _run_dir, *, ref, **_kwargs: _verified(ref, repaired_ref),
        seal_repair=lambda *_args, **_kwargs: child,
        receipt=receipt,
    )

    with pytest.raises(OSError, match="write failed"):
        authority.seal_completed_repair_candidate(
            repaired_ref,
            1,
            failure_status="runtime_failed",
        )

    assert state.current_capsule_ref == parent
    assert state.last_completed_repair_parent_ref is None
    assert state.last_completed_repair_child_ref is None
    assert state.last_completed_repair_code_sha256 is None
    assert step_record == expected_record
    assert records == []
    assert history == []
