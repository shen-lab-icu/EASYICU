"""Sidecar lifecycle for the sealed step-result envelope (M7, commit 2).

These tests exercise the producer + loader against a REAL ``EvidenceStore``
and the REAL ``StepEvidenceCommit`` transaction: a committed success is
recoverable, and a rolled-back / unpublished / legacy / tampered record is
never recognised as current authority.  No downstream consumer is wired.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.registration import StepEvidenceCommit
from easyicu.research_agent.authority.runtime_artifacts import (
    load_run_artifact_authority,
)
from easyicu.research_agent.execution.envelope_sidecar import (
    SIDECAR_PRODUCER,
    SIDECAR_SCHEMA_VERSION,
    LoadedStepResultEnvelopeSidecar,
    PreparedStepResultEnvelopeSidecar,
    StepResultEnvelopeSidecarQuery,
    StepResultEnvelopeSidecarUnavailable,
    load_current_step_result_envelope_sidecar,
    prepare_step_result_envelope_sidecar,
    publish_step_result_envelope_sidecar,
    publish_terminal_step_result_envelope_sidecar,
)
from easyicu.research_agent.execution.result_envelope import (
    StepResultEnvelope,
    normalize_step_result_shadow,
)

_STEP_ID = "05_primary_result"


def _script_evidence_id(store: EvidenceStore) -> str:
    record = store.register_text(
        kind="code",
        description="Step script.",
        text="print('analysis')\n",
        filename="step.py",
        produced_by_step=_STEP_ID,
        producer="coder",
        publish_aliases=False,
    )
    return record.evidence_id


def _ready_snapshot(tmp_path: Path, *, status: str = "executed_pending_review"):
    envelope = normalize_step_result_shadow(
        step_id=_STEP_ID,
        step_summary={"status": "completed", "primary_or": "1.5"},
        output_dir=tmp_path,
        status=status,
    )
    return envelope


def _prepared(
    store: EvidenceStore,
    tmp_path: Path,
    *,
    script_evidence_id: str,
    attempt_id: str = "attempt-1",
    checkpoint_id: str = "checkpoint-1",
    terminal_status: str = "ok",
) -> PreparedStepResultEnvelopeSidecar:
    prepared = prepare_step_result_envelope_sidecar(
        snapshot_envelope=_ready_snapshot(tmp_path),
        step_id=_STEP_ID,
        attempt_id=attempt_id,
        checkpoint_id=checkpoint_id,
        script_evidence_id=script_evidence_id,
        terminal_status=terminal_status,
    )
    assert prepared is not None
    return prepared


def _commit_alias(
    store: EvidenceStore,
    *,
    evidence_id: str,
    alias: str,
    register_numeric_claims: Callable[[], None] = lambda: None,
) -> None:
    """Promote one alias through the real success transaction."""

    StepEvidenceCommit(store).commit_validated_step(
        step_id=_STEP_ID,
        pending_aliases={evidence_id: [alias]},
        allowed_evidence_ids=[evidence_id],
        register_numeric_claims=register_numeric_claims,
    )


def _query(
    script_evidence_id: str,
    *,
    step_id: str = _STEP_ID,
    terminal_status: str = "ok",
    attempt_id: str | None = "attempt-1",
    checkpoint_id: str | None = "checkpoint-1",
) -> StepResultEnvelopeSidecarQuery:
    return StepResultEnvelopeSidecarQuery(
        step_id=step_id,
        terminal_status=terminal_status,
        script_evidence_id=script_evidence_id,
        attempt_id=attempt_id,
        checkpoint_id=checkpoint_id,
    )


# ---------------------------------------------------------------------------
# Prepare (pure, fail-closed)
# ---------------------------------------------------------------------------


def test_prepare_binds_terminal_status_and_full_metadata(tmp_path: Path) -> None:
    snapshot = _ready_snapshot(tmp_path, status="executed_pending_review")
    prepared = prepare_step_result_envelope_sidecar(
        snapshot_envelope=snapshot,
        step_id=_STEP_ID,
        attempt_id="attempt-1",
        checkpoint_id="checkpoint-1",
        script_evidence_id="code_step_abcd",
        terminal_status="ok",
    )

    assert prepared is not None
    assert prepared.envelope.status == "ok"
    assert prepared.envelope.shadow is True
    assert prepared.envelope.paper_authorized is False
    md = prepared.metadata
    assert md["sidecar_schema_version"] == SIDECAR_SCHEMA_VERSION
    assert md["step_id"] == _STEP_ID
    assert md["attempt_id"] == "attempt-1"
    assert md["checkpoint_id"] == "checkpoint-1"
    assert md["script_evidence_id"] == "code_step_abcd"
    assert md["envelope_schema_version"] == prepared.envelope.schema_version
    assert md["content_sha256"] == prepared.envelope.content_sha256
    assert md["source_summary_sha256"] == prepared.envelope.source_summary_sha256
    assert md["terminal_status"] == "ok"
    assert md["paper_authorized"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"snapshot_envelope": None},
        {"terminal_status": "contract_failed"},
        {"terminal_status": "executed_pending_review"},
        {"terminal_status": ""},
        {"attempt_id": ""},
        {"checkpoint_id": ""},
        {"script_evidence_id": ""},
    ],
)
def test_prepare_fails_closed_without_publishing(
    tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    base = {
        "snapshot_envelope": _ready_snapshot(tmp_path),
        "step_id": _STEP_ID,
        "attempt_id": "attempt-1",
        "checkpoint_id": "checkpoint-1",
        "script_evidence_id": "code_step_abcd",
        "terminal_status": "ok",
    }
    base.update(kwargs)
    assert prepare_step_result_envelope_sidecar(**base) is None


def test_prepare_rejects_step_identity_disagreement(tmp_path: Path) -> None:
    snapshot = _ready_snapshot(tmp_path)
    assert (
        prepare_step_result_envelope_sidecar(
            snapshot_envelope=snapshot,
            step_id="99_other_step",
            attempt_id="attempt-1",
            checkpoint_id="checkpoint-1",
            script_evidence_id="code_step_abcd",
            terminal_status="ok",
        )
        is None
    )


def test_prepare_performs_no_filesystem_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _ready_snapshot(tmp_path)  # built BEFORE the spies

    def forbid_read_bytes(self: Path) -> bytes:
        raise AssertionError("prepare re-read a file")

    def forbid_open(*args: Any, **kwargs: Any):
        raise AssertionError("prepare opened a file")

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    monkeypatch.setattr(Path, "read_text", forbid_read_bytes)
    monkeypatch.setattr(builtins, "open", forbid_open)
    prepared = prepare_step_result_envelope_sidecar(
        snapshot_envelope=snapshot,
        step_id=_STEP_ID,
        attempt_id="attempt-1",
        checkpoint_id="checkpoint-1",
        script_evidence_id="code_step_abcd",
        terminal_status="ok",
    )
    assert prepared is not None


# ---------------------------------------------------------------------------
# Publish + recover through the real success transaction
# ---------------------------------------------------------------------------


def test_committed_sidecar_is_recoverable_and_outside_raw_outputs(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "run"
    raw_out_dir = tmp_path / "run" / "steps" / _STEP_ID / "out"
    raw_out_dir.mkdir(parents=True)
    store = EvidenceStore(store_root)
    script_id = _script_evidence_id(store)
    prepared = _prepared(store, tmp_path, script_evidence_id=script_id)

    record = publish_step_result_envelope_sidecar(prepared, evidence_store=store)
    _commit_alias(store, evidence_id=record.evidence_id, alias=prepared.alias)

    # Written to the evidence directory, never the raw step output directory.
    assert record.relative_path.startswith("evidence/")
    assert (store_root / record.relative_path).is_file()
    assert not list(raw_out_dir.rglob("*result_envelope_sidecar*"))

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, LoadedStepResultEnvelopeSidecar)
    assert result.envelope.status == "ok"
    assert result.envelope.step_id == _STEP_ID
    assert result.envelope.paper_authorized is False
    assert result.evidence_id == record.evidence_id


def test_fresh_and_resume_use_the_same_loader(tmp_path: Path) -> None:
    store_root = tmp_path / "run"
    store = EvidenceStore(store_root)
    script_id = _script_evidence_id(store)
    prepared = _prepared(store, tmp_path, script_evidence_id=script_id)
    record = publish_step_result_envelope_sidecar(prepared, evidence_store=store)
    _commit_alias(store, evidence_id=record.evidence_id, alias=prepared.alias)

    fresh = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    # Resume reopens the durable store from disk and uses the SAME loader.
    reopened = EvidenceStore(store_root)
    resumed = load_current_step_result_envelope_sidecar(
        evidence_store=reopened, query=_query(script_id)
    )

    assert isinstance(fresh, LoadedStepResultEnvelopeSidecar)
    assert isinstance(resumed, LoadedStepResultEnvelopeSidecar)
    assert resumed.evidence_id == fresh.evidence_id
    assert resumed.envelope.content_sha256 == fresh.envelope.content_sha256


def test_publish_terminal_helper_chains_prepare_and_publish(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "run")
    script_id = _script_evidence_id(store)
    published = publish_terminal_step_result_envelope_sidecar(
        snapshot_envelope=_ready_snapshot(tmp_path),
        step_id=_STEP_ID,
        attempt_id="attempt-1",
        checkpoint_id="checkpoint-1",
        script_evidence_id=script_id,
        terminal_status="ok",
        evidence_store=store,
    )
    assert published is not None
    _commit_alias(store, evidence_id=published.evidence_id, alias=published.alias)

    loaded = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(loaded, LoadedStepResultEnvelopeSidecar)
    assert loaded.evidence_id == published.evidence_id


def test_publish_terminal_helper_publishes_nothing_when_fail_closed(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "run")
    published = publish_terminal_step_result_envelope_sidecar(
        snapshot_envelope=None,
        step_id=_STEP_ID,
        attempt_id="attempt-1",
        checkpoint_id="checkpoint-1",
        script_evidence_id="code_step_abcd",
        terminal_status="ok",
        evidence_store=store,
    )
    assert published is None
    assert not any(r.producer == SIDECAR_PRODUCER for r in store.records())


def test_uncommitted_sidecar_is_not_current_authority(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "run")
    script_id = _script_evidence_id(store)
    prepared = _prepared(store, tmp_path, script_evidence_id=script_id)

    # Registered (bytes on disk) but the success transaction never ran.
    record = publish_step_result_envelope_sidecar(prepared, evidence_store=store)

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "no_committed_alias"
    # The unpublished record itself is allowed to remain on disk.
    assert store.get(record.evidence_id) is not None


def test_rolled_back_commit_leaves_no_current_authority(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "run")
    script_id = _script_evidence_id(store)
    prepared = _prepared(store, tmp_path, script_evidence_id=script_id)
    record = publish_step_result_envelope_sidecar(prepared, evidence_store=store)

    def boom() -> None:
        raise RuntimeError("numeric claim registration failed")

    with pytest.raises(RuntimeError):
        _commit_alias(
            store,
            evidence_id=record.evidence_id,
            alias=prepared.alias,
            register_numeric_claims=boom,
        )

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "no_committed_alias"
    # Rolled back: the alias never became current even though the bytes exist.
    assert prepared.alias not in store.aliases()
    assert store.get(record.evidence_id) is not None


def _commit_attempt(
    store: EvidenceStore,
    tmp_path: Path,
    *,
    script_evidence_id: str,
    attempt_id: str,
    checkpoint_id: str,
) -> tuple[PreparedStepResultEnvelopeSidecar, Any]:
    """Prepare + publish + commit one successful attempt's sidecar."""

    prepared = _prepared(
        store,
        tmp_path,
        script_evidence_id=script_evidence_id,
        attempt_id=attempt_id,
        checkpoint_id=checkpoint_id,
    )
    record = publish_step_result_envelope_sidecar(prepared, evidence_store=store)
    _commit_alias(store, evidence_id=record.evidence_id, alias=prepared.alias)
    return prepared, record


def test_second_successful_attempt_supersedes_first_as_current(
    tmp_path: Path,
) -> None:
    """Identical content, new attempt: a new record and current alias moves."""

    store = EvidenceStore(tmp_path / "run")
    script_id = _script_evidence_id(store)
    prepared1, record1 = _commit_attempt(
        store,
        tmp_path,
        script_evidence_id=script_id,
        attempt_id="attempt-1",
        checkpoint_id="checkpoint-1",
    )
    prepared2, record2 = _commit_attempt(
        store,
        tmp_path,
        script_evidence_id=script_id,
        attempt_id="attempt-2",
        checkpoint_id="checkpoint-2",
    )

    # Same step/script/content, but a DISTINCT record per attempt.
    assert prepared1.alias == prepared2.alias
    assert record1.evidence_id != record2.evidence_id
    assert record1.metadata["content_sha256"] == record2.metadata["content_sha256"]
    # The current alias points at the second attempt.
    assert store.aliases()[prepared2.alias] == record2.evidence_id

    # The first attempt is now stale; only the second is recoverable.
    stale = load_current_step_result_envelope_sidecar(
        evidence_store=store,
        query=_query(script_id, attempt_id="attempt-1", checkpoint_id="checkpoint-1"),
    )
    assert isinstance(stale, StepResultEnvelopeSidecarUnavailable)
    assert stale.reason == "attempt_mismatch"

    loaded = load_current_step_result_envelope_sidecar(
        evidence_store=store,
        query=_query(script_id, attempt_id="attempt-2", checkpoint_id="checkpoint-2"),
    )
    assert isinstance(loaded, LoadedStepResultEnvelopeSidecar)
    assert loaded.evidence_id == record2.evidence_id


def test_rolled_back_third_attempt_keeps_second_as_current(tmp_path: Path) -> None:
    """A third attempt whose commit rolls back leaves the second current."""

    store = EvidenceStore(tmp_path / "run")
    script_id = _script_evidence_id(store)
    _commit_attempt(
        store,
        tmp_path,
        script_evidence_id=script_id,
        attempt_id="attempt-1",
        checkpoint_id="checkpoint-1",
    )
    _prepared2, record2 = _commit_attempt(
        store,
        tmp_path,
        script_evidence_id=script_id,
        attempt_id="attempt-2",
        checkpoint_id="checkpoint-2",
    )

    # Third attempt: bytes registered, but the success transaction rolls back.
    prepared3 = _prepared(
        store,
        tmp_path,
        script_evidence_id=script_id,
        attempt_id="attempt-3",
        checkpoint_id="checkpoint-3",
    )
    record3 = publish_step_result_envelope_sidecar(prepared3, evidence_store=store)

    def boom() -> None:
        raise RuntimeError("numeric claim registration failed")

    with pytest.raises(RuntimeError):
        _commit_alias(
            store,
            evidence_id=record3.evidence_id,
            alias=prepared3.alias,
            register_numeric_claims=boom,
        )

    # The current alias is unchanged: still the second attempt.
    assert store.aliases()[prepared3.alias] == record2.evidence_id
    loaded = load_current_step_result_envelope_sidecar(
        evidence_store=store,
        query=_query(script_id, attempt_id="attempt-2", checkpoint_id="checkpoint-2"),
    )
    assert isinstance(loaded, LoadedStepResultEnvelopeSidecar)
    assert loaded.evidence_id == record2.evidence_id

    # The rolled-back third attempt is never current.
    stale = load_current_step_result_envelope_sidecar(
        evidence_store=store,
        query=_query(script_id, attempt_id="attempt-3", checkpoint_id="checkpoint-3"),
    )
    assert isinstance(stale, StepResultEnvelopeSidecarUnavailable)
    assert stale.reason == "attempt_mismatch"


def test_query_requires_nonempty_attempt_and_checkpoint() -> None:
    """attempt_id/checkpoint_id are required and non-empty: no silent bypass."""

    # Omitted entirely -> missing-argument construction error.
    with pytest.raises(TypeError):
        StepResultEnvelopeSidecarQuery(  # type: ignore[call-arg]
            step_id=_STEP_ID,
            terminal_status="ok",
            script_evidence_id="code_x",
        )
    # Present but empty/blank -> fail-closed ValueError, never a wildcard query.
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            StepResultEnvelopeSidecarQuery(
                step_id=_STEP_ID,
                terminal_status="ok",
                script_evidence_id="code_x",
                attempt_id=bad,
                checkpoint_id="checkpoint-1",
            )
        with pytest.raises(ValueError):
            StepResultEnvelopeSidecarQuery(
                step_id=_STEP_ID,
                terminal_status="ok",
                script_evidence_id="code_x",
                attempt_id="attempt-1",
                checkpoint_id=bad,
            )


def test_legacy_store_without_sidecar_is_not_auto_promoted(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "run")
    script_id = _script_evidence_id(store)
    # A normal completed step with ordinary evidence but no envelope sidecar.
    store.register_text(
        kind="table",
        description="Legacy table.",
        text="a,b\n1,2\n",
        filename="table.csv",
        produced_by_step=_STEP_ID,
        producer="coder",
        aliases=[f"{_STEP_ID}_table"],
        publish_aliases=True,
    )

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "no_committed_alias"


# ---------------------------------------------------------------------------
# Loader binding + tamper negatives
# ---------------------------------------------------------------------------


def _committed_sidecar(
    tmp_path: Path,
) -> tuple[EvidenceStore, str, PreparedStepResultEnvelopeSidecar, Any]:
    store = EvidenceStore(tmp_path / "run")
    script_id = _script_evidence_id(store)
    prepared = _prepared(store, tmp_path, script_evidence_id=script_id)
    record = publish_step_result_envelope_sidecar(prepared, evidence_store=store)
    _commit_alias(store, evidence_id=record.evidence_id, alias=prepared.alias)
    return store, script_id, prepared, record


def test_loader_rejects_stale_attempt(tmp_path: Path) -> None:
    store, script_id, _prep, _rec = _committed_sidecar(tmp_path)
    result = load_current_step_result_envelope_sidecar(
        evidence_store=store,
        query=_query(script_id, attempt_id="attempt-2"),
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "attempt_mismatch"


def test_loader_rejects_wrong_checkpoint(tmp_path: Path) -> None:
    store, script_id, _prep, _rec = _committed_sidecar(tmp_path)
    result = load_current_step_result_envelope_sidecar(
        evidence_store=store,
        query=_query(script_id, checkpoint_id="checkpoint-2"),
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "checkpoint_mismatch"


def test_loader_rejects_wrong_script_binding(tmp_path: Path) -> None:
    store, _script_id, _prep, _rec = _committed_sidecar(tmp_path)
    result = load_current_step_result_envelope_sidecar(
        evidence_store=store,
        query=_query("code_step_forged"),
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "script_mismatch"


def test_loader_rejects_non_successful_terminal_status(tmp_path: Path) -> None:
    store, script_id, _prep, _rec = _committed_sidecar(tmp_path)
    result = load_current_step_result_envelope_sidecar(
        evidence_store=store,
        query=_query(script_id, terminal_status="contract_failed"),
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "non_successful_terminal_status"


def test_loader_rejects_wrong_step(tmp_path: Path) -> None:
    store, script_id, _prep, _rec = _committed_sidecar(tmp_path)
    result = load_current_step_result_envelope_sidecar(
        evidence_store=store,
        query=_query(script_id, step_id="06_other_step"),
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "no_committed_alias"


def test_loader_rejects_foreign_record_under_sidecar_alias(tmp_path: Path) -> None:
    """A malicious/foreign record claiming the sidecar alias fails closed."""

    store = EvidenceStore(tmp_path / "run")
    script_id = _script_evidence_id(store)
    prepared = _prepared(store, tmp_path, script_evidence_id=script_id)
    foreign = store.register_text(
        kind="table",  # not the sidecar kind
        description="Foreign record impersonating the sidecar alias.",
        text="a,b\n1,2\n",
        filename="foreign.csv",
        produced_by_step=_STEP_ID,
        producer="coder",
        publish_aliases=False,
    )
    _commit_alias(store, evidence_id=foreign.evidence_id, alias=prepared.alias)

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "kind_mismatch"


def test_loader_rejects_wrong_producer(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "run")
    script_id = _script_evidence_id(store)
    prepared = _prepared(store, tmp_path, script_evidence_id=script_id)
    impostor = store.register_text(
        kind="log",
        description="Right kind, wrong producer.",
        text=prepared.payload.decode("utf-8"),
        filename=prepared.filename,
        produced_by_step=_STEP_ID,
        script_evidence_id=script_id,
        producer="analyzer",  # not SIDECAR_PRODUCER
        metadata=dict(prepared.metadata),
        publish_aliases=False,
    )
    _commit_alias(store, evidence_id=impostor.evidence_id, alias=prepared.alias)

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "producer_mismatch"


def test_loader_rejects_wrong_sidecar_schema(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "run")
    script_id = _script_evidence_id(store)
    prepared = _prepared(store, tmp_path, script_evidence_id=script_id)
    bad_md = dict(prepared.metadata)
    bad_md["sidecar_schema_version"] = "easyicu.step_result_envelope_sidecar/999"
    record = store.register_text(
        kind="log",
        description="Wrong sidecar schema.",
        text=prepared.payload.decode("utf-8"),
        filename=prepared.filename,
        produced_by_step=_STEP_ID,
        script_evidence_id=script_id,
        evidence_id=prepared.evidence_id,
        producer=SIDECAR_PRODUCER,
        metadata=bad_md,
        publish_aliases=False,
    )
    _commit_alias(store, evidence_id=record.evidence_id, alias=prepared.alias)

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "sidecar_schema_mismatch"


def test_loader_rejects_paper_authority_metadata(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "run")
    script_id = _script_evidence_id(store)
    prepared = _prepared(store, tmp_path, script_evidence_id=script_id)
    bad_md = dict(prepared.metadata)
    bad_md["paper_authorized"] = True
    record = store.register_text(
        kind="log",
        description="Metadata forges paper authority.",
        text=prepared.payload.decode("utf-8"),
        filename=prepared.filename,
        produced_by_step=_STEP_ID,
        script_evidence_id=script_id,
        evidence_id=prepared.evidence_id,
        producer=SIDECAR_PRODUCER,
        metadata=bad_md,
        publish_aliases=False,
    )
    _commit_alias(store, evidence_id=record.evidence_id, alias=prepared.alias)

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "paper_authority_asserted"


def test_loader_rejects_tampered_bytes(tmp_path: Path) -> None:
    store, script_id, prepared, record = _committed_sidecar(tmp_path)
    target = store.root / record.relative_path
    tampered = json.loads(target.read_bytes())
    tampered["status"] = "hijacked"
    target.write_bytes(json.dumps(tampered).encode("utf-8"))

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    # Tampered bytes fail the descriptor-anchored digest check.
    assert result.reason == "artifact_path_unverified"


def test_loader_rejects_symlinked_sidecar(tmp_path: Path) -> None:
    store, script_id, prepared, record = _committed_sidecar(tmp_path)
    target = store.root / record.relative_path
    payload = target.read_bytes()
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_bytes(payload)
    target.unlink()
    target.symlink_to(elsewhere)

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    # A final symlink -- even to identical bytes -- is refused.
    assert result.reason == "artifact_path_unverified"


def test_loader_rejects_parent_directory_symlink(tmp_path: Path) -> None:
    """A symlinked evidence *parent* directory is refused, not just a final link."""

    store, script_id, prepared, record = _committed_sidecar(tmp_path)
    payload = (store.root / record.relative_path).read_bytes()
    # Relocate the whole evidence directory and symlink it back into place, so
    # the file itself is a regular file but a path *component* is a symlink.
    evidence_dir = store.root / "evidence"
    moved = store.root / "evidence_real"
    evidence_dir.rename(moved)
    evidence_dir.symlink_to(moved, target_is_directory=True)
    assert (store.root / record.relative_path).read_bytes() == payload
    assert (store.root / record.relative_path).is_file()

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "artifact_path_unverified"


def test_loader_rejects_relative_path_traversal(tmp_path: Path) -> None:
    """A record whose ``relative_path`` escapes the evidence dir is refused."""

    store, script_id, prepared, record = _committed_sidecar(tmp_path)
    # Plant identical bytes OUTSIDE the evidence directory, then point the
    # committed record at them via a traversal path.  Metadata is untrusted:
    # the descriptor-anchored guard must refuse the escape even though the
    # bytes match the registered digest.
    outside = store.root / "outside.json"
    outside.write_bytes((store.root / record.relative_path).read_bytes())
    record.relative_path = "evidence/../outside.json"

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "artifact_path_unverified"


def test_loader_rejects_internally_inconsistent_envelope(tmp_path: Path) -> None:
    """A sidecar whose envelope digest does not self-verify is rejected."""

    store = EvidenceStore(tmp_path / "run")
    script_id = _script_evidence_id(store)
    snapshot = _ready_snapshot(tmp_path)
    # Flip status WITHOUT recomputing the digest: an invalid envelope.
    forged_env = snapshot.model_copy(update={"status": "ok"})
    forged_bytes = (
        json.dumps(
            forged_env.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    md = {
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "step_id": _STEP_ID,
        "attempt_id": "attempt-1",
        "checkpoint_id": "checkpoint-1",
        "script_evidence_id": script_id,
        "envelope_schema_version": forged_env.schema_version,
        "content_sha256": forged_env.content_sha256,
        "source_summary_sha256": forged_env.source_summary_sha256,
        "terminal_status": "ok",
        "paper_authorized": False,
    }
    alias = "result_envelope_sidecar__" + _STEP_ID
    record = store.register_text(
        kind="log",
        description="Internally inconsistent envelope.",
        text=forged_bytes,
        filename=f"{alias}.json",
        produced_by_step=_STEP_ID,
        script_evidence_id=script_id,
        producer=SIDECAR_PRODUCER,
        metadata=md,
        publish_aliases=False,
    )
    _commit_alias(store, evidence_id=record.evidence_id, alias=alias)

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=_query(script_id)
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "envelope_digest_invalid"


# ---------------------------------------------------------------------------
# Live wiring: the real _execute_one_step success path publishes a recoverable
# sidecar (M7, commit 3).  Reuses the minimal typed 4-step pipeline harness --
# mock LLM + in-process runner, no Provider, Docker, or network.
# ---------------------------------------------------------------------------


def _load_trajectory_fixture() -> ModuleType:
    path = Path(__file__).with_name("test_trajectory_stability_pipeline_success.py")
    spec = importlib.util.spec_from_file_location(
        "_easyicu_sidecar_trajectory_fixture", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_minimal_pipeline(ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fixture = _load_trajectory_fixture()
    fixture._disable_unrelated_audits(monkeypatch)
    llm = fixture._PlanAndCoderLLM()
    runners_by_timeout: dict[float, object] = {}

    def runner_factory(*, workdir, timeout_seconds, **_kwargs):
        timeout = float(timeout_seconds)
        runner = runners_by_timeout.get(timeout)
        if runner is None:
            runner = fixture._HybridTrajectoryRunner(workdir=Path(workdir))
            runners_by_timeout[timeout] = runner
        return runner

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        timeout_seconds=17.0,
        standard_executor_timeout_seconds=1_234.0,
        runner_factory=runner_factory,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_replanning=False,
        enable_deterministic_code_fallback=True,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=2,
    )
    cohort = pd.DataFrame(
        {
            "stay_id": list(range(1, 25)),
            "marker_h0_6": np.linspace(-1.0, 1.0, 24),
            "marker_h6_12": np.linspace(-0.5, 1.5, 24),
            "death": [0, 1] * 12,
        }
    )
    result = pipeline.run(
        question="Assess fixed-window trajectory phenotypes.",
        cohort=cohort,
        cohort_name="trajectory_stability_success",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="04_characterization",
        stop_after_analysis=True,
    )
    return Path(result.workdir)


def _query_from_committed_record(store: EvidenceStore, *, alias: str, step_id: str):
    evidence_id = store.aliases().get(alias)
    assert evidence_id is not None, f"sidecar alias {alias!r} was not committed"
    record = next(r for r in store.records() if r.evidence_id == evidence_id)
    md = record.metadata or {}
    query = StepResultEnvelopeSidecarQuery(
        step_id=step_id,
        terminal_status="ok",
        script_evidence_id=str(md["script_evidence_id"]),
        attempt_id=str(md["attempt_id"]),
        checkpoint_id=str(md["checkpoint_id"]),
    )
    return record, query


def test_live_success_path_publishes_recoverable_sidecar(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _run_minimal_pipeline(ra, tmp_path, monkeypatch)
    store = EvidenceStore(run_dir)

    # Every ``ok`` step in the real run committed exactly one sidecar alias.
    for step_id in (
        "01_representation",
        "02_candidates",
        "03_stability",
        "04_characterization",
    ):
        alias = "result_envelope_sidecar__" + step_id
        record, query = _query_from_committed_record(
            store, alias=alias, step_id=step_id
        )
        assert record.producer == SIDECAR_PRODUCER
        assert record.kind == "log"
        assert record.metadata["sidecar_schema_version"] == SIDECAR_SCHEMA_VERSION
        assert record.metadata["paper_authorized"] is False
        # Written to the evidence directory, never the raw step output dir.
        assert record.relative_path.startswith("evidence/")
        assert (run_dir / record.relative_path).is_file()
        step_out = run_dir / "steps" / step_id
        if step_out.exists():
            assert not list(step_out.rglob("*result_envelope_sidecar*"))

        loaded = load_current_step_result_envelope_sidecar(
            evidence_store=store, query=query
        )
        assert isinstance(loaded, LoadedStepResultEnvelopeSidecar)
        assert loaded.envelope.status == "ok"
        assert loaded.envelope.step_id == step_id
        assert loaded.envelope.paper_authorized is False


def test_live_sidecar_fails_closed_on_stale_attempt_and_missing_step(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _run_minimal_pipeline(ra, tmp_path, monkeypatch)
    store = EvidenceStore(run_dir)
    _record, query = _query_from_committed_record(
        store, alias="result_envelope_sidecar__03_stability", step_id="03_stability"
    )

    # Stale attempt: the current step is bound to a different attempt id.
    stale = load_current_step_result_envelope_sidecar(
        evidence_store=store,
        query=StepResultEnvelopeSidecarQuery(
            step_id="03_stability",
            terminal_status="ok",
            script_evidence_id=query.script_evidence_id,
            attempt_id=str(query.attempt_id) + "-stale",
            checkpoint_id=query.checkpoint_id,
        ),
    )
    assert isinstance(stale, StepResultEnvelopeSidecarUnavailable)
    assert stale.reason == "attempt_mismatch"

    # No sidecar was ever published for a non-existent step.
    missing = load_current_step_result_envelope_sidecar(
        evidence_store=store,
        query=StepResultEnvelopeSidecarQuery(
            step_id="99_never_ran",
            terminal_status="ok",
            script_evidence_id=query.script_evidence_id,
            attempt_id=query.attempt_id,
            checkpoint_id=query.checkpoint_id,
        ),
    )
    assert isinstance(missing, StepResultEnvelopeSidecarUnavailable)
    assert missing.reason == "no_committed_alias"


def test_live_sidecar_fails_closed_on_tampered_bytes(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _run_minimal_pipeline(ra, tmp_path, monkeypatch)
    store = EvidenceStore(run_dir)
    record, query = _query_from_committed_record(
        store,
        alias="result_envelope_sidecar__04_characterization",
        step_id="04_characterization",
    )
    target = run_dir / record.relative_path
    tampered = json.loads(target.read_bytes())
    tampered["status"] = "hijacked"
    target.write_bytes(json.dumps(tampered).encode("utf-8"))

    result = load_current_step_result_envelope_sidecar(
        evidence_store=store, query=query
    )
    assert isinstance(result, StepResultEnvelopeSidecarUnavailable)
    assert result.reason == "artifact_path_unverified"


@pytest.mark.parametrize(
    "sidecar_behavior, expected_reason_fragment",
    [
        ("return_none", "sidecar_unavailable_for_successful_step"),
        ("raise", "sidecar_registration_failed"),
    ],
)
def test_live_ok_step_fails_closed_when_sidecar_cannot_publish(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_behavior: str,
    expected_reason_fragment: str,
) -> None:
    """status==ok must not commit silently if the sidecar cannot be sealed.

    A missing snapshot / fail-closed prepare (``return_none``) or a registration
    error (``raise``) converts the step to a typed ``contract_failed`` finding;
    no sidecar alias becomes current.
    """

    from easyicu.research_agent.execution import phase as phase_mod

    def _sidecar_none(**_kwargs: Any) -> None:
        return None

    def _sidecar_boom(**_kwargs: Any) -> None:
        raise OSError("simulated sidecar registration failure")

    monkeypatch.setattr(
        phase_mod,
        "publish_terminal_step_result_envelope_sidecar",
        _sidecar_none if sidecar_behavior == "return_none" else _sidecar_boom,
    )

    run_dir = _run_minimal_pipeline(ra, tmp_path, monkeypatch)
    store = EvidenceStore(run_dir)

    # No sidecar alias ever became current authority.
    assert not any(
        alias.startswith("result_envelope_sidecar__") for alias in store.aliases()
    )

    authority = load_run_artifact_authority(run_dir)
    assert authority is not None
    per_step = authority.get("per_step_records") or []
    sidecar_findings = [
        finding
        for record in per_step
        if record.get("status") == "contract_failed"
        for finding in (record.get("contract_findings") or [])
        if finding.get("validator") == "result_envelope_sidecar"
    ]
    assert sidecar_findings, "a status==ok step must fail closed on sidecar publish"
    finding = sidecar_findings[0]
    assert finding["severity"] == "error"
    assert expected_reason_fragment in str(finding["detail"]["reason"])
