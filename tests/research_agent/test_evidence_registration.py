"""Focused contracts for the extracted success-only evidence registrar."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.registration import (
    EvidenceRegistrar,
    step_owned_artifact_evidence_id,
)


def _register(
    store: EvidenceStore,
    *,
    step_id: str,
    evidence_id: str,
    filename: str,
) -> object:
    return store.register_text(
        kind="statistic",
        description=f"Historical candidate {evidence_id}.",
        text='{"estimate": 1}',
        filename=filename,
        produced_by_step=step_id,
        evidence_id=evidence_id,
        publish_aliases=False,
    )


def test_registrar_rejects_evidence_outside_exact_attempt_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvidenceStore(tmp_path)
    allowed = _register(
        store,
        step_id="01_model",
        evidence_id="allowed_result",
        filename="allowed.json",
    )
    foreign_attempt = _register(
        store,
        step_id="01_model",
        evidence_id="foreign_attempt_result",
        filename="foreign.json",
    )
    publisher_calls = 0
    original_publish = store.publish_step_success_aliases

    def counted_publish(*args, **kwargs):
        nonlocal publisher_calls
        publisher_calls += 1
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(store, "publish_step_success_aliases", counted_publish)

    with pytest.raises(ValueError, match="outside the current attempt"):
        EvidenceRegistrar(store).promote_validated_step(
            step_id="01_model",
            pending_aliases={
                foreign_attempt.evidence_id: ["primary_association"],
            },
            allowed_evidence_ids=[allowed.evidence_id],
        )

    assert publisher_calls == 0
    assert store.aliases() == {}


def test_registrar_preserves_parent_authority_and_publishes_child_role(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    parent = _register(
        store,
        step_id="01_model",
        evidence_id="parent_result",
        filename="parent.json",
    )
    store.publish_step_success_aliases(
        {parent.evidence_id: ["primary_association"]},
        step_id="01_model",
    )
    child = _register(
        store,
        step_id="02_figure",
        evidence_id="child_figure",
        filename="child.svg",
    )

    result = EvidenceRegistrar(store).promote_validated_step(
        step_id="02_figure",
        pending_aliases={
            child.evidence_id: ["primary_association", "association_figure"],
        },
        allowed_evidence_ids=[child.evidence_id],
    )

    assert result.retained_cross_step_aliases == {
        "primary_association": parent.evidence_id,
    }
    assert store.get("primary_association").evidence_id == parent.evidence_id
    assert store.get("association_figure").evidence_id == child.evidence_id


def test_registrar_allows_same_step_retry_to_replace_its_alias(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    first = _register(
        store,
        step_id="01_model",
        evidence_id="result_attempt_1",
        filename="first.json",
    )
    store.publish_step_success_aliases(
        {first.evidence_id: ["primary_association"]},
        step_id="01_model",
    )
    second = _register(
        store,
        step_id="01_model",
        evidence_id="result_attempt_2",
        filename="second.json",
    )

    EvidenceRegistrar(store).promote_validated_step(
        step_id="01_model",
        pending_aliases={second.evidence_id: ["primary_association"]},
        allowed_evidence_ids=[second.evidence_id],
    )

    assert store.get("primary_association").evidence_id == second.evidence_id


def test_identical_artifact_bytes_keep_distinct_step_authority(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    payload = '{"runtime":"docker","image_id":"sha256:fixed"}'
    first_path = tmp_path / "first" / "runner_provenance.json"
    second_path = tmp_path / "second" / "runner_provenance.json"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_text(payload, encoding="utf-8")
    second_path.write_text(payload, encoding="utf-8")
    artifact_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    records = []
    for step_id, script_id, source_path in (
        ("01_cohort", "code_step_01", first_path),
        ("02_table", "code_step_02", second_path),
    ):
        evidence_id = step_owned_artifact_evidence_id(
            kind="log",
            step_id=step_id,
            source_name=source_path.name,
            artifact_sha256=artifact_sha256,
            script_evidence_id=script_id,
        )
        record = store.register_file(
            kind="log",
            description="Immutable runner provenance.",
            source_path=source_path,
            produced_by_step=step_id,
            script_evidence_id=script_id,
            evidence_id=evidence_id,
            publish_aliases=False,
        )
        EvidenceRegistrar(store).promote_validated_step(
            step_id=step_id,
            pending_aliases={record.evidence_id: []},
            allowed_evidence_ids=[record.evidence_id],
        )
        records.append(record)

    assert records[0].sha256 == records[1].sha256
    assert records[0].evidence_id != records[1].evidence_id
    assert records[0].produced_by_step == "01_cohort"
    assert records[1].produced_by_step == "02_table"


def test_registrar_propagates_store_failure_without_local_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvidenceStore(tmp_path)
    record = _register(
        store,
        step_id="01_model",
        evidence_id="candidate",
        filename="candidate.json",
    )

    def fail_publish(*args, **kwargs):
        raise OSError("simulated persistence failure")

    monkeypatch.setattr(store, "publish_step_success_aliases", fail_publish)

    with pytest.raises(OSError, match="simulated persistence failure"):
        EvidenceRegistrar(store).promote_validated_step(
            step_id="01_model",
            pending_aliases={record.evidence_id: ["primary_association"]},
            allowed_evidence_ids=[record.evidence_id],
        )

    assert store.aliases() == {}
    assert not hasattr(EvidenceRegistrar(store), "current")


def test_execute_phase_uses_extracted_registrar_after_success_gate() -> None:
    # Batch 1c: the numeric-claim registration + alias promotion are no longer
    # inlined in run_execute_phase — they are delegated to
    # StepEvidenceCommit.commit_validated_step, which opens the store's
    # success_publication_transaction. The "numeric + alias commit as one
    # generation" guarantee is now locked by the AST structural contract in
    # test_step_evidence_commit.py; here we only lock the CALLER wiring: after the
    # status gate, the ok path routes through the commit boundary, and the failure
    # path stays fail-closed.
    from easyicu.research_agent.execution import phase as pipeline_execute

    module_source = inspect.getsource(pipeline_execute)
    execute_source = inspect.getsource(pipeline_execute.run_execute_phase)

    assert "def _filter_success_alias_bindings(" not in module_source
    assert "evidence.publish_step_success_aliases(" not in execute_source
    # The transaction is delegated, not inlined in the orchestrator anymore.
    assert "evidence.success_publication_transaction()" not in execute_source
    assert "evidence_registrar.promote_validated_step(" not in execute_source

    commit = execute_source.rindex("step_evidence_commit.commit_validated_step(")
    status_resolution = execute_source.rindex(
        'step_record["status"] = _step_status_from_contract_findings(',
        0,
        commit,
    )
    success_guard = execute_source.rindex(
        'if step_record["status"] == "ok":',
        status_resolution,
        commit,
    )
    # The numeric registration is handed to the commit boundary as a thunk (a bare
    # reference, invoked inside the transaction — see the component contract test).
    assert "register_numeric_claims=_register_current_step_numeric_claims" in (
        execute_source[success_guard:commit] + execute_source[commit : commit + 400]
    )
    terminal_checkpoint = execute_source.rindex(
        "_append_terminal_step_record(per_step_records, step_record)"
    )
    publication_failure = execute_source.index(
        'validator="result_evidence_authority"', commit
    )

    assert status_resolution < success_guard < commit < terminal_checkpoint
    assert (
        "EvidenceAuthorityIntegrityError" in execute_source[commit:publication_failure]
    )
    assert (
        "if not store_unavailable:"
        in execute_source[publication_failure:terminal_checkpoint]
    )
