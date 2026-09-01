from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.research_agent.authority import step_capsule as capsule_store
from easyicu.research_agent.authority.step_capsule import (
    ConceptAuditSeal,
    ContentRef,
    ExecutionOutput,
    ExecutionSeal,
    StepAuthorityCapsule,
    StepAuthorityCapsuleError,
    StepAuthorityCapsuleRef,
    concept_audit_authority_sha256,
    execution_seal_identity_sha256,
    load_verified_step_authority_capsule,
    put_content_blob,
    read_verified_content,
    seal_step_authority_capsule,
)
from easyicu.research_agent.contracts.execution_result import RunnerFailureCode

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _refs(run_dir: Path) -> dict[str, ContentRef]:
    return {
        "planner_scope": put_content_blob(
            run_dir,
            payload=b'{"step_id":"01_summary"}',
            media_type="application/json",
        ),
        "scoped_coder_context": put_content_blob(
            run_dir,
            payload=b'{"variables":["stay_id"]}',
            media_type="application/json",
        ),
        "resolved_inputs": put_content_blob(
            run_dir,
            payload=b'{"inputs":{}}',
            media_type="application/json",
        ),
        "candidate_code": put_content_blob(
            run_dir,
            payload=b"import json\nprint(json.dumps({'ok': True}))\n",
            media_type="text/x-python",
        ),
    }


def _candidate(run_dir: Path, **updates) -> StepAuthorityCapsule:
    payload = {
        "step_id": "01_summary",
        "stage": "candidate",
        "parent_capsule_sha256": None,
        "run_input_capsule_sha256": SHA_A,
        **_refs(run_dir),
        "typed_bindings_sha256": SHA_B,
        "upstream_authority_sha256": SHA_C,
        "candidate_origin": {
            "kind": "initial_generation",
            "authority_binding_sha256": SHA_B,
            "provider_category": "initial_generation",
            "provider_transport_id": "initial_generation:1",
            "logical_repair_attempt_id": None,
            "repair_ticket_sha256": None,
            "deterministic_reason_sha256": None,
        },
        "deterministic_gate_fingerprint": SHA_D,
        "engine_code_sha256": SHA_E,
        "validator_code_sha256": SHA_F,
        "prompt_pack_version": "2026-07-16",
        "prompt_pack_sha256": SHA_A,
        "concept_audit": None,
        "execution": None,
    }
    payload.update(updates)
    return StepAuthorityCapsule.model_validate(payload)


def _execution(
    run_dir: Path,
    candidate: StepAuthorityCapsule,
    *,
    returncode: int = 0,
    outputs_safe_to_collect: bool = True,
) -> ExecutionSeal:
    payload = {
        "execution_context_sha256": SHA_D,
        "code_sha256": candidate.candidate_code.sha256,
        "resolved_inputs_sha256": candidate.resolved_inputs.sha256,
        "returncode": returncode,
        "duration_seconds": 0.25,
        "timed_out": False,
        "outputs_safe_to_collect": outputs_safe_to_collect,
        "requested_network_policy": "none",
        "effective_isolation": "macos_sandbox_exec",
        "isolation_degraded": False,
        "isolation_degradation_reason": None,
        "runtime_provenance": put_content_blob(
            run_dir,
            payload=b'{"python":"3.13"}',
            media_type="application/json",
        ),
        "stdout": put_content_blob(
            run_dir,
            payload=b"",
            media_type="text/plain",
        ),
        "stderr": put_content_blob(
            run_dir,
            payload=b"",
            media_type="text/plain",
        ),
        "runner_log": None,
        "outputs": (),
    }
    payload["execution_identity_sha256"] = execution_seal_identity_sha256(payload)
    return ExecutionSeal.model_validate(payload)


def test_runner_failure_code_preserves_v1_identity_when_absent() -> None:
    legacy = {"returncode": 1, "timed_out": False}
    explicit_none = {**legacy, "runner_failure_code": None}
    typed = {
        **legacy,
        "runner_failure_code": RunnerFailureCode.ISOLATION_BACKEND_UNAVAILABLE,
    }

    assert execution_seal_identity_sha256(legacy) == execution_seal_identity_sha256(
        explicit_none
    )
    assert execution_seal_identity_sha256(typed) != execution_seal_identity_sha256(
        legacy
    )


def _audit(
    run_dir: Path,
    candidate: StepAuthorityCapsule,
    *,
    findings_payload: bytes = b"[]",
    result: str = "passed",
) -> ConceptAuditSeal:
    findings = put_content_blob(
        run_dir,
        payload=findings_payload,
        media_type="application/json",
    )
    binding = concept_audit_authority_sha256(
        candidate,
        audit_key=SHA_A,
        auditor_identity_sha256=SHA_B,
        environment_sha256=SHA_C,
        validator_implementation_sha256=SHA_D,
    )
    return ConceptAuditSeal(
        audit_key=SHA_A,
        audited_code_sha256=candidate.candidate_code.sha256,
        authority_binding_sha256=binding,
        result=result,
        findings=findings,
        auditor_identity_sha256=SHA_B,
        environment_sha256=SHA_C,
        validator_implementation_sha256=SHA_D,
    )


def test_content_blob_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    payload = b'{"same":true}'

    first = put_content_blob(
        tmp_path,
        payload=payload,
        media_type="application/json",
    )
    second = put_content_blob(
        tmp_path,
        payload=payload,
        media_type="application/json",
    )

    assert first == second
    assert read_verified_content(tmp_path, first) == payload
    stored = list((tmp_path / ".step_authority" / "blobs").rglob(first.sha256))
    assert len(stored) == 1


def test_capsule_roundtrip_is_deterministic_and_has_no_evidence_side_effect(
    tmp_path: Path,
) -> None:
    capsule = _candidate(tmp_path)

    first = seal_step_authority_capsule(tmp_path, capsule)
    second = seal_step_authority_capsule(tmp_path, capsule)
    verified = load_verified_step_authority_capsule(
        tmp_path,
        ref=first,
        expected_step_id="01_summary",
    )

    assert first == second
    assert verified.ref == first
    assert verified.capsule == capsule
    assert verified.candidate_code.startswith("import json")
    assert not (tmp_path / "evidence").exists()


def test_capsule_schema_rejects_unknown_fields_and_invalid_paths(
    tmp_path: Path,
) -> None:
    payload = _candidate(tmp_path).model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        StepAuthorityCapsule.model_validate(payload)

    missing_transport = _candidate(tmp_path).model_dump(mode="json")
    missing_transport["candidate_origin"]["provider_transport_id"] = None
    with pytest.raises(ValidationError):
        StepAuthorityCapsule.model_validate(missing_transport)

    with pytest.raises(ValidationError):
        StepAuthorityCapsuleRef(step_id="../escape", capsule_sha256=SHA_A)

    with pytest.raises(ValidationError):
        ExecutionOutput(
            logical_relative_path="../outside.csv",
            content=_refs(tmp_path)["resolved_inputs"],
        )


def test_stage_contract_requires_audit_and_execution_closure(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate_ref = seal_step_authority_capsule(tmp_path, candidate)
    audit = _audit(tmp_path, candidate)

    with pytest.raises(ValidationError):
        _candidate(
            tmp_path,
            stage="concept_audited",
            parent_capsule_sha256=candidate_ref.capsule_sha256,
        )

    execution = _execution(tmp_path, candidate)
    audited = _candidate(
        tmp_path,
        stage="concept_audited",
        parent_capsule_sha256=candidate_ref.capsule_sha256,
        concept_audit=audit,
    )
    audited_ref = seal_step_authority_capsule(tmp_path, audited)
    executed = _candidate(
        tmp_path,
        stage="executed_concept_audited",
        parent_capsule_sha256=audited_ref.capsule_sha256,
        concept_audit=audit,
        execution=execution,
    )

    ref = seal_step_authority_capsule(tmp_path, executed)
    assert load_verified_step_authority_capsule(tmp_path, ref=ref).capsule == executed

    bad_execution = execution.model_copy(update={"code_sha256": SHA_A})
    with pytest.raises(ValidationError):
        _candidate(
            tmp_path,
            stage="executed_concept_audited",
            parent_capsule_sha256=audited_ref.capsule_sha256,
            concept_audit=audit,
            execution=bad_execution,
        )


def test_failed_execution_can_be_sealed_before_concept_audit(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate_ref = seal_step_authority_capsule(tmp_path, candidate)
    failed = _candidate(
        tmp_path,
        stage="executed",
        parent_capsule_sha256=candidate_ref.capsule_sha256,
        execution=_execution(
            tmp_path,
            candidate,
            returncode=1,
            outputs_safe_to_collect=False,
        ),
    )

    ref = seal_step_authority_capsule(tmp_path, failed)
    recovered = load_verified_step_authority_capsule(tmp_path, ref=ref)

    assert recovered.capsule.execution is not None
    assert recovered.capsule.execution.returncode == 1
    assert recovered.capsule.concept_audit is None


def test_audit_result_is_derived_from_strict_findings(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate_ref = seal_step_authority_capsule(tmp_path, candidate)
    findings = put_content_blob(
        tmp_path,
        payload=json.dumps(
            [
                {
                    "validator": "llm_concept_auditor",
                    "severity": "error",
                    "message": "blocking semantic error",
                    "evidence_ids": [],
                    "detail": {"issue_code": "other"},
                }
            ]
        ).encode("utf-8"),
        media_type="application/json",
    )
    false_pass = _audit(
        tmp_path,
        candidate,
        findings_payload=read_verified_content(tmp_path, findings),
        result="passed",
    )
    audited = _candidate(
        tmp_path,
        stage="concept_audited",
        parent_capsule_sha256=candidate_ref.capsule_sha256,
        concept_audit=false_pass,
    )

    with pytest.raises(StepAuthorityCapsuleError, match="finding severities"):
        seal_step_authority_capsule(tmp_path, audited)

    provider_failure = put_content_blob(
        tmp_path,
        payload=json.dumps(
            [
                {
                    "validator": "llm_concept_auditor",
                    "severity": "error",
                    "message": "provider unavailable",
                    "evidence_ids": [],
                    "detail": {"issue_code": "llm_concept_audit_provider_failure"},
                }
            ]
        ).encode("utf-8"),
        media_type="application/json",
    )
    unavailable_audit = false_pass.model_copy(
        update={"result": "blocked", "findings": provider_failure}
    )
    unavailable = audited.model_copy(update={"concept_audit": unavailable_audit})
    with pytest.raises(StepAuthorityCapsuleError, match="cannot become"):
        seal_step_authority_capsule(tmp_path, unavailable)


def test_concept_audit_cannot_be_reused_for_different_code(tmp_path: Path) -> None:
    candidate_a = _candidate(tmp_path)
    audit_a = _audit(tmp_path, candidate_a)
    code_b = put_content_blob(
        tmp_path,
        payload=b"import json\nprint(json.dumps({'different': True}))\n",
        media_type="text/x-python",
    )
    origin_b = candidate_a.candidate_origin.model_copy(
        update={
            "authority_binding_sha256": SHA_C,
            "provider_transport_id": "initial_generation:2",
        }
    )
    candidate_b = _candidate(
        tmp_path,
        candidate_code=code_b,
        candidate_origin=origin_b,
    )
    candidate_b_ref = seal_step_authority_capsule(tmp_path, candidate_b)
    falsely_audited_b = _candidate(
        tmp_path,
        stage="concept_audited",
        parent_capsule_sha256=candidate_b_ref.capsule_sha256,
        candidate_code=code_b,
        candidate_origin=origin_b,
        concept_audit=audit_a,
    )

    with pytest.raises(StepAuthorityCapsuleError, match="different candidate code"):
        seal_step_authority_capsule(tmp_path, falsely_audited_b)


def test_stage_transition_cannot_erase_candidate_origin(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate_ref = seal_step_authority_capsule(tmp_path, candidate)
    audit = _audit(tmp_path, candidate)
    relabeled = _candidate(
        tmp_path,
        stage="concept_audited",
        parent_capsule_sha256=candidate_ref.capsule_sha256,
        candidate_origin={
            "kind": "legacy_adoption",
            "authority_binding_sha256": SHA_B,
            "provider_category": None,
            "provider_transport_id": None,
            "logical_repair_attempt_id": None,
            "repair_ticket_sha256": None,
            "deterministic_reason_sha256": None,
        },
        concept_audit=audit,
    )

    with pytest.raises(StepAuthorityCapsuleError, match="not bound"):
        seal_step_authority_capsule(tmp_path, relabeled)


def test_missing_grandparent_invalidates_descendant(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate_ref = seal_step_authority_capsule(tmp_path, candidate)
    audit = _audit(tmp_path, candidate)
    audited = _candidate(
        tmp_path,
        stage="concept_audited",
        parent_capsule_sha256=candidate_ref.capsule_sha256,
        concept_audit=audit,
    )
    audited_ref = seal_step_authority_capsule(tmp_path, audited)
    combined = _candidate(
        tmp_path,
        stage="executed_concept_audited",
        parent_capsule_sha256=audited_ref.capsule_sha256,
        concept_audit=audit,
        execution=_execution(tmp_path, candidate),
    )
    combined_ref = seal_step_authority_capsule(tmp_path, combined)
    candidate_path = next(
        (tmp_path / ".step_authority" / "capsules").rglob(
            f"{candidate_ref.capsule_sha256}.json"
        )
    )
    candidate_path.unlink()

    with pytest.raises(StepAuthorityCapsuleError, match="missing"):
        load_verified_step_authority_capsule(tmp_path, ref=combined_ref)


def test_tampered_blob_and_capsule_fail_closed(tmp_path: Path) -> None:
    capsule = _candidate(tmp_path)
    ref = seal_step_authority_capsule(tmp_path, capsule)
    blob_path = next(
        (tmp_path / ".step_authority" / "blobs").rglob(capsule.candidate_code.sha256)
    )
    blob_path.write_text("import os\nprint('tampered')\n", encoding="utf-8")

    with pytest.raises(StepAuthorityCapsuleError, match="digest"):
        load_verified_step_authority_capsule(tmp_path, ref=ref)

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    clean_ref = seal_step_authority_capsule(clean_dir, _candidate(clean_dir))
    capsule_path = next(
        (clean_dir / ".step_authority" / "capsules").rglob(
            f"{clean_ref.capsule_sha256}.json"
        )
    )
    capsule_path.write_text("{}", encoding="utf-8")
    with pytest.raises(StepAuthorityCapsuleError, match="digest"):
        load_verified_step_authority_capsule(clean_dir, ref=clean_ref)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_store_rejects_symlinked_authority_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".step_authority").symlink_to(outside, target_is_directory=True)

    with pytest.raises(StepAuthorityCapsuleError, match="symbolic link"):
        put_content_blob(
            tmp_path,
            payload=b"{}",
            media_type="application/json",
        )
    assert not list(outside.iterdir())


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_store_rejects_nested_symlink_and_never_writes_outside(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    store = tmp_path / ".step_authority"
    store.mkdir()
    (store / "blobs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(StepAuthorityCapsuleError, match="symbolic link"):
        put_content_blob(
            tmp_path,
            payload=b"{}",
            media_type="application/json",
        )
    assert not list(outside.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="descriptor traversal is POSIX-only")
def test_intermediate_directory_swap_cannot_redirect_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    put_content_blob(
        tmp_path,
        payload=b"seed",
        media_type="application/octet-stream",
    )
    store = tmp_path / ".step_authority"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "sha256" and dir_fd is not None and not swapped:
            swapped = True
            (store / "blobs").rename(store / "blobs_anchored")
            (store / "blobs").symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(capsule_store.os, "open", swapping_open)

    put_content_blob(
        tmp_path,
        payload=b"must stay anchored",
        media_type="application/octet-stream",
    )

    assert swapped is True
    assert not list(outside.iterdir())
    assert list((store / "blobs_anchored").rglob("*"))


def test_preexisting_digest_path_with_other_bytes_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capsule_store, "_sha256_bytes", lambda _payload: SHA_A)
    first = put_content_blob(
        tmp_path,
        payload=b"first",
        media_type="application/octet-stream",
    )

    with pytest.raises(StepAuthorityCapsuleError, match="conflicts"):
        put_content_blob(
            tmp_path,
            payload=b"second",
            media_type="application/octet-stream",
        )

    stored = next((tmp_path / ".step_authority" / "blobs").rglob(first.sha256))
    assert stored.read_bytes() == b"first"


def test_concurrent_identical_writers_publish_one_complete_object(
    tmp_path: Path,
) -> None:
    payload = json.dumps({"rows": list(range(1000))}).encode("utf-8")

    def write_once(_: int) -> ContentRef:
        return put_content_blob(
            tmp_path,
            payload=payload,
            media_type="application/json",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(write_once, range(32)))

    assert len(set(refs)) == 1
    assert read_verified_content(tmp_path, refs[0]) == payload


def test_concurrent_capsule_sealing_is_idempotent(tmp_path: Path) -> None:
    capsule = _candidate(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(
            pool.map(
                lambda _index: seal_step_authority_capsule(tmp_path, capsule),
                range(24),
            )
        )

    assert len(set(refs)) == 1
    assert (
        load_verified_step_authority_capsule(
            tmp_path,
            ref=refs[0],
        ).capsule
        == capsule
    )


def test_shared_capsule_ancestry_is_verified_once_per_root_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _candidate(tmp_path)
    base_ref = seal_step_authority_capsule(tmp_path, base)
    adopted_origin = base.candidate_origin.model_copy(
        update={
            "kind": "legacy_adoption",
            "provider_category": None,
            "provider_transport_id": None,
            "adopted_from_capsule_sha256": base_ref.capsule_sha256,
        }
    )
    adopted = _candidate(tmp_path, candidate_origin=adopted_origin)
    adopted_ref = seal_step_authority_capsule(tmp_path, adopted)
    audited = _candidate(
        tmp_path,
        stage="concept_audited",
        parent_capsule_sha256=adopted_ref.capsule_sha256,
        candidate_origin=adopted_origin,
        concept_audit=_audit(tmp_path, adopted),
    )

    real_candidate_check = capsule_store._is_candidate_python
    candidate_checks = 0

    def counted_candidate_check(code: str) -> bool:
        nonlocal candidate_checks
        candidate_checks += 1
        return real_candidate_check(code)

    monkeypatch.setattr(capsule_store, "_is_candidate_python", counted_candidate_check)

    seal_step_authority_capsule(tmp_path, audited)

    # The root, adopted parent, and shared base are each content-verified once.
    # The base capsule remains addressable from both ancestry edges without
    # re-reading and re-validating every referenced blob.
    assert candidate_checks == 3


def test_code_blob_must_be_utf8_executable_python(tmp_path: Path) -> None:
    invalid_code = put_content_blob(
        tmp_path,
        payload=b'{"format":"easyicu.code_patch/1","edits":[]}',
        media_type="text/x-python",
    )
    with pytest.raises(StepAuthorityCapsuleError, match="executable Python"):
        seal_step_authority_capsule(
            tmp_path,
            _candidate(tmp_path, candidate_code=invalid_code),
        )

    assignment_code = put_content_blob(
        tmp_path,
        payload=b"value = 1\n",
        media_type="text/x-python",
    )
    assert seal_step_authority_capsule(
        tmp_path,
        _candidate(tmp_path, candidate_code=assignment_code),
    ).capsule_sha256
