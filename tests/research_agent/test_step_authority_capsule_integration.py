from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from easyicu.research_agent.authority.coder_authority import HostCoderAuthority
from easyicu.research_agent.authority.provider_budget import (
    StepProviderCallBudget,
    load_provider_call_budget_state,
    provider_call_budget_receipt_path,
)
from easyicu.research_agent.pipeline import _load_resume_state
from easyicu.research_agent.repairs.coordination import RepairAuthorityBinding
from easyicu.research_agent.runner import RunResult
from easyicu.research_agent.runtime_artifacts import write_run_checkpoint
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ResearchContext,
    ValidationFinding,
)
from easyicu.research_agent.research_context.typed import ResearchContextV2
from easyicu.research_agent.authority.step_capsule import (
    StepAuthorityCapsuleError,
    load_verified_step_authority_capsule,
)
from easyicu.research_agent.authority.step_runtime import (
    StepAuthorityRuntimeError,
    adopt_frozen_scoped_coder_context,
    execution_context_sha256,
    load_checkpoint_selected_step_capsule,
    materialize_sealed_run_result,
    persist_candidate_code,
    prepare_step_authority_coordinates,
    repair_code_ref,
    seal_concept_audit_capsule,
    seal_execution_capsule,
    seal_initial_generation_candidate,
    seal_legacy_candidate,
    seal_repair_candidate_from_receipt,
)
from tests.research_agent.test_research_context_v2_authority_join import (
    _prepare_typed_run,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _coordinates(tmp_path: Path, *, scoped_coder_context=None):
    resolved = tmp_path / "resolved_inputs.json"
    resolved.write_text('{"inputs":{}}\n', encoding="utf-8")
    return prepare_step_authority_coordinates(
        run_dir=tmp_path,
        step_id="01_summary",
        run_input_capsule_sha256=SHA_A,
        planner_scope={"step_id": "01_summary", "method": "descriptive_summary"},
        scoped_coder_context=(
            scoped_coder_context
            if scoped_coder_context is not None
            else {"variables": ["stay_id"]}
        ),
        resolved_inputs_path=resolved,
        typed_bindings={},
        upstream_authority={"cohort_sha256": SHA_B},
        deterministic_gate_fingerprint=SHA_C,
        engine_code_sha256=SHA_D,
        validator_code_sha256=SHA_E,
        prompt_pack_version="2026-07-16",
        prompt_pack={"coder.txt": SHA_A},
    )


def _initial_candidate(tmp_path: Path, *, scoped_coder_context=None):
    coordinates = _coordinates(
        tmp_path,
        scoped_coder_context=scoped_coder_context,
    )
    code_ref = persist_candidate_code(
        coordinates,
        "import json\nprint(json.dumps({'ok': True}))\n",
    )
    receipt_path = provider_call_budget_receipt_path(
        tmp_path, step_id=coordinates.step_id
    )
    budget = StepProviderCallBudget(
        4,
        step_id=coordinates.step_id,
        receipt_path=receipt_path,
    )
    transport_id = budget.reserve_initial_generation(
        coordinates.initial_generation_binding()
    )
    assert budget.initial_generation_resume_status() == "unpaid_pending"
    budget.consume("initial_generation")
    budget.complete_initial_generation_transport(
        provider_transport_id=transport_id,
        after_code_sha256=code_ref.sha256,
        after_code_size_bytes=code_ref.size_bytes,
    )
    state = load_provider_call_budget_state(
        receipt_path,
        step_id=coordinates.step_id,
    )
    candidate_ref = seal_initial_generation_candidate(
        coordinates,
        code_ref=code_ref,
        receipt_state=state,
    )
    return coordinates, code_ref, candidate_ref, budget, receipt_path


def _repair_binding(coordinates, *, attempt_id: int, before_code_sha256: str):
    return RepairAuthorityBinding(
        step_id=coordinates.step_id,
        attempt_id=attempt_id,
        repair_class="concept",
        provider_category="concept_repair",
        before_code_sha256=before_code_sha256,
        step_spec_sha256=coordinates.planner_scope.sha256,
        resolved_inputs_sha256=coordinates.resolved_inputs.sha256,
        coder_context_sha256=coordinates.scoped_coder_context.sha256,
        repair_ticket_sha256=SHA_D,
        engine_validator_sha256=coordinates.deterministic_gate_fingerprint,
        prompt_pack_version=coordinates.prompt_pack_version,
        run_input_capsule_sha256=coordinates.run_input_capsule_sha256,
    )


def test_initial_generation_transport_persists_before_candidate_seal(
    tmp_path: Path,
) -> None:
    coordinates, code_ref, candidate_ref, budget, receipt_path = _initial_candidate(
        tmp_path
    )

    state = load_provider_call_budget_state(receipt_path, step_id="01_summary")
    assert state.schema_version == 6
    assert state.initial_generation is not None
    assert state.initial_generation["transport"]["state"] == "completed"
    assert budget.initial_generation_resume_status() == "completed"
    verified = load_verified_step_authority_capsule(tmp_path, ref=candidate_ref)
    assert verified.candidate_code.encode("utf-8")
    assert verified.capsule.candidate_code == code_ref
    assert (
        verified.capsule.candidate_origin.authority_binding_sha256
        == coordinates.authority_binding_sha256
    )


def test_paid_pending_initial_generation_fails_closed(tmp_path: Path) -> None:
    coordinates = _coordinates(tmp_path)
    receipt_path = provider_call_budget_receipt_path(tmp_path, step_id="01_summary")
    budget = StepProviderCallBudget(2, step_id="01_summary", receipt_path=receipt_path)
    budget.reserve_initial_generation(coordinates.initial_generation_binding())
    budget.consume("initial_generation")

    assert budget.initial_generation_resume_status() == "paid_pending"
    with pytest.raises(Exception, match="paid provider calls"):
        budget.reserve_initial_generation(coordinates.initial_generation_binding())


def test_repair_candidate_joins_receipt_before_and_after_code(tmp_path: Path) -> None:
    coordinates, _parent_code, parent_ref, budget, receipt_path = _initial_candidate(
        tmp_path
    )
    parent = load_verified_step_authority_capsule(tmp_path, ref=parent_ref)
    child_ref = persist_candidate_code(
        coordinates,
        "import json\nprint(json.dumps({'ok': 'repaired'}))\n",
    )
    binding = _repair_binding(
        coordinates,
        attempt_id=1,
        before_code_sha256=parent.capsule.candidate_code.sha256,
    )
    attempt_id = budget.reserve_logical_repair(
        "concept",
        max_repairs=2,
        binding=binding.payload(),
        binding_sha256=binding.sha256,
    )
    assert attempt_id == 1
    budget.consume("concept_repair_patch")
    budget.complete_logical_repair_transport(
        attempt_id=attempt_id,
        mode="minimal_patch",
        after_code_sha256=child_ref.sha256,
        after_code_size_bytes=child_ref.size_bytes,
    )
    receipt = load_provider_call_budget_state(receipt_path, step_id="01_summary")
    assert repair_code_ref(receipt, attempt_id=attempt_id) == child_ref

    sealed = seal_repair_candidate_from_receipt(
        coordinates,
        parent_ref=parent_ref,
        checkpoint_parent_ref=parent_ref,
        code_ref=child_ref,
        receipt_state=receipt,
        attempt_id=attempt_id,
        failure_status="concept_failed",
    )
    verified = load_verified_step_authority_capsule(tmp_path, ref=sealed)
    assert verified.capsule.candidate_origin.kind == "repair_patch"
    assert verified.capsule.candidate_origin.logical_repair_attempt_id == 1
    assert verified.capsule.candidate_origin.repair_ticket_sha256 == SHA_D


def test_repair_candidate_rejects_decoy_before_code_and_noncurrent_parent(
    tmp_path: Path,
) -> None:
    coordinates, _parent_code, parent_ref, budget, receipt_path = _initial_candidate(
        tmp_path
    )
    child_ref = persist_candidate_code(
        coordinates,
        "import json\nprint(json.dumps({'ok': 'repaired'}))\n",
    )
    binding = _repair_binding(
        coordinates,
        attempt_id=1,
        before_code_sha256=SHA_E,
    )
    attempt_id = budget.reserve_logical_repair(
        "concept", max_repairs=2, binding=binding.payload()
    )
    assert attempt_id == 1
    budget.consume("concept_repair_patch")
    budget.complete_logical_repair_transport(
        attempt_id=attempt_id,
        mode="patch",
        after_code_sha256=child_ref.sha256,
    )
    receipt = load_provider_call_budget_state(receipt_path, step_id="01_summary")
    with pytest.raises(StepAuthorityRuntimeError, match="before-code"):
        seal_repair_candidate_from_receipt(
            coordinates,
            parent_ref=parent_ref,
            checkpoint_parent_ref=parent_ref,
            code_ref=child_ref,
            receipt_state=receipt,
            attempt_id=attempt_id,
            failure_status="concept_failed",
        )

    other_ref = seal_legacy_candidate(
        coordinates,
        code_ref=persist_candidate_code(
            coordinates, "import json\nprint(json.dumps({'other': True}))\n"
        ),
    )
    with pytest.raises(StepAuthorityRuntimeError, match="newest checkpoint"):
        seal_repair_candidate_from_receipt(
            coordinates,
            parent_ref=parent_ref,
            checkpoint_parent_ref=other_ref,
            code_ref=child_ref,
            receipt_state=receipt,
            attempt_id=attempt_id,
            failure_status="concept_failed",
        )


def test_checkpoint_selector_never_discovers_or_falls_back_to_orphan(
    tmp_path: Path,
) -> None:
    coordinates, _code, selected_ref, _budget, _receipt = _initial_candidate(tmp_path)
    orphan_ref = seal_legacy_candidate(
        coordinates,
        code_ref=persist_candidate_code(
            coordinates, "import json\nprint(json.dumps({'orphan': True}))\n"
        ),
    )
    write_run_checkpoint(
        tmp_path / "manifest_partial.json",
        {
            "per_step_records": [
                {
                    "step_id": "01_summary",
                    "status": "candidate_checkpointed",
                    "step_authority_capsule_ref": selected_ref.model_dump(mode="json"),
                }
            ]
        },
    )

    selected = load_checkpoint_selected_step_capsule(tmp_path, step_id="01_summary")
    assert selected is not None and selected.ref == selected_ref
    assert selected.ref != orphan_ref

    selected_path = next(
        (tmp_path / ".step_authority" / "capsules").rglob(
            f"{selected_ref.capsule_sha256}.json"
        )
    )
    selected_path.unlink()
    with pytest.raises(StepAuthorityRuntimeError, match="checkpoint-selected"):
        load_checkpoint_selected_step_capsule(tmp_path, step_id="01_summary")


def test_resume_state_uses_monotonic_sequence_not_partial_file_preference(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "manifest_partial.json"
    final = tmp_path / "manifest.json"
    write_run_checkpoint(
        partial,
        {"per_step_records": [], "marker": "older_partial"},
    )
    write_run_checkpoint(
        final,
        {"per_step_records": [], "marker": "newer_final"},
    )
    final_mtime = final.stat().st_mtime_ns
    os.utime(partial, ns=(final_mtime + 1_000_000, final_mtime + 1_000_000))

    loaded = _load_resume_state(tmp_path)
    assert loaded is not None
    assert loaded["marker"] == "newer_final"
    assert loaded["checkpoint_sequence"] == 2


def test_execution_capsule_materializes_exact_synthetic_run_result(
    tmp_path: Path,
) -> None:
    coordinates, _code, candidate_ref, _budget, _receipt = _initial_candidate(tmp_path)
    step_dir = tmp_path / "original" / "01_summary"
    out_dir = step_dir / "outputs"
    out_dir.mkdir(parents=True)
    table = out_dir / "result.csv"
    table.write_text("n\n3\n", encoding="utf-8")
    log = step_dir / "run.log"
    log.write_text("runner log\n", encoding="utf-8")
    script = step_dir / "analysis.py"
    script.write_text(
        load_verified_step_authority_capsule(
            tmp_path, ref=candidate_ref
        ).candidate_code,
        encoding="utf-8",
    )
    context_digest = execution_context_sha256(
        code_sha256=load_verified_step_authority_capsule(
            tmp_path, ref=candidate_ref
        ).capsule.candidate_code.sha256,
        resolved_inputs_sha256=coordinates.resolved_inputs.sha256,
        cohort_sha256=SHA_A,
        universe_sha256=SHA_B,
        runner_identity="tests.FakeRunner",
        timeout_seconds=10.0,
        requested_network_policy="none",
    )
    result = RunResult(
        step_id="01_summary",
        script_path=script,
        cwd=step_dir,
        out_dir=out_dir,
        stdout="stdout\n",
        stderr="stderr\n",
        returncode=1,
        duration_seconds=0.5,
        artefacts=[table],
        timed_out=False,
        requested_network_policy="none",
        effective_isolation="test_isolation",
        isolation_degraded=False,
        isolation_degradation_reason=None,
        runtime_provenance={"python": "3.13"},
        outputs_safe_to_collect=True,
        runner_log_path=log,
    )
    executed_ref = seal_execution_capsule(
        coordinates,
        parent_ref=candidate_ref,
        run_result=result,
        execution_context_digest=context_digest,
    )
    verified = load_verified_step_authority_capsule(tmp_path, ref=executed_ref)
    destination = tmp_path / "steps" / "01_summary" / "outputs"
    destination.mkdir(parents=True)
    (destination / "result.csv").write_text("garbage", encoding="utf-8")

    replayed = materialize_sealed_run_result(
        tmp_path,
        verified,
        expected_execution_context_sha256=context_digest,
    )
    assert replayed.returncode == 1
    assert replayed.stdout == "stdout\n"
    assert replayed.stderr == "stderr\n"
    assert replayed.runtime_provenance == {"python": "3.13"}
    assert replayed.requested_network_policy == "none"
    assert replayed.effective_isolation == "test_isolation"
    assert replayed.runner_log_path is not None
    assert replayed.runner_log_path.read_text(encoding="utf-8") == "runner log\n"
    assert (replayed.out_dir / "result.csv").read_text(encoding="utf-8") == "n\n3\n"

    with pytest.raises(StepAuthorityRuntimeError, match="context is stale"):
        materialize_sealed_run_result(
            tmp_path,
            verified,
            expected_execution_context_sha256=SHA_E,
        )


def _sealed_execution_for_replay(tmp_path: Path):
    coordinates, _code, candidate_ref, budget, receipt = _initial_candidate(tmp_path)
    original = tmp_path / "original" / "01_summary"
    original_out = original / "outputs"
    original_out.mkdir(parents=True)
    table = original_out / "result.csv"
    table.write_text("n\n3\n", encoding="utf-8")
    script = original / "analysis.py"
    script.write_text(
        load_verified_step_authority_capsule(
            tmp_path, ref=candidate_ref
        ).candidate_code,
        encoding="utf-8",
    )
    context_digest = execution_context_sha256(
        code_sha256=load_verified_step_authority_capsule(
            tmp_path, ref=candidate_ref
        ).capsule.candidate_code.sha256,
        resolved_inputs_sha256=coordinates.resolved_inputs.sha256,
        cohort_sha256=SHA_A,
        universe_sha256=SHA_B,
        runner_identity="tests.FakeRunner",
        timeout_seconds=10.0,
        requested_network_policy="none",
    )
    result = RunResult(
        step_id="01_summary",
        script_path=script,
        cwd=original,
        out_dir=original_out,
        stdout="",
        stderr="",
        returncode=0,
        duration_seconds=0.1,
        artefacts=[table],
        requested_network_policy="none",
        effective_isolation="test",
        runtime_provenance={},
    )
    executed_ref = seal_execution_capsule(
        coordinates,
        parent_ref=candidate_ref,
        run_result=result,
        execution_context_digest=context_digest,
    )
    return (
        coordinates,
        load_verified_step_authority_capsule(tmp_path, ref=executed_ref),
        context_digest,
        budget,
        receipt,
    )


@pytest.mark.parametrize("state", ["backup_only", "out_and_backup"])
def test_execution_replay_recovers_interrupted_output_swap(
    tmp_path: Path,
    state: str,
) -> None:
    (
        _coordinates_value,
        verified,
        context_digest,
        _budget,
        _receipt,
    ) = _sealed_execution_for_replay(tmp_path)
    step_dir = tmp_path / "steps" / "01_summary"
    step_dir.mkdir(parents=True, exist_ok=True)
    out_dir = step_dir / "outputs"
    backup = step_dir / ".capsule-outputs-backup"
    backup.mkdir()
    (backup / "old.txt").write_text("old", encoding="utf-8")
    if state == "out_and_backup":
        out_dir.mkdir()
        (out_dir / "partial.txt").write_text("partial", encoding="utf-8")

    replayed = materialize_sealed_run_result(
        tmp_path,
        verified,
        expected_execution_context_sha256=context_digest,
    )

    assert (replayed.out_dir / "result.csv").read_text(encoding="utf-8") == "n\n3\n"
    assert not backup.exists()
    assert not (replayed.out_dir / "old.txt").exists()
    assert not (replayed.out_dir / "partial.txt").exists()


@pytest.mark.parametrize("unsafe_name", ["outputs", ".capsule-outputs-backup"])
def test_execution_replay_rejects_symlinked_swap_state(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    (
        _coordinates_value,
        verified,
        context_digest,
        _budget,
        _receipt,
    ) = _sealed_execution_for_replay(tmp_path)
    step_dir = tmp_path / "steps" / "01_summary"
    step_dir.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    (step_dir / unsafe_name).symlink_to(external, target_is_directory=True)

    with pytest.raises(StepAuthorityRuntimeError, match="symbolic link"):
        materialize_sealed_run_result(
            tmp_path,
            verified,
            expected_execution_context_sha256=context_digest,
        )
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_execution_context_binds_runtime_and_runner_configuration() -> None:
    common = {
        "code_sha256": SHA_A,
        "resolved_inputs_sha256": SHA_B,
        "cohort_sha256": SHA_C,
        "universe_sha256": SHA_D,
        "runner_identity": "tests.FakeRunner",
        "timeout_seconds": 10.0,
        "requested_network_policy": "none",
    }
    baseline = execution_context_sha256(
        **common,
        runtime_environment_sha256=SHA_A,
        runner_configuration_sha256=SHA_B,
    )
    assert (
        baseline == "72f6b213bcb83e309de2c5366fdea27ce28f84ffd3906d7bfce1fa6492b33b08"
    )
    assert baseline == execution_context_sha256(
        **common,
        runtime_environment_sha256=SHA_A,
        runner_configuration_sha256=SHA_B,
        trajectory_sha256=None,
        trajectory_authority_sha256=None,
    )
    assert baseline != execution_context_sha256(
        **common,
        runtime_environment_sha256=SHA_C,
        runner_configuration_sha256=SHA_B,
    )
    assert baseline != execution_context_sha256(
        **common,
        runtime_environment_sha256=SHA_A,
        runner_configuration_sha256=SHA_D,
    )
    with_trajectory = execution_context_sha256(
        **common,
        runtime_environment_sha256=SHA_A,
        runner_configuration_sha256=SHA_B,
        trajectory_sha256=SHA_C,
        trajectory_authority_sha256=SHA_D,
    )
    assert baseline != with_trajectory
    assert with_trajectory != execution_context_sha256(
        **common,
        runtime_environment_sha256=SHA_A,
        runner_configuration_sha256=SHA_B,
        trajectory_sha256=SHA_E,
        trajectory_authority_sha256=SHA_D,
    )
    assert with_trajectory != execution_context_sha256(
        **common,
        runtime_environment_sha256=SHA_A,
        runner_configuration_sha256=SHA_B,
        trajectory_sha256=SHA_C,
        trajectory_authority_sha256=SHA_E,
    )
    with pytest.raises(
        StepAuthorityRuntimeError,
        match="authority cannot be bound without trajectory bytes",
    ):
        execution_context_sha256(
            **common,
            runtime_environment_sha256=SHA_A,
            runner_configuration_sha256=SHA_B,
            trajectory_authority_sha256=SHA_D,
        )


def test_concept_audit_identity_can_supersede_without_rewriting_execution(
    tmp_path: Path,
) -> None:
    (
        coordinates,
        executed,
        _context_digest,
        budget,
        receipt_path,
    ) = _sealed_execution_for_replay(tmp_path)
    old_ref = seal_concept_audit_capsule(
        coordinates,
        parent_ref=executed.ref,
        findings=[],
        audit_key=SHA_A,
        auditor_identity_sha256=SHA_B,
        environment_sha256=SHA_C,
        validator_implementation_sha256=SHA_D,
    )
    old = load_verified_step_authority_capsule(tmp_path, ref=old_ref)
    replacement_ref = seal_concept_audit_capsule(
        coordinates,
        parent_ref=old_ref,
        findings=[
            ValidationFinding(
                validator="replacement_auditor",
                severity="error",
                message="replacement audit",
            )
        ],
        audit_key=SHA_E,
        auditor_identity_sha256=SHA_E,
        environment_sha256=SHA_C,
        validator_implementation_sha256=SHA_D,
    )
    replacement = load_verified_step_authority_capsule(
        tmp_path,
        ref=replacement_ref,
    )

    assert replacement.capsule.execution == old.capsule.execution
    assert replacement.capsule.concept_audit is not None
    assert replacement.capsule.concept_audit.auditor_identity_sha256 == SHA_E
    assert replacement.capsule.concept_audit.result == "blocked"
    assert replacement.capsule.parent_capsule_sha256 == old_ref.capsule_sha256

    repaired_ref = persist_candidate_code(
        coordinates,
        "import json\nprint(json.dumps({'ok': 'after replacement audit'}))\n",
    )
    binding = _repair_binding(
        coordinates,
        attempt_id=1,
        before_code_sha256=replacement.capsule.candidate_code.sha256,
    )
    attempt_id = budget.reserve_logical_repair(
        "concept",
        max_repairs=1,
        binding=binding.payload(),
        binding_sha256=binding.sha256,
    )
    budget.consume("concept_repair_patch")
    budget.complete_logical_repair_transport(
        attempt_id=attempt_id,
        mode="minimal_patch",
        after_code_sha256=repaired_ref.sha256,
        after_code_size_bytes=repaired_ref.size_bytes,
    )
    repaired_capsule_ref = seal_repair_candidate_from_receipt(
        coordinates,
        parent_ref=replacement_ref,
        checkpoint_parent_ref=replacement_ref,
        code_ref=repaired_ref,
        receipt_state=load_provider_call_budget_state(
            receipt_path,
            step_id=coordinates.step_id,
        ),
        attempt_id=attempt_id,
        failure_status="concept_failed",
    )
    assert (
        load_verified_step_authority_capsule(
            tmp_path,
            ref=repaired_capsule_ref,
        ).capsule.candidate_origin.kind
        == "repair_patch"
    )


def test_execution_identity_tamper_is_rejected_on_load(tmp_path: Path) -> None:
    coordinates, _code, candidate_ref, _budget, _receipt = _initial_candidate(tmp_path)
    step_dir = tmp_path / "original" / "01_summary"
    out_dir = step_dir / "outputs"
    out_dir.mkdir(parents=True)
    script = step_dir / "analysis.py"
    script.write_text(
        load_verified_step_authority_capsule(
            tmp_path, ref=candidate_ref
        ).candidate_code,
        encoding="utf-8",
    )
    context_digest = execution_context_sha256(
        code_sha256=load_verified_step_authority_capsule(
            tmp_path, ref=candidate_ref
        ).capsule.candidate_code.sha256,
        resolved_inputs_sha256=coordinates.resolved_inputs.sha256,
        cohort_sha256=SHA_A,
        universe_sha256=SHA_B,
        runner_identity="tests.FakeRunner",
        timeout_seconds=10.0,
        requested_network_policy="none",
    )
    result = RunResult(
        step_id="01_summary",
        script_path=script,
        cwd=step_dir,
        out_dir=out_dir,
        stdout="",
        stderr="",
        returncode=1,
        duration_seconds=0.1,
        artefacts=[],
        requested_network_policy="none",
        effective_isolation="test",
        runtime_provenance={},
    )
    executed_ref = seal_execution_capsule(
        coordinates,
        parent_ref=candidate_ref,
        run_result=result,
        execution_context_digest=context_digest,
    )
    capsule_path = next(
        (tmp_path / ".step_authority" / "capsules").rglob(
            f"{executed_ref.capsule_sha256}.json"
        )
    )
    payload = json.loads(capsule_path.read_text(encoding="utf-8"))
    payload["execution"]["returncode"] = 0
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    new_digest = __import__("hashlib").sha256(raw).hexdigest()
    tampered_dir = tmp_path / ".step_authority" / "capsules" / "sha256" / new_digest[:2]
    tampered_dir.mkdir(parents=True, exist_ok=True)
    tampered_path = tampered_dir / f"{new_digest}.json"
    tampered_path.write_bytes(raw)
    with pytest.raises(StepAuthorityCapsuleError, match="schema is invalid"):
        load_verified_step_authority_capsule(
            tmp_path,
            ref=executed_ref.model_copy(update={"capsule_sha256": new_digest}),
        )


def test_frozen_scoped_context_accepts_only_memory_metadata_drift(
    tmp_path: Path,
) -> None:
    first_context = ResearchContext(
        research_question="Summarize the cohort.",
        cohort=CohortDescriptor(
            cohort_name="capsule_context",
            database="synthetic",
            n_patients=3,
            n_stays=3,
        ),
        variables=[],
        target_outcome="death",
        notes="first run memory",
        created_at=datetime(2026, 7, 16, 10, tzinfo=timezone.utc),
    )
    coordinates, _code, candidate_ref, _budget, _receipt = _initial_candidate(
        tmp_path,
        scoped_coder_context=first_context.model_dump(mode="json"),
    )
    verified = load_verified_step_authority_capsule(tmp_path, ref=candidate_ref)
    resumed_context = first_context.model_copy(
        update={
            "notes": "newly appended run memory",
            "created_at": datetime(2026, 7, 16, 11, tzinfo=timezone.utc),
        }
    )
    resumed_coordinates = _coordinates(
        tmp_path,
        scoped_coder_context=resumed_context.model_dump(mode="json"),
    )

    adopted = adopt_frozen_scoped_coder_context(verified, resumed_coordinates)
    assert adopted is not None
    frozen_context, frozen_coordinates = adopted
    assert frozen_context.notes == "first run memory"
    assert frozen_coordinates.scoped_coder_context == coordinates.scoped_coder_context

    changed_science = resumed_context.model_copy(update={"target_outcome": "icu_death"})
    changed_coordinates = _coordinates(
        tmp_path,
        scoped_coder_context=changed_science.model_dump(mode="json"),
    )
    assert adopt_frozen_scoped_coder_context(verified, changed_coordinates) is None


def test_frozen_scoped_v2_context_preserves_typed_authority_on_adoption(
    tmp_path: Path,
) -> None:
    run_dir, _cohort_path, context, _identity, _cohort, _trajectory = (
        _prepare_typed_run(tmp_path)
    )
    first_context = context.model_copy(
        update={
            "notes": "first run memory",
            "created_at": datetime(2026, 7, 16, 10, tzinfo=timezone.utc),
        }
    )
    coordinates, _code, candidate_ref, _budget, _receipt = _initial_candidate(
        run_dir,
        scoped_coder_context=first_context.model_dump(mode="json"),
    )
    verified = load_verified_step_authority_capsule(run_dir, ref=candidate_ref)
    resumed_context = first_context.model_copy(
        update={
            "notes": "newly appended run memory",
            "created_at": datetime(2026, 7, 16, 11, tzinfo=timezone.utc),
        }
    )
    resumed_coordinates = _coordinates(
        run_dir,
        scoped_coder_context=resumed_context.model_dump(mode="json"),
    )

    adopted = adopt_frozen_scoped_coder_context(verified, resumed_coordinates)

    assert adopted is not None
    frozen_context, frozen_coordinates = adopted
    assert isinstance(frozen_context, ResearchContextV2)
    assert (
        frozen_context.materialized_inputs.model_dump(mode="json")
        == first_context.materialized_inputs.model_dump(mode="json")
    )
    assert frozen_coordinates.scoped_coder_context == coordinates.scoped_coder_context


def test_frozen_scoped_context_binds_host_authority_outside_user_notes(
    tmp_path: Path,
) -> None:
    context = ResearchContext(
        research_question="Summarize the cohort.",
        cohort=CohortDescriptor(
            cohort_name="capsule_host_authority",
            database="synthetic",
            n_patients=3,
            n_stays=3,
        ),
        variables=[],
        notes="user note with HOST-OWNED words",
        created_at=datetime(2026, 7, 16, 10, tzinfo=timezone.utc),
    )
    authority = HostCoderAuthority().append("exact schema receipt A")

    def wrapped(value: ResearchContext, host: HostCoderAuthority) -> dict:
        return {
            "research_context": value.model_dump(mode="json"),
            "host_coder_authority": host.payload(),
        }

    coordinates, _code, candidate_ref, _budget, _receipt = _initial_candidate(
        tmp_path,
        scoped_coder_context=wrapped(context, authority),
    )
    verified = load_verified_step_authority_capsule(tmp_path, ref=candidate_ref)
    resumed_context = context.model_copy(
        update={
            "notes": "different user run memory",
            "created_at": datetime(2026, 7, 16, 11, tzinfo=timezone.utc),
        }
    )
    same_authority_coordinates = _coordinates(
        tmp_path,
        scoped_coder_context=wrapped(resumed_context, authority),
    )

    adopted = adopt_frozen_scoped_coder_context(
        verified,
        same_authority_coordinates,
    )
    assert adopted is not None
    assert adopted[1].scoped_coder_context == coordinates.scoped_coder_context

    changed_authority_coordinates = _coordinates(
        tmp_path,
        scoped_coder_context=wrapped(
            resumed_context,
            HostCoderAuthority().append("exact schema receipt B"),
        ),
    )
    assert (
        adopt_frozen_scoped_coder_context(
            verified,
            changed_authority_coordinates,
        )
        is None
    )
    assert (
        changed_authority_coordinates.scoped_coder_context
        != same_authority_coordinates.scoped_coder_context
    )
