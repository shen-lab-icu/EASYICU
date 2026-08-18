"""Workflow integration for immutable step-authority capsules.

The storage module owns bytes.  This module joins those bytes to the current
checkpoint and provider receipt, reconstructs runner results, and enforces the
workflow-specific parent rules.  It never publishes evidence or marks a step
successful.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
import json
import os
import platform
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping, Optional, Sequence

from pydantic import ValidationError

from ..canonical_json import (
    canonical_json_bytes as _canonical_json_bytes,
    canonical_sha256 as _canonical_sha256,
    sha256_bytes as _sha256,
)
from ..contracts.method_packages import (
    BASELINE_PACKAGES,
    CURATED_METHOD_PACKAGES,
    FINGERPRINT_ONLY_DISTRIBUTIONS,
    OPTIONAL_BASELINE_PACKAGES,
)
from ..contracts.execution_result import RunnerFailureCode, RunResult
from ..gates.semantics import blocking_validator_findings
from .provider_budget import ProviderCallBudgetReceiptState
from ..authority.runtime_artifacts import (
    current_step_records,
    load_run_artifact_authority,
)
from ..schema import ResearchContext, ValidationFinding
from ..research_context.typed import parse_research_context
from .coder_authority import HostCoderAuthority
from .step_capsule import (
    CandidateOrigin,
    ConceptAuditSeal,
    ContentRef,
    ExecutionOutput,
    ExecutionSeal,
    StepAuthorityCapsule,
    StepAuthorityCapsuleError,
    StepAuthorityCapsuleRef,
    VerifiedStepAuthorityCapsule,
    concept_audit_authority_sha256,
    execution_seal_identity_sha256,
    input_representation_upgrade_matches,
    load_verified_step_authority_capsule,
    put_content_blob,
    read_verified_content,
    scoped_coder_representation_upgrade_matches,
    seal_step_authority_capsule,
)


class StepAuthorityRuntimeError(RuntimeError):
    """A checkpoint, receipt, capsule, or replay boundary is inconsistent."""


@dataclass(frozen=True)
class StepAuthorityCoordinates:
    """Fixed host-owned authority shared by every candidate for one step."""

    run_dir: Path
    step_id: str
    run_input_capsule_sha256: str
    planner_scope: ContentRef
    scoped_coder_context: ContentRef
    resolved_inputs: ContentRef
    typed_bindings_sha256: str
    upstream_authority_sha256: str
    deterministic_gate_fingerprint: str
    engine_code_sha256: str
    validator_code_sha256: str
    prompt_pack_version: str
    prompt_pack_sha256: str
    upstream_authority_payload: Optional[bytes] = None

    def initial_generation_binding(self) -> dict[str, object]:
        return {
            "schema_version": "easyicu.initial_generation_authority/1",
            "step_id": self.step_id,
            "run_input_capsule_sha256": self.run_input_capsule_sha256,
            "planner_scope_sha256": self.planner_scope.sha256,
            "scoped_coder_context_sha256": self.scoped_coder_context.sha256,
            "resolved_inputs_sha256": self.resolved_inputs.sha256,
            "typed_bindings_sha256": self.typed_bindings_sha256,
            "upstream_authority_sha256": self.upstream_authority_sha256,
            "deterministic_gate_fingerprint": self.deterministic_gate_fingerprint,
            "engine_code_sha256": self.engine_code_sha256,
            "validator_code_sha256": self.validator_code_sha256,
            "prompt_pack_version": self.prompt_pack_version,
            "prompt_pack_sha256": self.prompt_pack_sha256,
        }

    @property
    def authority_binding_sha256(self) -> str:
        return _canonical_sha256(self.initial_generation_binding())


def coordinates_from_verified_capsule(
    run_dir: str | Path,
    verified: VerifiedStepAuthorityCapsule,
) -> StepAuthorityCoordinates:
    """Recover the exact historical coordinates of a verified capsule."""

    capsule = verified.capsule
    return StepAuthorityCoordinates(
        run_dir=Path(run_dir).resolve(strict=True),
        step_id=capsule.step_id,
        run_input_capsule_sha256=capsule.run_input_capsule_sha256,
        planner_scope=capsule.planner_scope,
        scoped_coder_context=capsule.scoped_coder_context,
        resolved_inputs=capsule.resolved_inputs,
        typed_bindings_sha256=capsule.typed_bindings_sha256,
        upstream_authority_sha256=capsule.upstream_authority_sha256,
        deterministic_gate_fingerprint=capsule.deterministic_gate_fingerprint,
        engine_code_sha256=capsule.engine_code_sha256,
        validator_code_sha256=capsule.validator_code_sha256,
        prompt_pack_version=capsule.prompt_pack_version,
        prompt_pack_sha256=capsule.prompt_pack_sha256,
    )


def prepare_step_authority_coordinates(
    *,
    run_dir: str | Path,
    step_id: str,
    run_input_capsule_sha256: str,
    planner_scope: Mapping[str, object],
    scoped_coder_context: Mapping[str, object],
    resolved_inputs_path: str | Path,
    typed_bindings: Mapping[str, object],
    upstream_authority: Mapping[str, object],
    deterministic_gate_fingerprint: str,
    engine_code_sha256: str,
    validator_code_sha256: str,
    prompt_pack_version: str,
    prompt_pack: Mapping[str, object],
) -> StepAuthorityCoordinates:
    """Persist immutable step inputs and return their fixed authority tuple."""

    root = Path(run_dir)
    resolved_path = Path(resolved_inputs_path)
    try:
        resolved_payload = resolved_path.read_bytes()
        parsed = json.loads(resolved_payload)
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as exc:
        raise StepAuthorityRuntimeError(
            "resolved-input authority is missing or invalid"
        ) from exc
    if not isinstance(parsed, dict):
        raise StepAuthorityRuntimeError("resolved-input authority must be an object")
    return StepAuthorityCoordinates(
        run_dir=root,
        step_id=str(step_id),
        run_input_capsule_sha256=str(run_input_capsule_sha256),
        planner_scope=put_content_blob(
            root,
            payload=_canonical_json_bytes(dict(planner_scope)),
            media_type="application/json",
        ),
        scoped_coder_context=put_content_blob(
            root,
            payload=_canonical_json_bytes(dict(scoped_coder_context)),
            media_type="application/json",
        ),
        resolved_inputs=put_content_blob(
            root,
            payload=resolved_payload,
            media_type="application/json",
        ),
        typed_bindings_sha256=_canonical_sha256(dict(typed_bindings)),
        upstream_authority_sha256=_canonical_sha256(dict(upstream_authority)),
        deterministic_gate_fingerprint=str(deterministic_gate_fingerprint),
        engine_code_sha256=str(engine_code_sha256),
        validator_code_sha256=str(validator_code_sha256),
        prompt_pack_version=str(prompt_pack_version),
        prompt_pack_sha256=_canonical_sha256(dict(prompt_pack)),
        upstream_authority_payload=_canonical_json_bytes(dict(upstream_authority)),
    )


def persist_candidate_code(
    coordinates: StepAuthorityCoordinates,
    code: str,
) -> ContentRef:
    """Persist normalized candidate bytes before a provider receipt is terminal."""

    return put_content_blob(
        coordinates.run_dir,
        payload=str(code).encode("utf-8"),
        media_type="text/x-python",
    )


def _candidate_capsule(
    coordinates: StepAuthorityCoordinates,
    *,
    code_ref: ContentRef,
    origin: CandidateOrigin,
    parent_ref: Optional[StepAuthorityCapsuleRef] = None,
) -> StepAuthorityCapsule:
    return StepAuthorityCapsule(
        step_id=coordinates.step_id,
        stage="candidate",
        parent_capsule_sha256=(
            parent_ref.capsule_sha256 if parent_ref is not None else None
        ),
        run_input_capsule_sha256=coordinates.run_input_capsule_sha256,
        planner_scope=coordinates.planner_scope,
        scoped_coder_context=coordinates.scoped_coder_context,
        resolved_inputs=coordinates.resolved_inputs,
        typed_bindings_sha256=coordinates.typed_bindings_sha256,
        upstream_authority_sha256=coordinates.upstream_authority_sha256,
        candidate_code=code_ref,
        candidate_origin=origin,
        deterministic_gate_fingerprint=coordinates.deterministic_gate_fingerprint,
        engine_code_sha256=coordinates.engine_code_sha256,
        validator_code_sha256=coordinates.validator_code_sha256,
        prompt_pack_version=coordinates.prompt_pack_version,
        prompt_pack_sha256=coordinates.prompt_pack_sha256,
    )


def _capsule_matches_coordinates(
    capsule: StepAuthorityCapsule,
    coordinates: StepAuthorityCoordinates,
) -> bool:
    return (
        capsule.step_id == coordinates.step_id
        and capsule.run_input_capsule_sha256 == coordinates.run_input_capsule_sha256
        and capsule.planner_scope == coordinates.planner_scope
        and capsule.scoped_coder_context == coordinates.scoped_coder_context
        and capsule.resolved_inputs == coordinates.resolved_inputs
        and capsule.typed_bindings_sha256 == coordinates.typed_bindings_sha256
        and capsule.upstream_authority_sha256 == coordinates.upstream_authority_sha256
        and capsule.deterministic_gate_fingerprint
        == coordinates.deterministic_gate_fingerprint
        and capsule.engine_code_sha256 == coordinates.engine_code_sha256
        and capsule.validator_code_sha256 == coordinates.validator_code_sha256
        and capsule.prompt_pack_version == coordinates.prompt_pack_version
        and capsule.prompt_pack_sha256 == coordinates.prompt_pack_sha256
    )


def seal_initial_generation_candidate(
    coordinates: StepAuthorityCoordinates,
    *,
    code_ref: ContentRef,
    receipt_state: ProviderCallBudgetReceiptState,
) -> StepAuthorityCapsuleRef:
    """Seal initial code only after its exact provider transport is terminal."""

    entry = receipt_state.initial_generation
    if entry is None:
        raise StepAuthorityRuntimeError("initial-generation receipt is missing")
    binding = entry.get("binding")
    expected_binding = coordinates.initial_generation_binding()
    transport = entry.get("transport")
    if (
        binding != expected_binding
        or entry.get("binding_sha256") != coordinates.authority_binding_sha256
        or not isinstance(transport, dict)
        or transport.get("state") != "completed"
        or transport.get("after_code_sha256") != code_ref.sha256
        or transport.get("after_code_size_bytes") != code_ref.size_bytes
    ):
        raise StepAuthorityRuntimeError(
            "initial-generation receipt does not bind the persisted candidate"
        )
    origin = CandidateOrigin(
        kind="initial_generation",
        authority_binding_sha256=coordinates.authority_binding_sha256,
        provider_category="initial_generation",
        provider_transport_id=str(entry["provider_transport_id"]),
    )
    return seal_step_authority_capsule(
        coordinates.run_dir,
        _candidate_capsule(coordinates, code_ref=code_ref, origin=origin),
    )


def initial_generation_code_ref(
    coordinates: StepAuthorityCoordinates,
    receipt_state: ProviderCallBudgetReceiptState,
) -> ContentRef:
    """Recover the explicitly receipt-bound code blob without scanning storage."""

    entry = receipt_state.initial_generation
    if (
        entry is None
        or entry.get("binding") != coordinates.initial_generation_binding()
    ):
        raise StepAuthorityRuntimeError(
            "initial-generation receipt does not match current step authority"
        )
    transport = entry.get("transport")
    if not isinstance(transport, dict) or transport.get("state") != "completed":
        raise StepAuthorityRuntimeError("initial-generation result is not completed")
    ref = ContentRef(
        sha256=str(transport.get("after_code_sha256")),
        size_bytes=int(transport.get("after_code_size_bytes")),
        media_type="text/x-python",
    )
    read_verified_content(coordinates.run_dir, ref)
    return ref


_CONCEPT_REPAIR_CATEGORIES = {
    "compatibility_repair",
    "concept_repair",
    "post_mutation_concept_repair",
}
_OUTPUT_REPAIR_CATEGORIES = {"contract_repair", "visual_repair"}


def _validate_repair_parent_policy(
    parent: StepAuthorityCapsule,
    *,
    provider_category: str,
    failure_status: str,
) -> None:
    if provider_category in _CONCEPT_REPAIR_CATEGORIES:
        if parent.stage == "candidate":
            return
        if (
            parent.stage in {"concept_audited", "executed_concept_audited"}
            and parent.concept_audit is not None
            and parent.concept_audit.result == "blocked"
        ):
            return
    elif provider_category == "runtime_repair":
        if parent.execution is not None and (
            parent.execution.returncode != 0 or parent.execution.timed_out
        ):
            return
    elif provider_category in _OUTPUT_REPAIR_CATEGORIES:
        if parent.execution is not None and failure_status in {
            "contract_failed",
            "visual_failed",
        }:
            return
    elif provider_category == "critic_resume_repair":
        if failure_status == "critic_failed":
            return
    raise StepAuthorityRuntimeError(
        "repair route is incompatible with the checkpoint-selected parent stage"
    )


def seal_repair_candidate_from_receipt(
    coordinates: StepAuthorityCoordinates,
    *,
    parent_ref: StepAuthorityCapsuleRef,
    checkpoint_parent_ref: StepAuthorityCapsuleRef,
    code_ref: ContentRef,
    receipt_state: ProviderCallBudgetReceiptState,
    attempt_id: int,
    failure_status: str,
) -> StepAuthorityCapsuleRef:
    """Join a completed repair receipt to its exact current parent and bytes."""

    if parent_ref != checkpoint_parent_ref:
        raise StepAuthorityRuntimeError(
            "repair parent is not the capsule selected by the newest checkpoint"
        )
    parent = load_verified_step_authority_capsule(
        coordinates.run_dir,
        ref=parent_ref,
        expected_step_id=coordinates.step_id,
    )
    if not _capsule_matches_coordinates(parent.capsule, coordinates):
        raise StepAuthorityRuntimeError("repair parent authority is stale")
    if (
        isinstance(attempt_id, bool)
        or not isinstance(attempt_id, int)
        or not 1 <= attempt_id <= len(receipt_state.logical_repairs)
    ):
        raise StepAuthorityRuntimeError("repair attempt is not in the verified ledger")
    entry = receipt_state.logical_repairs[attempt_id - 1]
    binding = entry.get("binding")
    transport = entry.get("transport")
    if not isinstance(binding, dict) or not isinstance(transport, dict):
        raise StepAuthorityRuntimeError("repair receipt lacks bound transport data")
    provider_category = str(binding.get("provider_category") or "")
    _validate_repair_parent_policy(
        parent.capsule,
        provider_category=provider_category,
        failure_status=str(failure_status),
    )
    expected_binding_fields = {
        "step_id": coordinates.step_id,
        "attempt_id": attempt_id,
        "before_code_sha256": parent.capsule.candidate_code.sha256,
        "step_spec_sha256": coordinates.planner_scope.sha256,
        "resolved_inputs_sha256": coordinates.resolved_inputs.sha256,
        "coder_context_sha256": coordinates.scoped_coder_context.sha256,
        "engine_validator_sha256": coordinates.deterministic_gate_fingerprint,
        "prompt_pack_version": coordinates.prompt_pack_version,
        "run_input_capsule_sha256": coordinates.run_input_capsule_sha256,
    }
    if any(binding.get(key) != value for key, value in expected_binding_fields.items()):
        raise StepAuthorityRuntimeError(
            "repair receipt before-code or step authority does not match its parent"
        )
    if (
        transport.get("state") != "completed"
        or transport.get("after_code_sha256") != code_ref.sha256
    ):
        raise StepAuthorityRuntimeError(
            "repair transport does not bind the persisted candidate bytes"
        )
    mode = str(transport.get("mode") or "")
    if mode in {"patch", "minimal_patch", "full_rewrite_response_patch"}:
        kind = "repair_patch"
    elif mode in {"full_rewrite", "patch_response_full_script"}:
        kind = "repair_full_rewrite"
    else:
        raise StepAuthorityRuntimeError("repair transport mode is unsupported")
    origin = CandidateOrigin(
        kind=kind,
        authority_binding_sha256=str(entry.get("binding_sha256")),
        provider_category=provider_category,
        provider_transport_id=f"{provider_category}:{attempt_id}:{mode}",
        logical_repair_attempt_id=attempt_id,
        repair_ticket_sha256=str(binding.get("repair_ticket_sha256")),
    )
    return seal_step_authority_capsule(
        coordinates.run_dir,
        _candidate_capsule(
            coordinates,
            code_ref=code_ref,
            origin=origin,
            parent_ref=parent_ref,
        ),
    )


def repair_code_ref(
    receipt_state: ProviderCallBudgetReceiptState,
    *,
    attempt_id: int,
) -> ContentRef:
    """Recover only the persisted code explicitly named by a repair receipt."""

    if (
        isinstance(attempt_id, bool)
        or not isinstance(attempt_id, int)
        or not 1 <= attempt_id <= len(receipt_state.logical_repairs)
    ):
        raise StepAuthorityRuntimeError("repair attempt is not in the verified ledger")
    transport = receipt_state.logical_repairs[attempt_id - 1].get("transport")
    if not isinstance(transport, dict) or transport.get("state") != "completed":
        raise StepAuthorityRuntimeError("repair result is not completed")
    if transport.get("result_persistence") != "content_addressed":
        raise StepAuthorityRuntimeError(
            "completed repair has no content-addressed persisted result"
        )
    size_bytes = transport.get("after_code_size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
        raise StepAuthorityRuntimeError(
            "completed repair lacks the persisted code size required for recovery"
        )
    return ContentRef(
        sha256=str(transport.get("after_code_sha256")),
        size_bytes=size_bytes,
        media_type="text/x-python",
    )


def seal_legacy_candidate(
    coordinates: StepAuthorityCoordinates,
    *,
    code_ref: ContentRef,
    adopted_from_ref: Optional[StepAuthorityCapsuleRef] = None,
    input_representation_upgrade_proof: Optional[ContentRef] = None,
) -> StepAuthorityCapsuleRef:
    """Adopt pre-capsule or host-selected code without inventing provider lineage."""

    origin = CandidateOrigin(
        kind="legacy_adoption",
        authority_binding_sha256=coordinates.authority_binding_sha256,
        adopted_from_capsule_sha256=(
            adopted_from_ref.capsule_sha256 if adopted_from_ref is not None else None
        ),
        input_representation_upgrade_proof=input_representation_upgrade_proof,
    )
    return seal_step_authority_capsule(
        coordinates.run_dir,
        _candidate_capsule(coordinates, code_ref=code_ref, origin=origin),
    )


def seal_deterministic_candidate(
    coordinates: StepAuthorityCoordinates,
    *,
    parent_ref: StepAuthorityCapsuleRef,
    code_ref: ContentRef,
    reason: str,
) -> StepAuthorityCapsuleRef:
    """Seal a host-authorized deterministic mutation of the current candidate."""

    parent = load_verified_step_authority_capsule(
        coordinates.run_dir,
        ref=parent_ref,
        expected_step_id=coordinates.step_id,
    )
    if not _capsule_matches_coordinates(parent.capsule, coordinates):
        raise StepAuthorityRuntimeError("deterministic parent authority is stale")
    origin = CandidateOrigin(
        kind="deterministic_mutation",
        authority_binding_sha256=coordinates.authority_binding_sha256,
        deterministic_reason_sha256=_canonical_sha256(str(reason)),
    )
    return seal_step_authority_capsule(
        coordinates.run_dir,
        _candidate_capsule(
            coordinates,
            code_ref=code_ref,
            origin=origin,
            parent_ref=parent_ref,
        ),
    )


def execution_context_sha256(
    *,
    code_sha256: str,
    resolved_inputs_sha256: str,
    cohort_sha256: str,
    universe_sha256: str,
    runner_identity: str,
    timeout_seconds: float,
    requested_network_policy: str,
    runtime_environment_sha256: Optional[str] = None,
    runner_configuration_sha256: Optional[str] = None,
    trajectory_sha256: Optional[str] = None,
    trajectory_authority_sha256: Optional[str] = None,
) -> str:
    if trajectory_authority_sha256 is not None and trajectory_sha256 is None:
        raise StepAuthorityRuntimeError(
            "trajectory authority cannot be bound without trajectory bytes"
        )
    runtime_digest = (
        str(runtime_environment_sha256)
        if runtime_environment_sha256 is not None
        else current_execution_runtime_sha256()
    )
    runner_digest = (
        str(runner_configuration_sha256)
        if runner_configuration_sha256 is not None
        else _canonical_sha256({"runner_identity": runner_identity})
    )
    payload = {
        "schema": "easyicu.step_execution_context/1",
        "code_sha256": code_sha256,
        "resolved_inputs_sha256": resolved_inputs_sha256,
        "cohort_sha256": cohort_sha256,
        "universe_sha256": universe_sha256,
        "runner_identity": runner_identity,
        "timeout_seconds": float(timeout_seconds),
        "requested_network_policy": requested_network_policy,
        "runtime_environment_sha256": runtime_digest,
        "runner_configuration_sha256": runner_digest,
    }
    # Preserve the exact v1 payload for historical executions without a
    # trajectory.  When a trajectory is present it is a scientific execution
    # input and must invalidate replay if either its bytes or typed selector
    # changes.
    if trajectory_sha256 is not None:
        payload["trajectory_sha256"] = str(trajectory_sha256)
    if trajectory_authority_sha256 is not None:
        payload["trajectory_authority_sha256"] = str(trajectory_authority_sha256)
    return _canonical_sha256(payload)


def current_execution_runtime_sha256() -> str:
    """Fingerprint the host runtime that can change computed numeric results."""

    distribution_names = {
        "scikit-learn" if package == "sklearn" else package
        for package in (*BASELINE_PACKAGES, *OPTIONAL_BASELINE_PACKAGES)
    }
    distribution_names.update(package.pip_name for package in CURATED_METHOD_PACKAGES)
    distribution_names.update(FINGERPRINT_ONLY_DISTRIBUTIONS)
    package_versions: dict[str, str] = {}
    for package in sorted(distribution_names):
        try:
            package_versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            package_versions[package] = "unavailable"
    return _canonical_sha256(
        {
            "schema": "easyicu.execution_runtime_environment/1",
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "packages": package_versions,
        }
    )


def _media_type(path: Path) -> str:
    return {
        ".csv": "text/csv",
        ".html": "text/html",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".json": "application/json",
        ".md": "text/markdown",
        ".parquet": "application/x-parquet",
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".py": "text/x-python",
        ".svg": "image/svg+xml",
        ".txt": "text/plain",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }.get(path.suffix.lower(), "application/octet-stream")


def seal_execution_capsule(
    coordinates: StepAuthorityCoordinates,
    *,
    parent_ref: StepAuthorityCapsuleRef,
    run_result: RunResult,
    execution_context_digest: str,
) -> StepAuthorityCapsuleRef:
    """Seal one completed runner attempt without granting current authority."""

    parent = load_verified_step_authority_capsule(
        coordinates.run_dir,
        ref=parent_ref,
        expected_step_id=coordinates.step_id,
    )
    if not _capsule_matches_coordinates(parent.capsule, coordinates):
        raise StepAuthorityRuntimeError("execution parent authority is stale")
    if parent.capsule.stage not in {"candidate", "concept_audited"}:
        raise StepAuthorityRuntimeError(
            "execution parent is not an executable candidate"
        )
    if str(run_result.step_id) != coordinates.step_id:
        raise StepAuthorityRuntimeError("runner result belongs to another step")
    try:
        executed_script = Path(run_result.script_path).read_bytes()
    except OSError as exc:
        raise StepAuthorityRuntimeError(
            "executed script control copy is unreadable"
        ) from exc
    if _sha256(executed_script) != parent.capsule.candidate_code.sha256:
        raise StepAuthorityRuntimeError(
            "executed script does not match its candidate authority"
        )
    runtime_ref = put_content_blob(
        coordinates.run_dir,
        payload=_canonical_json_bytes(dict(run_result.runtime_provenance)),
        media_type="application/json",
    )
    stdout_ref = put_content_blob(
        coordinates.run_dir,
        payload=str(run_result.stdout).encode("utf-8"),
        media_type="text/plain",
    )
    stderr_ref = put_content_blob(
        coordinates.run_dir,
        payload=str(run_result.stderr).encode("utf-8"),
        media_type="text/plain",
    )
    runner_log_ref: Optional[ContentRef] = None
    if run_result.runner_log_path is not None:
        try:
            log_payload = Path(run_result.runner_log_path).read_bytes()
            log_payload.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise StepAuthorityRuntimeError("runner log is unreadable") from exc
        runner_log_ref = put_content_blob(
            coordinates.run_dir,
            payload=log_payload,
            media_type="text/plain",
        )
    outputs: list[ExecutionOutput] = []
    if run_result.outputs_safe_to_collect:
        out_root = Path(run_result.out_dir).resolve(strict=True)
        for path in sorted(Path(item) for item in run_result.artefacts):
            try:
                if path.is_symlink():
                    raise OSError("symbolic link")
                relative = path.resolve(strict=True).relative_to(out_root).as_posix()
                if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                    raise OSError("not a regular file")
                payload = path.read_bytes()
            except (OSError, ValueError) as exc:
                raise StepAuthorityRuntimeError(
                    "runner output is outside its sealed output directory"
                ) from exc
            outputs.append(
                ExecutionOutput(
                    logical_relative_path=relative,
                    content=put_content_blob(
                        coordinates.run_dir,
                        payload=payload,
                        media_type=_media_type(path),
                    ),
                )
            )
    outputs.sort(key=lambda item: item.logical_relative_path)
    execution_payload: dict[str, object] = {
        "execution_context_sha256": execution_context_digest,
        "code_sha256": parent.capsule.candidate_code.sha256,
        "resolved_inputs_sha256": coordinates.resolved_inputs.sha256,
        "returncode": int(run_result.returncode),
        "duration_seconds": float(run_result.duration_seconds),
        "timed_out": bool(run_result.timed_out),
        "outputs_safe_to_collect": bool(run_result.outputs_safe_to_collect),
        "requested_network_policy": str(run_result.requested_network_policy),
        "effective_isolation": str(run_result.effective_isolation),
        "isolation_degraded": bool(run_result.isolation_degraded),
        "isolation_degradation_reason": run_result.isolation_degradation_reason,
        "runtime_provenance": runtime_ref,
        "stdout": stdout_ref,
        "stderr": stderr_ref,
        "runner_log": runner_log_ref,
        "runner_failure_code": run_result.runner_failure_code,
        "outputs": tuple(outputs),
    }
    execution_payload["execution_identity_sha256"] = execution_seal_identity_sha256(
        execution_payload
    )
    execution = ExecutionSeal.model_validate(execution_payload)
    stage = (
        "executed_concept_audited"
        if parent.capsule.concept_audit is not None
        else "executed"
    )
    child = parent.capsule.model_copy(
        update={
            "stage": stage,
            "parent_capsule_sha256": parent_ref.capsule_sha256,
            "execution": execution,
        }
    )
    return seal_step_authority_capsule(coordinates.run_dir, child)


def seal_concept_audit_capsule(
    coordinates: StepAuthorityCoordinates,
    *,
    parent_ref: StepAuthorityCapsuleRef,
    findings: Sequence[ValidationFinding],
    audit_key: str,
    auditor_identity_sha256: str,
    environment_sha256: str,
    validator_implementation_sha256: str,
) -> StepAuthorityCapsuleRef:
    """Seal exact concept findings for the checkpoint-selected candidate."""

    parent = load_verified_step_authority_capsule(
        coordinates.run_dir,
        ref=parent_ref,
        expected_step_id=coordinates.step_id,
    )
    if not _capsule_matches_coordinates(parent.capsule, coordinates):
        raise StepAuthorityRuntimeError("concept-audit parent authority is stale")
    findings_ref = put_content_blob(
        coordinates.run_dir,
        payload=_canonical_json_bytes(
            [finding.model_dump(mode="json") for finding in findings]
        ),
        media_type="application/json",
    )
    result = "blocked" if blocking_validator_findings(findings) else "passed"
    audit = ConceptAuditSeal(
        audit_key=audit_key,
        audited_code_sha256=parent.capsule.candidate_code.sha256,
        authority_binding_sha256=concept_audit_authority_sha256(
            parent.capsule,
            audit_key=audit_key,
            auditor_identity_sha256=auditor_identity_sha256,
            environment_sha256=environment_sha256,
            validator_implementation_sha256=validator_implementation_sha256,
        ),
        result=result,
        findings=findings_ref,
        auditor_identity_sha256=auditor_identity_sha256,
        environment_sha256=environment_sha256,
        validator_implementation_sha256=validator_implementation_sha256,
    )
    stage = (
        "executed_concept_audited"
        if parent.capsule.execution is not None
        else "concept_audited"
    )
    child = parent.capsule.model_copy(
        update={
            "stage": stage,
            "parent_capsule_sha256": parent_ref.capsule_sha256,
            "concept_audit": audit,
        }
    )
    return seal_step_authority_capsule(coordinates.run_dir, child)


def read_concept_audit_findings(
    verified: VerifiedStepAuthorityCapsule,
    *,
    run_dir: str | Path,
) -> list[ValidationFinding]:
    audit = verified.capsule.concept_audit
    if audit is None:
        return []
    try:
        payload = json.loads(
            read_verified_content(run_dir, audit.findings).decode("utf-8")
        )
        if not isinstance(payload, list):
            raise TypeError("findings are not a list")
        return [ValidationFinding.model_validate(item) for item in payload]
    except (UnicodeDecodeError, ValueError, TypeError, ValidationError) as exc:
        raise StepAuthorityRuntimeError("sealed concept findings are invalid") from exc


def load_checkpoint_selected_step_capsule(
    run_dir: str | Path,
    *,
    step_id: str,
    checkpoint: Optional[Mapping[str, object]] = None,
) -> Optional[VerifiedStepAuthorityCapsule]:
    """Load only the capsule explicitly selected by the newest checkpoint."""

    authority = (
        dict(checkpoint)
        if checkpoint is not None
        else load_run_artifact_authority(run_dir)
    )
    if authority is None:
        return None
    raw_records = authority.get("per_step_records")
    if not isinstance(raw_records, list):
        raise StepAuthorityRuntimeError("checkpoint step ledger is invalid")
    latest = next(
        (
            record
            for record in current_step_records(raw_records)
            if str(record.get("step_id") or "") == str(step_id)
        ),
        None,
    )
    if latest is None or "step_authority_capsule_ref" not in latest:
        return None
    raw_ref = latest.get("step_authority_capsule_ref")
    try:
        ref = StepAuthorityCapsuleRef.model_validate(raw_ref)
        return load_verified_step_authority_capsule(
            run_dir,
            ref=ref,
            expected_step_id=str(step_id),
        )
    except (ValidationError, StepAuthorityCapsuleError, ValueError) as exc:
        raise StepAuthorityRuntimeError(
            "checkpoint-selected step capsule is missing, corrupt, or inconsistent"
        ) from exc


def load_explicit_failed_step_capsule(
    run_dir: str | Path,
    *,
    step_id: str,
    record: Mapping[str, object],
) -> Optional[VerifiedStepAuthorityCapsule]:
    """Load one explicitly recorded failed candidate for targeted resume.

    Failure retires current scientific authority, but it does not erase the
    immutable candidate bytes needed for a later deterministic repair.  This
    loader never scans capsule storage and never restores evidence aliases: it
    accepts only the exact ref and code digests written into the terminal step
    ledger.  Callers must additionally require an explicit resume cut.
    """

    status = str(record.get("status") or "").strip().lower()
    if status not in {"execution_failed", "contract_failed"}:
        return None
    if (
        record.get("provider_call_budget_receipt_invalid") is True
        or record.get("quarantined_requires_repair") is True
    ):
        return None
    if status == "execution_failed":
        returncode = record.get("returncode")
        timed_out = record.get("timed_out")
        if not (
            timed_out is True
            or (
                isinstance(returncode, int)
                and not isinstance(returncode, bool)
                and returncode != 0
            )
        ):
            return None
    else:
        if (
            record.get("returncode") != 0
            or record.get("timed_out") is not False
            or record.get("outputs_safe_to_collect") is not True
        ):
            return None

    executed_sha256 = str(record.get("executed_code_sha256") or "")
    approved_sha256 = str(record.get("concept_approved_code_sha256") or "")
    if (
        len(executed_sha256) != 64
        or any(char not in "0123456789abcdef" for char in executed_sha256)
        or approved_sha256 != executed_sha256
    ):
        return None
    try:
        ref = StepAuthorityCapsuleRef.model_validate(
            record.get("step_authority_capsule_ref")
        )
        verified = load_verified_step_authority_capsule(
            run_dir,
            ref=ref,
            expected_step_id=str(step_id),
        )
    except (ValidationError, StepAuthorityCapsuleError, ValueError) as exc:
        raise StepAuthorityRuntimeError(
            "explicit failed-step capsule is missing, corrupt, or inconsistent"
        ) from exc
    if verified.capsule.candidate_code.sha256 != executed_sha256:
        raise StepAuthorityRuntimeError(
            "explicit failed-step capsule code digest is inconsistent"
        )
    return verified


def load_explicit_success_step_capsule(
    run_dir: str | Path,
    *,
    step_id: str,
    record: Mapping[str, object],
) -> Optional[VerifiedStepAuthorityCapsule]:
    """Load exact historical success bytes for explicit current revalidation.

    A targeted ``stop_after`` run may temporarily omit later successful steps
    from the current selector while their append-only history remains intact.
    Re-entering one of those steps must not purchase a second initial draft.
    This returns candidate code only; current execution, audit, gates, and
    evidence registration remain mandatory.
    """

    if (
        str(record.get("status") or "").strip().lower() != "ok"
        or record.get("provider_call_budget_receipt_invalid") is True
        or record.get("quarantined_requires_repair") is True
        or record.get("returncode") != 0
        or record.get("timed_out") is not False
        or record.get("outputs_safe_to_collect") is not True
    ):
        return None
    executed_sha256 = str(record.get("executed_code_sha256") or "")
    approved_sha256 = str(record.get("concept_approved_code_sha256") or "")
    if (
        len(executed_sha256) != 64
        or any(char not in "0123456789abcdef" for char in executed_sha256)
        or approved_sha256 != executed_sha256
    ):
        return None
    try:
        ref = StepAuthorityCapsuleRef.model_validate(
            record.get("step_authority_capsule_ref")
        )
        verified = load_verified_step_authority_capsule(
            run_dir,
            ref=ref,
            expected_step_id=str(step_id),
        )
    except (ValidationError, StepAuthorityCapsuleError, ValueError) as exc:
        raise StepAuthorityRuntimeError(
            "explicit successful-step capsule is missing, corrupt, or inconsistent"
        ) from exc
    if verified.capsule.candidate_code.sha256 != executed_sha256:
        raise StepAuthorityRuntimeError(
            "explicit successful-step capsule code digest is inconsistent"
        )
    return verified


def load_explicit_executed_success_step_capsule(
    run_dir: str | Path,
    *,
    step_id: str,
    record: Mapping[str, object],
) -> Optional[VerifiedStepAuthorityCapsule]:
    """Load the exact successful execution observation named by one record.

    Unlike :func:`load_explicit_success_step_capsule`, which intentionally
    recovers candidate code for revalidation, this contract is product-input
    authority.  It therefore accepts only an executed capsule whose sealed
    sandbox result is successful and safe to collect.  The caller decides
    whether a record without a capsule is an older deterministic host path;
    an explicitly named but invalid capsule never falls back to a scan.
    """

    if str(record.get("status") or "").strip().lower() != "ok":
        return None
    raw_ref = record.get("step_authority_capsule_ref")
    if raw_ref is None:
        return None
    try:
        ref = StepAuthorityCapsuleRef.model_validate(raw_ref)
        verified = load_verified_step_authority_capsule(
            run_dir,
            ref=ref,
            expected_step_id=str(step_id),
        )
    except (ValidationError, StepAuthorityCapsuleError, ValueError) as exc:
        raise StepAuthorityRuntimeError(
            "explicit executed-step capsule is missing, corrupt, or inconsistent"
        ) from exc
    capsule = verified.capsule
    execution = capsule.execution
    if capsule.stage not in {"executed", "executed_concept_audited"} or execution is None:
        raise StepAuthorityRuntimeError(
            "explicit successful-step capsule does not seal execution"
        )
    if (
        execution.returncode != 0
        or execution.timed_out
        or not execution.outputs_safe_to_collect
    ):
        raise StepAuthorityRuntimeError(
            "explicit successful-step capsule does not seal a successful result"
        )
    recorded_code_sha256 = str(record.get("executed_code_sha256") or "")
    if recorded_code_sha256 and recorded_code_sha256 != execution.code_sha256:
        raise StepAuthorityRuntimeError(
            "explicit successful-step capsule code digest is inconsistent"
        )
    return verified


def load_latest_explicit_failed_step_capsule_from_history(
    run_dir: str | Path,
    *,
    step_id: str,
    records: Sequence[Mapping[str, object]],
) -> Optional[tuple[VerifiedStepAuthorityCapsule, Mapping[str, object]]]:
    """Load the newest failed candidate hidden by a dependency-status row.

    Dependency blocking and targeted resume checkpoints may append a newer row
    without candidate coordinates.  That row must not erase the last explicit
    immutable candidate.  Walk only the supplied append-only ledger (never the
    capsule directory), stop at the newest record that names a capsule, and
    accept it only through the existing failed terminal loader.  A newer
    ineligible or corrupt candidate therefore blocks fallback to older bytes;
    historical successes keep using their separate revalidation path.
    """

    normalized_step_id = str(step_id)
    for record in reversed(tuple(records)):
        if str(record.get("step_id") or "") != normalized_step_id:
            continue
        if "step_authority_capsule_ref" not in record:
            continue
        failed = load_explicit_failed_step_capsule(
            run_dir,
            step_id=normalized_step_id,
            record=record,
        )
        if failed is not None:
            return failed, record
        return None
    return None


_DEPENDENCY_CANDIDATE_FIELDS = (
    "dependency_blocked_candidate_capsule_ref",
    "dependency_blocked_candidate_code_sha256",
    "dependency_blocked_candidate_attempt_id",
    "dependency_blocked_candidate_source_status",
)


def _load_dependency_blocked_candidate(
    run_dir: str | Path,
    *,
    step_id: str,
    record: Mapping[str, object],
) -> Optional[VerifiedStepAuthorityCapsule]:
    present = [field in record for field in _DEPENDENCY_CANDIDATE_FIELDS]
    if not any(present):
        return None
    if not all(present):
        raise StepAuthorityRuntimeError(
            "dependency-blocked candidate coordinates are incomplete"
        )
    try:
        ref = StepAuthorityCapsuleRef.model_validate(
            record["dependency_blocked_candidate_capsule_ref"]
        )
        verified = load_verified_step_authority_capsule(
            run_dir,
            ref=ref,
            expected_step_id=step_id,
        )
    except (ValidationError, StepAuthorityCapsuleError, ValueError) as exc:
        raise StepAuthorityRuntimeError(
            "dependency-blocked candidate is missing, corrupt, or inconsistent"
        ) from exc
    expected_sha256 = str(record.get("dependency_blocked_candidate_code_sha256") or "")
    if verified.capsule.candidate_code.sha256 != expected_sha256:
        raise StepAuthorityRuntimeError(
            "dependency-blocked candidate code digest is inconsistent"
        )
    return verified


def dependency_blocked_candidate_metadata(
    run_dir: str | Path,
    *,
    step_id: str,
    current_record: Optional[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Carry exact failed-candidate coordinates through a dependency block."""

    if current_record is None:
        return {}
    carried = _load_dependency_blocked_candidate(
        run_dir,
        step_id=step_id,
        record=current_record,
    )
    if carried is not None:
        return {field: current_record[field] for field in _DEPENDENCY_CANDIDATE_FIELDS}
    failed = load_explicit_failed_step_capsule(
        run_dir,
        step_id=step_id,
        record=current_record,
    )
    source_record = current_record
    if failed is None:
        historical = load_latest_explicit_failed_step_capsule_from_history(
            run_dir,
            step_id=step_id,
            records=records,
        )
        if historical is None:
            return {}
        failed, source_record = historical
    return {
        "dependency_blocked_candidate_capsule_ref": failed.ref.model_dump(mode="json"),
        "dependency_blocked_candidate_code_sha256": (
            failed.capsule.candidate_code.sha256
        ),
        "dependency_blocked_candidate_attempt_id": str(
            source_record.get("attempt_id") or ""
        ),
        "dependency_blocked_candidate_source_status": str(
            source_record.get("status") or ""
        ),
    }


def select_explicit_step_capsule_for_targeted_resume(
    run_dir: str | Path,
    *,
    step_id: str,
    current_record: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    deterministic_gate_fingerprint: str,
) -> Optional[tuple[VerifiedStepAuthorityCapsule, dict[str, object]]]:
    """Select exact candidate bytes and monotonic replay metadata.

    This is orchestration glue for an explicit resume cut.  It keeps capsule
    selection out of the execute god-function while preserving the authority
    rules of the individual loaders.
    """

    source_record = current_record
    failed = load_explicit_failed_step_capsule(
        run_dir, step_id=step_id, record=source_record
    )
    success = (
        None
        if failed is not None
        else load_explicit_success_step_capsule(
            run_dir, step_id=step_id, record=source_record
        )
    )
    if (
        failed is None
        and success is None
        and str(current_record.get("status") or "").strip().lower()
        == "blocked_dependency_evidence"
    ):
        carried = _load_dependency_blocked_candidate(
            run_dir,
            step_id=step_id,
            record=current_record,
        )
        if carried is not None:
            failed = carried
        else:
            historical = load_latest_explicit_failed_step_capsule_from_history(
                run_dir,
                step_id=step_id,
                records=records,
            )
            if historical is not None:
                failed, source_record = historical
    if failed is not None:
        code_sha256 = str(
            source_record.get("dependency_blocked_candidate_code_sha256")
            or source_record.get("executed_code_sha256")
            or ""
        )
        if (
            source_record.get("explicit_failed_capsule_reused_code_sha256")
            == code_sha256
            and source_record.get("explicit_failed_capsule_reused_gate_fingerprint")
            == deterministic_gate_fingerprint
        ):
            return None
        return failed, {
            "explicit_failed_capsule_reused": True,
            "explicit_failed_capsule_reused_status": str(
                source_record.get("dependency_blocked_candidate_source_status")
                or source_record.get("status")
                or ""
            ),
            "explicit_failed_capsule_reused_attempt_id": str(
                source_record.get("dependency_blocked_candidate_attempt_id")
                or source_record.get("attempt_id")
                or ""
            ),
            "explicit_failed_capsule_reused_code_sha256": code_sha256,
            "explicit_failed_capsule_reused_gate_fingerprint": (
                deterministic_gate_fingerprint
            ),
        }
    if success is not None:
        return success, {
            "explicit_success_capsule_reused": True,
            "explicit_success_capsule_reused_attempt_id": str(
                source_record.get("attempt_id") or ""
            ),
        }
    return None


def capsule_matches_coordinates(
    verified: VerifiedStepAuthorityCapsule,
    coordinates: StepAuthorityCoordinates,
) -> bool:
    return _capsule_matches_coordinates(verified.capsule, coordinates)


def _decode_scoped_coder_context(
    payload: object,
) -> tuple[dict[str, object], HostCoderAuthority]:
    """Decode current wrapped contexts and legacy flat ResearchContext blobs."""

    if not isinstance(payload, dict):
        raise TypeError("scoped coder context is not an object")
    if set(payload) == {"research_context", "host_coder_authority"}:
        research_context = payload.get("research_context")
        if not isinstance(research_context, dict):
            raise TypeError("wrapped ResearchContext is not an object")
        authority = HostCoderAuthority.from_payload(payload.get("host_coder_authority"))
        return dict(research_context), authority
    return dict(payload), HostCoderAuthority()


def _scoped_coder_contexts_match_except_run_memory(
    current_payload: object,
    frozen_payload: object,
) -> tuple[bool, dict[str, object]]:
    current_context, current_authority = _decode_scoped_coder_context(current_payload)
    frozen_context, frozen_authority = _decode_scoped_coder_context(frozen_payload)
    current_comparable = dict(current_context)
    frozen_comparable = dict(frozen_context)
    for field in ("created_at", "notes"):
        current_comparable.pop(field, None)
        frozen_comparable.pop(field, None)
    return (
        current_comparable == frozen_comparable
        and current_authority == frozen_authority,
        frozen_context,
    )


def adopt_frozen_scoped_coder_context(
    verified: VerifiedStepAuthorityCapsule,
    coordinates: StepAuthorityCoordinates,
) -> Optional[tuple[ResearchContext, StepAuthorityCoordinates]]:
    """Reuse the frozen context only when drift is run-memory metadata alone.

    A resume may append prior-run memory to ``notes`` and refresh ``created_at``.
    Those fields must not invalidate already-audited code, but no exposure,
    outcome, cohort, variable, or other scientific field may change silently.
    """

    frozen_coordinates = replace(
        coordinates,
        scoped_coder_context=verified.capsule.scoped_coder_context,
    )
    if not _capsule_matches_coordinates(verified.capsule, frozen_coordinates):
        return None
    try:
        current_payload = json.loads(
            read_verified_content(
                coordinates.run_dir,
                coordinates.scoped_coder_context,
            ).decode("utf-8")
        )
        frozen_payload = json.loads(
            read_verified_content(
                coordinates.run_dir,
                verified.capsule.scoped_coder_context,
            ).decode("utf-8")
        )
        contexts_match, frozen_context_payload = (
            _scoped_coder_contexts_match_except_run_memory(
                current_payload,
                frozen_payload,
            )
        )
        if not contexts_match:
            return None
        frozen_context = parse_research_context(frozen_context_payload)
    except (UnicodeDecodeError, ValueError, TypeError, ValidationError) as exc:
        raise StepAuthorityRuntimeError(
            "checkpoint-selected scoped coder context is invalid"
        ) from exc
    return frozen_context, frozen_coordinates


def _resolved_inputs_representation_upgrade_proof(
    verified: VerifiedStepAuthorityCapsule,
    coordinates: StepAuthorityCoordinates,
) -> Optional[dict[str, object]]:
    """Prove that only digest-bound typed-table representation facts changed."""

    try:
        historical_manifest = json.loads(
            read_verified_content(
                coordinates.run_dir,
                verified.capsule.resolved_inputs,
            ).decode("utf-8")
        )
        current_manifest = json.loads(
            read_verified_content(
                coordinates.run_dir,
                coordinates.resolved_inputs,
            ).decode("utf-8")
        )
        current_upstream_authority = json.loads(
            coordinates.upstream_authority_payload or b"null"
        )
    except (
        UnicodeDecodeError,
        ValueError,
        TypeError,
        StepAuthorityCapsuleError,
    ) as exc:
        raise StepAuthorityRuntimeError(
            "resolved-input representation authority is invalid"
        ) from exc
    if not isinstance(historical_manifest, dict) or not isinstance(
        current_manifest, dict
    ):
        return None
    historical_inputs = historical_manifest.get("inputs")
    current_inputs = current_manifest.get("inputs")
    if not isinstance(historical_inputs, Mapping) or not isinstance(
        current_inputs, Mapping
    ):
        return None
    if not isinstance(current_upstream_authority, Mapping):
        return None
    historical_upstream = dict(current_upstream_authority)
    historical_upstream["resolved_input_bindings"] = historical_inputs
    if not input_representation_upgrade_matches(
        historical_manifest=historical_manifest,
        current_manifest=current_manifest,
        historical_typed_bindings_sha256=verified.capsule.typed_bindings_sha256,
        current_typed_bindings_sha256=coordinates.typed_bindings_sha256,
        historical_upstream_authority_sha256=(
            verified.capsule.upstream_authority_sha256
        ),
        current_upstream_authority_sha256=coordinates.upstream_authority_sha256,
        historical_upstream_authority=historical_upstream,
        current_upstream_authority=current_upstream_authority,
    ):
        return None
    return {
        "schema_version": "easyicu.input_representation_upgrade_proof/1",
        "historical_capsule_sha256": verified.ref.capsule_sha256,
        "historical_upstream_authority": historical_upstream,
        "current_upstream_authority": dict(current_upstream_authority),
    }


def adopt_candidate_for_control_plane_revalidation(
    verified: VerifiedStepAuthorityCapsule,
    coordinates: StepAuthorityCoordinates,
) -> Optional[
    tuple[ResearchContext, StepAuthorityCoordinates, StepAuthorityCapsuleRef]
]:
    """Adopt old code under new engine/gate/prompt authority, never old seals."""

    capsule = verified.capsule
    immutable_science_matches = (
        capsule.step_id == coordinates.step_id
        and capsule.run_input_capsule_sha256 == coordinates.run_input_capsule_sha256
        and capsule.planner_scope == coordinates.planner_scope
    )
    exact_input_authority = (
        capsule.resolved_inputs == coordinates.resolved_inputs
        and capsule.typed_bindings_sha256 == coordinates.typed_bindings_sha256
        and capsule.upstream_authority_sha256 == coordinates.upstream_authority_sha256
    )
    representation_upgrade_proof = None
    if (
        immutable_science_matches
        and not exact_input_authority
        and coordinates.upstream_authority_payload is not None
    ):
        representation_upgrade_proof = _resolved_inputs_representation_upgrade_proof(
            verified,
            coordinates,
        )
    if not immutable_science_matches or not (
        exact_input_authority or representation_upgrade_proof is not None
    ):
        return None
    try:
        current_payload = json.loads(
            read_verified_content(
                coordinates.run_dir,
                coordinates.scoped_coder_context,
            ).decode("utf-8")
        )
        frozen_payload = json.loads(
            read_verified_content(
                coordinates.run_dir,
                capsule.scoped_coder_context,
            ).decode("utf-8")
        )
        contexts_match, frozen_context_payload = (
            _scoped_coder_contexts_match_except_run_memory(
                current_payload,
                frozen_payload,
            )
        )
        representation_context_upgrade = bool(
            representation_upgrade_proof is not None
            and scoped_coder_representation_upgrade_matches(
                frozen_payload,
                current_payload,
            )
        )
        if not contexts_match and not representation_context_upgrade:
            return None
        selected_context_payload = (
            current_payload.get("research_context")
            if representation_context_upgrade
            and isinstance(current_payload, dict)
            and set(current_payload) == {"research_context", "host_coder_authority"}
            else frozen_context_payload
        )
        if not isinstance(selected_context_payload, dict):
            return None
        selected_context = parse_research_context(selected_context_payload)
    except (UnicodeDecodeError, ValueError, TypeError, ValidationError) as exc:
        raise StepAuthorityRuntimeError(
            "checkpoint-selected scoped coder context is invalid"
        ) from exc
    adopted_coordinates = (
        coordinates
        if representation_upgrade_proof is not None
        else replace(
            coordinates,
            scoped_coder_context=capsule.scoped_coder_context,
        )
    )
    proof_ref = (
        put_content_blob(
            adopted_coordinates.run_dir,
            payload=_canonical_json_bytes(representation_upgrade_proof),
            media_type="application/json",
        )
        if representation_upgrade_proof is not None
        else None
    )
    adopted_ref = seal_legacy_candidate(
        adopted_coordinates,
        code_ref=capsule.candidate_code,
        adopted_from_ref=verified.ref,
        input_representation_upgrade_proof=proof_ref,
    )
    return selected_context, adopted_coordinates, adopted_ref


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise StepAuthorityRuntimeError("replay destination is a symbolic link")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _directory_presence(path: Path, *, label: str) -> bool:
    """Return whether a replay directory exists without following links."""

    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(mode):
        raise StepAuthorityRuntimeError(f"{label} is a symbolic link")
    if not stat.S_ISDIR(mode):
        raise StepAuthorityRuntimeError(f"{label} is not a directory")
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize_sealed_run_result(
    run_dir: str | Path,
    verified: VerifiedStepAuthorityCapsule,
    *,
    expected_execution_context_sha256: str,
) -> RunResult:
    """Materialize one verified execution seal without invoking a runner."""

    execution = verified.capsule.execution
    if execution is None:
        raise StepAuthorityRuntimeError("selected capsule has no execution seal")
    if execution.execution_context_sha256 != expected_execution_context_sha256:
        raise StepAuthorityRuntimeError("sealed execution context is stale")
    if execution.execution_identity_sha256 != execution_seal_identity_sha256(execution):
        raise StepAuthorityRuntimeError("sealed execution identity is invalid")
    root = Path(run_dir).resolve(strict=True)
    steps_root = root / "steps"
    steps_root.mkdir(parents=True, exist_ok=True)
    step_dir = steps_root / verified.capsule.step_id
    if steps_root.is_symlink() or step_dir.is_symlink():
        raise StepAuthorityRuntimeError("replay step path is a symbolic link")
    step_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".capsule-outputs-", dir=step_dir))
    try:
        output_paths: list[Path] = []
        for output in execution.outputs:
            target = staging.joinpath(*Path(output.logical_relative_path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(
                target,
                read_verified_content(root, output.content),
            )
            output_paths.append(target)
        out_dir = step_dir / "outputs"
        backup = step_dir / ".capsule-outputs-backup"
        out_present = _directory_presence(
            out_dir,
            label="replay output destination",
        )
        backup_present = _directory_presence(
            backup,
            label="replay output backup",
        )
        # ``out + backup`` means the previous replay installed its sealed
        # output and crashed before cleanup. Removing the stale backup is safe:
        # neither directory is trusted, and the fresh staging tree below is
        # reconstructed exclusively from verified content-addressed blobs.
        if out_present and backup_present:
            shutil.rmtree(backup)
            _fsync_directory(step_dir)
            backup_present = False
        if out_present:
            os.replace(out_dir, backup)
            _fsync_directory(step_dir)
            backup_present = True
        try:
            os.replace(staging, out_dir)
            _fsync_directory(step_dir)
        except BaseException:
            if backup_present and not _directory_presence(
                out_dir,
                label="replay output destination",
            ):
                os.replace(backup, out_dir)
                _fsync_directory(step_dir)
            raise
        if backup_present:
            shutil.rmtree(backup)
            _fsync_directory(step_dir)
        output_paths = [out_dir / path.relative_to(staging) for path in output_paths]
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    script_path = step_dir / "analysis.py"
    _atomic_write(script_path, verified.candidate_code.encode("utf-8"))
    stdout = read_verified_content(root, execution.stdout).decode("utf-8")
    stderr = read_verified_content(root, execution.stderr).decode("utf-8")
    runner_log_path: Optional[Path] = None
    if execution.runner_log is not None:
        runner_log_path = step_dir / "run.log"
        _atomic_write(
            runner_log_path,
            read_verified_content(root, execution.runner_log),
        )
    try:
        runtime_provenance = json.loads(
            read_verified_content(root, execution.runtime_provenance).decode("utf-8")
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise StepAuthorityRuntimeError("sealed runtime provenance is invalid") from exc
    if not isinstance(runtime_provenance, dict):
        raise StepAuthorityRuntimeError("sealed runtime provenance is not an object")
    return RunResult(
        step_id=verified.capsule.step_id,
        script_path=script_path,
        cwd=step_dir,
        out_dir=step_dir / "outputs",
        stdout=stdout,
        stderr=stderr,
        returncode=execution.returncode,
        duration_seconds=execution.duration_seconds,
        artefacts=sorted(output_paths),
        timed_out=execution.timed_out,
        requested_network_policy=execution.requested_network_policy,
        effective_isolation=execution.effective_isolation,
        isolation_degraded=execution.isolation_degraded,
        isolation_degradation_reason=execution.isolation_degradation_reason,
        runtime_provenance=runtime_provenance,
        outputs_safe_to_collect=execution.outputs_safe_to_collect,
        runner_log_path=runner_log_path,
        runner_failure_code=(
            RunnerFailureCode(execution.runner_failure_code)
            if execution.runner_failure_code is not None
            else None
        ),
    )


__all__ = [
    "StepAuthorityCoordinates",
    "StepAuthorityRuntimeError",
    "adopt_candidate_for_control_plane_revalidation",
    "adopt_frozen_scoped_coder_context",
    "capsule_matches_coordinates",
    "coordinates_from_verified_capsule",
    "current_execution_runtime_sha256",
    "dependency_blocked_candidate_metadata",
    "execution_context_sha256",
    "initial_generation_code_ref",
    "load_checkpoint_selected_step_capsule",
    "load_explicit_failed_step_capsule",
    "load_explicit_executed_success_step_capsule",
    "load_latest_explicit_failed_step_capsule_from_history",
    "load_explicit_success_step_capsule",
    "select_explicit_step_capsule_for_targeted_resume",
    "materialize_sealed_run_result",
    "persist_candidate_code",
    "prepare_step_authority_coordinates",
    "read_concept_audit_findings",
    "repair_code_ref",
    "seal_concept_audit_capsule",
    "seal_deterministic_candidate",
    "seal_execution_capsule",
    "seal_initial_generation_candidate",
    "seal_legacy_candidate",
    "seal_repair_candidate_from_receipt",
]
