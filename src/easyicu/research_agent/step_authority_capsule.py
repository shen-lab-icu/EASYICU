"""Immutable, content-addressed authority for one research-agent step.

This module owns bytes, not workflow status. A capsule can recover the exact
candidate code, audit binding, or completed runner result after a crash, but it
does not make that capsule current, mark a step successful, register evidence,
or publish semantic aliases. The newest monotonic run checkpoint remains the
only selector of current step authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .code_patch import looks_like_executable_python
from .schema import ValidationFinding

STEP_AUTHORITY_CAPSULE_SCHEMA_VERSION = "easyicu.step_authority_capsule/1"
STEP_AUTHORITY_CAPSULE_REF_SCHEMA_VERSION = "easyicu.step_authority_capsule_ref/1"
_STORE_DIRECTORY = ".step_authority"
_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_SAFE_STEP_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_NON_SEALABLE_AUDIT_ISSUE_CODES = frozenset(
    {
        "llm_concept_audit_provider_failure",
        "llm_concept_audit_response_invalid",
    }
)

Sha256 = Annotated[str, Field(pattern=_DIGEST_PATTERN)]
MediaType = Literal[
    "application/json",
    "application/octet-stream",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/x-parquet",
    "image/jpeg",
    "image/png",
    "image/svg+xml",
    "text/csv",
    "text/html",
    "text/markdown",
    "text/plain",
    "text/x-python",
]


class StepAuthorityCapsuleError(RuntimeError):
    """Capsule bytes or storage cannot be trusted."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ContentRef(_StrictFrozenModel):
    """Digest-bound reference to one immutable byte string."""

    sha256: Sha256
    size_bytes: int = Field(ge=0)
    media_type: MediaType


class StepAuthorityCapsuleRef(_StrictFrozenModel):
    """Checkpoint-safe reference to one immutable step capsule."""

    schema_version: Literal[STEP_AUTHORITY_CAPSULE_REF_SCHEMA_VERSION] = (
        STEP_AUTHORITY_CAPSULE_REF_SCHEMA_VERSION
    )
    step_id: str
    capsule_sha256: Sha256

    @field_validator("step_id")
    @classmethod
    def _safe_step_id(cls, value: str) -> str:
        return _validated_step_id(value)


class CandidateOrigin(_StrictFrozenModel):
    """Exact provider or host transaction that produced candidate bytes."""

    kind: Literal[
        "initial_generation",
        "repair_patch",
        "repair_full_rewrite",
        "deterministic_mutation",
        "legacy_adoption",
    ]
    authority_binding_sha256: Sha256
    provider_category: Optional[str] = None
    provider_transport_id: Optional[str] = None
    logical_repair_attempt_id: Optional[int] = Field(default=None, ge=1)
    repair_ticket_sha256: Optional[Sha256] = None
    deterministic_reason_sha256: Optional[Sha256] = None

    @field_validator("provider_category")
    @classmethod
    def _safe_provider_category(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not _SAFE_CATEGORY_PATTERN.fullmatch(value):
            raise ValueError("provider_category must be a normalized category")
        return value

    @field_validator("provider_transport_id")
    @classmethod
    def _safe_transport_id(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9:._-]{0,127}", value
        ):
            raise ValueError("provider_transport_id is not normalized")
        return value

    @model_validator(mode="after")
    def _coordinates_match_origin(self) -> "CandidateOrigin":
        provider_bound = self.provider_category is not None
        transport_bound = self.provider_transport_id is not None
        repair_bound = (
            self.logical_repair_attempt_id is not None
            and self.repair_ticket_sha256 is not None
        )
        if self.kind == "initial_generation":
            if (
                self.provider_category != "initial_generation"
                or not transport_bound
                or repair_bound
                or self.logical_repair_attempt_id is not None
                or self.repair_ticket_sha256 is not None
                or self.deterministic_reason_sha256 is not None
            ):
                raise ValueError(
                    "initial generation requires its own provider transport"
                )
        elif self.kind in {"repair_patch", "repair_full_rewrite"}:
            if (
                not provider_bound
                or not transport_bound
                or not repair_bound
                or self.deterministic_reason_sha256 is not None
            ):
                raise ValueError("repair origin requires exact ledger coordinates")
        elif self.kind == "deterministic_mutation":
            if (
                provider_bound
                or transport_bound
                or self.logical_repair_attempt_id is not None
                or self.repair_ticket_sha256 is not None
                or self.deterministic_reason_sha256 is None
            ):
                raise ValueError("deterministic mutation cannot claim provider calls")
        elif (
            provider_bound
            or transport_bound
            or self.logical_repair_attempt_id is not None
            or self.repair_ticket_sha256 is not None
            or self.deterministic_reason_sha256 is not None
        ):
            raise ValueError("legacy adoption cannot invent generation authority")
        return self


class ConceptAuditSeal(_StrictFrozenModel):
    """Exact digest-cache identity for the deterministic/LLM concept gate."""

    audit_key: Sha256
    audited_code_sha256: Sha256
    authority_binding_sha256: Sha256
    result: Literal["passed", "blocked"]
    findings: ContentRef
    auditor_identity_sha256: Sha256
    environment_sha256: Sha256
    validator_implementation_sha256: Sha256

    @model_validator(mode="after")
    def _findings_are_json(self) -> "ConceptAuditSeal":
        if self.findings.media_type != "application/json":
            raise ValueError("concept-audit findings must be application/json")
        return self


class ExecutionOutput(_StrictFrozenModel):
    """One sealed runner output under a canonical logical relative path."""

    logical_relative_path: str
    content: ContentRef

    @field_validator("logical_relative_path")
    @classmethod
    def _safe_logical_path(cls, value: str) -> str:
        return _validated_logical_path(value)


class ExecutionSeal(_StrictFrozenModel):
    """Completed sandbox result, including failures that must not be rerun."""

    execution_identity_sha256: Sha256
    code_sha256: Sha256
    resolved_inputs_sha256: Sha256
    returncode: int
    duration_seconds: float = Field(ge=0)
    timed_out: bool
    outputs_safe_to_collect: bool
    requested_network_policy: str = Field(min_length=1, max_length=128)
    effective_isolation: str = Field(min_length=1, max_length=128)
    isolation_degraded: bool
    isolation_degradation_reason: Optional[str] = Field(default=None, max_length=1024)
    runtime_provenance: ContentRef
    stdout: ContentRef
    stderr: ContentRef
    runner_log: Optional[ContentRef] = None
    outputs: tuple[ExecutionOutput, ...] = ()

    @model_validator(mode="after")
    def _outputs_are_canonical(self) -> "ExecutionSeal":
        paths = [item.logical_relative_path for item in self.outputs]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("execution outputs must be unique and path-sorted")
        if not self.outputs_safe_to_collect and self.outputs:
            raise ValueError("unsafe execution results cannot seal output files")
        if self.runtime_provenance.media_type != "application/json":
            raise ValueError("runtime_provenance must be application/json")
        if (
            self.stdout.media_type != "text/plain"
            or self.stderr.media_type != "text/plain"
        ):
            raise ValueError("stdout and stderr must be text/plain")
        if self.runner_log is not None and self.runner_log.media_type != "text/plain":
            raise ValueError("runner_log must be text/plain")
        if self.isolation_degraded and not self.isolation_degradation_reason:
            raise ValueError("degraded isolation requires a reason")
        if not self.isolation_degraded and self.isolation_degradation_reason:
            raise ValueError("non-degraded isolation cannot claim a degradation reason")
        return self


class StepAuthorityCapsule(_StrictFrozenModel):
    """Path-free immutable snapshot of one step authority stage."""

    schema_version: Literal[STEP_AUTHORITY_CAPSULE_SCHEMA_VERSION] = (
        STEP_AUTHORITY_CAPSULE_SCHEMA_VERSION
    )
    step_id: str
    stage: Literal[
        "candidate",
        "executed",
        "concept_audited",
        "executed_concept_audited",
    ]
    parent_capsule_sha256: Optional[Sha256] = None

    run_input_capsule_sha256: Sha256
    planner_scope: ContentRef
    scoped_coder_context: ContentRef
    resolved_inputs: ContentRef
    typed_bindings_sha256: Sha256
    upstream_authority_sha256: Sha256

    candidate_code: ContentRef
    candidate_origin: CandidateOrigin
    deterministic_gate_fingerprint: Sha256
    engine_code_sha256: Sha256
    validator_code_sha256: Sha256
    prompt_pack_version: str = Field(min_length=1, max_length=128)
    prompt_pack_sha256: Sha256

    concept_audit: Optional[ConceptAuditSeal] = None
    execution: Optional[ExecutionSeal] = None

    @field_validator("step_id")
    @classmethod
    def _safe_step_id(cls, value: str) -> str:
        return _validated_step_id(value)

    @model_validator(mode="after")
    def _stage_is_closed(self) -> "StepAuthorityCapsule":
        json_refs = (
            self.planner_scope,
            self.scoped_coder_context,
            self.resolved_inputs,
        )
        if any(ref.media_type != "application/json" for ref in json_refs):
            raise ValueError("planner/context/resolved-input refs must be JSON")
        if self.candidate_code.media_type != "text/x-python":
            raise ValueError("candidate_code must be text/x-python")

        if self.stage == "candidate":
            if self.concept_audit is not None or self.execution is not None:
                raise ValueError("candidate stage cannot claim audit or execution")
            if self.candidate_origin.kind in {
                "repair_patch",
                "repair_full_rewrite",
                "deterministic_mutation",
            }:
                if self.parent_capsule_sha256 is None:
                    raise ValueError("mutated candidates require a parent capsule")
        elif self.stage == "executed":
            if self.parent_capsule_sha256 is None or self.execution is None:
                raise ValueError("executed stage requires parent and execution")
            if self.concept_audit is not None:
                raise ValueError("executed stage cannot claim concept audit")
        elif self.stage == "concept_audited":
            if self.parent_capsule_sha256 is None or self.concept_audit is None:
                raise ValueError("concept_audited stage requires parent and audit")
            if self.execution is not None:
                raise ValueError("concept_audited stage cannot claim execution")
        else:
            if (
                self.parent_capsule_sha256 is None
                or self.concept_audit is None
                or self.execution is None
            ):
                raise ValueError(
                    "executed_concept_audited stage requires parent, audit, and execution"
                )
        if self.execution is not None:
            if self.execution.code_sha256 != self.candidate_code.sha256:
                raise ValueError("execution is not bound to candidate code")
            if self.execution.resolved_inputs_sha256 != self.resolved_inputs.sha256:
                raise ValueError("execution is not bound to resolved inputs")
        return self


@dataclass(frozen=True)
class VerifiedStepAuthorityCapsule:
    """A fully verified capsule plus decoded candidate source."""

    ref: StepAuthorityCapsuleRef
    capsule: StepAuthorityCapsule
    candidate_code: str


def _validated_step_id(value: str) -> str:
    text = str(value or "")
    if not _SAFE_STEP_PATTERN.fullmatch(text) or text in {".", ".."}:
        raise ValueError("step_id must be one safe path component")
    return text


def _validated_logical_path(value: str) -> str:
    text = str(value or "")
    path = PurePosixPath(text)
    if (
        not text
        or "\x00" in text
        or "\\" in text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        raise ValueError("logical output path must be canonical and relative")
    return text


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def concept_audit_authority_sha256(
    capsule: StepAuthorityCapsule,
    *,
    audit_key: str,
    auditor_identity_sha256: str,
    environment_sha256: str,
    validator_implementation_sha256: str,
) -> str:
    """Bind a cache/audit identity to exact candidate and input authority."""

    payload = {
        "schema": "easyicu.step_concept_audit_authority/1",
        "step_id": capsule.step_id,
        "audit_key": audit_key,
        "candidate_code_sha256": capsule.candidate_code.sha256,
        "run_input_capsule_sha256": capsule.run_input_capsule_sha256,
        "planner_scope_sha256": capsule.planner_scope.sha256,
        "scoped_coder_context_sha256": capsule.scoped_coder_context.sha256,
        "resolved_inputs_sha256": capsule.resolved_inputs.sha256,
        "typed_bindings_sha256": capsule.typed_bindings_sha256,
        "upstream_authority_sha256": capsule.upstream_authority_sha256,
        "prompt_pack_sha256": capsule.prompt_pack_sha256,
        "environment_sha256": environment_sha256,
        "auditor_identity_sha256": auditor_identity_sha256,
        "validator_implementation_sha256": validator_implementation_sha256,
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _run_root(run_dir: str | Path) -> Path:
    candidate = Path(run_dir).expanduser()
    try:
        if candidate.is_symlink():
            raise StepAuthorityCapsuleError("run directory must not be a symbolic link")
        resolved = candidate.resolve(strict=True)
        if not stat.S_ISDIR(candidate.stat(follow_symlinks=False).st_mode):
            raise StepAuthorityCapsuleError("run directory is not a directory")
    except (FileNotFoundError, OSError) as exc:
        raise StepAuthorityCapsuleError(
            "run directory is missing or unreadable"
        ) from exc
    return resolved


def _object_parts(*, kind: str, digest: str) -> tuple[str, ...]:
    if kind not in {"blobs", "capsules"}:
        raise StepAuthorityCapsuleError("unknown authority object family")
    return (_STORE_DIRECTORY, kind, "sha256", digest[:2])


def _open_store_directory_fd(
    root: Path,
    *,
    parts: tuple[str, ...],
    create: bool,
) -> int:
    """Traverse from an opened run root using no-follow ``openat`` calls."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        current_fd = os.open(root, flags)
    except OSError as exc:
        raise StepAuthorityCapsuleError("cannot open run directory authority") from exc
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise StepAuthorityCapsuleError(
                        f"cannot create authority store component: {part}"
                    ) from exc
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except OSError as exc:
                state = "missing, unreadable, or a symbolic link"
                raise StepAuthorityCapsuleError(
                    f"authority store component is {state}: {part}"
                ) from exc
            info = os.fstat(next_fd)
            if not stat.S_ISDIR(info.st_mode):
                os.close(next_fd)
                raise StepAuthorityCapsuleError(
                    f"authority store component is not a directory: {part}"
                )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _fallback_store_directory(
    root: Path,
    *,
    parts: tuple[str, ...],
    create: bool,
) -> Path:
    """Non-POSIX fallback; production paths use descriptor-anchored traversal."""

    current = root
    for part in parts:  # pragma: no cover - POSIX production and CI
        current = current / part
        if create:
            current.mkdir(mode=0o700, exist_ok=True)
        if current.is_symlink() or not current.is_dir():
            raise StepAuthorityCapsuleError(
                f"authority store component is missing or a symbolic link: {part}"
            )
    return current


def _read_from_directory_fd(parent_fd: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise StepAuthorityCapsuleError("authority object is not a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            return handle.read()
    except StepAuthorityCapsuleError:
        raise
    except (FileNotFoundError, OSError) as exc:
        raise StepAuthorityCapsuleError(
            f"authority object is missing, unreadable, or a symbolic link: {name}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_immutable(
    root: Path,
    *,
    parts: tuple[str, ...],
    name: str,
    payload: bytes,
) -> None:
    if os.name != "posix":  # pragma: no cover - production and CI use POSIX runners
        parent = _fallback_store_directory(root, parts=parts, create=True)
        target = parent / name
        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.read_bytes() != payload:
                raise StepAuthorityCapsuleError(
                    "existing content-addressed object conflicts with payload"
                )
            return
        descriptor, temporary_name = tempfile.mkstemp(dir=parent, prefix=".capsule.")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary_name, target)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        return

    parent_fd = _open_store_directory_fd(root, parts=parts, create=True)
    temporary_name = f".{name}.{secrets.token_hex(8)}.tmp"
    descriptor: Optional[int] = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_from_directory_fd(parent_fd, name) != payload:
                raise StepAuthorityCapsuleError(
                    "existing content-addressed object conflicts with payload"
                )
        os.fsync(parent_fd)
    except StepAuthorityCapsuleError:
        raise
    except OSError as exc:
        raise StepAuthorityCapsuleError(
            "cannot publish immutable authority object"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except OSError:
            pass
        os.close(parent_fd)


def _read_object(
    root: Path,
    *,
    kind: str,
    digest: str,
    size_bytes: Optional[int] = None,
) -> bytes:
    parts = _object_parts(kind=kind, digest=digest)
    name = digest if kind == "blobs" else f"{digest}.json"
    if os.name == "posix":
        parent_fd = _open_store_directory_fd(root, parts=parts, create=False)
    else:  # pragma: no cover - production and CI use POSIX runners
        parent = _fallback_store_directory(root, parts=parts, create=False)
        payload = (parent / name).read_bytes()
        parent_fd = None
    try:
        if parent_fd is not None:
            payload = _read_from_directory_fd(parent_fd, name)
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
    if _sha256_bytes(payload) != digest:
        raise StepAuthorityCapsuleError(
            "authority object digest does not match its reference"
        )
    if size_bytes is not None and len(payload) != size_bytes:
        raise StepAuthorityCapsuleError(
            "authority object size does not match its reference"
        )
    return payload


def put_content_blob(
    run_dir: str | Path,
    *,
    payload: bytes,
    media_type: MediaType,
) -> ContentRef:
    """Atomically publish bytes without selecting them as current authority."""

    if not isinstance(payload, bytes):
        raise TypeError("content payload must be bytes")
    root = _run_root(run_dir)
    digest = _sha256_bytes(payload)
    ref = ContentRef(
        sha256=digest,
        size_bytes=len(payload),
        media_type=media_type,
    )
    _publish_immutable(
        root,
        parts=_object_parts(kind="blobs", digest=digest),
        name=digest,
        payload=payload,
    )
    return ref


def read_verified_content(run_dir: str | Path, ref: ContentRef) -> bytes:
    """Read one referenced blob and verify size plus digest before returning it."""

    try:
        verified_ref = ContentRef.model_validate(ref.model_dump(mode="json"))
    except (AttributeError, ValidationError) as exc:
        raise StepAuthorityCapsuleError("invalid content reference") from exc
    root = _run_root(run_dir)
    return _read_object(
        root,
        kind="blobs",
        digest=verified_ref.sha256,
        size_bytes=verified_ref.size_bytes,
    )


def _load_capsule_model(root: Path, digest: str) -> StepAuthorityCapsule:
    payload = _read_object(root, kind="capsules", digest=digest)
    try:
        capsule = StepAuthorityCapsule.model_validate_json(payload)
    except (ValidationError, ValueError) as exc:
        raise StepAuthorityCapsuleError("authority capsule schema is invalid") from exc
    return capsule


def _authority_identity(capsule: StepAuthorityCapsule) -> tuple[object, ...]:
    return (
        capsule.step_id,
        capsule.run_input_capsule_sha256,
        capsule.planner_scope,
        capsule.scoped_coder_context,
        capsule.resolved_inputs,
        capsule.typed_bindings_sha256,
        capsule.upstream_authority_sha256,
        capsule.deterministic_gate_fingerprint,
        capsule.engine_code_sha256,
        capsule.validator_code_sha256,
        capsule.prompt_pack_version,
        capsule.prompt_pack_sha256,
    )


def _verify_parent_transition(
    capsule: StepAuthorityCapsule,
    *,
    parent: StepAuthorityCapsule,
) -> None:
    if _authority_identity(parent) != _authority_identity(capsule):
        raise StepAuthorityCapsuleError("parent capsule authority binding disagrees")
    if capsule.stage in {"concept_audited", "executed"}:
        if (
            parent.stage != "candidate"
            or parent.candidate_code != capsule.candidate_code
            or parent.candidate_origin != capsule.candidate_origin
        ):
            raise StepAuthorityCapsuleError("stage is not bound to its candidate")
    elif capsule.stage == "executed_concept_audited":
        if (
            parent.stage not in {"concept_audited", "executed"}
            or parent.candidate_code != capsule.candidate_code
            or parent.candidate_origin != capsule.candidate_origin
        ):
            raise StepAuthorityCapsuleError(
                "execution is not bound to audited candidate"
            )
        if (
            parent.stage == "concept_audited"
            and parent.concept_audit != capsule.concept_audit
        ) or (parent.stage == "executed" and parent.execution != capsule.execution):
            raise StepAuthorityCapsuleError("combined stage rewrites its parent seal")


def _verify_audit_findings(root: Path, audit: ConceptAuditSeal) -> None:
    payload = _read_object(
        root,
        kind="blobs",
        digest=audit.findings.sha256,
        size_bytes=audit.findings.size_bytes,
    )
    try:
        raw = json.loads(payload)
        if not isinstance(raw, list):
            raise TypeError("findings payload is not a list")
        findings = [ValidationFinding.model_validate(item) for item in raw]
    except (UnicodeDecodeError, ValueError, TypeError, ValidationError) as exc:
        raise StepAuthorityCapsuleError(
            "concept-audit findings do not match the strict cache schema"
        ) from exc
    if any(
        str((finding.detail or {}).get("issue_code") or "")
        in _NON_SEALABLE_AUDIT_ISSUE_CODES
        for finding in findings
    ):
        raise StepAuthorityCapsuleError(
            "provider or schema failures cannot become concept-audit authority"
        )
    derived = (
        "blocked" if any(item.severity == "error" for item in findings) else "passed"
    )
    if audit.result != derived:
        raise StepAuthorityCapsuleError(
            "concept-audit result disagrees with sealed finding severities"
        )


def _verify_capsule_contents(
    root: Path,
    capsule: StepAuthorityCapsule,
    *,
    ancestry: Optional[frozenset[str]] = None,
) -> str:
    current_digest = _sha256_bytes(
        _canonical_json_bytes(capsule.model_dump(mode="json"))
    )
    visited = ancestry or frozenset()
    if current_digest in visited:
        raise StepAuthorityCapsuleError("capsule ancestry contains a cycle")
    if len(visited) >= 256:
        raise StepAuthorityCapsuleError("capsule ancestry exceeds verification bound")
    visited = visited | {current_digest}

    json_refs = [
        capsule.planner_scope,
        capsule.scoped_coder_context,
        capsule.resolved_inputs,
    ]
    for ref in json_refs:
        payload = _read_object(
            root,
            kind="blobs",
            digest=ref.sha256,
            size_bytes=ref.size_bytes,
        )
        try:
            json.loads(payload)
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise StepAuthorityCapsuleError("referenced JSON blob is invalid") from exc

    code_bytes = _read_object(
        root,
        kind="blobs",
        digest=capsule.candidate_code.sha256,
        size_bytes=capsule.candidate_code.size_bytes,
    )
    try:
        code = code_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StepAuthorityCapsuleError("candidate code is not UTF-8") from exc
    if not looks_like_executable_python(code):
        raise StepAuthorityCapsuleError("candidate code is not executable Python")

    if capsule.concept_audit is not None:
        audit = capsule.concept_audit
        if audit.audited_code_sha256 != capsule.candidate_code.sha256:
            raise StepAuthorityCapsuleError(
                "concept audit is bound to different candidate code"
            )
        expected_binding = concept_audit_authority_sha256(
            capsule,
            audit_key=audit.audit_key,
            auditor_identity_sha256=audit.auditor_identity_sha256,
            environment_sha256=audit.environment_sha256,
            validator_implementation_sha256=audit.validator_implementation_sha256,
        )
        if audit.authority_binding_sha256 != expected_binding:
            raise StepAuthorityCapsuleError(
                "concept audit authority binding disagrees with candidate inputs"
            )
        _verify_audit_findings(root, capsule.concept_audit)

    if capsule.execution is not None:
        execution = capsule.execution
        runtime_payload = _read_object(
            root,
            kind="blobs",
            digest=execution.runtime_provenance.sha256,
            size_bytes=execution.runtime_provenance.size_bytes,
        )
        try:
            runtime_provenance = json.loads(runtime_payload)
            if not isinstance(runtime_provenance, dict):
                raise TypeError("runtime provenance is not an object")
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise StepAuthorityCapsuleError(
                "runtime provenance JSON is invalid"
            ) from exc
        text_refs = [execution.stdout, execution.stderr]
        if execution.runner_log is not None:
            text_refs.append(execution.runner_log)
        for ref in text_refs:
            payload = _read_object(
                root,
                kind="blobs",
                digest=ref.sha256,
                size_bytes=ref.size_bytes,
            )
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise StepAuthorityCapsuleError(
                    "execution text stream is not UTF-8"
                ) from exc
        for output in execution.outputs:
            _read_object(
                root,
                kind="blobs",
                digest=output.content.sha256,
                size_bytes=output.content.size_bytes,
            )
    if capsule.parent_capsule_sha256 is not None:
        parent = _load_capsule_model(root, capsule.parent_capsule_sha256)
        _verify_capsule_contents(root, parent, ancestry=visited)
        _verify_parent_transition(capsule, parent=parent)
    return code


def seal_step_authority_capsule(
    run_dir: str | Path,
    capsule: StepAuthorityCapsule,
) -> StepAuthorityCapsuleRef:
    """Verify referenced bytes, then atomically seal canonical capsule JSON."""

    try:
        untrusted_payload = _canonical_json_bytes(capsule.model_dump(mode="json"))
        verified_capsule = StepAuthorityCapsule.model_validate_json(untrusted_payload)
    except (AttributeError, ValidationError) as exc:
        raise StepAuthorityCapsuleError("invalid step authority capsule") from exc
    root = _run_root(run_dir)
    _verify_capsule_contents(root, verified_capsule)
    payload = _canonical_json_bytes(verified_capsule.model_dump(mode="json"))
    digest = _sha256_bytes(payload)
    _publish_immutable(
        root,
        parts=_object_parts(kind="capsules", digest=digest),
        name=f"{digest}.json",
        payload=payload,
    )
    return StepAuthorityCapsuleRef(
        step_id=verified_capsule.step_id,
        capsule_sha256=digest,
    )


def load_verified_step_authority_capsule(
    run_dir: str | Path,
    *,
    ref: StepAuthorityCapsuleRef,
    expected_step_id: Optional[str] = None,
) -> VerifiedStepAuthorityCapsule:
    """Load one explicitly selected capsule; never scan for a newer candidate."""

    try:
        verified_ref = StepAuthorityCapsuleRef.model_validate(
            ref.model_dump(mode="json")
        )
    except (AttributeError, ValidationError) as exc:
        raise StepAuthorityCapsuleError(
            "invalid step authority capsule reference"
        ) from exc
    if expected_step_id is not None:
        expected = _validated_step_id(expected_step_id)
        if verified_ref.step_id != expected:
            raise StepAuthorityCapsuleError("capsule reference belongs to another step")

    root = _run_root(run_dir)
    capsule = _load_capsule_model(root, verified_ref.capsule_sha256)
    if capsule.step_id != verified_ref.step_id:
        raise StepAuthorityCapsuleError("capsule payload belongs to another step")
    code = _verify_capsule_contents(root, capsule)
    return VerifiedStepAuthorityCapsule(
        ref=verified_ref,
        capsule=capsule,
        candidate_code=code,
    )


__all__ = [
    "CandidateOrigin",
    "ConceptAuditSeal",
    "ContentRef",
    "ExecutionOutput",
    "ExecutionSeal",
    "STEP_AUTHORITY_CAPSULE_REF_SCHEMA_VERSION",
    "STEP_AUTHORITY_CAPSULE_SCHEMA_VERSION",
    "StepAuthorityCapsule",
    "StepAuthorityCapsuleError",
    "StepAuthorityCapsuleRef",
    "VerifiedStepAuthorityCapsule",
    "concept_audit_authority_sha256",
    "load_verified_step_authority_capsule",
    "put_content_blob",
    "read_verified_content",
    "seal_step_authority_capsule",
]
