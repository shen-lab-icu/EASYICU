"""Fail-closed pre-run authorization gate for a real Canonical9 ``aware`` run.

This gate is enforced INSIDE the real launcher (``tools/run_research_agent_bench.py``)
before any pipeline, subprocess, Provider, or data load.  It never launches a run,
calls a Provider, runs Docker, reads patient data, or grants paper authority.

The binding is an **operator freeze declaration** verified against the **actual
parsed invocation** — not against the declaration's own restated fields.  Every
identity is PINNED by the operator ahead of the run, and the gate proves that BOTH
(a) the on-disk artifacts still hash to those pins AND (b) the real command line
the launcher is about to execute matches those pins knob-for-knob.  No runtime
value is ever written back into the receipt and then trusted as a binding.

Reused, never re-implemented:

* ``evaluator/input_freeze_v1.py`` — the strict typed loader + canonical digest.
  The tracked ``canonical_input_freeze_v1`` is an *assessment authority*: its
  schema forces exactly ``e2/e3/h2`` and every case ``state == "blocked"``, so it
  can NEVER be forged into a runnable full-9 authority (``cases=[]``, a missing /
  duplicate / out-of-order case, or an unknown field are rejected by the loader).
* ``ExpectedExecutionIdentity`` (engine) — clean-tree commit, docker image digest,
  non-secret provider authorization, prompt pack, network policy, ``paper_eligible``.
* ``FIGURE2_TASK_IDS`` (evaluator) — exact ordered 9-task coverage.
* ``_benchmark_input_authority_sha256`` (launcher file branch) — the launcher
  delegates to :func:`production_cohort_input_sha256` here so the frozen per-task
  input digest and the runtime input digest are computed by ONE algorithm.

The authorized path additionally requires a real **typed production input
authority** covering the exact nine tasks in fixed order with per-task input and
provenance digests.  Its ``authority_digest`` (the nine-task mapping-table summary)
must equal both the operator pin and ``ExecutionIdentity.input_authority_sha256``;
and each task's *real* cohort file must hash to that task's frozen ``input_sha256``.

Before any of those cohort bytes are read, the gate also requires an ordered,
digest-bound scientific-protocol authority for E2, H2, and H3.  Each selected
KnowHow card must carry a real clinical-and-methods review attestation bound to
its exact reviewable content.  This module verifies those approvals but never
creates them or promotes the current ``curated_mvp`` cards.

The one operation this gate never performs is launching the real run.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.authority.execution_identity import (
    ExpectedExecutionIdentity,
)

from .evaluator.input_freeze_v1 import (
    CanonicalInputFreezeError,
    canonical_input_freeze_manifest_sha256,
    load_canonical_input_freeze_manifest,
)
from .evaluator.rubric_v1 import FIGURE2_TASK_IDS

REALRUN_AUTHORIZATION_SCHEMA = "easyicu.figure2_realrun_authorization/2"
OPERATOR_FREEZE_DECLARATION_SCHEMA = "easyicu.figure2_operator_freeze_declaration/2"
PRODUCTION_INPUT_AUTHORITY_SCHEMA = "easyicu.figure2_production_input_authority/1"
AUTHORIZED_ARMS: tuple[str, ...] = ("aware",)
_MAX_DOC_BYTES = 8 * 1024 * 1024
_COHORT_READ_BLOCK = 1024 * 1024

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Sha1Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
# A docker image digest as stored on the frozen identity — bare 64-hex or the
# registry ``sha256:`` form; the gate compares it verbatim to the identity's value.
ImageDigest = Annotated[str, Field(pattern=r"^(sha256:)?[0-9a-f]{64}$")]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_regular_file(path: Path) -> bytes:
    fd: int | None = None
    try:
        fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        meta = os.fstat(fd)
        if not stat.S_ISREG(meta.st_mode):
            raise ValueError("authorization input must be a regular file")
        if meta.st_size > _MAX_DOC_BYTES:
            raise ValueError("authorization input exceeds the size limit")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if fd is not None:
            os.close(fd)


def _require_safe_path(path: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() or resolved.is_symlink():
        raise ValueError("path must be an absolute, non-symlink path")
    return resolved.resolve(strict=True)


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(_read_regular_file(_require_safe_path(path))).hexdigest()


# ---------------------------------------------------------------------------
# Shared per-task cohort input digest (ONE algorithm for freeze + runtime)
# ---------------------------------------------------------------------------


def _launcher_input_canonical_bytes(payload: object) -> bytes:
    """The exact canonical encoding used by the launcher input-authority hash."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def production_cohort_input_sha256(path: Path | str) -> str:
    """Per-task cohort FILE input digest.

    This is byte-for-byte the launcher's ``_benchmark_input_authority_sha256`` file
    branch; the launcher imports and delegates to this function so the frozen
    per-task ``input_sha256`` and the runtime-bound digest are the SAME algorithm.
    """

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("benchmark cohort must be a regular non-symlink file")
    content = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(_COHORT_READ_BLOCK), b""):
            content.update(block)
    payload = {
        "kind": "file",
        "content_sha256": content.hexdigest(),
        "size_bytes": int(candidate.stat().st_size),
    }
    return hashlib.sha256(_launcher_input_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------------------
# Minimal, strict JSONL manifest reader (handoff manifest only, never cohorts)
# ---------------------------------------------------------------------------


def _reject_duplicate_jsonl_keys(pairs: Sequence[tuple[str, object]]) -> dict:
    decoded: dict[str, object] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError(f"duplicate JSON key: {key!r}")
        decoded[key] = value
    return decoded


def resolve_strict_jsonl_path(path: Path | str) -> Path:
    """The ONE strict JSONL resolution shared by the gate and the launcher.

    Absolute, regular, non-symlink, existing.  A relative path, a symlink, a
    directory, or a missing file RAISES — it is never downgraded to "not a
    canonical run".  The launcher runs exactly this resolved path, so the gate and
    the launcher can never diverge on which file was authorized.
    """

    safe = _require_safe_path(path)  # absolute + non-symlink + resolve(strict=True)
    if not safe.is_file():
        raise ValueError("ehrflowbench JSONL must be a regular file")
    return safe


@dataclass
class CanonicalJsonlRow:
    task_id: str
    cohort_path: str
    cohort_authority_path: str = ""
    cohort_authority_ref: Optional[Mapping[str, Any]] = None
    trajectory_path: str = ""
    trajectory_authority_path: str = ""
    trajectory_authority_ref: Optional[Mapping[str, Any]] = None


def read_canonical_jsonl_rows(path: Path | str) -> tuple[CanonicalJsonlRow, ...]:
    """Strictly read the handoff rows (keys, cohort paths, typed authority refs).

    Only the handoff manifest is read, never the cohort payloads.  Rows are decoded
    strictly (duplicate JSON keys rejected) so an ambiguous manifest cannot silently
    authorize.  The JSONL path itself must satisfy :func:`resolve_strict_jsonl_path`.
    """

    safe = resolve_strict_jsonl_path(path)
    raw = _read_regular_file(safe)
    rows: list[CanonicalJsonlRow] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line, object_pairs_hook=_reject_duplicate_jsonl_keys)
        if not isinstance(obj, dict):
            raise ValueError("benchmark JSONL row must be an object")
        key = obj.get("key")
        if key is None:
            key = obj.get("id")
        cohort = obj.get("cohort_path")
        if cohort is None:
            cohort = obj.get("cohort")
        cohort_ref = obj.get("cohort_authority_ref")
        cohort_authority_path = obj.get("cohort_authority_path")
        trajectory_path = obj.get("trajectory_path")
        trajectory_authority_path = obj.get("trajectory_authority_path")
        traj_ref = obj.get("trajectory_authority_ref")
        rows.append(
            CanonicalJsonlRow(
                task_id=str(key) if key is not None else "",
                cohort_path=str(cohort) if cohort is not None else "",
                cohort_authority_path=(
                    str(cohort_authority_path)
                    if cohort_authority_path is not None
                    else ""
                ),
                cohort_authority_ref=(
                    dict(cohort_ref) if isinstance(cohort_ref, Mapping) else None
                ),
                trajectory_path=(
                    str(trajectory_path) if trajectory_path is not None else ""
                ),
                trajectory_authority_path=(
                    str(trajectory_authority_path)
                    if trajectory_authority_path is not None
                    else ""
                ),
                trajectory_authority_ref=(
                    dict(traj_ref) if isinstance(traj_ref, Mapping) else None
                ),
            )
        )
    return tuple(rows)


def read_canonical_jsonl_invocation(
    path: Path | str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(task_ids, cohort_paths)`` from an EHRFlow handoff JSONL."""

    rows = read_canonical_jsonl_rows(path)
    return (
        tuple(row.task_id for row in rows),
        tuple(row.cohort_path for row in rows),
    )


def production_provenance_sha256(
    cohort_authority: Optional[Mapping[str, Any]],
    trajectory_authority: Optional[Mapping[str, Any]],
) -> str:
    """Canonical per-task provenance digest over the typed sidecar authorities."""

    body = {
        "cohort_authority": dict(cohort_authority) if cohort_authority else None,
        "trajectory_authority": (
            dict(trajectory_authority) if trajectory_authority else None
        ),
    }
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def jsonl_references_canonical9(task_ids: Sequence[str]) -> bool:
    """True if the JSONL declares ANY frozen Canonical9 task id.

    Used for *mandatory activation*: a run that references even one canonical task
    cannot bypass the gate by omitting the declaration.  Full authorization still
    requires the exact ordered nine.
    """

    canonical = set(FIGURE2_TASK_IDS)
    return any(task_id in canonical for task_id in task_ids)


# ---------------------------------------------------------------------------
# Typed production input authority (the future authorized full-9 input)
# ---------------------------------------------------------------------------


class ProductionInputTask(_StrictFrozenModel):
    task_id: str
    input_sha256: Sha256  # == production_cohort_input_sha256(cohort file)
    provenance_sha256: Sha256


class ProductionInputAuthority(_StrictFrozenModel):
    """A real, exact-nine, byte- and provenance-digested production input authority."""

    schema_version: Literal["easyicu.figure2_production_input_authority/1"]
    submission_profile_ref: str
    tasks: tuple[ProductionInputTask, ...]
    authority_digest: Sha256

    @model_validator(mode="after")
    def _exact_nine_and_self_consistent(self) -> "ProductionInputAuthority":
        if tuple(task.task_id for task in self.tasks) != tuple(FIGURE2_TASK_IDS):
            raise ValueError(
                "production input authority must cover the exact ordered nine tasks"
            )
        body = self.model_dump(mode="json", exclude={"authority_digest"})
        if (
            hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
            != self.authority_digest
        ):
            raise ValueError("production input authority digest mismatch")
        return self

    def frozen_input_by_task(self) -> dict[str, str]:
        """The per-task frozen input digest map (task_id -> input_sha256)."""

        return {task.task_id: task.input_sha256 for task in self.tasks}

    @classmethod
    def build(
        cls, *, submission_profile_ref: str, tasks: Sequence[ProductionInputTask]
    ) -> "ProductionInputAuthority":
        body = {
            "schema_version": PRODUCTION_INPUT_AUTHORITY_SCHEMA,
            "submission_profile_ref": submission_profile_ref,
            "tasks": [task.model_dump(mode="json") for task in tasks],
        }
        digest = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
        return cls(
            schema_version=PRODUCTION_INPUT_AUTHORITY_SCHEMA,
            submission_profile_ref=submission_profile_ref,
            tasks=tuple(tasks),
            authority_digest=digest,
        )


def load_production_input_authority(
    path: Path,
) -> tuple[ProductionInputAuthority, str]:
    """Strict-load a production input authority, rejecting a v1 assessment manifest."""

    safe = _require_safe_path(path)
    raw = _read_regular_file(safe)
    # A v1 assessment manifest (blocked e2/e3/h2) must never masquerade as a full-9
    # production authority; detect it explicitly with the strict v1 loader.
    try:
        load_canonical_input_freeze_manifest(safe)
    except CanonicalInputFreezeError:
        pass
    else:
        raise ValueError(
            "canonical_input_freeze assessment manifest is not a full-9 "
            "production input authority"
        )
    authority = ProductionInputAuthority.model_validate_json(raw, strict=True)
    return authority, hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Canonical execution config — every run-semantics knob folded into ONE digest
# ---------------------------------------------------------------------------

CANONICAL_EXECUTION_CONFIG_SCHEMA = "easyicu.figure2_canonical_execution_config/2"


class CanonicalExecutionConfig(_StrictFrozenModel):
    """Pure parse/normalization of every knob that changes canonical run semantics.

    No Provider, runner, or data load.  The declaration pins
    ``execution_config_sha256``; the gate rebuilds this object from the real argv and
    compares digests, so a new flag can never be silently omitted from the binding.
    A canonical run must have ``stop_after_step_id is None``.
    """

    schema_version: Literal["easyicu.figure2_canonical_execution_config/2"]
    stop_after_step_id: Optional[str]
    seed: int
    llm_seed: Optional[int]
    disable_replanning: bool
    max_total_steps: Optional[int]
    max_code_repair_attempts: Optional[int]
    max_step_llm_repair_attempts: Optional[int]
    max_step_provider_calls: int
    max_provider_attempts_per_run: int
    max_provider_attempts_per_batch: int
    max_total_tokens_per_run: int
    max_total_tokens_per_batch: int
    max_estimated_cost_usd_per_batch: float
    max_wall_clock_seconds_per_task: float
    provider_input_cost_usd_per_million_tokens: float
    provider_output_cost_usd_per_million_tokens: float
    timeout_seconds: float
    standard_executor_timeout_seconds: float
    request_timeout_seconds: float
    transport_max_attempts: int
    provider_base_url: str
    llm_stream_enabled: bool
    enable_repro_envelope: bool
    enable_cost_tracking: bool
    strict_evidence: bool
    writer_digest_widened: bool
    enable_pubmed: bool
    case: Optional[str]
    development_sample_size: Optional[int]
    development_sample_seed: int
    models: tuple[str, ...]
    reasoning_effort_profile: Literal["provider_default", "adaptive_v1"]

    def digest(self) -> str:
        return hashlib.sha256(
            _canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()


def build_canonical_execution_config(
    *,
    seed: int,
    timeout_seconds: float,
    standard_executor_timeout_seconds: float,
    request_timeout_seconds: float = 180.0,
    stop_after_step_id: object = None,
    llm_seed: object = None,
    disable_replanning: bool = False,
    max_total_steps: object = None,
    max_code_repair_attempts: object = None,
    max_step_llm_repair_attempts: object = None,
    max_step_provider_calls: int = 9,
    max_provider_attempts_per_run: int = 192,
    max_provider_attempts_per_batch: int = 1_728,
    max_total_tokens_per_run: int = 2_000_000,
    max_total_tokens_per_batch: int = 18_000_000,
    max_estimated_cost_usd_per_batch: float = 100.0,
    max_wall_clock_seconds_per_task: float = 21_600.0,
    provider_input_cost_usd_per_million_tokens: float = 10.0,
    provider_output_cost_usd_per_million_tokens: float = 30.0,
    enable_repro_envelope: bool = True,
    enable_cost_tracking: bool = True,
    strict_evidence: bool = False,
    writer_digest_widened: bool = False,
    enable_pubmed: bool = False,
    case: object = None,
    development_sample_size: object = None,
    development_sample_seed: int = 20260719,
    models: Sequence[str] = (),
    reasoning_effort_profile: str = "provider_default",
    transport_max_attempts: int = 1,
    provider_base_url: str = "https://openrouter.ai/api/v1",
    llm_stream_enabled: bool = False,
) -> CanonicalExecutionConfig:
    """Normalize argv into the canonical config (pure — no Provider/runner/data)."""

    def _opt_int(value: object) -> Optional[int]:
        return int(value) if value is not None else None

    return CanonicalExecutionConfig(
        schema_version=CANONICAL_EXECUTION_CONFIG_SCHEMA,
        stop_after_step_id=(
            str(stop_after_step_id).strip() or None if stop_after_step_id else None
        ),
        seed=int(seed),
        llm_seed=_opt_int(llm_seed),
        disable_replanning=bool(disable_replanning),
        max_total_steps=_opt_int(max_total_steps),
        max_code_repair_attempts=_opt_int(max_code_repair_attempts),
        max_step_llm_repair_attempts=_opt_int(max_step_llm_repair_attempts),
        max_step_provider_calls=int(max_step_provider_calls),
        max_provider_attempts_per_run=int(max_provider_attempts_per_run),
        max_provider_attempts_per_batch=int(max_provider_attempts_per_batch),
        max_total_tokens_per_run=int(max_total_tokens_per_run),
        max_total_tokens_per_batch=int(max_total_tokens_per_batch),
        max_estimated_cost_usd_per_batch=float(
            max_estimated_cost_usd_per_batch
        ),
        max_wall_clock_seconds_per_task=float(max_wall_clock_seconds_per_task),
        provider_input_cost_usd_per_million_tokens=float(
            provider_input_cost_usd_per_million_tokens
        ),
        provider_output_cost_usd_per_million_tokens=float(
            provider_output_cost_usd_per_million_tokens
        ),
        timeout_seconds=float(timeout_seconds),
        standard_executor_timeout_seconds=float(standard_executor_timeout_seconds),
        request_timeout_seconds=float(request_timeout_seconds),
        transport_max_attempts=int(transport_max_attempts),
        provider_base_url=str(provider_base_url).strip(),
        llm_stream_enabled=bool(llm_stream_enabled),
        enable_repro_envelope=bool(enable_repro_envelope),
        enable_cost_tracking=bool(enable_cost_tracking),
        strict_evidence=bool(strict_evidence),
        writer_digest_widened=bool(writer_digest_widened),
        enable_pubmed=bool(enable_pubmed),
        case=(str(case).strip() or None if case else None),
        development_sample_size=_opt_int(development_sample_size),
        development_sample_seed=int(development_sample_seed),
        models=tuple(str(model) for model in (models or ())),
        reasoning_effort_profile=str(reasoning_effort_profile),
    )


# ---------------------------------------------------------------------------
# Operator freeze declaration (all pins REQUIRED)
# ---------------------------------------------------------------------------


class OperatorFreezeDeclaration(_StrictFrozenModel):
    """Operator-frozen pins fixed BEFORE the run; the gate only verifies them."""

    schema_version: Literal["easyicu.figure2_operator_freeze_declaration/2"]
    expected_execution_identity_sha256: Sha256
    input_authority_digest: Sha256
    scientific_protocol_authority_digest: Sha256
    input_freeze_manifest_sha256: Sha256
    rubric_sha256: Sha256
    execution_config_sha256: Sha256
    code_commit_sha: Sha1Hex
    runner: Literal["docker"]
    network_policy: Literal["none", "disabled"]
    runner_image_digest: ImageDigest
    provider: str
    model: str
    submission_profile_ref: str
    ehrflowbench_jsonl_path: str
    ehrflowbench_jsonl_sha256: Sha256
    task_ids: tuple[str, ...]
    arms: tuple[str, ...]
    cross_run_memory: Literal[False]
    output_root: str
    batch_id: str

    @model_validator(mode="after")
    def _declared_intent_is_canonical(self) -> "OperatorFreezeDeclaration":
        if self.task_ids != tuple(FIGURE2_TASK_IDS):
            raise ValueError("declaration must pin the exact ordered nine task ids")
        if self.arms != AUTHORIZED_ARMS:
            raise ValueError("declaration must pin exactly the aware arm")
        if self.provider.strip().lower() in {"", "mock"}:
            raise ValueError("declaration provider must be a real (non-mock) provider")
        if not self.model.strip():
            raise ValueError("declaration must pin a concrete model")
        if not self.submission_profile_ref.strip():
            raise ValueError("declaration must pin a submission profile ref")
        if not Path(self.output_root).is_absolute():
            raise ValueError("declaration output_root must be absolute")
        if not Path(self.ehrflowbench_jsonl_path).is_absolute():
            raise ValueError("declaration ehrflowbench_jsonl_path must be absolute")
        if not self.batch_id.startswith("batch_"):
            raise ValueError("declaration batch_id must start with 'batch_'")
        if Path(self.output_root).name != self.batch_id:
            raise ValueError("declaration output_root must be the batch_id directory")
        return self


# ---------------------------------------------------------------------------
# The concrete parsed invocation the launcher is about to execute
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealRunInvocation:
    """The real, already-parsed launch intent handed to the gate.

    Built from the actual argv (and the handoff JSONL keys/cohort paths), NOT from
    the declaration.  Every field here is compared against a declared pin so an
    authorized declaration cannot be paired with a divergent command line.
    """

    arms: tuple[str, ...]
    task_ids: tuple[str, ...]
    task_cohort_paths: tuple[tuple[str, str], ...]
    ehrflowbench_jsonl_path: Optional[Path]
    provider: str
    model: str
    submission_profile_enabled: bool
    submission_profile_ref: Optional[str]
    runner: str
    out_root: Path
    require_paper_acceptance: bool
    execution_config: CanonicalExecutionConfig
    reuse_existing: bool = False
    repeat: int = 1
    force_writer_probe: bool = False
    development_sample_size: Optional[int] = None
    allow_host_runner: bool = False
    allow_mock_aware: bool = False
    resume_run_id: Optional[str] = None
    resume_from_step_id: Optional[str] = None
    cross_run_memory: bool = False

    def cohort_by_task(self) -> dict[str, str]:
        return {task_id: cohort for task_id, cohort in self.task_cohort_paths}


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


class RealRunAuthorizationIssue(_StrictFrozenModel):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,127}$")
    detail: str = Field(min_length=1, max_length=2048)


class RealRunAuthorization(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_realrun_authorization/2"]
    status: Literal["authorized", "blocked"]
    expected_task_ids: tuple[str, ...]
    declaration_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_authority_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    scientific_protocol_authority_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    batch_id: str | None = None
    frozen_input_by_task: dict[str, Sha256] = Field(default_factory=dict)
    issues: tuple[RealRunAuthorizationIssue, ...] = ()

    @model_validator(mode="after")
    def _validate_status(self) -> "RealRunAuthorization":
        if self.expected_task_ids != tuple(FIGURE2_TASK_IDS):
            raise ValueError("authorization must retain exact Canonical9 order")
        if self.status == "authorized":
            if self.issues:
                raise ValueError("authorized receipt cannot carry issues")
            if (
                self.declaration_sha256 is None
                or self.input_authority_digest is None
                or self.scientific_protocol_authority_digest is None
            ):
                raise ValueError(
                    "authorized receipt must bind declaration, input authority, "
                    "and scientific protocol authority"
                )
            if self.batch_id is None or not self.batch_id.startswith("batch_"):
                raise ValueError("authorized receipt must bind a batch id")
            if tuple(self.frozen_input_by_task) != tuple(FIGURE2_TASK_IDS):
                raise ValueError("authorized receipt must bind every frozen task input")
        elif not self.issues:
            raise ValueError("blocked receipt must carry at least one issue")
        return self


@dataclass(frozen=True)
class RealRunAuthorizationRequest:
    """The declaration file + the actual artifacts + the concrete run intent."""

    declaration_path: Path
    expected_execution_identity_path: Path
    input_freeze_path: Path
    rubric_path: Path
    invocation: RealRunInvocation
    production_input_authority_path: Optional[Path] = None
    scientific_protocol_authority_path: Optional[Path] = None
    # The LIVE checkout state (git_sha / git_dirty). ``None`` means "measure it now"
    # via ``capture_code_version()`` — the real launcher leaves it None; tests inject
    # a value so the running-tree binding is exercised without a real git tree.
    live_code_version: Optional[Mapping[str, Any]] = None


class RealRunAuthorizationBlocked(RuntimeError):
    def __init__(self, authorization: "RealRunAuthorization") -> None:
        codes = ", ".join(issue.code for issue in authorization.issues)
        super().__init__(f"real-run authorization blocked: {codes}")
        self.authorization = authorization


def _issue(code: str, detail: str) -> RealRunAuthorizationIssue:
    return RealRunAuthorizationIssue(code=code, detail=detail)


def _load_declaration(
    path: Path,
) -> tuple[Optional[OperatorFreezeDeclaration], Optional[str], list]:
    issues: list[RealRunAuthorizationIssue] = []
    try:
        safe = _require_safe_path(path)
        raw = _read_regular_file(safe)
        declaration = OperatorFreezeDeclaration.model_validate_json(raw, strict=True)
        return declaration, hashlib.sha256(raw).hexdigest(), issues
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "OPERATOR_DECLARATION_INVALID",
                f"cannot load operator freeze declaration: {type(exc).__name__}: {exc}",
            )
        )
        return None, None, issues


def _verify_fresh_output_root(root: Path, batch_id: str) -> list:
    """Require an unallocated batch directory that can be atomically reserved.

    ``out_root`` is the batch root itself, not a shared parent.  Verification is
    intentionally separate from reservation; the latter uses ``mkdir`` without
    ``exist_ok`` immediately after authorization, so concurrent processes cannot
    both obtain the same paper-facing batch identity.
    """

    issues: list[RealRunAuthorizationIssue] = []
    try:
        candidate = Path(root).expanduser()
        if not candidate.is_absolute() or candidate.is_symlink():
            raise ValueError("output root must be an absolute, non-symlink path")
        if candidate.name != batch_id:
            raise ValueError("output root must be named exactly for the declared batch")
        parent = candidate.parent
        parent_info = parent.lstat()
        if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
            raise ValueError("output-root parent must be an existing real directory")
        if candidate.exists():
            raise ValueError("batch root already exists; refusing diagnostic reuse")
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue("OUTPUT_ROOT_NOT_FRESH", str(exc)))
    return issues


def _verify_invocation_binding(
    declaration: OperatorFreezeDeclaration, invocation: RealRunInvocation
) -> list:
    """Every real command-line knob must equal the operator's declared pin."""

    issues: list[RealRunAuthorizationIssue] = []

    def bad(code: str, detail: str) -> None:
        issues.append(_issue(code, detail))

    # Arms — must be exactly the aware arm and equal to the pin.
    if tuple(invocation.arms) != AUTHORIZED_ARMS:
        bad(
            "INVOCATION_ARM_NOT_AWARE",
            f"real arms {list(invocation.arms)} are not exactly {list(AUTHORIZED_ARMS)}",
        )
    if tuple(invocation.arms) != tuple(declaration.arms):
        bad("INVOCATION_ARM_MISMATCH", "real arms differ from the declared pin")

    # Task ids — exact ordered nine, equal to the pin.
    if tuple(invocation.task_ids) != tuple(FIGURE2_TASK_IDS):
        bad(
            "INVOCATION_TASKS_NOT_CANONICAL",
            "real task ids are not the exact ordered Canonical9",
        )
    if tuple(invocation.task_ids) != tuple(declaration.task_ids):
        bad("INVOCATION_TASKS_MISMATCH", "real task ids differ from the declared pin")

    # EHRFlow JSONL — strict resolution (absolute/regular/non-symlink), exact path
    # AND exact bytes.  The launcher runs this same strict path.
    try:
        if invocation.ehrflowbench_jsonl_path is None:
            raise ValueError("no ehrflowbench JSONL supplied for a canonical run")
        actual_jsonl = resolve_strict_jsonl_path(invocation.ehrflowbench_jsonl_path)
        declared_jsonl = Path(declaration.ehrflowbench_jsonl_path).expanduser()
        if not declared_jsonl.is_absolute() or actual_jsonl != declared_jsonl.resolve(
            strict=True
        ):
            raise ValueError("real JSONL path differs from the declared pin")
        if _sha256_of(actual_jsonl) != declaration.ehrflowbench_jsonl_sha256:
            raise ValueError("real JSONL bytes differ from the declared pin")
    except Exception as exc:  # noqa: BLE001
        bad("INVOCATION_JSONL_MISMATCH", f"{type(exc).__name__}: {exc}")

    # Provider / model — real, non-mock, equal to the pin.
    if invocation.provider.strip().lower() == "mock":
        bad("INVOCATION_PROVIDER_IS_MOCK", "a real run cannot use the mock provider")
    if invocation.provider != declaration.provider:
        bad(
            "INVOCATION_PROVIDER_MISMATCH",
            "real provider differs from the declared pin",
        )
    if invocation.model != declaration.model:
        bad("INVOCATION_MODEL_MISMATCH", "real model differs from the declared pin")
    if invocation.allow_mock_aware:
        bad("INVOCATION_MOCK_AWARE", "--allow-mock-aware is never valid for a real run")

    # Submission profile — enabled and equal to the pin.
    if not invocation.submission_profile_enabled:
        bad("INVOCATION_PROFILE_DISABLED", "a real run requires --submission-profile")
    if (invocation.submission_profile_ref or "") != declaration.submission_profile_ref:
        bad(
            "INVOCATION_PROFILE_MISMATCH",
            "real submission profile differs from the declared pin",
        )

    # Runner — docker, equal to the pin, never the host runner.
    if invocation.runner != "docker":
        bad(
            "INVOCATION_RUNNER_NOT_DOCKER",
            f"real runner {invocation.runner!r} is not the docker runner",
        )
    if invocation.runner != declaration.runner:
        bad("INVOCATION_RUNNER_MISMATCH", "real runner differs from the declared pin")
    if invocation.allow_host_runner:
        bad("INVOCATION_HOST_RUNNER", "--allow-host-runner is never paper authority")

    # Output root — resolved exact equality with the pin.
    try:
        real_root = Path(invocation.out_root).expanduser().resolve()
        declared_root = Path(declaration.output_root).expanduser().resolve()
        if real_root != declared_root:
            raise ValueError("real output root differs from the declared pin")
    except Exception as exc:  # noqa: BLE001
        bad("INVOCATION_OUTPUT_ROOT_MISMATCH", f"{type(exc).__name__}: {exc}")

    # Fresh single canonical run — reject every run-semantics-altering flag.
    if invocation.repeat != 1:
        bad("UNSAFE_RUN_FLAG_REPEAT", "--repeat is not a fresh single canonical run")
    if invocation.reuse_existing:
        bad("UNSAFE_RUN_FLAG_REUSE", "--reuse-existing is not a fresh canonical run")
    if invocation.force_writer_probe:
        bad(
            "UNSAFE_RUN_FLAG_FORCE_WRITER",
            "--force-writer-probe is diagnostic only, never archival",
        )
    if invocation.development_sample_size is not None:
        bad(
            "UNSAFE_RUN_FLAG_DEV_SAMPLE",
            "--development-sample-size is not a paper cohort",
        )

    # A real canonical run must be scored by the paper-acceptance gate.
    if not invocation.require_paper_acceptance:
        bad(
            "PAPER_ACCEPTANCE_NOT_REQUIRED",
            "a real canonical run must set --require-figure2-paper-acceptance",
        )

    # Frozen execution-config digest — folds every run-semantics knob into one pin
    # so no flag (stop-after, seed, replan, budgets, timeouts, pubmed, ...) can be
    # silently omitted from the binding.
    config = invocation.execution_config
    if config.stop_after_step_id is not None:
        bad(
            "EXECUTION_CONFIG_INVALID",
            "--stop-after-step-id would run an incomplete but paid canonical run",
        )
    if config.case is not None:
        bad(
            "EXECUTION_CONFIG_INVALID",
            "--case is a mutable named fixture and cannot select a Canonical9 run",
        )
    if len(config.models) > 1:
        bad(
            "EXECUTION_CONFIG_INVALID",
            "a Canonical9 run must bind one concrete model, not a model matrix",
        )
    try:
        from easyicu.research_agent.authority.provider_hard_stop import (
            ProviderHardStopLimits,
        )

        ProviderHardStopLimits(
            max_provider_attempts_per_run=config.max_provider_attempts_per_run,
            max_provider_attempts_per_batch=config.max_provider_attempts_per_batch,
            max_total_tokens_per_run=config.max_total_tokens_per_run,
            max_total_tokens_per_batch=config.max_total_tokens_per_batch,
            max_estimated_cost_usd_per_batch=(
                config.max_estimated_cost_usd_per_batch
            ),
            max_wall_clock_seconds_per_task=config.max_wall_clock_seconds_per_task,
            input_cost_usd_per_million_tokens=(
                config.provider_input_cost_usd_per_million_tokens
            ),
            output_cost_usd_per_million_tokens=(
                config.provider_output_cost_usd_per_million_tokens
            ),
        )
        if config.max_step_provider_calls <= 0:
            raise ValueError("max_step_provider_calls must be positive")
        if config.transport_max_attempts <= 0:
            raise ValueError("transport_max_attempts must be positive")
        if not config.provider_base_url:
            raise ValueError("provider_base_url must be explicitly frozen")
        if not config.enable_cost_tracking:
            raise ValueError("formal runs require cost tracking")
    except (TypeError, ValueError) as exc:
        bad(
            "EXECUTION_CONFIG_INVALID",
            f"Provider hard-stop configuration is invalid: {exc}",
        )
    if config.digest() != declaration.execution_config_sha256:
        bad(
            "EXECUTION_CONFIG_MISMATCH",
            "real execution config digest differs from the declared pin",
        )
    return issues


def _verify_sidecar_digest(
    sidecar: Path, *, expected_sha256: str, expected_size: int
) -> None:
    """A typed materialization-authority sidecar must hash+size to its declared ref."""

    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError(f"sidecar authority is not a regular file: {sidecar.name}")
    data = _read_regular_file(_require_safe_path(sidecar))
    if len(data) != int(expected_size):
        raise ValueError(f"sidecar authority size differs from ref: {sidecar.name}")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"sidecar authority digest differs from ref: {sidecar.name}")


def _verify_task_provenance(task, row: Optional[CanonicalJsonlRow]) -> None:
    """Bind the task's frozen ``provenance_sha256`` to the REAL typed sidecars."""

    from easyicu.research_agent.intake import materialized_metadata
    from easyicu.research_agent.intake import materialized_trajectory

    if row is None or not row.cohort_authority_ref:
        raise ValueError(
            f"task {task.task_id} lacks a declared typed cohort materialization "
            "authority (provenance cannot be verified)"
        )
    cohort_ref = materialized_metadata.MaterializedCohortAuthorityRef.from_dict(
        row.cohort_authority_ref
    )
    cohort_path = _require_safe_path(Path(row.cohort_path))
    cohort_dir = cohort_path.parent
    declared_cohort_authority = _require_safe_path(Path(row.cohort_authority_path))
    expected_cohort_authority = _require_safe_path(cohort_dir / cohort_ref.file)
    if declared_cohort_authority != expected_cohort_authority:
        raise ValueError(
            f"task {task.task_id} cohort_authority_path does not select the "
            "authority declared by cohort_authority_ref"
        )
    _verify_sidecar_digest(
        declared_cohort_authority,
        expected_sha256=cohort_ref.sha256,
        expected_size=cohort_ref.size,
    )
    verified_cohort = materialized_metadata.load_verified_materialized_cohort_authority(
        cohort_path,
        expected_authority=cohort_ref,
    )
    if verified_cohort is None:
        raise ValueError(
            f"task {task.task_id} cohort authority did not verify against its cohort"
        )
    # A structural retrofit can establish typed *columns*, but it cannot itself
    # establish patient-level paper authority.  Preserve its source seal kind in
    # the materializer's immutable semantic provenance, then reject it here rather
    # than allowing a valid-looking sidecar to upgrade a legacy source into a real
    # Canonical9 input.  A future, separately reviewed retrofit path needs an
    # explicit source-attestation contract; this P4 gate must never infer one.
    cohort_provenance = getattr(verified_cohort, "provenance", None)
    export_provenance = (
        cohort_provenance.get("export_authority")
        if isinstance(cohort_provenance, Mapping)
        else None
    )
    if isinstance(export_provenance, Mapping) and export_provenance.get("seal_kind"):
        raise ValueError(
            f"task {task.task_id} source export uses seal_kind "
            f"{export_provenance['seal_kind']!r}; structural retrofit provenance "
            "is not a production paper authority"
        )
    trajectory_norm = None
    has_trajectory = bool(row.trajectory_path)
    has_trajectory_ref = row.trajectory_authority_ref is not None
    has_trajectory_authority_path = bool(row.trajectory_authority_path)
    if (
        has_trajectory != has_trajectory_ref
        or has_trajectory != has_trajectory_authority_path
    ):
        raise ValueError(
            f"task {task.task_id} trajectory path and typed authority must be "
            "declared together"
        )
    if has_trajectory:
        trajectory_ref = (
            materialized_trajectory.MaterializedTrajectoryAuthorityRef.from_dict(
                row.trajectory_authority_ref
            )
        )
        trajectory_path = _require_safe_path(Path(row.trajectory_path))
        declared_trajectory_authority = _require_safe_path(
            Path(row.trajectory_authority_path)
        )
        expected_trajectory_authority = _require_safe_path(
            trajectory_path.parent / trajectory_ref.file
        )
        if declared_trajectory_authority != expected_trajectory_authority:
            raise ValueError(
                f"task {task.task_id} trajectory_authority_path does not select "
                "the authority declared by trajectory_authority_ref"
            )
        _verify_sidecar_digest(
            declared_trajectory_authority,
            expected_sha256=trajectory_ref.sha256,
            expected_size=trajectory_ref.size,
        )
        verified_trajectory = (
            materialized_trajectory.load_verified_materialized_trajectory_authority(
                trajectory_path,
                expected_authority=trajectory_ref,
                expected_universe_authority=cohort_ref,
            )
        )
        if verified_trajectory is None:
            raise ValueError(
                f"task {task.task_id} trajectory authority did not verify against "
                "its trajectory and cohort authority"
            )
        trajectory_norm = trajectory_ref.to_dict()
    provenance = production_provenance_sha256(cohort_ref.to_dict(), trajectory_norm)
    if provenance != task.provenance_sha256:
        raise ValueError(
            f"task {task.task_id} provenance differs from the frozen production "
            "authority (a sidecar/provenance was replaced)"
        )


def _verify_production_input_authority(
    request: RealRunAuthorizationRequest,
    declaration: OperatorFreezeDeclaration,
    identity,
) -> tuple[list, Optional[ProductionInputAuthority]]:
    """The typed production authority + each real cohort/provenance must hold."""

    issues: list[RealRunAuthorizationIssue] = []
    if request.production_input_authority_path is None:
        issues.append(
            _issue(
                "PRODUCTION_INPUT_AUTHORITY_ABSENT",
                "no typed production input authority supplied; the full-9 input is "
                "not yet frozen for a real run (the v1 assessment stays blocked)",
            )
        )
        return issues, None
    try:
        authority, _ = load_production_input_authority(
            request.production_input_authority_path
        )
        if authority.authority_digest != declaration.input_authority_digest:
            raise ValueError("production authority digest differs from the pin")
        if (
            identity is not None
            and identity.input_authority_sha256 != authority.authority_digest
        ):
            raise ValueError(
                "execution identity input authority differs from the production "
                "authority digest"
            )
        if authority.submission_profile_ref != declaration.submission_profile_ref:
            raise ValueError(
                "production authority submission profile differs from the pin"
            )
        # Re-read the strictly-resolved JSONL rows (the path+bytes are already pinned
        # by the invocation binding) so per-task input AND provenance bind to reality.
        rows_by_task = {
            row.task_id: row
            for row in read_canonical_jsonl_rows(
                request.invocation.ehrflowbench_jsonl_path
            )
        }
        cohort_by_task = request.invocation.cohort_by_task()
        for task in authority.tasks:
            cohort_path = cohort_by_task.get(task.task_id)
            if not cohort_path:
                raise ValueError(
                    f"invocation is missing a cohort path for task {task.task_id}"
                )
            # (a) the REAL cohort file must hash to the frozen per-task input digest,
            #     using the launcher's own algorithm.
            actual = production_cohort_input_sha256(cohort_path)
            if actual != task.input_sha256:
                raise ValueError(
                    f"cohort for task {task.task_id} does not hash to its frozen "
                    "production input authority"
                )
            # (b) the typed materialization sidecar(s) must bind to the frozen
            #     per-task provenance digest (a swapped sidecar fails closed here).
            _verify_task_provenance(task, rows_by_task.get(task.task_id))
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue("PRODUCTION_INPUT_AUTHORITY_INVALID", f"{type(exc).__name__}: {exc}")
        )
        authority = None
    return issues, authority


def _verify_scientific_protocol_authority(
    request: RealRunAuthorizationRequest,
    declaration: OperatorFreezeDeclaration,
) -> tuple[list[RealRunAuthorizationIssue], Optional[str]]:
    """Verify E2/H2/H3 review authority before any cohort bytes are read."""

    if request.scientific_protocol_authority_path is None:
        return [
            _issue(
                "SCIENTIFIC_PROTOCOL_AUTHORITY_ABSENT",
                "no digest-bound, dual-reviewed E2/H2/H3 protocol authority was "
                "supplied; formal Canonical9 execution remains blocked",
            )
        ], None
    try:
        from .scientific_protocol_authority import (
            load_verified_scientific_protocol_authority,
        )

        authority, _file_sha256 = load_verified_scientific_protocol_authority(
            request.scientific_protocol_authority_path
        )
        if (
            authority.authority_digest
            != declaration.scientific_protocol_authority_digest
        ):
            raise ValueError(
                "scientific protocol authority digest differs from the operator pin"
            )
        return [], authority.authority_digest
    except Exception as exc:  # noqa: BLE001
        return [
            _issue(
                "SCIENTIFIC_PROTOCOL_AUTHORITY_INVALID",
                f"{type(exc).__name__}: {exc}",
            )
        ], None


def verify_realrun_authorization(
    request: RealRunAuthorizationRequest,
) -> RealRunAuthorization:
    """Fail-closed pre-run authorization; ``authorized`` only if every pin holds."""

    declaration, declaration_sha, issues = _load_declaration(request.declaration_path)
    invocation = request.invocation

    # Runtime intent (independent of the declaration bytes).
    if (
        invocation.resume_run_id is not None
        or invocation.resume_from_step_id is not None
    ):
        issues.append(
            _issue(
                "NON_FRESH_RUN",
                "a canonical run must be fresh; refusing to resume a diagnostic run",
            )
        )
    if invocation.cross_run_memory:
        issues.append(
            _issue(
                "CROSS_RUN_MEMORY_ENABLED",
                "cross-run memory / experience bank must be disabled",
            )
        )

    if declaration is None:
        return RealRunAuthorization(
            schema_version=REALRUN_AUTHORIZATION_SCHEMA,
            status="blocked",
            expected_task_ids=tuple(FIGURE2_TASK_IDS),
            issues=tuple(issues),
        )

    # Scientific decisions are the first post-declaration hard stop.  Missing or
    # tampered E2/H2/H3 review evidence returns immediately so the gate cannot
    # continue into production-cohort digest or sidecar verification.
    protocol_issues, protocol_digest = _verify_scientific_protocol_authority(
        request,
        declaration,
    )
    issues.extend(protocol_issues)
    if protocol_issues:
        return RealRunAuthorization(
            schema_version=REALRUN_AUTHORIZATION_SCHEMA,
            status="blocked",
            expected_task_ids=tuple(FIGURE2_TASK_IDS),
            declaration_sha256=declaration_sha,
            input_authority_digest=declaration.input_authority_digest,
            issues=tuple(issues),
        )

    # Fresh output root (the real out_root, keyed by the declared run id).
    issues.extend(_verify_fresh_output_root(invocation.out_root, declaration.batch_id))

    # Declaration <-> real invocation, knob for knob.
    issues.extend(_verify_invocation_binding(declaration, invocation))

    # Live checkout: the tree that will actually run must be clean AND equal to the
    # declared commit pin — not merely the commit baked into the frozen identity.
    try:
        version = request.live_code_version
        if version is None:
            from easyicu.research_agent.authority.runtime_artifacts import (
                capture_code_version,
            )

            version = capture_code_version() or {}
        live_sha = version.get("git_sha")
        if not live_sha or live_sha != declaration.code_commit_sha:
            raise ValueError("live checkout commit differs from the declared pin")
        if version.get("git_dirty") is not False:
            raise ValueError("live checkout tree is dirty")
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue("LIVE_CHECKOUT_MISMATCH", f"{type(exc).__name__}: {exc}"))

    # 1) Execution identity: pin the file bytes, then require paper eligibility and
    #    equality with every declared engine pin.
    identity = None
    try:
        actual_identity_sha = _sha256_of(request.expected_execution_identity_path)
        if actual_identity_sha != declaration.expected_execution_identity_sha256:
            raise ValueError("execution identity file sha differs from the pin")
        identity = ExpectedExecutionIdentity.model_validate_json(
            _read_regular_file(
                _require_safe_path(request.expected_execution_identity_path)
            ),
            strict=True,
        ).execution_identity
        if not identity.paper_eligible:
            raise ValueError("frozen execution identity is not paper eligible")
        if not identity.git_sha or identity.git_dirty is not False:
            raise ValueError("execution identity is not a clean-tree commit")
        if identity.git_sha != declaration.code_commit_sha:
            raise ValueError("execution identity commit differs from the declared pin")
        if identity.runner != declaration.runner:
            raise ValueError("execution identity runner differs from the declared pin")
        if identity.submission_profile_ref != declaration.submission_profile_ref:
            raise ValueError("execution identity profile ref differs from the pin")
        if identity.network_policy != declaration.network_policy:
            raise ValueError("execution identity network policy differs from the pin")
        if identity.runner_image_digest != declaration.runner_image_digest:
            raise ValueError("execution identity image digest differs from the pin")
        if identity.input_authority_sha256 is None:
            raise ValueError("execution identity lacks a bound input authority sha")
        if identity.input_authority_sha256 != declaration.input_authority_digest:
            raise ValueError(
                "execution identity input_authority_sha256 differs from the pin"
            )
        provider_manifest = identity.provider_authorization
        clients = provider_manifest.get("clients")
        if not isinstance(clients, list) or not clients:
            raise ValueError("execution identity has no Provider client binding")
        expected_base_url = invocation.execution_config.provider_base_url.rstrip("/")
        for client in clients:
            if (
                not isinstance(client, Mapping)
                or str(client.get("provider") or "") != declaration.provider
                or str(client.get("model") or "") != declaration.model
                or str(client.get("base_url") or "").rstrip("/")
                != expected_base_url
            ):
                raise ValueError(
                    "execution identity Provider/model/base URL differs from "
                    "the invocation freeze"
                )
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue("EXECUTION_IDENTITY_MISMATCH", f"{type(exc).__name__}: {exc}")
        )

    # 2) Input freeze v1: reuse the strict loader; the acknowledged assessment must
    #    authenticate against its pin (it is always blocked, never authorizing).
    try:
        actual_freeze_sha = canonical_input_freeze_manifest_sha256(
            _require_safe_path(request.input_freeze_path)
        )
        if actual_freeze_sha != declaration.input_freeze_manifest_sha256:
            raise ValueError("input freeze manifest sha differs from the pin")
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue("INPUT_FREEZE_INVALID", f"{type(exc).__name__}: {exc}"))

    # 3) Production input authority: the ONLY thing that can authorize the input,
    #    bound per-task to the real cohorts.  Preserve the verified mapping in the
    #    authorization result so the launcher never re-opens this mutable file.
    authority_issues, production_authority = _verify_production_input_authority(
        request, declaration, identity
    )
    issues.extend(authority_issues)

    # 4) Rubric identity.
    try:
        actual_rubric_sha = _sha256_of(request.rubric_path)
        if actual_rubric_sha != declaration.rubric_sha256:
            raise ValueError("rubric sha differs from the pin")
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue("RUBRIC_IDENTITY_INVALID", f"{type(exc).__name__}: {exc}"))

    status: Literal["authorized", "blocked"] = "authorized" if not issues else "blocked"
    return RealRunAuthorization(
        schema_version=REALRUN_AUTHORIZATION_SCHEMA,
        status=status,
        expected_task_ids=tuple(FIGURE2_TASK_IDS),
        declaration_sha256=declaration_sha,
        input_authority_digest=declaration.input_authority_digest,
        scientific_protocol_authority_digest=protocol_digest,
        batch_id=(declaration.batch_id if status == "authorized" else None),
        frozen_input_by_task=(
            production_authority.frozen_input_by_task()
            if status == "authorized" and production_authority is not None
            else {}
        ),
        issues=tuple(issues),
    )


def enforce_realrun_authorization(
    request: RealRunAuthorizationRequest,
) -> RealRunAuthorization:
    """Verify and RAISE ``RealRunAuthorizationBlocked`` unless authorized."""

    authorization = verify_realrun_authorization(request)
    if authorization.status != "authorized":
        raise RealRunAuthorizationBlocked(authorization)
    return authorization


# ---------------------------------------------------------------------------
# Batch identity: pre-run receipt + post-run child ledger
# ---------------------------------------------------------------------------

BATCH_RECEIPT_SCHEMA = "easyicu.figure2_batch_authorization_receipt/2"
BATCH_LEDGER_SCHEMA = "easyicu.figure2_batch_ledger/2"


@dataclass(frozen=True)
class RealRunBatchBinding:
    """One irrevocably reserved Canonical9 batch handed to the launcher."""

    batch_id: str
    declaration_sha256: str
    input_authority_digest: str
    scientific_protocol_authority_digest: str
    frozen_input_by_task: Mapping[str, str]
    batch_root: Path | None = None
    receipt_sha256: str | None = None


def _write_new_regular_file(path: Path, raw: bytes) -> None:
    """Create one regular file exactly once, without following a replacement link."""

    fd: int | None = None
    try:
        fd = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        remaining = memoryview(raw)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("short write while publishing authorization artifact")
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        if fd is not None:
            os.close(fd)


def _batch_receipt_bytes(binding: RealRunBatchBinding, *, generated_at: str) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": BATCH_RECEIPT_SCHEMA,
                "batch_id": binding.batch_id,
                "declaration_sha256": binding.declaration_sha256,
                "input_authority_digest": binding.input_authority_digest,
                "scientific_protocol_authority_digest": (
                    binding.scientific_protocol_authority_digest
                ),
                "expected_task_ids": list(FIGURE2_TASK_IDS),
                "generated_at": generated_at,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def reserve_authorized_batch_root(
    out_root: Path, binding: RealRunBatchBinding, *, generated_at: str | None = None
) -> RealRunBatchBinding:
    """Atomically reserve a fresh batch root and publish its immutable receipt.

    This is the transition from a pure successful authorization to a launchable
    batch.  It deliberately uses ``mkdir`` without ``exist_ok``: two concurrent
    launchers with the same declaration cannot both proceed.  The only cleanup on
    failure is the just-created, still-empty directory owned by this call.
    """

    root = Path(out_root).expanduser()
    if binding.batch_root is not None or binding.receipt_sha256 is not None:
        raise ValueError("a batch binding may only be reserved once")
    if not root.is_absolute() or root.is_symlink() or root.name != binding.batch_id:
        raise ValueError("batch root must be the declared absolute non-symlink batch")
    parent = root.parent
    parent_info = parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise ValueError("batch-root parent must be an existing real directory")
    os.mkdir(root, mode=0o700)
    try:
        reserved_root = _require_safe_path(root)
        if reserved_root != root:
            raise ValueError("batch root resolution changed during reservation")
        provisional = RealRunBatchBinding(
            batch_id=binding.batch_id,
            declaration_sha256=binding.declaration_sha256,
            input_authority_digest=binding.input_authority_digest,
            scientific_protocol_authority_digest=(
                binding.scientific_protocol_authority_digest
            ),
            frozen_input_by_task=dict(binding.frozen_input_by_task),
            batch_root=reserved_root,
        )
        receipt_path = write_batch_authorization_receipt(
            reserved_root,
            provisional,
            generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        )
        receipt_sha = hashlib.sha256(_read_regular_file(receipt_path)).hexdigest()
        return RealRunBatchBinding(
            batch_id=provisional.batch_id,
            declaration_sha256=provisional.declaration_sha256,
            input_authority_digest=provisional.input_authority_digest,
            scientific_protocol_authority_digest=(
                provisional.scientific_protocol_authority_digest
            ),
            frozen_input_by_task=provisional.frozen_input_by_task,
            batch_root=reserved_root,
            receipt_sha256=receipt_sha,
        )
    except BaseException:
        try:
            root.rmdir()
        except OSError:
            pass
        raise


def write_batch_authorization_receipt(
    out_root: Path, binding: RealRunBatchBinding, *, generated_at: str
) -> Path:
    """Persist the one pre-run receipt inside an already reserved batch root."""

    root = Path(out_root).expanduser()
    if binding.batch_root is None or root != binding.batch_root:
        raise ValueError("receipt root must equal the reserved batch binding root")
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise ValueError("receipt root must be a real directory")
    path = root / "figure2_realrun_authorization_receipt.json"
    _write_new_regular_file(
        path, _batch_receipt_bytes(binding, generated_at=generated_at)
    )
    return path


def verify_batch_authorization_receipt(binding: RealRunBatchBinding) -> Path:
    """Re-read the pre-run receipt and prove it still binds this reserved batch."""

    if binding.batch_root is None or binding.receipt_sha256 is None:
        raise ValueError("batch binding was not atomically reserved")
    root = _require_safe_path(binding.batch_root)
    if root != binding.batch_root or root.name != binding.batch_id or not root.is_dir():
        raise ValueError("reserved batch root is invalid")
    path = root / "figure2_realrun_authorization_receipt.json"
    raw = _read_regular_file(_require_safe_path(path))
    if hashlib.sha256(raw).hexdigest() != binding.receipt_sha256:
        raise ValueError("batch authorization receipt bytes changed after reservation")
    try:
        receipt = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_jsonl_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError("batch authorization receipt is not strict JSON") from exc
    if not isinstance(receipt, dict) or receipt != {
        "schema_version": BATCH_RECEIPT_SCHEMA,
        "batch_id": binding.batch_id,
        "declaration_sha256": binding.declaration_sha256,
        "input_authority_digest": binding.input_authority_digest,
        "scientific_protocol_authority_digest": (
            binding.scientific_protocol_authority_digest
        ),
        "expected_task_ids": list(FIGURE2_TASK_IDS),
        "generated_at": (
            receipt.get("generated_at") if isinstance(receipt, dict) else None
        ),
    }:
        raise ValueError("batch authorization receipt does not match the binding")
    if not isinstance(receipt["generated_at"], str) or not receipt["generated_at"]:
        raise ValueError("batch authorization receipt timestamp is invalid")
    return path


def build_batch_ledger(
    results_payload: Mapping[str, object],
    out_root: Path,
    binding: RealRunBatchBinding,
) -> dict:
    """POST-run: map each Canonical9 aware child run back to the batch/declaration.

    Records per task: child ``run_id``, workdir, the ``manifest.json`` sha256, the
    execution identity sha, and the bound input-authority digest — and whether it
    equals the frozen per-task value.  ``complete`` is True only when all nine map.
    """

    if binding.batch_root is None:
        raise ValueError("post-run ledger requires an atomically reserved batch")
    root = _require_safe_path(binding.batch_root)
    requested_root = Path(out_root).expanduser().resolve(strict=True)
    if requested_root != root:
        raise ValueError("post-run ledger root differs from the reserved batch root")
    verify_batch_authorization_receipt(binding)
    raw_scores = results_payload.get("scores")
    scores = raw_scores if isinstance(raw_scores, list) else []
    by_key = {
        str(score.get("item_key")): score for score in scores if isinstance(score, dict)
    }
    children: list[dict] = []
    complete = True
    for task_id in FIGURE2_TASK_IDS:
        score = by_key.get(task_id)
        aware = score.get("aware") if isinstance(score, dict) else None
        if not isinstance(aware, dict):
            children.append({"task_id": task_id, "status": "missing_aware_score"})
            complete = False
            continue
        manifest_sha = None
        actual_run_id = None
        actual_identity_sha = None
        actual_input_digest = None
        workdir_text = aware.get("workdir")
        status = "recorded"
        try:
            workdir = _require_safe_path(Path(str(workdir_text)))
            workdir.relative_to(root)
            raw_manifest = _read_regular_file(
                _require_safe_path(workdir / "manifest.json")
            )
            manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
            manifest = json.loads(
                raw_manifest.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_jsonl_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")
                ),
            )
            if not isinstance(manifest, dict):
                raise ValueError("manifest is not a JSON object")
            actual_run_id = manifest.get("run_id")
            if not isinstance(actual_run_id, str) or not actual_run_id:
                raise ValueError("manifest lacks a canonical run_id")
            expected_workdir = root / task_id / "aware" / actual_run_id
            if workdir != expected_workdir:
                raise ValueError(
                    "child manifest workdir does not match batch/task/arm/run"
                )
            from easyicu.research_agent.authority.execution_identity import (
                ExecutionIdentity,
            )

            actual_identity = ExecutionIdentity.model_validate(
                manifest.get("execution_identity"), strict=True
            )
            actual_identity_sha = actual_identity.identity_sha256
            actual_input_digest = actual_identity.input_authority_sha256
            if aware.get("run_id") not in {None, actual_run_id}:
                raise ValueError("score run_id disagrees with the child manifest")
            score_identity = aware.get("execution_identity")
            if isinstance(score_identity, Mapping) and (
                score_identity.get("identity_sha256") != actual_identity_sha
                or score_identity.get("input_authority_sha256") != actual_input_digest
            ):
                raise ValueError("score identity disagrees with the child manifest")
        except Exception as exc:  # noqa: BLE001
            status = f"manifest_unreadable: {type(exc).__name__}"
            complete = False
        if actual_input_digest != binding.frozen_input_by_task.get(task_id):
            status = "input_authority_mismatch"
            complete = False
        children.append(
            {
                "task_id": task_id,
                "run_id": actual_run_id,
                "workdir": str(workdir_text),
                "manifest_sha256": manifest_sha,
                "identity_sha256": actual_identity_sha,
                "input_authority_sha256": actual_input_digest,
                "status": status,
            }
        )
    return {
        "schema_version": BATCH_LEDGER_SCHEMA,
        "batch_id": binding.batch_id,
        "declaration_sha256": binding.declaration_sha256,
        "scientific_protocol_authority_digest": (
            binding.scientific_protocol_authority_digest
        ),
        "expected_task_ids": list(FIGURE2_TASK_IDS),
        "complete": complete,
        "children": children,
    }


def write_batch_ledger(ledger: Mapping[str, object], out_root: Path) -> Path:
    root = Path(out_root).expanduser()
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise ValueError("batch ledger root must be a real directory")
    path = root / "figure2_batch_ledger.json"
    _write_new_regular_file(
        path,
        (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return path


# ---------------------------------------------------------------------------
# Post-run per-manifest cross-check (launcher layer; evaluator stays locked)
# ---------------------------------------------------------------------------


def verify_results_frozen_input_authority(
    results_payload: Mapping[str, object],
    frozen_by_task: Mapping[str, str],
) -> list[tuple[str, str]]:
    """Return ``(task_id, reason)`` for every aware manifest whose bound input
    authority does not equal that task's frozen per-task digest.

    An empty list means every frozen task produced an aware score whose
    ``execution_identity.input_authority_sha256`` equals its frozen ``input_sha256``.
    """

    mismatches: list[tuple[str, str]] = []
    raw_scores = results_payload.get("scores")
    if not isinstance(raw_scores, list):
        return [("<results>", "scores ledger is not a list")]
    seen: set[str] = set()
    for score in raw_scores:
        if not isinstance(score, dict):
            mismatches.append(("<score>", "score row is not an object"))
            continue
        key = score.get("item_key")
        if key not in frozen_by_task:
            continue
        seen.add(str(key))
        expected = frozen_by_task[key]
        aware = score.get("aware")
        identity = aware.get("execution_identity") if isinstance(aware, dict) else None
        actual = (
            identity.get("input_authority_sha256")
            if isinstance(identity, dict)
            else None
        )
        if actual != expected:
            mismatches.append(
                (
                    str(key),
                    f"manifest input authority {actual!r} != frozen {expected!r}",
                )
            )
    for task_id in frozen_by_task:
        if task_id not in seen:
            mismatches.append((task_id, "no aware score produced for a frozen task"))
    return mismatches


def write_realrun_authorization_receipt(
    authorization: RealRunAuthorization, out_path: Path
) -> Path:
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(authorization.model_dump_json(indent=2), encoding="utf-8")
    return path


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="figure2-realrun-authority",
        description=(
            "Verify (never launch) the pre-run freeze authority against the REAL "
            "invocation flags. 0=authorized, 2=blocked."
        ),
    )
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--input-freeze", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--production-input-authority", default=None)
    parser.add_argument("--scientific-protocol-authority", required=True)
    # The real invocation flags (mirrors the launcher; never self-derived).
    parser.add_argument("--ehrflowbench-jsonl", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--submission-profile", action="store_true")
    parser.add_argument("--submission-profile-ref", default=None)
    parser.add_argument("--runner", default=None)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--require-figure2-paper-acceptance", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--force-writer-probe", action="store_true")
    parser.add_argument("--development-sample-size", type=int, default=None)
    parser.add_argument("--allow-host-runner", action="store_true")
    parser.add_argument("--allow-mock-aware", action="store_true")
    parser.add_argument("--resume-run-id", default=None)
    parser.add_argument("--resume-from-step-id", default=None)
    parser.add_argument("--enable-cross-run-memory", action="store_true")
    parser.add_argument(
        "--execution-config",
        required=True,
        help="Path to the canonical execution-config JSON pinned by the operator.",
    )
    parser.add_argument("--receipt-out", default=None)
    args = parser.parse_args(argv)

    execution_config = CanonicalExecutionConfig.model_validate_json(
        _read_regular_file(_require_safe_path(args.execution_config)), strict=True
    )
    task_ids, cohort_paths = read_canonical_jsonl_invocation(args.ehrflowbench_jsonl)
    invocation = RealRunInvocation(
        arms=tuple(args.arms),
        task_ids=task_ids,
        task_cohort_paths=tuple(zip(task_ids, cohort_paths)),
        ehrflowbench_jsonl_path=Path(args.ehrflowbench_jsonl),
        provider=str(args.provider),
        model=str(args.model),
        submission_profile_enabled=bool(args.submission_profile),
        submission_profile_ref=args.submission_profile_ref,
        runner=str(args.runner or "auto"),
        out_root=Path(args.out_root),
        require_paper_acceptance=bool(args.require_figure2_paper_acceptance),
        execution_config=execution_config,
        reuse_existing=bool(args.reuse_existing),
        repeat=int(args.repeat),
        force_writer_probe=bool(args.force_writer_probe),
        development_sample_size=args.development_sample_size,
        allow_host_runner=bool(args.allow_host_runner),
        allow_mock_aware=bool(args.allow_mock_aware),
        resume_run_id=args.resume_run_id,
        resume_from_step_id=args.resume_from_step_id,
        cross_run_memory=bool(args.enable_cross_run_memory),
    )
    request = RealRunAuthorizationRequest(
        declaration_path=Path(args.declaration),
        expected_execution_identity_path=Path(args.expected_identity),
        input_freeze_path=Path(args.input_freeze),
        rubric_path=Path(args.rubric),
        invocation=invocation,
        production_input_authority_path=(
            Path(args.production_input_authority)
            if args.production_input_authority
            else None
        ),
        scientific_protocol_authority_path=Path(
            args.scientific_protocol_authority
        ),
    )
    authorization = verify_realrun_authorization(request)
    if args.receipt_out:
        write_realrun_authorization_receipt(authorization, Path(args.receipt_out))
    print(authorization.model_dump_json(indent=2))
    return 0 if authorization.status == "authorized" else 2


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1:]))


__all__ = [
    "REALRUN_AUTHORIZATION_SCHEMA",
    "OPERATOR_FREEZE_DECLARATION_SCHEMA",
    "PRODUCTION_INPUT_AUTHORITY_SCHEMA",
    "CANONICAL_EXECUTION_CONFIG_SCHEMA",
    "BATCH_RECEIPT_SCHEMA",
    "BATCH_LEDGER_SCHEMA",
    "AUTHORIZED_ARMS",
    "production_cohort_input_sha256",
    "production_provenance_sha256",
    "resolve_strict_jsonl_path",
    "read_canonical_jsonl_rows",
    "read_canonical_jsonl_invocation",
    "jsonl_references_canonical9",
    "CanonicalJsonlRow",
    "CanonicalExecutionConfig",
    "build_canonical_execution_config",
    "ProductionInputTask",
    "ProductionInputAuthority",
    "load_production_input_authority",
    "OperatorFreezeDeclaration",
    "RealRunInvocation",
    "RealRunAuthorizationIssue",
    "RealRunAuthorization",
    "RealRunAuthorizationRequest",
    "RealRunAuthorizationBlocked",
    "RealRunBatchBinding",
    "verify_realrun_authorization",
    "enforce_realrun_authorization",
    "verify_results_frozen_input_authority",
    "write_realrun_authorization_receipt",
    "reserve_authorized_batch_root",
    "write_batch_authorization_receipt",
    "verify_batch_authorization_receipt",
    "build_batch_ledger",
    "write_batch_ledger",
]
