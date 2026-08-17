"""Persist the Progressive Planner contract chain as governed evidence.

This module owns the boundary between planning contracts and EvidenceStore.  It
does not plan, compile, execute, or infer authority from mutable pipeline state.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..canonical_json import canonical_sha256
from .progressive_contract import (
    ProgressiveFoundationMaterialization,
    ProgressivePlanCompileReceipt,
    ProgressivePlanOutline,
    ProgressivePlannerCheckpoint,
    ProgressivePlanSkeleton,
    ProgressiveStepMaterialization,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_RESUME_CHECKPOINT_BYTES = 16 * 1024 * 1024


class ProgressivePlanningArtifactError(ValueError):
    """A planning artifact chain is incomplete or authority-inconsistent."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(f"{reason_code}: {message}")
        self.reason_code = reason_code


class ProgressiveCompilerFinding(BaseModel):
    """Closed compiler diagnostic safe to persist without raw model text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: Literal["easyicu.planning.progressive_compiler_v1"]
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    step_id: str | None = Field(default=None, max_length=160)
    step_index: int | None = Field(default=None, ge=0)
    path: str | None = Field(default=None, max_length=240)
    message: str = Field(min_length=1, max_length=2_000)
    findings: list["ProgressiveCompilerFinding"] = Field(
        default_factory=list,
        max_length=20,
    )


class ProgressiveCompileReplayAttempt(BaseModel):
    """One schema-validated candidate rejected by the deterministic compiler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int = Field(ge=0, le=20)
    step_schema_authority_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    materialization: ProgressiveStepMaterialization
    materialization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_finding: ProgressiveCompilerFinding

    @model_validator(mode="after")
    def _materialization_digest_matches(self) -> "ProgressiveCompileReplayAttempt":
        observed = canonical_sha256(self.materialization.model_dump(mode="json"))
        if observed != self.materialization_sha256:
            raise ValueError("compile replay materialization digest mismatch")
        return self


class ProgressiveCompileFailureReplay(BaseModel):
    """Content-addressed failed-candidate set for zero-provider replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.progressive_compile_failure_replay/1"] = (
        "easyicu.progressive_compile_failure_replay/1"
    )
    owner: Literal["easyicu.planning.progressive_artifacts_v1"] = (
        "easyicu.planning.progressive_artifacts_v1"
    )
    request_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prefix_checkpoint_sequence: int = Field(ge=0)
    prefix_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempts: list[ProgressiveCompileReplayAttempt] = Field(
        min_length=1,
        max_length=20,
    )
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _artifact_digest_matches(self) -> "ProgressiveCompileFailureReplay":
        unsigned = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if canonical_sha256(unsigned) != self.artifact_sha256:
            raise ValueError("compile replay artifact digest mismatch")
        return self


class ProgressiveEvidenceRegistrar(Protocol):
    """Small EvidenceStore surface required by this owner module."""

    def get(self, evidence_id_or_alias: str) -> object | None: ...

    def register_file(self, **kwargs: Any) -> object: ...


@dataclass(frozen=True)
class ProgressivePlanningArtifactPaths:
    """Paths for the persisted outline-to-compile authority chain."""

    outline: Path
    foundation: Path
    materializations: Path
    skeleton: Path
    compile_receipt: Path


@dataclass(frozen=True)
class ProgressiveResumePersistenceReceipt:
    """Auditable summary after a validated Dev replay chain is imported."""

    source_checkpoint_sha256: str
    source_sequence: int
    reused_materialization_count: int
    new_checkpoint_count: int


@dataclass
class ProgressivePlannerCheckpointEmitter:
    """Build one typed append-only checkpoint chain in memory.

    Persistence remains the recorder's responsibility. Keeping sequence and
    predecessor bookkeeping here prevents the provider loop from owning
    artifact-chain mechanics.
    """

    callback: Callable[[ProgressivePlannerCheckpoint], None] | None
    request_authority_sha256: str
    source_checkpoint: ProgressivePlannerCheckpoint | None = None
    _sequence: int = field(init=False)
    _previous_checkpoint_sha256: str | None = field(init=False)

    def __post_init__(self) -> None:
        self._sequence = (
            int(self.source_checkpoint.sequence) + 1
            if self.source_checkpoint is not None
            else 0
        )
        self._previous_checkpoint_sha256 = (
            self.source_checkpoint.checkpoint_sha256
            if self.source_checkpoint is not None
            else None
        )

    def emit(
        self,
        *,
        stage: str,
        outline: ProgressivePlanOutline,
        foundation: ProgressiveFoundationMaterialization | None,
        materializations: Sequence[ProgressiveStepMaterialization],
        prompt_metrics: Mapping[str, Any],
    ) -> None:
        if self.callback is None:
            return
        body: dict[str, Any] = {
            "schema_version": "easyicu.progressive_planner_checkpoint/1",
            "sequence": self._sequence,
            "stage": stage,
            "request_authority_sha256": self.request_authority_sha256,
            "previous_checkpoint_sha256": self._previous_checkpoint_sha256,
            "outline": outline.model_dump(mode="json"),
            "foundation": (
                foundation.model_dump(mode="json")
                if foundation is not None
                else None
            ),
            "materializations": [
                item.model_dump(mode="json") for item in materializations
            ],
            "prompt_metrics": json.loads(
                json.dumps(prompt_metrics, ensure_ascii=False)
            ),
        }
        body["checkpoint_sha256"] = canonical_sha256(body)
        checkpoint = ProgressivePlannerCheckpoint.model_validate(body)
        self.callback(checkpoint)
        self._previous_checkpoint_sha256 = checkpoint.checkpoint_sha256
        self._sequence += 1


@dataclass
class ProgressivePlannerCheckpointRecorder:
    """Persist normal checkpoints immediately and buffer Dev continuations.

    A resumed source chain is not registered against the current run until the
    Planner has independently verified its dependency authority and recompiled
    its prefix.  The pipeline calls :meth:`persist_validated_resume` only after
    that gate succeeds, including when a later suffix call fails.
    """

    run_dir: Path
    evidence: ProgressiveEvidenceRegistrar
    prompt_pack_version: str
    source_chain: tuple[ProgressivePlannerCheckpoint, ...] = ()
    _pending: list[ProgressivePlannerCheckpoint] = field(default_factory=list)
    _persisted: bool = False
    _latest_checkpoint: ProgressivePlannerCheckpoint | None = field(
        init=False,
        default=None,
    )

    def __post_init__(self) -> None:
        if self.source_chain:
            self._latest_checkpoint = self.source_chain[-1]

    @property
    def latest_checkpoint(self) -> ProgressivePlannerCheckpoint | None:
        return self._latest_checkpoint

    def record(self, checkpoint: ProgressivePlannerCheckpoint) -> None:
        if self._persisted:
            raise ProgressivePlanningArtifactError(
                "progressive_resume_checkpoint_recorder_closed",
                "validated resume checkpoints have already been persisted",
            )
        if self.source_chain:
            self._pending.append(checkpoint)
            self._latest_checkpoint = checkpoint
            return
        persist_progressive_planner_checkpoint(
            run_dir=self.run_dir,
            evidence=self.evidence,
            checkpoint=checkpoint,
            prompt_pack_version=self.prompt_pack_version,
        )
        self._latest_checkpoint = checkpoint

    def persist_validated_resume(self) -> ProgressiveResumePersistenceReceipt:
        if not self.source_chain:
            raise ProgressivePlanningArtifactError(
                "progressive_resume_source_chain_missing",
                "checkpoint recorder has no development source chain",
            )
        terminal = self.source_chain[-1]
        if not self._persisted:
            for checkpoint in (*self.source_chain, *self._pending):
                persist_progressive_planner_checkpoint(
                    run_dir=self.run_dir,
                    evidence=self.evidence,
                    checkpoint=checkpoint,
                    prompt_pack_version=self.prompt_pack_version,
                )
            self._persisted = True
        return ProgressiveResumePersistenceReceipt(
            source_checkpoint_sha256=terminal.checkpoint_sha256,
            source_sequence=int(terminal.sequence),
            reused_materialization_count=len(terminal.materializations),
            new_checkpoint_count=len(self._pending),
        )


class ProgressivePlanningStepAuthority(BaseModel):
    """One materialized step and the exact schema that governed its response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    materialization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    structured_output_authority_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class ProgressivePlanningAuthority(BaseModel):
    """Content-addressed root joining progressive planning to normalized Plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.progressive_planning_authority/2"] = (
        "easyicu.progressive_planning_authority/2"
    )
    planner_strategy: Literal["progressive_v2"] = "progressive_v2"
    outline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outline_structured_output_authority_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    foundation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_structured_output_authority_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    ordered_steps: tuple[ProgressivePlanningStepAuthority, ...]
    compiled_skeleton_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_analysis_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_plan_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outline_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialization_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    skeleton_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compile_receipt_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    planner_prompt_metrics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_plan_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_plan_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strict_transport_bound: bool
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _verify_authority(self) -> "ProgressivePlanningAuthority":
        if not self.ordered_steps:
            raise ValueError("progressive planning authority has no steps")
        step_ids = [item.step_id for item in self.ordered_steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("progressive planning authority repeats a step id")
        observed_strict = bool(
            self.outline_structured_output_authority_sha256
            and self.foundation_structured_output_authority_sha256
            and all(
                item.structured_output_authority_sha256
                for item in self.ordered_steps
            )
        )
        if self.strict_transport_bound is not observed_strict:
            raise ValueError("progressive strict transport projection mismatch")
        unsigned = self.model_dump(mode="json", exclude={"authority_sha256"})
        if canonical_sha256(unsigned) != self.authority_sha256:
            raise ValueError("progressive planning authority digest mismatch")
        return self


def _authority_digest(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    digest = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ProgressivePlanningArtifactError(
            "progressive_schema_authority_invalid",
            f"{field} must be null or a lowercase SHA-256 digest",
        )
    return digest


def _record_sha256(record: object) -> str | None:
    raw = (
        record.get("sha256")
        if isinstance(record, Mapping)
        else getattr(record, "sha256", None)
    )
    digest = str(raw or "").strip().lower()
    return digest if _SHA256_RE.fullmatch(digest) else None


def _verified_source_bytes(
    *,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    evidence_id: str,
    filename: str,
) -> tuple[bytes, str]:
    record = evidence.get(evidence_id)
    expected_sha256 = _record_sha256(record) if record is not None else None
    path = Path(run_dir) / filename
    if expected_sha256 is None:
        raise ProgressivePlanningArtifactError(
            "progressive_source_evidence_missing",
            f"{evidence_id} has no registered SHA-256 authority",
        )
    if path.is_symlink() or not path.is_file():
        raise ProgressivePlanningArtifactError(
            "progressive_source_artifact_missing",
            f"{filename} is absent or not a regular file",
        )
    content = path.read_bytes()
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ProgressivePlanningArtifactError(
            "progressive_source_artifact_digest_mismatch",
            f"{filename} differs from EvidenceStore authority",
        )
    return content, observed_sha256


def _write_and_register_once(
    evidence: ProgressiveEvidenceRegistrar,
    *,
    content: str,
    evidence_id: str,
    description: str,
    source_path: Path,
    inputs: Sequence[str],
    producer: str,
    generation_mode: str,
    prompt_pack_version: str,
) -> None:
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    existing = evidence.get(evidence_id)
    if existing is not None:
        existing_sha256 = _record_sha256(existing)
        if existing_sha256 != content_sha256:
            raise ProgressivePlanningArtifactError(
                "progressive_existing_evidence_identity_mismatch",
                f"{evidence_id} already identifies different content",
            )
        source_path.write_text(content, encoding="utf-8")
        return
    source_path.write_text(content, encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=description,
        source_path=source_path,
        evidence_id=evidence_id,
        inputs=list(inputs),
        producer=producer,
        generation_mode=generation_mode,
        prompt_pack_version=prompt_pack_version,
    )


def persist_progressive_planner_checkpoint(
    *,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    checkpoint: ProgressivePlannerCheckpoint,
    prompt_pack_version: str,
) -> Path:
    """Persist one append-only outline/foundation/prefix checkpoint."""

    sequence = int(checkpoint.sequence)
    evidence_id = f"progressive_planner_checkpoint_{sequence:03d}"
    inputs = ["research_context"]
    if sequence:
        previous_id = f"progressive_planner_checkpoint_{sequence - 1:03d}"
        previous_bytes, _digest = _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id=previous_id,
            filename=f"{previous_id}.json",
        )
        try:
            previous = ProgressivePlannerCheckpoint.model_validate_json(
                previous_bytes
            )
        except Exception as exc:
            raise ProgressivePlanningArtifactError(
                "progressive_checkpoint_predecessor_invalid",
                str(exc),
            ) from exc
        if (
            checkpoint.previous_checkpoint_sha256
            != previous.checkpoint_sha256
        ):
            raise ProgressivePlanningArtifactError(
                "progressive_checkpoint_chain_mismatch",
                "checkpoint predecessor digest does not match sequence authority",
            )
        if (
            checkpoint.request_authority_sha256
            != previous.request_authority_sha256
        ):
            raise ProgressivePlanningArtifactError(
                "progressive_checkpoint_request_authority_drift",
                "checkpoint request authority changed within one planning chain",
            )
        inputs.append(previous_id)
    path = Path(run_dir) / f"{evidence_id}.json"
    _write_and_register_once(
        evidence,
        content=checkpoint.model_dump_json(indent=2),
        evidence_id=evidence_id,
        description=(
            "Append-only Progressive Planner checkpoint after a host-validated "
            f"{checkpoint.stage} boundary."
        ),
        source_path=path,
        inputs=inputs,
        producer="progressive_planner_checkpoint",
        generation_mode="deterministic_skill",
        prompt_pack_version=prompt_pack_version,
    )
    return path


def persist_progressive_compile_failure_replay(
    *,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    attempts: Sequence[ProgressiveCompileReplayAttempt],
    prefix_checkpoint: ProgressivePlannerCheckpoint,
    prompt_pack_version: str,
) -> Path:
    """Persist normalized failed candidates for zero-provider compiler replay."""

    validated_attempts = [
        ProgressiveCompileReplayAttempt.model_validate(item)
        for item in attempts
    ]
    body: dict[str, Any] = {
        "schema_version": "easyicu.progressive_compile_failure_replay/1",
        "owner": "easyicu.planning.progressive_artifacts_v1",
        "request_authority_sha256": (
            prefix_checkpoint.request_authority_sha256
        ),
        "prefix_checkpoint_sequence": int(prefix_checkpoint.sequence),
        "prefix_checkpoint_sha256": prefix_checkpoint.checkpoint_sha256,
        "attempts": [item.model_dump(mode="json") for item in validated_attempts],
    }
    body["artifact_sha256"] = canonical_sha256(body)
    replay = ProgressiveCompileFailureReplay.model_validate(body)
    evidence_id = "progressive_compile_failure_replay"
    checkpoint_id = (
        f"progressive_planner_checkpoint_{prefix_checkpoint.sequence:03d}"
    )
    path = Path(run_dir) / f"{evidence_id}.json"
    _write_and_register_once(
        evidence,
        content=replay.model_dump_json(indent=2),
        evidence_id=evidence_id,
        description=(
            "Schema-validated Progressive Planner candidates rejected by the "
            "deterministic host compiler; safe for zero-provider replay."
        ),
        source_path=path,
        inputs=("research_context", checkpoint_id),
        producer="progressive_compile_failure_replay",
        generation_mode="llm_validated_diagnostic",
        prompt_pack_version=prompt_pack_version,
    )
    return path


def load_progressive_compile_failure_replay(
    *,
    replay_path: Path,
    expected_artifact_sha256: str,
) -> ProgressiveCompileFailureReplay:
    """Load one digest-bound failed-candidate set without calling a provider."""

    path = Path(replay_path)
    expected = str(expected_artifact_sha256 or "").strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise ProgressivePlanningArtifactError(
            "progressive_compile_replay_digest_invalid",
            "expected compile replay artifact digest is not a lowercase SHA-256",
        )
    if path.is_symlink() or not path.is_file():
        raise ProgressivePlanningArtifactError(
            "progressive_compile_replay_missing",
            "compile replay artifact is absent or not a regular file",
        )
    if path.stat().st_size > _MAX_RESUME_CHECKPOINT_BYTES:
        raise ProgressivePlanningArtifactError(
            "progressive_compile_replay_too_large",
            "compile replay artifact exceeds the bounded reader limit",
        )
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ProgressivePlanningArtifactError(
            "progressive_compile_replay_digest_mismatch",
            "compile replay artifact differs from its external authority",
        )
    try:
        return ProgressiveCompileFailureReplay.model_validate_json(content)
    except Exception as exc:
        raise ProgressivePlanningArtifactError(
            "progressive_compile_replay_invalid",
            "compile replay artifact failed its closed typed contract",
        ) from exc


def load_progressive_planner_checkpoint_chain(
    *,
    last_checkpoint_path: Path,
    expected_artifact_sha256: str,
) -> tuple[ProgressivePlannerCheckpoint, ...]:
    """Load one complete, append-only checkpoint chain for development replay.

    The caller supplies the SHA-256 of the terminal *file*, while every parsed
    checkpoint verifies its own canonical content digest.  Requiring the
    canonical sibling filenames from ``000`` through the terminal sequence
    prevents a single cumulative checkpoint from silently standing in for an
    unavailable predecessor chain.
    """

    expected_digest = str(expected_artifact_sha256 or "").strip().lower()
    if not _SHA256_RE.fullmatch(expected_digest):
        raise ProgressivePlanningArtifactError(
            "progressive_resume_artifact_digest_invalid",
            "expected terminal checkpoint artifact SHA-256 is invalid",
        )

    terminal_path = Path(last_checkpoint_path).expanduser()

    def read_checkpoint(
        path: Path,
    ) -> tuple[bytes, ProgressivePlannerCheckpoint]:
        if path.is_symlink():
            raise ProgressivePlanningArtifactError(
                "progressive_resume_checkpoint_symlink",
                f"resume checkpoint must not be a symbolic link: {path.name}",
            )
        try:
            stat = path.stat()
        except OSError as exc:
            raise ProgressivePlanningArtifactError(
                "progressive_resume_checkpoint_missing",
                f"resume checkpoint is unavailable: {path.name}",
            ) from exc
        if not path.is_file():
            raise ProgressivePlanningArtifactError(
                "progressive_resume_checkpoint_not_regular",
                f"resume checkpoint is not a regular file: {path.name}",
            )
        if stat.st_size <= 0 or stat.st_size > _MAX_RESUME_CHECKPOINT_BYTES:
            raise ProgressivePlanningArtifactError(
                "progressive_resume_checkpoint_size_invalid",
                f"resume checkpoint size is outside the accepted range: {path.name}",
            )
        try:
            raw = path.read_bytes()
            checkpoint = ProgressivePlannerCheckpoint.model_validate_json(raw)
        except Exception as exc:
            raise ProgressivePlanningArtifactError(
                "progressive_resume_checkpoint_invalid",
                f"resume checkpoint failed typed validation: {path.name}",
            ) from exc
        return raw, checkpoint

    terminal_bytes, terminal = read_checkpoint(terminal_path)
    terminal_filename = (
        f"progressive_planner_checkpoint_{terminal.sequence:03d}.json"
    )
    if terminal_path.name != terminal_filename:
        raise ProgressivePlanningArtifactError(
            "progressive_resume_checkpoint_filename_mismatch",
            "terminal checkpoint filename does not match its typed sequence",
        )
    if hashlib.sha256(terminal_bytes).hexdigest() != expected_digest:
        raise ProgressivePlanningArtifactError(
            "progressive_resume_artifact_digest_mismatch",
            "terminal checkpoint file does not match caller authority",
        )

    chain: list[ProgressivePlannerCheckpoint] = []
    previous: ProgressivePlannerCheckpoint | None = None
    for sequence in range(terminal.sequence + 1):
        path = terminal_path.parent / (
            f"progressive_planner_checkpoint_{sequence:03d}.json"
        )
        _raw, checkpoint = read_checkpoint(path)
        if checkpoint.sequence != sequence:
            raise ProgressivePlanningArtifactError(
                "progressive_resume_checkpoint_sequence_mismatch",
                f"checkpoint sequence does not match filename: {path.name}",
            )
        if (
            checkpoint.request_authority_sha256
            != terminal.request_authority_sha256
        ):
            raise ProgressivePlanningArtifactError(
                "progressive_resume_checkpoint_request_authority_drift",
                "checkpoint request authority changes within the source chain",
            )
        if previous is not None and (
            checkpoint.previous_checkpoint_sha256
            != previous.checkpoint_sha256
        ):
            raise ProgressivePlanningArtifactError(
                "progressive_resume_checkpoint_chain_mismatch",
                "checkpoint predecessor digest does not close the source chain",
            )
        chain.append(checkpoint)
        previous = checkpoint

    if chain[-1].checkpoint_sha256 != terminal.checkpoint_sha256:
        raise ProgressivePlanningArtifactError(
            "progressive_resume_terminal_checkpoint_mismatch",
            "terminal checkpoint changed while its source chain was loaded",
        )
    return tuple(chain)


def persist_progressive_planning_artifacts(
    *,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    outline: ProgressivePlanOutline,
    foundation: ProgressiveFoundationMaterialization,
    materializations: Sequence[ProgressiveStepMaterialization],
    skeleton: ProgressivePlanSkeleton,
    compile_receipt: ProgressivePlanCompileReceipt,
    prompt_metrics: Mapping[str, Any],
    prompt_pack_version: str,
) -> ProgressivePlanningArtifactPaths:
    """Write and register one complete outline/materialize/compile evidence chain."""
    if not materializations:
        raise ProgressivePlanningArtifactError(
            "progressive_materialization_ledger_empty",
            "at least one current-step materialization is required",
        )
    if any(item.foundation is not None for item in materializations):
        raise ProgressivePlanningArtifactError(
            "progressive_step_repeats_sealed_foundation",
            "step materializations must keep foundation=null after it is sealed",
        )

    observed_outline_sha256 = canonical_sha256(outline.model_dump(mode="json"))
    if prompt_metrics.get("outline_sha256") != observed_outline_sha256:
        raise ProgressivePlanningArtifactError(
            "progressive_outline_identity_mismatch",
            "prompt metrics do not identify the persisted outline",
        )
    if foundation.outline_sha256 != observed_outline_sha256:
        raise ProgressivePlanningArtifactError(
            "progressive_foundation_outline_identity_mismatch",
            "plan foundation does not identify the persisted outline",
        )
    if prompt_metrics.get("final_skeleton_sha256") != compile_receipt.skeleton_sha256:
        raise ProgressivePlanningArtifactError(
            "progressive_skeleton_identity_mismatch",
            "prompt metrics and compile receipt identify different skeletons",
        )

    raw_step_schema_digests = prompt_metrics.get(
        "step_materialization_schema_sha256"
    )
    if not isinstance(raw_step_schema_digests, list) or len(
        raw_step_schema_digests
    ) != len(materializations):
        raise ProgressivePlanningArtifactError(
            "progressive_step_schema_authority_count_mismatch",
            "one schema authority entry is required per materialized step",
        )
    step_schema_digests = [
        _authority_digest(value, field=f"step_schema[{index}]")
        for index, value in enumerate(raw_step_schema_digests)
    ]
    outline_schema_digest = _authority_digest(
        prompt_metrics.get("structured_output_authority_sha256"),
        field="outline_schema",
    )
    foundation_schema_digest = _authority_digest(
        prompt_metrics.get("foundation_structured_output_authority_sha256"),
        field="foundation_schema",
    )

    outline_path = run_dir / "progressive_plan_outline.json"
    _write_and_register_once(
        evidence,
        content=outline.model_dump_json(indent=2),
        evidence_id="progressive_plan_outline",
        description=(
            "Retrieval-informed high-level scientific outline before any "
            "executable step detail."
        ),
        source_path=outline_path,
        inputs=("research_context",),
        producer="progressive_planner",
        generation_mode="llm",
        prompt_pack_version=prompt_pack_version,
    )

    foundation_path = run_dir / "progressive_plan_foundation.json"
    _write_and_register_once(
        evidence,
        content=foundation.model_dump_json(indent=2),
        evidence_id="progressive_plan_foundation",
        description=(
            "Outline-bound plan-wide cohort, label, robustness, and know-how "
            "choices, separated from executable step detail."
        ),
        source_path=foundation_path,
        inputs=("progressive_plan_outline", "research_context"),
        producer="progressive_planner",
        generation_mode="llm",
        prompt_pack_version=prompt_pack_version,
    )

    materializations_path = run_dir / "progressive_step_materializations.json"
    materializations_content = (
        json.dumps(
            {
                "schema_version": (
                    "easyicu.progressive_step_materialization_ledger/2"
                ),
                "outline_sha256": observed_outline_sha256,
                "outline_structured_output_authority_sha256": (
                    outline_schema_digest
                ),
                "foundation_sha256": canonical_sha256(
                    foundation.model_dump(mode="json")
                ),
                "foundation_structured_output_authority_sha256": (
                    foundation_schema_digest
                ),
                "materializations": [
                    {
                        "step_id": item.step.step_id,
                        "structured_output_authority_sha256": (
                            step_schema_digests[index]
                        ),
                        "materialization": item.model_dump(mode="json"),
                    }
                    for index, item in enumerate(materializations)
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _write_and_register_once(
        evidence,
        content=materializations_content,
        evidence_id="progressive_step_materializations",
        description=(
            "Ordered current-step strict materializations with their run-bound "
            "schema authority digests."
        ),
        source_path=materializations_path,
        inputs=(
            "progressive_plan_outline",
            "progressive_plan_foundation",
            "research_context",
        ),
        producer="progressive_planner",
        generation_mode="llm",
        prompt_pack_version=prompt_pack_version,
    )

    skeleton_path = run_dir / "progressive_plan_skeleton.json"
    _write_and_register_once(
        evidence,
        content=skeleton.model_dump_json(indent=2),
        evidence_id="progressive_plan_skeleton",
        description=(
            "Host-assembled scientific skeleton produced from coordinate-bound "
            "step materializations."
        ),
        source_path=skeleton_path,
        inputs=(
            "progressive_plan_outline",
            "progressive_plan_foundation",
            "progressive_step_materializations",
            "research_context",
        ),
        producer="progressive_planner_compiler",
        generation_mode="deterministic_skill",
        prompt_pack_version=prompt_pack_version,
    )

    receipt_path = run_dir / "progressive_plan_compile_receipt.json"
    _write_and_register_once(
        evidence,
        content=compile_receipt.model_dump_json(indent=2),
        evidence_id="progressive_plan_compile_receipt",
        description=(
            "Host-derived immutable-prefix and AnalysisPlan compilation receipt "
            "for Progressive Planner v2."
        ),
        source_path=receipt_path,
        inputs=("progressive_plan_skeleton", "research_context"),
        producer="progressive_plan_compiler",
        generation_mode="deterministic_skill",
        prompt_pack_version=prompt_pack_version,
    )
    return ProgressivePlanningArtifactPaths(
        outline=outline_path,
        foundation=foundation_path,
        materializations=materializations_path,
        skeleton=skeleton_path,
        compile_receipt=receipt_path,
    )


def persist_progressive_planning_authority(
    *,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    proposed_plan_sha256: str,
    normalized_plan_sha256: str,
    normalized_plan_authority_sha256: str,
    normalized_plan_evidence_id: str,
    normalized_plan_filename: str,
    prompt_pack_version: str,
) -> ProgressivePlanningAuthority:
    """Verify the on-disk chain and bind it to one normalized plan authority."""

    sources = {
        "outline": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="progressive_plan_outline",
            filename="progressive_plan_outline.json",
        ),
        "foundation": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="progressive_plan_foundation",
            filename="progressive_plan_foundation.json",
        ),
        "materializations": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="progressive_step_materializations",
            filename="progressive_step_materializations.json",
        ),
        "skeleton": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="progressive_plan_skeleton",
            filename="progressive_plan_skeleton.json",
        ),
        "compile_receipt": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="progressive_plan_compile_receipt",
            filename="progressive_plan_compile_receipt.json",
        ),
        "prompt_metrics": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="planner_prompt_metrics",
            filename="planner_prompt_metrics.json",
        ),
        "analysis_plan": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="analysis_plan",
            filename="analysis_plan.json",
        ),
        "normalized_plan": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id=normalized_plan_evidence_id,
            filename=normalized_plan_filename,
        ),
    }
    try:
        outline = ProgressivePlanOutline.model_validate_json(sources["outline"][0])
        foundation = ProgressiveFoundationMaterialization.model_validate_json(
            sources["foundation"][0]
        )
        ledger = json.loads(sources["materializations"][0])
        skeleton = ProgressivePlanSkeleton.model_validate_json(
            sources["skeleton"][0]
        )
        receipt = ProgressivePlanCompileReceipt.model_validate_json(
            sources["compile_receipt"][0]
        )
        prompt_metrics = json.loads(sources["prompt_metrics"][0])
    except Exception as exc:
        raise ProgressivePlanningArtifactError(
            "progressive_authority_source_invalid",
            str(exc),
        ) from exc
    if not isinstance(ledger, dict) or not isinstance(prompt_metrics, dict):
        raise ProgressivePlanningArtifactError(
            "progressive_authority_source_invalid",
            "materialization ledger and prompt metrics must be objects",
        )
    if ledger.get("schema_version") != (
        "easyicu.progressive_step_materialization_ledger/2"
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_materialization_ledger_version_mismatch",
            "separate-foundation planning requires materialization ledger v2",
        )
    entries = ledger.get("materializations")
    if not isinstance(entries, list) or not entries:
        raise ProgressivePlanningArtifactError(
            "progressive_materialization_ledger_invalid",
            "materialization ledger must contain ordered steps",
        )
    try:
        materializations = [
            ProgressiveStepMaterialization.model_validate(item["materialization"])
            for item in entries
            if isinstance(item, dict)
        ]
    except Exception as exc:
        raise ProgressivePlanningArtifactError(
            "progressive_materialization_ledger_invalid",
            str(exc),
        ) from exc
    if len(materializations) != len(entries):
        raise ProgressivePlanningArtifactError(
            "progressive_materialization_ledger_invalid",
            "every ledger row must be one materialization object",
        )
    if any(item.foundation is not None for item in materializations):
        raise ProgressivePlanningArtifactError(
            "progressive_step_repeats_sealed_foundation",
            "materialization ledger repeats the separately sealed foundation",
        )
    outline_step_ids = [item.step_id for item in outline.steps]
    materialized_step_ids = [item.step.step_id for item in materializations]
    skeleton_step_ids = [item.step_id for item in skeleton.steps]
    ledger_step_ids = [str(item.get("step_id") or "") for item in entries]
    if not (
        outline_step_ids
        == ledger_step_ids
        == materialized_step_ids
        == skeleton_step_ids
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_step_order_identity_mismatch",
            "outline, ledger, materializations, and skeleton disagree on step order",
        )
    outline_sha256 = canonical_sha256(outline.model_dump(mode="json"))
    if (
        ledger.get("outline_sha256") != outline_sha256
        or prompt_metrics.get("outline_sha256") != outline_sha256
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_outline_identity_mismatch",
            "authority sources disagree on the outline digest",
        )
    if foundation.outline_sha256 != outline_sha256:
        raise ProgressivePlanningArtifactError(
            "progressive_foundation_outline_identity_mismatch",
            "foundation and outline digests disagree",
        )
    foundation_sha256 = canonical_sha256(foundation.model_dump(mode="json"))
    if ledger.get("foundation_sha256") != foundation_sha256:
        raise ProgressivePlanningArtifactError(
            "progressive_foundation_identity_mismatch",
            "materialization ledger does not identify the persisted foundation",
        )
    if skeleton.cohort != foundation.foundation.cohort or (
        list(skeleton.display_labels) != list(foundation.foundation.display_labels)
        or list(skeleton.robustness_intents)
        != list(foundation.foundation.robustness_intents)
        or list(skeleton.know_how_decisions)
        != list(foundation.foundation.know_how_decisions)
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_foundation_skeleton_identity_mismatch",
            "compiled skeleton plan-wide fields differ from the sealed foundation",
        )
    skeleton_sha256 = canonical_sha256(skeleton.model_dump(mode="json"))
    if (
        skeleton_sha256 != receipt.skeleton_sha256
        or prompt_metrics.get("final_skeleton_sha256") != skeleton_sha256
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_skeleton_identity_mismatch",
            "authority sources disagree on the compiled skeleton digest",
        )
    if receipt.analysis_plan_sha256 != proposed_plan_sha256:
        raise ProgressivePlanningArtifactError(
            "progressive_compiled_plan_identity_mismatch",
            "compiler receipt does not identify the normalized lineage proposal",
        )
    outline_schema_sha256 = _authority_digest(
        ledger.get("outline_structured_output_authority_sha256"),
        field="outline_schema",
    )
    if outline_schema_sha256 != _authority_digest(
        prompt_metrics.get("structured_output_authority_sha256"),
        field="prompt_metrics.outline_schema",
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_schema_authority_mismatch",
            "outline schema authority differs between ledger and prompt metrics",
        )
    foundation_schema_sha256 = _authority_digest(
        ledger.get("foundation_structured_output_authority_sha256"),
        field="foundation_schema",
    )
    if foundation_schema_sha256 != _authority_digest(
        prompt_metrics.get("foundation_structured_output_authority_sha256"),
        field="prompt_metrics.foundation_schema",
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_schema_authority_mismatch",
            "foundation schema authority differs between ledger and prompt metrics",
        )
    metrics_step_schema = prompt_metrics.get("step_materialization_schema_sha256")
    if not isinstance(metrics_step_schema, list) or len(metrics_step_schema) != len(
        entries
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_step_schema_authority_count_mismatch",
            "prompt metrics do not contain one schema digest per step",
        )
    ordered_steps: list[ProgressivePlanningStepAuthority] = []
    for index, (entry, materialization) in enumerate(
        zip(entries, materializations, strict=True)
    ):
        entry_schema_sha256 = _authority_digest(
            entry.get("structured_output_authority_sha256"),
            field=f"ledger.step_schema[{index}]",
        )
        if entry_schema_sha256 != _authority_digest(
            metrics_step_schema[index],
            field=f"prompt_metrics.step_schema[{index}]",
        ):
            raise ProgressivePlanningArtifactError(
                "progressive_schema_authority_mismatch",
                f"step schema authority differs at index {index}",
            )
        ordered_steps.append(
            ProgressivePlanningStepAuthority(
                step_id=materialization.step.step_id,
                materialization_sha256=canonical_sha256(
                    materialization.model_dump(mode="json")
                ),
                structured_output_authority_sha256=entry_schema_sha256,
            )
        )
    body: dict[str, Any] = {
        "schema_version": "easyicu.progressive_planning_authority/2",
        "planner_strategy": "progressive_v2",
        "outline_sha256": outline_sha256,
        "outline_structured_output_authority_sha256": outline_schema_sha256,
        "foundation_sha256": foundation_sha256,
        "foundation_structured_output_authority_sha256": (
            foundation_schema_sha256
        ),
        "ordered_steps": [item.model_dump(mode="json") for item in ordered_steps],
        "compiled_skeleton_sha256": skeleton_sha256,
        "compiled_analysis_plan_sha256": receipt.analysis_plan_sha256,
        "normalized_plan_sha256": str(normalized_plan_sha256),
        "normalized_plan_authority_sha256": str(
            normalized_plan_authority_sha256
        ),
        "outline_artifact_sha256": sources["outline"][1],
        "foundation_artifact_sha256": sources["foundation"][1],
        "materialization_ledger_sha256": sources["materializations"][1],
        "skeleton_artifact_sha256": sources["skeleton"][1],
        "compile_receipt_artifact_sha256": sources["compile_receipt"][1],
        "planner_prompt_metrics_sha256": sources["prompt_metrics"][1],
        "analysis_plan_artifact_sha256": sources["analysis_plan"][1],
        "normalized_plan_artifact_sha256": sources["normalized_plan"][1],
        "strict_transport_bound": bool(
            outline_schema_sha256
            and foundation_schema_sha256
            and all(
                item.structured_output_authority_sha256 for item in ordered_steps
            )
        ),
    }
    body["authority_sha256"] = canonical_sha256(body)
    authority = ProgressivePlanningAuthority.model_validate(body)
    _write_and_register_once(
        evidence,
        content=authority.model_dump_json(indent=2),
        evidence_id="progressive_planning_authority",
        description=(
            "Content-addressed Progressive Planner outline, per-step schema, "
            "compile, and normalized-plan authority root."
        ),
        source_path=Path(run_dir) / "progressive_planning_authority.json",
        inputs=(
            "progressive_plan_outline",
            "progressive_plan_foundation",
            "progressive_step_materializations",
            "progressive_plan_skeleton",
            "progressive_plan_compile_receipt",
            "planner_prompt_metrics",
            "analysis_plan",
            normalized_plan_evidence_id,
        ),
        producer="progressive_planning_authority",
        generation_mode="deterministic_skill",
        prompt_pack_version=prompt_pack_version,
    )
    return authority


__all__ = [
    "ProgressiveCompileFailureReplay",
    "ProgressiveCompileReplayAttempt",
    "ProgressiveCompilerFinding",
    "ProgressivePlanningArtifactError",
    "ProgressivePlanningArtifactPaths",
    "ProgressivePlanningAuthority",
    "ProgressivePlanningStepAuthority",
    "ProgressivePlannerCheckpointRecorder",
    "ProgressiveResumePersistenceReceipt",
    "load_progressive_compile_failure_replay",
    "load_progressive_planner_checkpoint_chain",
    "persist_progressive_compile_failure_replay",
    "persist_progressive_planner_checkpoint",
    "persist_progressive_planning_artifacts",
    "persist_progressive_planning_authority",
]
