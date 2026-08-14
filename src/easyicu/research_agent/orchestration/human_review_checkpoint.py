"""Durable, digest-bound pause handoff for human plan review.

This is intentionally narrower than a generic run execution context.  It owns
only the boundary between a completed Plan phase and the first Execute side
effect.  Live providers, locks, evidence stores and runners are reconstructed
by the pipeline from their existing owners; this file persists the immutable
coordinates needed to prove that reconstruction is the state a human saw.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..canonical_json import canonical_sha256
from .workflow import HumanReviewRequest


CHECKPOINT_FILENAME = "human_review_checkpoint.json"
DEFAULT_CHECKPOINT_TTL = timedelta(days=7)
_MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024


class HumanReviewCheckpointError(RuntimeError):
    """Stable fail-closed error for an unusable durable review pause."""

    reason_code = "human_review_checkpoint_invalid"


class HumanReviewCheckpointExpired(HumanReviewCheckpointError):
    reason_code = "human_review_checkpoint_expired"


class HumanReviewCheckpointConsumed(HumanReviewCheckpointError):
    reason_code = "human_review_checkpoint_consumed"


class HumanReviewCheckpoint(BaseModel):
    """Complete typed coordinates for reconstructing one Plan-phase pause."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.human_review_checkpoint/1"] = (
        "easyicu.human_review_checkpoint/1"
    )
    state: Literal["pending", "consumed", "completed", "failed"] = "pending"
    run_id: str = Field(min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=200)
    created_at: str
    expires_at: str
    pipeline_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_identity: dict[str, Any]
    environment_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    llm_signature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_input_capsule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_activation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_capabilities: tuple[str, ...]
    runtime_capabilities_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_bundle: Optional[dict[str, Any]] = None
    runtime_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requests: tuple[HumanReviewRequest, ...]
    request_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_handoff: dict[str, Any]
    plan_handoff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_coordinates: dict[str, Any]
    execution_coordinates_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    consumed_decision_sha256: Optional[str] = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _digests_bind_payloads(self) -> "HumanReviewCheckpoint":
        checks = (
            (
                "environment_identity_sha256",
                self.environment_identity_sha256,
                self.environment_identity,
            ),
            (
                "runtime_capabilities_sha256",
                self.runtime_capabilities_sha256,
                list(self.runtime_capabilities),
            ),
            ("runtime_bundle_sha256", self.runtime_bundle_sha256, self.runtime_bundle),
            (
                "request_set_sha256",
                self.request_set_sha256,
                [item.model_dump(mode="json") for item in self.requests],
            ),
            ("plan_handoff_sha256", self.plan_handoff_sha256, self.plan_handoff),
            (
                "execution_coordinates_sha256",
                self.execution_coordinates_sha256,
                self.execution_coordinates,
            ),
        )
        for label, observed, payload in checks:
            if canonical_sha256(payload) != observed:
                raise ValueError(f"{label} does not bind its checkpoint payload")
        unsigned = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if canonical_sha256(unsigned) != self.checkpoint_sha256:
            raise ValueError("checkpoint_sha256 does not bind checkpoint contents")
        if not self.requests:
            raise ValueError("human-review checkpoint must contain review requests")
        return self

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        pipeline_config_sha256: str,
        environment_identity: Mapping[str, Any],
        llm_signature_sha256: str,
        run_input_capsule_sha256: str,
        capability_activation_sha256: str,
        runtime_capabilities: Sequence[str],
        runtime_bundle: Optional[Mapping[str, Any]],
        requests: Sequence[HumanReviewRequest],
        plan_handoff: Mapping[str, Any],
        execution_coordinates: Mapping[str, Any],
        now: Optional[datetime] = None,
        ttl: timedelta = DEFAULT_CHECKPOINT_TTL,
    ) -> "HumanReviewCheckpoint":
        instant = now or datetime.now(timezone.utc)
        if instant.tzinfo is None:
            raise ValueError("checkpoint creation time must be timezone-aware")
        request_payload = [item.model_dump(mode="json") for item in requests]
        capabilities = tuple(sorted({str(item) for item in runtime_capabilities}))
        bundle = dict(runtime_bundle) if runtime_bundle is not None else None
        body: dict[str, Any] = {
            "schema_version": "easyicu.human_review_checkpoint/1",
            "state": "pending",
            "run_id": str(run_id),
            "thread_id": str(run_id),
            "created_at": instant.isoformat(),
            "expires_at": (instant + ttl).isoformat(),
            "pipeline_config_sha256": str(pipeline_config_sha256),
            "environment_identity": dict(environment_identity),
            "environment_identity_sha256": canonical_sha256(environment_identity),
            "llm_signature_sha256": str(llm_signature_sha256),
            "run_input_capsule_sha256": str(run_input_capsule_sha256),
            "capability_activation_sha256": str(capability_activation_sha256),
            "runtime_capabilities": capabilities,
            "runtime_capabilities_sha256": canonical_sha256(list(capabilities)),
            "runtime_bundle": bundle,
            "runtime_bundle_sha256": canonical_sha256(bundle),
            "requests": request_payload,
            "request_set_sha256": canonical_sha256(request_payload),
            "plan_handoff": dict(plan_handoff),
            "plan_handoff_sha256": canonical_sha256(plan_handoff),
            "execution_coordinates": dict(execution_coordinates),
            "execution_coordinates_sha256": canonical_sha256(
                execution_coordinates
            ),
            "consumed_decision_sha256": None,
        }
        body["checkpoint_sha256"] = canonical_sha256(body)
        return cls.model_validate(body)

    def transitioned(
        self,
        state: Literal["consumed", "completed", "failed"],
        *,
        decision_sha256: Optional[str] = None,
    ) -> "HumanReviewCheckpoint":
        body = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        body["state"] = state
        if decision_sha256 is not None:
            body["consumed_decision_sha256"] = str(decision_sha256)
        body["checkpoint_sha256"] = canonical_sha256(body)
        return type(self).model_validate(body)


def checkpoint_path(run_dir: Path) -> Path:
    return Path(run_dir) / CHECKPOINT_FILENAME


def write_checkpoint(path: Path, checkpoint: HumanReviewCheckpoint) -> Path:
    """Atomically publish one private checkpoint and fsync its directory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            checkpoint.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    if len(raw) > _MAX_CHECKPOINT_BYTES:
        raise HumanReviewCheckpointError("human-review checkpoint exceeds size limit")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_checkpoint(
    path: Path,
    *,
    require_pending: bool = True,
    now: Optional[datetime] = None,
) -> HumanReviewCheckpoint:
    target = Path(path)
    try:
        if target.is_symlink() or not target.is_file():
            raise HumanReviewCheckpointError("human-review checkpoint is missing")
        raw = target.read_bytes()
        if len(raw) > _MAX_CHECKPOINT_BYTES:
            raise HumanReviewCheckpointError(
                "human-review checkpoint exceeds size limit"
            )
        checkpoint = HumanReviewCheckpoint.model_validate_json(raw)
    except HumanReviewCheckpointError:
        raise
    except Exception as exc:
        raise HumanReviewCheckpointError(
            "human-review checkpoint is corrupt or invalid"
        ) from exc
    if require_pending and checkpoint.state != "pending":
        raise HumanReviewCheckpointConsumed(
            f"human-review checkpoint is already {checkpoint.state}"
        )
    instant = now or datetime.now(timezone.utc)
    try:
        expires = datetime.fromisoformat(checkpoint.expires_at)
    except ValueError as exc:  # pragma: no cover - model already digest checked
        raise HumanReviewCheckpointError("checkpoint expiry is invalid") from exc
    if instant >= expires:
        raise HumanReviewCheckpointExpired("human-review checkpoint has expired")
    return checkpoint


__all__ = [
    "CHECKPOINT_FILENAME",
    "DEFAULT_CHECKPOINT_TTL",
    "HumanReviewCheckpoint",
    "HumanReviewCheckpointConsumed",
    "HumanReviewCheckpointError",
    "HumanReviewCheckpointExpired",
    "checkpoint_path",
    "load_checkpoint",
    "write_checkpoint",
]
