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


class HumanReviewCheckpointPhaseUncertain(HumanReviewCheckpointError):
    reason_code = "human_review_checkpoint_phase_uncertain"


class HumanReviewCheckpoint(BaseModel):
    """Complete typed coordinates for reconstructing one Plan-phase pause."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "easyicu.human_review_checkpoint/1",
        "easyicu.human_review_checkpoint/2",
        "easyicu.human_review_checkpoint/3",
    ] = (
        "easyicu.human_review_checkpoint/3"
    )
    state: Literal[
        "pending",
        "approved_pending_execution",
        "executing",
        "write_in_progress",
        "finalize_in_progress",
        "rejected",
        "consumed",
        "completed",
        "failed",
    ] = "pending"
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
    approved_decisions: tuple[dict[str, Any], ...] = ()
    approved_decision_records: tuple[dict[str, Any], ...] = ()
    approved_decisions_sha256: Optional[str] = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    execution_start_receipt: Optional[dict[str, Any]] = None
    execution_start_receipt_sha256: Optional[str] = Field(
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
        decision_fields = (
            bool(self.approved_decisions),
            bool(self.approved_decision_records),
            self.approved_decisions_sha256 is not None,
            self.consumed_decision_sha256 is not None,
        )
        if any(decision_fields) and not all(decision_fields):
            raise ValueError("durable decisions must be present as a complete set")
        if self.approved_decisions_sha256 is not None:
            approved_payload = {
                "decisions": list(self.approved_decisions),
                "records": list(self.approved_decision_records),
            }
            if canonical_sha256(approved_payload) != self.approved_decisions_sha256:
                raise ValueError(
                    "approved_decisions_sha256 does not bind approved decisions"
                )
        if self.execution_start_receipt_sha256 is not None:
            if canonical_sha256(self.execution_start_receipt) != (
                self.execution_start_receipt_sha256
            ):
                raise ValueError(
                    "execution_start_receipt_sha256 does not bind its receipt"
                )
        if self.schema_version.endswith(("/2", "/3")) and self.state in {
            "approved_pending_execution",
            "executing",
            "write_in_progress",
            "finalize_in_progress",
            "rejected",
            "completed",
        }:
            if (
                not self.approved_decisions
                or not self.approved_decision_records
                or self.approved_decisions_sha256 is None
                or self.consumed_decision_sha256 is None
            ):
                raise ValueError(
                    "approved checkpoint state requires exact durable decisions"
                )
        if self.schema_version.endswith(("/2", "/3")) and self.state in {
            "executing",
            "write_in_progress",
            "finalize_in_progress",
            "completed",
        } and (
            self.execution_start_receipt is None
            or self.execution_start_receipt_sha256 is None
        ):
            raise ValueError("executing checkpoint state requires a start receipt")
        unsigned = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if self.schema_version in {
            "easyicu.human_review_checkpoint/1",
            "easyicu.human_review_checkpoint/2",
        }:
            # Older checkpoints predate later optional fields. Preserve their
            # original canonical bytes rather than hashing injected defaults.
            unsigned = {
                key: value
                for key, value in unsigned.items()
                if key in self.model_fields_set
            }
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
            "schema_version": "easyicu.human_review_checkpoint/3",
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
            "approved_decisions": [],
            "approved_decision_records": [],
            "approved_decisions_sha256": None,
            "execution_start_receipt": None,
            "execution_start_receipt_sha256": None,
        }
        body["checkpoint_sha256"] = canonical_sha256(body)
        return cls.model_validate(body)

    def transitioned(
        self,
        state: Literal[
            "approved_pending_execution",
            "executing",
            "write_in_progress",
            "finalize_in_progress",
            "rejected",
            "consumed",
            "completed",
            "failed",
        ],
        *,
        decision_sha256: Optional[str] = None,
    ) -> "HumanReviewCheckpoint":
        body = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        body["state"] = state
        if decision_sha256 is not None:
            body["consumed_decision_sha256"] = str(decision_sha256)
        body["checkpoint_sha256"] = canonical_sha256(body)
        return type(self).model_validate(body)

    def approved(
        self,
        *,
        decisions: Sequence[Mapping[str, Any]],
        decision_records: Sequence[Mapping[str, Any]],
        decision_sha256: str,
    ) -> "HumanReviewCheckpoint":
        """Persist the exact approved decision set before execution starts."""

        decision_payload = tuple(dict(item) for item in decisions)
        record_payload = tuple(dict(item) for item in decision_records)
        if canonical_sha256(list(decision_payload)) != str(decision_sha256):
            raise HumanReviewCheckpointError(
                "approved decision digest does not bind decision payloads"
            )
        if self.state == "approved_pending_execution":
            if (
                self.consumed_decision_sha256 == str(decision_sha256)
                and self.approved_decisions == decision_payload
                and self.approved_decision_records == record_payload
            ):
                return self
            raise HumanReviewCheckpointConsumed(
                "checkpoint was approved with a different decision set"
            )
        if self.state != "pending":
            raise HumanReviewCheckpointConsumed(
                f"checkpoint cannot be approved from state {self.state!r}"
            )
        body = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        body["state"] = "approved_pending_execution"
        body["consumed_decision_sha256"] = str(decision_sha256)
        body["approved_decisions"] = list(decision_payload)
        body["approved_decision_records"] = list(record_payload)
        body["approved_decisions_sha256"] = canonical_sha256(
            {
                "decisions": list(decision_payload),
                "records": list(record_payload),
            }
        )
        body["checkpoint_sha256"] = canonical_sha256(body)
        return type(self).model_validate(body)

    def decision_recorded(
        self,
        *,
        decisions: Sequence[Mapping[str, Any]],
        decision_records: Sequence[Mapping[str, Any]],
        decision_sha256: str,
    ) -> "HumanReviewCheckpoint":
        """Durably stage exact decisions before writing their evidence files."""

        decision_payload = tuple(dict(item) for item in decisions)
        record_payload = tuple(dict(item) for item in decision_records)
        if canonical_sha256(list(decision_payload)) != str(decision_sha256):
            raise HumanReviewCheckpointError(
                "recorded decision digest does not bind decision payloads"
            )
        if not decision_payload or not record_payload:
            raise HumanReviewCheckpointError("recorded decision set is empty")
        if self.approved_decisions_sha256 is not None:
            if (
                self.consumed_decision_sha256 == str(decision_sha256)
                and self.approved_decisions == decision_payload
                and self.approved_decision_records == record_payload
            ):
                return self
            raise HumanReviewCheckpointConsumed(
                "checkpoint already records a different decision set"
            )
        if self.state != "pending":
            raise HumanReviewCheckpointConsumed(
                f"checkpoint cannot record decisions from state {self.state!r}"
            )
        body = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        body["consumed_decision_sha256"] = str(decision_sha256)
        body["approved_decisions"] = list(decision_payload)
        body["approved_decision_records"] = list(record_payload)
        body["approved_decisions_sha256"] = canonical_sha256(
            {
                "decisions": list(decision_payload),
                "records": list(record_payload),
            }
        )
        body["checkpoint_sha256"] = canonical_sha256(body)
        return type(self).model_validate(body)

    def decision_committed(self) -> "HumanReviewCheckpoint":
        """Commit staged decisions after their evidence has been persisted."""

        if self.state in {"approved_pending_execution", "rejected"}:
            return self
        if self.state != "pending" or self.approved_decisions_sha256 is None:
            raise HumanReviewCheckpointError(
                "checkpoint has no staged durable decision to commit"
            )
        rejected = any(
            str(record.get("decision") or "") == "rejected"
            for record in self.approved_decision_records
        )
        return self.transitioned(
            "rejected" if rejected else "approved_pending_execution"
        )

    def execution_started(
        self,
        *,
        now: Optional[datetime] = None,
    ) -> "HumanReviewCheckpoint":
        """Bind the durable execution start before the first Execute side effect."""

        if self.state == "executing":
            return self
        if self.state != "approved_pending_execution":
            raise HumanReviewCheckpointConsumed(
                f"checkpoint cannot start execution from state {self.state!r}"
            )
        instant = now or datetime.now(timezone.utc)
        receipt = {
            "schema_version": "easyicu.human_review_execution_start/1",
            "run_id": self.run_id,
            "approved_plan_handoff_sha256": self.plan_handoff_sha256,
            "decision_set_sha256": self.consumed_decision_sha256,
            "execution_coordinates_sha256": self.execution_coordinates_sha256,
            "started_at": instant.isoformat(),
        }
        body = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        body["state"] = "executing"
        body["execution_start_receipt"] = receipt
        body["execution_start_receipt_sha256"] = canonical_sha256(receipt)
        body["checkpoint_sha256"] = canonical_sha256(body)
        return type(self).model_validate(body)

    def execution_phase_started(
        self,
        state: Literal["write_in_progress", "finalize_in_progress"],
    ) -> "HumanReviewCheckpoint":
        """Persist an irreversible post-analysis phase before its side effects."""

        allowed_from = {
            "write_in_progress": {"executing", "write_in_progress"},
            "finalize_in_progress": {"write_in_progress", "finalize_in_progress"},
        }
        if self.state == state:
            return self
        if self.state not in allowed_from[state]:
            raise HumanReviewCheckpointConsumed(
                f"checkpoint cannot enter {state!r} from state {self.state!r}"
            )
        return self.transitioned(state)


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


def completed_review_authorizes_exact_retry(
    path: Path,
    *,
    pipeline_config_sha256: str,
    run_input_capsule_sha256: str,
    plan_payload: Mapping[str, Any],
) -> bool:
    """Verify that a completed approval covers one exact execution retry.

    A failed execution resumes through :meth:`ResearchAgentPipeline.run` so it
    can reuse the normal checkpoint and per-step ledger.  Re-entering the Plan
    phase must not manufacture a second approval for the same plan: doing so
    both misrepresents the operator's action and collides with the immutable
    decision evidence.  This verifier is the fail-closed bridge between the
    completed review checkpoint and that execution-only retry.

    ``False`` means the checkpoint is not a completed execution approval and
    the normal review workflow still applies.  A completed checkpoint whose
    authority no longer matches raises instead of silently falling back to a
    new, visually indistinguishable approval.
    """

    checkpoint = load_checkpoint(path, require_pending=False)
    if checkpoint.state != "completed":
        return False
    if checkpoint.pipeline_config_sha256 != str(pipeline_config_sha256):
        raise HumanReviewCheckpointError(
            "completed review pipeline configuration changed before retry"
        )
    if checkpoint.run_input_capsule_sha256 != str(run_input_capsule_sha256):
        raise HumanReviewCheckpointError(
            "completed review run-input capsule changed before retry"
        )
    reviewed_plan = checkpoint.plan_handoff.get("plan")
    if not isinstance(reviewed_plan, Mapping) or canonical_sha256(
        reviewed_plan
    ) != canonical_sha256(plan_payload):
        raise HumanReviewCheckpointError(
            "completed review analysis plan changed before retry"
        )
    if checkpoint.execution_start_receipt is None:
        raise HumanReviewCheckpointError(
            "completed review has no execution-start receipt"
        )
    receipt = checkpoint.execution_start_receipt
    if (
        str(receipt.get("run_id") or "") != checkpoint.run_id
        or str(receipt.get("approved_plan_handoff_sha256") or "")
        != checkpoint.plan_handoff_sha256
        or str(receipt.get("decision_set_sha256") or "")
        != str(checkpoint.consumed_decision_sha256 or "")
    ):
        raise HumanReviewCheckpointError(
            "completed review execution-start receipt is not bound to the checkpoint"
        )
    requests = {item.review_id: item for item in checkpoint.requests}
    decisions = {str(item.get("review_id") or ""): item for item in checkpoint.approved_decisions}
    if not requests or set(decisions) != set(requests):
        raise HumanReviewCheckpointError(
            "completed review decision set does not cover every request"
        )
    for review_id, request in requests.items():
        decision = decisions[review_id]
        if (
            str(decision.get("decision") or "") != "approved"
            or str(decision.get("authority_sha256") or "")
            != request.authority_sha256
        ):
            raise HumanReviewCheckpointError(
                "completed review decision does not approve its exact request"
            )
    return True


__all__ = [
    "CHECKPOINT_FILENAME",
    "DEFAULT_CHECKPOINT_TTL",
    "HumanReviewCheckpoint",
    "HumanReviewCheckpointConsumed",
    "HumanReviewCheckpointError",
    "HumanReviewCheckpointExpired",
    "HumanReviewCheckpointPhaseUncertain",
    "checkpoint_path",
    "completed_review_authorizes_exact_retry",
    "load_checkpoint",
    "write_checkpoint",
]
