"""Host-owned, one-use authority for preparation review decisions.

Only a host action already recorded for the bound researcher conversation can
mint a durable receipt.  A self-hashed model/client checkpoint is not a human
decision.  This receipt grants preparation only, never scientific execution.
The caller must still validate source, scope, code and producer closure.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from easyicu import state_paths
from easyicu.research_agent.authority.filesystem import AnchoredDirectory, AuthorityFilesystemError
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision
from easyicu.webserver.host_action_contracts import (
    HostActionRequest, RawPreparationReviewDecision, decision_key, host_action_id,
)

_HEX = re.compile(r"^[0-9a-f]{64}$")


class RawPreparationHostBinding(BaseModel):
    """Host-owned coordinates; values alone do not authorize preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    project_id: str = Field(min_length=1, max_length=160)
    session_id: str = Field(min_length=1, max_length=160)
    study_context_id: str = Field(min_length=1, max_length=160)
    run_id: str = Field(min_length=1, max_length=160)
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")


def _private_directory() -> Path:
    root = state_paths.state_root() / "private-preparation-decisions"
    if root.is_symlink():
        raise ValueError("raw_preparation_host_directory_invalid")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise ValueError("raw_preparation_host_directory_invalid")
    # Opening pins and rejects linked ancestors on supported platforms.
    with AnchoredDirectory.open(root) as directory:
        directory.assert_still_selected()
    return root


def _publish_once(directory: AnchoredDirectory, name: str, data: bytes) -> None:
    temp_name, descriptor = directory.create_temporary(stem=name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        directory.assert_still_selected()
        directory.publish_temporary_no_clobber(temp_name, name)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        directory.unlink(temp_name, missing_ok=True)


def _host_key() -> bytes:
    root = _private_directory()
    with AnchoredDirectory.open(root) as directory:
        try:
            _publish_once(directory, "host.key", secrets.token_bytes(32))
        except FileExistsError:
            pass
        with directory.open_regular("host.key") as stream:
            info = os.fstat(stream.fileno())
            if info.st_mode & 0o077 or info.st_uid != os.getuid():
                raise ValueError("raw_preparation_host_key_invalid")
            key = stream.read(33)
        directory.assert_still_selected()
    if len(key) != 32:
        raise ValueError("raw_preparation_host_key_invalid")
    return key


def _signature(body: Mapping[str, Any]) -> str:
    return hmac.new(_host_key(), canonical_json(dict(body)).encode(), hashlib.sha256).hexdigest()


def record_host_review(*, host_action: HostActionRequest, pending: RawPreparationCheckpoint,
                       decision: str, study_context_id: str, job_id: str,
                       run_id: str, scientific_configuration_sha256: str) -> RawPreparationCheckpoint:
    """Seal the decision already accepted and recorded by the host action owner.

    This function cannot offer an action, authenticate a browser, or start a job.
    The host-action submission must first match the current offered decision,
    validate the conversation binding and record the decision's tagged job.
    A rejected or missing audit record never authorizes data access.
    """
    from easyicu.webserver import jobs
    from easyicu.webserver.agent_pipeline_runs import server_reviewer_identity
    from easyicu.webserver.pi_copilot.service import get_pi_copilot_service

    if (not isinstance(host_action, HostActionRequest)
            or host_action.action_code != "review_raw_preparation"
            or not isinstance(host_action.decision, RawPreparationReviewDecision)
            or decision not in {"approved", "rejected"} or pending.state != "pending"):
        raise ValueError("raw_preparation_host_review_invalid")
    selected = host_action.decision
    if (selected.checkpoint_sha256 != pending.checkpoint_sha256
            or selected.review_authority_sha256 != pending.request.authority_sha256
            or selected.run_id != run_id
            or selected.scientific_configuration_sha256 != scientific_configuration_sha256):
        raise ValueError("raw_preparation_host_review_mismatch")
    owner = get_pi_copilot_service()
    owner.host_action_session(host_action.session_id, project_id=host_action.project_id,
                              study_context_id=study_context_id)
    key = decision_key(selected, review_decision=decision)
    job = jobs.MANAGER.get(job_id)
    tag = getattr(job, "host_action", None) if job is not None else None
    expected = {"session_id": host_action.session_id, "project_id": host_action.project_id,
                "study_context_id": study_context_id, "action_code": host_action.action_code,
                "decision_key": key, "action_id": host_action_id(host_action.session_id, host_action.action_code, key)}
    if tag is None or any(getattr(tag, name, None) != value for name, value in expected.items()):
        raise ValueError("raw_preparation_host_action_unrecorded")
    row = owner.host_action_child_job(host_action.session_id, project_id=host_action.project_id,
                                      action_key=key)
    if not row or row.get("child_job_id") != job_id:
        raise ValueError("raw_preparation_host_action_unrecorded")
    scope_values = [v for k, v in pending.request.payload.items() if k != "plan_review_authority"]
    if len(scope_values) != 1 or canonical_sha256(scope_values[0]) != selected.scope_sha256:
        raise ValueError("raw_preparation_host_scope_mismatch")
    reviewer = server_reviewer_identity()
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("raw_preparation_host_reviewer_missing")
    decided = HumanReviewDecision(review_id=pending.request.review_id,
                                  authority_sha256=pending.request.authority_sha256,
                                  decision=decision, reviewer=reviewer,
                                  decided_at=datetime.now(timezone.utc).isoformat())
    result = pending.decide(decided)
    body = {"schema_version": "easyicu.raw-preparation-host-decision/1",
            "actor": "host_user", "reviewer": reviewer, "decision": decision,
            "project_id": host_action.project_id, "session_id": host_action.session_id,
            "study_context_id": study_context_id, "run_id": selected.run_id,
            "scientific_configuration_sha256": scientific_configuration_sha256,
            "nonce": selected.nonce, "scope_sha256": selected.scope_sha256,
            "pending_checkpoint_sha256": pending.checkpoint_sha256,
            "decided_checkpoint_sha256": result.checkpoint_sha256,
            "request_authority_sha256": pending.request.authority_sha256,
            "decision_key": key, "job_id": job_id, "expires_at": pending.expires_at}
    receipt = {**body, "signature": _signature(body)}
    with AnchoredDirectory.open(_private_directory()) as directory:
        try:
            _publish_once(directory, selected.nonce + ".json", canonical_json(receipt).encode())
        except FileExistsError as exc:
            raise ValueError("raw_preparation_host_nonce_replayed") from exc
    return result


def verify_host_review(*, checkpoint: RawPreparationCheckpoint, project_id: str, session_id: str,
                       study_context_id: str, run_id: str, nonce: str,
                       scope: Mapping[str, Any], now: datetime | None = None) -> Mapping[str, Any]:
    """Verify the host-sealed user decision; no preparation side effect occurs."""
    if not isinstance(nonce, str) or not _HEX.fullmatch(nonce):
        raise ValueError("raw_preparation_host_nonce_invalid")
    try:
        with AnchoredDirectory.open(_private_directory()) as directory:
            payload = directory.read_bytes(nonce + ".json", max_bytes=64 * 1024)
            directory.assert_still_selected()
        receipt = json.loads(payload)
        if not isinstance(receipt, dict):
            raise ValueError
        signature = receipt.pop("signature")
        if not isinstance(signature, str) or not hmac.compare_digest(_signature(receipt), signature):
            raise ValueError
    except (AuthorityFilesystemError, OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("raw_preparation_host_receipt_invalid") from exc
    instant = now or datetime.now(timezone.utc)
    try:
        expires = datetime.fromisoformat(str(receipt["expires_at"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("raw_preparation_host_receipt_invalid") from exc
    if instant.tzinfo is None or expires.tzinfo is None or instant >= expires:
        raise ValueError("raw_preparation_host_receipt_expired")
    expected = {"schema_version": "easyicu.raw-preparation-host-decision/1", "actor": "host_user",
                "decision": "approved", "project_id": project_id, "session_id": session_id,
                "study_context_id": study_context_id, "run_id": run_id, "nonce": nonce,
                "scope_sha256": canonical_sha256(dict(scope)),
                "decided_checkpoint_sha256": checkpoint.checkpoint_sha256,
                "request_authority_sha256": checkpoint.request.authority_sha256}
    if checkpoint.state != "approved" or any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("raw_preparation_host_binding_mismatch")
    with AnchoredDirectory.open(_private_directory()) as directory:
        if not directory.is_absent(nonce + ".used"):
            raise ValueError("raw_preparation_host_nonce_replayed")
        directory.assert_still_selected()
    return receipt


def consume_host_review(*, checkpoint: RawPreparationCheckpoint, project_id: str, session_id: str,
                        study_context_id: str, run_id: str, nonce: str,
                        scope: Mapping[str, Any], now: datetime | None = None) -> None:
    """Atomically consume an approved decision before the first source read.

    A failed/crashed producer cannot simply replay it. Recovery needs a
    separate durable lifecycle and a cleanup proof, which is not supplied here.
    """
    receipt = verify_host_review(checkpoint=checkpoint, project_id=project_id, session_id=session_id,
                                 study_context_id=study_context_id, run_id=run_id, nonce=nonce,
                                 scope=scope, now=now)
    marker = canonical_json({"decision_sha256": canonical_sha256(receipt), "nonce": nonce}).encode()
    with AnchoredDirectory.open(_private_directory()) as directory:
        try:
            _publish_once(directory, nonce + ".used", marker)
        except FileExistsError as exc:
            raise ValueError("raw_preparation_host_nonce_replayed") from exc
