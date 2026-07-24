"""Transactional sidecar lifecycle for the sealed step-result envelope.

This module owns the *producer* and *recovery* halves of the completed-step
envelope authority, and nothing else:

* :func:`prepare_step_result_envelope_sidecar` re-binds an already-sealed
  shadow snapshot to the final ``ok`` step status and builds the canonical
  sidecar payload plus its binding metadata.  It is pure -- no cohort, CSV,
  JSON, or artifact bytes are re-read.
* :func:`publish_step_result_envelope_sidecar` registers that payload as one
  evidence record with ``publish_aliases=False``.  The bytes land in the
  EvidenceStore's own content-addressed directory (outside the raw step
  output directory); the caller adds the returned alias to
  ``pending_success_aliases`` so the EXISTING
  ``StepEvidenceCommit.success_publication_transaction`` promotes it as part
  of the same durable generation.  No second transaction is introduced.
* :func:`load_current_step_result_envelope_sidecar` recovers the envelope for
  a current successful step.  It resolves ONLY through the committed alias
  table (never ``EvidenceStore.get``'s fuzzy prefix fallback), so an
  unpublished / rolled-back / legacy record is never recognised as current
  authority, and it fails closed on every producer, schema, status, binding,
  digest, symlink, or tamper mismatch.  The on-disk artifact is resolved
  through the shared descriptor-anchored :func:`verified_run_evidence_path`
  guard, which rejects a parent-directory symlink, a ``..`` / absolute /
  escaped relative path, a final symlink, a non-regular file, and a digest
  mismatch in one check -- a bare ``Path.is_symlink()`` is not sufficient.

The evidence identity binds the full step attempt: ``step_id`` + ``attempt_id``
+ ``checkpoint_id`` + ``script_evidence_id`` + ``content_sha256``.  A second
successful attempt of the same step/script/content therefore produces a *new*
record; the step-scoped alias re-points to the latest attempt, so a query for
the earlier attempt is stale and only the current attempt is recoverable.

The envelope stays ``shadow=True`` / ``paper_authorized=False``: a sidecar is
recovery metadata, never a grant of paper authority.  No downstream
consumer (registered-output validator, Writer, readiness, scorer, Jury,
figure/source-data) is switched here -- this module only establishes the
lifecycle a future consumer can adopt.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, Union

from easyicu.research_agent.schema import EvidenceRecord

from ..authority.runtime_artifacts import verified_run_evidence_path
from .result_envelope import (
    StepResultEnvelope,
    rebind_step_result_status,
    verify_step_result_envelope,
)

SIDECAR_SCHEMA_VERSION = "easyicu.step_result_envelope_sidecar/2"
SIDECAR_PRODUCER = "step_result_envelope_sidecar"
SIDECAR_EVIDENCE_KIND = "log"
SIDECAR_GENERATION_MODE = "system"

# Only a successful terminal step publishes a sidecar.  Any other terminal or
# in-flight status yields no sidecar (permissive: the step still commits).
SUCCESSFUL_TERMINAL_STATUS = "ok"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALIAS_PREFIX = "result_envelope_sidecar__"
_EVIDENCE_ID_PREFIX = "log_result_envelope_sidecar_"


def _safe_step_token(step_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", step_id).strip("_")


def _sidecar_alias(step_id: str) -> str:
    return f"{_ALIAS_PREFIX}{_safe_step_token(step_id)}"


def _sidecar_filename(step_id: str) -> str:
    return f"{_sidecar_alias(step_id)}.json"


def _sidecar_evidence_id(
    *,
    step_id: str,
    attempt_id: str,
    checkpoint_id: str,
    script_evidence_id: str,
    content_sha256: str,
) -> str:
    """Deterministic identity bound to the full step attempt.

    Artifact bytes alone are not an authority identity (two steps -- or two
    attempts of one step -- can emit an identical envelope shell), so the id
    binds the producing step, the exact attempt and checkpoint, and the script
    as well as the content digest.  A second successful attempt of the same
    step/script/content therefore yields a *different* id (a new record) rather
    than colliding with the first; a forger cannot fabricate a matching id
    without matching all five; and the loader re-derives and checks it.
    """

    token = hashlib.sha256(
        "\0".join(
            (
                step_id,
                attempt_id,
                checkpoint_id,
                script_evidence_id,
                content_sha256,
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"{_EVIDENCE_ID_PREFIX}{token}"


def _canonical_envelope_bytes(envelope: StepResultEnvelope) -> bytes:
    return (
        json.dumps(
            envelope.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class PreparedStepResultEnvelopeSidecar:
    """A terminal-bound envelope ready to publish inside the success commit."""

    envelope: StepResultEnvelope
    payload: bytes
    evidence_id: str
    alias: str
    filename: str
    metadata: Mapping[str, Any]

    @property
    def step_id(self) -> str:
        return str(self.metadata["step_id"])


def prepare_step_result_envelope_sidecar(
    *,
    snapshot_envelope: StepResultEnvelope | None,
    step_id: str,
    attempt_id: str,
    checkpoint_id: str,
    script_evidence_id: str,
    terminal_status: str,
) -> PreparedStepResultEnvelopeSidecar | None:
    """Rebind + serialise a sidecar payload, or ``None`` to publish nothing.

    Pure and fail-closed: returns ``None`` (no sidecar, step still commits)
    when the sealed snapshot is absent, the terminal status is not the
    successful terminal status, a binding coordinate is missing, the snapshot
    step identity disagrees, or the rebind rejects an inconsistent envelope.
    Never re-reads cohort/CSV/JSON/artifact bytes.
    """

    if snapshot_envelope is None:
        return None
    if terminal_status != SUCCESSFUL_TERMINAL_STATUS:
        return None
    step_id = str(step_id or "").strip()
    attempt_id = str(attempt_id or "").strip()
    checkpoint_id = str(checkpoint_id or "").strip()
    script_evidence_id = str(script_evidence_id or "").strip()
    if not (step_id and attempt_id and checkpoint_id and script_evidence_id):
        return None
    if snapshot_envelope.step_id != step_id:
        return None
    try:
        rebound = rebind_step_result_status(snapshot_envelope, status=terminal_status)
    except ValueError:
        return None
    # Type-level Literals already pin these; assert defensively so a future
    # schema change can never publish an authority-bearing sidecar.
    if rebound.shadow is not True or rebound.paper_authorized is not False:
        return None
    payload = _canonical_envelope_bytes(rebound)
    evidence_id = _sidecar_evidence_id(
        step_id=step_id,
        attempt_id=attempt_id,
        checkpoint_id=checkpoint_id,
        script_evidence_id=script_evidence_id,
        content_sha256=rebound.content_sha256,
    )
    metadata: dict[str, Any] = {
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "step_id": step_id,
        "attempt_id": attempt_id,
        "checkpoint_id": checkpoint_id,
        "script_evidence_id": script_evidence_id,
        "envelope_schema_version": rebound.schema_version,
        "content_sha256": rebound.content_sha256,
        "source_summary_sha256": rebound.source_summary_sha256,
        "terminal_status": terminal_status,
        "paper_authorized": False,
    }
    return PreparedStepResultEnvelopeSidecar(
        envelope=rebound,
        payload=payload,
        evidence_id=evidence_id,
        alias=_sidecar_alias(step_id),
        filename=_sidecar_filename(step_id),
        metadata=metadata,
    )


class _SidecarRegisterStore(Protocol):
    """Narrow EvidenceStore surface required to register a sidecar."""

    def register_text(
        self,
        *,
        kind: str,
        description: str,
        text: str,
        filename: str,
        produced_by_step: str | None = ...,
        script_evidence_id: str | None = ...,
        evidence_id: str | None = ...,
        aliases: Sequence[str] | None = ...,
        producer: str | None = ...,
        generation_mode: str | None = ...,
        metadata: dict[str, Any] | None = ...,
        publish_aliases: bool = ...,
    ) -> EvidenceRecord: ...


def publish_step_result_envelope_sidecar(
    prepared: PreparedStepResultEnvelopeSidecar,
    *,
    evidence_store: _SidecarRegisterStore,
) -> EvidenceRecord:
    """Register the sidecar bytes with ``publish_aliases=False``.

    The bytes are written to the EvidenceStore's own content-addressed
    directory -- outside the raw step output directory.  The alias is NOT
    made current here; the caller adds ``prepared.alias`` to the step's
    ``pending_success_aliases`` so the existing success transaction promotes
    it atomically with the numeric claims and result aliases.
    """

    return evidence_store.register_text(
        kind=SIDECAR_EVIDENCE_KIND,
        description=(
            "Sealed step-result envelope sidecar (shadow, non-authoritative) "
            f"for step {prepared.step_id}."
        ),
        text=prepared.payload.decode("utf-8"),
        filename=prepared.filename,
        produced_by_step=prepared.step_id,
        script_evidence_id=str(prepared.metadata["script_evidence_id"]),
        evidence_id=prepared.evidence_id,
        aliases=[prepared.alias],
        producer=SIDECAR_PRODUCER,
        generation_mode=SIDECAR_GENERATION_MODE,
        metadata=dict(prepared.metadata),
        publish_aliases=False,
    )


@dataclass(frozen=True)
class PublishedStepResultEnvelopeSidecar:
    """Identity a caller adds to the step's pending success aliases."""

    evidence_id: str
    alias: str


def publish_terminal_step_result_envelope_sidecar(
    *,
    snapshot_envelope: StepResultEnvelope | None,
    step_id: str,
    attempt_id: str,
    checkpoint_id: str,
    script_evidence_id: str,
    terminal_status: str,
    evidence_store: _SidecarRegisterStore,
) -> PublishedStepResultEnvelopeSidecar | None:
    """Rebind + register the terminal envelope sidecar in one call.

    This is the single seam the live success path invokes so the step
    orchestrator only wires the returned ``evidence_id``/``alias`` into its
    pending success aliases (promoted by the existing transaction).  Returns
    ``None`` -- publishing nothing -- whenever :func:`prepare_step_result_
    envelope_sidecar` fails closed.
    """

    prepared = prepare_step_result_envelope_sidecar(
        snapshot_envelope=snapshot_envelope,
        step_id=step_id,
        attempt_id=attempt_id,
        checkpoint_id=checkpoint_id,
        script_evidence_id=script_evidence_id,
        terminal_status=terminal_status,
    )
    if prepared is None:
        return None
    record = publish_step_result_envelope_sidecar(
        prepared, evidence_store=evidence_store
    )
    return PublishedStepResultEnvelopeSidecar(
        evidence_id=record.evidence_id, alias=prepared.alias
    )


@dataclass(frozen=True)
class StepResultEnvelopeSidecarQuery:
    """The current successful step's coordinates the loader binds against.

    ``attempt_id`` and ``checkpoint_id`` are required and non-empty: a caller
    must not omit them to bypass the current-attempt binding.  A query that
    names an older attempt of a re-run step is therefore stale, never a match.
    """

    step_id: str
    terminal_status: str
    script_evidence_id: str
    attempt_id: str
    checkpoint_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "step_id",
            "terminal_status",
            "script_evidence_id",
            "attempt_id",
            "checkpoint_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "step-result envelope sidecar query requires a non-empty "
                    f"{field_name}"
                )


@dataclass(frozen=True)
class LoadedStepResultEnvelopeSidecar:
    """A recovered, fully-verified current-step envelope."""

    envelope: StepResultEnvelope
    evidence_id: str
    terminal_status: str


@dataclass(frozen=True)
class StepResultEnvelopeSidecarUnavailable:
    """Fail-closed: no current envelope authority is recoverable."""

    reason: str


StepResultEnvelopeSidecarLoad = Union[
    LoadedStepResultEnvelopeSidecar,
    StepResultEnvelopeSidecarUnavailable,
]


class _SidecarLoadStore(Protocol):
    """Narrow EvidenceStore surface required to recover a sidecar."""

    root: Path

    def aliases(self) -> Mapping[str, str]: ...

    def records(self) -> Sequence[EvidenceRecord]: ...


def _unavailable(reason: str) -> StepResultEnvelopeSidecarUnavailable:
    return StepResultEnvelopeSidecarUnavailable(reason=reason)


def load_current_step_result_envelope_sidecar(
    *,
    evidence_store: _SidecarLoadStore,
    query: StepResultEnvelopeSidecarQuery,
) -> StepResultEnvelopeSidecarLoad:
    """Recover the envelope for the current successful step, or fail closed.

    Resolution is strictly through the committed alias table.  A sidecar that
    was never published, whose success transaction rolled back, or that
    belongs to a legacy checkpoint has no current alias and is reported
    :class:`StepResultEnvelopeSidecarUnavailable` -- it is never auto-promoted
    into current authority.  Every producer, schema, binding, status, digest,
    symlink, and tamper check must pass exactly.
    """

    step_id = str(query.step_id or "").strip()
    terminal_status = str(query.terminal_status or "").strip()
    script_evidence_id = str(query.script_evidence_id or "").strip()
    attempt_id = str(query.attempt_id or "").strip()
    checkpoint_id = str(query.checkpoint_id or "").strip()
    if not step_id or not script_evidence_id or not attempt_id or not checkpoint_id:
        return _unavailable("incomplete_query")
    if terminal_status != SUCCESSFUL_TERMINAL_STATUS:
        return _unavailable("non_successful_terminal_status")

    alias = _sidecar_alias(step_id)
    # Committed aliases only -- deliberately NOT EvidenceStore.get, whose
    # prefix fallback could bind an ambiguous or unpublished record.
    evidence_id = evidence_store.aliases().get(alias)
    if not evidence_id:
        return _unavailable("no_committed_alias")
    record = next(
        (r for r in evidence_store.records() if r.evidence_id == evidence_id),
        None,
    )
    if record is None:
        return _unavailable("committed_alias_without_record")

    if record.kind != SIDECAR_EVIDENCE_KIND:
        return _unavailable("kind_mismatch")
    if (record.producer or "") != SIDECAR_PRODUCER:
        return _unavailable("producer_mismatch")
    if (record.produced_by_step or "") != step_id:
        return _unavailable("step_mismatch")

    md = record.metadata or {}
    if md.get("sidecar_schema_version") != SIDECAR_SCHEMA_VERSION:
        return _unavailable("sidecar_schema_mismatch")
    if str(md.get("step_id") or "") != step_id:
        return _unavailable("metadata_step_mismatch")
    if str(md.get("script_evidence_id") or "") != script_evidence_id:
        return _unavailable("script_mismatch")
    if md.get("terminal_status") != terminal_status:
        return _unavailable("terminal_status_mismatch")
    if md.get("paper_authorized") is not False:
        return _unavailable("paper_authority_asserted")
    if str(md.get("attempt_id") or "") != attempt_id:
        return _unavailable("attempt_mismatch")
    if str(md.get("checkpoint_id") or "") != checkpoint_id:
        return _unavailable("checkpoint_mismatch")

    # Descriptor-anchored resolution: rejects a parent-directory symlink, a
    # ``..`` / absolute / escaped relative path, a final symlink, a non-regular
    # file, and a digest mismatch in one guard.  ``record.relative_path`` is
    # ``evidence/<id>__<file>`` relative to the store root, which is exactly the
    # run directory ``verified_run_evidence_path`` anchors under.
    verified_path = verified_run_evidence_path(evidence_store.root, record)
    if verified_path is None:
        return _unavailable("artifact_path_unverified")
    # ``verified_run_evidence_path`` already digest-checked the file, but we must
    # read it again to parse the envelope.  Treat that second read as untrusted:
    # a read error, or bytes that no longer match the committed digest (a file
    # swapped in the TOCTOU window), returns typed unavailable rather than
    # letting an OSError escape or a changed payload be parsed as authority.
    try:
        raw = verified_path.read_bytes()
    except OSError:
        return _unavailable("artifact_read_failed")
    if hashlib.sha256(raw).hexdigest() != record.sha256:
        return _unavailable("artifact_digest_mismatch")

    try:
        envelope = StepResultEnvelope.model_validate(json.loads(raw))
    except (ValueError, TypeError):
        return _unavailable("unparseable_envelope")
    if not verify_step_result_envelope(envelope):
        return _unavailable("envelope_digest_invalid")
    if envelope.step_id != step_id:
        return _unavailable("envelope_step_mismatch")
    if envelope.status != terminal_status:
        return _unavailable("envelope_status_mismatch")
    if envelope.shadow is not True or envelope.paper_authorized is not False:
        return _unavailable("envelope_authority_asserted")
    content_sha256 = envelope.content_sha256
    if str(md.get("content_sha256") or "") != content_sha256:
        return _unavailable("content_digest_mismatch")
    if md.get("source_summary_sha256") != envelope.source_summary_sha256:
        return _unavailable("source_digest_mismatch")
    if str(md.get("envelope_schema_version") or "") != envelope.schema_version:
        return _unavailable("envelope_schema_mismatch")
    # Exact evidence-id binding: re-derive from the recovered content + the
    # full step/attempt/checkpoint/script coordinates and require the committed
    # id to match.  A stale attempt therefore cannot resolve even if its alias
    # somehow survived.
    expected_evidence_id = _sidecar_evidence_id(
        step_id=step_id,
        attempt_id=attempt_id,
        checkpoint_id=checkpoint_id,
        script_evidence_id=script_evidence_id,
        content_sha256=content_sha256,
    )
    if record.evidence_id != expected_evidence_id:
        return _unavailable("evidence_id_binding_mismatch")

    return LoadedStepResultEnvelopeSidecar(
        envelope=envelope,
        evidence_id=record.evidence_id,
        terminal_status=terminal_status,
    )


__all__ = [
    "LoadedStepResultEnvelopeSidecar",
    "PreparedStepResultEnvelopeSidecar",
    "PublishedStepResultEnvelopeSidecar",
    "SIDECAR_EVIDENCE_KIND",
    "SIDECAR_PRODUCER",
    "SIDECAR_SCHEMA_VERSION",
    "SUCCESSFUL_TERMINAL_STATUS",
    "StepResultEnvelopeSidecarLoad",
    "StepResultEnvelopeSidecarQuery",
    "StepResultEnvelopeSidecarUnavailable",
    "load_current_step_result_envelope_sidecar",
    "prepare_step_result_envelope_sidecar",
    "publish_step_result_envelope_sidecar",
    "publish_terminal_step_result_envelope_sidecar",
]
