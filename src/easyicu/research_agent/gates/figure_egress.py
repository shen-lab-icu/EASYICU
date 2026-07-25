"""Authorization gate for sending rendered figure bytes to an external VLM.

Ordinary agent prompts never carry raw records: they are built from
``outbound_safe_context_payload()``, a deny-by-default projection that keeps
variable definitions, cohort sizes and aggregate statistics and drops
everything else. VLM visual QA does not go through that projection — it reads
the PNG/SVG off disk and base64s the whole file into a chat completion.

A figure is not automatically aggregate-only. Generated code routinely renders
per-patient trajectories, small-cell strata, rare category labels and local
filesystem paths into figure text. Provider-level authorization answers "may
this client talk to that URL", not "may this *image* leave the machine", so
this module adds the missing question.

The gate is deny-by-default for external destinations and requires, per image:

* the run explicitly enabled external figure upload;
* the path resolves inside the current run directory;
* it matches a registered ``kind="figure"`` evidence record;
* the file's SHA-256 still equals the registered digest;
* the record belongs to the active checkpoint, when one is supplied;
* the host's own privacy audit cleared *this* figure, still hashes to its
  registered digest, and the sources it inspected have not been re-registered
  since — the record's ``aggregate_only`` flag is an index into that audit,
  never the authorization itself.

Local (loopback) and offline mock destinations are exempt: nothing leaves the
machine, and forcing the same declaration there would only push users to
disable visual QA entirely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..authority.evidence_store import sha256_of_file
from .figure_privacy import (
    FIGURE_PRIVACY_RECEIPT_SCHEMA,
    TRUSTED_AUDIT_VERSIONS,
)

#: Metadata key carrying the verdict that the rendered image holds no
#: patient-level marks, identifiers or small-cell strata. Reading it is *not*
#: the authorization — see :func:`_verify_host_privacy_authorization`.
AGGREGATE_ONLY_METADATA_KEY = "aggregate_only"

#: The only basis this gate accepts. A record whose ``aggregate_only`` came
#: from anywhere else is a producer asserting its own compliance.
HOST_PRIVACY_AUDIT_BASIS = "host_privacy_audit"

#: Metadata keys the host's privacy audit owns. Anything that can register
#: evidence from outside the host — the MCP ``bind_evidence`` tool, generated
#: code writing its own metadata — must be refused these keys: they *are* the
#: authorization, not a claim about it.
RESERVED_PRIVACY_METADATA_KEYS = frozenset(
    {
        "aggregate_only",
        "aggregate_only_audit_version",
        "aggregate_only_basis",
        "aggregate_only_mark_count_verified",
        "aggregate_only_reason",
        "aggregate_only_roles",
        "aggregate_only_sources_inspected",
        "figure_privacy_audit_evidence_id",
    }
)

#: Producers whose figure records the host wrote itself, in-process, through
#: the deterministic figure skill.
TRUSTED_FIGURE_PRODUCERS = frozenset({"publication_figure_skill"})

#: Destinations that never put bytes on the network.
LOCAL_DESTINATIONS = frozenset({"local", "mock"})

#: The gate cleared these bytes to leave. Nothing has been sent yet.
TRANSPORT_AUTHORIZED = "authorized"
#: The provider call returned without raising. This says the client finished
#: sending, not that the remote retained, processed or deleted the image.
TRANSPORT_COMPLETED = "transport_completed"
#: The provider call raised. Bytes may or may not have reached the wire.
TRANSPORT_FAILED = "transport_failed"
#: Authorized, but the caller never reported an outcome — the run was killed
#: mid-call, or a code path forgot to close the loop. Never silently upgraded.
TRANSPORT_UNKNOWN = "transport_unknown"

TRANSPORT_OUTCOMES = frozenset(
    {
        TRANSPORT_AUTHORIZED,
        TRANSPORT_COMPLETED,
        TRANSPORT_FAILED,
        TRANSPORT_UNKNOWN,
    }
)


class FigureEgressError(RuntimeError):
    """Raised when a figure may not be uploaded to an external provider."""


@dataclass
class FigureEgressPolicy:
    """Per-run authority for uploading figure bytes to an external provider.

    ``allow_external_upload`` is the operator's decision and defaults to False.
    ``evidence``/``run_dir`` supply the binding: without them an external
    upload cannot be authorized at all, because there is nothing to check the
    bytes against.
    """

    allow_external_upload: bool = False
    evidence: Optional[Any] = None
    run_dir: Optional[Path] = None
    active_step_evidence_ids: Optional[frozenset] = None
    #: One entry per image the gate cleared, in order, each carrying its own
    #: ``transport`` outcome. Authorization and transport are recorded
    #: separately because the gate runs *before* the provider call: a list
    #: written at authorization time says "these were allowed to go", and
    #: reading it as "these were sent" overstates every failed upload.
    uploaded: List[Dict[str, str]] = field(default_factory=list)

    def record_upload(self, entries: Sequence[Mapping[str, str]]) -> None:
        for entry in entries:
            item = dict(entry)
            item.setdefault("transport", TRANSPORT_AUTHORIZED)
            self.uploaded.append(item)

    def record_transport_outcome(
        self,
        entries: Sequence[Mapping[str, str]],
        outcome: str,
    ) -> None:
        """Close the loop on entries this policy authorized.

        Matched by image digest, so a re-authorized image updates its own row
        rather than the first one that happens to share a path.
        """

        if outcome not in TRANSPORT_OUTCOMES:
            raise ValueError(f"unknown figure transport outcome {outcome!r}")
        digests = {str(entry.get("sha256") or "") for entry in entries}
        for item in self.uploaded:
            if item.get("transport") != TRANSPORT_AUTHORIZED:
                continue
            if str(item.get("sha256") or "") in digests:
                item["transport"] = outcome

    def transport_summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.uploaded:
            key = str(item.get("transport") or TRANSPORT_UNKNOWN)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))


def _figure_records(evidence: Any) -> Dict[str, Any]:
    """Index the run's figure evidence by resolved absolute path."""

    try:
        records = list(evidence.records())
    except Exception as exc:  # pragma: no cover - defensive
        raise FigureEgressError(f"cannot read the evidence store: {exc}") from exc
    return {str(record.relative_path): record for record in records}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _verify_host_privacy_authorization(
    record: Any,
    *,
    evidence: Any,
    run_dir: Path,
) -> Dict[str, Any]:
    """Prove the ``aggregate_only`` verdict was reached by this host.

    ``metadata["aggregate_only"] is True`` is a *string in a dict*. Every path
    that can put a record in the store can put that string there — most
    directly the MCP ``bind_evidence`` tool, which accepts caller-supplied
    metadata, producer and generation mode. Trusting the flag therefore made
    the whole gate a formality: register any PNG as ``kind="figure"`` with
    ``{"aggregate_only": true}`` and it ships.

    So the flag is treated as an *index* into the audit that produced it, and
    the audit is verified end to end: it exists, it is the host's own schema
    and version, it hashes to its registered digest, it cleared *this* figure,
    and the sources it inspected are still the sources the store records.

    Returns the receipt so the caller can put it in the egress entry.
    """

    metadata = _mapping(getattr(record, "metadata", None))
    evidence_id = str(getattr(record, "evidence_id", "") or "<unknown>")

    def deny(detail: str) -> "FigureEgressError":
        return FigureEgressError(
            f"figure evidence {evidence_id} is not cleared for external upload: "
            f"{detail}. Only the host's own privacy audit can clear a figure; a "
            "producer-declared aggregate_only flag is not an authorization."
        )

    if not bool(metadata.get(AGGREGATE_ONLY_METADATA_KEY)):
        raise deny(
            f"it does not declare {AGGREGATE_ONLY_METADATA_KEY}=True, so the "
            "rendered image may carry patient-level marks, identifiers or "
            "small-cell strata"
        )

    basis = str(metadata.get("aggregate_only_basis") or "")
    if basis != HOST_PRIVACY_AUDIT_BASIS:
        raise deny(
            f"its aggregate_only basis is {basis or '<absent>'!r}, not "
            f"{HOST_PRIVACY_AUDIT_BASIS!r}"
        )

    producer = str(getattr(record, "producer", "") or "")
    if producer not in TRUSTED_FIGURE_PRODUCERS:
        raise deny(
            f"it was produced by {producer or '<unknown>'!r}, which is not one "
            "of the host figure producers "
            f"({', '.join(sorted(TRUSTED_FIGURE_PRODUCERS))})"
        )

    declared_version = str(metadata.get("aggregate_only_audit_version") or "")
    if declared_version not in TRUSTED_AUDIT_VERSIONS:
        raise deny(
            f"its audit version {declared_version or '<absent>'!r} is not one "
            f"this host honours ({', '.join(sorted(TRUSTED_AUDIT_VERSIONS))})"
        )

    audit_id = str(metadata.get("figure_privacy_audit_evidence_id") or "")
    if not audit_id:
        raise deny("it names no privacy-audit evidence record")
    try:
        audit_record = evidence.get(audit_id)
    except Exception as exc:  # noqa: BLE001 - an unreadable store clears nothing
        raise deny(f"its privacy audit {audit_id!r} could not be read: {exc}") from exc
    if audit_record is None:
        raise deny(f"its privacy audit {audit_id!r} does not resolve to a record")

    audit_path = Path(run_dir) / str(getattr(audit_record, "relative_path", "") or "")
    if not audit_path.is_file():
        raise deny(f"its privacy audit {audit_id!r} is missing from the run directory")
    audit_sha = sha256_of_file(audit_path)
    registered_audit_sha = str(getattr(audit_record, "sha256", "") or "")
    if audit_sha != registered_audit_sha:
        raise deny(
            f"its privacy audit {audit_id!r} no longer matches its registered "
            f"digest ({audit_sha[:12]}… != {registered_audit_sha[:12]}…)"
        )

    try:
        receipt = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - unparseable audit clears nothing
        raise deny(f"its privacy audit {audit_id!r} is unreadable: {exc}") from exc
    if not isinstance(receipt, Mapping):
        raise deny(f"its privacy audit {audit_id!r} is not an audit receipt")
    if str(receipt.get("schema") or "") != FIGURE_PRIVACY_RECEIPT_SCHEMA:
        raise deny(
            f"its privacy audit declares schema {receipt.get('schema')!r}, not "
            f"{FIGURE_PRIVACY_RECEIPT_SCHEMA!r}"
        )
    if str(receipt.get("audit_version") or "") != declared_version:
        raise deny(
            "its metadata and its privacy audit disagree about the audit "
            f"version ({declared_version!r} vs {receipt.get('audit_version')!r})"
        )
    if receipt.get("aggregate_only") is not True:
        raise deny(
            "its privacy audit did not clear it: "
            + ("; ".join(str(r) for r in receipt.get("reasons") or ()) or "unproven")
        )

    figure_id = str(metadata.get("figure_id") or "")
    if not figure_id:
        raise deny("it declares no figure_id to match against its privacy audit")
    if str(receipt.get("figure_id") or "") != figure_id:
        raise deny(
            f"its privacy audit cleared figure {receipt.get('figure_id')!r}, not "
            f"{figure_id!r}"
        )

    # The clearance is about the artefacts the audit read. If a source has been
    # re-registered since, the verdict no longer describes what was drawn.
    cleared_sources = _mapping(receipt.get("source_sha256"))
    declared_sources = _mapping(metadata.get("source_evidence_sha256"))
    source_ids = [
        str(item) for item in (metadata.get("source_evidence_ids") or ()) if str(item)
    ]
    if not source_ids:
        raise deny("it names no source evidence, so the audit inspected nothing")
    for source_id in source_ids:
        cleared = str(cleared_sources.get(source_id) or "")
        if not cleared:
            raise deny(f"its privacy audit did not inspect source {source_id!r}")
        if str(declared_sources.get(source_id) or "") != cleared:
            raise deny(
                f"its metadata and its privacy audit disagree about source "
                f"{source_id!r}"
            )
        try:
            current = evidence.get(source_id)
        except Exception as exc:  # noqa: BLE001
            raise deny(f"source {source_id!r} could not be re-read: {exc}") from exc
        if current is None:
            raise deny(f"source {source_id!r} is no longer registered")
        if str(getattr(current, "sha256", "") or "") != cleared:
            raise deny(
                f"source {source_id!r} changed after the audit cleared it; the "
                "verdict does not describe the artefacts behind this image"
            )

    return {
        "evidence_id": audit_id,
        "sha256": audit_sha,
        "audit_version": declared_version,
        "mark_count_verified": bool(receipt.get("mark_count_verified")),
    }


def authorize_figure_upload(
    paths: Sequence[Path],
    *,
    policy: Optional[FigureEgressPolicy],
    destination: str,
) -> List[Dict[str, str]]:
    """Return one provenance entry per image, or raise :class:`FigureEgressError`.

    ``destination`` is the provider-transport classification from
    :func:`..providers.factory.provider_transport_destination`.
    """

    entries: List[Dict[str, str]] = []
    if destination in LOCAL_DESTINATIONS:
        for path in paths:
            entries.append(
                {
                    "path": str(path),
                    "sha256": sha256_of_file(Path(path)),
                    "destination": destination,
                }
            )
        return entries

    if policy is None or not policy.allow_external_upload:
        raise FigureEgressError(
            "external figure upload is disabled for this run; a rendered figure "
            "can carry patient-level marks, identifiers or small-cell strata, so "
            "it is not covered by the text outbound projection. Enable it "
            "explicitly with allow_external_figure_upload=True after confirming "
            "the figures are aggregate-only, or use a loopback/local VLM."
        )
    if policy.evidence is None or policy.run_dir is None:
        raise FigureEgressError(
            "external figure upload requires an evidence-bound run: without the "
            "evidence store and run directory the uploaded bytes cannot be "
            "matched to a registered, hashed figure artefact"
        )

    run_dir = Path(policy.run_dir).resolve()
    indexed = _figure_records(policy.evidence)

    for raw_path in paths:
        path = Path(raw_path)
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise FigureEgressError(f"figure {path} cannot be read: {exc}") from exc
        try:
            relative = resolved.relative_to(run_dir)
        except ValueError as exc:
            raise FigureEgressError(
                f"figure {path} is outside the run directory {run_dir}; only "
                "artefacts this run produced and registered may be uploaded"
            ) from exc

        record = indexed.get(str(relative))
        if record is None:
            raise FigureEgressError(
                f"figure {relative} is not a registered evidence artefact; an "
                "unregistered image has no provenance and must not be uploaded"
            )
        if str(record.kind) != "figure":
            raise FigureEgressError(
                f"evidence {record.evidence_id} is kind={record.kind!r}, not a "
                "figure; only registered figures may be uploaded"
            )

        actual = sha256_of_file(resolved)
        if actual != str(record.sha256):
            raise FigureEgressError(
                f"figure {relative} no longer matches its registered digest "
                f"({actual[:12]}… != {str(record.sha256)[:12]}…); the bytes about "
                "to be uploaded are not the audited artefact"
            )

        active = policy.active_step_evidence_ids
        if active is not None and str(record.evidence_id) not in active:
            raise FigureEgressError(
                f"figure evidence {record.evidence_id} does not belong to the "
                "active checkpoint; a stale artefact must not be uploaded"
            )

        audit = _verify_host_privacy_authorization(
            record, evidence=policy.evidence, run_dir=run_dir
        )

        entries.append(
            {
                "path": str(relative),
                "evidence_id": str(record.evidence_id),
                "sha256": actual,
                "destination": destination,
                "privacy_audit_evidence_id": audit["evidence_id"],
                "privacy_audit_sha256": audit["sha256"],
                "privacy_audit_version": audit["audit_version"],
                "transport": TRANSPORT_AUTHORIZED,
            }
        )

    if policy is not None:
        policy.record_upload(entries)
    return entries


class FigureEgressReceiptError(RuntimeError):
    """Raised when this run cannot record what it sent to an external provider.

    Kept distinct from :class:`FigureEgressError` (which *prevents* an upload)
    because it describes the opposite situation: bytes may already have left,
    and the run has no way to say so. It must not be demoted to a warning.
    """


def register_figure_egress_receipt(
    *,
    policy: Optional[FigureEgressPolicy],
    evidence: Any,
    run_dir: Path,
    phase: str = "completed",
) -> Optional[Any]:
    """Persist what this run is about to send, or did send, to an external provider.

    ``FigureEgressPolicy.uploaded`` is an in-memory list on an object the write
    phase builds and drops, so an authorized upload left no trace: the finished
    run could not answer which images left the host. The receipt is written
    whenever a policy existed — an empty list is the meaningful evidence that
    nothing was uploaded, and is exactly what a privacy reviewer needs to see.

    Two phases, because a single post-upload record is lost precisely when it
    matters most (upload succeeded, host then failed):

    * ``intent`` — written before any byte can leave, recording that this run
      is authorized to upload and under which policy;
    * ``completed`` — written after the visual-QA call returns *or raises*,
      recording what actually went out.

    Each entry carries its own ``transport`` outcome, because the gate runs
    before the provider call: ``authorized`` and ``transport_completed`` are
    different facts, and an entry nobody closed the loop on is recorded as
    ``transport_unknown`` rather than counted as a success.

    A failure to write either one raises :class:`FigureEgressReceiptError`.
    """

    if policy is None:
        return None
    if phase not in {"intent", "completed"}:
        raise ValueError(f"unknown figure-egress receipt phase {phase!r}")
    uploads = [dict(item) for item in (getattr(policy, "uploaded", ()) or ())]
    if phase == "completed":
        # An entry still marked "authorized" at the end means nobody reported
        # what happened to it. That is a third state, not a success.
        for item in uploads:
            if item.get("transport") == TRANSPORT_AUTHORIZED:
                item["transport"] = TRANSPORT_UNKNOWN
    counts: Dict[str, int] = {}
    for item in uploads:
        key = str(item.get("transport") or TRANSPORT_UNKNOWN)
        counts[key] = counts.get(key, 0) + 1
    payload = {
        "schema": "easyicu.figure_egress_receipt/3",
        "phase": phase,
        "allow_external_upload": bool(policy.allow_external_upload),
        "authorized_count": len(uploads),
        "transport_counts": dict(sorted(counts.items())),
        "uploads": uploads,
    }
    evidence_id = (
        "figure_egress_receipt"
        if phase == "completed"
        else "figure_egress_authorization_intent"
    )
    receipt_path = Path(run_dir) / f"{evidence_id}.json"
    try:
        receipt_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return evidence.register_file(
            kind="log",
            description=(
                "Which rendered figures, if any, were authorized to leave the "
                "host for external visual review, with their evidence ids and "
                f"digests ({phase} phase)."
            ),
            source_path=receipt_path,
            evidence_id=evidence_id,
            producer="pipeline",
            generation_mode="system",
            metadata={
                "authorized_count": len(uploads),
                "transport_counts": dict(sorted(counts.items())),
                "egress_phase": phase,
            },
            on_sha_change="new_id",
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed blocker
        raise FigureEgressReceiptError(
            f"figure-egress {phase} receipt could not be recorded ({exc}); this "
            "run cannot account for image bytes sent to an external provider, "
            "so it must not be treated as a complete manuscript run"
        ) from exc


__all__ = [
    "AGGREGATE_ONLY_METADATA_KEY",
    "HOST_PRIVACY_AUDIT_BASIS",
    "LOCAL_DESTINATIONS",
    "RESERVED_PRIVACY_METADATA_KEYS",
    "TRANSPORT_AUTHORIZED",
    "TRANSPORT_COMPLETED",
    "TRANSPORT_FAILED",
    "TRANSPORT_OUTCOMES",
    "TRANSPORT_UNKNOWN",
    "TRUSTED_FIGURE_PRODUCERS",
    "FigureEgressError",
    "FigureEgressPolicy",
    "FigureEgressReceiptError",
    "authorize_figure_upload",
    "register_figure_egress_receipt",
]
