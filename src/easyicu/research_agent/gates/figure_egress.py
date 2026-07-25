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
* the record declares aggregate-only content.

Local (loopback) and offline mock destinations are exempt: nothing leaves the
machine, and forcing the same declaration there would only push users to
disable visual QA entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..authority.evidence_store import sha256_of_file

#: Metadata key a figure producer sets to declare the rendered image carries no
#: patient-level marks, identifiers or small-cell strata.
AGGREGATE_ONLY_METADATA_KEY = "aggregate_only"

#: Destinations that never put bytes on the network.
LOCAL_DESTINATIONS = frozenset({"local", "mock"})


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
    #: SHA-256 of every image this run actually sent externally, in order.
    uploaded: List[Dict[str, str]] = field(default_factory=list)

    def record_upload(self, entries: Sequence[Mapping[str, str]]) -> None:
        self.uploaded.extend(dict(entry) for entry in entries)


def _figure_records(evidence: Any) -> Dict[str, Any]:
    """Index the run's figure evidence by resolved absolute path."""

    try:
        records = list(evidence.records())
    except Exception as exc:  # pragma: no cover - defensive
        raise FigureEgressError(f"cannot read the evidence store: {exc}") from exc
    return {str(record.relative_path): record for record in records}


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

        metadata = record.metadata if isinstance(record.metadata, Mapping) else {}
        if not bool(metadata.get(AGGREGATE_ONLY_METADATA_KEY)):
            raise FigureEgressError(
                f"figure evidence {record.evidence_id} does not declare "
                f"{AGGREGATE_ONLY_METADATA_KEY}=True; the producer must assert "
                "the rendered image carries no patient-level marks, identifiers "
                "or small-cell strata before it can leave the machine"
            )

        entries.append(
            {
                "path": str(relative),
                "evidence_id": str(record.evidence_id),
                "sha256": actual,
                "destination": destination,
            }
        )

    if policy is not None:
        policy.record_upload(entries)
    return entries


__all__ = [
    "AGGREGATE_ONLY_METADATA_KEY",
    "LOCAL_DESTINATIONS",
    "FigureEgressError",
    "FigureEgressPolicy",
    "authorize_figure_upload",
]
