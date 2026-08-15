"""Verify digest-bound source coordinates against a registered evidence view."""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class SourceFingerprintRecord(Protocol):
    """Smallest registered-record surface needed by this integrity rule."""

    sha256: str


class SourceFingerprintLookup(Protocol):
    """Read-only evidence lookup accepted by the fingerprint verifier."""

    def get(self, evidence_id_or_alias: str) -> SourceFingerprintRecord | None: ...


def registered_source_fingerprints_match(
    evidence: SourceFingerprintLookup,
    metadata: Mapping[str, Any],
) -> bool:
    """Return whether every declared source still has its registered digest.

    Both the publication-figure reuse check and the readiness gate consume this
    one projection. Absent or empty coordinates are a negative answer: an
    artifact that names no source cannot have its sources verified.
    """

    source_ids = metadata.get("source_evidence_ids")
    if isinstance(source_ids, str):
        ids = [source_ids]
    elif isinstance(source_ids, (list, tuple, set)):
        ids = [str(evidence_id) for evidence_id in source_ids if str(evidence_id)]
    else:
        ids = []
    single = metadata.get("source_evidence_id")
    if single and str(single) not in ids:
        ids.append(str(single))
    fingerprints = metadata.get("source_evidence_sha256")
    if not ids or not isinstance(fingerprints, dict) or not fingerprints:
        return False
    for evidence_id in ids:
        source = evidence.get(evidence_id)
        if source is None or fingerprints.get(evidence_id) != source.sha256:
            return False
    return True


__all__ = [
    "SourceFingerprintLookup",
    "SourceFingerprintRecord",
    "registered_source_fingerprints_match",
]
