"""Path-free, browser-safe provenance for an evidence-bound manuscript.

The manuscript remains the scientific source of truth.  This module only
projects its already-bound numeric footnotes into a reader contract that can
show, for each printed number, the exact JSON field and the registered code /
data artefacts that produced it.  It never computes or upgrades a result.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.numeric_claim_identity import NumericClaim
from easyicu.research_agent.schema import EvidenceRecord


SCHEMA_VERSION = "easyicu.manuscript-provenance/1"

_FOOTNOTE_DEFINITION = re.compile(
    r"^\[\^(?P<id>[A-Za-z0-9_-]+)\]:\s*(?P<body>.+)$", re.MULTILINE
)
_BOUND_NUMBER = re.compile(
    r"(?P<display>[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)%?)"
    r"\[\^(?P<id>[A-Za-z0-9_-]+)\]"
)
_INTERNAL_EVIDENCE_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\(evidence/[^)\n]+\)")
_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+)$")


class ManuscriptProvenanceError(ValueError):
    """The bound manuscript and EvidenceStore disagree."""


def _parse_definition(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in body.split("; "):
        key, separator, value = part.partition("=")
        if separator and key.strip():
            fields[key.strip()] = value.strip()
    return fields


def _claim_identity(claim: NumericClaim) -> tuple[str, str, str, str]:
    return (
        claim.step_id,
        claim.source_field,
        claim.evidence_id,
        claim.value,
    )


def _definition_identity(fields: Mapping[str, str]) -> tuple[str, str, str, str]:
    required = ("step", "field", "evidence", "value")
    missing = [key for key in required if not fields.get(key)]
    if missing:
        raise ManuscriptProvenanceError(
            "numeric footnote is missing required provenance fields: "
            + ", ".join(missing)
        )
    return (
        fields["step"],
        fields["field"],
        fields["evidence"],
        fields["value"],
    )


def _safe_artifact(record: EvidenceRecord, *, role: str) -> dict[str, Any]:
    """Project immutable identifiers only; never expose a host path or rows."""

    return {
        "evidence_id": record.evidence_id,
        "role": role,
        "kind": record.kind,
        "description": record.description[:500],
        "sha256": record.sha256,
        "produced_by_step": record.produced_by_step,
        "producer": record.producer,
        "generation_mode": record.generation_mode,
    }


def _json_pointer(source_field: str) -> str:
    """Convert dotted fields and array indexes into an RFC 6901 pointer."""

    tokens: list[str] = []
    for dotted_part in source_field.split("."):
        head, *indexes = re.split(r"\[([^\]]+)\]", dotted_part)
        if head:
            tokens.append(head)
        tokens.extend(index for index in indexes if index)
    escaped = [token.replace("~", "~0").replace("/", "~1") for token in tokens]
    return "/" + "/".join(escaped)


def _verify_record(evidence: EvidenceStore, record: EvidenceRecord) -> None:
    """Fail closed if a registered artifact is missing, escaped, or stale."""

    root = evidence.root.resolve()
    path = (root / record.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ManuscriptProvenanceError(
            f"evidence path escapes run root: {record.evidence_id}"
        ) from exc
    if not path.is_file():
        raise ManuscriptProvenanceError(
            f"evidence file is missing: {record.evidence_id}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record.sha256:
        raise ManuscriptProvenanceError(
            f"evidence digest is stale: {record.evidence_id}"
        )


def _related_artifacts(
    record: EvidenceRecord,
    *,
    evidence: EvidenceStore,
    records_by_id: Mapping[str, EvidenceRecord],
    records: Sequence[EvidenceRecord],
) -> list[dict[str, Any]]:
    selected: list[tuple[EvidenceRecord, str]] = [(record, "source_json")]
    if record.script_evidence_id:
        script = records_by_id.get(record.script_evidence_id)
        if script is not None:
            selected.append((script, "analysis_code"))
    for evidence_id in record.inputs:
        parent = records_by_id.get(evidence_id)
        if parent is not None:
            selected.append((parent, "input_data"))
    if record.produced_by_step:
        for sibling in records:
            if sibling.produced_by_step != record.produced_by_step:
                continue
            if sibling.kind not in {"code", "table", "figure"}:
                continue
            role = "analysis_code" if sibling.kind == "code" else "supporting_artifact"
            selected.append((sibling, role))

    public: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item, role in selected:
        if item.evidence_id in seen:
            continue
        seen.add(item.evidence_id)
        _verify_record(evidence, item)
        public.append(_safe_artifact(item, role=role))
        if len(public) >= 12:
            break
    return public


def _reader_blocks(markdown: str) -> list[dict[str, Any]]:
    body = _FOOTNOTE_DEFINITION.sub("", markdown).strip()
    body = _INTERNAL_EVIDENCE_LINK.sub(lambda match: f"[{match.group('label')}]", body)
    blocks: list[dict[str, Any]] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = " ".join(item.strip() for item in paragraph if item.strip()).strip()
        paragraph.clear()
        if text:
            blocks.append({"kind": "paragraph", "segments": _reader_segments(text)})

    for line in body.splitlines():
        heading = _HEADING.match(line)
        if heading:
            flush_paragraph()
            blocks.append(
                {
                    "kind": "heading",
                    "level": min(len(heading.group("hashes")), 4),
                    "segments": _reader_segments(heading.group("text").strip()),
                }
            )
        elif not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line)
    flush_paragraph()
    return blocks


def _reader_segments(text: str) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    cursor = 0
    for match in _BOUND_NUMBER.finditer(text):
        if match.start() > cursor:
            segments.append({"kind": "text", "text": text[cursor : match.start()]})
        segments.append(
            {
                "kind": "claim",
                "text": match.group("display"),
                "claim_id": match.group("id"),
            }
        )
        cursor = match.end()
    if cursor < len(text):
        segments.append({"kind": "text", "text": text[cursor:]})
    return segments or [{"kind": "text", "text": text}]


def strip_numeric_provenance(markdown: str) -> str:
    """Return the same bound prose with numeric markers/definitions removed.

    This is used by provider-free reporting replays: the writer text and
    evidence links remain byte-for-byte unchanged while every numeric claim is
    rebound against the current deterministic contract.
    """

    without_definitions = _FOOTNOTE_DEFINITION.sub("", markdown)
    return (
        re.sub(
            r"\[\^[A-Za-z0-9_-]+\]",
            "",
            without_definitions,
        ).rstrip()
        + "\n"
    )


def build_manuscript_provenance(
    *,
    manuscript: str,
    evidence: EvidenceStore,
    binding_map: Mapping[str, NumericClaim] | None = None,
    claim_ceiling: str = "analysis_only",
) -> dict[str, Any]:
    """Build a digest-bound, path-free reader projection.

    Every referenced numeric footnote must resolve to exactly one registered
    NumericClaim and one current EvidenceRecord.  Any disagreement fails
    closed instead of producing a plausible-looking reader link.
    """

    definitions = {
        match.group("id"): _parse_definition(match.group("body"))
        for match in _FOOTNOTE_DEFINITION.finditer(manuscript)
    }
    occurrences: dict[str, list[str]] = {}
    for match in _BOUND_NUMBER.finditer(manuscript):
        occurrences.setdefault(match.group("id"), []).append(match.group("display"))
    missing_definitions = sorted(set(occurrences) - set(definitions))
    if missing_definitions:
        raise ManuscriptProvenanceError(
            "numeric markers have no provenance definition: "
            + ", ".join(missing_definitions)
        )

    numeric_claims = list(evidence.numeric_claims())
    records = list(evidence.records())
    records_by_id = {record.evidence_id: record for record in records}
    claims: list[dict[str, Any]] = []
    for claim_id, displays in occurrences.items():
        expected = _definition_identity(definitions[claim_id])
        if binding_map is not None and claim_id in binding_map:
            matches = [binding_map[claim_id]]
        else:
            matches = [
                claim for claim in numeric_claims if _claim_identity(claim) == expected
            ]
        if len(matches) != 1 or _claim_identity(matches[0]) != expected:
            raise ManuscriptProvenanceError(
                f"{claim_id} does not resolve to exactly one registered NumericClaim"
            )
        claim = matches[0]
        record = records_by_id.get(claim.evidence_id)
        if record is None:
            raise ManuscriptProvenanceError(
                f"{claim_id} references missing evidence {claim.evidence_id}"
            )
        _verify_record(evidence, record)
        claims.append(
            {
                "claim_id": claim_id,
                "display_value": displays[0],
                "source_value": claim.value,
                "canonical_value": claim.canonical,
                "step_id": claim.step_id,
                "source_field": claim.source_field,
                "source_json_pointer": _json_pointer(claim.source_field),
                "evidence": _safe_artifact(record, role="source_json"),
                "related_artifacts": _related_artifacts(
                    record,
                    evidence=evidence,
                    records_by_id=records_by_id,
                    records=records,
                ),
                "effect_scale": (
                    claim.effect_scale.value if claim.effect_scale is not None else None
                ),
                "estimand": claim.estimand.value
                if claim.estimand is not None
                else None,
                "occurrence_count": len(displays),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "evidence_bound_manuscript_reader",
        "manuscript_sha256": hashlib.sha256(manuscript.encode("utf-8")).hexdigest(),
        "claim_count": len(claims),
        "claim_ceiling": claim_ceiling,
        "publication_authorized": False,
        "article_blocks": _reader_blocks(manuscript),
        "claims": claims,
        "integrity": {
            "path_values_returned": False,
            "patient_rows_returned": False,
            "raw_data_returned": False,
            "numeric_claims_verified": True,
        },
    }


__all__ = [
    "ManuscriptProvenanceError",
    "SCHEMA_VERSION",
    "build_manuscript_provenance",
    "strip_numeric_provenance",
]
