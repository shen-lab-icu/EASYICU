"""Path-free, browser-safe provenance for an evidence-bound manuscript.

The manuscript remains the scientific source of truth.  This module only
projects its already-bound numeric footnotes into a reader contract that can
show, for each printed number, the exact JSON field and the registered code /
data artefacts that produced it.  It never computes or upgrades a result.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.numeric_claim_identity import NumericClaim
from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.contracts.cohort_product_keys import is_closed_cohort_product_key
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


def verified_evidence_json(
    evidence: EvidenceStore,
    evidence_id_or_alias: str,
    *,
    exact_evidence_id: bool = False,
    expected_kind: str | None = None,
    expected_producer: str | None = None,
) -> dict[str, Any]:
    """Read a registered JSON object with exact identity and byte verification."""

    record = evidence.get(evidence_id_or_alias)
    if (
        record is None
        or (exact_evidence_id and record.evidence_id != evidence_id_or_alias)
        or (expected_kind is not None and record.kind != expected_kind)
        or (expected_producer is not None and record.producer != expected_producer)
    ):
        raise RuntimeError(
            f"writer evidence {evidence_id_or_alias!r} lost its registered identity"
        )
    path = verified_run_evidence_path(evidence.root, record)
    if path is None:
        raise RuntimeError(
            f"writer evidence {evidence_id_or_alias!r} failed digest verification"
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"writer evidence {evidence_id_or_alias!r} became unreadable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != record.sha256:
        raise RuntimeError(
            f"writer evidence {evidence_id_or_alias!r} changed while being read"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"writer evidence {evidence_id_or_alias!r} is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"writer evidence {evidence_id_or_alias!r} is not a JSON object"
        )
    return payload


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


def _record_status(evidence: EvidenceStore, record: EvidenceRecord) -> str:
    """Return current / stale / missing / escaped without raising."""

    root = evidence.root.resolve()
    path = (root / record.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return "escaped"
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record.sha256:
        return "stale"
    return "current"


def _verify_record(evidence: EvidenceStore, record: EvidenceRecord) -> None:
    """Fail closed if a registered artifact is missing, escaped, or stale."""

    status = _record_status(evidence, record)
    if status == "escaped":
        raise ManuscriptProvenanceError(
            f"evidence path escapes run root: {record.evidence_id}"
        )
    if status == "missing":
        raise ManuscriptProvenanceError(
            f"evidence file is missing: {record.evidence_id}"
        )
    if status == "stale":
        raise ManuscriptProvenanceError(
            f"evidence digest is stale: {record.evidence_id}"
        )


def _related_artifacts(
    record: EvidenceRecord,
    *,
    evidence: EvidenceStore,
    records_by_id: Mapping[str, EvidenceRecord],
    records: Sequence[EvidenceRecord],
    display_index: Mapping[str, Mapping[str, str]] | None = None,
    verify: str = "raise",
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
        status = _record_status(evidence, item)
        if verify == "raise":
            _verify_record(evidence, item)
        projected = _safe_artifact(item, role=role)
        projected["status"] = status
        display = (display_index or {}).get(item.sha256)
        if display:
            projected["display_id"] = str(display["display_id"])
            projected["display_contract_sha256"] = str(
                display.get("contract_sha256") or ""
            )
        public.append(projected)
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


_METHOD_SUMMARY_FIELDS = (
    "intent",
    "planned_analysis_role",
    "time_window_label",
    "exposure_label",
    "outcome_label",
    "population_label",
)


def _summarize_method(record: Mapping[str, Any] | None) -> dict[str, str]:
    """Keep only short scalar step coordinates; never expose rows or paths."""

    if not isinstance(record, Mapping):
        return {}
    summary: dict[str, str] = {}
    for key in _METHOD_SUMMARY_FIELDS:
        value = record.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            text = str(value).strip()
            if text:
                summary[key] = text[:400]
    return summary


def _safe_reader_notes(notes: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Project bounded verification notes without paths or patient content."""

    public: list[dict[str, Any]] = []
    for note in notes or []:
        if not isinstance(note, Mapping):
            continue
        code = str(note.get("code") or "").strip()[:120]
        text = str(note.get("text") or "").strip()[:600]
        if not code or not text:
            continue
        projected: dict[str, Any] = {
            "code": code,
            "severity": str(note.get("severity") or "warning").strip()[:20],
            "text": text,
        }
        evidence_id = str(note.get("evidence_id") or "").strip()
        if evidence_id:
            projected["evidence_id"] = evidence_id[:160]
        sha256 = str(note.get("sha256") or "").strip().lower()
        if re.fullmatch(r"[a-f0-9]{64}", sha256):
            projected["sha256"] = sha256
        public.append(projected)
        if len(public) >= 24:
            break
    return public


_LINEAGE_KINDS = (
    "statistic", "analysis_input", "cohort", "column_definitions",
    "concept_dictionary", "data_source",
)
_BROWSER_SAFE_INTEGER = 2 ** 53 - 1


def _lineage_token(value: Any) -> str | None:
    """Identifiers, never arbitrary paths, URLs, credentials or prose."""
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value) else None


def _lineage_digest(value: Any) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) else None


def _lineage_count(facts: dict[str, Any], key: str, value: Any) -> None:
    if type(value) is int and 0 <= value <= _BROWSER_SAFE_INTEGER:
        facts[key] = value


class _ClaimLineage:
    """Read only this store's exact identities; cache verification per build.

    Node digests always identify the openable evidence record, including when
    that record is a metadata receipt for a cohort or dictionary. Content
    digests inside a receipt are separate facts, never substituted for its SHA.
    No mutable working copy, installed dictionary or source directory is read.
    """

    def __init__(self, evidence: EvidenceStore, records: Mapping[str, EvidenceRecord]):
        self.evidence = evidence
        self.records = records
        self.cache: dict[str, tuple[str, str | None, dict[str, Any]]] = {}

    def _read(self, evidence_id: str) -> tuple[str, str | None, dict[str, Any]]:
        if evidence_id in self.cache:
            return self.cache[evidence_id]
        record = self.records.get(evidence_id)
        result: tuple[str, str | None, dict[str, Any]] = ("missing", "artifact_unregistered", {})
        if record is not None and _lineage_token(evidence_id) and _lineage_digest(record.sha256):
            path = verified_run_evidence_path(self.evidence.root, record)
            if path is None:
                # Only distinguish stale bytes after ruling out escaped or
                # symlinked coordinates. Never read a target outside this run.
                relative = Path(record.relative_path)
                safe = (not relative.is_absolute() and ".." not in relative.parts
                        and bool(relative.parts) and relative.parts[0] == "evidence")
                candidate = self.evidence.root
                for part in relative.parts:
                    candidate /= part
                    safe = safe and not candidate.is_symlink()
                if safe and candidate.is_file():
                    result = ("stale", "artifact_digest_mismatch", {})
                else:
                    result = ("missing", "artifact_unavailable", {})
            elif path.suffix.lower() != ".json":
                result = ("current", None, {})
            elif path.stat().st_size > 2 * 1024 * 1024:
                result = ("missing", "artifact_json_too_large", {})
            else:
                try:
                    payload = verified_evidence_json(self.evidence, evidence_id, exact_evidence_id=True)
                    result = ("current", None, payload)
                except RuntimeError:
                    result = ("missing", "artifact_json_invalid", {})
        self.cache[evidence_id] = result
        return result

    def _node(self, kind: str, evidence_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        status, reason, payload = self._read(evidence_id)
        record = self.records.get(evidence_id)
        node = {
            "kind": kind, "status": status,
            "sha256": _lineage_digest(record.sha256) if record else None,
            "evidence_id": _lineage_token(evidence_id) if record else None,
            "facts": {}, "missing_reason": reason,
        }
        return node, payload

    def _reaches_cohort(self, record: EvidenceRecord, digest: str) -> bool:
        """Follow registered input edges, including derived analysis cohorts."""
        pending, seen = [record], set()
        while pending:
            current = pending.pop()
            if current.evidence_id in seen:
                continue
            seen.add(current.evidence_id)
            if self._read(current.evidence_id)[0] != "current":
                continue
            if current.sha256 == digest:
                return True
            pending.extend(self.records[key] for key in current.inputs if key in self.records)
        return False

    def _input_bindings(self, summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        bindings = summary.get("input_bindings")
        if isinstance(bindings, list):
            rows = [row for row in bindings if isinstance(row, Mapping)]
            cohorts = [row for row in rows if is_closed_cohort_product_key(row.get("input_key"))]
            return cohorts or rows
        # Archived executors recorded the consumed digest directly. Join it
        # only to an exact, unique table in this registry, never to a filename.
        digests = {summary[key] for key in (
            "input_sha256", "analysis_input_sha256", "counting_process_input_sha256",
        ) if _lineage_digest(summary.get(key))}
        if not digests:
            return []
        matches = [row for row in self.records.values() if row.kind == "table" and row.sha256 in digests]
        if len(digests) != 1:
            return [{}, {}]
        digest = next(iter(digests))
        return [{"loaded": True, "sha256": digest, "evidence_id": row.evidence_id}
                for row in matches] or [{"loaded": True, "sha256": digest}]

    def build(
        self, claim: NumericClaim, record: EvidenceRecord, method_summary: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        nodes = [{"kind": kind, "status": "missing", "sha256": None,
                  "evidence_id": None, "facts": {}, "missing_reason": "upstream_unproven"}
                 for kind in _LINEAGE_KINDS]
        nodes[0], summary = self._node("statistic", record.evidence_id)
        if nodes[0]["status"] != "current":
            return nodes
        if _lineage_token(claim.step_id):
            nodes[0]["facts"]["step_id"] = claim.step_id
        method = summary.get("method", method_summary.get("intent"))
        if isinstance(method, str) and re.fullmatch(r"[\w .()+-]{1,160}", method):
            nodes[0]["facts"]["method"] = method
        bindings = self._input_bindings(summary)
        if len(bindings) != 1:
            nodes[1]["missing_reason"] = "analysis_input_ambiguous" if bindings else "analysis_input_not_recorded"
            return nodes
        binding = bindings[0]
        if binding.get("loaded") is not True:
            nodes[1]["missing_reason"] = "analysis_input_not_loaded"
            return nodes
        input_id = _lineage_token(binding.get("evidence_id")) or ""
        nodes[1], _ = self._node("analysis_input", input_id)
        if nodes[1]["status"] != "current":
            return nodes
        if self.records[input_id].kind != "table":
            nodes[1].update(status="missing", missing_reason="analysis_input_not_table")
            return nodes
        if not _lineage_digest(binding.get("sha256")):
            nodes[1].update(status="missing", missing_reason="input_digest_not_recorded")
            return nodes
        if nodes[1]["sha256"] != binding["sha256"]:
            nodes[1].update(status="stale", missing_reason="input_digest_mismatch")
            return nodes
        _lineage_count(nodes[1]["facts"], "rows", binding.get("row_count"))
        capsule_node, capsule = self._node("cohort", "run_input_capsule")
        if capsule_node["status"] != "current":
            nodes[2] = capsule_node
            return nodes
        cohort_sha = _lineage_digest(capsule.get("cohort_sha256"))
        if not cohort_sha or not self._reaches_cohort(self.records[input_id], cohort_sha):
            nodes[2]["missing_reason"] = "cohort_input_unlinked"
            return nodes
        context_id = _lineage_token(capsule.get("context_evidence_id")) or ""
        nodes[2], context = self._node("cohort", context_id)
        if nodes[2]["status"] != "current":
            return nodes
        if nodes[2]["sha256"] != capsule.get("context_sha256"):
            nodes[2].update(status="stale", missing_reason="context_digest_mismatch")
            return nodes
        cohort = context.get("cohort")
        cohort = cohort if isinstance(cohort, Mapping) else {}
        provenance = cohort.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        typed = context.get("materialized_inputs")
        typed = typed.get("cohort") if isinstance(typed, Mapping) else None
        typed = typed if isinstance(typed, Mapping) else {}
        if typed and typed.get("cohort_sha256") != cohort_sha:
            nodes[2].update(status="stale", missing_reason="cohort_digest_mismatch")
            return nodes
        _lineage_count(nodes[2]["facts"], "rows", typed.get("cohort_rows", cohort.get("n_stays")))
        window = provenance.get("materialized_cohort_window_hours")
        if isinstance(window, list) and len(window) == 2:
            for key, value in zip(("window_start_hours", "window_end_hours"), window, strict=True):
                if (type(value) in {int, float} and abs(value) <= _BROWSER_SAFE_INTEGER
                        and math.isfinite(value)):
                    nodes[2]["facts"][key] = value
        columns = typed.get("column_bindings")
        if isinstance(columns, Mapping):
            nodes[3], _ = self._node("column_definitions", context_id)
            nodes[3]["facts"]["count"] = len(columns)
        else:
            nodes[3]["missing_reason"] = "column_definitions_not_recorded"
        nodes[4], fingerprint = self._node("concept_dictionary", "concept_dict_fingerprint")
        if nodes[4]["status"] == "current":
            for key, field in (("base_sha256", "concept_dict_sha"), ("overlay_sha256", "sofa2_dict_sha")):
                if _lineage_digest(fingerprint.get(field)):
                    nodes[4]["facts"][key] = fingerprint[field]
            _lineage_count(nodes[4]["facts"], "concept_count", fingerprint.get("concept_count"))
            if "base_sha256" not in nodes[4]["facts"]:
                nodes[4].update(status="missing", missing_reason="dictionary_digest_not_recorded", facts={})
        nodes[5], _ = self._node("data_source", context_id)
        source = provenance.get("source_identity")
        source = source if isinstance(source, Mapping) else {}
        facts = nodes[5]["facts"]
        for key in ("database", "version", "registration_id", "source_id"):
            value = source.get(key, cohort.get(key))
            if _lineage_token(value):
                facts[key] = value
        if not any(key in facts for key in ("registration_id", "source_id")):
            nodes[5].update(status="missing", missing_reason="source_identity_not_recorded")
        # The release owner supplies public metadata. A database/demo label is
        # never proof that this run used an official archive.
        from easyicu.webserver.demo_source_contracts import get_source

        try:
            official = get_source(facts.get("source_id"))
        except KeyError:
            return nodes
        matches = (bool(_lineage_digest(source.get("archive_sha256")))
                   and source.get("archive_sha256") == official.archive_sha256
                   and facts.get("database") == official.database
                   and facts.get("version") == official.version)
        facts["official_sha256_matches"] = matches
        if matches:
            facts["doi"] = official.citation_url.removeprefix("https://doi.org/")
            facts["physionet_slug"] = official.landing_page.split("/content/", 1)[1].split("/", 1)[0]
        return nodes


def build_manuscript_provenance(
    *,
    manuscript: str,
    evidence: EvidenceStore,
    binding_map: Mapping[str, NumericClaim] | None = None,
    claim_ceiling: str = "analysis_only",
    verify: str = "raise",
    display_index: Mapping[str, Mapping[str, str]] | None = None,
    method_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    reader_notes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a digest-bound, path-free reader projection.

    Every referenced numeric footnote must resolve to exactly one registered
    NumericClaim and one current EvidenceRecord.  With ``verify="raise"`` the
    publication/binding path fails closed on any disagreement instead of
    producing a plausible-looking reader link.  ``verify="mark"`` is reserved
    for the reader build: missing or stale links are marked explicitly so the
    projection can still be reviewed, while publication paths keep raising.
    """

    if verify not in {"raise", "mark"}:
        raise ValueError("verify must be 'raise' or 'mark'")

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
    lineage = _ClaimLineage(evidence, records_by_id)
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
        status = _record_status(evidence, record)
        if verify == "raise":
            _verify_record(evidence, record)
        claim_payload: dict[str, Any] = {
            "claim_id": claim_id,
            "display_value": displays[0],
            "source_value": claim.value,
            "canonical_value": claim.canonical,
            "step_id": claim.step_id,
            "source_field": claim.source_field,
            "source_json_pointer": _json_pointer(claim.source_field),
            "evidence": _safe_artifact(record, role="source_json"),
            "related_artifacts": (
                _related_artifacts(
                    record,
                    evidence=evidence,
                    records_by_id=records_by_id,
                    records=records,
                    display_index=display_index,
                    verify=verify,
                )
                if status == "current"
                else []
            ),
            "effect_scale": (
                claim.effect_scale.value if claim.effect_scale is not None else None
            ),
            "estimand": claim.estimand.value if claim.estimand is not None else None,
            "occurrence_count": len(displays),
            "status": status,
        }
        claim_payload["evidence"]["status"] = status
        if status == "current":
            related_statuses = {
                item["status"] for item in claim_payload["related_artifacts"]
            }
            claim_payload["status"] = next(
                (state for state in ("escaped", "missing", "stale") if state in related_statuses),
                "current",
            )
        method_summary = _summarize_method(
            (method_summaries or {}).get(claim.step_id)
        )
        if method_summary:
            claim_payload["method_summary"] = method_summary
        claim_payload["lineage"] = lineage.build(claim, record, method_summary)
        claims.append(claim_payload)

    blocks = _reader_blocks(manuscript)
    notes = _safe_reader_notes(reader_notes)
    if notes:
        blocks.append({"kind": "verification_notes", "notes": notes})

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "evidence_bound_manuscript_reader",
        "manuscript_sha256": hashlib.sha256(manuscript.encode("utf-8")).hexdigest(),
        "claim_count": len(claims),
        "claim_ceiling": claim_ceiling,
        "publication_authorized": False,
        "article_blocks": blocks,
        "claims": claims,
        "integrity": {
            "path_values_returned": False,
            "patient_rows_returned": False,
            "raw_data_returned": False,
            "numeric_claims_verified": all(
                claim.get("status") == "current" for claim in claims
            ),
        },
    }


__all__ = [
    "ManuscriptProvenanceError",
    "SCHEMA_VERSION",
    "build_manuscript_provenance",
    "strip_numeric_provenance",
]
