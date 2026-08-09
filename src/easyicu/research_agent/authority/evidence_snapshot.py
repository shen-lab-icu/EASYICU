"""Schema-light, fail-closed loader for the EvidenceStore authority state.

The historical flat JSON files remain compatibility projections. New stores
stage a complete logical state behind run-root coordinates, so readers observe
one generation of records, aliases, and numeric claims rather than a mixture
of independently replaced files. A persistent transaction receipt moves from
``prepared`` to ``committed`` last; the root/head mirrors and previous payload
make interrupted writes recoverable without blessing a partial generation.

This module intentionally imports no research-agent schema or runtime module;
identity-critical readers can share it without expanding the package's import
cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..canonical_json import (
    canonical_json_bytes as _shared_canonical_json_bytes,
    sha256_bytes as _sha256_bytes,
)

EVIDENCE_AUTHORITY_SCHEMA_VERSION = "easyicu.evidence_authority/1"
EVIDENCE_AUTHORITY_MARKER_SCHEMA_VERSION = "easyicu.evidence_authority_marker/1"
EVIDENCE_AUTHORITY_ROOT_MARKER_SCHEMA_VERSION = (
    "easyicu.evidence_authority_root_marker/1"
)
EVIDENCE_AUTHORITY_HEAD_SCHEMA_VERSION = "easyicu.evidence_authority_head/1"
EVIDENCE_AUTHORITY_TRANSACTION_SCHEMA_VERSION = (
    "easyicu.evidence_authority_transaction/1"
)
EVIDENCE_AUTHORITY_FILENAME = "evidence_authority.json"
EVIDENCE_AUTHORITY_PREVIOUS_FILENAME = "evidence_authority.previous.json"
EVIDENCE_AUTHORITY_MARKER_FILENAME = "evidence_authority_v1.marker.json"
EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME = ".easyicu_evidence_authority_v1.marker.json"
EVIDENCE_AUTHORITY_HEAD_FILENAME = ".easyicu_evidence_authority_head.json"
EVIDENCE_AUTHORITY_TRANSACTION_FILENAME = ".easyicu_evidence_authority_transaction.json"

_INDEX_FILENAME = "evidence_index.json"
_ALIASES_FILENAME = "evidence_aliases.json"
_NUMERIC_FILENAME = "numeric_claims.json"
_PROJECTION_FILENAMES = (_INDEX_FILENAME, _ALIASES_FILENAME, _NUMERIC_FILENAME)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class EvidenceAuthorityIntegrityError(RuntimeError):
    """Raised when the selected evidence generation cannot be verified."""


@dataclass(frozen=True)
class EvidenceAuthoritySnapshot:
    """Raw, schema-neutral view selected for one EvidenceStore instance."""

    source: str
    generation: Optional[int]
    payload_sha256: Optional[str]
    previous_payload_sha256: Optional[str]
    records: Tuple[Dict[str, Any], ...]
    aliases: Dict[str, str]
    numeric_claims: Tuple[Dict[str, Any], ...]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return _shared_canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceAuthorityIntegrityError(
            f"evidence authority contains non-canonical JSON: {exc}"
        ) from exc


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise EvidenceAuthorityIntegrityError(
            f"evidence {label} is missing, non-regular, or a symbolic link"
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise EvidenceAuthorityIntegrityError(
            f"evidence {label} is unreadable: {exc}"
        ) from exc


def _json_object(path: Path, *, label: str) -> object:
    payload = _regular_file_bytes(path, label=label)
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, TypeError) as exc:
        raise EvidenceAuthorityIntegrityError(
            f"evidence {label} is corrupt or invalid JSON: {exc}"
        ) from exc


def _validated_state(
    *,
    records: object,
    aliases: object,
    numeric_claims: object,
    label: str,
) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, str], Tuple[Dict[str, Any], ...]]:
    if not isinstance(records, list):
        raise EvidenceAuthorityIntegrityError(f"evidence {label} records is not a list")
    if not isinstance(aliases, dict):
        raise EvidenceAuthorityIntegrityError(
            f"evidence {label} aliases is not an object"
        )
    if not isinstance(numeric_claims, list):
        raise EvidenceAuthorityIntegrityError(
            f"evidence {label} numeric claims is not a list"
        )

    normalized_records: list[Dict[str, Any]] = []
    evidence_ids: set[str] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise EvidenceAuthorityIntegrityError(
                f"evidence {label} record {index} is not an object"
            )
        evidence_id = str(raw.get("evidence_id") or "").strip()
        if not evidence_id or evidence_id in evidence_ids:
            raise EvidenceAuthorityIntegrityError(
                f"evidence {label} has missing or duplicate evidence ids"
            )
        evidence_ids.add(evidence_id)
        normalized_records.append(dict(raw))

    normalized_aliases: Dict[str, str] = {}
    for raw_alias, raw_evidence_id in aliases.items():
        alias = str(raw_alias).strip()
        evidence_id = str(raw_evidence_id).strip()
        if not alias or not evidence_id or evidence_id not in evidence_ids:
            raise EvidenceAuthorityIntegrityError(
                f"evidence {label} aliases references unknown authority"
            )
        normalized_aliases[alias] = evidence_id

    normalized_claims: list[Dict[str, Any]] = []
    for index, raw in enumerate(numeric_claims):
        if not isinstance(raw, dict):
            raise EvidenceAuthorityIntegrityError(
                f"evidence {label} numeric claim {index} is not an object"
            )
        normalized_claims.append(dict(raw))

    return (
        tuple(normalized_records),
        normalized_aliases,
        tuple(normalized_claims),
    )


def build_evidence_authority_payload(
    *,
    generation: int,
    previous_payload_sha256: Optional[str],
    records: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, str],
    numeric_claims: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build one canonical, self-digesting full-state generation."""

    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise EvidenceAuthorityIntegrityError(
            "evidence authority generation must be a non-negative integer"
        )
    previous = (
        str(previous_payload_sha256).strip().lower()
        if previous_payload_sha256 is not None
        else None
    )
    if generation == 0 and previous is not None:
        raise EvidenceAuthorityIntegrityError(
            "generation zero cannot reference a previous evidence authority"
        )
    if generation > 0 and (previous is None or _SHA256_RE.fullmatch(previous) is None):
        raise EvidenceAuthorityIntegrityError(
            "later evidence generations require the prior authority digest"
        )
    normalized_records, normalized_aliases, normalized_claims = _validated_state(
        records=list(records),
        aliases=dict(aliases),
        numeric_claims=list(numeric_claims),
        label="candidate authority",
    )
    body: Dict[str, Any] = {
        "schema_version": EVIDENCE_AUTHORITY_SCHEMA_VERSION,
        "generation": generation,
        "previous_payload_sha256": previous,
        "records": list(normalized_records),
        "aliases": normalized_aliases,
        "numeric_claims": list(normalized_claims),
    }
    return {**body, "payload_sha256": _sha256_bytes(_canonical_json_bytes(body))}


def evidence_authority_text(payload: Mapping[str, Any]) -> str:
    """Return stable human-readable bytes for the selected generation."""

    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )


def validate_evidence_authority_payload(raw: object) -> EvidenceAuthoritySnapshot:
    if not isinstance(raw, dict):
        raise EvidenceAuthorityIntegrityError("evidence authority is not an object")
    expected = {
        "schema_version",
        "generation",
        "previous_payload_sha256",
        "records",
        "aliases",
        "numeric_claims",
        "payload_sha256",
    }
    if set(raw) != expected:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority has missing or unknown fields"
        )
    if raw.get("schema_version") != EVIDENCE_AUTHORITY_SCHEMA_VERSION:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority schema version is unsupported"
        )
    claimed = str(raw.get("payload_sha256") or "").strip().lower()
    body = {key: raw[key] for key in expected if key != "payload_sha256"}
    observed = _sha256_bytes(_canonical_json_bytes(body))
    if claimed != observed:
        raise EvidenceAuthorityIntegrityError("evidence authority digest is invalid")
    rebuilt = build_evidence_authority_payload(
        generation=raw["generation"],
        previous_payload_sha256=raw["previous_payload_sha256"],
        records=raw["records"],
        aliases=raw["aliases"],
        numeric_claims=raw["numeric_claims"],
    )
    if rebuilt != raw:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority is not in canonical normalized form"
        )
    return EvidenceAuthoritySnapshot(
        source="authority",
        generation=int(raw["generation"]),
        payload_sha256=claimed,
        previous_payload_sha256=raw["previous_payload_sha256"],
        records=tuple(dict(item) for item in raw["records"]),
        aliases={str(key): str(value) for key, value in raw["aliases"].items()},
        numeric_claims=tuple(dict(item) for item in raw["numeric_claims"]),
    )


def projection_sha256(evidence_dir: Path) -> Dict[str, Optional[str]]:
    """Hash the exact legacy projection set without following symlinks."""

    digests: Dict[str, Optional[str]] = {}
    for filename in _PROJECTION_FILENAMES:
        path = evidence_dir / filename
        if not path.exists() and not path.is_symlink():
            digests[filename] = None
            continue
        digests[filename] = _sha256_bytes(_regular_file_bytes(path, label=filename))
    return digests


def build_evidence_authority_marker(
    projection_digests: Mapping[str, Optional[str]],
) -> Dict[str, Any]:
    if set(projection_digests) != set(_PROJECTION_FILENAMES):
        raise EvidenceAuthorityIntegrityError(
            "evidence authority marker projection set is incomplete"
        )
    normalized: Dict[str, Optional[str]] = {}
    for filename in _PROJECTION_FILENAMES:
        value = projection_digests[filename]
        if value is not None:
            value = str(value).strip().lower()
            if _SHA256_RE.fullmatch(value) is None:
                raise EvidenceAuthorityIntegrityError(
                    "evidence authority marker contains an invalid digest"
                )
        normalized[filename] = value
    body = {
        "schema_version": EVIDENCE_AUTHORITY_MARKER_SCHEMA_VERSION,
        "legacy_projection_sha256": normalized,
    }
    return {**body, "payload_sha256": _sha256_bytes(_canonical_json_bytes(body))}


def evidence_authority_marker_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )


def _validated_marker(raw: object) -> Dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "legacy_projection_sha256",
        "payload_sha256",
    }:
        raise EvidenceAuthorityIntegrityError("evidence authority marker is invalid")
    if raw.get("schema_version") != EVIDENCE_AUTHORITY_MARKER_SCHEMA_VERSION:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority marker schema is unsupported"
        )
    rebuilt = build_evidence_authority_marker(raw["legacy_projection_sha256"])
    if rebuilt != raw:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority marker digest is invalid"
        )
    return dict(raw)


def build_evidence_authority_root_marker(
    *,
    legacy_projection_sha256: Mapping[str, Optional[str]],
    selected_generation: Optional[int],
    selected_payload_sha256: Optional[str],
) -> Dict[str, Any]:
    """Build the external high-water mirror for the evidence selector.

    The migration baseline distinguishes a real legacy store from a deleted
    modern ledger. Once a generation is committed, the selected generation
    and digest make unilateral rollback of the inner authority plus ``head``
    detectable. This is defense against partial deletion/rollback; no set of
    local files can resist a writer that restores every anchor atomically.
    """

    baseline = build_evidence_authority_marker(legacy_projection_sha256)[
        "legacy_projection_sha256"
    ]
    if selected_generation is None:
        if selected_payload_sha256 is not None:
            raise EvidenceAuthorityIntegrityError(
                "unselected evidence root marker cannot contain a payload digest"
            )
        selected_digest = None
    else:
        if (
            isinstance(selected_generation, bool)
            or not isinstance(selected_generation, int)
            or selected_generation < 0
        ):
            raise EvidenceAuthorityIntegrityError(
                "evidence root marker generation must be a non-negative integer"
            )
        selected_digest = str(selected_payload_sha256 or "").strip().lower()
        if _SHA256_RE.fullmatch(selected_digest) is None:
            raise EvidenceAuthorityIntegrityError(
                "evidence root marker contains an invalid selected digest"
            )
    body = {
        "schema_version": EVIDENCE_AUTHORITY_ROOT_MARKER_SCHEMA_VERSION,
        "legacy_projection_sha256": baseline,
        "selected_generation": selected_generation,
        "selected_payload_sha256": selected_digest,
    }
    return {**body, "payload_sha256": _sha256_bytes(_canonical_json_bytes(body))}


def evidence_authority_root_marker_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )


def validate_evidence_authority_root_marker(raw: object) -> Dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "legacy_projection_sha256",
        "selected_generation",
        "selected_payload_sha256",
        "payload_sha256",
    }:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority root marker is invalid"
        )
    if raw.get("schema_version") != EVIDENCE_AUTHORITY_ROOT_MARKER_SCHEMA_VERSION:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority root marker schema is unsupported"
        )
    rebuilt = build_evidence_authority_root_marker(
        legacy_projection_sha256=raw["legacy_projection_sha256"],
        selected_generation=raw["selected_generation"],
        selected_payload_sha256=raw["selected_payload_sha256"],
    )
    if rebuilt != raw:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority root marker digest is invalid"
        )
    return dict(raw)


def build_evidence_authority_head(
    *,
    generation: int,
    payload_sha256: str,
) -> Dict[str, Any]:
    """Build the external staged selector for one candidate generation."""

    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise EvidenceAuthorityIntegrityError(
            "evidence authority head generation must be a non-negative integer"
        )
    digest = str(payload_sha256 or "").strip().lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority head contains an invalid payload digest"
        )
    body = {
        "schema_version": EVIDENCE_AUTHORITY_HEAD_SCHEMA_VERSION,
        "generation": generation,
        "payload_sha256": digest,
    }
    return {**body, "head_sha256": _sha256_bytes(_canonical_json_bytes(body))}


def evidence_authority_head_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )


def _validated_head(raw: object) -> Dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "generation",
        "payload_sha256",
        "head_sha256",
    }:
        raise EvidenceAuthorityIntegrityError("evidence authority head is invalid")
    if raw.get("schema_version") != EVIDENCE_AUTHORITY_HEAD_SCHEMA_VERSION:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority head schema is unsupported"
        )
    rebuilt = build_evidence_authority_head(
        generation=raw["generation"],
        payload_sha256=raw["payload_sha256"],
    )
    if rebuilt != raw:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority head digest is invalid"
        )
    return dict(raw)


def build_evidence_authority_transaction(
    *,
    state: str,
    from_generation: Optional[int],
    from_payload_sha256: Optional[str],
    candidate_generation: int,
    candidate_payload_sha256: str,
) -> Dict[str, Any]:
    """Build the durable prepare/commit receipt for one transition."""

    normalized_state = str(state or "").strip().lower()
    if normalized_state not in {"prepared", "committed"}:
        raise EvidenceAuthorityIntegrityError(
            "evidence transaction state must be prepared or committed"
        )

    if from_generation is None:
        if from_payload_sha256 is not None or candidate_generation != 0:
            raise EvidenceAuthorityIntegrityError(
                "bootstrap transaction coordinates are inconsistent"
            )
        from_digest = None
    else:
        if (
            isinstance(from_generation, bool)
            or not isinstance(from_generation, int)
            or from_generation < 0
            or candidate_generation != from_generation + 1
        ):
            raise EvidenceAuthorityIntegrityError(
                "evidence transaction generations are not consecutive"
            )
        from_digest = str(from_payload_sha256 or "").strip().lower()
        if _SHA256_RE.fullmatch(from_digest) is None:
            raise EvidenceAuthorityIntegrityError(
                "evidence transaction contains an invalid predecessor digest"
            )
    if (
        isinstance(candidate_generation, bool)
        or not isinstance(candidate_generation, int)
        or candidate_generation < 0
    ):
        raise EvidenceAuthorityIntegrityError(
            "evidence transaction candidate generation is invalid"
        )
    candidate_digest = str(candidate_payload_sha256 or "").strip().lower()
    if _SHA256_RE.fullmatch(candidate_digest) is None:
        raise EvidenceAuthorityIntegrityError(
            "evidence transaction contains an invalid candidate digest"
        )
    body = {
        "schema_version": EVIDENCE_AUTHORITY_TRANSACTION_SCHEMA_VERSION,
        "state": normalized_state,
        "from_generation": from_generation,
        "from_payload_sha256": from_digest,
        "candidate_generation": candidate_generation,
        "candidate_payload_sha256": candidate_digest,
    }
    return {**body, "payload_sha256": _sha256_bytes(_canonical_json_bytes(body))}


def evidence_authority_transaction_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )


def validate_evidence_authority_transaction(raw: object) -> Dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "state",
        "from_generation",
        "from_payload_sha256",
        "candidate_generation",
        "candidate_payload_sha256",
        "payload_sha256",
    }:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority transaction witness is invalid"
        )
    if raw.get("schema_version") != EVIDENCE_AUTHORITY_TRANSACTION_SCHEMA_VERSION:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority transaction schema is unsupported"
        )
    rebuilt = build_evidence_authority_transaction(
        state=raw["state"],
        from_generation=raw["from_generation"],
        from_payload_sha256=raw["from_payload_sha256"],
        candidate_generation=raw["candidate_generation"],
        candidate_payload_sha256=raw["candidate_payload_sha256"],
    )
    if rebuilt != raw:
        raise EvidenceAuthorityIntegrityError(
            "evidence authority transaction digest is invalid"
        )
    return dict(raw)


def _load_legacy_snapshot(
    evidence_dir: Path, *, source: str
) -> EvidenceAuthoritySnapshot:
    present = {
        filename
        for filename in _PROJECTION_FILENAMES
        if (evidence_dir / filename).exists() or (evidence_dir / filename).is_symlink()
    }
    if not present:
        return EvidenceAuthoritySnapshot(
            source=source,
            generation=None,
            payload_sha256=None,
            previous_payload_sha256=None,
            records=(),
            aliases={},
            numeric_claims=(),
        )
    valid_layouts = [
        {_INDEX_FILENAME, _ALIASES_FILENAME},
        {_INDEX_FILENAME, _ALIASES_FILENAME, _NUMERIC_FILENAME},
    ]
    if present not in valid_layouts:
        raise EvidenceAuthorityIntegrityError(
            f"evidence legacy authority has an incomplete file set: {sorted(present)}"
        )
    records = _json_object(evidence_dir / _INDEX_FILENAME, label="index")
    aliases = _json_object(evidence_dir / _ALIASES_FILENAME, label="aliases")
    numeric_claims: object = []
    if _NUMERIC_FILENAME in present:
        numeric_claims = _json_object(
            evidence_dir / _NUMERIC_FILENAME,
            label="numeric claims",
        )
    normalized_records, normalized_aliases, normalized_claims = _validated_state(
        records=records,
        aliases=aliases,
        numeric_claims=numeric_claims,
        label="legacy authority",
    )
    return EvidenceAuthoritySnapshot(
        source=source,
        generation=None,
        payload_sha256=None,
        previous_payload_sha256=None,
        records=normalized_records,
        aliases=normalized_aliases,
        numeric_claims=normalized_claims,
    )


def load_current_evidence_snapshot(root: Path) -> EvidenceAuthoritySnapshot:
    """Load exactly the selected generation; never scan or fall back."""

    run_root = Path(root).expanduser().resolve()
    evidence_dir = run_root / "evidence"
    root_marker_path = run_root / EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME
    head_path = run_root / EVIDENCE_AUTHORITY_HEAD_FILENAME
    transaction_path = run_root / EVIDENCE_AUTHORITY_TRANSACTION_FILENAME
    root_marker_exists = root_marker_path.exists() or root_marker_path.is_symlink()
    head_exists = head_path.exists() or head_path.is_symlink()
    transaction_exists = transaction_path.exists() or transaction_path.is_symlink()
    if evidence_dir.is_symlink():
        raise EvidenceAuthorityIntegrityError(
            "evidence authority directory is a symbolic link"
        )
    if not evidence_dir.exists():
        if root_marker_exists or head_exists or transaction_exists:
            raise EvidenceAuthorityIntegrityError(
                "modern evidence authority anchor exists but evidence is missing"
            )
        return _load_legacy_snapshot(evidence_dir, source="fresh")
    if not evidence_dir.is_dir():
        raise EvidenceAuthorityIntegrityError(
            "evidence authority path is not a directory"
        )

    authority_path = evidence_dir / EVIDENCE_AUTHORITY_FILENAME
    previous_path = evidence_dir / EVIDENCE_AUTHORITY_PREVIOUS_FILENAME
    marker_path = evidence_dir / EVIDENCE_AUTHORITY_MARKER_FILENAME
    authority_exists = authority_path.exists() or authority_path.is_symlink()
    previous_exists = previous_path.exists() or previous_path.is_symlink()
    marker_exists = marker_path.exists() or marker_path.is_symlink()

    root_marker: Optional[Dict[str, Any]] = None
    if root_marker_exists:
        root_marker = validate_evidence_authority_root_marker(
            _json_object(root_marker_path, label="root authority marker")
        )
    transaction: Optional[Dict[str, Any]] = None
    if transaction_exists:
        transaction = validate_evidence_authority_transaction(
            _json_object(transaction_path, label="authority transaction witness")
        )

    if head_exists:
        if root_marker is None or transaction is None:
            raise EvidenceAuthorityIntegrityError(
                "modern evidence authority lacks its root marker or transaction receipt"
            )
        if not authority_exists or not marker_exists:
            raise EvidenceAuthorityIntegrityError(
                "selected modern evidence authority files are missing"
            )
        marker = _validated_marker(_json_object(marker_path, label="authority marker"))
        if (
            marker["legacy_projection_sha256"]
            != root_marker["legacy_projection_sha256"]
        ):
            raise EvidenceAuthorityIntegrityError(
                "evidence authority markers identify different migration baselines"
            )
        head = _validated_head(_json_object(head_path, label="authority head"))
        current = validate_evidence_authority_payload(
            _json_object(authority_path, label="authority")
        )
        candidate_coordinate = (
            int(transaction["candidate_generation"]),
            str(transaction["candidate_payload_sha256"]),
        )
        from_coordinate = (
            transaction["from_generation"],
            transaction["from_payload_sha256"],
        )
        root_coordinate = (
            root_marker["selected_generation"],
            root_marker["selected_payload_sha256"],
        )
        head_coordinate = (int(head["generation"]), str(head["payload_sha256"]))
        current_coordinate = (
            int(current.generation),
            str(current.payload_sha256),
        )

        if transaction["state"] == "committed":
            if not (
                root_coordinate
                == head_coordinate
                == current_coordinate
                == candidate_coordinate
            ):
                raise EvidenceAuthorityIntegrityError(
                    "committed evidence authority selectors disagree or were rolled back"
                )
            if current.previous_payload_sha256 != transaction["from_payload_sha256"]:
                raise EvidenceAuthorityIntegrityError(
                    "committed evidence authority has the wrong predecessor"
                )
            return current

        allowed = {candidate_coordinate}
        if from_coordinate[0] is not None:
            allowed.add((int(from_coordinate[0]), str(from_coordinate[1])))
        if root_coordinate not in allowed and root_coordinate != (None, None):
            raise EvidenceAuthorityIntegrityError(
                "prepared evidence root selector has unrelated coordinates"
            )
        if head_coordinate not in allowed or current_coordinate not in allowed:
            raise EvidenceAuthorityIntegrityError(
                "prepared evidence authority contains unrelated staged coordinates"
            )
        if current_coordinate == candidate_coordinate and (
            current.previous_payload_sha256 != transaction["from_payload_sha256"]
        ):
            raise EvidenceAuthorityIntegrityError(
                "prepared evidence candidate has the wrong predecessor"
            )
        if from_coordinate == (None, None):
            current_digests = projection_sha256(evidence_dir)
            if current_digests != root_marker["legacy_projection_sha256"]:
                raise EvidenceAuthorityIntegrityError(
                    "prepared bootstrap changed the legacy migration baseline"
                )
            return _load_legacy_snapshot(
                evidence_dir, source="root_marker_legacy_prepared"
            )

        selected: EvidenceAuthoritySnapshot
        if current_coordinate == from_coordinate:
            selected = current
        else:
            selected = validate_evidence_authority_payload(
                _json_object(previous_path, label="previous authority")
            )
            if (
                selected.generation != from_coordinate[0]
                or selected.payload_sha256 != from_coordinate[1]
            ):
                raise EvidenceAuthorityIntegrityError(
                    "prepared evidence transition lacks its selected predecessor"
                )
        return EvidenceAuthoritySnapshot(
            source="authority_previous_recovery",
            generation=selected.generation,
            payload_sha256=selected.payload_sha256,
            previous_payload_sha256=selected.previous_payload_sha256,
            records=selected.records,
            aliases=selected.aliases,
            numeric_claims=selected.numeric_claims,
        )

    if root_marker is not None:
        if root_marker["selected_generation"] is not None:
            raise EvidenceAuthorityIntegrityError(
                "evidence authority head is missing after modern selection"
            )
        current_digests = projection_sha256(evidence_dir)
        if current_digests != root_marker["legacy_projection_sha256"]:
            raise EvidenceAuthorityIntegrityError(
                "evidence authority head is missing after the migration baseline changed"
            )
        source = "root_marker_legacy"
        if transaction is not None:
            if (
                transaction["state"] != "prepared"
                or transaction["from_generation"] is not None
                or transaction["from_payload_sha256"] is not None
                or transaction["candidate_generation"] != 0
            ):
                raise EvidenceAuthorityIntegrityError(
                    "bootstrap transaction receipt has invalid coordinates"
                )
            source = "root_marker_legacy_prepared"
        return _load_legacy_snapshot(evidence_dir, source=source)

    if authority_exists or previous_exists or marker_exists or transaction_exists:
        raise EvidenceAuthorityIntegrityError(
            "modern evidence authority files lack their permanent root anchor"
        )

    return _load_legacy_snapshot(evidence_dir, source="legacy")


__all__ = [
    "EVIDENCE_AUTHORITY_FILENAME",
    "EVIDENCE_AUTHORITY_HEAD_FILENAME",
    "EVIDENCE_AUTHORITY_MARKER_FILENAME",
    "EVIDENCE_AUTHORITY_PREVIOUS_FILENAME",
    "EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME",
    "EVIDENCE_AUTHORITY_ROOT_MARKER_SCHEMA_VERSION",
    "EVIDENCE_AUTHORITY_SCHEMA_VERSION",
    "EVIDENCE_AUTHORITY_TRANSACTION_FILENAME",
    "EVIDENCE_AUTHORITY_TRANSACTION_SCHEMA_VERSION",
    "EvidenceAuthorityIntegrityError",
    "EvidenceAuthoritySnapshot",
    "build_evidence_authority_head",
    "build_evidence_authority_marker",
    "build_evidence_authority_payload",
    "build_evidence_authority_root_marker",
    "build_evidence_authority_transaction",
    "evidence_authority_head_text",
    "evidence_authority_marker_text",
    "evidence_authority_root_marker_text",
    "evidence_authority_transaction_text",
    "evidence_authority_text",
    "load_current_evidence_snapshot",
    "projection_sha256",
    "validate_evidence_authority_root_marker",
    "validate_evidence_authority_transaction",
]
