"""Bounded, digest-pinned previews for registered Research Agent evidence.

This module owns the browser-preview boundary for evidence records.  Callers
provide an evidence id and the digest already shown to the user; they never
provide a host path.  The owner resolves the record from ``evidence_index``,
rehashes the file, and dispatches to a small read-only renderer contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Dict, Mapping


SCHEMA_VERSION = "easyicu.web-evidence-preview/1"
MAX_PREVIEW_BYTES = 512 * 1024
MAX_TABLE_ROWS = 100
MAX_TABLE_COLUMNS = 24

_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_HOST_PATH = re.compile(
    r"(?:file://|(?<![A-Za-z0-9])/(?:Users|home|private|tmp|var|etc|opt|Volumes)/|"
    r"\b[A-Za-z]:\\)",
    re.I,
)
_CODE_SUFFIXES = {".py", ".r", ".sql", ".jl", ".sh", ".js", ".ts"}
_RAW_TABLE_SUFFIXES = {".parquet", ".feather", ".arrow", ".ipc"}
_LANGUAGES = {
    ".py": "python",
    ".r": "r",
    ".sql": "sql",
    ".jl": "julia",
    ".sh": "shell",
    ".js": "javascript",
    ".ts": "typescript",
}


class EvidencePreviewError(ValueError):
    """Stable, owner-attributable evidence preview failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)


def is_identifier_column(name: Any) -> bool:
    """Return whether a table column is a direct patient/stay identifier."""

    token = re.sub(r"[^a-z0-9]+", "", str(name or "").lower())
    return token in {
        "stayid",
        "subjectid",
        "patientid",
        "hadmid",
        "icustayid",
        "patientunitstayid",
        "recordid",
        "mrn",
    }


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidencePreviewError(
            "evidence_preview_file_unavailable",
            "The registered evidence file is unavailable or is a symbolic link.",
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise EvidencePreviewError(
                "evidence_preview_file_not_regular",
                "Only regular registered evidence files can be previewed.",
            )
        if info.st_size > max_bytes:
            raise EvidencePreviewError(
                "evidence_preview_file_too_large",
                "The registered evidence file exceeds the bounded preview size.",
            )
        chunks = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise EvidencePreviewError(
                "evidence_preview_file_too_large",
                "The registered evidence file exceeds the bounded preview size.",
            )
        return raw
    finally:
        os.close(descriptor)


def _record_path(run_dir: Path, relative_path: Any) -> Path:
    text = str(relative_path or "").strip().replace("\\", "/")
    candidate = Path(text)
    if (
        not text
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise EvidencePreviewError(
            "evidence_preview_path_invalid",
            "The evidence registry contains an invalid relative path.",
        )
    resolved_root = run_dir.resolve()
    current = resolved_root
    for part in candidate.parts:
        current = current / part
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise EvidencePreviewError(
                    "evidence_preview_symlink_forbidden",
                    "Symbolic links are not allowed in registered evidence paths.",
                )
        except FileNotFoundError:
            break
    resolved = (resolved_root / candidate).resolve(strict=False)
    if resolved_root not in resolved.parents:
        raise EvidencePreviewError(
            "evidence_preview_path_escape",
            "The evidence registry path escapes the selected run.",
        )
    return resolved


def _load_record(run_dir: Path, evidence_id: str) -> Mapping[str, Any]:
    index_path = run_dir / "evidence" / "evidence_index.json"
    try:
        raw = _read_regular_file(index_path, max_bytes=4 * 1024 * 1024)
        records = json.loads(raw.decode("utf-8"))
    except EvidencePreviewError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidencePreviewError(
            "evidence_preview_index_invalid",
            "The evidence registry is not valid UTF-8 JSON.",
        ) from exc
    if not isinstance(records, list):
        raise EvidencePreviewError(
            "evidence_preview_index_invalid",
            "The evidence registry must be a list of evidence records.",
        )
    matches = [
        row
        for row in records
        if isinstance(row, Mapping) and row.get("evidence_id") == evidence_id
    ]
    if len(matches) != 1:
        code = (
            "evidence_preview_not_found"
            if not matches
            else "evidence_preview_id_ambiguous"
        )
        raise EvidencePreviewError(
            code, "The requested evidence id is not unique in this run."
        )
    return matches[0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidencePreviewError(
            "evidence_preview_file_unavailable",
            "The registered evidence file is unavailable or is a symbolic link.",
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise EvidencePreviewError(
                "evidence_preview_file_not_regular",
                "Only regular registered evidence files can be previewed.",
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _base_payload(record: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": str(record.get("evidence_id") or ""),
        "sha256": str(record.get("sha256") or ""),
        "kind": str(record.get("kind") or "artifact"),
        "role": str(record.get("produced_by_step") or ""),
        "description": str(record.get("description") or "")[:500],
        "display_name": path.name,
    }


def _metadata_only(
    base: Mapping[str, Any], *, reason: str, size: int | None = None
) -> Dict[str, Any]:
    return {
        **base,
        "renderer": "metadata",
        "previewable": False,
        "withheld_reason": reason,
        "bytes": size,
    }


def _decode_text(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidencePreviewError(
            "evidence_preview_encoding_unsupported",
            "Only UTF-8 text evidence can be previewed.",
        ) from exc
    if _HOST_PATH.search(text):
        raise EvidencePreviewError(
            "evidence_preview_host_path_detected",
            "The evidence text contains a host path and was withheld.",
        )
    return text


def _code_preview(base: Mapping[str, Any], path: Path, raw: bytes) -> Dict[str, Any]:
    text = _decode_text(raw)
    return {
        **base,
        "renderer": "code",
        "previewable": True,
        "language": _LANGUAGES.get(path.suffix.lower(), "text"),
        "line_count": len(text.splitlines()),
        "text": text,
    }


def _json_preview(base: Mapping[str, Any], raw: bytes) -> Dict[str, Any]:
    text = _decode_text(raw)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvidencePreviewError(
            "evidence_preview_json_invalid",
            "The registered evidence file is not valid JSON.",
        ) from exc
    return {**base, "renderer": "json", "previewable": True, "value": value}


def _csv_preview(base: Mapping[str, Any], path: Path, raw: bytes) -> Dict[str, Any]:
    text = _decode_text(raw)
    try:
        rows = csv.reader(text.splitlines())
        source_headers = next(rows, [])
        if any(is_identifier_column(header) for header in source_headers):
            return _metadata_only(
                base,
                reason="direct_identifier_columns_withheld",
                size=path.stat().st_size,
            )
        headers = source_headers[:MAX_TABLE_COLUMNS]
        values = [
            list(row[:MAX_TABLE_COLUMNS])
            for _, row in zip(range(MAX_TABLE_ROWS + 1), rows)
        ]
    except csv.Error as exc:
        raise EvidencePreviewError(
            "evidence_preview_csv_invalid",
            "The registered evidence file is not valid CSV.",
        ) from exc
    return {
        **base,
        "renderer": "table",
        "previewable": True,
        "headers": headers,
        "rows": values[:MAX_TABLE_ROWS],
        "rows_truncated": len(values) > MAX_TABLE_ROWS,
        "columns_truncated": len(source_headers) > MAX_TABLE_COLUMNS,
    }


def build_evidence_preview(
    run_dir: str | Path, evidence_id: str, expected_sha256: str
) -> Dict[str, Any]:
    """Build one safe browser projection from the run-owned evidence registry."""

    clean_id = str(evidence_id or "").strip()
    clean_expected = str(expected_sha256 or "").strip().lower()
    if not _EVIDENCE_ID.fullmatch(clean_id):
        raise EvidencePreviewError(
            "evidence_preview_id_invalid", "The evidence id format is invalid."
        )
    if not _SHA256.fullmatch(clean_expected):
        raise EvidencePreviewError(
            "evidence_preview_sha_invalid", "A valid expected SHA-256 is required."
        )
    root = Path(run_dir).resolve()
    record = _load_record(root, clean_id)
    registered_sha = str(record.get("sha256") or "").strip().lower()
    if registered_sha != clean_expected:
        raise EvidencePreviewError(
            "evidence_preview_sha_mismatch",
            "The requested digest does not match the registered evidence record.",
        )
    path = _record_path(root, record.get("relative_path"))
    base = _base_payload(record, path)
    size = path.stat().st_size
    suffix = path.suffix.lower()
    kind = str(record.get("kind") or "").lower()
    if suffix in _RAW_TABLE_SUFFIXES or size > MAX_PREVIEW_BYTES:
        if _sha256_file(path) != registered_sha:
            raise EvidencePreviewError(
                "evidence_preview_digest_mismatch",
                "The evidence file digest no longer matches its registry record.",
            )
        reason = (
            "patient_level_rows_withheld"
            if suffix in _RAW_TABLE_SUFFIXES
            else "preview_size_limit"
        )
        return _metadata_only(base, reason=reason, size=size)
    raw = _read_regular_file(path, max_bytes=MAX_PREVIEW_BYTES)
    if hashlib.sha256(raw).hexdigest() != registered_sha:
        raise EvidencePreviewError(
            "evidence_preview_digest_mismatch",
            "The evidence file digest no longer matches its registry record.",
        )
    try:
        if kind == "code" and suffix in _CODE_SUFFIXES:
            return _code_preview(base, path, raw)
        if suffix == ".json" and kind in {"statistic", "log", "table"}:
            return _json_preview(base, raw)
        if suffix == ".csv" and kind == "table":
            result_owned = bool(record.get("script_evidence_id")) and str(
                record.get("produced_by_step") or ""
            ) not in {"", "cohort_definition"}
            if result_owned:
                return _csv_preview(base, path, raw)
            return _metadata_only(
                base, reason="non_result_table_rows_withheld", size=size
            )
    except EvidencePreviewError as exc:
        if exc.code in {
            "evidence_preview_host_path_detected",
            "evidence_preview_encoding_unsupported",
        }:
            return _metadata_only(base, reason=exc.code, size=size)
        raise
    return _metadata_only(base, reason="unsupported_evidence_type", size=size)
