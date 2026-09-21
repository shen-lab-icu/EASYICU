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
from typing import Any, Dict, Iterable, Mapping


SCHEMA_VERSION = "easyicu.web-evidence-preview/2"
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


def _load_records(run_dir: Path) -> list[Mapping[str, Any]]:
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
    return [row for row in records if isinstance(row, Mapping)]


def _registered_evidence_root(run_dir: str | Path) -> Path:
    """Resolve a wrapper run to its declared concrete pipeline run."""

    root = Path(run_dir).resolve()
    if (root / "evidence" / "evidence_index.json").is_file():
        return root
    try:
        raw = _read_regular_file(
            root / "source_run_manifest.json", max_bytes=4 * 1024 * 1024
        )
        manifest = json.loads(raw.decode("utf-8"))
    except (EvidencePreviewError, UnicodeDecodeError, json.JSONDecodeError):
        return root
    if not isinstance(manifest, Mapping):
        return root
    run_id = str(manifest.get("run_id") or "").strip()
    if not _EVIDENCE_ID.fullmatch(run_id):
        return root
    pipeline_root = root / "pipeline"
    candidate = pipeline_root / run_id
    try:
        if stat.S_ISLNK(os.lstat(pipeline_root).st_mode):
            return root
        if stat.S_ISLNK(os.lstat(candidate).st_mode):
            return root
        resolved_pipeline = pipeline_root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except FileNotFoundError:
        return root
    if (
        resolved_candidate.parent != resolved_pipeline
        or not resolved_candidate.is_dir()
        or not (resolved_candidate / "evidence" / "evidence_index.json").is_file()
    ):
        return root
    return resolved_candidate


def project_review_evidence_refs(
    run_dir: str | Path,
    references: Iterable[Any],
) -> list[Dict[str, Any]]:
    """Resolve scientific-review file labels to digest-pinned evidence records.

    Scientific readiness findings intentionally store compact, human-readable
    references such as ``reviewer_report.json`` or
    ``preplan_literature_bundle.json.screening_decisions``.  The browser must
    not turn those labels into paths.  This projection joins them against the
    immutable evidence registry and returns only the coordinates already
    accepted by :func:`build_evidence_preview`.
    """

    root = _registered_evidence_root(run_dir)
    try:
        records = _load_records(root)
    except EvidencePreviewError:
        return []

    candidates: list[tuple[Mapping[str, Any], str]] = []
    for record in records:
        evidence_id = str(record.get("evidence_id") or "").strip()
        digest = str(record.get("sha256") or "").strip().lower()
        relative_path = str(record.get("relative_path") or "").strip()
        if (
            not _EVIDENCE_ID.fullmatch(evidence_id)
            or not _SHA256.fullmatch(digest)
            or not relative_path
        ):
            continue
        basename = Path(relative_path).name
        published_name = basename.split("__", 1)[-1]
        candidates.append((record, published_name))

    projected: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw_reference in references:
        reference = str(raw_reference or "").strip()[:240]
        if not reference or reference in seen:
            continue
        # A suffix after the registered JSON filename is a field locator, not a
        # filesystem coordinate.  Preserve it as a JSON pointer hint only.
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+\.(?:json|md|csv|tsv|txt))(?:\.([A-Za-z0-9_.-]+))?",
            reference,
            re.I,
        )
        if not match:
            continue
        file_name = match.group(1)
        pointer_suffix = str(match.group(2) or "").strip(".")
        record = next(
            (row for row, published in candidates if published == file_name),
            None,
        )
        if record is None:
            continue
        seen.add(reference)
        evidence_id = str(record.get("evidence_id") or "")
        projected.append(
            {
                "reference": reference,
                "evidence_id": evidence_id,
                "sha256": str(record.get("sha256") or "").lower(),
                "kind": str(record.get("kind") or "artifact")[:80],
                "label": reference,
                "description": str(record.get("description") or "")[:500],
                **(
                    {"pointer": "/" + pointer_suffix.replace(".", "/")}
                    if pointer_suffix
                    else {}
                ),
            }
        )
    return projected


def _verified_code_locator(
    run_dir: Path,
    record: Mapping[str, Any],
    *,
    focus_token: str = "",
) -> Dict[str, Any] | None:
    """Project one digest-verified code record without exposing its run path."""

    if str(record.get("kind") or "").strip().lower() != "code":
        return None
    evidence_id = str(record.get("evidence_id") or "").strip()
    digest = str(record.get("sha256") or "").strip().lower()
    if not _EVIDENCE_ID.fullmatch(evidence_id) or not _SHA256.fullmatch(digest):
        return None
    try:
        path = _record_path(run_dir, record.get("relative_path"))
        if path.suffix.lower() not in _CODE_SUFFIXES or not path.is_file():
            return None
        if _sha256_file(path) != digest:
            return None
    except EvidencePreviewError:
        return None
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    locator = {
        "evidence_id": evidence_id,
        "sha256": digest,
        "display_name": _source_file_name(record, path),
        "language": _LANGUAGES.get(path.suffix.lower(), "text"),
        "description": str(record.get("description") or "")[:500],
        "produced_by_step": str(
            record.get("produced_by_step")
            or metadata.get("promoted_from_step_id")
            or ""
        )[:160],
        "producer": str(record.get("producer") or "")[:160],
        "generation_mode": str(record.get("generation_mode") or "")[:160],
    }
    clean_focus = str(focus_token or "").strip()
    if clean_focus and len(clean_focus) <= 160:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        needle = clean_focus.casefold()
        match = next(
            (index for index, line in enumerate(lines) if needle in line.casefold()),
            None,
        )
        if match is not None:
            locator["focus_start_line"] = max(1, match + 1 - 12)
            locator["focus_end_line"] = min(len(lines), match + 1 + 24)
    return locator


def project_figure_source_code(
    run_dir: str | Path, gallery: Mapping[str, Any]
) -> Dict[str, Any]:
    """Attach safe source-code locators to figure-gallery rows.

    The persisted gallery remains unchanged.  Supporting figures bind through
    their exact ``steps/<step_id>/outputs`` directory; promoted publication
    figures bind through the registered renderer's ``metadata.figure_id``.
    Ambiguous, missing, or digest-mismatched code fails closed by omitting the
    browser action.
    """

    projected = dict(gallery or {})
    figures = gallery.get("figures") if isinstance(gallery, Mapping) else None
    if not isinstance(figures, list):
        return projected
    root = _registered_evidence_root(run_dir)
    try:
        records = _load_records(root)
    except EvidencePreviewError:
        projected["figures"] = [dict(row) if isinstance(row, Mapping) else row for row in figures]
        return projected

    def code_for_figure_records(
        figure_records: list[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        if len(figure_records) != 1:
            return []
        figure_record = figure_records[0]
        metadata = figure_record.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        script_ids = {
            str(value or "").strip()
            for value in (
                figure_record.get("script_evidence_id"),
                metadata.get("script_evidence_id"),
            )
            if str(value or "").strip()
        }
        if script_ids:
            return [
                record
                for record in records
                if str(record.get("kind") or "").lower() == "code"
                and str(record.get("evidence_id") or "") in script_ids
            ]
        step_id = str(
            figure_record.get("produced_by_step")
            or metadata.get("step_id")
            or ""
        ).strip()
        if not step_id:
            return []
        return [
            record
            for record in records
            if str(record.get("kind") or "").lower() == "code"
            and str(record.get("produced_by_step") or "") == step_id
        ]

    def source_for(row: Mapping[str, Any]) -> Dict[str, Any] | None:
        relative_path = str(row.get("relative_path") or "").strip().replace("\\", "/")
        step_match = re.fullmatch(r"steps/([^/]+)/outputs/[^/]+", relative_path)
        candidates: list[Mapping[str, Any]] = []
        source_evidence_id = str(row.get("source_evidence_id") or "").strip()
        if source_evidence_id:
            candidates = code_for_figure_records(
                [
                    record
                    for record in records
                    if str(record.get("kind") or "").lower() == "figure"
                    and str(record.get("evidence_id") or "") == source_evidence_id
                ]
            )
        elif step_match:
            step_id = step_match.group(1)
            candidates = [
                record
                for record in records
                if str(record.get("kind") or "").lower() == "code"
                and str(record.get("produced_by_step") or "") == step_id
            ]
        else:
            figure_id = str(row.get("figure_id") or "").strip()
            if figure_id:
                candidates = [
                    record
                    for record in records
                    if str(record.get("kind") or "").lower() == "code"
                    and isinstance(record.get("metadata"), Mapping)
                    and str((record.get("metadata") or {}).get("figure_id") or "") == figure_id
                ]
                if not candidates:
                    candidates = code_for_figure_records(
                        [
                            record
                            for record in records
                            if str(record.get("kind") or "").lower() == "figure"
                            and isinstance(record.get("metadata"), Mapping)
                            and str((record.get("metadata") or {}).get("figure_id") or "") == figure_id
                            and str(record.get("relative_path") or "").lower().endswith(".png")
                        ]
                    )
        if len(candidates) != 1:
            return None
        candidate = candidates[0]
        candidate_metadata = candidate.get("metadata")
        candidate_metadata = (
            candidate_metadata if isinstance(candidate_metadata, Mapping) else {}
        )
        # Presentation galleries are allowed to omit their internal figure id.
        # The verified renderer record still carries the exact figure id (or, for
        # a planner step, the producing step id), so use that stable registry
        # metadata to open the source at the code that generated this figure.
        focus_token = next(
            (
                value
                for value in (
                    str(row.get("figure_id") or "").strip(),
                    str(candidate_metadata.get("figure_id") or "").strip(),
                    str(candidate.get("produced_by_step") or "").strip(),
                )
                if value
            ),
            "",
        )
        return _verified_code_locator(
            root,
            candidate,
            focus_token=focus_token,
        )

    projected_figures: list[Any] = []
    for item in figures:
        if not isinstance(item, Mapping):
            projected_figures.append(item)
            continue
        row = dict(item)
        source = source_for(item)
        if source:
            row["source_code"] = source
        projected_figures.append(row)
    projected["figures"] = projected_figures
    return projected


def _load_record(
    run_dir: Path, evidence_id: str
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    records = _load_records(run_dir)
    matches = [row for row in records if row.get("evidence_id") == evidence_id]
    if len(matches) != 1:
        code = (
            "evidence_preview_not_found"
            if not matches
            else "evidence_preview_id_ambiguous"
        )
        raise EvidencePreviewError(
            code, "The requested evidence id is not unique in this run."
        )
    return matches[0], records


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


def _source_file_name(record: Mapping[str, Any], path: Path) -> str:
    prefix = f"{str(record.get('evidence_id') or '')}__"
    return path.name[len(prefix) :] if path.name.startswith(prefix) else path.name


def _record_link(
    run_dir: Path,
    records: list[Mapping[str, Any]],
    evidence_id: Any,
    *,
    relation: str,
) -> Dict[str, Any]:
    clean_id = str(evidence_id or "").strip()
    matches = [row for row in records if str(row.get("evidence_id") or "") == clean_id]
    if len(matches) != 1:
        return {
            "relation": relation,
            "evidence_id": clean_id,
            "status": "unregistered" if not matches else "ambiguous",
        }
    record = matches[0]
    digest = str(record.get("sha256") or "").strip().lower()
    try:
        path = _record_path(run_dir, record.get("relative_path"))
        if not path.is_file() or not _SHA256.fullmatch(digest):
            raise EvidencePreviewError(
                "evidence_preview_lineage_unavailable",
                "The registered lineage file is unavailable.",
            )
    except EvidencePreviewError as exc:
        return {
            "relation": relation,
            "evidence_id": clean_id,
            "status": exc.code,
        }
    return {
        "relation": relation,
        "status": "registered",
        "evidence_id": clean_id,
        "kind": str(record.get("kind") or "artifact"),
        "description": str(record.get("description") or "")[:500],
        "sha256": digest,
        "display_name": _source_file_name(record, path),
        "relative_path": str(record.get("relative_path") or ""),
        "produced_by_step": str(record.get("produced_by_step") or ""),
        "producer": str(record.get("producer") or ""),
        "generation_mode": str(record.get("generation_mode") or ""),
    }


def _declared_lineage(
    run_dir: Path,
    record: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    declared: list[tuple[str, str]] = []
    script_evidence_id = str(record.get("script_evidence_id") or "").strip()
    if script_evidence_id:
        declared.append((script_evidence_id, "analysis_code"))
    for item in record.get("inputs") or []:
        clean_id = str(item or "").strip()
        if clean_id:
            declared.append((clean_id, "input_data"))
    links: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for evidence_id, relation in declared:
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        links.append(
            _record_link(
                run_dir,
                records,
                evidence_id,
                relation=relation,
            )
        )
    return links


def _run_authority(run_dir: Path, records: list[Mapping[str, Any]]) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "not_recorded", "links": []}
    try:
        raw = _read_regular_file(manifest_path, max_bytes=4 * 1024 * 1024)
        manifest = json.loads(raw.decode("utf-8"))
    except (EvidencePreviewError, UnicodeDecodeError, json.JSONDecodeError):
        return {"status": "invalid", "links": []}
    if not isinstance(manifest, Mapping):
        return {"status": "invalid", "links": []}
    code_version = manifest.get("code_version")
    code_version = code_version if isinstance(code_version, Mapping) else {}
    identity = manifest.get("execution_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    plan = manifest.get("current_plan_authority")
    plan = plan if isinstance(plan, Mapping) else {}
    links: list[Dict[str, Any]] = []
    plan_id = str(plan.get("evidence_id") or "").strip()
    if plan_id:
        plan_link = _record_link(
            run_dir, records, plan_id, relation="run_plan_authority"
        )
        declared_plan_sha = str(plan.get("sha256") or "").strip().lower()
        if (
            plan_link.get("status") == "registered"
            and declared_plan_sha
            and plan_link.get("sha256") != declared_plan_sha
        ):
            plan_link = {
                "relation": "run_plan_authority",
                "evidence_id": plan_id,
                "status": "authority_digest_mismatch",
            }
        links.append(plan_link)
    for evidence_id, relation in (
        ("research_context", "run_research_context"),
        ("cohort_locked", "run_cohort_authority"),
    ):
        if any(str(row.get("evidence_id") or "") == evidence_id for row in records):
            links.append(_record_link(run_dir, records, evidence_id, relation=relation))
    return {
        "status": "recorded",
        "run_id": str(manifest.get("run_id") or run_dir.name),
        "git_sha": str(code_version.get("git_sha") or ""),
        "git_branch": str(code_version.get("git_branch") or ""),
        "git_dirty": code_version.get("git_dirty"),
        "package_version": str(code_version.get("package_version") or ""),
        "runner": str(identity.get("runner") or ""),
        "runner_image_digest": str(identity.get("runner_image_digest") or ""),
        "network_policy": str(identity.get("network_policy") or ""),
        "prompt_pack_version": str(manifest.get("prompt_pack_version") or ""),
        "prompt_pack_sha256": str(identity.get("prompt_pack_sha256") or ""),
        "environment_identity_sha256": str(
            identity.get("environment_identity_sha256") or ""
        ),
        "input_authority_sha256": str(identity.get("input_authority_sha256") or ""),
        "execution_identity_sha256": str(identity.get("identity_sha256") or ""),
        "paper_eligible": identity.get("paper_eligible"),
        "links": links,
    }


def _base_payload(
    record: Mapping[str, Any],
    path: Path,
    *,
    lineage: list[Dict[str, Any]],
    run_authority: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": str(record.get("evidence_id") or ""),
        "sha256": str(record.get("sha256") or ""),
        "kind": str(record.get("kind") or "artifact"),
        "role": str(record.get("produced_by_step") or ""),
        "description": str(record.get("description") or "")[:500],
        "display_name": _source_file_name(record, path),
        "registered_name": path.name,
        "relative_path": str(record.get("relative_path") or ""),
        "producer": str(record.get("producer") or ""),
        "generation_mode": str(record.get("generation_mode") or ""),
        "prompt_pack_version": str(record.get("prompt_pack_version") or ""),
        "created_at": str(record.get("created_at") or ""),
        "declared_lineage": lineage,
        "run_authority": dict(run_authority),
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
    record, records = _load_record(root, clean_id)
    registered_sha = str(record.get("sha256") or "").strip().lower()
    if registered_sha != clean_expected:
        raise EvidencePreviewError(
            "evidence_preview_sha_mismatch",
            "The requested digest does not match the registered evidence record.",
        )
    path = _record_path(root, record.get("relative_path"))
    base = _base_payload(
        record,
        path,
        lineage=_declared_lineage(root, record, records),
        run_authority=_run_authority(root, records),
    )
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
