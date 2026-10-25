"""Registry-bound aggregate figure formats for the governed run-file owner."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from easyicu.webserver import run_refs
from easyicu.webserver.figure_presentation import verified_presentation_gallery
from easyicu.webserver.research_evidence_preview import (
    EvidencePreviewError,
    _load_records,
    _read_regular_file,
    _record_path,
    _registered_evidence_root,
)

_MEDIA_TYPES = {
    "pdf": "application/pdf", "svg": "image/svg+xml",
    "tiff": "image/tiff", "png": "image/png",
}
_MAX_BYTES = 16 * 1024 * 1024
_LOCATOR_FIELDS = {"evidence_id", "sha256", "kind"}


def _path(root: Path, relative: Any) -> Path:
    # Reject noncanonical spellings before pathlib can collapse them.
    if not isinstance(relative, str) or "\\" in relative or any(
        part in {"", ".", ".."} for part in relative.split("/")
    ):
        raise ValueError("invalid figure path")
    return _record_path(root, relative)


def _records(root: Path) -> list[Mapping[str, Any]]:
    _path(root, "evidence/evidence_index.json")
    return _load_records(root)


def _gallery(root: Path) -> Mapping[str, Any]:
    raw = _read_regular_file(_path(root, "figure_gallery.json"), max_bytes=4 * 1024 * 1024)
    gallery = json.loads(raw.decode("utf-8"))
    if not isinstance(gallery, Mapping):
        raise ValueError("invalid figure gallery")
    return verified_presentation_gallery(root, gallery) or gallery


def _rows(gallery: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    figures = gallery.get("figures")
    return [row for row in figures if isinstance(row, Mapping)] if isinstance(figures, list) else []


def figure_export_locators(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Project declared formats only after verifying aliases and registry bytes."""
    exports = row.get("exports")
    if not isinstance(exports, Mapping):
        return {}
    try:
        records = _records(root)
        image = _path(root, row.get("relative_path"))
    except (EvidencePreviewError, OSError, ValueError):
        return {}
    locators = {}
    for fmt in _MEDIA_TYPES:
        try:
            path = _path(root, exports.get(fmt))
            if path.suffix.lower() != f".{fmt}" or path.with_suffix(".png") != image:
                continue
            raw = _read_regular_file(path, max_bytes=_MAX_BYTES)
            if not raw:
                continue
            digest = hashlib.sha256(raw).hexdigest()
            for record in records:
                evidence_id = str(record.get("evidence_id") or "")
                if (record.get("kind") != "figure" or record.get("sha256") != digest
                        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", evidence_id)
                        or sum(item.get("evidence_id") == evidence_id for item in records) != 1):
                    continue
                registered = _path(root, record.get("relative_path"))
                if registered.suffix.lower() != f".{fmt}":
                    continue
                verified = _read_regular_file(registered, max_bytes=_MAX_BYTES)
                if hashlib.sha256(verified).hexdigest() == digest:
                    locators[fmt] = {
                        "evidence_id": evidence_id, "sha256": digest, "kind": "figure",
                    }
                    break
        except (EvidencePreviewError, OSError, ValueError):
            continue
    return locators


def project_gallery_exports(run_dir: Path, gallery: Mapping[str, Any]) -> dict[str, Any]:
    """Refresh browser locators, including a wrapper's declared pipeline gallery.

    A persisted locator is never download authority. The canonical gallery and
    evidence registry are rechecked on each read; host paths stay on the server.
    """
    projected = dict(gallery)
    sources: dict[str, Mapping[str, Any]] = {}
    root = _registered_evidence_root(run_dir)
    try:
        root.relative_to(run_dir.resolve())
        source = _gallery(root)
        sources = {
            str(row.get("relative_path") or ""): row
            for row in _rows(source)
        }
    except (EvidencePreviewError, OSError, ValueError):
        pass
    projected["figures"] = [
        {**row, "export_locators": figure_export_locators(root, sources.get(str(row.get("relative_path") or ""), {}))}
        for row in _rows(gallery)
    ]
    projected["run_ref"] = run_refs.issue(str(run_dir))
    return projected


def read_figure_export_bytes(run_dir: Path, locator: Any) -> dict[str, Any]:
    """Resolve one advertised figure identity; callers must first govern the run."""
    def refused(code: str) -> dict[str, Any]:
        return {"ok": False, "error": f"figure_export_{code}"}

    if not isinstance(locator, Mapping):
        return refused("locator_invalid")
    fmt = locator.get("format")
    if not isinstance(fmt, str) or fmt not in _MEDIA_TYPES:
        return refused("format_not_allowed")
    if (set(locator) != _LOCATOR_FIELDS | {"format"} or locator.get("kind") != "figure"
            or not isinstance(locator.get("evidence_id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", locator["evidence_id"])
            or not isinstance(locator.get("sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", locator["sha256"])):
        return refused("locator_invalid")
    root = _registered_evidence_root(run_dir)
    try:
        root.relative_to(run_dir.resolve())
        records = _records(root)
        matches = [record for record in records if record.get("evidence_id") == locator["evidence_id"]]
        if len(matches) != 1 or matches[0].get("kind") != "figure":
            return refused("not_validated")
        record = matches[0]
        if record.get("sha256") != locator["sha256"]:
            return refused("digest_mismatch")
        path = _path(root, record.get("relative_path"))
        if path.suffix.lower() != f".{fmt}":
            return refused("format_not_allowed")
        raw = _read_regular_file(path, max_bytes=_MAX_BYTES)
        if hashlib.sha256(raw).hexdigest() != locator["sha256"]:
            return refused("digest_mismatch")
    except (EvidencePreviewError, OSError, ValueError):
        return refused("path_unsafe")
    try:
        projected = project_gallery_exports(run_dir, _gallery(run_dir))
    except (EvidencePreviewError, OSError, ValueError):
        return refused("not_validated")
    identity = {key: locator[key] for key in _LOCATOR_FIELDS}
    if not any(row["export_locators"].get(fmt) == identity for row in projected["figures"]):
        return refused("not_validated")
    return {
        "ok": True, "name": f"{locator['evidence_id']}.{fmt}",
        "content": raw, "media_type": _MEDIA_TYPES[fmt],
    }
