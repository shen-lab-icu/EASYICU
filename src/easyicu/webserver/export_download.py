"""Controlled browser downloads for registered EasyICU export packages.

The registered-source id is the public coordinate.  Host paths stay inside
this owner, and only files named by the sealed export manifest are bundled.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from easyicu.webserver import sources, state_paths


MAX_EXPORT_DOWNLOAD_BYTES = 512 * 1024 * 1024


class ExportDownloadError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ExportDownloadBundle:
    path: Path
    filename: str
    source_id: str
    source_bytes: int
    file_count: int


def _source_for_id(source_id: str, registry: Mapping[str, Any]) -> Mapping[str, Any]:
    clean = str(source_id or "").strip()
    if not re.fullmatch(r"src_[a-f0-9]{12}", clean):
        raise ExportDownloadError(
            "registered_export_source_id_invalid",
            "The registered export download requires an exact source id.",
        )
    source = next(
        (
            row
            for row in (registry.get("sources") or [])
            if isinstance(row, Mapping)
            and row.get("ok")
            and str(row.get("id") or "") == clean
        ),
        None,
    )
    if source is None:
        raise ExportDownloadError(
            "registered_export_source_not_found",
            "The requested registered export is unavailable.",
            status_code=404,
        )
    return source


def _manifest_files(root: Path, manifest: Mapping[str, Any]) -> list[Path]:
    names: list[str] = ["_manifest.json"]
    if (root / "README.md").is_file():
        names.append("README.md")
    for field in ("readme", "feature_definitions_csv"):
        value = manifest.get(field)
        if isinstance(value, str) and value.strip():
            names.append(value)
    definitions = manifest.get("feature_definitions")
    if isinstance(definitions, str) and definitions.strip():
        names.append(definitions)
    elif isinstance(definitions, Mapping):
        for row in definitions.get("files") or []:
            if isinstance(row, Mapping) and row.get("file"):
                names.append(str(row["file"]))
    metadata = manifest.get("column_metadata")
    if isinstance(metadata, Mapping) and metadata.get("file"):
        names.append(str(metadata["file"]))
    for group in ("files", "definition_files"):
        for row in manifest.get(group) or []:
            if isinstance(row, Mapping) and row.get("file"):
                names.append(str(row["file"]))

    files: list[Path] = []
    root_resolved = root.resolve(strict=True)
    for name in dict.fromkeys(names):
        relative = Path(name)
        if relative.is_absolute() or not relative.parts or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ExportDownloadError(
                "registered_export_manifest_file_invalid",
                "The export manifest contains an unsafe file coordinate.",
            )
        candidate = root / relative
        try:
            candidate_is_symlink = candidate.is_symlink()
        except OSError as exc:
            raise ExportDownloadError(
                "registered_export_manifest_file_invalid",
                "The export manifest contains an unsafe file coordinate.",
            ) from exc
        if candidate_is_symlink:
            raise ExportDownloadError(
                "registered_export_manifest_symlink_blocked",
                "Registered export downloads do not follow symbolic links.",
            )
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ExportDownloadError(
                "registered_export_manifest_file_missing",
                "A file declared by the export manifest is unavailable.",
            ) from exc
        if root_resolved not in resolved.parents or not resolved.is_file():
            raise ExportDownloadError(
                "registered_export_manifest_file_invalid",
                "The export manifest contains an unsafe file coordinate.",
            )
        files.append(resolved)
    return files


def prepare_registered_export_bundle(
    source_id: str,
    *,
    registry: Optional[Mapping[str, Any]] = None,
    temp_dir: Optional[Path] = None,
    max_source_bytes: int = MAX_EXPORT_DOWNLOAD_BYTES,
) -> ExportDownloadBundle:
    """Build one temporary ZIP from the exact registered manifest allowlist."""

    current = registry if registry is not None else sources.load_registry()
    source = _source_for_id(source_id, current)
    root = Path(str(source.get("path") or ""))
    manifest_path = root / "_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportDownloadError(
            "registered_export_manifest_unreadable",
            "The registered export manifest cannot be read safely.",
        ) from exc
    if not isinstance(manifest, Mapping):
        raise ExportDownloadError(
            "registered_export_manifest_invalid",
            "The registered export manifest is invalid.",
        )

    files = _manifest_files(root, manifest)
    source_bytes = sum(path.stat().st_size for path in files)
    if source_bytes > max_source_bytes:
        raise ExportDownloadError(
            "registered_export_download_too_large",
            "This export is too large for a browser bundle; use the local Data Extraction workspace instead.",
            status_code=413,
        )

    bundle_dir = Path(temp_dir) if temp_dir is not None else state_paths.state_root() / "downloads"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f"easyicu-{source_id}-", suffix=".zip", dir=bundle_dir
    )
    os.close(descriptor)
    archive_path = Path(raw_path)
    try:
        with zipfile.ZipFile(
            archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in files:
                archive.write(path, arcname=path.relative_to(root.resolve(strict=True)))
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return ExportDownloadBundle(
        path=archive_path,
        filename=f"easyicu_export_{source_id}.zip",
        source_id=source_id,
        source_bytes=source_bytes,
        file_count=len(files),
    )


def iter_bundle_and_cleanup(bundle: ExportDownloadBundle, chunk_size: int = 1024 * 1024):
    try:
        with bundle.path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                yield chunk
    finally:
        bundle.path.unlink(missing_ok=True)


__all__ = [
    "ExportDownloadBundle",
    "ExportDownloadError",
    "iter_bundle_and_cleanup",
    "prepare_registered_export_bundle",
]
