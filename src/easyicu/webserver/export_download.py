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
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Mapping, Optional

from easyicu.export_publication import (
    INTEGRITY_UNRECORDED,
    UNPUBLISHED_CODE,
    ExportMemberIntegrityError,
    ExportMemberUnreadableError,
    NativeExportUnpublishedError,
    export_relative_path,
    integrity_status,
    member_unreadable,
    refusal_message,
    require_listed_members_present,
    require_published_native_export,
    verify_member_integrity,
)

from easyicu.webserver import sources, state_paths
from easyicu.webserver.hosted import require_instance_path, require_path


INTEGRITY_RECEIPT_NAME = "EASYICU_MEMBER_INTEGRITY.json"


def _require_readable(path: Path) -> None:
    """An unverified member must at least parse (footer / header)."""

    try:
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            _ = pq.ParquetFile(path).metadata
        elif path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig") as handle:
                handle.readline()
    except Exception as exc:
        raise member_unreadable(path, exc) from exc


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
    #: ``{member: verified|unrecorded}``; also written into the archive.
    member_integrity: Mapping[str, str] = dataclass_field(default_factory=dict)


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
    require_path(root, instance_owned=True)
    manifest_path = root / "_manifest.json"
    require_path(manifest_path, instance_owned=True)
    try:
        require_published_native_export(root)
    except NativeExportUnpublishedError as exc:
        raise ExportDownloadError(
            UNPUBLISHED_CODE,
            "The registered export is unpublished (an extraction was interrupted).",
            status_code=409,
        ) from exc
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
    # Never zip a module file that differs from, or is missing from, what the
    # manifest published. An explicitly legacy manifest without digests is
    # downloadable only outside formal scope, after a format-readability
    # check, and the archive states that its members are unverified.
    member_integrity: dict[str, str] = {}
    try:
        require_listed_members_present(root, dict(manifest))
        for row in manifest.get("files") or []:
            if not (isinstance(row, Mapping) and row.get("file")):
                continue
            name = str(row["file"])
            status = verify_member_integrity(root, name, manifest=dict(manifest))
            if status == INTEGRITY_UNRECORDED:
                _require_readable(root / name)
            member_integrity[name] = status
    except (ExportMemberIntegrityError, ExportMemberUnreadableError) as exc:
        # Exactly "integrity unrecorded · re-export to verify" when no digest
        # was ever recorded; the path is relative to the user's export root.
        raise ExportDownloadError(
            exc.code,
            f"{refusal_message(exc)}: {export_relative_path(root, exc.member)}",
            status_code=409,
        ) from exc
    source_bytes = sum(path.stat().st_size for path in files)
    if source_bytes > max_source_bytes:
        raise ExportDownloadError(
            "registered_export_download_too_large",
            "This export is too large for a browser bundle; use the local Data Extraction workspace instead.",
            status_code=413,
        )

    bundle_dir = Path(temp_dir) if temp_dir is not None else state_paths.state_root() / "downloads"
    require_instance_path(bundle_dir)
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
            archive.writestr(INTEGRITY_RECEIPT_NAME, json.dumps(
                {**integrity_status(member_integrity.values(), manifest=dict(manifest)),
                 "members": member_integrity},
                indent=2, sort_keys=True,
            ))
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return ExportDownloadBundle(
        path=archive_path,
        filename=f"easyicu_export_{source_id}.zip",
        source_id=source_id,
        source_bytes=source_bytes,
        file_count=len(files),
        member_integrity=member_integrity,
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
