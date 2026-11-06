"""Publication state of a native export directory (Owner: extraction export).

``extract_database`` writes each module's raw output into the export directory
first and only canonicalises it (stay id, dtypes, episode window, declared
bounds) when ``_publish_native_export_v2`` seals the package with a root
``_manifest.json``. An interrupted or failed run therefore leaves files that
look like a finished export but hold unpublished values.

Contract:

* ``begin_native_export`` writes ``_IN_PROGRESS`` before any module output and
  refuses a directory that already carries the marker (a stale partial run).
  The caller must choose a fresh directory or remove the stale one; nothing is
  cleaned automatically, so a partial run's evidence is never destroyed.
* ``finish_native_export`` removes the marker only after the root manifest has
  been written. A crash between the two leaves both, and readers refuse it.
* ``require_published_native_export`` is the reader check: an export carrying
  the marker, or lacking ``_manifest.json``, raises
  :class:`NativeExportUnpublishedError` (code ``native_export_unpublished``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Union

IN_PROGRESS_MARKER = "_IN_PROGRESS"
NATIVE_MANIFEST = "_manifest.json"
UNPUBLISHED_CODE = "native_export_unpublished"

#: ``reason`` values carried by :class:`NativeExportUnpublishedError`.
REASON_IN_PROGRESS = "in_progress_marker"
REASON_MANIFEST_MISSING = "manifest_missing"
REASON_STALE_PARTIAL = "stale_partial_export"

_MARKER_TEXT = (
    "EasyICU native export in progress or interrupted. The files in this "
    "directory are unpublished raw module outputs; do not use them.\n"
)


class NativeExportUnpublishedError(ValueError):
    """An export directory is not a published native export."""

    code = UNPUBLISHED_CODE

    def __init__(self, message: str, *, reason: str, path: Union[str, Path]) -> None:
        super().__init__(message)
        self.reason = reason
        self.path = str(path)


def in_progress_marker_path(export_dir: Union[str, Path]) -> Path:
    return Path(export_dir) / IN_PROGRESS_MARKER


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return  # Not supported on this platform (for example Windows).
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def begin_native_export(export_dir: Union[str, Path]) -> Path:
    """Mark ``export_dir`` as in progress; refuse a stale partial export."""

    root = Path(export_dir)
    marker = in_progress_marker_path(root)
    if os.path.lexists(marker):
        raise NativeExportUnpublishedError(
            f"This folder holds an unfinished export from an interrupted run: {root} "
            f"({IN_PROGRESS_MARKER} present). Choose a new output folder, or move "
            "this one aside after checking it. EasyICU does not delete or reuse it.",
            reason=REASON_STALE_PARTIAL,
            path=root,
        )
    root.mkdir(parents=True, exist_ok=True)
    with open(marker, "x", encoding="utf-8") as handle:
        handle.write(_MARKER_TEXT)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(root)
    return marker


def finish_native_export(export_dir: Union[str, Path]) -> None:
    """Remove the marker; the root manifest must already be in place."""

    root = Path(export_dir)
    if not (root / NATIVE_MANIFEST).is_file():
        raise NativeExportUnpublishedError(
            f"{root}: refusing to clear {IN_PROGRESS_MARKER} before {NATIVE_MANIFEST} exists",
            reason=REASON_MANIFEST_MISSING,
            path=root,
        )
    in_progress_marker_path(root).unlink(missing_ok=True)
    _fsync_directory(root)


def unpublished_reason(export_dir: Union[str, Path]) -> str | None:
    """Return why ``export_dir`` is not a published native export, or None."""

    root = Path(export_dir)
    if os.path.lexists(in_progress_marker_path(root)):
        return REASON_IN_PROGRESS
    if not (root / NATIVE_MANIFEST).is_file():
        return REASON_MANIFEST_MISSING
    return None


LEGACY_MANIFEST = "easyicu_export_manifest.json"
#: A prepared source proves itself by its own preparation receipt.
SOURCE_PREPARATION_MANIFESTS = (
    "conversion_manifest.json",
    "community_preparation_manifest.json",
)
#: Files only an EasyICU export writes: producer module manifests, the
#: content-addressed column-metadata sidecar and the feature definitions.
_EXPORT_ARTIFACT_GLOBS = (
    "*.manifest.json",
    "column_metadata.sha256-*.json",
    "feature_definitions.json",
    "feature_definitions.csv",
)
#: Module data files an export writes as ``{module}.{ext}``.
_MODULE_DATA_SUFFIXES = (".parquet", ".csv", ".csv.gz", ".xlsx")


def _has_root_manifest(root: Path) -> bool:
    return (root / NATIVE_MANIFEST).is_file() or (root / LEGACY_MANIFEST).is_file()


def _holds_export_files(root: Path, module_names: Iterable[str]) -> bool:
    try:
        if any(next(root.glob(pattern), None) is not None for pattern in _EXPORT_ARTIFACT_GLOBS):
            return True
        return any(
            (root / f"{module}{suffix}").exists()
            for module in module_names
            for suffix in _MODULE_DATA_SUFFIXES
        )
    except OSError:
        return False


def folder_unpublished_reason(
    export_dir: Union[str, Path], *, module_names: Iterable[str] = ()
) -> str | None:
    """Why a folder served as a data source is an unpublished export, or None.

    For entry points that accept both exports and sources. A published export
    has no marker and a root manifest (native, or the older
    ``easyicu_export_manifest.json`` package). A prepared source is recognised
    by its own preparation receipt. Absent both, a folder holding any export
    file (a ``{module}`` data file named after an extraction module, a
    producer manifest, the metadata sidecar or feature definitions) is an
    unpublished export: missing export sidecars never prove a source. Any
    other folder (a raw CSV source) is left to its own detection contract.
    """

    root = Path(export_dir)
    if os.path.lexists(in_progress_marker_path(root)):
        return REASON_IN_PROGRESS
    if _has_root_manifest(root):
        return None
    if any((root / name).is_file() for name in SOURCE_PREPARATION_MANIFESTS):
        return None
    if _holds_export_files(root, module_names):
        return REASON_MANIFEST_MISSING
    return None


def require_published_export_package(export_dir: Union[str, Path]) -> Path:
    """For export-only readers: no marker, and a native or legacy root manifest.

    Returns the root manifest path (native first).
    """

    root = Path(export_dir)
    require_no_in_progress_marker(root)
    for name in (NATIVE_MANIFEST, LEGACY_MANIFEST):
        if (root / name).is_file():
            return root / name
    raise NativeExportUnpublishedError(
        f"{root} is not a published export (no {NATIVE_MANIFEST} or {LEGACY_MANIFEST})",
        reason=REASON_MANIFEST_MISSING,
        path=root,
    )


def adopt_native_export(export_dir: Union[str, Path]) -> Path:
    """Take over an export folder for an explicit resume of the same transaction.

    Unlike :func:`begin_native_export`, an existing marker is not refused: the
    caller (an explicit ``--resume`` of the same candidate) is the transaction
    that left it. A folder without a marker gets one before any mutation.
    """

    root = Path(export_dir)
    marker = in_progress_marker_path(root)
    if os.path.lexists(marker):
        if marker.is_symlink() or not marker.is_file():
            raise NativeExportUnpublishedError(
                f"{root}: {IN_PROGRESS_MARKER} is not a regular file",
                reason=REASON_IN_PROGRESS,
                path=root,
            )
        return marker
    return begin_native_export(root)


def require_no_in_progress_marker(export_dir: Union[str, Path]) -> None:
    """Refuse an export directory that carries the in-progress marker.

    For readers that also accept the older ``easyicu_export_manifest.json``
    packages: the marker check applies to every package kind.
    """

    root = Path(export_dir)
    if os.path.lexists(in_progress_marker_path(root)):
        raise NativeExportUnpublishedError(
            f"{root} is an unpublished native export ({IN_PROGRESS_MARKER} present)",
            reason=REASON_IN_PROGRESS,
            path=root,
        )


def require_published_native_export(export_dir: Union[str, Path]) -> Path:
    """Return the root manifest path of a published native export, else raise."""

    root = Path(export_dir)
    reason = unpublished_reason(root)
    if reason == REASON_IN_PROGRESS:
        require_no_in_progress_marker(root)
    if reason == REASON_MANIFEST_MISSING:
        raise NativeExportUnpublishedError(
            f"{root} is not a published native export ({NATIVE_MANIFEST} missing)",
            reason=REASON_MANIFEST_MISSING,
            path=root,
        )
    return root / NATIVE_MANIFEST
