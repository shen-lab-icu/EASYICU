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
from typing import Union

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
#: Files only an EasyICU export writes: producer module manifests, the
#: content-addressed column-metadata sidecar and the feature definitions. A raw
#: or prepared source folder holds none of them.
_EXPORT_ARTIFACT_GLOBS = (
    "*.manifest.json",
    "column_metadata.sha256-*.json",
    "feature_definitions.json",
    "feature_definitions.csv",
)


def is_export_folder(export_dir: Union[str, Path]) -> bool:
    """Whether ``export_dir`` is (or was being written as) an EasyICU export."""

    root = Path(export_dir)
    for name in (IN_PROGRESS_MARKER, NATIVE_MANIFEST, LEGACY_MANIFEST):
        if os.path.lexists(root / name):
            return True
    try:
        return any(next(root.glob(pattern), None) is not None for pattern in _EXPORT_ARTIFACT_GLOBS)
    except OSError:
        return False


def folder_unpublished_reason(export_dir: Union[str, Path]) -> str | None:
    """Why a folder served as a data source is an unpublished export, or None.

    A raw or prepared source folder is never an export and passes. An export
    folder must carry no marker and a root manifest (native, or the older
    ``easyicu_export_manifest.json`` packages).
    """

    root = Path(export_dir)
    if os.path.lexists(in_progress_marker_path(root)):
        return REASON_IN_PROGRESS
    if not is_export_folder(root):
        return None
    if (root / NATIVE_MANIFEST).is_file() or (root / LEGACY_MANIFEST).is_file():
        return None
    return REASON_MANIFEST_MISSING


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
