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
REASON_AMBIGUOUS = "export_source_ambiguous"

_MARKER_TEXT = (
    "EasyICU native export in progress or interrupted. The files in this "
    "directory are unpublished raw module outputs; do not use them.\n"
)


class SourceReceiptInvalidError(ValueError):
    """A folder relies on a source preparation receipt that its owner rejects."""

    code = "source_receipt_invalid"

    def __init__(self, message: str, *, reason: str, path: Union[str, Path]) -> None:
        super().__init__(message)
        self.reason = reason
        self.path = str(path)


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
    ``easyicu_export_manifest.json`` package). A legitimate source folder
    holds only its own source tables. Without a root manifest, a folder
    holding any export file (a ``{module}`` data file named after an
    extraction module, a producer manifest, the metadata sidecar or feature
    definitions) is an unpublished export (``manifest_missing``), and
    ``export_source_ambiguous`` when a source preparation receipt sits beside
    it: missing export sidecars never prove a source, and a receipt never
    turns export files into one. A receipt that classifies a folder as a
    source must satisfy its owner's validator (``storage_layout`` for the
    conversion receipt, ``databases.detection`` for the community receipt),
    else :class:`SourceReceiptInvalidError` (``source_receipt_invalid``). A
    raw CSV folder (no receipt) is left to its own detection contract.
    """

    root = Path(export_dir)
    if os.path.lexists(in_progress_marker_path(root)):
        return REASON_IN_PROGRESS
    if _has_root_manifest(root):
        return None
    if _holds_export_files(root, module_names):
        # A legitimate source holds only its own source tables. A preparation
        # receipt beside export files (an export written into, or copied from,
        # a prepared source folder) cannot make them a source.
        if any(os.path.lexists(root / name) for name in SOURCE_PREPARATION_MANIFESTS):
            return REASON_AMBIGUOUS
        return REASON_MANIFEST_MISSING
    _validate_source_receipts(root)
    return None


def _validate_source_receipts(root: Path) -> None:
    """A receipt that classifies a folder as a source must satisfy its owner."""

    from .databases.detection import DatabaseDetectionError, validate_community_receipt
    from .storage_layout import StorageIntegrityError, validate_conversion_manifest

    conversion, community = SOURCE_PREPARATION_MANIFESTS
    try:
        if os.path.lexists(root / conversion):
            validate_conversion_manifest(root)
        if os.path.lexists(root / community):
            validate_community_receipt(root)
    except StorageIntegrityError as exc:
        raise SourceReceiptInvalidError(
            f"{root}: {exc}", reason=getattr(exc, "reason", "conversion_manifest_unreadable"),
            path=root,
        ) from exc
    except DatabaseDetectionError as exc:
        raise SourceReceiptInvalidError(f"{root}: {exc}", reason=exc.code, path=root) from exc


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


# --- member integrity ----------------------------------------------------------
#
# A published manifest records each module file's size and SHA-256
# (``parquet_bytes``/``parquet_sha256`` from extraction, ``bytes``/``sha256``
# from the web writer). Readers that consume a member call
# :func:`verify_member_integrity` (or, when they already hashed the bytes,
# :func:`require_member_digest`) so a truncated, corrupted or replaced file is
# refused instead of read. Formal release tools (EX-A01, EX-A03,
# reextract_native_export_v2, derivation_context) keep their own full check.
#
# Limits, stated plainly:
# * Cost: a member is hashed once per process per stat identity
#   ``(realpath, size, mtime_ns, ctime_ns, ino, dev)`` (``_DIGEST_MEMO``).
#   ctime cannot be set from userland, so any in-place rewrite changes the key;
#   a file changed within 2 s of its hash is never memoised (``_RACY_WINDOW_NS``)
#   because a coarse kernel clock can give a same-tick rewrite the same ctime.
#   The memo is a convenience, not a formal check: someone able to forge stat
#   fields (root, a filesystem without ctime, a raw block write) can make a
#   changed file reuse an old digest. Formal paths re-hash every time.
#   Drop the memo by making ``_member_sha256`` call ``_hash_file`` directly.
# * A legacy or pre-digest manifest that records no digest for any member
#   cannot be compared (``INTEGRITY_UNRECORDED``; older full exports even list
#   only part of their files). Readers still refuse an unreadable member. As
#   soon as a manifest records one digest, every consumed member must be listed
#   with one.

INTEGRITY_FAILED_CODE = "export_member_integrity_failed"
UNREADABLE_CODE = "export_member_unreadable"

REASON_MEMBER_UNRECORDED = "member_not_in_manifest"
REASON_DIGEST_UNRECORDED = "member_digest_not_recorded"
REASON_SIZE_MISMATCH = "member_size_mismatch"
REASON_SHA256_MISMATCH = "member_sha256_mismatch"
REASON_MANIFEST_UNREADABLE = "manifest_unreadable"

INTEGRITY_VERIFIED = "verified"
INTEGRITY_UNRECORDED = "unrecorded_legacy_manifest"

_DIGEST_KEYS = (("parquet_sha256", "parquet_bytes"), ("sha256", "bytes"))
_DIGEST_MEMO: dict[tuple, str] = {}
_DIGEST_MEMO_LIMIT = 4096


class ExportMemberIntegrityError(ValueError):
    """A consumed export member does not match its published manifest record."""

    code = INTEGRITY_FAILED_CODE

    def __init__(self, message: str, *, reason: str, path: Union[str, Path],
                 member: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.path = str(path)
        self.member = member


class ExportMemberUnreadableError(ValueError):
    """A published export member cannot be read (the cause is chained)."""

    code = UNREADABLE_CODE

    def __init__(self, message: str, *, path: Union[str, Path], member: str = "") -> None:
        super().__init__(message)
        self.path = str(path)
        self.member = member or Path(path).name


def member_unreadable(path: Union[str, Path], exc: BaseException) -> ExportMemberUnreadableError:
    """The typed error a reader raises (``from exc``) when a member cannot be read."""

    return ExportMemberUnreadableError(
        f"export member cannot be read: {Path(path).name} ({type(exc).__name__})", path=path
    )


def _hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


#: A file changed within this window of its hash is not memoised ("racily
#: clean", as in git's index): kernels stamp ctime from a coarse clock, so a
#: rewrite in the same tick can keep the same stat key.
_RACY_WINDOW_NS = 2_000_000_000


def _member_sha256(path: Path) -> str:
    import time

    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
           stat.st_ino, stat.st_dev)
    cached = _DIGEST_MEMO.get(key)
    if cached is not None:
        return cached
    started_ns = time.time_ns()
    digest = _hash_file(path)
    if max(stat.st_mtime_ns, stat.st_ctime_ns) < started_ns - _RACY_WINDOW_NS:
        if len(_DIGEST_MEMO) >= _DIGEST_MEMO_LIMIT:
            _DIGEST_MEMO.clear()
        _DIGEST_MEMO[key] = digest
    return digest


def member_digest_record(path: Union[str, Path]) -> dict:
    """The ``sha256``/``bytes`` a writer records for one freshly written member."""

    target = Path(path)
    return {"sha256": _hash_file(target), "bytes": target.stat().st_size}


def clear_integrity_memo() -> None:
    _DIGEST_MEMO.clear()


def recorded_member_digest(entry: object) -> tuple[str, int | None] | None:
    """The ``(sha256, size)`` a manifest entry records, or ``None``."""

    if not isinstance(entry, dict):
        return None
    for sha_key, size_key in _DIGEST_KEYS:
        sha = entry.get(sha_key)
        if isinstance(sha, str) and sha:
            size = entry.get(size_key)
            return sha, size if isinstance(size, int) and not isinstance(size, bool) else None
    return None


def manifest_member_entries(manifest: object) -> list[dict]:
    """Every member record of a native or legacy manifest (``file`` keyed)."""

    if not isinstance(manifest, dict):
        return []
    entries: list[dict] = []
    for key in ("files", "modules", "exported_files"):
        values = manifest.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict) and value.get("file"):
                entries.append(value)
            elif isinstance(value, str) and value:
                entries.append({"file": value})
    return entries


def require_member_digest(entries: Iterable[dict], member: str, *, sha256: str,
                          size: int, path: Union[str, Path]) -> str:
    """Compare already-computed bytes facts with the manifest record of ``member``."""

    entries = list(entries)
    relative = Path(member).as_posix()
    entry = next((e for e in entries if Path(str(e.get("file"))).as_posix() == relative), None)
    records_digests = any(recorded_member_digest(e) is not None for e in entries)
    if not records_digests:
        # A legacy or pre-digest manifest (older full exports list only part
        # of their files): nothing to compare. Readers still refuse an
        # unreadable member through ExportMemberUnreadableError.
        return INTEGRITY_UNRECORDED
    if entry is None:
        raise ExportMemberIntegrityError(
            f"{relative} is not listed in the export manifest",
            reason=REASON_MEMBER_UNRECORDED, path=path, member=relative,
        )
    recorded = recorded_member_digest(entry)
    if recorded is None:
        raise ExportMemberIntegrityError(
            f"the export manifest records no digest for {relative}",
            reason=REASON_DIGEST_UNRECORDED, path=path, member=relative,
        )
    expected_sha, expected_size = recorded
    if expected_size is not None and expected_size != size:
        raise ExportMemberIntegrityError(
            f"{relative} is {size} bytes; the manifest records {expected_size}",
            reason=REASON_SIZE_MISMATCH, path=path, member=relative,
        )
    if expected_sha != sha256:
        raise ExportMemberIntegrityError(
            f"{relative} does not match the SHA-256 its export manifest records",
            reason=REASON_SHA256_MISMATCH, path=path, member=relative,
        )
    return INTEGRITY_VERIFIED


def load_published_manifest(package_root: Union[str, Path]) -> dict:
    """The parsed root manifest of a published export (marker refused)."""

    import json

    manifest_path = require_published_export_package(package_root)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ExportMemberIntegrityError(
            f"{manifest_path.name} cannot be read", reason=REASON_MANIFEST_UNREADABLE,
            path=manifest_path, member=manifest_path.name,
        ) from exc
    if not isinstance(payload, dict):
        raise ExportMemberIntegrityError(
            f"{manifest_path.name} is not an object", reason=REASON_MANIFEST_UNREADABLE,
            path=manifest_path, member=manifest_path.name,
        )
    return payload


def verify_member_integrity(package_root: Union[str, Path], member: Union[str, Path],
                            *, manifest: dict | None = None) -> str:
    """Refuse a consumed member whose size or SHA-256 differs from the manifest.

    ``member`` is the file path or its path relative to ``package_root``.
    Returns :data:`INTEGRITY_VERIFIED`, or :data:`INTEGRITY_UNRECORDED` for a
    manifest that records no digests at all. Memoised per stat identity.
    """

    root = Path(package_root)
    payload = manifest if manifest is not None else load_published_manifest(root)
    candidate = Path(member)
    path = candidate if candidate.is_absolute() else root / candidate
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = candidate.as_posix()
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise member_unreadable(path, exc) from exc
    entries = manifest_member_entries(payload)
    # Cheap size refusal first; hash only when the record has a digest.
    entry = next((e for e in entries if Path(str(e.get("file"))).as_posix() == relative), None)
    recorded = recorded_member_digest(entry) if entry is not None else None
    sha = _member_sha256(path) if recorded is not None and (
        recorded[1] is None or recorded[1] == size) else ""
    return require_member_digest(entries, relative, sha256=sha, size=size, path=path)


def verify_package_members(package_root: Union[str, Path], members: Iterable[Union[str, Path]],
                           *, manifest: dict | None = None) -> dict:
    """Verify several members against one manifest read; returns the manifest."""

    payload = manifest if manifest is not None else load_published_manifest(package_root)
    for member in members:
        verify_member_integrity(package_root, member, manifest=payload)
    return payload
