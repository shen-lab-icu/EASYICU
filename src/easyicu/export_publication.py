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
from contextlib import contextmanager
from contextvars import ContextVar
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
# * Unrecorded (supervisor policy, delta27e): a manifest whose writer never
#   recorded digests (no ``schema_version``, or the exact web format written
#   before 6064eaf6a, :data:`LEGACY_WEB_PRE_KIND`) cannot be compared. Only the
#   non-formal UI path accepts it, and every consumer shows and records
#   ``integrity: unrecorded`` per member ("integrity unrecorded · re-export to
#   verify"). Formal scope (:func:`formal_integrity_scope`, release tools,
#   benchmark inputs) refuses it. Readers still refuse an unreadable member.
# * Anti-downgrade: the decision is per manifest, from its writer contract
#   (:func:`manifest_requires_digests`), never from whether digests happen to
#   be present. A native v2 extraction manifest, or any manifest declaring
#   ``member_integrity``, must list every consumed member with a digest;
#   deleting the digest fields refuses (``member_digest_not_recorded``), as
#   does any manifest carrying ``export_kind`` or ``contract_revision``.
#   Limit: deleting ``export_kind``/``contract_revision``/``member_integrity``
#   and the digests from a manifest that has data_path, cohort_contract and
#   column_metadata forges the pre-6064eaf6a signature: visible as unrecorded
#   in the UI, refused by every formal path, never certified.

INTEGRITY_FAILED_CODE = "export_member_integrity_failed"
UNREADABLE_CODE = "export_member_unreadable"

REASON_MEMBER_UNRECORDED = "member_not_in_manifest"
REASON_DIGEST_UNRECORDED = "member_digest_not_recorded"
REASON_SIZE_MISMATCH = "member_size_mismatch"
REASON_SHA256_MISMATCH = "member_sha256_mismatch"
REASON_MANIFEST_UNREADABLE = "manifest_unreadable"
REASON_INTEGRITY_UNRECORDED = "member_integrity_unrecorded"
REASON_DIGEST_CONFLICT = "member_digest_conflict"
REASON_MEMBER_MISSING = "member_missing"
REASON_CHANGED_DURING_HASH = "member_changed_during_hash"
REASON_MEMBER_PATH_UNSAFE = "member_path_unsafe"

INTEGRITY_VERIFIED = "verified"
INTEGRITY_UNRECORDED = "unrecorded"
INTEGRITY_UNRECORDED_LABEL = "integrity unrecorded · re-export to verify"

#: Declared by every writer that records member digests (native extraction,
#: the web writer, EX-A02/A03 republication). Its presence makes digests
#: mandatory for that manifest.
MEMBER_INTEGRITY_KEY = "member_integrity"
MEMBER_INTEGRITY_DECLARATION = {"algorithm": "sha256", "scope": "every_member"}
_NATIVE_V2_SCHEMA = "easyicu_native_export_v2"
_FORMAL_ENV = "EASYICU_FORMAL_EXPORT_INTEGRITY"

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


#: The one explicitly identified legacy web format: the web writer before
#: 6064eaf6a (2026-10-24) labelled its exports ``easyicu_native_export_v2``
#: but recorded no ``export_kind``, no ``contract_revision`` and no member
#: digests. Supervisor ruling (r13exportsha delta27e, option a): exactly
#: this signature is legacy; any extra or missing field stays required.
LEGACY_WEB_PRE_KIND = "legacy_web_pre_6064eaf6a"
LEGACY_NO_SCHEMA = "legacy_no_schema_version"
_PRE_KIND_WEB_PRESENT = ("data_path", "cohort_contract", "column_metadata")
_PRE_KIND_WEB_ABSENT = ("export_kind", "contract_revision", MEMBER_INTEGRITY_KEY)


def _is_pre_kind_web_export(manifest: dict) -> bool:
    if manifest.get("schema_version") != _NATIVE_V2_SCHEMA:
        return False
    if not all(key in manifest for key in _PRE_KIND_WEB_PRESENT):
        return False
    if any(key in manifest for key in _PRE_KIND_WEB_ABSENT):
        return False
    # No member may carry a digest: a partly digested manifest is not this format.
    return not any(recorded_member_digest(entry) for entry in manifest_member_entries(manifest))


def legacy_manifest_format(manifest: object) -> str | None:
    """The explicitly identified legacy format of ``manifest``, or ``None``."""

    if not isinstance(manifest, dict) or MEMBER_INTEGRITY_KEY in manifest:
        return None
    if manifest.get("schema_version") is None:
        return LEGACY_NO_SCHEMA
    if _is_pre_kind_web_export(manifest):
        return LEGACY_WEB_PRE_KIND
    return None


def manifest_integrity_contract(manifest: object) -> str:
    """``required`` or ``legacy``: this manifest's writer contract for digests.

    Decided per manifest from its schema/version (anti-downgrade), never from
    whether digests happen to be present:

    * ``required``: a manifest declaring ``member_integrity``, or a native v2
      extraction manifest (it has always recorded ``parquet_sha256``).
    * ``legacy``, only for explicitly identified older formats: a manifest
      without ``schema_version`` (pre-v2 native, ``easyicu_export_manifest.json``
      full exports), or the web writer's format before 6064eaf6a
      (:data:`LEGACY_WEB_PRE_KIND`, exact signature only).
    * any other schema is ``required`` (never silently unrecorded).
    """

    if not isinstance(manifest, dict):
        raise ExportMemberIntegrityError(
            "the export manifest is not an object", reason=REASON_MANIFEST_UNREADABLE,
            path="", member=NATIVE_MANIFEST,
        )
    if legacy_manifest_format(manifest) is not None:
        return "legacy"
    # Native v2 extraction, any web export carrying export_kind, and and any unrecognised schema: digests are required,
    # so an unknown format can never read as unrecorded.
    return "required"


def manifest_requires_digests(manifest: object) -> bool:
    """True when the manifest's writer contract records every member's digest."""

    return manifest_integrity_contract(manifest) == "required"


_FORMAL_SCOPE: "ContextVar[bool]" = ContextVar("easyicu_formal_export_integrity", default=False)


def formal_integrity_active() -> bool:
    """True inside :func:`formal_integrity_scope` or with the formal env flag."""

    return _FORMAL_SCOPE.get() or os.environ.get(_FORMAL_ENV) == "1"


@contextmanager
def formal_integrity_scope(*, propagate_to_children: bool = True):
    """Formal/release/benchmark scope: an unrecorded manifest is refused.

    ``propagate_to_children`` also sets the environment flag so worker
    processes started inside the scope inherit it.
    """

    token = _FORMAL_SCOPE.set(True)
    previous = os.environ.get(_FORMAL_ENV)
    if propagate_to_children:
        os.environ[_FORMAL_ENV] = "1"
    try:
        yield
    finally:
        _FORMAL_SCOPE.reset(token)
        if propagate_to_children:
            if previous is None:
                os.environ.pop(_FORMAL_ENV, None)
            else:
                os.environ[_FORMAL_ENV] = previous


_HASH_CHUNK = 1024 * 1024


def _digest_chunks(handle) -> Iterable[bytes]:
    """Chunks of an open member (a seam for the hash-race regression test)."""

    while chunk := handle.read(_HASH_CHUNK):
        yield chunk


def _identity(stat: os.stat_result) -> tuple:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _hash_file(path: Path) -> str:
    """SHA-256 of one open member; refuses a file that changes while hashed.

    The same open descriptor is fstat-ed before and after the read, and the
    path must still name it afterwards: a same-size in-place write, a rename
    or a replacement during the hash refuses (``member_changed_during_hash``).
    """

    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for chunk in _digest_chunks(handle):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            current = os.stat(path)
    except OSError as exc:
        raise member_unreadable(path, exc) from exc
    if _identity(before) != _identity(after) or _identity(after) != _identity(current):
        raise ExportMemberIntegrityError(
            f"{path.name} changed while it was being verified",
            reason=REASON_CHANGED_DURING_HASH, path=path, member=path.name,
        )
    return digest.hexdigest()


#: A file changed within this window of its hash is not memoised ("racily
#: clean", as in git's index): kernels stamp ctime from a coarse clock, so a
#: rewrite in the same tick can keep the same stat key.
_RACY_WINDOW_NS = 2_000_000_000


def _member_sha256(path: Path) -> str:
    import time

    try:
        stat = path.stat()
    except OSError as exc:
        raise member_unreadable(path, exc) from exc
    key = (str(path.resolve()), *_identity(stat))
    cached = _DIGEST_MEMO.get(key)
    if cached is not None:
        return cached
    started_ns = time.time_ns()
    digest = _hash_file(path)  # refuses (and so never memoises) a changing file
    if _identity(path.stat()) != _identity(stat):
        raise ExportMemberIntegrityError(
            f"{path.name} changed while it was being verified",
            reason=REASON_CHANGED_DURING_HASH, path=path, member=path.name,
        )
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
            elif isinstance(value, str) and value and key != "modules":
                # ``modules`` may be a list of module *names*; only its
                # ``file``-keyed records are members (as in intake).
                entries.append({"file": value})
    return entries


def manifest_member_records(manifest: object) -> dict[str, tuple[str, int | None] | None]:
    """``{member: (sha256, size) | None}``: one normalised record per member.

    Legacy manifests list members in redundant inventories (``files``,
    ``modules``, ``exported_files``); compatible digests merge, a missing
    size is filled from another record, and two different digests or sizes
    for one member are refused (``member_digest_conflict``). Shared by every
    reader, intake included, so list order never decides.
    """

    merged: dict[str, tuple[str, int | None] | None] = {}
    for entry in manifest_member_entries(manifest):
        name = Path(str(entry.get("file"))).as_posix()
        record = recorded_member_digest(entry)
        current = merged.get(name)
        if name not in merged or current is None:
            merged[name] = record
            continue
        if record is None:
            continue
        if record[0] != current[0] or (
            record[1] is not None and current[1] is not None and record[1] != current[1]
        ):
            raise ExportMemberIntegrityError(
                f"the export manifest records two different digests for {name}",
                reason=REASON_DIGEST_CONFLICT, path=name, member=name,
            )
        merged[name] = (current[0], current[1] if current[1] is not None else record[1])
    return merged


def require_member_digest(manifest: dict, member: str, *, sha256: str, size: int,
                          path: Union[str, Path], formal: bool | None = None) -> str:
    """Compare already-computed bytes facts with the manifest record of ``member``.

    ``formal`` (default: :func:`formal_integrity_active`) refuses unrecorded.
    """

    relative = Path(member).as_posix()
    records = manifest_member_records(manifest)
    if manifest_integrity_contract(manifest) == "legacy" and not any(records.values()):
        # An explicitly legacy manifest whose writer never recorded digests:
        # nothing to compare. Formal scope refuses it; the UI path shows it.
        if formal if formal is not None else formal_integrity_active():
            raise ExportMemberIntegrityError(
                f"{INTEGRITY_UNRECORDED_LABEL}: {relative} (its export manifest "
                "records no digests; a formal consumer requires them)",
                reason=REASON_INTEGRITY_UNRECORDED, path=path, member=relative,
            )
        return INTEGRITY_UNRECORDED
    if relative not in records:
        raise ExportMemberIntegrityError(
            f"{relative} is not listed in the export manifest",
            reason=REASON_MEMBER_UNRECORDED, path=path, member=relative,
        )
    recorded = records[relative]
    if recorded is None:
        raise ExportMemberIntegrityError(
            f"{INTEGRITY_UNRECORDED_LABEL}: {relative} (the export manifest "
            "records no digest for it)",
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


def _member_path(root: Path, member: Union[str, Path]) -> Path:
    """An absolute member path: absolute, caller-relative inside root, or root-relative.

    A relative path that already starts at the root (``export/x.parquet`` for
    root ``export``) is the caller's file path, not a member to join again.
    """

    candidate = Path(member)
    if candidate.is_absolute():
        return candidate
    root_abs = root.absolute()
    caller = candidate.absolute()
    try:
        caller.relative_to(root_abs)
        return caller
    except ValueError:
        return root_abs / candidate


def verify_member_integrity(package_root: Union[str, Path], member: Union[str, Path],
                            *, manifest: dict | None = None,
                            formal: bool | None = None) -> str:
    """Refuse a consumed member whose size or SHA-256 differs from the manifest.

    ``member`` is an absolute path, a caller-relative path inside the root, or
    a path relative to ``package_root``. Returns :data:`INTEGRITY_VERIFIED`, or
    :data:`INTEGRITY_UNRECORDED` for an explicitly legacy manifest without
    digests (refused in formal scope). Memoised per stat identity.
    """

    root = Path(package_root).absolute()
    payload = manifest if manifest is not None else load_published_manifest(root)
    path = _member_path(root, member)
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = Path(member).as_posix()
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise member_unreadable(path, exc) from exc
    records = manifest_member_records(payload)
    recorded = records.get(relative)
    # Cheap size refusal first; hash only when the record has a digest.
    sha = _member_sha256(path) if recorded is not None and (
        recorded[1] is None or recorded[1] == size) else ""
    return require_member_digest(payload, relative, sha256=sha, size=size, path=path,
                                 formal=formal)


def require_listed_members_present(package_root: Union[str, Path], manifest: dict) -> None:
    """Every member the manifest lists must exist (never a silently smaller export)."""

    root = Path(package_root).absolute()
    for name in manifest_member_records(manifest):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or "\x00" in name:
            # Never probe outside the package for a listed member.
            raise ExportMemberIntegrityError(
                f"{name!r} is not a path inside the export",
                reason=REASON_MEMBER_PATH_UNSAFE, path=root, member=name,
            )
        target = root / relative  # manifest names are root-relative, never cwd-relative
        if not target.is_file():
            raise ExportMemberIntegrityError(
                f"{name} is listed in the export manifest but missing",
                reason=REASON_MEMBER_MISSING, path=target, member=name,
            )


def verify_package_members(package_root: Union[str, Path], members: Iterable[Union[str, Path]],
                           *, manifest: dict | None = None,
                           formal: bool | None = None) -> dict[str, str]:
    """Verify several members against one manifest read: ``{member: status}``."""

    payload = manifest if manifest is not None else load_published_manifest(package_root)
    require_listed_members_present(package_root, payload)
    return {
        Path(member).as_posix(): verify_member_integrity(
            package_root, member, manifest=payload, formal=formal)
        for member in members
    }


def require_verified_export_package(export_dir: Union[str, Path], *,
                                    formal: bool | None = True) -> dict:
    """Package-level gate for whole-export readers (tools, discovery, audits).

    Publication gate, then every manifest-listed member present and verified
    against its recorded digest. ``formal`` defaults to True: an explicitly
    legacy manifest without digests is refused (``member_integrity_unrecorded``);
    ``formal=None`` follows :func:`formal_integrity_active` and lets the caller
    show the returned :func:`integrity_status` instead. Members a reader opens
    that the manifest does not list are the reader's own per-file check.
    """

    root = Path(export_dir).absolute()
    manifest = load_published_manifest(root)
    names = list(manifest_member_records(manifest))
    if not names:
        # Nothing listed means nothing verifiable: never a vacuous "verified".
        if formal if formal is not None else formal_integrity_active():
            raise ExportMemberIntegrityError(
                f"{INTEGRITY_UNRECORDED_LABEL}: {root.name} (its export manifest "
                "lists no members; a formal consumer requires them)",
                reason=REASON_INTEGRITY_UNRECORDED, path=root, member="",
            )
        return {"status": INTEGRITY_UNRECORDED, "label": INTEGRITY_UNRECORDED_LABEL,
                "unrecorded_members": 0, "legacy_format": legacy_manifest_format(manifest)}
    statuses = verify_package_members(root, [root / name for name in names],
                                      manifest=manifest, formal=formal)
    return integrity_status(statuses.values(), manifest=manifest)


#: Reasons whose user-facing text is exactly :data:`INTEGRITY_UNRECORDED_LABEL`:
#: the bytes cannot be verified because no digest was ever recorded.
UNRECORDED_REASONS = frozenset({REASON_DIGEST_UNRECORDED, REASON_INTEGRITY_UNRECORDED})


def refusal_message(exc: BaseException) -> str:
    """The human message for a member refusal (no host paths).

    An unrecorded digest reads exactly "integrity unrecorded · re-export to
    verify"; a damaged or unreadable member asks for a new extraction.
    """

    if getattr(exc, "reason", None) in UNRECORDED_REASONS:
        return INTEGRITY_UNRECORDED_LABEL
    if isinstance(exc, ExportMemberUnreadableError) or getattr(exc, "code", None) == UNREADABLE_CODE:
        return "export member cannot be read · re-run the extraction"
    return "export member does not match its published manifest · re-run the extraction"


def export_relative_path(package_root: Union[str, Path], member: str) -> str:
    """``<export folder>/<member>``: the member relative to the user's export root."""

    name = Path(package_root).name
    member = Path(str(member or "")).as_posix()
    if not member or Path(member).is_absolute() or ".." in Path(member).parts:
        return name
    return f"{name}/{member}"


def integrity_status(statuses: Iterable[str], *, manifest: object = None) -> dict:
    """The visible summary of consumed members' integrity statuses.

    With ``manifest``, an unrecorded summary also names its legacy format
    (``legacy_format``, e.g. :data:`LEGACY_WEB_PRE_KIND`).
    """

    values = list(statuses)
    unrecorded = sum(value == INTEGRITY_UNRECORDED for value in values)
    if not values:
        return {"status": None, "label": None, "unrecorded_members": 0}
    status = INTEGRITY_UNRECORDED if unrecorded else INTEGRITY_VERIFIED
    summary = {
        "status": status,
        "label": INTEGRITY_UNRECORDED_LABEL if unrecorded else None,
        "unrecorded_members": unrecorded,
    }
    if unrecorded and manifest is not None:
        summary["legacy_format"] = legacy_manifest_format(manifest)
    return summary
