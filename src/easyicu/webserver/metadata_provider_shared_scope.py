"""An explicitly enrolled, deployment-scoped metadata Provider coordinator.

An operator must retire old code and enumerate every participating state home
before activation. The coordinator cannot police unregistered old binaries or
external Provider clients. Migrated unknown requests remain reserved; neither
their original files nor their run budgets are modified.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys
import tempfile
from typing import Any

from easyicu.runtime.platform_compat import lock_exclusive, unlock_file

SHARED_ROOT_ENV = "EASYICU_METADATA_PROVIDER_SHARED_ROOT"
_DB = "metadata-provider-shared.sqlite3"
_LOCAL = "metadata-provider-slots.sqlite3"
_ENROLLMENT = "metadata-provider-shared-enrollment.json"
_COLUMNS = ("attempt_id", "account", "task_id", "ledger_path", "guard_path",
            "guard_nonce", "worker_sha256", "created", "request_sha256")
_VERSION = "easyicu.metadata-shared-scope/1"
_MAX_HOMES = 128
_MAX_SLOTS_PER_HOME = 10_000


class SharedScopeError(RuntimeError):
    """The declared shared deployment cannot be verified."""


def read_enrollment(state: Path) -> str | None:
    """A bound home never silently falls back to a local capacity ledger."""
    path = state / _ENROLLMENT
    if path.is_symlink():
        raise SharedScopeError("metadata_shared_enrollment_invalid")
    if not path.exists():
        return None
    _private_dir(state)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise SharedScopeError("metadata_shared_enrollment_invalid") from exc
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_size > 4096 or info.st_mode & 0o077
                or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
            raise SharedScopeError("metadata_shared_enrollment_invalid")
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            payload = json.loads(stream.read(4097))
        if (not isinstance(payload, dict) or set(payload) != {"version", "state", "root"}
                or payload["version"] != _VERSION or payload["state"] != str(state)
                or not isinstance(payload["root"], str) or not Path(payload["root"]).is_absolute()):
            raise SharedScopeError("metadata_shared_enrollment_invalid")
        return payload["root"]
    except (OSError, TypeError, ValueError) as exc:
        raise SharedScopeError("metadata_shared_enrollment_invalid") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _write_enrollment(state: Path, root: Path) -> None:
    existing = read_enrollment(state)
    if existing is not None:
        if existing != str(root):
            raise SharedScopeError("metadata_shared_enrollment_conflict")
        return
    data = json.dumps({"version": _VERSION, "state": str(state), "root": str(root)},
                      separators=(",", ":")).encode()
    fd = os.open(state / _ENROLLMENT, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(state, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _private_dir(path: Path) -> Path:
    # POSIX ownership and permission checks are part of this authority; do not
    # treat Windows mode bits as equivalent ACL evidence.
    if os.name != "posix":
        raise SharedScopeError("metadata_shared_scope_platform_unsupported")
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise SharedScopeError("metadata_shared_scope_directory_invalid")
    resolved = path.resolve(strict=True)
    if resolved != path or any(parent.is_symlink() for parent in path.parents):
        raise SharedScopeError("metadata_shared_scope_directory_invalid")
    info = path.stat()
    if info.st_mode & 0o077 or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
        raise SharedScopeError("metadata_shared_scope_directory_not_private")
    return path


def _local_rows(state: Path, *, lock: bool = False) -> tuple[str, list[tuple[Any, ...]], sqlite3.Connection | None]:
    """Read, optionally lock, the exact legacy database without creating it."""
    path = state / _LOCAL
    if path.is_symlink():
        raise SharedScopeError("metadata_shared_legacy_database_invalid")
    if not path.exists():
        return "absent", [], None
    info = path.stat()
    if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077
            or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
        raise SharedScopeError("metadata_shared_legacy_database_invalid")
    connection = sqlite3.connect(path.as_uri() + ("?mode=rw" if lock else "?mode=ro"),
                                 uri=True, timeout=1, isolation_level=None)
    try:
        if lock:
            connection.execute("BEGIN IMMEDIATE")
        names = {row[1] for row in connection.execute("PRAGMA table_info(slots)")}
        if not set(_COLUMNS[:-1]).issubset(names):
            raise SharedScopeError("metadata_shared_legacy_schema_invalid")
        selected = ", ".join(_COLUMNS if "request_sha256" in names else (*_COLUMNS[:-1], "NULL"))
        rows = connection.execute(f"SELECT {selected} FROM slots ORDER BY attempt_id LIMIT ?",
                                  (_MAX_SLOTS_PER_HOME + 1,)).fetchall()
        if len(rows) > _MAX_SLOTS_PER_HOME:
            raise SharedScopeError("metadata_shared_legacy_slot_limit_exceeded")
        encoded = json.dumps(rows, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest(), rows, connection
    except BaseException:
        connection.close()
        raise


def open_enrolled_scope(root: Path, state: Path) -> sqlite3.Connection:
    """Open the shared DB only if this home and its legacy snapshot match."""
    root = _private_dir(root)
    state = _private_dir(state)
    path = root / _DB
    if path.is_symlink() or not path.is_file():
        raise SharedScopeError("metadata_shared_scope_unavailable")
    info = path.stat()
    if info.st_mode & 0o077 or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
        raise SharedScopeError("metadata_shared_scope_unavailable")
    connection = sqlite3.connect(path, timeout=5, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        version = connection.execute("SELECT version FROM scope").fetchall()
        members = connection.execute("SELECT state_path, legacy_sha256 FROM members ORDER BY state_path LIMIT ?",
                                     (_MAX_HOMES + 1,)).fetchall()
        if (len(version) != 1 or version[0][0] != _VERSION or not members or len(members) > _MAX_HOMES
                or str(state) not in {row[0] for row in members}):
            raise SharedScopeError("metadata_shared_scope_membership_or_legacy_changed")
        # A stale legacy process in *any* enrolled home invalidates the whole
        # scope; checking just the caller would let another home bypass it.
        for member in members:
            member_state = _private_dir(Path(member[0]))
            digest, _, local = _local_rows(member_state)
            if local is not None:
                local.close()
            if member[1] != digest:
                raise SharedScopeError("metadata_shared_scope_membership_or_legacy_changed")
        connection.execute("PRAGMA synchronous=FULL")
        return connection
    except BaseException:
        connection.close()
        raise


def preview_scope(root: Path, homes: list[Path]) -> tuple[tuple[str, str, int], ...]:
    """Read the exact enrollment choices for the operator, with no mutation."""
    _private_dir(root)
    states = [_private_dir(home / ".easyicu") for home in homes]
    if not states or len(states) > _MAX_HOMES or len({str(state) for state in states}) != len(states):
        raise SharedScopeError("metadata_shared_scope_members_invalid")
    preview = []
    for state in sorted(states):
        digest, rows, connection = _local_rows(state)
        if connection is not None:
            connection.close()
        preview.append((str(state), digest, len(rows)))
    return tuple(preview)


def initialize_scope(root: Path, homes: list[Path], *,
                     expected_preview: tuple[tuple[str, str, int], ...] | None = None) -> tuple[int, int]:
    """Atomically publish an operator-attested, explicitly enumerated scope.

    The CLI obtains the operator's attestation. Enrollment markers are written
    to each home, while the existing slot and budget ledgers are preserved.
    Callers must ensure older binaries cannot subsequently use this deployment.
    """
    root = _private_dir(root)
    states = [_private_dir(home / ".easyicu") for home in homes]
    if not states or len(states) > _MAX_HOMES or len({str(state) for state in states}) != len(states):
        raise SharedScopeError("metadata_shared_scope_members_invalid")
    destination = root / _DB
    if destination.exists() or destination.is_symlink():
        raise SharedScopeError("metadata_shared_scope_already_exists")
    with ExitStack() as stack:
        copied: dict[str, tuple[Any, ...]] = {}
        snapshots: list[tuple[str, str]] = []
        for state in sorted(states):
            lock_path = state / "webserver-single-process.lock"
            if lock_path.is_symlink():
                raise SharedScopeError("metadata_shared_instance_lock_invalid")
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0), 0o600)
            stack.callback(os.close, fd)
            lock_info = os.fstat(fd)
            if (not stat.S_ISREG(lock_info.st_mode) or lock_info.st_mode & 0o077
                    or (hasattr(os, "getuid") and lock_info.st_uid != os.getuid())):
                raise SharedScopeError("metadata_shared_instance_lock_invalid")
            try:
                lock_exclusive(fd, blocking=False)
            except (BlockingIOError, OSError) as exc:
                raise SharedScopeError("metadata_shared_instance_still_running") from exc
            stack.callback(unlock_file, fd)
            digest, rows, local = _local_rows(state, lock=True)
            if local is not None:
                stack.callback(local.close)
            snapshots.append((str(state), digest))
            if expected_preview is not None and (str(state), digest, len(rows)) not in expected_preview:
                raise SharedScopeError("metadata_shared_scope_changed_since_review")
            for row in rows:
                key = str(row[0])
                if key in copied and copied[key] != row:
                    raise SharedScopeError("metadata_shared_attempt_collision")
                copied[key] = row
        if expected_preview is not None and len(expected_preview) != len(snapshots):
            raise SharedScopeError("metadata_shared_scope_changed_since_review")
        # Write each binding before publishing the coordinator. If interrupted
        # mid-enrollment, those homes fail closed until the same scope is
        # completed; a missing environment setting never reopens a local slot.
        for state in sorted(states):
            _write_enrollment(state, root)
        descriptor, temp = tempfile.mkstemp(prefix=".metadata-shared-", dir=root)
        os.close(descriptor)
        temporary = Path(temp)
        try:
            connection = sqlite3.connect(temporary, isolation_level=None)
            try:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("CREATE TABLE scope (version TEXT NOT NULL)")
                connection.execute("INSERT INTO scope VALUES (?)", (_VERSION,))
                connection.execute("CREATE TABLE members (state_path TEXT PRIMARY KEY, legacy_sha256 TEXT NOT NULL)")
                connection.executemany("INSERT INTO members VALUES (?, ?)", snapshots)
                connection.execute("CREATE TABLE slots (attempt_id TEXT PRIMARY KEY, account TEXT NOT NULL, task_id TEXT NOT NULL, ledger_path TEXT NOT NULL, guard_path TEXT NOT NULL, guard_nonce TEXT, worker_sha256 TEXT NOT NULL, created REAL NOT NULL, request_sha256 TEXT)")
                connection.executemany("INSERT INTO slots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", copied.values())
                connection.execute("CREATE TABLE imported_slots (attempt_id TEXT PRIMARY KEY)")
                connection.executemany("INSERT INTO imported_slots VALUES (?)", ((key,) for key in copied))
                connection.execute("COMMIT")
            finally:
                connection.close()
            os.chmod(temporary, 0o600)
            os.link(temporary, destination)  # never replace an existing scope
            directory = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
    return len(states), len(copied)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shared_root", type=Path, help="Existing private coordinator directory")
    parser.add_argument("--home", type=Path, action="append", required=True,
                        help="Each enrolled EASYICU_HOME; repeat for all instances")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        raise SharedScopeError("metadata_shared_scope_human_attestation_requires_terminal")
    preview = preview_scope(args.shared_root, args.home)
    print(f"Coordinator: {args.shared_root}")
    for state, _digest, count in preview:
        print(f"  {state}: {count} retained legacy request(s)")
    print("This scope covers only these homes and identical endpoint/credential fingerprints.")
    print("Stop all listed instances and prevent every old or unlisted instance from using this deployment.")
    print("Unknown requests and budgets remain reserved. Type ENROLL to attest the complete list: ", end="", flush=True)
    if input().strip() != "ENROLL":
        raise SharedScopeError("metadata_shared_scope_attestation_not_given")
    homes, slots = initialize_scope(args.shared_root, args.home, expected_preview=preview)
    print(f"Enrolled {homes} homes; retained {slots} legacy slots. Configure {SHARED_ROOT_ENV} for every instance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
