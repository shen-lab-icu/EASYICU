"""Durable bounded slots for EasyICU's supervised metadata-only requests.

An unknown remote request keeps its slot indefinitely. This scope covers only
requests admitted by this owner, not external clients using the same account.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sqlite3
import stat
import time
from typing import Any, Mapping

from easyicu.webserver import state_paths


class MetadataProviderCapacityExhausted(RuntimeError):
    """The local admission limit was reached before starting a new request."""


def _limit() -> int:
    """Default one; a higher metadata-owner limit needs explicit configuration.

    Other EasyICU paths and external consumers are not counted. An operator
    must provision their shared Provider capacity before setting this bound.
    """
    raw = os.environ.get("EASYICU_METADATA_PROVIDER_MAX_INFLIGHT", "1")
    if raw not in {"1", "2"}:
        raise ValueError("metadata_provider_concurrency_configuration_invalid")
    return int(raw)


def account_identity(public: Mapping[str, Any]) -> str:
    endpoint = str(public.get("endpoint_fingerprint") or "")
    credential = str(public.get("credential_fingerprint") or "")
    if not endpoint or not credential:
        raise ValueError("metadata_provider_account_identity_missing")
    return hashlib.sha256((endpoint + "\0" + credential).encode()).hexdigest()


def _open() -> sqlite3.Connection:
    root = state_paths.state_root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / "metadata-provider-slots.sqlite3"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("metadata_provider_slot_path_invalid")
        if os.name != "nt":
            os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    connection = sqlite3.connect(path, timeout=5, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("CREATE TABLE IF NOT EXISTS slots (attempt_id TEXT PRIMARY KEY, account TEXT NOT NULL, task_id TEXT NOT NULL, ledger_path TEXT NOT NULL, guard_path TEXT NOT NULL, guard_nonce TEXT, worker_sha256 TEXT NOT NULL, created REAL NOT NULL, request_sha256 TEXT)")
    if "request_sha256" not in {row[1] for row in connection.execute("PRAGMA table_info(slots)")}:
        try:
            connection.execute("ALTER TABLE slots ADD COLUMN request_sha256 TEXT")
        except sqlite3.OperationalError as exc:
            # Another process may have completed the same additive migration.
            if "duplicate column name" not in str(exc).lower():
                raise
    return connection


def reserve(*, attempt_id: str, account: str, task_id: str, ledger_path: Path,
            guard_path: Path, worker_sha256: str, request_sha256: str | None = None) -> None:
    connection = _open()
    try:
        connection.execute("BEGIN IMMEDIATE")
        count = connection.execute("SELECT COUNT(*) FROM slots WHERE account=?", (account,)).fetchone()[0]
        if count >= _limit():
            raise MetadataProviderCapacityExhausted("metadata_provider_concurrency_exhausted")
        connection.execute("INSERT INTO slots (attempt_id, account, task_id, ledger_path, guard_path, guard_nonce, worker_sha256, created, request_sha256) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)",
                           (attempt_id, account, task_id, str(ledger_path), str(guard_path), worker_sha256, time.time(), request_sha256))
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def bind_guard(attempt_id: str, nonce: str) -> None:
    connection = _open()
    try:
        cursor = connection.execute("UPDATE slots SET guard_nonce=? WHERE attempt_id=? AND guard_nonce IS NULL", (nonce, attempt_id))
        if cursor.rowcount != 1:
            raise RuntimeError("metadata_provider_slot_binding_invalid")
    finally:
        connection.close()


def get(attempt_id: str) -> dict[str, Any] | None:
    connection = _open()
    try:
        row = connection.execute("SELECT * FROM slots WHERE attempt_id=?", (attempt_id,)).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def has_capacity(account: str) -> bool:
    """A preflight only; reservation under BEGIN IMMEDIATE remains authoritative."""
    connection = _open()
    try:
        count = connection.execute("SELECT COUNT(*) FROM slots WHERE account=?", (account,)).fetchone()[0]
        return count < _limit()
    finally:
        connection.close()


def release(attempt_id: str) -> None:
    connection = _open()
    try:
        connection.execute("DELETE FROM slots WHERE attempt_id=?", (attempt_id,))
    finally:
        connection.close()
