"""Durable bounded slots for EasyICU's supervised metadata-only requests.

An unknown remote request keeps its slot indefinitely. This scope covers only
requests admitted by this owner, not external clients using the same account.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import stat
import time
from typing import Any, Callable, Mapping

from easyicu.webserver import state_paths


_LOG = logging.getLogger(__name__)

# A slot is reclaimed only after its request cannot still be running: the
# request timeout (stored with the slot; this default when a row predates it)
# plus a margin has passed, and the worker is verifiably gone.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 1800.0
_RECLAIM_MARGIN_SECONDS = 60.0
_RECLAIM_LOG = "metadata-provider-reclaims.jsonl"
_PROTECTED_TASKS: Callable[[], frozenset[str]] | None = None


class MetadataProviderCapacityExhausted(RuntimeError):
    """The local admission limit was reached before starting a new request."""


def set_protected_tasks_provider(provider: Callable[[], frozenset[str]] | None) -> None:
    """Tasks research auto-resume may still continue keep their slot.

    Resuming an interrupted task re-uses the slot its in-flight request holds,
    so a slot of such a task is never reclaimed here.
    """
    global _PROTECTED_TASKS
    _PROTECTED_TASKS = provider


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
    from easyicu.webserver.metadata_provider_shared_scope import (
        SHARED_ROOT_ENV, SharedScopeError, open_enrolled_scope, read_enrollment,
    )

    shared = str(os.environ.get(SHARED_ROOT_ENV) or "").strip()
    enrollment = read_enrollment(root)
    if enrollment is not None and shared != enrollment:
        raise SharedScopeError("metadata_shared_scope_configuration_required")
    if shared and enrollment is None:
        raise SharedScopeError("metadata_shared_scope_unavailable")
    if shared:
        # No fallback: a missing enrollment or changed legacy snapshot must
        # not silently open a new independent capacity ledger.
        try:
            return open_enrolled_scope(Path(shared), root)
        except SharedScopeError:
            raise
        except (OSError, sqlite3.DatabaseError, ValueError) as exc:
            raise SharedScopeError("metadata_shared_scope_unavailable") from exc
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
    for column, kind in (("request_sha256", "TEXT"), ("request_timeout", "REAL"),
                         ("owner_pid", "INTEGER"), ("owner_created", "REAL")):
        if column not in {row[1] for row in connection.execute("PRAGMA table_info(slots)")}:
            try:
                connection.execute(f"ALTER TABLE slots ADD COLUMN {column} {kind}")
            except sqlite3.OperationalError as exc:
                # Another process may have completed the same additive migration.
                if "duplicate column name" not in str(exc).lower():
                    raise
    return connection


def _columns(connection: sqlite3.Connection) -> set[str]:
    return {row[1] for row in connection.execute("PRAGMA table_info(slots)")}


def _owner_identity() -> tuple[int, float] | tuple[None, None]:
    try:
        import psutil

        process = psutil.Process()
        return process.pid, process.create_time()
    except Exception:  # psutil is optional on Core installs; no owner evidence then.
        return None, None


def _owner_alive(pid: Any, created: Any) -> bool:
    """Whether the process that reserved a slot still exists (same start time)."""
    if type(pid) is not int or pid <= 0 or not isinstance(created, (int, float)):
        return False
    try:
        import psutil

        process = psutil.Process(pid)
        return process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def reserve(*, attempt_id: str, account: str, task_id: str, ledger_path: Path,
            guard_path: Path, worker_sha256: str, request_sha256: str | None = None,
            timeout_seconds: float | None = None) -> None:
    for reclaimed_once in (False, True):
        connection = _open()
        try:
            connection.execute("BEGIN IMMEDIATE")
            count = connection.execute("SELECT COUNT(*) FROM slots WHERE account=?", (account,)).fetchone()[0]
            if count >= _limit():
                connection.execute("ROLLBACK")
                connection.close()
                connection = None
                # A slot whose worker is verifiably gone must not block new
                # requests forever; try once, then refuse as before.
                if not reclaimed_once and reclaim_orphaned(account=account):
                    continue
                raise MetadataProviderCapacityExhausted("metadata_provider_concurrency_exhausted")
            columns = _columns(connection)
            owner_pid, owner_created = _owner_identity()
            fields = {"attempt_id": attempt_id, "account": account, "task_id": task_id,
                      "ledger_path": str(ledger_path), "guard_path": str(guard_path),
                      "guard_nonce": None, "worker_sha256": worker_sha256, "created": time.time(),
                      "request_sha256": request_sha256}
            for column, value in (("request_timeout", timeout_seconds),
                                  ("owner_pid", owner_pid), ("owner_created", owner_created)):
                if column in columns:
                    fields[column] = value
            connection.execute(
                f"INSERT INTO slots ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})",
                tuple(fields.values()))
            connection.execute("COMMIT")
            return
        except BaseException:
            if connection is not None:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
            raise
        finally:
            if connection is not None:
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
        from easyicu.webserver.metadata_provider_shared_scope import SHARED_ROOT_ENV

        if os.environ.get(SHARED_ROOT_ENV) and connection.execute(
            "SELECT 1 FROM imported_slots WHERE attempt_id=?", (attempt_id,)
        ).fetchone() is not None:
            raise RuntimeError("metadata_provider_imported_slot_requires_terminal_proof")
        connection.execute("DELETE FROM slots WHERE attempt_id=?", (attempt_id,))
    finally:
        connection.close()


def _slot_is_orphaned(row: Mapping[str, Any], now: float, protected: frozenset[str]) -> tuple[str, float, float] | None:
    """The reclaim reason with the slot's age and timeout, or ``None`` to keep it."""
    if row["task_id"] in protected:
        return None
    timeout = float(row.get("request_timeout") or DEFAULT_REQUEST_TIMEOUT_SECONDS)
    age = now - float(row["created"])
    if age <= timeout + _RECLAIM_MARGIN_SECONDS:
        return None
    # Another running instance of the same state folder keeps its own slot.
    if _owner_alive(row.get("owner_pid"), row.get("owner_created")):
        return None
    if row.get("guard_nonce") is None:
        # The guard directory and its nonce are written before the supervised
        # worker is spawned (guarded_call), so no request left this slot.
        return "slot_reclaimed_never_sent", age, timeout
    from easyicu.runtime.host_execution_lease import (
        HostExecutionLeaseError, assert_host_execution_quiescent, read_host_execution_receipt,
    )

    guard = Path(row["guard_path"])
    try:
        receipt = read_host_execution_receipt(guard)
        if (receipt is None or receipt.get("nonce") != row["guard_nonce"]
                or receipt.get("script_sha256") != row["worker_sha256"]):
            return None
        assert_host_execution_quiescent(guard, allow_missing=False)
    except (OSError, ValueError, HostExecutionLeaseError):
        return None
    return "slot_reclaimed_worker_gone", age, timeout


def reclaim_orphaned(*, account: str | None = None) -> int:
    """Release slots whose request cannot still be running; return how many.

    A slot is kept unless every proof holds: its task is not one research
    auto-resume may continue, its age exceeds the request timeout plus a
    margin, the process that reserved it no longer exists, and either no
    worker was ever prepared or the worker's lifecycle receipt (nonce and
    script digest bound) shows it quiesced. Imported shared-scope slots need
    terminal proof from their own owner and are never reclaimed here.
    """
    from easyicu.webserver.metadata_provider_shared_scope import SHARED_ROOT_ENV

    if os.environ.get(SHARED_ROOT_ENV):
        return 0
    protected = frozenset(_PROTECTED_TASKS()) if _PROTECTED_TASKS is not None else frozenset()
    reclaimed: list[dict[str, Any]] = []
    connection = _open()
    try:
        query = "SELECT * FROM slots" + (" WHERE account=?" if account else "")
        rows = [dict(row) for row in connection.execute(query, (account,) if account else ())]
        now = time.time()
        for row in rows:
            verdict = _slot_is_orphaned(row, now, protected)
            if verdict is None:
                continue
            reason, age, timeout = verdict
            if connection.execute("DELETE FROM slots WHERE attempt_id=? AND created=?",
                                  (row["attempt_id"], row["created"])).rowcount == 1:
                reclaimed.append({"reason": reason, "attempt_id": row["attempt_id"],
                                  "task_id": row["task_id"], "age_seconds": round(age, 1),
                                  "timeout_seconds": timeout, "at": now})
    finally:
        connection.close()
    for record in reclaimed:
        _LOG.warning("metadata provider slot reclaimed: %s", record)
        try:
            with open(state_paths.state_root() / _RECLAIM_LOG, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            pass
    return len(reclaimed)
