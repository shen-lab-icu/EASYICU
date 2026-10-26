"""Private, bounded lifecycle receipts for the local Web job manager.

This is a journal, not a workflow executor. It stores only identifiers, states,
timestamps and event cursors. Runners, arguments, credentials, event payloads
and results stay with their existing owners. A lost running job is interrupted;
only an existing scientific checkpoint owner may authorize its continuation.
The Web deployment lease must be held before opening this store.

For running jobs the cursor is a reserved upper bound, so streaming can publish
an event batch without a disk write per token. Terminal receipts use the exact
cursor. Recovery emits a gap marker before its terminal event when necessary.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_STATES = {"running", "done", "failed", "cancelled"}
_COLUMNS = (
    "id", "kind", "status", "created", "finished", "cancel_requested",
    "next_event_seq", "interrupted",
)


class JobJournalError(RuntimeError):
    """The host cannot safely read or record local job lifecycle state."""


def _validated(record: Mapping[str, Any]) -> dict[str, Any]:
    # Explicit projection: free-form error messages or result dictionaries must
    # never accidentally become persistence input when the Job API grows.
    try:
        result = {key: record[key] for key in _COLUMNS}
    except (KeyError, TypeError) as exc:
        raise JobJournalError("job_journal_invalid_record") from exc
    if any(
        not isinstance(result[key], str) or not _IDENTIFIER.fullmatch(result[key])
        for key in ("id", "kind")
    ) or not isinstance(result["status"], str) or result["status"] not in _STATES:
        raise JobJournalError("job_journal_invalid_record")
    for key in ("created", "finished"):
        value = result[key]
        if key == "finished" and value is None:
            continue
        if type(value) not in {float, int} or not math.isfinite(value) or not 0 <= value <= 253402300799:
            raise JobJournalError("job_journal_invalid_record")
    if (
        type(result["next_event_seq"]) is not int
        or not 0 <= result["next_event_seq"] < 2**63 - 1
        or any(type(result[key]) not in {bool, int} or result[key] not in (0, 1)
               for key in ("cancel_requested", "interrupted"))
        or (result["status"] == "running") != (result["finished"] is None)
    ):
        raise JobJournalError("job_journal_invalid_record")
    return result


class JobJournal:
    """Atomic SQLite lifecycle writes under the existing single-process lease."""

    def __init__(self, path: Path, *, max_completed: int = 100) -> None:
        if max_completed < 0:
            raise ValueError("max_completed must be non-negative")
        self.path = Path(path)
        self.max_completed = max_completed
        self._lock = threading.RLock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode) or self.path.is_symlink():
                    raise JobJournalError("job_journal_unsafe_path")
                if os.name != "nt":
                    os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
            with self._connect() as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version not in {0, 1}:
                    raise JobJournalError("job_journal_unsupported_version")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS jobs ("
                    "id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, "
                    "created REAL NOT NULL, finished REAL, cancel_requested INTEGER NOT NULL, "
                    "next_event_seq INTEGER NOT NULL, interrupted INTEGER NOT NULL)"
                )
                connection.execute("PRAGMA user_version = 1")
        except (OSError, sqlite3.Error) as exc:
            raise JobJournalError("job_journal_unavailable") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA synchronous = FULL")
            with connection:
                yield connection
        finally:
            connection.close()

    def create(self, record: Mapping[str, Any]) -> None:
        """Persist acceptance before the runner starts; duplicate IDs fail closed."""
        values = _validated(record)
        with self._lock:
            try:
                with self._connect() as connection:
                    connection.execute(
                        "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        tuple(values[key] for key in _COLUMNS),
                    )
            except sqlite3.Error as exc:
                raise JobJournalError("job_journal_write_failed") from exc

    def save(self, record: Mapping[str, Any]) -> None:
        """Update an accepted job, never reopen a terminal receipt."""
        values = _validated(record)
        with self._lock:
            try:
                with self._connect() as connection:
                    cursor = connection.execute(
                        "UPDATE jobs SET kind=?, status=?, created=?, finished=?, "
                        "cancel_requested=?, next_event_seq=?, interrupted=? "
                        "WHERE id=? AND status='running'",
                        tuple(values[key] for key in _COLUMNS[1:]) + (values["id"],),
                    )
                    if not cursor.rowcount:
                        prior = connection.execute("SELECT * FROM jobs WHERE id=?", (values["id"],)).fetchone()
                        if prior is None or dict(prior) != values:
                            raise JobJournalError("job_journal_terminal_receipt_changed")
                    connection.execute(
                        "DELETE FROM jobs WHERE id IN (SELECT id FROM jobs "
                        "WHERE status != 'running' ORDER BY finished DESC, id DESC LIMIT -1 OFFSET ?)",
                        (self.max_completed,),
                    )
            except sqlite3.Error as exc:
                raise JobJournalError("job_journal_write_failed") from exc

    def reopen_interrupted(self, record: Mapping[str, Any]) -> None:
        """Reattach an execution owner to an interrupted, never-cancelled ID.

        Reopening the transport does not grant scientific execution; the durable
        runner must still validate its original authority before any action.
        """
        values = _validated(record)
        if values["status"] != "running" or values["cancel_requested"] or values["interrupted"]:
            raise JobJournalError("job_journal_invalid_resume")
        with self._lock:
            try:
                with self._connect() as connection:
                    prior = connection.execute("SELECT * FROM jobs WHERE id=?", (values["id"],)).fetchone()
                    if (prior is None or prior["kind"] != values["kind"]
                        or prior["status"] != "failed" or not prior["interrupted"]
                        or prior["cancel_requested"] or values["created"] != prior["created"]
                        or values["next_event_seq"] < prior["next_event_seq"]):
                        raise JobJournalError("job_journal_resume_not_interrupted")
                    cursor = connection.execute(
                        "UPDATE jobs SET status='running', finished=NULL, interrupted=0, next_event_seq=? "
                        "WHERE id=? AND status='failed' AND interrupted=1 AND cancel_requested=0",
                        (values["next_event_seq"], values["id"]),
                    )
                    if cursor.rowcount != 1:
                        raise JobJournalError("job_journal_resume_conflict")
            except sqlite3.Error as exc:
                raise JobJournalError("job_journal_write_failed") from exc

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            try:
                with self._connect() as connection:
                    rows = connection.execute("SELECT * FROM jobs ORDER BY created, id").fetchall()
                return [_validated(dict(row)) for row in rows]
            except (sqlite3.Error, KeyError, TypeError, ValueError) as exc:
                raise JobJournalError("job_journal_read_failed") from exc
