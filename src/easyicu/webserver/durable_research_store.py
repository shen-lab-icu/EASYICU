"""Private command outbox for research workflows; never stores runner closures.

The deployment lease owns admission. DBOS owns workflow replay; the scientific
adapter owns authorization. Prepared commands cannot execute until the ordinary
submission owner has consumed its grant and bound the study to the job.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import threading
import time
from typing import Any, Iterator


class DurableResearchError(RuntimeError):
    """A durable command cannot safely advance."""


def runtime_digest() -> str:
    """Bind continuation to the installed Python code and clinical dictionaries."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted([*root.rglob("*.py"), *(root / "data").rglob("*.json")]):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class ResearchCommandStore:
    """Crash-safe admission and bounded recovery attempts under a private home."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if self.path.is_symlink() or not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise DurableResearchError("research_command_unsafe_path")
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        with self._connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise DurableResearchError("research_command_schema_invalid")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS commands (id TEXT PRIMARY KEY, "
                "payload TEXT NOT NULL, code_sha256 TEXT NOT NULL, status TEXT NOT NULL, "
                "attempts INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, reason TEXT)"
            )
            connection.execute("PRAGMA user_version=1")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = sqlite3.connect(self.path, timeout=5)
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("PRAGMA synchronous=FULL")
                with connection:
                    yield connection
            finally:
                connection.close()

    def prepare(self, job_id: str, payload: dict[str, Any]) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
            raise DurableResearchError("research_command_id_invalid")
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode()) > 128 * 1024:
            raise DurableResearchError("research_command_too_large")
        with self._connection() as connection:
            connection.execute("INSERT INTO commands VALUES (?, ?, ?, 'prepared', 0, ?, NULL)",
                               (job_id, encoded, runtime_digest(), time.time()))

    def activate(self, job_id: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute("UPDATE commands SET status='active' WHERE id=? AND status='prepared'", (job_id,))
            if cursor.rowcount != 1:
                raise DurableResearchError("research_command_activation_conflict")

    def prepare_active_handoff(self, job_id: str, payload: dict[str, Any]) -> None:
        """Atomically persist and activate one independently accepted handoff."""
        if (not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id)
                or payload.get("schema_version") != "easyicu.durable-preparation/3"):
            raise DurableResearchError("preparation_handoff_command_invalid")
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode()) > 128 * 1024:
            raise DurableResearchError("research_command_too_large")
        with self._connection() as connection:
            connection.execute("INSERT INTO commands VALUES (?, ?, ?, 'active', 0, ?, NULL)",
                               (job_id, encoded, runtime_digest(), time.time()))

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM commands WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["payload"] = json.loads(value["payload"])
        if (not isinstance(value["payload"], dict) or value["status"] not in {"prepared", "active", "terminal"}
            or type(value["attempts"]) is not int or value["attempts"] < 0):
            raise DurableResearchError("research_command_record_invalid")
        return value

    def active_ids(self) -> list[str]:
        with self._connection() as connection:
            return [row[0] for row in connection.execute("SELECT id FROM commands WHERE status='active' ORDER BY created")]

    def known_ids(self) -> set[str]:
        """Identify existing commands without re-admitting terminal jobs."""
        with self._connection() as connection:
            return {row[0] for row in connection.execute("SELECT id FROM commands")}

    def bind_provider(self, job_id: str, binding: dict[str, Any]) -> None:
        """Remember the original budget task before scientific work can proceed."""
        with self._connection() as connection:
            row = connection.execute("SELECT payload, status FROM commands WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return  # Legacy/non-durable work remains with its existing owner.
            if row["status"] != "active":
                raise DurableResearchError("research_command_not_active")
            payload = json.loads(row["payload"])
            prior = payload.get("provider_binding")
            if prior is not None:
                # Continuations may use another wrapper but can never switch
                # the original paid task or ledger to acquire a new budget.
                for key in ("hard_stop_ledger_path", "hard_stop_task_id", "hard_stop_declaration_sha256"):
                    if prior.get(key) != binding.get(key):
                        raise DurableResearchError("research_auto_resume_budget_binding_changed")
            payload["provider_binding"] = binding
            connection.execute("UPDATE commands SET payload=? WHERE id=? AND status='active'",
                               (json.dumps(payload, allow_nan=False, sort_keys=True), job_id))

    def request_cancel(self, job_id: str) -> None:
        """Independent cancel receipt; never frees capacity of a live runner."""
        with self._connection() as connection:
            row = connection.execute("SELECT payload FROM commands WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise DurableResearchError("research_command_missing")
            payload = json.loads(row["payload"])
            payload["_cancel_requested"] = True
            connection.execute("UPDATE commands SET payload=? WHERE id=?", (json.dumps(payload), job_id))

    def begin_attempt(self, job_id: str) -> int:
        with self._connection() as connection:
            cursor = connection.execute("UPDATE commands SET attempts=attempts+1 WHERE id=? AND status='active' AND attempts<4", (job_id,))
            if cursor.rowcount != 1:
                raise DurableResearchError("research_auto_resume_attempt_limit")
            return int(connection.execute("SELECT attempts FROM commands WHERE id=?", (job_id,)).fetchone()[0])

    def finish(self, job_id: str, reason: str) -> None:
        # Only machine codes enter the outbox, never exception text or results.
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", reason):
            reason = "research_auto_resume_failed"
        with self._connection() as connection:
            connection.execute("UPDATE commands SET status='terminal', reason=? WHERE id=? AND status!='terminal'", (reason, job_id))
