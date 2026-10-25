"""Bounded conversation follow-ups; saved text never carries tool authority."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from .contracts import PiCopilotError, utc_now


class FollowupQueue:
    MAX_PENDING = 4

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._rows: list[dict[str, Any]] = []
        self._unavailable = False
        if path.exists():
            try:
                if path.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("oversized follow-up store")
                raw = json.loads(path.read_text(encoding="utf-8"))
                if (not isinstance(raw, list) or len(raw) > 320
                    or any(not isinstance(row, dict) or not isinstance(row.get("message"), str)
                        or len(row["message"]) > 12000
                        or not all(isinstance(row.get(key), str) and row[key] for key in ("id", "session_id", "project_id"))
                        or not isinstance(row.get("binding"), dict)
                        or not isinstance(row.get("status"), str)
                        or row.get("status") not in {"queued", "dispatching", "submitted", "interrupted", "cancelled"}
                        for row in raw)):
                    raise ValueError("invalid follow-up store")
                self._rows = raw
                for row in self._rows:
                    if row.get("status") in {"queued", "dispatching"}:
                        row.update(status="interrupted", error="server_restarted")
                self._save()
            except (OSError, ValueError, PiCopilotError):
                # A damaged queue must not silently become a fresh executable queue.
                self._unavailable = True

    def _save(self) -> None:
        if self._unavailable:
            raise PiCopilotError("pi_followup_store_unavailable", "Saved follow-up messages could not be read.", status_code=409)
        try:
            self._write_store()
        except OSError as exc:
            self._unavailable = True
            for row in self._rows:
                if row.get("status") in {"queued", "dispatching"}:
                    row.update(status="interrupted", error="queue_persistence_failed")
            raise PiCopilotError(
                "pi_followup_store_unavailable", "Saved follow-up messages could not be written.", status_code=503,
            ) from exc

    def _write_store(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".followups-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self._rows, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            rows = [dict(row) for row in self._rows if row.get("session_id") == session_id]
            return {"items": [{key: value for key, value in row.items() if key != "binding"} for row in rows],
                    "max_pending": self.MAX_PENDING, "available": not self._unavailable}

    def enqueue(self, *, session_id: str, project_id: str, message: str, binding: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._unavailable:
                raise PiCopilotError("pi_followup_store_unavailable", "Saved follow-up messages could not be read.", status_code=409)
            pending = [row for row in self._rows if row.get("status") in {"queued", "dispatching"}]
            if sum(row.get("session_id") == session_id for row in pending) >= self.MAX_PENDING or len(pending) >= 64:
                raise PiCopilotError("pi_followup_queue_full", "The follow-up queue is full.", status_code=429)
            # Retain a small visible terminal history as well as every pending message.
            terminal = [row for row in self._rows if row.get("status") not in {"queued", "dispatching"}]
            own = [row for row in terminal if row.get("session_id") == session_id][-7:]
            other = [row for row in terminal if row.get("session_id") != session_id][-248:]
            row = {"id": "followup_" + uuid.uuid4().hex[:16], "session_id": session_id,
                   "project_id": project_id, "message": message, "binding": binding,
                   "status": "queued", "created_at": utc_now()}
            self._rows = [*other, *own, *pending, row]
            self._save()
            return dict(row)

    def take(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            if self._unavailable:
                return None
            row = next((row for row in self._rows if row.get("session_id") == session_id and row.get("status") == "queued"), None)
            if row is None:
                return None
            row["status"] = "dispatching"
            self._save()
            return dict(row)

    def update(self, item_id: str, **values: Any) -> None:
        with self._lock:
            row = next((row for row in self._rows if row.get("id") == item_id), None)
            if row is not None:
                if row.get("status") == "cancelled" and values.get("status") in {"interrupted", "submitted"}:
                    return
                row.update(values)
                self._save()

    def cancel(self, session_id: str, item_id: str | None = None, *, error: str = "user_cancelled") -> None:
        with self._lock:
            for row in self._rows:
                if row.get("session_id") == session_id and row.get("status") in ({"queued", "interrupted"} if item_id else {"queued", "dispatching"}) and (item_id is None or row.get("id") == item_id):
                    row.update(status="cancelled" if error == "user_cancelled" else "interrupted", error=error)
            self._save()
