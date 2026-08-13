"""Persist browser-safe Pi lifecycle replay without owning scientific results."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .contracts import PiCopilotError, utc_now

SCHEMA_VERSION = "easyicu.pi-conversation-replay/1"
MAX_REPLAY_BYTES = 1024 * 1024
MAX_TURNS = 48
MAX_EVENTS_PER_TURN = 120
MAX_CHILD_JOBS = 16


class PiConversationReplayStore:
    """Own only the safe UX/audit projection of one Pi conversation.

    Pi's private JSONL remains the transcript owner. Research Agent, extraction,
    and idea-mining remain the scientific artifact owners. This store keeps the
    small lifecycle and job projections needed to reopen a truthful Web replay.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (
            Path(root)
            if root is not None
            else Path.home() / ".easyicu" / "pi-copilot-replay"
        )
        self._lock = threading.RLock()

    def _path(self, session_id: str) -> Path:
        identity = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()
        return self.root / f"{identity}.json"

    @staticmethod
    def _empty(session_id: str, project_id: str) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": str(session_id),
            "project_id": str(project_id),
            "updated_at": utc_now(),
            "turns": [],
            "child_jobs": [],
        }

    def _read(self, session_id: str, project_id: str) -> Dict[str, Any]:
        path = self._path(session_id)
        try:
            if path.stat().st_size > MAX_REPLAY_BYTES:
                raise PiCopilotError(
                    "pi_replay_store_too_large",
                    "The Pi conversation replay exceeds its bounded contract.",
                    status_code=500,
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._empty(session_id, project_id)
        except json.JSONDecodeError as exc:
            raise PiCopilotError(
                "pi_replay_store_invalid",
                "The Pi conversation replay is invalid JSON.",
                status_code=500,
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
        ):
            raise PiCopilotError(
                "pi_replay_store_invalid",
                "The Pi conversation replay has an invalid schema.",
                status_code=500,
            )
        if (
            payload.get("session_id") != session_id
            or payload.get("project_id") != project_id
        ):
            raise PiCopilotError(
                "pi_replay_scope_mismatch",
                "The Pi conversation replay belongs to another project or session.",
                status_code=409,
            )
        if not isinstance(payload.get("turns"), list) or not isinstance(
            payload.get("child_jobs"), list
        ):
            raise PiCopilotError(
                "pi_replay_store_invalid",
                "The Pi conversation replay has an invalid shape.",
                status_code=500,
            )
        return payload

    def _write(self, payload: Dict[str, Any]) -> None:
        payload["updated_at"] = utc_now()
        payload["turns"] = list(payload.get("turns") or [])[-MAX_TURNS:]
        payload["child_jobs"] = list(payload.get("child_jobs") or [])[-MAX_CHILD_JOBS:]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > MAX_REPLAY_BYTES:
            raise PiCopilotError(
                "pi_replay_store_too_large",
                "The Pi conversation replay exceeds its bounded contract.",
                status_code=500,
            )
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(self.root),
            prefix=".pi-replay-",
            suffix=".tmp",
            delete=False,
        )
        tmp = Path(handle.name)
        try:
            with handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.chmod(0o600)
            tmp.replace(self._path(str(payload["session_id"])))
        finally:
            tmp.unlink(missing_ok=True)

    def start_turn(
        self,
        *,
        session_id: str,
        project_id: str,
        job_id: str,
        allowed_actions: list[str],
    ) -> None:
        with self._lock:
            payload = self._read(session_id, project_id)
            turns = [
                row
                for row in payload["turns"]
                if isinstance(row, dict) and row.get("job_id") != job_id
            ]
            turns.append(
                {
                    "job_id": job_id,
                    "status": "running",
                    "started_at": utc_now(),
                    "ended_at": None,
                    "allowed_actions": sorted(set(allowed_actions))[:16],
                    "events": [],
                }
            )
            payload["turns"] = turns
            self._write(payload)

    def append_event(
        self,
        *,
        session_id: str,
        project_id: str,
        job_id: str,
        event: Mapping[str, Any],
    ) -> None:
        with self._lock:
            payload = self._read(session_id, project_id)
            turn = next(
                (
                    row
                    for row in reversed(payload["turns"])
                    if isinstance(row, dict) and row.get("job_id") == job_id
                ),
                None,
            )
            if turn is None:
                raise PiCopilotError(
                    "pi_replay_turn_missing",
                    "The Pi replay turn was not initialized.",
                    status_code=409,
                )
            turn["events"] = (list(turn.get("events") or []) + [dict(event)])[
                -MAX_EVENTS_PER_TURN:
            ]
            self._write(payload)

    def finish_turn(
        self,
        *,
        session_id: str,
        project_id: str,
        job_id: str,
        status: str,
    ) -> None:
        if status not in {"done", "failed", "cancelled", "interrupted"}:
            raise ValueError(f"invalid Pi replay status: {status}")
        with self._lock:
            payload = self._read(session_id, project_id)
            for turn in reversed(payload["turns"]):
                if isinstance(turn, dict) and turn.get("job_id") == job_id:
                    turn["status"] = status
                    turn["ended_at"] = utc_now()
                    self._write(payload)
                    return

    def archive_child_job(
        self,
        *,
        session_id: str,
        project_id: str,
        job: Mapping[str, Any],
    ) -> Dict[str, Any]:
        job_id = str(job.get("job_id") or "").strip()
        if not job_id:
            raise PiCopilotError(
                "pi_replay_child_job_invalid",
                "The projected child job is missing its identifier.",
                status_code=409,
            )
        with self._lock:
            payload = self._read(session_id, project_id)
            referenced = any(
                str(event.get("job_id") or "") == job_id
                for turn in payload["turns"]
                if isinstance(turn, dict)
                for event in (turn.get("events") or [])
                if isinstance(event, dict)
            )
            already_archived = any(
                isinstance(row, dict) and row.get("job_id") == job_id
                for row in payload["child_jobs"]
            )
            if not referenced and not already_archived:
                raise PiCopilotError(
                    "pi_replay_child_job_unbound",
                    "This child job was not submitted by the bound Pi conversation.",
                    status_code=409,
                )
            payload["child_jobs"] = [
                row
                for row in payload["child_jobs"]
                if isinstance(row, dict) and row.get("job_id") != job_id
            ] + [dict(job)]
            self._write(payload)
            return dict(job)

    def snapshot(self, *, session_id: str, project_id: str) -> Dict[str, Any]:
        with self._lock:
            payload = self._read(session_id, project_id)
        public = {
            "schema_version": payload["schema_version"],
            "updated_at": payload.get("updated_at"),
            "turns": list(payload["turns"]),
            "child_jobs": list(payload["child_jobs"]),
        }
        normalized = json.dumps(
            public,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        public["replay_sha256"] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return public

    def retire(self, session_id: str) -> None:
        with self._lock:
            self._path(session_id).unlink(missing_ok=True)
