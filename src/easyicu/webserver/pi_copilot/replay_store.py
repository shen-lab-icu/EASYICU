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
from .projections import project_pi_replay_event

from easyicu.webserver import state_paths

SCHEMA_VERSION = "easyicu.pi-conversation-replay/1"
MAX_REPLAY_BYTES = 8 * 1024 * 1024
MAX_TURNS = 256
MAX_EVENTS_PER_TURN = 500
MAX_CHILD_JOBS = 64
DEFAULT_TURN_PAGE_SIZE = 48
MAX_TURN_PAGE_SIZE = 100


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
            else state_paths.state_root() / "pi-copilot-replay"
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
                    "The Copilot conversation replay exceeds its bounded contract.",
                    status_code=500,
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._empty(session_id, project_id)
        except json.JSONDecodeError as exc:
            raise PiCopilotError(
                "pi_replay_store_invalid",
                "The Copilot conversation replay is invalid JSON.",
                status_code=500,
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
        ):
            raise PiCopilotError(
                "pi_replay_store_invalid",
                "The Copilot conversation replay has an invalid schema.",
                status_code=500,
            )
        if (
            payload.get("session_id") != session_id
            or payload.get("project_id") != project_id
        ):
            raise PiCopilotError(
                "pi_replay_scope_mismatch",
                "The Copilot conversation replay belongs to another project or session.",
                status_code=409,
            )
        if not isinstance(payload.get("turns"), list) or not isinstance(
            payload.get("child_jobs"), list
        ):
            raise PiCopilotError(
                "pi_replay_store_invalid",
                "The Copilot conversation replay has an invalid shape.",
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
                "The Copilot conversation replay exceeds its bounded contract.",
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

    def record_host_action(
        self,
        *,
        session_id: str,
        project_id: str,
        action_id: str,
        action_code: str,
        action_key: Optional[str] = None,
        child_job_id: Optional[str] = None,
        status: str = "running",
    ) -> Dict[str, Any]:
        """Persist one typed host action in the same chronological replay.

        These actions are explicit button clicks owned by the EasyICU host, not
        model-authored transcript entries.  Keeping the enum and child-job
        coordinate here lets the browser reopen the interaction without
        inventing a user prompt or an assistant answer.
        """

        if status not in {"running", "done", "failed", "cancelled"}:
            raise ValueError(f"invalid host action status: {status}")
        with self._lock:
            payload = self._read(session_id, project_id)
            existing = next(
                (
                    row
                    for row in payload["turns"]
                    if isinstance(row, dict) and row.get("job_id") == action_id
                ),
                None,
            )
            if existing is not None:
                normalized_key = str(action_key or "").strip()
                if normalized_key and not existing.get("action_key"):
                    existing["action_key"] = normalized_key
                    self._write(payload)
                return dict(existing)
            now = utc_now()
            turn = {
                "job_id": action_id,
                "kind": "host_action",
                "action_code": str(action_code),
                "action_key": str(action_key or "").strip() or None,
                "child_job_id": str(child_job_id or "") or None,
                "status": status,
                "started_at": now,
                "ended_at": None if status == "running" else now,
                "allowed_actions": [],
                "events": [],
            }
            payload["turns"].append(turn)
            self._write(payload)
            return dict(turn)

    def finish_host_actions_for_child_job(
        self,
        *,
        session_id: str,
        project_id: str,
        child_job_id: str,
        status: str,
    ) -> None:
        if status not in {"done", "failed", "cancelled", "interrupted"}:
            raise ValueError(f"invalid child job status: {status}")
        with self._lock:
            payload = self._read(session_id, project_id)
            changed = False
            for turn in payload["turns"]:
                if (
                    isinstance(turn, dict)
                    and turn.get("kind") == "host_action"
                    and turn.get("child_job_id") == child_job_id
                ):
                    turn["status"] = status
                    turn["ended_at"] = utc_now()
                    changed = True
            if changed:
                self._write(payload)

    def running_host_action_child_job_ids(
        self,
        *,
        session_id: str,
        project_id: str,
    ) -> list[str]:
        """Return active-branch child jobs still presented as running."""

        with self._lock:
            payload = self._read(session_id, project_id)
        return list(
            dict.fromkeys(
                str(turn.get("child_job_id") or "").strip()
                for turn in payload["turns"]
                if isinstance(turn, dict)
                and not turn.get("superseded")
                and turn.get("kind") == "host_action"
                and turn.get("status") == "running"
                and str(turn.get("child_job_id") or "").strip()
            )
        )

    def append_event(
        self,
        *,
        session_id: str,
        project_id: str,
        job_id: str,
        event: Mapping[str, Any],
    ) -> None:
        projected = project_pi_replay_event(event)
        if projected is None:
            return
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
            turn["events"] = (list(turn.get("events") or []) + [projected])[
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

    def supersede_from_turn_index(
        self,
        *,
        session_id: str,
        project_id: str,
        turn_index: int,
    ) -> None:
        """Hide an abandoned conversation branch without deleting its audit rows."""
        if turn_index < 0:
            raise ValueError("turn_index must be non-negative")
        with self._lock:
            payload = self._read(session_id, project_id)
            active = [
                row
                for row in payload["turns"]
                if isinstance(row, dict) and not row.get("superseded")
            ]
            model_turns = [
                row for row in active if row.get("kind") != "host_action"
            ]
            if turn_index > len(model_turns):
                raise PiCopilotError(
                    "pi_replay_branch_invalid",
                    "The Copilot replay branch is inconsistent with the transcript.",
                    status_code=409,
                )
            cutoff = len(active)
            if turn_index < len(model_turns):
                cutoff = active.index(model_turns[turn_index])
            for row in active[cutoff:]:
                row["superseded"] = True
                row["superseded_at"] = utc_now()
            self._write(payload)

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
                str(turn.get("child_job_id") or "") == job_id
                or any(
                    str(event.get("job_id") or "") == job_id
                    for event in (turn.get("events") or [])
                    if isinstance(event, dict)
                )
                for turn in payload["turns"]
                if isinstance(turn, dict)
            )
            already_archived = any(
                isinstance(row, dict) and row.get("job_id") == job_id
                for row in payload["child_jobs"]
            )
            if not referenced and not already_archived:
                raise PiCopilotError(
                    "pi_replay_child_job_unbound",
                    "This child job was not submitted by the bound Copilot conversation.",
                    status_code=409,
                )
            payload["child_jobs"] = [
                row
                for row in payload["child_jobs"]
                if isinstance(row, dict) and row.get("job_id") != job_id
            ] + [dict(job)]
            self._write(payload)
            return dict(job)

    @staticmethod
    def _turn_page(
        turns: list[Any],
        *,
        cursor: Optional[str] = None,
        limit: int = DEFAULT_TURN_PAGE_SIZE,
    ) -> Dict[str, Any]:
        """Return one reverse-cursor page while preserving chronological order."""

        total = len(turns)
        page_size = max(1, min(MAX_TURN_PAGE_SIZE, int(limit)))
        if cursor is None or str(cursor).strip() == "":
            end = total
        else:
            raw = str(cursor).strip()
            if not raw.isdigit() or int(raw) > total:
                raise PiCopilotError(
                    "pi_replay_cursor_invalid",
                    "The Copilot conversation replay cursor is invalid.",
                    status_code=400,
                )
            end = int(raw)
        start = max(0, end - page_size)
        return {
            "items": list(turns[start:end]),
            "start": start,
            "end": end,
            "total": total,
            "has_more": start > 0,
            "next_cursor": str(start) if start > 0 else None,
        }

    def snapshot(
        self,
        *,
        session_id: str,
        project_id: str,
        cursor: Optional[str] = None,
        limit: int = DEFAULT_TURN_PAGE_SIZE,
    ) -> Dict[str, Any]:
        with self._lock:
            payload = self._read(session_id, project_id)
        active_turns = [
            row
            for row in payload["turns"]
            if isinstance(row, dict) and not row.get("superseded")
        ]
        active_child_job_ids = {
            str(event.get("job_id") or "").strip()
            for turn in active_turns
            for event in (turn.get("events") or [])
            if isinstance(event, Mapping) and str(event.get("job_id") or "").strip()
        }
        active_child_job_ids.update(
            str(turn.get("child_job_id") or "").strip()
            for turn in active_turns
            if str(turn.get("child_job_id") or "").strip()
        )
        active_child_jobs = [
            row
            for row in payload["child_jobs"]
            if isinstance(row, dict)
            and str(row.get("job_id") or "").strip() in active_child_job_ids
        ]
        turn_page = self._turn_page(
            active_turns,
            cursor=cursor,
            limit=limit,
        )
        digest_payload = {
            "schema_version": payload["schema_version"],
            "updated_at": payload.get("updated_at"),
            "turns": active_turns,
            # Child jobs are a projection of the active conversation branch,
            # not a second independent history. Superseded branches remain in
            # the private audit file but must not reappear in the Web dialogue.
            "child_jobs": active_child_jobs,
        }
        public = {
            **digest_payload,
            "turns": list(turn_page["items"]),
            "turn_page": turn_page,
        }
        normalized = json.dumps(
            digest_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        public["replay_sha256"] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return public

    def retire(self, session_id: str) -> None:
        with self._lock:
            self._path(session_id).unlink(missing_ok=True)
