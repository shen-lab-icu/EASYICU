"""Durable run/batch stop-loss for real Provider transport attempts.

The existing :mod:`provider_budget` ledger limits one analysis step.  This
module adds the two outer scopes needed by paid benchmark batches:

* one task/run; and
* the complete batch.

Every transport attempt reserves its worst-case token and cost allowance and
atomically persists that reservation *before* the transport starts.  A
successful response releases the conservative difference between the
reservation and provider-reported usage.

A failed or interrupted attempt stays charged, because a remote provider may
have accepted work even when the client never received usage metadata -- but
charged for what could actually be at risk, not for the worst case.  Both
worst-case terms are deliberate over-reservations with no report to settle
against, and each has an explicit release keyed to what the caller itself
authorized: the completion hold falls to ``requested_completion_tokens``, and
the prompt hold to ``estimate_prompt_tokens`` of the bytes actually sent.  See
``finish_transport_attempt``; each release records the measured run that
motivated it.

The ledger intentionally contains no prompts, responses, credentials, or
patient data.  It is also the batch progress checkpoint: task transitions and
their run directory/cost summary are written after every item.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
from threading import Lock
import time
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence
import uuid


PROVIDER_HARD_STOP_SCHEMA = "easyicu.provider_hard_stop_ledger/1"
PROVIDER_PROMPT_OVERHEAD_TOKEN_RESERVATION = 4096
# OpenAI OAuth gateways can legitimately strip caller-supplied output caps
# before forwarding to the ChatGPT internal endpoint. Reserve the published
# GPT-5 maximum output envelope before every paid attempt so token/cost stop
# losses remain pre-transport even when the serving path cannot enforce the
# requested per-response cap. Successful calls release the difference.
PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR = 128_000
_MAX_LEDGER_BYTES = 16 * 1024 * 1024
_TERMINAL_TASK_STATES = frozenset(
    {"completed", "failed", "batch_canary_blocked", "budget_exhausted"}
)


class ProviderHardStopError(RuntimeError):
    """Base class for durable global Provider stop-loss failures."""


class ProviderHardStopExceeded(ProviderHardStopError):
    """Raised before transport when a frozen run/batch ceiling is unavailable."""

    def __init__(self, *, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


class ProviderHardStopLedgerError(ProviderHardStopError):
    """Raised when the durable ledger cannot be trusted or persisted."""


@dataclass(frozen=True)
class ProviderHardStopLimits:
    """Frozen ceilings and explicit price assumptions for one batch."""

    max_provider_attempts_per_run: int
    max_provider_attempts_per_batch: int
    max_total_tokens_per_run: int
    max_total_tokens_per_batch: int
    max_estimated_cost_usd_per_batch: float
    max_wall_clock_seconds_per_task: float
    input_cost_usd_per_million_tokens: float
    output_cost_usd_per_million_tokens: float

    def __post_init__(self) -> None:
        integer_fields = (
            "max_provider_attempts_per_run",
            "max_provider_attempts_per_batch",
            "max_total_tokens_per_run",
            "max_total_tokens_per_batch",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        numeric_fields = (
            "max_estimated_cost_usd_per_batch",
            "max_wall_clock_seconds_per_task",
            "input_cost_usd_per_million_tokens",
            "output_cost_usd_per_million_tokens",
        )
        for name in numeric_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_estimated_cost_usd_per_batch <= 0:
            raise ValueError("max_estimated_cost_usd_per_batch must be positive")
        if self.max_wall_clock_seconds_per_task <= 0:
            raise ValueError("max_wall_clock_seconds_per_task must be positive")
        if self.max_provider_attempts_per_batch < self.max_provider_attempts_per_run:
            raise ValueError(
                "batch Provider-attempt ceiling cannot be below the per-run ceiling"
            )
        if self.max_total_tokens_per_batch < self.max_total_tokens_per_run:
            raise ValueError("batch token ceiling cannot be below the per-run ceiling")

    def canonical_payload(self) -> Dict[str, object]:
        return dict(asdict(self))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _payload_digest(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    """Publish one complete owner-only ledger and fsync its directory."""

    parent = path.parent
    parent_info = parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise ProviderHardStopLedgerError(
            "Provider hard-stop ledger parent must be a real directory"
        )
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ProviderHardStopLedgerError(
            "Provider hard-stop ledger destination must be a regular file"
        )
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    body = dict(payload)
    body["sha256"] = _payload_digest(body)
    raw = _canonical_bytes(body)
    old_mask = None
    descriptor: Optional[int] = None
    if hasattr(signal, "pthread_sigmask"):
        old_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            {signal.SIGINT, signal.SIGTERM},
        )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short Provider hard-stop ledger write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception as exc:
        raise ProviderHardStopLedgerError(
            f"Could not persist Provider hard-stop ledger: {path}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if old_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


def load_provider_hard_stop_ledger(path: Path) -> Dict[str, object]:
    """Strictly load and digest-check a persisted ledger."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ProviderHardStopLedgerError(
            "Provider hard-stop ledger must be a regular non-symlink file"
        )
    raw = candidate.read_bytes()
    if len(raw) > _MAX_LEDGER_BYTES:
        raise ProviderHardStopLedgerError("Provider hard-stop ledger is too large")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ProviderHardStopLedgerError(
            "Provider hard-stop ledger is not strict JSON"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PROVIDER_HARD_STOP_SCHEMA
        or payload.get("sha256") != _payload_digest(payload)
    ):
        raise ProviderHardStopLedgerError(
            "Provider hard-stop ledger schema or digest is invalid"
        )
    return payload


def _prompt_payload_bytes(messages: Sequence[Any]) -> int:
    """The caller-visible prompt bytes this attempt is about to transmit."""

    return sum(
        len(str(getattr(message, "content", "") or "").encode("utf-8"))
        for message in messages
    )


def _prompt_token_reservation(messages: Sequence[Any]) -> int:
    """Conservative bound including provider-side prompt framing.

    Message bytes alone bound tokens produced from caller-visible content, but
    OpenAI-compatible gateways may prepend model instructions that are absent
    from the request payload while still reporting them as prompt usage.  Luna
    on the local gateway has demonstrated this behaviour.  Keep a fixed,
    provider-independent allowance so a normal successful response is not
    misclassified as a budget overflow; successful calls release the unused
    reservation back to their provider-reported usage.
    """

    prompt_bytes = sum(
        len(str(getattr(message, "content", "") or "").encode("utf-8"))
        for message in messages
    )
    # Every tokenizer token consumes at least one encoded byte for normal
    # message content. The small term covers caller-visible roles/framing; the
    # larger constant covers provider-owned instructions and special tokens.
    return max(
        1,
        prompt_bytes
        + 16 * len(messages)
        + 64
        + PROVIDER_PROMPT_OVERHEAD_TOKEN_RESERVATION,
    )


def _safe_score_summary(score: Optional[Mapping[str, object]]) -> Dict[str, object]:
    if not isinstance(score, Mapping):
        return {}
    aware = score.get("aware")
    selected = aware if isinstance(aware, Mapping) else None
    if selected is None:
        naive = score.get("naive")
        selected = naive if isinstance(naive, Mapping) else None
    if selected is None:
        return {}
    cost = selected.get("cost_summary")
    return {
        "run_id": selected.get("run_id"),
        "workdir": selected.get("workdir"),
        "paper_authorized": bool(selected.get("paper_authorized")),
        "provider_calls_reported": (
            cost.get("n_calls") if isinstance(cost, Mapping) else None
        ),
        "tokens_reported": (
            cost.get("total_tokens") if isinstance(cost, Mapping) else None
        ),
        "estimated_cost_usd_reported": (
            cost.get("total_cost_usd") if isinstance(cost, Mapping) else None
        ),
    }


def _safe_error_summary(error: Optional[str]) -> Optional[Dict[str, str]]:
    """Retain diagnostic identity without persisting exception/body contents."""

    if error is None:
        return None
    text = str(error)
    raw_type = text.split(":", 1)[0].strip()
    safe_type = "".join(
        character
        for character in raw_type
        if character.isalnum() or character in {"_", ".", "-"}
    )[:128]
    return {
        "type": safe_type or "Error",
        "message_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


class ProviderHardStopLedger:
    """Thread-safe durable accounting shared by every task in one batch."""

    def __init__(
        self,
        *,
        path: Path,
        task_ids: Sequence[str],
        limits: ProviderHardStopLimits,
        batch_id: Optional[str] = None,
        declaration_sha256: Optional[str] = None,
        resume_existing: bool = False,
    ) -> None:
        normalized_ids = tuple(str(task_id).strip() for task_id in task_ids)
        if (
            not normalized_ids
            or any(not task_id for task_id in normalized_ids)
            or len(set(normalized_ids)) != len(normalized_ids)
        ):
            raise ValueError("Provider hard-stop task ids must be unique and non-empty")
        self.path = Path(path).expanduser()
        if not self.path.is_absolute():
            raise ValueError("Provider hard-stop ledger path must be absolute")
        self.limits = limits
        self._lock = Lock()
        self._task_started_monotonic: Dict[str, float] = {}
        now = _utc_now()
        initial_payload: Dict[str, object] = {
            "schema_version": PROVIDER_HARD_STOP_SCHEMA,
            "batch_id": str(batch_id).strip() if batch_id else None,
            "declaration_sha256": (
                str(declaration_sha256).strip() if declaration_sha256 else None
            ),
            "created_at": now,
            "updated_at": now,
            "limits": limits.canonical_payload(),
            "task_order": list(normalized_ids),
            "tasks": [
                {
                    "task_id": task_id,
                    "status": "pending",
                    "started_at": None,
                    "finished_at": None,
                    "elapsed_seconds": None,
                    "error": None,
                    "blocked_by": None,
                    "score_summary": {},
                    "calls": [],
                }
                for task_id in normalized_ids
            ],
            "totals": {},
            "terminal": False,
        }
        if self.path.exists():
            if not resume_existing:
                raise FileExistsError(self.path)
            loaded = load_provider_hard_stop_ledger(self.path)
            if (
                loaded.get("limits") != limits.canonical_payload()
                or loaded.get("task_order") != list(normalized_ids)
                or loaded.get("batch_id") != initial_payload["batch_id"]
                or loaded.get("declaration_sha256")
                != initial_payload["declaration_sha256"]
            ):
                raise ProviderHardStopLedgerError(
                    "Existing Provider hard-stop ledger differs from this invocation"
                )
            loaded.pop("sha256", None)
            self._payload = loaded
            now_monotonic = time.monotonic()
            now_utc = datetime.now(timezone.utc)
            for task in self._tasks_locked():
                if task.get("status") != "running":
                    continue
                try:
                    started_at = datetime.fromisoformat(str(task["started_at"]))
                    elapsed = max(0.0, (now_utc - started_at).total_seconds())
                except (KeyError, TypeError, ValueError):
                    elapsed = self.limits.max_wall_clock_seconds_per_task
                self._task_started_monotonic[str(task["task_id"])] = (
                    now_monotonic - elapsed
                )
        else:
            self._payload = initial_payload
            with self._lock:
                self._persist_locked()

    def _tasks_locked(self) -> list[Dict[str, object]]:
        tasks = self._payload.get("tasks")
        if not isinstance(tasks, list) or any(
            not isinstance(task, dict) for task in tasks
        ):
            raise ProviderHardStopLedgerError("Provider hard-stop tasks are invalid")
        return tasks  # type: ignore[return-value]

    def _task_locked(self, task_id: str) -> Dict[str, object]:
        matches = [
            task for task in self._tasks_locked() if task.get("task_id") == str(task_id)
        ]
        if len(matches) != 1:
            raise ProviderHardStopLedgerError(
                f"Provider hard-stop task is not uniquely declared: {task_id!r}"
            )
        return matches[0]

    def _totals_locked(self) -> Dict[str, object]:
        attempts = 0
        accounted_tokens = 0
        estimated_cost = 0.0
        completed = 0
        failed = 0
        for task in self._tasks_locked():
            status = str(task.get("status") or "")
            completed += int(status == "completed")
            failed += int(status in _TERMINAL_TASK_STATES - {"completed"})
            calls = task.get("calls")
            if not isinstance(calls, list):
                raise ProviderHardStopLedgerError(
                    "Provider hard-stop call history is invalid"
                )
            for call in calls:
                if not isinstance(call, dict):
                    raise ProviderHardStopLedgerError(
                        "Provider hard-stop call record is invalid"
                    )
                attempts += 1
                accounted_tokens += int(call.get("accounted_tokens") or 0)
                estimated_cost += float(call.get("accounted_estimated_cost_usd") or 0.0)
        return {
            "provider_attempts": attempts,
            "accounted_tokens": accounted_tokens,
            "accounted_estimated_cost_usd": round(estimated_cost, 12),
            "completed_tasks": completed,
            "failed_or_blocked_tasks": failed,
        }

    def _persist_locked(self) -> None:
        self._payload["updated_at"] = _utc_now()
        self._payload["totals"] = self._totals_locked()
        statuses = {str(task.get("status") or "") for task in self._tasks_locked()}
        self._payload["terminal"] = bool(statuses and statuses <= _TERMINAL_TASK_STATES)
        _atomic_write(self.path, self._payload)

    def start_task(
        self,
        task_id: str,
        *,
        reopen_terminal: bool = False,
    ) -> "TaskProviderHardStop":
        """Start one task or explicitly reopen a resumable terminal attempt.

        Normal batch reuse keeps ``completed`` tasks immutable.  An explicit
        step-level run resume may reopen only ``completed`` or ``failed`` tasks;
        their calls and elapsed wall clock remain cumulative so a resume cannot
        reset any Provider stop-loss.  Budget exhaustion and canary blocking
        are never resumable through this path.
        """

        normalized = str(task_id).strip()
        with self._lock:
            task = self._task_locked(normalized)
            if task.get("status") == "running":
                if normalized not in self._task_started_monotonic:
                    raise ProviderHardStopLedgerError(
                        "Running Provider hard-stop task lost its wall clock"
                    )
                return TaskProviderHardStop(self, normalized)
            status = str(task.get("status") or "")
            if status == "completed" and not reopen_terminal:
                return TaskProviderHardStop(self, normalized)
            if reopen_terminal and status in {"completed", "failed"}:
                calls = task.get("calls")
                if not isinstance(calls, list) or any(
                    isinstance(call, Mapping) and call.get("state") == "in_progress"
                    for call in calls
                ):
                    raise ProviderHardStopLedgerError(
                        "Terminal Provider hard-stop task has an unsafe call history"
                    )
                try:
                    prior_elapsed = float(task.get("elapsed_seconds") or 0.0)
                except (TypeError, ValueError) as exc:
                    raise ProviderHardStopLedgerError(
                        "Terminal Provider hard-stop task elapsed time is invalid"
                    ) from exc
                if not math.isfinite(prior_elapsed) or prior_elapsed < 0.0:
                    raise ProviderHardStopLedgerError(
                        "Terminal Provider hard-stop task elapsed time is invalid"
                    )
                attempts = task.get("terminal_attempts")
                if attempts is None:
                    attempts = []
                    task["terminal_attempts"] = attempts
                if not isinstance(attempts, list) or any(
                    not isinstance(attempt, Mapping) for attempt in attempts
                ):
                    raise ProviderHardStopLedgerError(
                        "Provider hard-stop terminal attempt history is invalid"
                    )
                attempts.append(
                    {
                        "status": status,
                        "started_at": task.get("started_at"),
                        "finished_at": task.get("finished_at"),
                        "elapsed_seconds": prior_elapsed,
                        "error": task.get("error"),
                        "score_summary": task.get("score_summary"),
                    }
                )
                task["resume_count"] = len(attempts)
                task["status"] = "running"
                task["finished_at"] = None
                task["elapsed_seconds"] = None
                task["error"] = None
                task["blocked_by"] = None
                task["score_summary"] = {}
                if not task.get("started_at"):
                    task["started_at"] = _utc_now()
                self._task_started_monotonic[normalized] = (
                    time.monotonic() - prior_elapsed
                )
                self._persist_locked()
                return TaskProviderHardStop(self, normalized)
            if status != "pending":
                raise ProviderHardStopLedgerError(
                    f"Provider hard-stop task cannot start from {status!r}"
                )
            task["status"] = "running"
            task["started_at"] = _utc_now()
            self._task_started_monotonic[normalized] = time.monotonic()
            self._persist_locked()
        return TaskProviderHardStop(self, normalized)

    def mark_task_blocked(self, task_id: str, *, blocked_by: str) -> None:
        with self._lock:
            task = self._task_locked(task_id)
            if task.get("status") != "pending":
                raise ProviderHardStopLedgerError(
                    "Only a pending task can be marked batch-canary blocked"
                )
            task["status"] = "batch_canary_blocked"
            task["blocked_by"] = str(blocked_by)
            task["finished_at"] = _utc_now()
            self._persist_locked()

    def _elapsed_locked(self, task_id: str) -> float:
        started = self._task_started_monotonic.get(task_id)
        if started is None:
            raise ProviderHardStopLedgerError(
                f"Provider hard-stop task has no live clock: {task_id!r}"
            )
        return max(0.0, time.monotonic() - started)

    def assert_task_active(self, task_id: str) -> float:
        with self._lock:
            task = self._task_locked(task_id)
            if task.get("status") != "running":
                raise ProviderHardStopExceeded(
                    code="TASK_NOT_RUNNING",
                    detail=f"task {task_id!r} is {task.get('status')!r}",
                )
            elapsed = self._elapsed_locked(task_id)
            remaining = self.limits.max_wall_clock_seconds_per_task - elapsed
            if remaining <= 0:
                task["status"] = "budget_exhausted"
                task["finished_at"] = _utc_now()
                task["elapsed_seconds"] = round(elapsed, 6)
                task["error"] = "TASK_WALL_CLOCK_EXHAUSTED"
                self._persist_locked()
                raise ProviderHardStopExceeded(
                    code="TASK_WALL_CLOCK_EXHAUSTED",
                    detail=(
                        f"task {task_id!r} elapsed {elapsed:.3f}s; "
                        f"limit={self.limits.max_wall_clock_seconds_per_task:.3f}s"
                    ),
                )
            return remaining

    def _task_totals_locked(self, task: Mapping[str, object]) -> Dict[str, float]:
        raw_calls = task.get("calls")
        calls = raw_calls if isinstance(raw_calls, list) else []
        return {
            "attempts": float(len(calls)),
            "tokens": float(
                sum(
                    int(call.get("accounted_tokens") or 0)
                    for call in calls
                    if isinstance(call, Mapping)
                )
            ),
        }

    def reserve_transport_attempt(
        self,
        *,
        task_id: str,
        role: Optional[str],
        model: str,
        messages: Sequence[Any],
        max_tokens: int,
        prior_attempt_id: Optional[str],
    ) -> str:
        """Reserve one maximum-cost transport attempt before network delivery."""

        # A second reservation in one logical call means the previous raw
        # transport failed and the reviewed client is about to retry. Close and
        # persist that attempt before evaluating whether another one is
        # affordable. This keeps the on-disk ledger truthful even when the new
        # attempt is denied by a wall-clock, token, cost, or attempt ceiling.
        if prior_attempt_id:
            self.finish_transport_attempt(
                task_id=task_id,
                attempt_id=prior_attempt_id,
                usage=None,
                error_type="TransportRetry",
            )
        self.assert_task_active(task_id)
        prompt_reserve = _prompt_token_reservation(messages)
        prompt_payload_bytes = _prompt_payload_bytes(messages)
        requested_completion = max(1, int(max_tokens))
        completion_reserve = max(
            requested_completion,
            PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR,
        )
        token_reserve = prompt_reserve + completion_reserve
        cost_reserve = (
            prompt_reserve * self.limits.input_cost_usd_per_million_tokens
            + completion_reserve * self.limits.output_cost_usd_per_million_tokens
        ) / 1_000_000.0
        with self._lock:
            task = self._task_locked(task_id)
            calls = task.get("calls")
            if not isinstance(calls, list):
                raise ProviderHardStopLedgerError(
                    "Provider hard-stop call history is invalid"
                )
            task_totals = self._task_totals_locked(task)
            batch_totals = self._totals_locked()
            if (
                int(task_totals["attempts"]) + 1
                > self.limits.max_provider_attempts_per_run
            ):
                raise ProviderHardStopExceeded(
                    code="RUN_PROVIDER_ATTEMPT_LIMIT",
                    detail=(
                        f"task {task_id!r} reached "
                        f"{self.limits.max_provider_attempts_per_run} attempts"
                    ),
                )
            if (
                int(batch_totals["provider_attempts"]) + 1
                > self.limits.max_provider_attempts_per_batch
            ):
                raise ProviderHardStopExceeded(
                    code="BATCH_PROVIDER_ATTEMPT_LIMIT",
                    detail=(
                        "batch reached "
                        f"{self.limits.max_provider_attempts_per_batch} attempts"
                    ),
                )
            if (
                int(task_totals["tokens"]) + token_reserve
                > self.limits.max_total_tokens_per_run
            ):
                raise ProviderHardStopExceeded(
                    code="RUN_TOKEN_LIMIT",
                    detail=(
                        f"task {task_id!r} cannot reserve {token_reserve} tokens "
                        f"under {self.limits.max_total_tokens_per_run}"
                    ),
                )
            if (
                int(batch_totals["accounted_tokens"]) + token_reserve
                > self.limits.max_total_tokens_per_batch
            ):
                raise ProviderHardStopExceeded(
                    code="BATCH_TOKEN_LIMIT",
                    detail=(
                        f"batch cannot reserve {token_reserve} tokens under "
                        f"{self.limits.max_total_tokens_per_batch}"
                    ),
                )
            if (
                float(batch_totals["accounted_estimated_cost_usd"]) + cost_reserve
                > self.limits.max_estimated_cost_usd_per_batch + 1e-12
            ):
                raise ProviderHardStopExceeded(
                    code="BATCH_COST_LIMIT",
                    detail=(
                        f"batch cannot reserve ${cost_reserve:.6f} under "
                        f"${self.limits.max_estimated_cost_usd_per_batch:.6f}"
                    ),
                )
            attempt_id = uuid.uuid4().hex
            calls.append(
                {
                    "attempt_id": attempt_id,
                    "sequence_in_task": len(calls) + 1,
                    "state": "in_progress",
                    "role": str(role) if role else None,
                    "model": str(model),
                    "started_at": _utc_now(),
                    "finished_at": None,
                    "prompt_token_reservation": prompt_reserve,
                    "prompt_payload_bytes": prompt_payload_bytes,
                    "provider_prompt_overhead_token_reservation": (
                        PROVIDER_PROMPT_OVERHEAD_TOKEN_RESERVATION
                    ),
                    "requested_completion_tokens": requested_completion,
                    "completion_token_reservation": completion_reserve,
                    "provider_completion_token_reservation_floor": (
                        PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR
                    ),
                    "accounted_tokens": token_reserve,
                    "reported_prompt_tokens": None,
                    "reported_completion_tokens": None,
                    "accounted_estimated_cost_usd": cost_reserve,
                    "error_type": None,
                }
            )
            # The reservation is durable before the caller can touch transport.
            self._persist_locked()
            return attempt_id

    def finish_transport_attempt(
        self,
        *,
        task_id: str,
        attempt_id: str,
        usage: Optional[Mapping[str, object]],
        error_type: Optional[str],
    ) -> None:
        with self._lock:
            task = self._task_locked(task_id)
            calls = task.get("calls")
            matches = [
                call
                for call in (calls if isinstance(calls, list) else [])
                if isinstance(call, dict) and call.get("attempt_id") == attempt_id
            ]
            if len(matches) != 1 or matches[0].get("state") != "in_progress":
                raise ProviderHardStopLedgerError(
                    "Provider attempt is not a unique in-progress call"
                )
            call = matches[0]
            call["finished_at"] = _utc_now()
            call["error_type"] = str(error_type) if error_type else None
            if error_type is not None or not isinstance(usage, Mapping):
                call["state"] = (
                    "failed_usage_unknown"
                    if error_type is not None
                    else "completed_usage_unreported"
                )
                # A RESERVATION IS A HOLD, AND EVERY HOLD NEEDS A RELEASE PATH.
                #
                # ``completion_reserve`` is deliberately taken as
                # ``max(requested_completion, FLOOR)`` before transport, because
                # a gateway may return more than we asked for and we must not
                # start a call we could not afford in the worst case.  A
                # reported call then releases that hold down to the provider's
                # own numbers.  An UNREPORTED call had no release path at all,
                # so it stayed charged at the floor forever.
                #
                # MEASURED on a survival-analysis fixture, 2026-08-03.  The
                # local gateway was answering
                # HTTP 500 in 0.98 s with ``Post ".../responses": EOF`` -- the
                # upstream connection died while the request was still being
                # sent, about a quarter of the time a successful call needs to
                # come back (3.5-7.9 s).  Every one of the 14 attempts asked for
                # 4,096 completion tokens (2,048 for repair).  The 10 that died
                # were each charged the 128,000 floor: 1,848,481 of the run's
                # 2,000,000 tokens, and $45.39 of the batch's $100, for output
                # that provably never existed.  The run died at step 3 of 9 --
                # not on any analysis defect, on its own accounting.
                #
                # The floor exists to absorb a provider REPORTING more than it
                # was asked for; with no report there is nothing to absorb, and
                # ``requested_completion`` is the true ceiling on what the
                # provider could have produced for this request.  So release the
                # completion hold to what the caller actually authorized and
                # keep the prompt reservation in full -- those bytes may have
                # reached the provider and been billed.  Attempt storms stay
                # bounded by ``max_provider_attempts_per_{run,batch}``, which is
                # the guard that owns retry count; the token ceiling should
                # charge tokens that could actually be at risk.
                #
                # Same 14 attempts under this rule: 699,021 tokens instead of
                # 1,944,205, leaving 1.3M for the analysis that was starved.
                requested_completion = int(call.get("requested_completion_tokens") or 0)
                held_completion = int(call.get("completion_token_reservation") or 0)
                if 0 < requested_completion < held_completion:
                    released = held_completion - requested_completion
                    call["completion_token_reservation"] = requested_completion
                    call["accounted_tokens"] = max(
                        0, int(call.get("accounted_tokens") or 0) - released
                    )
                    call["accounted_estimated_cost_usd"] = max(
                        0.0,
                        float(call.get("accounted_estimated_cost_usd") or 0.0)
                        - (released * self.limits.output_cost_usd_per_million_tokens)
                        / 1_000_000.0,
                    )
                    call["unreported_completion_hold_released"] = released

                # THE PROMPT HOLD NEEDS THE SAME RELEASE PATH, FOR THE SAME
                # REASON.
                #
                # ``_prompt_token_reservation`` bounds a prompt by its UTF-8
                # byte count -- one token per byte, which is a true ceiling and
                # about four times the truth. Its own docstring says why that
                # is safe: "successful calls release the unused reservation
                # back to their provider-reported usage." A failed call had no
                # such release and kept the byte-denominated hold forever.
                #
                # MEASURED on verify12, 2026-08-04, from the batch's durable
                # ledger: a successful call was charged 23,436 tokens and a
                # failed one 90,542 -- a call that returned nothing cost 3.9x
                # one that returned an answer. Over the m1 run, 19 of 39 calls
                # failed and took 707,014 tokens, 35% of the 2,000,000 run
                # ceiling and 2.75x the 256,708 the successful calls actually
                # used. All nine analysis steps then passed and the manuscript
                # writer was refused a 150,931-token reservation.
                #
                # The release lands on the estimator this codebase already
                # calibrated for exactly this question rather than a new
                # number: ``estimate_prompt_tokens`` divides by
                # ``CONSERVATIVE_BYTES_PER_TOKEN = 3.0``, deliberately below
                # the 3.7685 minimum measured over real receipts, so the
                # charge still over-counts every observed ratio. The provider
                # framing allowance is already in tokens and is kept whole.
                # A ledger written before this field existed keeps its full
                # hold: no bytes recorded, nothing to release.
                payload_bytes = int(call.get("prompt_payload_bytes") or 0)
                held_prompt = int(call.get("prompt_token_reservation") or 0)
                if payload_bytes > 0:
                    from ..providers.prompt_budget import estimate_prompt_tokens

                    released_prompt = (
                        estimate_prompt_tokens(payload_bytes)
                        + PROVIDER_PROMPT_OVERHEAD_TOKEN_RESERVATION
                    )
                    if 0 < released_prompt < held_prompt:
                        given_back = held_prompt - released_prompt
                        call["prompt_token_reservation"] = released_prompt
                        call["accounted_tokens"] = max(
                            0, int(call.get("accounted_tokens") or 0) - given_back
                        )
                        call["accounted_estimated_cost_usd"] = max(
                            0.0,
                            float(call.get("accounted_estimated_cost_usd") or 0.0)
                            - (
                                given_back
                                * self.limits.input_cost_usd_per_million_tokens
                            )
                            / 1_000_000.0,
                        )
                        call["unreported_prompt_hold_released"] = given_back
                self._persist_locked()
                return
            prompt_tokens = max(0, int(usage.get("prompt_tokens") or 0))
            completion_tokens = max(0, int(usage.get("completion_tokens") or 0))
            component_total = prompt_tokens + completion_tokens
            reported_total = max(
                component_total,
                max(0, int(usage.get("total_tokens") or 0)),
            )
            reserved_total = int(call.get("accounted_tokens") or 0)
            reserved_completion = int(call.get("completion_token_reservation") or 0)
            overflow_code: Optional[str] = None
            overflow_detail: Optional[str] = None
            if completion_tokens > reserved_completion:
                overflow_code = "PROVIDER_COMPLETION_USAGE_EXCEEDED_RESERVATION"
                overflow_detail = (
                    f"Provider reported {completion_tokens} completion tokens after "
                    f"a {reserved_completion}-token completion reservation"
                )
            elif reported_total > reserved_total:
                overflow_code = "PROVIDER_USAGE_EXCEEDED_RESERVATION"
                overflow_detail = (
                    f"Provider reported {reported_total} tokens after a "
                    f"{reserved_total}-token reservation"
                )
            if overflow_code is not None:
                call["state"] = "completed_usage_overflow"
                call["reported_prompt_tokens"] = prompt_tokens
                call["reported_completion_tokens"] = completion_tokens
                call["reported_total_tokens"] = reported_total
                call["accounted_tokens"] = reported_total
                unknown_tokens = reported_total - component_total
                call["accounted_estimated_cost_usd"] = (
                    prompt_tokens * self.limits.input_cost_usd_per_million_tokens
                    + completion_tokens * self.limits.output_cost_usd_per_million_tokens
                    + unknown_tokens
                    * max(
                        self.limits.input_cost_usd_per_million_tokens,
                        self.limits.output_cost_usd_per_million_tokens,
                    )
                ) / 1_000_000.0
                self._persist_locked()
                raise ProviderHardStopExceeded(
                    code=overflow_code,
                    detail=str(overflow_detail),
                )
            call["state"] = "completed"
            call["reported_prompt_tokens"] = prompt_tokens
            call["reported_completion_tokens"] = completion_tokens
            call["reported_total_tokens"] = reported_total
            call["accounted_tokens"] = reported_total
            unknown_tokens = reported_total - component_total
            call["accounted_estimated_cost_usd"] = (
                prompt_tokens * self.limits.input_cost_usd_per_million_tokens
                + completion_tokens * self.limits.output_cost_usd_per_million_tokens
                + unknown_tokens
                * max(
                    self.limits.input_cost_usd_per_million_tokens,
                    self.limits.output_cost_usd_per_million_tokens,
                )
            ) / 1_000_000.0
            self._persist_locked()

    def finish_task(
        self,
        task_id: str,
        *,
        score: Optional[Mapping[str, object]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            if self._task_locked(task_id).get("status") == "completed":
                return
        # Check the wall clock before granting a completed state.
        if error is None:
            self.assert_task_active(task_id)
        with self._lock:
            task = self._task_locked(task_id)
            if task.get("status") != "running":
                if task.get("status") == "budget_exhausted":
                    return
                raise ProviderHardStopLedgerError(
                    f"Provider hard-stop task cannot finish from {task.get('status')!r}"
                )
            elapsed = self._elapsed_locked(task_id)
            task["status"] = "failed" if error is not None else "completed"
            task["finished_at"] = _utc_now()
            task["elapsed_seconds"] = round(elapsed, 6)
            task["error"] = _safe_error_summary(error)
            task["score_summary"] = _safe_score_summary(score)
            self._persist_locked()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            # Canonical JSON round-trip returns an isolated plain-data snapshot.
            payload = dict(self._payload)
            payload["sha256"] = _payload_digest(payload)
            return json.loads(json.dumps(payload, allow_nan=False))

    def task_accounting_summary(self, task_id: str) -> Dict[str, object]:
        """Return actual, unknown, and conservative accounting for one task."""

        with self._lock:
            task = self._task_locked(task_id)
            calls = task.get("calls")
            if not isinstance(calls, list):
                raise ProviderHardStopLedgerError(
                    "Provider hard-stop call history is invalid"
                )
            reported_calls = 0
            reported_prompt = 0
            reported_completion = 0
            reported_total = 0
            reported_cost = 0.0
            unknown_calls = 0
            unknown_accounted_tokens = 0
            unknown_accounted_cost = 0.0
            accounted_tokens = 0
            accounted_cost = 0.0
            unknown_states: Dict[str, int] = {}
            for raw_call in calls:
                if not isinstance(raw_call, Mapping):
                    raise ProviderHardStopLedgerError(
                        "Provider hard-stop call record is invalid"
                    )
                call_tokens = int(raw_call.get("accounted_tokens") or 0)
                call_cost = float(raw_call.get("accounted_estimated_cost_usd") or 0.0)
                accounted_tokens += call_tokens
                accounted_cost += call_cost
                raw_reported_total = raw_call.get("reported_total_tokens")
                if raw_reported_total is not None:
                    reported_calls += 1
                    reported_prompt += int(raw_call.get("reported_prompt_tokens") or 0)
                    reported_completion += int(
                        raw_call.get("reported_completion_tokens") or 0
                    )
                    reported_total += int(raw_reported_total or 0)
                    reported_cost += call_cost
                    continue
                unknown_calls += 1
                unknown_accounted_tokens += call_tokens
                unknown_accounted_cost += call_cost
                state = str(raw_call.get("state") or "unknown")
                unknown_states[state] = unknown_states.get(state, 0) + 1
            return {
                "schema_version": "easyicu.provider_task_cost_accounting/1",
                "task_id": str(task_id),
                "provider_reported": {
                    "n_calls": reported_calls,
                    "prompt_tokens": reported_prompt,
                    "completion_tokens": reported_completion,
                    "total_tokens": reported_total,
                    "estimated_cost_usd": round(reported_cost, 12),
                },
                "usage_unknown": {
                    "n_calls": unknown_calls,
                    "accounted_tokens": unknown_accounted_tokens,
                    "accounted_estimated_cost_usd": round(
                        unknown_accounted_cost,
                        12,
                    ),
                    "states": dict(sorted(unknown_states.items())),
                },
                "conservative_upper_bound": {
                    "n_calls": len(calls),
                    "total_tokens": accounted_tokens,
                    "estimated_cost_usd": round(accounted_cost, 12),
                    "source": "durable_provider_hard_stop_ledger",
                },
            }


class TaskProviderHardStop:
    """One task-scoped view over the shared durable batch ledger."""

    def __init__(self, ledger: ProviderHardStopLedger, task_id: str) -> None:
        self.ledger = ledger
        self.task_id = str(task_id)

    def assert_active(self) -> float:
        return self.ledger.assert_task_active(self.task_id)

    def accounting_summary(self) -> Dict[str, object]:
        """Return the durable accounting view for this task."""

        return self.ledger.task_accounting_summary(self.task_id)

    def cap_timeout(self, requested_seconds: float) -> float:
        remaining = self.assert_active()
        return max(0.001, min(float(requested_seconds), remaining))

    def finish(
        self,
        *,
        score: Optional[Mapping[str, object]] = None,
        error: Optional[str] = None,
    ) -> None:
        self.ledger.finish_task(self.task_id, score=score, error=error)


class _ActiveHardStopCall:
    def __init__(
        self,
        *,
        task: TaskProviderHardStop,
        role: Optional[str],
        model: str,
        messages: Sequence[Any],
        max_tokens: int,
    ) -> None:
        self.task = task
        self.role = role
        self.model = str(model)
        self.messages = tuple(messages)
        self.max_tokens = int(max_tokens)
        self.attempt_ids: list[str] = []
        self._active_attempt_id: Optional[str] = None

    def reserve_attempt(self) -> float:
        prior = self._active_attempt_id
        # The ledger closes ``prior`` durably before it considers a new
        # reservation. Clear the local pointer first so a denied retry is not
        # later finalized a second time by ``fail()``.
        self._active_attempt_id = None
        attempt_id = self.task.ledger.reserve_transport_attempt(
            task_id=self.task.task_id,
            role=self.role,
            model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
            prior_attempt_id=prior,
        )
        self.attempt_ids.append(attempt_id)
        self._active_attempt_id = attempt_id
        return self.task.assert_active()

    def complete(self, usage: Optional[Mapping[str, object]]) -> None:
        if self._active_attempt_id is None:
            raise ProviderHardStopLedgerError(
                "Reviewed Provider transport did not reserve a hard-stop attempt"
            )
        self.task.ledger.finish_transport_attempt(
            task_id=self.task.task_id,
            attempt_id=self._active_attempt_id,
            usage=usage,
            error_type=None,
        )
        self._active_attempt_id = None

    def fail(self, error_type: str) -> None:
        if self._active_attempt_id is None:
            return
        self.task.ledger.finish_transport_attempt(
            task_id=self.task.task_id,
            attempt_id=self._active_attempt_id,
            usage=None,
            error_type=error_type,
        )
        self._active_attempt_id = None


_ACTIVE_HARD_STOP_CALL: ContextVar[Optional[_ActiveHardStopCall]] = ContextVar(
    "easyicu_active_provider_hard_stop_call",
    default=None,
)


@contextmanager
def provider_hard_stop_call_scope(
    *,
    task: TaskProviderHardStop,
    role: Optional[str],
    model: str,
    messages: Sequence[Any],
    max_tokens: int,
) -> Iterator[_ActiveHardStopCall]:
    """Activate task/batch accounting around one logical Provider call."""

    task.assert_active()
    state = _ActiveHardStopCall(
        task=task,
        role=role,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    token = _ACTIVE_HARD_STOP_CALL.set(state)
    try:
        yield state
    finally:
        _ACTIVE_HARD_STOP_CALL.reset(token)


def consume_active_provider_hard_stop_attempt() -> Optional[float]:
    """Reserve the active run/batch attempt, if a hard-stop scope is active."""

    state = _ACTIVE_HARD_STOP_CALL.get()
    if state is not None:
        return state.reserve_attempt()
    return None


__all__ = [
    "PROVIDER_HARD_STOP_SCHEMA",
    "PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR",
    "PROVIDER_PROMPT_OVERHEAD_TOKEN_RESERVATION",
    "ProviderHardStopError",
    "ProviderHardStopExceeded",
    "ProviderHardStopLedger",
    "ProviderHardStopLedgerError",
    "ProviderHardStopLimits",
    "TaskProviderHardStop",
    "consume_active_provider_hard_stop_attempt",
    "load_provider_hard_stop_ledger",
    "provider_hard_stop_call_scope",
]
