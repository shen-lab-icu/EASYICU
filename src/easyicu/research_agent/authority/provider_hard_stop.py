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

try:  # pragma: no branch - selected once per platform
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]

try:  # pragma: no branch - selected once per platform
    import msvcrt
except ImportError:  # pragma: no cover - unavailable on POSIX
    msvcrt = None  # type: ignore[assignment]


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
_LEDGER_PROCESS_LOCK = Lock()


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


def validate_provider_transport_reservation_capacity(
    limits: ProviderHardStopLimits,
) -> None:
    """Reject ceilings that cannot fund even one minimum Provider attempt.

    The ledger intentionally permits tiny ceilings so unit tests and direct
    diagnostic callers can exercise denial paths.  Production launchers call
    this preflight before starting a batch, preventing a configuration that is
    guaranteed to fail only after planning and analysis have already run.
    """

    minimum_tokens = (
        PROVIDER_PROMPT_OVERHEAD_TOKEN_RESERVATION
        + PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR
    )
    if limits.max_total_tokens_per_run < minimum_tokens:
        raise ValueError(
            "max_total_tokens_per_run cannot fund one minimum Provider "
            f"transport reservation: require at least {minimum_tokens}, got "
            f"{limits.max_total_tokens_per_run}"
        )
    if limits.max_total_tokens_per_batch < minimum_tokens:
        raise ValueError(
            "max_total_tokens_per_batch cannot fund one minimum Provider "
            f"transport reservation: require at least {minimum_tokens}, got "
            f"{limits.max_total_tokens_per_batch}"
        )


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


@contextmanager
def _exclusive_ledger_file_lock(path: Path) -> Iterator[None]:
    """Serialize ledger transactions using a stable, non-symlink sidecar."""

    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise ProviderHardStopLedgerError(
            "Provider hard-stop ledger parent is unavailable"
        ) from exc
    try:
        parent_info = parent.lstat()
    except OSError as exc:
        raise ProviderHardStopLedgerError(
            "Provider hard-stop ledger parent is unavailable"
        ) from exc
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise ProviderHardStopLedgerError(
            "Provider hard-stop ledger parent must be a real directory"
        )
    lock_path = parent / f".{path.name}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    acquired = False
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        opened = os.fstat(descriptor)
        current = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise ProviderHardStopLedgerError(
                "Provider hard-stop ledger lock must be a stable regular file"
            )
        try:
            os.fchmod(descriptor, 0o600)
        except (AttributeError, OSError):  # Windows has no reliable mode bits.
            pass
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - exercised on Windows
            # Windows permits locking a byte beyond EOF. Lock first, then seed
            # an empty sidecar so two first-openers cannot race by writing the
            # byte each other is about to lock.
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:  # pragma: no cover - supported platforms provide one backend
            raise ProviderHardStopLedgerError(
                "No cross-process Provider hard-stop ledger lock is available"
            )
        acquired = True
        if msvcrt is not None and fcntl is None and os.fstat(descriptor).st_size == 0:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        current = lock_path.lstat()
        if (
            stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise ProviderHardStopLedgerError(
                "Provider hard-stop ledger lock changed during acquisition"
            )
        yield
    except ProviderHardStopLedgerError:
        raise
    except OSError as exc:
        raise ProviderHardStopLedgerError(
            "Could not lock Provider hard-stop ledger"
        ) from exc
    finally:
        if descriptor is not None:
            if acquired:
                try:
                    if fcntl is not None:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    elif msvcrt is not None:  # pragma: no cover - Windows
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            os.close(descriptor)


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    """Publish one complete owner-only ledger and fsync its directory."""

    parent = path.parent
    parent_info = parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise ProviderHardStopLedgerError(
            "Provider hard-stop ledger parent must be a real directory"
        )
    if path.is_symlink() or (path.exists() and not path.is_file()):
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


def _prompt_payload_bytes(
    messages: Sequence[Any], *, additional_payload_bytes: int = 0
) -> int:
    """The caller-visible prompt bytes this attempt is about to transmit."""

    return max(0, int(additional_payload_bytes)) + sum(
        len(str(getattr(message, "content", "") or "").encode("utf-8"))
        for message in messages
    )


def _prompt_token_reservation(
    messages: Sequence[Any], *, additional_payload_bytes: int = 0
) -> int:
    """Conservative bound including provider-side prompt framing.

    Message bytes alone bound tokens produced from caller-visible content, but
    OpenAI-compatible gateways may prepend model instructions that are absent
    from the request payload while still reporting them as prompt usage.  Luna
    on the local gateway has demonstrated this behaviour.  Keep a fixed,
    provider-independent allowance so a normal successful response is not
    misclassified as a budget overflow; successful calls release the unused
    reservation back to their provider-reported usage.
    """

    prompt_bytes = max(0, int(additional_payload_bytes)) + sum(
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
        self._task_ids = normalized_ids
        self._batch_id = str(batch_id).strip() if batch_id else None
        self._declaration_sha256 = (
            str(declaration_sha256).strip() if declaration_sha256 else None
        )
        self._lock = Lock()
        # Live monotonic clocks exist only while Provider-backed execution is
        # active.  Human-review pauses persist their cumulative active seconds
        # in the ledger and deliberately hold no live clock, so time spent by a
        # reviewer cannot consume the execution stop-loss.
        self._task_started_monotonic: Dict[str, float] = {}
        self._task_active_started_at: Dict[str, str] = {}
        now = _utc_now()
        initial_payload: Dict[str, object] = {
            "schema_version": PROVIDER_HARD_STOP_SCHEMA,
            "batch_id": self._batch_id,
            "declaration_sha256": self._declaration_sha256,
            "created_at": now,
            "updated_at": now,
            "limits": limits.canonical_payload(),
            "task_order": list(normalized_ids),
            "tasks": [
                {
                    "task_id": task_id,
                    "status": "pending",
                    "started_at": None,
                    "active_started_at": None,
                    "finished_at": None,
                    "elapsed_seconds": 0.0,
                    "paused_at": None,
                    "paused_active_started_at": None,
                    "review_checkpoint_at": None,
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
        self._payload = initial_payload
        with self._lock, _LEDGER_PROCESS_LOCK, _exclusive_ledger_file_lock(self.path):
            if self.path.is_symlink() or self.path.exists():
                if not resume_existing:
                    raise FileExistsError(self.path)
                self._reload_locked()
            else:
                self._persist_locked()

    def _validate_invocation_locked(self, payload: Mapping[str, object]) -> None:
        if (
            payload.get("limits") != self.limits.canonical_payload()
            or payload.get("task_order") != list(self._task_ids)
            or payload.get("batch_id") != self._batch_id
            or payload.get("declaration_sha256") != self._declaration_sha256
        ):
            raise ProviderHardStopLedgerError(
                "Existing Provider hard-stop ledger differs from this invocation"
            )

    def _synchronize_live_clocks_locked(self) -> None:
        running_ids: set[str] = set()
        now_monotonic = time.monotonic()
        now_utc = datetime.now(timezone.utc)
        for task in self._tasks_locked():
            task_id = str(task.get("task_id") or "")
            if task.get("status") != "running":
                continue
            running_ids.add(task_id)
            active_anchor = str(task.get("active_started_at") or task.get("started_at"))
            if (
                task_id in self._task_started_monotonic
                and self._task_active_started_at.get(task_id) == active_anchor
            ):
                continue
            try:
                prior_elapsed = float(task.get("elapsed_seconds") or 0.0)
                active_started_at = datetime.fromisoformat(active_anchor)
                elapsed = prior_elapsed + max(
                    0.0, (now_utc - active_started_at).total_seconds()
                )
                if not math.isfinite(elapsed) or elapsed < 0.0:
                    raise ValueError("invalid elapsed wall clock")
            except (TypeError, ValueError):
                elapsed = self.limits.max_wall_clock_seconds_per_task
            self._task_started_monotonic[task_id] = now_monotonic - elapsed
            self._task_active_started_at[task_id] = active_anchor
        for task_id in set(self._task_started_monotonic) - running_ids:
            self._task_started_monotonic.pop(task_id, None)
            self._task_active_started_at.pop(task_id, None)

    def _reload_locked(self) -> None:
        loaded = load_provider_hard_stop_ledger(self.path)
        self._validate_invocation_locked(loaded)
        loaded.pop("sha256", None)
        self._payload = loaded
        self._synchronize_live_clocks_locked()

    @contextmanager
    def _durable_transaction(self) -> Iterator[None]:
        """Lock, digest-check, and reload the durable ledger before use."""

        with self._lock, _LEDGER_PROCESS_LOCK, _exclusive_ledger_file_lock(self.path):
            self._reload_locked()
            yield

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
        with self._durable_transaction():
            task = self._task_locked(normalized)
            if task.get("status") == "running":
                if normalized not in self._task_started_monotonic:
                    raise ProviderHardStopLedgerError(
                        "Running Provider hard-stop task lost its wall clock"
                    )
                return TaskProviderHardStop(self, normalized)
            status = str(task.get("status") or "")
            if status == "paused":
                # A restarted host may attach to a human-review pause without
                # restarting its active clock or resetting any cumulative use.
                return TaskProviderHardStop(self, normalized)
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
                task["elapsed_seconds"] = prior_elapsed
                task["error"] = None
                task["blocked_by"] = None
                task["score_summary"] = {}
                if not task.get("started_at"):
                    task["started_at"] = _utc_now()
                task["active_started_at"] = _utc_now()
                task["paused_active_started_at"] = None
                task["review_checkpoint_at"] = None
                self._task_started_monotonic[normalized] = (
                    time.monotonic() - prior_elapsed
                )
                self._task_active_started_at[normalized] = str(
                    task["active_started_at"]
                )
                self._persist_locked()
                return TaskProviderHardStop(self, normalized)
            if status != "pending":
                raise ProviderHardStopLedgerError(
                    f"Provider hard-stop task cannot start from {status!r}"
                )
            task["status"] = "running"
            started_at = _utc_now()
            task["started_at"] = started_at
            task["active_started_at"] = started_at
            task["elapsed_seconds"] = 0.0
            task["paused_at"] = None
            task["paused_active_started_at"] = None
            task["review_checkpoint_at"] = None
            self._task_started_monotonic[normalized] = time.monotonic()
            self._task_active_started_at[normalized] = started_at
            self._persist_locked()
        return TaskProviderHardStop(self, normalized)

    def pause_task(self, task_id: str) -> None:
        """Pause one live task without charging human-review wait time."""

        normalized = str(task_id).strip()
        with self._durable_transaction():
            task = self._task_locked(normalized)
            if task.get("status") == "paused":
                return
            if task.get("status") != "running":
                raise ProviderHardStopLedgerError(
                    f"Provider hard-stop task cannot pause from {task.get('status')!r}"
                )
            elapsed = self._elapsed_locked(normalized)
            task["status"] = "paused"
            task["paused_at"] = _utc_now()
            task["paused_active_started_at"] = task.get("active_started_at")
            # A completed resume may reach a later review pause. That pause
            # establishes a new checkpoint; only a crash while still running
            # is allowed to reuse the prior anchor.
            task["review_checkpoint_at"] = None
            task["elapsed_seconds"] = round(elapsed, 6)
            task["active_started_at"] = None
            self._task_started_monotonic.pop(normalized, None)
            self._task_active_started_at.pop(normalized, None)
            self._persist_locked()

    def reconcile_review_pause(self, task_id: str, *, paused_at: str) -> None:
        """Converge a crash-window running task to its durable review pause.

        The human-review checkpoint is fsynced separately from this ledger. If
        the process dies between those writes, restart must charge activity only
        through the checkpoint timestamp, not through the later restart time.
        """

        normalized = str(task_id).strip()
        with self._durable_transaction():
            task = self._task_locked(normalized)
            try:
                checkpoint_at = datetime.fromisoformat(str(paused_at))
                if checkpoint_at.tzinfo is None:
                    raise ValueError("review pause timestamp must be timezone-aware")
                recorded_raw = task.get("review_checkpoint_at")
                recorded_at = (
                    datetime.fromisoformat(str(recorded_raw))
                    if recorded_raw is not None
                    else None
                )
                if recorded_at is not None and recorded_at.tzinfo is None:
                    raise ValueError("review checkpoint anchor must be timezone-aware")
            except (TypeError, ValueError) as exc:
                raise ProviderHardStopLedgerError(
                    "Provider hard-stop review pause timestamp is invalid"
                ) from exc

            status = task.get("status")
            if recorded_at is not None:
                if recorded_at != checkpoint_at:
                    raise ProviderHardStopLedgerError(
                        "Provider hard-stop review checkpoint changed after pause"
                    )
                if status == "paused":
                    return
            if status not in {"running", "paused"}:
                raise ProviderHardStopLedgerError(
                    "Provider hard-stop review pause cannot reconcile from "
                    f"{status!r}"
                )
            try:
                prior_elapsed = float(task.get("elapsed_seconds") or 0.0)
                if status == "running" and recorded_at is not None:
                    # A prior resume reopened the active clock and crashed before
                    # committing the still-pending decision. The exact persisted
                    # anchor is the only timestamp authorized to rewind it.
                    elapsed = prior_elapsed
                elif status == "running":
                    active_started_at = datetime.fromisoformat(
                        str(task.get("active_started_at") or task["started_at"])
                    )
                    if active_started_at.tzinfo is None:
                        raise ValueError(
                            "review active timestamp must be timezone-aware"
                        )
                    active_delta = (checkpoint_at - active_started_at).total_seconds()
                    if active_delta < 0.0:
                        raise ValueError("review checkpoint predates active execution")
                    elapsed = prior_elapsed + active_delta
                else:
                    actual_paused_at = datetime.fromisoformat(str(task["paused_at"]))
                    active_started_at = datetime.fromisoformat(
                        str(
                            task.get("paused_active_started_at")
                            or task.get("started_at")
                        )
                    )
                    if (
                        actual_paused_at.tzinfo is None
                        or active_started_at.tzinfo is None
                        or not active_started_at <= checkpoint_at <= actual_paused_at
                    ):
                        raise ValueError(
                            "review checkpoint is outside the paused active segment"
                        )
                    rewind = (actual_paused_at - checkpoint_at).total_seconds()
                    elapsed = prior_elapsed - rewind
                if not math.isfinite(elapsed) or elapsed < 0.0:
                    raise ValueError("invalid reconciled elapsed wall clock")
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderHardStopLedgerError(
                    "Provider hard-stop review pause timestamp is invalid"
                ) from exc
            task["status"] = "paused"
            task["paused_at"] = checkpoint_at.isoformat()
            task["review_checkpoint_at"] = checkpoint_at.isoformat()
            task["elapsed_seconds"] = round(elapsed, 6)
            task["active_started_at"] = None
            task["paused_active_started_at"] = None
            self._task_started_monotonic.pop(normalized, None)
            self._task_active_started_at.pop(normalized, None)
            self._persist_locked()

    def resume_task(self, task_id: str) -> None:
        """Resume a paused task under its cumulative active-time ceiling."""

        normalized = str(task_id).strip()
        with self._durable_transaction():
            task = self._task_locked(normalized)
            if task.get("status") == "running":
                return
            if task.get("status") != "paused":
                raise ProviderHardStopLedgerError(
                    f"Provider hard-stop task cannot resume from {task.get('status')!r}"
                )
            try:
                prior_elapsed = float(task.get("elapsed_seconds") or 0.0)
            except (TypeError, ValueError) as exc:
                raise ProviderHardStopLedgerError(
                    "Paused Provider hard-stop task elapsed time is invalid"
                ) from exc
            if not math.isfinite(prior_elapsed) or prior_elapsed < 0.0:
                raise ProviderHardStopLedgerError(
                    "Paused Provider hard-stop task elapsed time is invalid"
                )
            if prior_elapsed >= self.limits.max_wall_clock_seconds_per_task:
                task["status"] = "budget_exhausted"
                task["finished_at"] = _utc_now()
                task["active_started_at"] = None
                task["error"] = "TASK_WALL_CLOCK_EXHAUSTED"
                self._task_started_monotonic.pop(normalized, None)
                self._task_active_started_at.pop(normalized, None)
                self._persist_locked()
                raise ProviderHardStopExceeded(
                    code="TASK_WALL_CLOCK_EXHAUSTED",
                    detail=(
                        f"task {task_id!r} active execution reached "
                        f"{prior_elapsed:.3f}s; limit="
                        f"{self.limits.max_wall_clock_seconds_per_task:.3f}s"
                    ),
                )
            task["status"] = "running"
            task["paused_at"] = None
            task["elapsed_seconds"] = prior_elapsed
            task["active_started_at"] = _utc_now()
            task["paused_active_started_at"] = None
            self._task_started_monotonic[normalized] = (
                time.monotonic() - prior_elapsed
            )
            self._task_active_started_at[normalized] = str(task["active_started_at"])
            self._persist_locked()

    def mark_task_blocked(self, task_id: str, *, blocked_by: str) -> None:
        with self._durable_transaction():
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

    def _assert_task_active_locked(self, task_id: str) -> float:
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
            task["active_started_at"] = None
            task["error"] = "TASK_WALL_CLOCK_EXHAUSTED"
            self._task_started_monotonic.pop(str(task_id), None)
            self._task_active_started_at.pop(str(task_id), None)
            self._persist_locked()
            raise ProviderHardStopExceeded(
                code="TASK_WALL_CLOCK_EXHAUSTED",
                detail=(
                    f"task {task_id!r} elapsed {elapsed:.3f}s; "
                    f"limit={self.limits.max_wall_clock_seconds_per_task:.3f}s"
                ),
            )
        return remaining

    def assert_task_active(self, task_id: str) -> float:
        with self._durable_transaction():
            return self._assert_task_active_locked(task_id)

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
        additional_prompt_payload_bytes: int = 0,
        logical_call_id: Optional[str] = None,
    ) -> str:
        """Reserve one maximum-cost transport attempt before network delivery."""

        if logical_call_id is not None and (
            not isinstance(logical_call_id, str) or not logical_call_id.strip()
        ):
            raise ValueError("logical_call_id must be a non-empty string or None")

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
        prompt_reserve = _prompt_token_reservation(
            messages,
            additional_payload_bytes=additional_prompt_payload_bytes,
        )
        prompt_payload_bytes = _prompt_payload_bytes(
            messages,
            additional_payload_bytes=additional_prompt_payload_bytes,
        )
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
        with self._durable_transaction():
            self._assert_task_active_locked(task_id)
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
                    "logical_call_id": (
                        str(logical_call_id) if logical_call_id else None
                    ),
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
        with self._durable_transaction():
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
            def reported_tokens(name: str) -> Optional[int]:
                if not isinstance(usage, Mapping) or usage.get(name) is None:
                    return None
                value = usage[name]
                if isinstance(value, bool) or not isinstance(value, int):
                    return None
                return value if value >= 0 else None

            reported_prompt = reported_tokens("prompt_tokens")
            reported_completion = reported_tokens("completion_tokens")
            reported_total_value = reported_tokens("total_tokens")
            component_usage_complete = (
                reported_prompt is not None
                and reported_completion is not None
                and reported_prompt + reported_completion > 0
            )
            usage_consistent = not (
                reported_total_value is not None
                and (
                    any(
                        component is not None and component > reported_total_value
                        for component in (reported_prompt, reported_completion)
                    )
                    or (
                        component_usage_complete
                        and reported_prompt + reported_completion
                        > reported_total_value
                    )
                )
            )
            has_reported_token_usage = usage_consistent and (
                component_usage_complete
                or bool(reported_total_value and reported_total_value > 0)
            )
            if error_type is not None or not has_reported_token_usage:
                call["state"] = (
                    "failed_usage_unknown"
                    if error_type is not None
                    else "completed_usage_unreported"
                )
                # Unknown usage cannot release either hold. The prompt byte
                # bound and completion floor are what made the decision
                # pre-transport; only a provider receipt proves a lower charge.
                self._persist_locked()
                return
            prompt_tokens = reported_prompt or 0
            completion_tokens = reported_completion or 0
            component_total = prompt_tokens + completion_tokens
            reported_total = max(
                component_total,
                reported_total_value or 0,
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
        with self._durable_transaction():
            task = self._task_locked(task_id)
            if task.get("status") == "completed":
                return
            # Check the wall clock before granting a completed state.
            if error is None:
                self._assert_task_active_locked(task_id)
            status = str(task.get("status") or "")
            if status not in {"running", "paused"}:
                if task.get("status") == "budget_exhausted":
                    return
                raise ProviderHardStopLedgerError(
                    f"Provider hard-stop task cannot finish from {task.get('status')!r}"
                )
            if status == "paused":
                if error is None:
                    raise ProviderHardStopLedgerError(
                        "A paused Provider hard-stop task cannot complete successfully"
                    )
                try:
                    elapsed = float(task.get("elapsed_seconds") or 0.0)
                except (TypeError, ValueError) as exc:
                    raise ProviderHardStopLedgerError(
                        "Paused Provider hard-stop task elapsed time is invalid"
                    ) from exc
            else:
                elapsed = self._elapsed_locked(task_id)
            task["status"] = "failed" if error is not None else "completed"
            task["finished_at"] = _utc_now()
            task["elapsed_seconds"] = round(elapsed, 6)
            task["active_started_at"] = None
            task["error"] = _safe_error_summary(error)
            task["score_summary"] = _safe_score_summary(score)
            self._task_started_monotonic.pop(str(task_id), None)
            self._task_active_started_at.pop(str(task_id), None)
            self._persist_locked()

    def snapshot(self) -> Dict[str, object]:
        with self._durable_transaction():
            # Canonical JSON round-trip returns an isolated plain-data snapshot.
            payload = dict(self._payload)
            payload["sha256"] = _payload_digest(payload)
            return json.loads(json.dumps(payload, allow_nan=False))

    def task_accounting_summary(self, task_id: str) -> Dict[str, object]:
        """Return actual, unknown, and conservative accounting for one task."""

        with self._durable_transaction():
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

    def pause(self) -> None:
        """Pause active execution while a human decision is outstanding."""

        self.ledger.pause_task(self.task_id)

    def resume(self) -> None:
        """Resume execution without charging the intervening human wait."""

        self.ledger.resume_task(self.task_id)

    def reconcile_review_pause(self, *, paused_at: str) -> None:
        """Recover the exact active-time boundary from a durable checkpoint."""

        self.ledger.reconcile_review_pause(self.task_id, paused_at=paused_at)

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
        additional_prompt_payload_bytes: int = 0,
        logical_call_id: Optional[str] = None,
    ) -> None:
        self.task = task
        self.role = role
        self.model = str(model)
        self.messages = tuple(messages)
        self.max_tokens = int(max_tokens)
        self.additional_prompt_payload_bytes = max(
            0, int(additional_prompt_payload_bytes)
        )
        self.logical_call_id = logical_call_id
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
            additional_prompt_payload_bytes=self.additional_prompt_payload_bytes,
            prior_attempt_id=prior,
            logical_call_id=self.logical_call_id,
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
    additional_prompt_payload_bytes: int = 0,
    logical_call_id: Optional[str] = None,
) -> Iterator[_ActiveHardStopCall]:
    """Activate task/batch accounting around one logical Provider call."""

    task.assert_active()
    state = _ActiveHardStopCall(
        task=task,
        role=role,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        additional_prompt_payload_bytes=additional_prompt_payload_bytes,
        logical_call_id=logical_call_id,
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
    "validate_provider_transport_reservation_capacity",
]
