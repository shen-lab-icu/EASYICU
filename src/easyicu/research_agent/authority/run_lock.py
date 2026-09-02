"""Cross-process exclusive writer lock for one research-agent run.

The lock lives outside the run directory so it can be acquired before the
pipeline creates or mutates any run-owned file.  Lock files are deliberately
kept after release: unlinking a flock file can split contenders across two
inodes and allow concurrent writers.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import socket
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

try:
    import fcntl
except ImportError as exc:  # pragma: no cover - EasyICU targets macOS/Linux.
    raise RuntimeError(
        "Research-agent run locking requires POSIX fcntl support."
    ) from exc


P = ParamSpec("P")
R = TypeVar("R")

_ACTIVE_RUN_ID: ContextVar[str | None] = ContextVar(
    "easyicu_active_locked_run_id", default=None
)


class RunExecutionLockError(RuntimeError):
    """Raised when another process already owns the requested run writer lock."""


def _new_run_id() -> str:
    return (
        "run_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:6]
    )


def _validated_run_id(run_id: str) -> str:
    candidate = str(run_id).strip()
    if (
        not candidate
        or candidate in {".", ".."}
        or "/" in candidate
        or "\\" in candidate
        or "\x00" in candidate
    ):
        raise ValueError(
            "run_id must be one non-empty path component without '/' or '\\'"
        )
    return candidate


def _canonical_run_dir(*, workdir: Path, run_id: str) -> Path:
    return (Path(workdir).expanduser().resolve() / run_id).resolve()


def _lock_path(*, workdir: Path, run_id: str) -> Path:
    run_dir = _canonical_run_dir(workdir=workdir, run_id=run_id)
    digest = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()
    return Path(workdir).expanduser().resolve() / ".run_locks" / f"{digest}.lock"


@dataclass
class RunExecutionLock:
    """An acquired non-blocking POSIX flock held by an open file descriptor."""

    run_id: str
    run_dir: Path
    path: Path
    _handle: Any
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._released = True

    def __enter__(self) -> RunExecutionLock:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def acquire_run_execution_lock(*, workdir: Path, run_id: str) -> RunExecutionLock:
    """Acquire the sole writer lease for ``run_id`` without waiting.

    A conflict is reported immediately with the last holder receipt.  The
    receipt is diagnostic only; kernel flock ownership is the authority, so a
    stale receipt after a crash never blocks a later run.
    """

    resolved_run_id = _validated_run_id(run_id)

    run_dir = _canonical_run_dir(workdir=workdir, run_id=resolved_run_id)
    path = _lock_path(workdir=workdir, run_id=resolved_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        try:
            handle.seek(0)
            raw_holder = handle.read().strip()
        except OSError:
            raw_holder = ""
        finally:
            handle.close()
        holder = ""
        if raw_holder:
            try:
                payload = json.loads(raw_holder)
                holder = (
                    f" Holder: pid={payload.get('pid', 'unknown')}, "
                    f"host={payload.get('hostname', 'unknown')}, "
                    f"acquired_at={payload.get('acquired_at', 'unknown')}."
                )
            except (TypeError, ValueError):
                holder = " Holder receipt is unreadable."
        raise RunExecutionLockError(
            "Research-agent run is already being written by another process: "
            f"run_id={resolved_run_id!r}, run_dir={run_dir}.{holder} "
            "Wait for that run/resume to finish before retrying."
        ) from exc
    except Exception:
        handle.close()
        raise

    receipt = {
        "schema_version": "easyicu.run_execution_lock.v1",
        "run_id": resolved_run_id,
        "run_dir": str(run_dir),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        handle.seek(0)
        handle.truncate()
        json.dump(receipt, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    except Exception:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        raise
    return RunExecutionLock(
        run_id=resolved_run_id,
        run_dir=run_dir,
        path=path,
        _handle=handle,
    )


def current_locked_run_id() -> str:
    """Return the run id selected and locked by ``exclusive_run_execution``."""

    run_id = _ACTIVE_RUN_ID.get()
    if not run_id:
        raise RuntimeError("No research-agent run execution lock is active.")
    return run_id


def exclusive_run_execution(function: Callable[P, R]) -> Callable[P, R]:
    """Decorate a pipeline ``run`` method with a whole-call writer lease."""

    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = signature.bind_partial(*args, **kwargs)
        pipeline = bound.arguments.get("self")
        if pipeline is None:
            raise RuntimeError("exclusive_run_execution requires a bound method")
        resume_run_id = bound.arguments.get("resume_run_id")
        run_id = _validated_run_id(resume_run_id) if resume_run_id else _new_run_id()
        token = _ACTIVE_RUN_ID.set(run_id)
        try:
            with acquire_run_execution_lock(
                workdir=Path(pipeline.workdir),
                run_id=run_id,
            ):
                from .run_heartbeat import run_heartbeat_scope

                with run_heartbeat_scope(run_id=run_id):
                    return function(*args, **kwargs)
        finally:
            _ACTIVE_RUN_ID.reset(token)

    wrapped.__easyicu_run_execution_locked__ = True
    return wrapped


__all__ = [
    "RunExecutionLock",
    "RunExecutionLockError",
    "acquire_run_execution_lock",
    "current_locked_run_id",
    "exclusive_run_execution",
]
