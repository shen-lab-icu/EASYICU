"""Fail-fast single-process lease for the current in-memory Web lifecycle."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO, Optional


class UnsupportedWebDeployment(RuntimeError):
    pass


_HANDLE: Optional[IO[str]] = None
_DEPTH = 0


def acquire_single_process_lease(path: Optional[Path] = None) -> None:
    """Reject a second worker until session/job ownership is externalized."""

    global _HANDLE, _DEPTH
    if _HANDLE is not None:
        _DEPTH += 1
        return
    selected = path or (Path.home() / ".easyicu" / "webserver-single-process.lock")
    selected.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    handle = selected.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise UnsupportedWebDeployment(
            "EasyICU Web currently requires exactly one server worker"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    os.fchmod(handle.fileno(), 0o600)
    _HANDLE = handle
    _DEPTH = 1


def release_single_process_lease() -> None:
    global _HANDLE, _DEPTH
    if _HANDLE is None:
        return
    _DEPTH -= 1
    if _DEPTH > 0:
        return
    fcntl.flock(_HANDLE.fileno(), fcntl.LOCK_UN)
    _HANDLE.close()
    _HANDLE = None
    _DEPTH = 0
