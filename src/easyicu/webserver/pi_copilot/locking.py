"""Fail-closed cross-process locks for Pi filesystem authorities.

The lock file is coordination metadata only. Scientific/project policy remains
owned by the caller; this module owns the portable exclusive-lock contract and
the stable diagnostic emitted when that contract cannot be acquired.
"""

from __future__ import annotations

import contextlib
import os
import stat
import time
from collections.abc import Iterator
from pathlib import Path

from .contracts import PiCopilotError

try:  # pragma: no cover - platform branch
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # pragma: no cover - platform branch
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]


@contextlib.contextmanager
def exclusive_file_lock(
    path: Path,
    *,
    code: str,
    timeout_seconds: float = 5.0,
) -> Iterator[None]:
    """Acquire an OS-released exclusive lock or fail with an owner code."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise PiCopilotError(
            code,
            "The Pi filesystem authority lock could not be opened.",
            status_code=503,
        ) from exc

    acquired = False
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise PiCopilotError(
                code,
                "The Pi filesystem authority lock is not a regular file.",
                status_code=503,
            )
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while not acquired:
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    if os.fstat(descriptor).st_size == 0:
                        os.write(descriptor, b"0")
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:  # pragma: no cover - unsupported platform
                    raise PiCopilotError(
                        code,
                        "No cross-process file-lock backend is available.",
                        status_code=503,
                    )
                acquired = True
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise PiCopilotError(
                        code,
                        "The Pi filesystem authority lock timed out.",
                        status_code=503,
                    )
                time.sleep(0.02)
        yield
    finally:
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


__all__ = ["exclusive_file_lock"]
