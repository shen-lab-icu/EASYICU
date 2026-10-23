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

from easyicu.runtime.platform_compat import PlatformCapabilityError, lock_exclusive, open_nofollow, unlock_file


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
        descriptor = open_nofollow(lock_path, flags, 0o600)
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
                lock_exclusive(descriptor, blocking=False)
                acquired = True
            except PlatformCapabilityError as exc:
                raise PiCopilotError(code, "No cross-process file-lock backend is available.", status_code=503) from exc
            except BlockingIOError:
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
                unlock_file(descriptor)
            except OSError:
                pass
        os.close(descriptor)


__all__ = ["exclusive_file_lock"]
