"""Dump every thread's Python stack on SIGUSR1, without a debugger.

Hosted servers block ptrace (Yama scope 1), so a stalled web server could not
be sampled with py-spy or gdb (pilot1, 2026-11-16). After startup,
``kill -USR1 <pid>`` writes all thread stacks to the server's stderr (the
web log) and the server keeps running. No-op where the signal does not exist.
"""

from __future__ import annotations

import faulthandler
import signal
import sys
from typing import IO, Optional


def install(file: Optional[IO[str]] = None) -> bool:
    """Register the dump; return whether the platform supports it."""
    signum = getattr(signal, "SIGUSR1", None)
    if signum is None:
        return False
    faulthandler.register(signum, file=file or sys.stderr, all_threads=True, chain=False)
    return True


__all__ = ["install"]
