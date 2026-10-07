"""The host's read of a standard executor's stop record.

``contracts.executor_stop`` owns the record and its vocabulary.  This module
reads the file a failed attempt left in its step's output directory, without
following a link at the directory or at the record and without waiting on a
special file, and hands the bytes to the contract's parser.  A record the
parser refuses names no stop.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Optional

from ..contracts.executor_stop import (
    EXECUTOR_STOP_RECORD_MAX_BYTES,
    EXECUTOR_STOP_RECORD_NAME,
    ExecutorStopRecordError,
    RecordedExecutorStop,
    parse_executor_stop_record,
)

__all__ = ["read_executor_stop_record"]

_NO_FOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NON_BLOCKING = getattr(os, "O_NONBLOCK", 0)


def _read_bounded(fd: int) -> bytes:
    chunks: list[bytes] = []
    remaining = EXECUTOR_STOP_RECORD_MAX_BYTES + 1
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_executor_stop_record(
    out_dir: Path, *, expected_owner: str
) -> tuple[Optional[RecordedExecutorStop], Optional[str]]:
    """The stop a failed attempt recorded, and why a record was refused.

    ``(stop, None)`` for an accepted record, ``(None, rejection)`` for a record
    that names no stop, and ``(None, None)`` when there is no record.
    """

    try:
        directory = os.open(
            out_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _NO_FOLLOW
        )
    except FileNotFoundError:
        return None, None
    except OSError:
        return None, "record_unreadable"
    try:
        try:
            # Non-blocking, so a FIFO at the record's name is refused below
            # instead of holding the host until something writes to it.
            fd = os.open(
                EXECUTOR_STOP_RECORD_NAME,
                os.O_RDONLY | _NO_FOLLOW | _NON_BLOCKING,
                dir_fd=directory,
            )
        except FileNotFoundError:
            return None, None
        except OSError:
            return None, "record_unreadable"
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                return None, "record_unreadable"
            if info.st_size > EXECUTOR_STOP_RECORD_MAX_BYTES:
                return None, "record_too_large"
            payload = _read_bounded(fd)
        finally:
            os.close(fd)
    finally:
        os.close(directory)
    try:
        return parse_executor_stop_record(payload, expected_owner=expected_owner), None
    except ExecutorStopRecordError as exc:
        return None, exc.code

