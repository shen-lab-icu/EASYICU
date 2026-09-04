"""Crash-durable, no-overwrite publication for one immutable artifact file."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from uuid import uuid4


class ImmutablePublicationError(OSError):
    """An immutable artifact could not be staged or durably published."""

    reason_code = "IMMUTABLE_ARTIFACT_PUBLICATION_FAILED"
    owner = "figure2.immutable_file_publication_v1"


def _require_real_parent(path: Path) -> Path:
    parent = path.parent
    try:
        info = parent.lstat()
    except OSError as exc:
        raise ImmutablePublicationError(
            "immutable artifact parent is unavailable"
        ) from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ImmutablePublicationError(
            "immutable artifact parent must be a real directory"
        )
    return parent


def _write_stage(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def publish_immutable_bytes(payload: bytes, destination: Path) -> Path:
    """Publish complete bytes exactly once and make the directory entry durable."""

    if not isinstance(payload, bytes):
        raise TypeError("immutable artifact payload must be bytes")
    target = Path(destination)
    parent = _require_real_parent(target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    staging = parent / f".{target.name}.{uuid4().hex}.stage"
    try:
        try:
            _write_stage(staging, payload)
        except FileExistsError as exc:
            raise ImmutablePublicationError(
                "immutable artifact staging name already exists"
            ) from exc
        if staging.read_bytes() != payload:
            raise ImmutablePublicationError(
                "immutable artifact staging verification failed"
            )
        os.link(staging, target, follow_symlinks=False)
        if target.read_bytes() != payload:
            raise ImmutablePublicationError(
                "immutable artifact publication verification failed"
            )
        staging.unlink()
        parent_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ImmutablePublicationError(
            f"immutable artifact could not be published durably: {exc}"
        ) from exc
    finally:
        staging.unlink(missing_ok=True)
    return target


__all__ = ["ImmutablePublicationError", "publish_immutable_bytes"]
