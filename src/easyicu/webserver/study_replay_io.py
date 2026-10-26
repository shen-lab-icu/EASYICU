"""Bounded file snapshots and atomic publication for offline replay bundles."""

from __future__ import annotations

from contextlib import ExitStack
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any

from .study_replay_policy import ReplayExportError, secret_values

MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_BYTES = 32 * 1024 * 1024


def identity(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value
    ):
        raise ReplayExportError("replay_identity_invalid")
    return value


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def relative_parts(relative: str) -> tuple[str, ...]:
    if (
        not isinstance(relative, str)
        or "\\" in relative
        or ":" in relative
        or relative.startswith("/")
        or any(p in {"", ".", ".."} for p in relative.split("/"))
    ):
        raise ReplayExportError("replay_path_forbidden")
    return PurePosixPath(relative).parts


def read_snapshot(root: Path, relative: str, *, optional: bool = False) -> bytes | None:
    """Open each directory without following links; hash and parse these bytes."""
    parts = relative_parts(relative)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        with ExitStack() as stack:
            descriptor = os.open(root, flags | os.O_DIRECTORY)
            stack.callback(os.close, descriptor)
            for part in parts[:-1]:
                descriptor = os.open(part, flags | os.O_DIRECTORY, dir_fd=descriptor)
                stack.callback(os.close, descriptor)
            descriptor = os.open(parts[-1], flags, dir_fd=descriptor)
            stack.callback(os.close, descriptor)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ReplayExportError("replay_path_forbidden")
            if info.st_size > MAX_FILE_BYTES:
                raise ReplayExportError("replay_size_limit")
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                raw = handle.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES:
                raise ReplayExportError("replay_size_limit")
            return raw
    except FileNotFoundError:
        if optional:
            return None
        raise ReplayExportError("replay_file_unavailable") from None
    except OSError as exc:
        code = (
            "replay_path_forbidden"
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}
            else "replay_file_unavailable"
        )
        raise ReplayExportError(code) from None


def parse_json(raw: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("nonfinite number")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ReplayExportError("replay_record_invalid") from None


class ReplayFiles:
    """Keep each input snapshot once, with private-path-free input receipts."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.inputs: dict[str, str] = {}
        self.secrets: set[str] = set()

    def json(self, name: str, *, optional: bool = False) -> Any:
        raw = read_snapshot(self.root, name, optional=optional)
        if raw is None:
            return None
        self.inputs[name] = digest(raw)
        value = parse_json(raw)
        self.secrets.update(secret_values(value))
        return value


def write_bundle(destination: Path, files: dict[str, bytes]) -> None:
    """Publish a fully checked bundle; never merge with an existing directory."""
    destination = Path(destination)
    if sum(map(len, files.values())) > MAX_BUNDLE_BYTES:
        raise ReplayExportError("replay_size_limit")
    for name in files:
        relative_parts(name)
    staging = None
    try:
        if destination.exists() or destination.is_symlink():
            raise ReplayExportError("replay_destination_exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".replay-", dir=destination.parent))
        for name, raw in sorted(files.items()):
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        staging.rename(destination)
    except OSError:
        raise ReplayExportError("replay_write_failed") from None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
