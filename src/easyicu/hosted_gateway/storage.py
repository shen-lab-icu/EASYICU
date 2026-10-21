"""Private files and serialized, atomic administrator updates."""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .config import https_origin


def account_name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value):
        raise ValueError("names must use letters, digits, '_' or '-' (1 to 64 characters)")
    return value


def private_directory(path: Path) -> None:
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("state directories must be administrator-owned directories with mode 0700")


def read_private(path: Path, *, limit: int = 1024 * 1024) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.geteuid()):
            raise ValueError("sensitive files must be administrator-owned regular files with mode 0600")
        result = handle.read(limit + 1)
    if len(result) > limit:
        raise ValueError("private file exceeds its size limit")
    return result


def read_secret(path: Path) -> bytes:
    value = read_private(path, limit=4096)
    if len(value) < 32:
        raise ValueError("gateway secret must contain at least 32 bytes")
    return value


def socket_identity(path: Path) -> tuple:
    """Detect directory case aliases even while an instance socket is stopped."""
    path = path.resolve()
    try:
        info = path.parent.stat()
    except FileNotFoundError:
        # An unavailable instance still has an account; routing reports 503.
        return ("path", str(path))
    return ("directory", info.st_dev, info.st_ino, path.name)


def read_users(path: Path) -> dict:
    users = json.loads(read_private(path))
    if not isinstance(users, dict):
        raise ValueError("users file must be an object")
    sockets = set()
    for name, user in users.items():
        account_name(name)
        if not isinstance(user, dict):
            raise ValueError("invalid user record")
        socket = Path(user["socket"])
        identity = socket_identity(socket)
        if not socket.is_absolute() or identity in sockets:
            raise ValueError("each user must have a distinct absolute instance socket")
        sockets.add(identity)
        password = user["password"]
        if (password.get("algorithm") != "scrypt-n32768-r8-p3"
                or not re.fullmatch("[0-9a-f]{32}", password["salt"])
                or not re.fullmatch("[0-9a-f]{64}", password["hash"])
                or not isinstance(user["version"], str) or not user["version"]):
            raise ValueError("invalid password or session version")
    return users


def write_users(path: Path, users: dict) -> None:
    private_directory(path.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".users-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write((json.dumps(users, ensure_ascii=False, indent=2) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def admin_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_directory(path.parent)
    descriptor = os.open(path.with_suffix(path.suffix + ".lock"),
                         os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "rb+") as handle:
        info = os.fstat(handle.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.geteuid()):
            raise ValueError("admin lock must be private and administrator-owned")
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def initialize_secret(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_directory(path.parent)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        read_secret(path)
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(secrets.token_bytes(32))
        handle.flush()
        os.fsync(handle.fileno())


def instance_socket(users_root: Path, name: str, public_origin: str) -> Path:
    """Read only b1's provisioning contract, never model keys or instance data."""
    instance = users_root.expanduser().absolute() / account_name(name)
    private_directory(instance)
    private_directory(instance / "run")
    # b1 stores JSON-encoded values in this administrator-owned file.
    selected = {}
    for line in read_private(instance / "instance.env").decode("utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"EASYICU_PUBLIC_ORIGIN", "EASYICU_HOSTED", "EASYICU_WEB_TRUST_PROXY"}:
            selected[key] = json.loads(value)
    if (https_origin(selected.get("EASYICU_PUBLIC_ORIGIN", "")) != public_origin
            or selected.get("EASYICU_HOSTED") != "1"
            or selected.get("EASYICU_WEB_TRUST_PROXY") != "1"):
        raise ValueError("instance must be hosted and trust this gateway's public origin")
    socket = instance / "run/web.sock"
    if socket.is_symlink():
        raise ValueError("instance socket cannot be a symlink")
    return socket.resolve()  # The instance can be stopped during account creation.
