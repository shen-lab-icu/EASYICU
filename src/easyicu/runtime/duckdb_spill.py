"""Own private DuckDB spill storage, including normal-exit and stale cleanup.

Each independent engine gets a PID/token directory, even within one process.
An override names the parent of our owned root, never a shared block directory.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import re
import shutil
import time
from typing import TYPE_CHECKING, Iterator
from uuid import uuid4
import weakref

if TYPE_CHECKING:
    import duckdb

from easyicu.state_paths import state_root

_OWNED_ROOT = "duckdb-spill"
_PRIVATE_NAME = re.compile(r"^pid-([1-9][0-9]*)-[0-9a-f]{32}$")
_directories: dict[Path, tuple[int, weakref.ReferenceType | None]] = {}


class DuckDBSpillError(RuntimeError):
    """The required private spill path could not be created or configured."""

    def __init__(self, path: Path):
        self.path = path
        super().__init__(f"Cannot configure private DuckDB spill directory: {path}")


def _pid_alive(pid: int) -> bool:
    # Windows' os.kill(pid, 0) can terminate a process. psutil is optional;
    # without it, conservatively retain stale directories on that platform.
    if os.name == "nt":
        try:
            import psutil
        except ImportError:
            return True
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
    except OSError as error:
        return error.errno != errno.ESRCH
    return True


def _reap_stale(root: Path) -> None:
    cutoff = time.time() - 3600
    for path in root.iterdir():
        match = _PRIVATE_NAME.fullmatch(path.name)
        if match is None or path.is_symlink():
            continue
        try:
            if (
                path.is_dir()
                and path.stat().st_mtime < cutoff
                and not _pid_alive(int(match[1]))
            ):
                shutil.rmtree(path)
        except OSError:
            # A concurrent reaper or inaccessible old directory cannot justify
            # falling back to cwd, nor prevent creating a fresh private path.
            continue


def _remove(path: Path, pid: int) -> None:
    if os.getpid() != pid:  # forked children never delete their parent's spill
        return
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError:
        return  # retained for an exit retry, or the next stale sweep
    _directories.pop(path, None)


def _cleanup() -> None:
    for path, (pid, reference) in list(_directories.items()):
        if pid != os.getpid():
            continue
        connection = reference() if reference else None
        if connection is not None:
            connection.close()
        _remove(path, pid)


atexit.register(_cleanup)


def _new_directory() -> Path:
    override = os.environ.get("EASYICU_DUCKDB_TEMP_DIR", "").strip()
    parent = Path(override).expanduser() if override else state_root() / "cache"
    root = parent.absolute() / _OWNED_ROOT
    path = root / f"pid-{os.getpid()}-{uuid4().hex}"
    try:
        if root.is_symlink():
            raise OSError("the owned spill root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        _reap_stale(root)
        path.mkdir(mode=0o700)
    except OSError as error:
        raise DuckDBSpillError(path) from error
    _directories[path] = (os.getpid(), None)
    return path


def connect(*args, **kwargs) -> duckdb.DuckDBPyConnection:
    """Create an engine with private spill storage; never use a cwd fallback."""
    import duckdb

    path = _new_directory()
    config = dict(kwargs.pop("config", {}) or {})
    config["temp_directory"] = str(path)
    maximum = os.environ.get("EASYICU_DUCKDB_MAX_TEMP_SIZE")
    if maximum:
        config["max_temp_directory_size"] = maximum
    try:
        connection = duckdb.connect(*args, config=config, **kwargs)
    except Exception as error:
        _remove(path, os.getpid())
        raise DuckDBSpillError(path) from error
    pid = os.getpid()
    _directories[path] = (pid, weakref.ref(connection))
    weakref.finalize(connection, _remove, path, pid).atexit = False
    return connection


@contextmanager
def connection(*args, **kwargs) -> Iterator[duckdb.DuckDBPyConnection]:
    """Close a short-lived engine and remove its directory on scope exit."""
    conn = connect(*args, **kwargs)
    path = Path(conn.execute("SELECT current_setting('temp_directory')").fetchone()[0])
    try:
        yield conn
    finally:
        conn.close()
        _remove(path, os.getpid())
