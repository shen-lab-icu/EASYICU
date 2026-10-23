"""Host OS primitives shared by the Web and research authorities.

POSIX retains flock, openat/O_NOFOLLOW, hard links and directory fsync.
Windows uses CRT byte locks (released on process death), and pins every
directory ancestor with Win32 handles that deny delete/rename sharing. Reparse
points, unavailable pinning and unstable file identities fail closed. Windows
publishes with os.replace / no-clobber os.rename; file contents are fsynced but
there is no portable directory-fsync guarantee after power loss.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import errno
import logging
import os
from pathlib import Path
import stat
import sys
import time

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
try:
    import msvcrt
except ImportError:  # POSIX
    msvcrt = None
try:
    import resource
except ImportError:  # Windows
    resource = None


class PlatformCapabilityError(OSError, RuntimeError):
    """A required host guarantee is unavailable; ``reason`` is machine readable."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(detail or reason)


def is_windows() -> bool:
    return os.name == "nt" or sys.platform == "win32"


def lock_exclusive(descriptor: int, *, blocking: bool = True) -> None:
    """Lock the stable inode, without changing file contents or its position.

    Windows permits locking byte zero beyond EOF, so seed/truncate races are
    avoided. LK_LOCK has a finite retry count; use LK_NBLCK with our own retry
    for callers whose POSIX contract is an unbounded blocking flock.
    """
    if not is_windows() and fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return
    if msvcrt is None:
        raise PlatformCapabilityError("exclusive_lock_unavailable")
    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        while True:
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if not blocking:
                    raise BlockingIOError(
                        errno.EAGAIN, "exclusive file lock is held"
                    ) from exc
                time.sleep(0.02)
    finally:
        os.lseek(descriptor, position, os.SEEK_SET)


def unlock_file(descriptor: int) -> None:
    if not is_windows() and fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return
    if msvcrt is None:
        raise PlatformCapabilityError("exclusive_lock_unavailable")
    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    finally:
        os.lseek(descriptor, position, os.SEEK_SET)


def _reject_reparse(info: os.stat_result) -> None:
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise PlatformCapabilityError(
            "reparse_point_refused", "symbolic link/reparse point refused"
        )


def _identity(info: os.stat_result) -> tuple[int, int]:
    if not info.st_ino:
        raise PlatformCapabilityError("file_identity_unavailable")
    return info.st_dev, info.st_ino


def _windows_api():
    """Load Win32 lazily so importing the host owner is portable."""
    import ctypes
    from ctypes import wintypes

    if not hasattr(ctypes, "WinDLL"):
        raise PlatformCapabilityError("directory_pinning_unavailable")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL

    return ctypes, wintypes, kernel


def _windows_handle_info(handle) -> tuple[int, int, int]:
    ctypes, wintypes, kernel = _windows_api()

    class FileInfo(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("created", wintypes.FILETIME),
            ("accessed", wintypes.FILETIME),
            ("written", wintypes.FILETIME),
            ("volume", wintypes.DWORD),
            ("size_hi", wintypes.DWORD),
            ("size_lo", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("index_hi", wintypes.DWORD),
            ("index_lo", wintypes.DWORD),
        ]

    inspect = kernel.GetFileInformationByHandle
    inspect.argtypes = [wintypes.HANDLE, ctypes.POINTER(FileInfo)]
    inspect.restype = wintypes.BOOL
    info = FileInfo()
    if not inspect(handle, ctypes.byref(info)):
        raise PlatformCapabilityError("file_identity_unavailable")
    return info.attributes, info.volume, (info.index_hi << 32) | info.index_lo


def descriptor_identity(descriptor: int) -> tuple[int, int]:
    """CRT fstat may omit Windows inode identity; query the native handle."""
    if not is_windows():
        return _identity(os.fstat(descriptor))
    if msvcrt is None:
        raise PlatformCapabilityError("file_identity_unavailable")
    attributes, volume, inode = _windows_handle_info(msvcrt.get_osfhandle(descriptor))
    if attributes & 0x400 or not inode:
        raise PlatformCapabilityError("file_identity_unavailable")
    return volume, inode


def _pin_windows_directory(path: Path, stack: ExitStack) -> None:
    """Deny ancestor rename AND reparse mutation until the transaction ends.

    A share-write handle would allow setting a junction in an already pinned
    directory. Deny both write and delete sharing; ordinary child operations
    still work. Native directory handles are never passed to CRT os.open.
    """
    ctypes, _wintypes, kernel = _windows_api()
    before = os.lstat(path)
    _reject_reparse(before)
    # FILE_READ_ATTRIBUTES, SHARE_READ, OPEN_EXISTING,
    # FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT.
    handle = kernel.CreateFileW(str(path), 0x80, 0x1, None, 3, 0x02200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise PlatformCapabilityError(
            "directory_pinning_failed", str(ctypes.WinError(ctypes.get_last_error()))
        )
    stack.callback(kernel.CloseHandle, handle)
    attributes, volume, inode = _windows_handle_info(handle)
    after = os.lstat(path)
    _reject_reparse(after)
    if (
        attributes & 0x400
        or not attributes & 0x10
        or _identity(before) != _identity(after)
        or (volume, inode) != _identity(after)
    ):
        raise PlatformCapabilityError("directory_selector_changed")


class SecureDirectory:
    """A pinned directory: openat on POSIX, rename-denying handles on Windows."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().absolute()
        self.fd: int | None = None
        self._stack = ExitStack()
        try:
            if is_windows():
                for part in (*reversed(self.path.parents), self.path):
                    if part != Path(self.path.anchor):
                        _require_component(part.name)
                    _pin_windows_directory(part, self._stack)
                self._info = os.lstat(self.path)
            else:
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                self.fd = os.open(self.path.anchor, flags)
                for part in self.path.parts[1:]:
                    next_fd = os.open(part, flags, dir_fd=self.fd)
                    os.close(self.fd)
                    self.fd = next_fd
                self._info = os.fstat(self.fd)
            _reject_reparse(self._info)
            if not stat.S_ISDIR(self._info.st_mode):
                raise PlatformCapabilityError("directory_required")
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self._stack.close()

    @property
    def identity(self) -> tuple[int, int]:
        return _identity(self._info)

    def assert_still_selected(self) -> None:
        with SecureDirectory(self.path) as selected:
            if selected.identity != self.identity:
                raise PlatformCapabilityError("directory_selector_changed")

    def stat(self, name: str) -> os.stat_result:
        _require_component(name)
        if self.fd is not None:
            return os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        return os.lstat(self.path / name)

    def open(self, name: str, flags: int, mode: int = 0o600) -> int:
        _require_component(name)
        if self.fd is not None:
            return open_nofollow(name, flags, mode, dir_fd=self.fd)
        return _windows_open_nofollow(self.path / name, flags, mode)

    def unlink(self, name: str) -> None:
        _require_component(name)
        if self.fd is not None:
            os.unlink(name, dir_fd=self.fd)
        else:
            os.unlink(self.path / name)

    def replace(self, source: str, target: str) -> None:
        _require_component(source)
        _require_component(target)
        if self.fd is not None:
            os.replace(source, target, src_dir_fd=self.fd, dst_dir_fd=self.fd)
        else:
            os.replace(self.path / source, self.path / target)

    def publish(self, source: str, target: str) -> None:
        """No-clobber publication (Windows rename refuses an existing target)."""
        _require_component(source)
        _require_component(target)
        if self.fd is not None:
            os.link(
                source,
                target,
                src_dir_fd=self.fd,
                dst_dir_fd=self.fd,
                follow_symlinks=False,
            )
        else:
            os.rename(self.path / source, self.path / target)

    def fsync(self) -> None:
        if self.fd is not None:
            os.fsync(self.fd)


def _require_component(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise PlatformCapabilityError("invalid_path_component")
    if is_windows() and (
        ":" in name
        or name.endswith((".", " "))
        or name.split(".")[0].upper()
        in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
    ):
        raise PlatformCapabilityError("windows_path_alias_refused")


@contextmanager
def secure_directory_tree(root: Path, parts: tuple[str, ...], *, create: bool):
    """Create a subtree while its existing ancestors stay pinned."""
    with ExitStack() as stack:
        current = stack.enter_context(SecureDirectory(root))
        for part in parts:
            _require_component(part)
            if create:
                try:
                    if current.fd is None:
                        os.mkdir(current.path / part, mode=0o700)
                    else:
                        os.mkdir(part, mode=0o700, dir_fd=current.fd)
                except FileExistsError:
                    pass
            current = stack.enter_context(SecureDirectory(current.path / part))
        yield current


def _windows_open_nofollow(path: Path, flags: int, mode: int) -> int:
    # Another writer may win first creation of a stable lock sidecar. Recheck
    # that inode instead of weakening EXCL or following a replacement link.
    for _attempt in range(8):
        try:
            return _windows_open_once(path, flags, mode)
        except FileExistsError:
            if not flags & os.O_CREAT or flags & os.O_EXCL:
                raise
    raise PlatformCapabilityError("file_selector_unstable")


def _windows_open_once(path: Path, flags: int, mode: int) -> int:
    _require_component(path.name)
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        before = None
    if before is not None:
        _reject_reparse(before)
        if not stat.S_ISREG(before.st_mode):
            raise PlatformCapabilityError("regular_file_required")
    # Defer truncation, and never let O_CREAT follow a newly introduced dangling
    # link. Existing files must still exist; new files must be created with EXCL.
    safe_flags = (flags & ~os.O_TRUNC) | getattr(os, "O_BINARY", 0)
    if flags & os.O_CREAT and not flags & os.O_EXCL:
        safe_flags = (
            safe_flags & ~os.O_CREAT if before is not None else safe_flags | os.O_EXCL
        )
    descriptor = os.open(path, safe_flags, mode)
    try:
        after = os.lstat(path)
        opened = os.fstat(descriptor)
        identity = descriptor_identity(descriptor)
        _reject_reparse(after)
        if (
            not stat.S_ISREG(opened.st_mode)
            or identity != _identity(after)
            or (before is not None and _identity(before) != identity)
        ):
            raise PlatformCapabilityError("file_selector_changed")
        if flags & os.O_TRUNC:
            os.ftruncate(descriptor, 0)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_nofollow(
    path, flags: int, mode: int = 0o600, *, dir_fd: int | None = None
) -> int:
    if not is_windows():
        return os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), mode, dir_fd=dir_fd)
    if dir_fd is not None:
        raise PlatformCapabilityError("directory_fd_unavailable")
    selected = Path(path).absolute()
    with SecureDirectory(selected.parent):
        return _windows_open_nofollow(selected, flags, mode)


def fsync_directory(path: Path) -> None:
    """Windows: file fsync + atomic rename, without directory durability."""
    if not is_windows():
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_replace(source: Path, target: Path) -> None:
    os.replace(source, target)


def chmod_private_fd(descriptor: int) -> None:
    """Set POSIX owner mode; Windows access remains governed by inherited ACLs."""
    if not is_windows():
        os.fchmod(descriptor, 0o600)


_DEGRADATIONS: set[str] = set()


def _record_degradation(reason: str) -> None:
    if reason not in _DEGRADATIONS:
        _DEGRADATIONS.add(reason)
        logging.getLogger(__name__).warning("Host capability degraded: %s", reason)


def recorded_degradations() -> tuple[str, ...]:
    return tuple(sorted(_DEGRADATIONS))


def set_resource_limit(name: str, limits: tuple[int, int]) -> bool:
    """Apply a POSIX limit, or record an explicit Windows no-op."""
    if is_windows() or resource is None:
        _record_degradation("resource_limits_unavailable")
        return False
    resource.setrlimit(getattr(resource, name), limits)
    return True


def peak_rss_mb() -> float:
    if not is_windows() and resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    _record_degradation("peak_rss_unavailable")
    return 0.0
