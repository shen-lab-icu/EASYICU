"""Descriptor-anchored filesystem primitives for authority publication.

Authority writers must not validate a directory by pathname and then keep
using that pathname: a concurrent rename can replace the verified directory
with a symbolic link before the first write.  This module opens every path
component with ``openat``/``O_NOFOLLOW`` and keeps the final directory
descriptor alive for the complete transaction.

The helpers are deliberately science-neutral.  They publish bytes and files;
they do not select cohorts, variables, methods, or estimands.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import secrets
import stat
import tempfile
from typing import BinaryIO, Callable, Optional


class AuthorityFilesystemError(RuntimeError):
    """Raised when an authority path cannot be used without following links."""


def _component(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise AuthorityFilesystemError("authority object name is not one component")
    return name


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_directory_tree(path: Path) -> int:
    """Open an absolute directory one no-follow component at a time."""

    candidate = Path(path).expanduser()
    candidate = candidate if candidate.is_absolute() else candidate.absolute()
    if os.name != "posix":  # pragma: no cover - production and CI are POSIX
        if candidate.is_symlink() or not candidate.is_dir():
            raise AuthorityFilesystemError(
                "authority root must be an existing real directory"
            )
        try:
            return os.open(candidate, os.O_RDONLY)
        except OSError as exc:
            raise AuthorityFilesystemError("cannot open authority root") from exc

    flags = _directory_flags()
    try:
        current_fd = os.open(candidate.anchor or "/", flags)
    except OSError as exc:
        raise AuthorityFilesystemError("cannot open authority filesystem root") from exc
    try:
        for part in candidate.parts[1:]:
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except OSError as exc:
                raise AuthorityFilesystemError(
                    f"authority directory component is missing or unsafe: {part}"
                ) from exc
            info = os.fstat(next_fd)
            if not stat.S_ISDIR(info.st_mode):
                os.close(next_fd)
                raise AuthorityFilesystemError(
                    f"authority directory component is not a directory: {part}"
                )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


@dataclass(slots=True)
class AnchoredDirectory:
    """One verified directory held open for a complete authority transaction."""

    path: Path
    fd: int
    _closed: bool = False

    @classmethod
    def open(cls, path: Path) -> "AnchoredDirectory":
        candidate = Path(path).expanduser()
        candidate = candidate if candidate.is_absolute() else candidate.absolute()
        return cls(path=candidate, fd=_open_directory_tree(candidate))

    def __enter__(self) -> "AnchoredDirectory":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            os.close(self.fd)
            self._closed = True

    @property
    def identity(self) -> tuple[int, int]:
        self._require_open()
        info = os.fstat(self.fd)
        return int(info.st_dev), int(info.st_ino)

    def assert_still_selected(self) -> None:
        """Fail if the original pathname no longer selects this directory."""

        self._require_open()
        selected_fd: Optional[int] = None
        try:
            selected_fd = _open_directory_tree(self.path)
            selected = os.fstat(selected_fd)
            held = os.fstat(self.fd)
            if (selected.st_dev, selected.st_ino) != (held.st_dev, held.st_ino):
                raise AuthorityFilesystemError(
                    "authority directory selector changed during publication"
                )
        finally:
            if selected_fd is not None:
                os.close(selected_fd)

    def _require_open(self) -> None:
        if self._closed:
            raise AuthorityFilesystemError("authority directory is closed")

    def stat(self, name: str) -> os.stat_result:
        self._require_open()
        name = _component(name)
        try:
            return os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except OSError as exc:
            raise AuthorityFilesystemError(
                f"cannot inspect authority object: {name}"
            ) from exc

    def is_absent(self, name: str) -> bool:
        self._require_open()
        name = _component(name)
        try:
            os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return True
        except OSError as exc:
            raise AuthorityFilesystemError(
                f"cannot inspect authority object: {name}"
            ) from exc
        return False

    def require_absent(self, *names: str) -> None:
        for name in names:
            if not self.is_absent(name):
                raise AuthorityFilesystemError(
                    f"authority target already exists: {_component(name)}"
                )

    def open_regular(self, name: str) -> BinaryIO:
        """Open one regular file relative to this directory without links."""

        self._require_open()
        name = _component(name)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(name, flags, dir_fd=self.fd)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise AuthorityFilesystemError(
                    f"authority object is not a regular file: {name}"
                )
            handle = os.fdopen(descriptor, "rb")
            descriptor = None
            return handle
        except AuthorityFilesystemError:
            raise
        except OSError as exc:
            raise AuthorityFilesystemError(
                f"cannot open authority object: {name}"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def read_bytes(
        self,
        name: str,
        *,
        max_bytes: int,
        expected_size: Optional[int] = None,
        expected_sha256: Optional[str] = None,
    ) -> bytes:
        with self.open_regular(name) as handle:
            info = os.fstat(handle.fileno())
            size = int(info.st_size)
            if size > max_bytes or (
                expected_size is not None and size != expected_size
            ):
                raise AuthorityFilesystemError(
                    f"authority object size mismatch: {_component(name)}"
                )
            payload = handle.read(size + 1)
        if len(payload) != size:
            raise AuthorityFilesystemError(
                f"authority object changed while reading: {_component(name)}"
            )
        if (
            expected_sha256 is not None
            and hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise AuthorityFilesystemError(
                f"authority object digest mismatch: {_component(name)}"
            )
        return payload

    def create_temporary(self, *, stem: str) -> tuple[str, int]:
        self._require_open()
        stem = _component(stem)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        for _ in range(32):
            name = f".{stem}.{secrets.token_hex(8)}.tmp"
            try:
                return name, os.open(name, flags, 0o600, dir_fd=self.fd)
            except FileExistsError:
                continue
            except OSError as exc:
                raise AuthorityFilesystemError(
                    "cannot create temporary authority object"
                ) from exc
        raise AuthorityFilesystemError("cannot allocate temporary authority object")

    def unlink(self, name: str, *, missing_ok: bool = True) -> None:
        self._require_open()
        try:
            os.unlink(_component(name), dir_fd=self.fd)
        except FileNotFoundError:
            if not missing_ok:
                raise
        except OSError as exc:
            raise AuthorityFilesystemError(
                f"cannot remove authority object: {name}"
            ) from exc

    def replace_temporary(
        self,
        temporary_name: str,
        target_name: str,
        *,
        require_absent: bool,
    ) -> None:
        self._require_open()
        temporary_name = _component(temporary_name)
        target_name = _component(target_name)
        if require_absent:
            self.require_absent(target_name)
        try:
            os.replace(
                temporary_name,
                target_name,
                src_dir_fd=self.fd,
                dst_dir_fd=self.fd,
            )
            os.fsync(self.fd)
        except OSError as exc:
            raise AuthorityFilesystemError(
                f"cannot publish authority object: {target_name}"
            ) from exc

    def replace_bytes(
        self,
        name: str,
        payload: bytes,
        *,
        require_absent: bool = False,
    ) -> None:
        name = _component(name)
        temporary_name, descriptor = self.create_temporary(stem=name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            self.replace_temporary(temporary_name, name, require_absent=require_absent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            self.unlink(temporary_name, missing_ok=True)

    def publish_immutable_bytes(self, name: str, payload: bytes) -> None:
        """Publish one write-once object or verify identical existing bytes."""

        name = _component(name)
        temporary_name, descriptor = self.create_temporary(stem=name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(
                    temporary_name,
                    name,
                    src_dir_fd=self.fd,
                    dst_dir_fd=self.fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                existing = self.read_bytes(name, max_bytes=max(len(payload), 1))
                if existing != payload:
                    raise AuthorityFilesystemError(
                        "existing immutable authority object conflicts with payload"
                    )
            os.fsync(self.fd)
        except AuthorityFilesystemError:
            raise
        except OSError as exc:
            raise AuthorityFilesystemError(
                f"cannot publish immutable authority object: {name}"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            self.unlink(temporary_name, missing_ok=True)


def publish_write_once_bytes(
    path: Path,
    payload: bytes,
    *,
    temp_prefix: str,
    conflict_error: Callable[[str], BaseException],
    conflict_message: str,
    race_message: Optional[str] = None,
) -> None:
    """Publish an immutable receipt, or verify the published bytes match.

    The path-anchored sibling of :meth:`AnchoredDirectory.publish_immutable_bytes`,
    for callers that hold a plain ``Path`` rather than an open directory
    descriptor. Six callers -- reviewed memory, the coder resource snapshot, the
    cross-run memory store and the three capability records -- each wrote this
    sequence out by hand, differing only in exception type and temp prefix. An
    integrity rule with six implementations has six chances to drift apart.

    The hard link, not ``os.replace``, is what makes this write-*once*: replace
    would let a second writer silently overwrite a receipt someone else already
    published, so the loser of a race would never learn it lost. Losing to
    identical bytes is a successful publish; losing to different bytes raises.

    ``conflict_error`` is a factory rather than an exception class so callers
    keep their own typed error and their own wording -- the point is one
    algorithm, not one vocabulary.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise conflict_error(conflict_message)
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=temp_prefix, dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise conflict_error(race_message or conflict_message) from None
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


__all__ = [
    "AnchoredDirectory",
    "AuthorityFilesystemError",
    "publish_write_once_bytes",
]
