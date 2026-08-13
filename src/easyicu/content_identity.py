"""Typed content receipts for cache and conversion freshness checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union


CONTENT_RECEIPT_SCHEMA_VERSION = 1


class ContentIdentityError(OSError):
    """A stable content receipt could not be established for a source file."""

    def __init__(self, code: str, path: Union[str, Path], message: str) -> None:
        super().__init__(message)
        self.code = code
        self.path = Path(path)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(stat: Any) -> Tuple[int, int, int, int, int]:
    return (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(stat.st_dev),
        int(stat.st_ino),
    )


def file_content_receipt(path: Union[str, Path]) -> dict[str, Any]:
    """Hash one stable file snapshot and bind it to cheap stat evidence."""

    resolved = Path(path)
    for _attempt in range(2):
        try:
            before = resolved.stat()
            sha256 = _sha256_file(resolved)
            after = resolved.stat()
        except OSError as exc:
            raise ContentIdentityError(
                "content_identity_unreadable",
                resolved,
                f"Could not read source content identity for {resolved}: {exc}",
            ) from exc
        if _stat_identity(before) == _stat_identity(after):
            return {
                "schema_version": CONTENT_RECEIPT_SCHEMA_VERSION,
                "size_bytes": int(after.st_size),
                "mtime_ns": int(after.st_mtime_ns),
                "ctime_ns": int(after.st_ctime_ns),
                "device": int(after.st_dev),
                "inode": int(after.st_ino),
                "sha256": sha256,
            }
    raise ContentIdentityError(
        "content_identity_changed_during_read",
        resolved,
        f"Source changed while its content identity was being read: {resolved}",
    )


def verify_content_receipt(
    path: Union[str, Path], receipt: object
) -> tuple[bool, Optional[dict[str, Any]]]:
    """Verify a receipt, hashing only when its cheap stat evidence changed."""

    if not isinstance(receipt, Mapping):
        return False, None
    required = {
        "schema_version",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
        "device",
        "inode",
        "sha256",
    }
    if not required.issubset(receipt):
        return False, None
    if receipt.get("schema_version") != CONTENT_RECEIPT_SCHEMA_VERSION:
        return False, None

    resolved = Path(path)
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise ContentIdentityError(
            "content_identity_unreadable",
            resolved,
            f"Could not stat source content for {resolved}: {exc}",
        ) from exc
    current_stat = _stat_identity(stat)
    receipt_stat = (
        int(receipt["size_bytes"]),
        int(receipt["mtime_ns"]),
        int(receipt["ctime_ns"]),
        int(receipt["device"]),
        int(receipt["inode"]),
    )
    if current_stat == receipt_stat:
        return True, dict(receipt)
    if int(stat.st_size) != int(receipt["size_bytes"]):
        return False, None

    current = file_content_receipt(resolved)
    return current["sha256"] == str(receipt["sha256"]), current


__all__ = [
    "CONTENT_RECEIPT_SCHEMA_VERSION",
    "ContentIdentityError",
    "file_content_receipt",
    "verify_content_receipt",
]
