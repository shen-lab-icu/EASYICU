"""Typed content receipts for cache and conversion freshness checks."""

from __future__ import annotations

import hashlib
import json
import os
import threading
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


_CONTENT_RECEIPT_INDEX = ".easyicu_content_receipts.json"
_CONTENT_RECEIPT_LOCK = threading.RLock()


def _load_receipt_index(index_path: Optional[Path], root: Path) -> dict[str, dict]:
    if index_path is None or not index_path.is_file():
        return {}
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("schema_version") == 2:
        roots = payload.get("roots")
        files = roots.get(str(root), {}) if isinstance(roots, dict) else {}
        return files if isinstance(files, dict) else {}
    if payload.get("schema_version") != 1 or payload.get("root") != str(root):
        return {}
    files = payload.get("files")
    return files if isinstance(files, dict) else {}


def _save_receipt_index(
    index_path: Optional[Path], root: Path, receipts: dict[str, dict]
) -> None:
    if index_path is None:
        return
    index_path.parent.mkdir(parents=True, exist_ok=True)
    roots = {}
    previous = ""
    if index_path.is_file():
        try:
            previous = index_path.read_text(encoding="utf-8")
            old = json.loads(previous)
            if not isinstance(old, dict):
                old = {}
            if old.get("schema_version") == 2 and isinstance(old.get("roots"), dict):
                roots = old["roots"]
            elif old.get("schema_version") == 1 and isinstance(old.get("files"), dict):
                roots[str(old.get("root"))] = old["files"]
        except (OSError, ValueError):
            pass
    roots[str(root)] = receipts
    payload = json.dumps(
        {"schema_version": 2, "roots": roots},
        sort_keys=True,
        separators=(",", ":"),
    )
    if payload == previous:
        return
    temporary = index_path.with_name(
        f"{index_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, index_path)


def _current_receipt(path: Path, previous: object) -> dict:
    matches, current = verify_content_receipt(path, previous)
    if matches and current is not None:
        return current
    return file_content_receipt(path)


def data_path_fingerprint(
    data_path: Union[str, Path],
    *,
    exclude_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Fingerprint dataset content with a persistent stat-to-digest index."""
    root = Path(data_path).expanduser().resolve()
    excluded = Path(exclude_dir).expanduser().resolve() if exclude_dir else None
    index_path = excluded / _CONTENT_RECEIPT_INDEX if excluded else None
    excluded_subtree = (
        excluded
        if excluded is not None
        and excluded != root
        and excluded.is_relative_to(root)
        else None
    )
    digest = hashlib.sha256(str(root).encode())

    suffixes = {".parquet", ".csv", ".gz", ".json"}
    if root.is_file():
        files = [root]
    else:
        files = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in suffixes
            and (index_path is None or path != index_path)
            and (
                excluded_subtree is None
                or not path.is_relative_to(excluded_subtree)
            )
        ]

    with _CONTENT_RECEIPT_LOCK:
        previous = _load_receipt_index(index_path, root)
        current_receipts: dict[str, dict] = {}
        for path in sorted(
            files,
            key=lambda item: item.name
            if root.is_file()
            else str(item.relative_to(root)),
        ):
            relative = path.name if root.is_file() else str(path.relative_to(root))
            receipt = _current_receipt(path, previous.get(relative))
            current_receipts[relative] = receipt
            digest.update(f"{relative}:{receipt['sha256']}\n".encode())
        _save_receipt_index(index_path, root, current_receipts)
        return digest.hexdigest()

__all__ = [
    "data_path_fingerprint",
    "CONTENT_RECEIPT_SCHEMA_VERSION",
    "ContentIdentityError",
    "file_content_receipt",
    "verify_content_receipt",
]
