"""Staging primitives for a future reviewed SIC raw-data producer.

No route invokes this module. A file snapshot alone does not prove that the
exporter consumed it, that the roster covers all reads, or that a research plan
is authorized. The caller must establish those independent facts before using
this primitive; no patient data are read by importing this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Iterable

from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.canonical_json import canonical_sha256


@dataclass(frozen=True)
class SnapshotFile:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class StagedRawSnapshot:
    """Content-verified staged files; explicitly not a producer attestation."""

    source_identity_sha256: str
    files: tuple[SnapshotFile, ...]
    schema_version: str = "easyicu.raw-source-stage/1"

    def public_receipt(self) -> dict:
        content = {
            "schema_version": self.schema_version,
            "source_identity_sha256": self.source_identity_sha256,
            "files": [dict(relative_path=f.relative_path, sha256=f.sha256,
                           size_bytes=f.size_bytes) for f in self.files],
            "source_identity_verified": False,
            "point_in_time_source_snapshot_verified": False,
            "actual_consumption_verified": False,
            "read_roster_complete": False,
        }
        return {**content, "snapshot_sha256": canonical_sha256(content)}


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("raw_snapshot_invalid_relative_path")
    path = PurePosixPath(value)
    if (path.is_absolute() or not path.parts or path.as_posix() != value
            or any(p in {".", ".."} for p in path.parts)):
        raise ValueError("raw_snapshot_invalid_relative_path")
    return path


def _hash_file(handle) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def stage_reviewed_source_files(*, source_root: Path, staging_root: Path,
                                relative_files: Iterable[str], source_identity_sha256: str) -> StagedRawSnapshot:
    """Copy a *host-authorized* roster to a new private staging directory.

    This low-level function is intentionally not a research authorization
    check. Future integration must first verify a signed preparation grant for
    the exact source and complete roster. A directory is untrusted until all
    entries have been copied and the returned receipt has been verified; a
    failure leaves no success receipt and may leave partial private files.
    """
    import re

    source_root, staging_root = Path(source_root), Path(staging_root)
    if (not source_root.is_absolute() or not staging_root.is_absolute()
            or ".." in source_root.parts or ".." in staging_root.parts
            or not re.fullmatch(r"[0-9a-f]{64}", source_identity_sha256)):
        raise ValueError("raw_snapshot_arguments_invalid")
    if source_root == staging_root or source_root in staging_root.parents or staging_root in source_root.parents:
        raise ValueError("raw_snapshot_source_destination_overlap")
    paths = tuple(_relative(value) for value in relative_files)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("raw_snapshot_roster_invalid")
    paths = tuple(sorted(paths, key=lambda item: item.as_posix()))
    # The host owns this new directory; never overwrite a previous snapshot.
    with AnchoredDirectory.open(staging_root.parent) as parent:
        parent.create_private_directory(staging_root.name)
        parent.assert_still_selected()
    captured = []
    with AnchoredDirectory.open(source_root) as source_anchor, AnchoredDirectory.open(staging_root) as staged_anchor:
        for relative in paths:
            source_parent = source_root.joinpath(*relative.parts[:-1])
            target_parent = staging_root.joinpath(*relative.parts[:-1])
            target_path = staging_root
            for component in relative.parts[:-1]:
                with AnchoredDirectory.open(target_path) as parent:
                    try:
                        parent.create_private_directory(component)
                    except FileExistsError:
                        pass
                    parent.assert_still_selected()
                target_path = target_path / component
            with AnchoredDirectory.open(source_parent) as source_dir, AnchoredDirectory.open(target_parent) as target_dir:
                target_dir.require_absent(relative.name)
                temp_name, descriptor = target_dir.create_temporary(stem=relative.name)
                try:
                    with source_dir.open_regular(relative.name) as src, os.fdopen(descriptor, "wb") as dst:
                        descriptor = -1
                        digest = hashlib.sha256()
                        size = 0
                        for chunk in iter(lambda: src.read(1024 * 1024), b""):
                            dst.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                        dst.flush()
                        os.fsync(dst.fileno())
                    expected = digest.hexdigest()
                    # Reopen the selected name, so replacing the original inode
                    # during the copy is detected as well as in-place writes.
                    with source_dir.open_regular(relative.name) as src:
                        if _hash_file(src) != (expected, size):
                            raise ValueError("raw_snapshot_source_changed")
                    source_dir.assert_still_selected()
                    target_dir.assert_still_selected()
                    source_anchor.assert_still_selected()
                    staged_anchor.assert_still_selected()
                    target_dir.publish_temporary_no_clobber(temp_name, relative.name)
                    with target_dir.open_regular(relative.name) as copied:
                        if _hash_file(copied) != (expected, size):
                            raise ValueError("raw_snapshot_staged_bytes_changed")
                    captured.append(SnapshotFile(relative.as_posix(), expected, size))
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
                    target_dir.unlink(temp_name, missing_ok=True)
        source_anchor.assert_still_selected()
        staged_anchor.assert_still_selected()
    result = StagedRawSnapshot(source_identity_sha256, tuple(captured))
    verify_staged_source_files(staging_root, result)
    return result


def verify_staged_source_files(staging_root: Path, receipt: StagedRawSnapshot) -> None:
    """Recheck staged bytes, including names, before and after a consumer."""
    root = Path(staging_root)
    with AnchoredDirectory.open(root) as anchor:
        for entry in receipt.files:
            relative = _relative(entry.relative_path)
            with AnchoredDirectory.open(root.joinpath(*relative.parts[:-1])) as directory:
                with directory.open_regular(relative.name) as handle:
                    if _hash_file(handle) != (entry.sha256, entry.size_bytes):
                        raise ValueError("raw_snapshot_staged_bytes_changed")
                directory.assert_still_selected()
        anchor.assert_still_selected()
