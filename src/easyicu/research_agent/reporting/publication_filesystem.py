"""No-follow filesystem transactions for publication-bundle promotion."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from ..authority.filesystem import AnchoredDirectory, AuthorityFilesystemError


def require_real_output_dir(out_dir: Path, run_dir: Optional[Path]) -> None:
    """Refuse an agent-created link that redirects host promotion writes."""

    if out_dir.is_symlink():
        raise ValueError("publication output directory is a symlink")
    if run_dir is None:
        return
    try:
        relative = out_dir.absolute().relative_to(run_dir.absolute())
        out_dir.resolve().relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError("publication output directory escapes run root") from exc
    current = run_dir
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("publication output directory contains a symlink")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace a file without following a pre-existing destination link."""

    with AnchoredDirectory.open(path.parent) as directory:
        directory.assert_still_selected()
        directory.replace_bytes(path.name, payload)


def _read_step_summary_bytes(out_dir: Path) -> Optional[bytes]:
    with AnchoredDirectory.open(out_dir) as directory:
        if directory.is_absent("step_summary.json"):
            return None
        with directory.open_regular("step_summary.json") as handle:
            return handle.read()


def _copy_publication_file(
    source: Path,
    target: Path,
    *,
    out_dir: Path,
    run_dir: Optional[Path],
) -> None:
    """Copy a regular run file to a real output path without following links."""

    if source.is_symlink() or not source.is_file():
        raise ValueError("publication source is a symlink or not a regular file")
    if run_dir is not None and not source.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("publication source escapes run root")
    try:
        relative_parent = target.parent.absolute().relative_to(out_dir.absolute())
        target.parent.resolve().relative_to(out_dir.resolve())
    except ValueError as exc:
        raise ValueError("publication destination escapes output directory") from exc
    current = out_dir
    for part in relative_parent.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("publication destination contains a symlink")
    if target.is_symlink():
        raise ValueError("publication destination is a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)

    with AnchoredDirectory.open(source.parent) as source_dir:
        with AnchoredDirectory.open(target.parent) as target_dir:
            with source_dir.open_regular(source.name) as source_handle:
                temporary_name, descriptor = target_dir.create_temporary(
                    stem=target.name
                )
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        descriptor = -1
                        shutil.copyfileobj(source_handle, handle)
                        handle.flush()
                        os.fsync(handle.fileno())
                    source_dir.assert_still_selected()
                    target_dir.assert_still_selected()
                    target_dir.replace_temporary(
                        temporary_name, target.name, require_absent=False
                    )
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
                    target_dir.unlink(temporary_name, missing_ok=True)


def _seal_corrupt_step_summary(out_dir: Path) -> Optional[str]:
    """Quarantine an unparseable ``step_summary.json`` before it is overwritten.

    Promotion paths rebuild the summary from rescued figure exports; wiping
    an unreadable file first would destroy the "corrupt" evidence itself.
    The raw bytes are preserved as ``step_summary.corrupt.<sha8>.json``
    (sha8 of the corrupt content, so identical corruption seals
    idempotently) and the sealed filename is returned for the rescue
    record.  Returns ``None`` when there is nothing corrupt to seal.
    """

    if (out_dir / "step_summary.json").is_symlink():
        raise ValueError("step_summary.json is a symlink")
    try:
        raw = _read_step_summary_bytes(out_dir)
    except AuthorityFilesystemError as exc:
        raise ValueError("step_summary.json cannot be preserved") from exc
    if raw is None:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return None
    digest = hashlib.sha256(raw).hexdigest()[:8]
    sealed_name = f"step_summary.corrupt.{digest}.json"
    try:
        with AnchoredDirectory.open(out_dir) as directory:
            directory.publish_immutable_bytes(sealed_name, raw)
    except AuthorityFilesystemError as exc:
        raise ValueError("corrupt summary seal cannot be preserved") from exc
    return sealed_name
