"""Recoverable maintenance for Pi transcript files outside the session index."""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .contracts import PiCopilotError


DEFAULT_ORPHAN_GRACE_SECONDS = 7 * 24 * 60 * 60
_QUARANTINE_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}$")


def _utc_batch_id(now: float) -> str:
    stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(4)}"


def _resolved_references(paths: Iterable[str | Path]) -> set[Path]:
    return {
        Path(path).expanduser().absolute()
        for path in paths
        if str(path or "").strip()
    }


@dataclass(frozen=True)
class SessionStorageInventory:
    session_files: int
    referenced_files: int
    unreferenced_files: int
    eligible_files: int
    total_bytes: int
    unreferenced_bytes: int
    eligible_bytes: int
    eligible_paths: tuple[Path, ...]

    def public_projection(self) -> dict[str, int]:
        return {
            "session_files": self.session_files,
            "referenced_files": self.referenced_files,
            "unreferenced_files": self.unreferenced_files,
            "eligible_files": self.eligible_files,
            "total_bytes": self.total_bytes,
            "unreferenced_bytes": self.unreferenced_bytes,
            "eligible_bytes": self.eligible_bytes,
        }


class SessionStorageMaintenance:
    def __init__(
        self,
        session_dir: Path,
        *,
        grace_seconds: int = DEFAULT_ORPHAN_GRACE_SECONDS,
    ) -> None:
        self.session_dir = Path(session_dir).expanduser().absolute()
        self.quarantine_root = self.session_dir / "quarantine"
        self.grace_seconds = max(60, int(grace_seconds))

    def inventory(
        self,
        referenced_paths: Iterable[str | Path],
        *,
        now: Optional[float] = None,
    ) -> SessionStorageInventory:
        references = _resolved_references(referenced_paths)
        observed_now = time.time() if now is None else float(now)
        files: list[tuple[Path, os.stat_result]] = []
        if self.session_dir.is_dir() and not self.session_dir.is_symlink():
            for candidate in self.session_dir.iterdir():
                try:
                    metadata = candidate.lstat()
                except OSError:
                    continue
                if (
                    candidate.suffix == ".jsonl"
                    and candidate.is_file()
                    and not candidate.is_symlink()
                ):
                    files.append((candidate.absolute(), metadata))
        unreferenced = [row for row in files if row[0] not in references]
        eligible = [
            row
            for row in unreferenced
            if observed_now - float(row[1].st_mtime) >= self.grace_seconds
        ]
        return SessionStorageInventory(
            session_files=len(files),
            referenced_files=sum(path in references for path, _ in files),
            unreferenced_files=len(unreferenced),
            eligible_files=len(eligible),
            total_bytes=sum(metadata.st_size for _, metadata in files),
            unreferenced_bytes=sum(metadata.st_size for _, metadata in unreferenced),
            eligible_bytes=sum(metadata.st_size for _, metadata in eligible),
            eligible_paths=tuple(path for path, _ in eligible),
        )

    def quarantine(
        self,
        referenced_paths: Iterable[str | Path],
        *,
        confirm: bool,
        now: Optional[float] = None,
    ) -> dict[str, object]:
        if not confirm:
            raise PiCopilotError(
                "pi_session_quarantine_confirmation_required",
                "Explicit confirmation is required before moving transcript files.",
                status_code=409,
            )
        references = _resolved_references(referenced_paths)
        observed_now = time.time() if now is None else float(now)
        inventory = self.inventory(references, now=observed_now)
        if not inventory.eligible_paths:
            return {
                "status": "nothing_to_quarantine",
                "quarantine_id": None,
                "moved_files": 0,
                "moved_bytes": 0,
                "inventory": inventory.public_projection(),
            }
        quarantine_id = _utc_batch_id(observed_now)
        destination = self.quarantine_root / quarantine_id
        destination.mkdir(parents=True, exist_ok=False, mode=0o700)
        moved: list[dict[str, object]] = []
        try:
            for source in inventory.eligible_paths:
                if source in references:
                    continue
                try:
                    metadata = source.lstat()
                except OSError:
                    continue
                if source.is_symlink() or not source.is_file():
                    continue
                target = destination / source.name
                if target.exists() or target.is_symlink():
                    raise PiCopilotError(
                        "pi_session_quarantine_collision",
                        "A quarantine target already exists.",
                        status_code=409,
                    )
                os.replace(source, target)
                moved.append(
                    {
                        "file": source.name,
                        "size_bytes": int(metadata.st_size),
                        "mtime_epoch": float(metadata.st_mtime),
                    }
                )
            manifest = {
                "schema_version": "easyicu.pi-session-quarantine/1",
                "quarantine_id": quarantine_id,
                "created_at_epoch": observed_now,
                "files": moved,
            }
            self._write_manifest(destination, manifest)
        except Exception as exc:
            for row in reversed(moved):
                target = destination / str(row["file"])
                source = self.session_dir / str(row["file"])
                if target.is_file() and not target.is_symlink() and not source.exists():
                    try:
                        os.replace(target, source)
                    except OSError:
                        pass
            try:
                destination.rmdir()
            except OSError:
                pass
            if isinstance(exc, PiCopilotError):
                raise
            raise PiCopilotError(
                "pi_session_quarantine_io_error",
                "Transcript quarantine could not be completed safely.",
                status_code=500,
            ) from exc
        return {
            "status": "quarantined",
            "quarantine_id": quarantine_id,
            "moved_files": len(moved),
            "moved_bytes": sum(int(row["size_bytes"]) for row in moved),
            "inventory": inventory.public_projection(),
        }

    def restore(self, quarantine_id: str, *, confirm: bool) -> dict[str, object]:
        clean_id = str(quarantine_id or "").strip()
        if not confirm:
            raise PiCopilotError(
                "pi_session_restore_confirmation_required",
                "Explicit confirmation is required before restoring transcript files.",
                status_code=409,
            )
        if _QUARANTINE_ID.fullmatch(clean_id) is None:
            raise PiCopilotError(
                "pi_session_quarantine_id_invalid",
                "The transcript quarantine identifier is invalid.",
                status_code=422,
            )
        source_root = self.quarantine_root / clean_id
        try:
            manifest = json.loads(
                (source_root / "manifest.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise PiCopilotError(
                "pi_session_quarantine_not_found",
                "The transcript quarantine manifest is unavailable.",
                status_code=404,
            ) from exc
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if (
            manifest.get("schema_version") != "easyicu.pi-session-quarantine/1"
            or manifest.get("quarantine_id") != clean_id
            or not isinstance(files, list)
        ):
            raise PiCopilotError(
                "pi_session_quarantine_invalid",
                "The transcript quarantine manifest is invalid.",
                status_code=409,
            )
        self.session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        moves: list[tuple[Path, Path]] = []
        for row in files:
            name = str(row.get("file") or "") if isinstance(row, dict) else ""
            if Path(name).name != name or not name.endswith(".jsonl"):
                raise PiCopilotError(
                    "pi_session_quarantine_invalid",
                    "The transcript quarantine manifest contains an invalid file.",
                    status_code=409,
                )
            source = source_root / name
            target = self.session_dir / name
            if target.exists() or target.is_symlink():
                raise PiCopilotError(
                    "pi_session_restore_collision",
                    "A transcript with the same name already exists.",
                    status_code=409,
                )
            if not source.is_file() or source.is_symlink():
                raise PiCopilotError(
                    "pi_session_quarantine_invalid",
                    "A quarantined transcript is unavailable.",
                    status_code=409,
                )
            moves.append((source, target))
        restored: list[tuple[Path, Path]] = []
        try:
            for source, target in moves:
                os.replace(source, target)
                restored.append((source, target))
        except OSError as exc:
            for source, target in reversed(restored):
                if target.is_file() and not target.is_symlink() and not source.exists():
                    try:
                        os.replace(target, source)
                    except OSError:
                        pass
            raise PiCopilotError(
                "pi_session_restore_io_error",
                "Transcript restore could not be completed safely.",
                status_code=500,
            ) from exc
        return {
            "status": "restored",
            "quarantine_id": clean_id,
            "restored_files": len(restored),
        }

    @staticmethod
    def _write_manifest(destination: Path, payload: dict[str, object]) -> None:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination,
            prefix=".manifest-",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            temporary.replace(destination / "manifest.json")
        finally:
            temporary.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_ORPHAN_GRACE_SECONDS",
    "SessionStorageInventory",
    "SessionStorageMaintenance",
]
