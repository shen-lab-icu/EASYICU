"""Private restart-recovery index for Web Research Agent review pauses."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.canonical_json import canonical_sha256


class WebReviewRecoveryError(RuntimeError):
    """The private Web recovery index is absent, corrupt, or drifted."""


_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: Dict[str, threading.RLock] = {}


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Serialize index reads and read-modify-write cycles across hosts/threads."""

    selected = Path(path)
    selected.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = str(selected.resolve())
    with _LOCKS_GUARD:
        thread_lock = _PATH_LOCKS.setdefault(key, threading.RLock())
    lock_path = selected.with_name(f".{selected.name}.lock")
    with thread_lock:
        if lock_path.is_symlink():
            raise WebReviewRecoveryError("Web review recovery lock cannot be a symlink")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            flock(descriptor, LOCK_EX)
            yield
        finally:
            flock(descriptor, LOCK_UN)
            os.close(descriptor)


class WebReviewRecoveryRecord(BaseModel):
    """Non-secret coordinates needed to reconstruct one paused Web run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "easyicu.web-review-recovery/1", "easyicu.web-review-recovery/2"
    ] = (
        "easyicu.web-review-recovery/2"
    )
    run_id: str
    wrapper_dir: str
    study: Dict[str, Any]
    scientific_configuration_sha256: str
    provider_meta: Dict[str, Any]
    provider_public: Dict[str, Any]
    credential_source: Literal["pi_verified", "scientific_provider"]
    pipeline_config: Dict[str, Any]
    pipeline_config_sha256: Optional[str] = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    acquisition_projection: Dict[str, Any]
    hard_stop_ledger_path: str
    hard_stop_task_id: str
    hard_stop_declaration_sha256: str
    created_at: float = Field(ge=0.0)
    record_sha256: str

    @model_validator(mode="after")
    def _verify_digest(self) -> "WebReviewRecoveryRecord":
        if (
            self.schema_version == "easyicu.web-review-recovery/2"
            and self.pipeline_config_sha256 is None
        ):
            raise ValueError("Web review recovery v2 requires a pipeline digest")
        payload = self.model_dump(mode="json", exclude={"record_sha256"})
        if canonical_sha256(payload) != self.record_sha256:
            raise ValueError("Web review recovery digest mismatch")
        return self

    @classmethod
    def create(cls, **values: Any) -> "WebReviewRecoveryRecord":
        values = dict(values)
        values.pop("schema_version", None)
        payload = {
            "schema_version": "easyicu.web-review-recovery/2",
            **values,
        }
        payload["record_sha256"] = canonical_sha256(payload)
        return cls.model_validate(payload)


def default_store_path() -> Path:
    return Path.home() / ".easyicu" / "web_review_recovery.json"


def _read(path: Path) -> Dict[str, Any]:
    if path.is_symlink():
        raise WebReviewRecoveryError("Web review recovery index cannot be a symlink")
    if not path.exists():
        return {"schema_version": "easyicu.web-review-recovery-index/1", "records": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise WebReviewRecoveryError("Web review recovery index is corrupt") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "easyicu.web-review-recovery-index/1"
        or not isinstance(payload.get("records"), dict)
    ):
        raise WebReviewRecoveryError("Web review recovery index has an invalid schema")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp_name = tempfile.mkstemp(prefix=".web-review-", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        path.chmod(0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass


def put_record(
    record: WebReviewRecoveryRecord,
    *,
    path: Optional[Path] = None,
    max_records: int = 128,
) -> None:
    selected = path or default_store_path()
    with _locked(selected):
        payload = _read(selected)
        records = dict(payload["records"])
        if record.run_id not in records and len(records) >= max_records:
            raise WebReviewRecoveryError(
                "Web review recovery capacity is full; no pending review was evicted"
            )
        records[record.run_id] = record.model_dump(mode="json")
        _write(selected, {**payload, "records": records})


def get_record(run_id: str, *, path: Optional[Path] = None) -> Optional[WebReviewRecoveryRecord]:
    selected = path or default_store_path()
    with _locked(selected):
        payload = _read(selected)
        raw = payload["records"].get(str(run_id))
    if raw is None:
        return None
    try:
        return WebReviewRecoveryRecord.model_validate(raw)
    except Exception as exc:
        raise WebReviewRecoveryError("Web review recovery record is corrupt") from exc


def remove_record(run_id: str, *, path: Optional[Path] = None) -> None:
    selected = path or default_store_path()
    with _locked(selected):
        payload = _read(selected)
        records = dict(payload["records"])
        if records.pop(str(run_id), None) is not None:
            _write(selected, {**payload, "records": records})
