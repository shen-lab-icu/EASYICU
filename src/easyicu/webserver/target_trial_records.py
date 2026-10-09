"""Where the host keeps the target trial records a study binds by digest.

Owner
-----
A study's ``target_trial_design`` section keeps the digest of the record the
host compiled for its approval card, not the record itself
(``research_agent.planning.target_trial_configuration``): a study's
configuration is small metadata, and a record with its protocol, elements and
capture entries does not fit beside the rest of a study.  This module keeps
each record, with the population spec it was compiled from, in a file under
the host's state that its digest names.  A record is written once and never
changes; a read checks the record against its digest.  The section's digest
is part of the study's scientific configuration, so the record stays bound to
the study through it.

Nothing here reads a patient row: a record holds none.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from easyicu.research_agent.planning.target_trial_configuration import (
    TargetTrialCompileRecord,
    TargetTrialDesignError,
    load_target_trial_compile_record,
)
from easyicu.webserver import state_paths

#: A kept record is a few dozen kilobytes; anything far larger is not one.
MAX_RECORD_BYTES = 512 * 1024
_SHA256_CHARS = frozenset("0123456789abcdef")
_LOCK = threading.Lock()


class TargetTrialRecordError(RuntimeError):
    """A kept target trial record cannot be written or read as it was kept."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def records_root() -> Path:
    """The directory of every study's kept records, under the host's state."""

    return state_paths.state_root() / "target-trial-compiles"


def _record_path(study_id: str, compile_sha256: str) -> Path:
    study = str(study_id or "").strip()
    digest = str(compile_sha256 or "").strip()
    if not study or len(digest) != 64 or not set(digest) <= _SHA256_CHARS:
        raise TargetTrialRecordError(
            "target_trial_record_coordinates_invalid",
            "A kept record is named by its study and its digest.",
        )
    study_key = hashlib.sha256(study.encode("utf-8")).hexdigest()[:24]
    return records_root() / study_key / "records" / f"{digest}.json"


def _encoded(kept: TargetTrialCompileRecord) -> bytes:
    return json.dumps(
        kept.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def keep_target_trial_record(study_id: str, kept: TargetTrialCompileRecord) -> None:
    """Keep ``kept`` for the study; keeping the same record again is a no-op."""

    path = _record_path(study_id, kept.compile_sha256)
    encoded = _encoded(kept)
    if len(encoded) > MAX_RECORD_BYTES:
        raise TargetTrialRecordError(
            "target_trial_record_too_large",
            "The compile record exceeds the size a kept record may have.",
            max_bytes=MAX_RECORD_BYTES,
        )
    with _LOCK:
        if path.exists():
            if _read_bytes(path) != encoded:
                raise TargetTrialRecordError(
                    "target_trial_record_identity_drift",
                    "Another record is already kept under this digest.",
                )
            return
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=".target-trial-record-",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_bytes(path: Path) -> bytes:
    try:
        if path.stat().st_size > MAX_RECORD_BYTES:
            raise TargetTrialRecordError(
                "target_trial_record_invalid",
                "The kept record exceeds the size a kept record may have.",
            )
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise TargetTrialRecordError(
            "target_trial_record_missing",
            "The host keeps no record under the digest the study names.",
        ) from exc
    except OSError as exc:
        raise TargetTrialRecordError(
            "target_trial_record_unreadable",
            "The kept record cannot be read.",
        ) from exc


def load_target_trial_record(
    study_id: str, compile_sha256: str
) -> TargetTrialCompileRecord:
    """The record kept for the study under ``compile_sha256``, checked against it."""

    path = _record_path(study_id, compile_sha256)
    try:
        payload: Any = json.loads(_read_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetTrialRecordError(
            "target_trial_record_invalid",
            "The kept record is not the JSON object it was kept as.",
        ) from exc
    try:
        kept = load_target_trial_compile_record(payload)
    except TargetTrialDesignError as exc:
        raise TargetTrialRecordError(
            "target_trial_record_invalid",
            "The kept record breaks its contract.",
            field=exc.field,
        ) from exc
    if kept.compile_sha256 != compile_sha256:
        raise TargetTrialRecordError(
            "target_trial_record_invalid",
            "The kept record is not the one its file is named for.",
        )
    return kept


__all__ = [
    "MAX_RECORD_BYTES",
    "TargetTrialRecordError",
    "keep_target_trial_record",
    "load_target_trial_record",
    "records_root",
]
