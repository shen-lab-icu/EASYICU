"""Private restart-recovery index for Web Research Agent review pauses."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.canonical_json import canonical_sha256

from easyicu.webserver import state_paths

try:  # pragma: no branch - selected once per platform
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]

try:  # pragma: no branch - selected once per platform
    import msvcrt
except ImportError:  # pragma: no cover - unavailable on POSIX
    msvcrt = None  # type: ignore[assignment]


class WebReviewRecoveryError(RuntimeError):
    """The private Web recovery index is absent, corrupt, or drifted."""


_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: Dict[str, threading.RLock] = {}
_INDEX_SCHEMA = "easyicu.web-review-recovery-index/2"
_SEED_FILENAME = "web_review_recovery_seed.json"
_DEFAULT_MAX_ROOTS = 32
_DEFAULT_MAX_CANDIDATES = 256
_MAX_RUN_DIRS_PER_CANDIDATE = 16


def _acquire_file_lock(descriptor: int) -> None:
    if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return
    if msvcrt is None:  # pragma: no cover - supported platforms provide one
        raise WebReviewRecoveryError("No cross-process file lock is available")
    if os.fstat(descriptor).st_size == 0:
        os.write(descriptor, b"\0")
        os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)


def _release_file_lock(descriptor: int) -> None:
    if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return
    if msvcrt is not None:  # pragma: no branch - paired with acquisition
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)


def _chmod_private_fd(descriptor: int) -> None:
    try:
        os.fchmod(descriptor, 0o600)
    except (AttributeError, OSError):  # Windows has no reliable POSIX mode bits.
        pass


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
        acquired = False
        try:
            _chmod_private_fd(descriptor)
            _acquire_file_lock(descriptor)
            acquired = True
            yield
        finally:
            if acquired:
                _release_file_lock(descriptor)
            os.close(descriptor)


class WebReviewRecoveryRecord(BaseModel):
    """Non-secret coordinates needed to reconstruct one paused Web run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "easyicu.web-review-recovery/1",
        "easyicu.web-review-recovery/2",
        "easyicu.web-review-recovery/3",
        "easyicu.web-review-recovery/4",
    ] = (
        "easyicu.web-review-recovery/4"
    )
    run_id: str
    wrapper_dir: str
    study: Dict[str, Any]
    scientific_configuration_sha256: str
    provider_meta: Dict[str, Any]
    provider_public: Dict[str, Any]
    credential_source: Literal["pi_verified", "scientific_provider"]
    budget_mode: Literal["planner_canary", "full_reviewed"] = "full_reviewed"
    prepared_package_binding: Optional[Dict[str, Any]] = None
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
            self.schema_version
            in {
                "easyicu.web-review-recovery/2",
                "easyicu.web-review-recovery/3",
                "easyicu.web-review-recovery/4",
            }
            and self.pipeline_config_sha256 is None
        ):
            raise ValueError("Web review recovery requires a pipeline digest")
        if (
            self.schema_version == "easyicu.web-review-recovery/4"
            and not self.prepared_package_binding
        ):
            raise ValueError("Web review recovery v4 requires a package binding")
        payload = self.model_dump(mode="json", exclude={"record_sha256"})
        if self.schema_version == "easyicu.web-review-recovery/1":
            payload.pop("pipeline_config_sha256", None)
        if self.schema_version in {
            "easyicu.web-review-recovery/1",
            "easyicu.web-review-recovery/2",
        }:
            payload.pop("budget_mode", None)
        if self.schema_version != "easyicu.web-review-recovery/4":
            payload.pop("prepared_package_binding", None)
        if canonical_sha256(payload) != self.record_sha256:
            raise ValueError("Web review recovery digest mismatch")
        return self

    @classmethod
    def create(cls, **values: Any) -> "WebReviewRecoveryRecord":
        values = dict(values)
        values.pop("schema_version", None)
        values.setdefault("budget_mode", "full_reviewed")
        payload = {
            "schema_version": "easyicu.web-review-recovery/4",
            **values,
        }
        payload["record_sha256"] = canonical_sha256(payload)
        return cls.model_validate(payload)


class WebReviewRecoverySeed(BaseModel):
    """Digest-bound Web coordinates persisted before the Planner starts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "easyicu.web-review-recovery-seed/1",
        "easyicu.web-review-recovery-seed/2",
        "easyicu.web-review-recovery-seed/3",
    ] = (
        "easyicu.web-review-recovery-seed/3"
    )
    wrapper_dir: str
    study: Dict[str, Any]
    scientific_configuration_sha256: str
    provider_meta: Dict[str, Any]
    provider_public: Dict[str, Any]
    credential_source: Literal["pi_verified", "scientific_provider"]
    budget_mode: Literal["planner_canary", "full_reviewed"] = "full_reviewed"
    prepared_package_binding: Optional[Dict[str, Any]] = None
    pipeline_config: Dict[str, Any]
    pipeline_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    acquisition_projection: Dict[str, Any]
    hard_stop_ledger_path: str
    hard_stop_task_id: str
    hard_stop_declaration_sha256: str
    created_at: float = Field(ge=0.0)
    seed_sha256: str

    @model_validator(mode="after")
    def _verify_digest(self) -> "WebReviewRecoverySeed":
        payload = self.model_dump(mode="json", exclude={"seed_sha256"})
        if (
            self.schema_version == "easyicu.web-review-recovery-seed/3"
            and not self.prepared_package_binding
        ):
            raise ValueError("Web review recovery seed v3 requires a package binding")
        if self.schema_version == "easyicu.web-review-recovery-seed/1":
            payload.pop("budget_mode", None)
        if self.schema_version != "easyicu.web-review-recovery-seed/3":
            payload.pop("prepared_package_binding", None)
        if canonical_sha256(payload) != self.seed_sha256:
            raise ValueError("Web review recovery seed digest mismatch")
        return self

    @classmethod
    def create(cls, **values: Any) -> "WebReviewRecoverySeed":
        values = dict(values)
        values.pop("schema_version", None)
        values.setdefault("budget_mode", "full_reviewed")
        payload = {
            "schema_version": "easyicu.web-review-recovery-seed/3",
            **values,
        }
        payload["seed_sha256"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    def record(self, run_id: str) -> WebReviewRecoveryRecord:
        values = self.model_dump(
            mode="json",
            exclude={"schema_version", "seed_sha256"},
        )
        return WebReviewRecoveryRecord.create(run_id=run_id, **values)


def default_store_path() -> Path:
    return state_paths.state_root() / "web_review_recovery.json"


def _read(path: Path) -> Dict[str, Any]:
    if path.is_symlink():
        raise WebReviewRecoveryError("Web review recovery index cannot be a symlink")
    if not path.exists():
        return {"schema_version": _INDEX_SCHEMA, "records": {}, "work_roots": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise WebReviewRecoveryError("Web review recovery index is corrupt") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        not in {"easyicu.web-review-recovery-index/1", _INDEX_SCHEMA}
        or not isinstance(payload.get("records"), dict)
        or (
            "work_roots" in payload
            and not isinstance(payload.get("work_roots"), list)
        )
    ):
        raise WebReviewRecoveryError("Web review recovery index has an invalid schema")
    return {
        "schema_version": _INDEX_SCHEMA,
        "records": payload["records"],
        "work_roots": payload.get("work_roots") or [],
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp_name = tempfile.mkstemp(prefix=".web-review-", dir=str(path.parent))
    try:
        _chmod_private_fd(fd)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
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


def register_pipeline_work_root(
    root: Path,
    *,
    path: Optional[Path] = None,
    max_roots: int = _DEFAULT_MAX_ROOTS,
) -> Path:
    """Persist one host-selected project root before pipeline work begins."""

    selected_root = Path(root).expanduser().resolve()
    if Path(root).expanduser().is_symlink():
        raise WebReviewRecoveryError("Web review pipeline work root cannot be a symlink")
    selected = path or default_store_path()
    with _locked(selected):
        payload = _read(selected)
        stored_roots = [
            str(Path(value).expanduser().resolve()) for value in payload["work_roots"]
        ]
        temporary_root = Path(tempfile.gettempdir()).resolve()
        roots = []
        for value in stored_roots:
            candidate = Path(value)
            try:
                candidate.relative_to(temporary_root)
                temporary = True
            except ValueError:
                temporary = False
            if not temporary or candidate.is_dir():
                roots.append(value)
        rendered = str(selected_root)
        if rendered not in roots:
            if len(roots) >= max_roots:
                raise WebReviewRecoveryError(
                    "Web review pipeline work-root capacity is full"
                )
            roots.append(rendered)
        if roots != stored_roots:
            _write(selected, {**payload, "work_roots": roots})
    return selected_root


def unregister_pipeline_work_root_if_unused(
    root: Path,
    *,
    path: Optional[Path] = None,
) -> None:
    """Forget a root only after its last local recovery seed is gone."""

    selected_root = Path(root).expanduser().resolve()
    try:
        next(selected_root.glob(f"*/run_*/.runtime/{_SEED_FILENAME}"))
    except (StopIteration, OSError):
        pass
    else:
        return
    selected = path or default_store_path()
    with _locked(selected):
        payload = _read(selected)
        rendered = str(selected_root)
        roots = [value for value in payload["work_roots"] if str(value) != rendered]
        if roots != payload["work_roots"]:
            _write(selected, {**payload, "work_roots": roots})


def recovery_seed_path(wrapper_dir: Path) -> Path:
    return Path(wrapper_dir) / ".runtime" / _SEED_FILENAME


def put_recovery_seed(seed: WebReviewRecoverySeed) -> Path:
    wrapper_dir = Path(seed.wrapper_dir).expanduser().resolve()
    path = recovery_seed_path(wrapper_dir)
    if path.is_symlink():
        raise WebReviewRecoveryError("Web review recovery seed cannot be a symlink")
    _write(path, seed.model_dump(mode="json"))
    return path


def remove_recovery_seed(wrapper_dir: Path) -> None:
    path = recovery_seed_path(wrapper_dir)
    if path.is_symlink():
        raise WebReviewRecoveryError("Web review recovery seed cannot be a symlink")
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _record_from_seed_path(path: Path, *, root: Path) -> list[WebReviewRecoveryRecord]:
    """Validate a local seed and existing checkpoints without running pipeline code."""

    if path.is_symlink() or path.parent.is_symlink():
        return []
    wrapper_dir = path.parent.parent.resolve()
    try:
        wrapper_dir.relative_to(root)
    except ValueError:
        return []
    if wrapper_dir.is_symlink() or wrapper_dir.parent.is_symlink():
        return []
    try:
        seed = WebReviewRecoverySeed.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if Path(seed.wrapper_dir).expanduser().resolve() != wrapper_dir:
        return []

    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.human_review_checkpoint import (
        load_checkpoint,
    )

    try:
        config = PipelineConfig.from_recovery_payload(
            seed.pipeline_config,
            expected_digest=seed.pipeline_config_sha256,
        )
    except ValueError:
        return []
    pipeline_root = (wrapper_dir / "pipeline").resolve()
    if Path(config.workdir).resolve() != pipeline_root or pipeline_root.is_symlink():
        return []

    records: list[WebReviewRecoveryRecord] = []
    run_dirs: list[Path] = []
    try:
        # ``Path.iterdir`` is lazy: creating the iterator succeeds even when a
        # stale recovery seed points at a directory that has since disappeared.
        # Materialize only the reviewed bound inside the exception boundary so
        # one obsolete test/run seed cannot prevent the whole Web app starting.
        for index, run_dir in enumerate(pipeline_root.iterdir()):
            if index >= _MAX_RUN_DIRS_PER_CANDIDATE:
                break
            run_dirs.append(run_dir)
    except OSError:
        return []
    for run_dir in run_dirs:
        if len(records) >= 4:
            break
        if not run_dir.is_dir() or run_dir.is_symlink():
            continue
        checkpoint_path = run_dir / "human_review_checkpoint.json"
        if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
            continue
        try:
            checkpoint = load_checkpoint(checkpoint_path, require_pending=False)
        except Exception:
            continue
        if (
            checkpoint.run_id != run_dir.name
            or checkpoint.pipeline_config_sha256 != seed.pipeline_config_sha256
            or checkpoint.state
            not in {"pending", "approved_pending_execution", "executing"}
        ):
            continue
        records.append(seed.record(checkpoint.run_id))
    return records


def _bounded_seed_paths(root: Path, *, limit: int) -> tuple[list[Path], int]:
    """Inspect at most ``limit`` studies and wrapper directories in one root."""

    paths: list[Path] = []
    wrappers_inspected = 0
    try:
        studies = root.iterdir()
    except OSError:
        return paths, wrappers_inspected
    for study_index, study_dir in enumerate(studies):
        if study_index >= limit:
            break
        if not study_dir.is_dir() or study_dir.is_symlink():
            continue
        try:
            wrappers = study_dir.iterdir()
        except OSError:
            continue
        for wrapper_dir in wrappers:
            if wrappers_inspected >= limit:
                return paths, wrappers_inspected
            wrappers_inspected += 1
            if (
                not wrapper_dir.name.startswith("run_")
                or not wrapper_dir.is_dir()
                or wrapper_dir.is_symlink()
            ):
                continue
            seed_path = recovery_seed_path(wrapper_dir)
            if seed_path.is_file() and not seed_path.is_symlink():
                paths.append(seed_path)
    return paths, wrappers_inspected


def reconcile_records(
    *,
    path: Optional[Path] = None,
    max_records: int = 128,
    max_roots: int = _DEFAULT_MAX_ROOTS,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
) -> int:
    """Import exact durable checkpoints from configured roots, within hard bounds."""

    selected = path or default_store_path()
    with _locked(selected):
        payload = _read(selected)
        configured = list(payload["work_roots"])[:max_roots]
        records = dict(payload["records"])
        imported = 0
        inspected = 0
        for raw_root in configured:
            root_path = Path(str(raw_root)).expanduser()
            if root_path.is_symlink():
                continue
            root = root_path.resolve()
            if not root.is_dir():
                continue
            remaining = max_candidates - inspected
            seed_paths, root_inspected = _bounded_seed_paths(root, limit=remaining)
            inspected += root_inspected
            for seed_path in seed_paths:
                for record in _record_from_seed_path(seed_path, root=root):
                    if record.run_id in records:
                        continue
                    if len(records) >= max_records:
                        break
                    records[record.run_id] = record.model_dump(mode="json")
                    imported += 1
            if inspected >= max_candidates or len(records) >= max_records:
                break
        if imported:
            _write(selected, {**payload, "records": records})
        return imported


def get_record(run_id: str, *, path: Optional[Path] = None) -> Optional[WebReviewRecoveryRecord]:
    selected = path or default_store_path()
    with _locked(selected):
        payload = _read(selected)
        raw = payload["records"].get(str(run_id))
    if raw is None:
        reconcile_records(path=selected)
        with _locked(selected):
            raw = _read(selected)["records"].get(str(run_id))
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
