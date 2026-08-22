"""Host-owned ResearchProject to StudyContext authority mapping."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .contracts import PiCopilotError, utc_now
from .locking import exclusive_file_lock

from easyicu.webserver import state_paths

_SCHEMA_VERSION = "easyicu.pi-project-authority/1"
_MAX_PROJECTS = 200


class ProjectStudyContextMigrationReceipt(BaseModel):
    """Stable receipt for the Host-owned project initialization boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.project-studycontext-migration/1"] = (
        "easyicu.project-studycontext-migration/1"
    )
    status: Literal["migrated", "initialized"]
    source_schema: str = Field(min_length=1, max_length=160)
    source_digest: str = Field(min_length=64, max_length=64)
    migrated_fields: list[str] = Field(default_factory=list, max_length=16)
    created_at: str = Field(default_factory=utc_now)


class ProjectAuthorityBinding(BaseModel):
    """One immutable scientific namespace owned by one research project."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(min_length=1, max_length=160)
    study_context_id: str = Field(min_length=1, max_length=160)
    migration_receipt: Optional[ProjectStudyContextMigrationReceipt] = None
    created_at: str = Field(default_factory=utc_now)


class ProjectAuthorityStore:
    """Persist and enforce the one-project/one-StudyContext relationship."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or state_paths.state_root() / "pi_project_authority.json")
        self._lock = threading.RLock()

    @staticmethod
    def _clean(value: str, *, code: str) -> str:
        clean = str(value or "").strip()
        if not clean or len(clean) > 160:
            raise PiCopilotError(
                code, "Project authority identifiers must be 1-160 characters."
            )
        return clean

    def _read(self) -> list[ProjectAuthorityBinding]:
        try:
            if self.path.stat().st_size > 512 * 1024:
                raise PiCopilotError(
                    "pi_project_authority_store_too_large",
                    "The project authority store exceeds its bounded contract.",
                    status_code=500,
                )
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as exc:
            raise PiCopilotError(
                "pi_project_authority_store_invalid",
                "The project authority store is invalid JSON.",
                status_code=500,
            ) from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != _SCHEMA_VERSION:
            raise PiCopilotError(
                "pi_project_authority_store_invalid",
                "The project authority store has an unsupported shape.",
                status_code=500,
            )
        rows = raw.get("bindings")
        if not isinstance(rows, list) or len(rows) > _MAX_PROJECTS:
            raise PiCopilotError(
                "pi_project_authority_store_invalid",
                "The project authority store has invalid bindings.",
                status_code=500,
            )
        try:
            return [ProjectAuthorityBinding.model_validate(row) for row in rows]
        except Exception as exc:
            raise PiCopilotError(
                "pi_project_authority_store_invalid",
                "The project authority store contains an invalid binding.",
                status_code=500,
            ) from exc

    def _write(self, rows: list[ProjectAuthorityBinding]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "updated_at": utc_now(),
            "bindings": [row.model_dump(mode="json") for row in rows],
        }
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.path.parent),
            prefix=".pi-project-authority-",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            temporary.replace(self.path)
            try:
                self.path.chmod(0o600)
            except OSError:
                pass
        finally:
            temporary.unlink(missing_ok=True)

    def resolve(self, project_id: str) -> Optional[str]:
        clean_project = self._clean(project_id, code="pi_project_binding_required")
        with self._lock:
            binding = next(
                (row for row in self._read() if row.project_id == clean_project),
                None,
            )
        return binding.study_context_id if binding else None

    def binding(self, project_id: str) -> Optional[ProjectAuthorityBinding]:
        clean_project = self._clean(project_id, code="pi_project_binding_required")
        with self._lock:
            return next(
                (row for row in self._read() if row.project_id == clean_project),
                None,
            )

    def bindings(self) -> tuple[ProjectAuthorityBinding, ...]:
        """Return one immutable snapshot for read-only host composition."""
        with self._lock:
            return tuple(self._read())

    def bind(
        self,
        project_id: str,
        study_context_id: str,
        *,
        migration_receipt: Optional[ProjectStudyContextMigrationReceipt] = None,
    ) -> str:
        clean_project = self._clean(project_id, code="pi_project_binding_required")
        clean_study = self._clean(
            study_context_id,
            code="pi_project_study_context_binding_required",
        )
        with self._lock:
            with exclusive_file_lock(
                self.path.with_name(self.path.name + ".lock"),
                code="pi_project_authority_lock_unavailable",
            ):
                rows = self._read()
                project_binding = next(
                    (row for row in rows if row.project_id == clean_project),
                    None,
                )
                if project_binding:
                    if project_binding.study_context_id != clean_study:
                        raise PiCopilotError(
                            "pi_project_study_context_mismatch",
                            "This research project is already bound to another StudyContext.",
                            status_code=409,
                            details={"project_id": clean_project},
                        )
                    return clean_study
                context_binding = next(
                    (row for row in rows if row.study_context_id == clean_study),
                    None,
                )
                if context_binding:
                    raise PiCopilotError(
                        "pi_study_context_project_mismatch",
                        "This StudyContext is already owned by another research project.",
                        status_code=409,
                        details={"project_id": clean_project},
                    )
                if len(rows) >= _MAX_PROJECTS:
                    raise PiCopilotError(
                        "pi_project_authority_capacity_reached",
                        "The bounded project authority store is full.",
                        status_code=409,
                    )
                rows.insert(
                    0,
                    ProjectAuthorityBinding(
                        project_id=clean_project,
                        study_context_id=clean_study,
                        migration_receipt=migration_receipt,
                    ),
                )
                self._write(rows)
        return clean_study

    def assert_matches(self, project_id: str, study_context_id: Optional[str]) -> str:
        clean_project = self._clean(project_id, code="pi_project_binding_required")
        mapped = self.resolve(clean_project)
        if mapped is None:
            raise PiCopilotError(
                "pi_project_initialization_required",
                "The research project has no authoritative StudyContext binding.",
                status_code=409,
                details={"project_id": clean_project},
            )
        if mapped != str(study_context_id or "").strip():
            raise PiCopilotError(
                "pi_session_project_authority_mismatch",
                "The Copilot session StudyContext does not belong to this research project.",
                status_code=409,
                details={"project_id": clean_project},
            )
        return mapped


__all__ = [
    "ProjectAuthorityBinding",
    "ProjectAuthorityStore",
    "ProjectStudyContextMigrationReceipt",
]
