"""Host-owned ResearchProject to StudyContext authority mapping.

A binding records which StudyContext one research project owns; the store
refuses a second context for a project and a second project for a context.

The file a project is read from holds at most :data:`_MAX_PROJECTS` bindings.
Projects leave the project list, and their folders move, without telling this
store, so a count of every binding ever made filled it for good and refused
every new project. A new binding that finds the file full now moves the oldest
binding to an archive file beside it; nothing is deleted. Lookups and both
refusals read the two files together, and reopening an archived project, which
binds it again, moves its binding back.
"""

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
_ARCHIVE_SCHEMA_VERSION = "easyicu.pi-project-authority-archive/1"
_MAX_PROJECTS = 200
_MAX_STORE_BYTES = 512 * 1024
# Each binding names one StudyContext, so the archive admits as many bindings
# as the StudyContext store admits contexts.
_MAX_ARCHIVED_PROJECTS = 4000
_MAX_ARCHIVE_BYTES = 4 * 1024 * 1024


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
        self.archive_path = self.path.with_name(self.path.stem + "_archive.json")
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
        return self._read_rows(
            self.path, _SCHEMA_VERSION, _MAX_PROJECTS, _MAX_STORE_BYTES
        )

    def _read_archive(self) -> list[ProjectAuthorityBinding]:
        return self._read_rows(
            self.archive_path,
            _ARCHIVE_SCHEMA_VERSION,
            _MAX_ARCHIVED_PROJECTS,
            _MAX_ARCHIVE_BYTES,
        )

    @staticmethod
    def _read_rows(
        path: Path, schema_version: str, max_rows: int, max_bytes: int
    ) -> list[ProjectAuthorityBinding]:
        try:
            if path.stat().st_size > max_bytes:
                raise PiCopilotError(
                    "pi_project_authority_store_too_large",
                    "The project authority store exceeds its bounded contract.",
                    status_code=500,
                )
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as exc:
            raise PiCopilotError(
                "pi_project_authority_store_invalid",
                "The project authority store is invalid JSON.",
                status_code=500,
            ) from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != schema_version:
            raise PiCopilotError(
                "pi_project_authority_store_invalid",
                "The project authority store has an unsupported shape.",
                status_code=500,
            )
        rows = raw.get("bindings")
        if not isinstance(rows, list) or len(rows) > max_rows:
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
        self._write_rows(self.path, _SCHEMA_VERSION, rows, _MAX_STORE_BYTES)

    def _write_archive(self, rows: list[ProjectAuthorityBinding]) -> None:
        self._write_rows(
            self.archive_path, _ARCHIVE_SCHEMA_VERSION, rows, _MAX_ARCHIVE_BYTES
        )

    @staticmethod
    def _write_rows(
        path: Path,
        schema_version: str,
        rows: list[ProjectAuthorityBinding],
        max_bytes: int,
    ) -> None:
        payload = {
            "schema_version": schema_version,
            "updated_at": utc_now(),
            "bindings": [row.model_dump(mode="json") for row in rows],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if len(text.encode("utf-8")) > max_bytes:
            # Refused before writing, so a reader never meets a file it rejects.
            raise PiCopilotError(
                "pi_project_authority_capacity_reached",
                "The bounded project authority store is full.",
                status_code=409,
            )
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=".pi-project-authority-",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            temporary.replace(path)
            try:
                path.chmod(0o600)
            except OSError:
                pass
        finally:
            temporary.unlink(missing_ok=True)

    def _known(
        self,
    ) -> tuple[list[ProjectAuthorityBinding], list[ProjectAuthorityBinding]]:
        """The active bindings, then the archived ones the active file lacks.

        A move between the files writes the receiving file first, so an
        interrupted one leaves the same binding in both. Only an identical copy
        is a leftover of that kind; two different bindings for one project, or
        two projects for one context, mean the store itself is invalid.
        """

        rows = self._read()
        active = {row.project_id: row for row in rows}
        archived: list[ProjectAuthorityBinding] = []
        for row in self._read_archive():
            copy = active.get(row.project_id)
            if copy is None:
                archived.append(row)
            elif copy != row:
                raise PiCopilotError(
                    "pi_project_authority_store_invalid",
                    "The project authority store and its archive disagree.",
                    status_code=500,
                )
        owners: dict[str, str] = {}
        for row in rows + archived:
            owner = owners.setdefault(row.study_context_id, row.project_id)
            if owner != row.project_id:
                raise PiCopilotError(
                    "pi_project_authority_store_invalid",
                    "The project authority store and its archive disagree.",
                    status_code=500,
                )
        return rows, archived

    def resolve(self, project_id: str) -> Optional[str]:
        binding = self.binding(project_id)
        return binding.study_context_id if binding else None

    def binding(self, project_id: str) -> Optional[ProjectAuthorityBinding]:
        clean_project = self._clean(project_id, code="pi_project_binding_required")

        def find(
            rows: list[ProjectAuthorityBinding],
        ) -> Optional[ProjectAuthorityBinding]:
            return next((row for row in rows if row.project_id == clean_project), None)

        with self._lock:
            # Active, archive, then active again: a binding moving either way
            # is in the file read last before it leaves the other one.
            return (
                find(self._read()) or find(self._read_archive()) or find(self._read())
            )

    def bindings(self) -> tuple[ProjectAuthorityBinding, ...]:
        """Return one immutable snapshot for read-only host composition."""
        with self._lock:
            rows, archived = self._known()
            seen = {row.project_id for row in rows + archived}
            # A binding restored while the snapshot was read is in the active
            # file by now.
            late = [row for row in self._read() if row.project_id not in seen]
            return tuple(rows + archived + late)

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
                rows, archived = self._known()
                known = rows + archived
                project_binding = next(
                    (row for row in known if row.project_id == clean_project),
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
                    if project_binding in rows:
                        return clean_study
                    # Reopening an archived project brings its binding back.
                    binding = project_binding
                else:
                    context_binding = next(
                        (row for row in known if row.study_context_id == clean_study),
                        None,
                    )
                    if context_binding:
                        raise PiCopilotError(
                            "pi_study_context_project_mismatch",
                            "This StudyContext is already owned by another research project.",
                            status_code=409,
                            details={"project_id": clean_project},
                        )
                    binding = ProjectAuthorityBinding(
                        project_id=clean_project,
                        study_context_id=clean_study,
                        migration_receipt=migration_receipt,
                    )
                rows.insert(0, binding)
                evicted = rows[_MAX_PROJECTS:]
                del rows[_MAX_PROJECTS:]
                restored = binding in archived
                if evicted:
                    if len(evicted) + len(archived) > _MAX_ARCHIVED_PROJECTS:
                        raise PiCopilotError(
                            "pi_project_authority_capacity_reached",
                            "The bounded project authority store is full.",
                            status_code=409,
                        )
                    # The oldest binding reaches the archive before it leaves
                    # the active file, so no interrupted write can drop it.
                    archived = evicted + archived
                    self._write_archive(archived)
                self._write(rows)
                if restored:
                    self._write_archive([row for row in archived if row != binding])
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
