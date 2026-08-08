"""Authoritative FastAPI-side owner for Pi Copilot sessions and turns."""

from __future__ import annotations

import json
import hashlib
import os
import secrets
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from easyicu.webserver import (
    agent_runs,
    guided_sessions,
    jobs,
    provider_gate,
    settings,
    study_contexts,
)

from .contracts import (
    MAX_MESSAGE_CHARS,
    AuthorityBinding,
    PiCopilotError,
    PiSessionRecord,
    ToolExecutionContext,
    utc_now,
)
from .gateway import PiGatewayClient
from .project_authority import (
    ProjectAuthorityStore,
    ProjectStudyContextMigrationReceipt,
)
from .provider_config import PiProviderConfigStore
from .projections import reject_sensitive_message
from .workspace import ProjectWorkspace

MAX_SESSIONS = 100
ALLOWED_TURN_ACTIONS = frozenset(
    {"configure", "run", "cancel", "workspace_write"}
)


class PiCopilotService:
    """Bind Pi UX sessions to current EasyICU scientific authority."""

    def __init__(
        self,
        *,
        store_path: Optional[Path] = None,
        gateway: Optional[PiGatewayClient] = None,
        provider_store: Optional[PiProviderConfigStore] = None,
        project_store: Optional[ProjectAuthorityStore] = None,
    ) -> None:
        self.store_path = (
            Path(store_path)
            if store_path is not None
            else Path.home() / ".easyicu" / "pi_copilot_sessions.json"
        )
        self.gateway = gateway or PiGatewayClient()
        self.provider_store = (
            provider_store
            or getattr(self.gateway, "provider_store", None)
            or PiProviderConfigStore()
        )
        self.project_store = project_store or ProjectAuthorityStore(
            None
            if store_path is None
            else self.store_path.with_name(f"{self.store_path.stem}.projects.json")
        )
        self._lock = threading.RLock()
        self._active_message_jobs: Dict[str, str] = {}
        self._busy_sessions: set[str] = set()
        self._pending_retirements: Dict[str, PiSessionRecord] = {}
        self._project_initialization_locks: Dict[str, threading.RLock] = {}
        workspace_base = Path(
            getattr(
                self.gateway,
                "cwd",
                self.store_path.parent / "pi_project_workspace",
            )
        )
        self.workspace_root = workspace_base.resolve()
        self.workspace = ProjectWorkspace(self.workspace_root)

    def _project_initialization_lock(self, project_id: str) -> threading.RLock:
        with self._lock:
            return self._project_initialization_locks.setdefault(
                project_id,
                threading.RLock(),
            )

    def _read_records(self) -> list[PiSessionRecord]:
        try:
            if self.store_path.stat().st_size > 2 * 1024 * 1024:
                raise PiCopilotError(
                    "pi_session_store_too_large",
                    "The Pi Copilot metadata store exceeds its bounded contract.",
                    status_code=500,
                )
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as exc:
            raise PiCopilotError(
                "pi_session_store_invalid",
                "The Pi Copilot metadata store is invalid JSON.",
                status_code=500,
            ) from exc
        rows = raw.get("sessions") if isinstance(raw, dict) else None
        if not isinstance(rows, list):
            raise PiCopilotError(
                "pi_session_store_invalid",
                "The Pi Copilot metadata store has an invalid shape.",
                status_code=500,
            )
        records = []
        for row in rows[:MAX_SESSIONS]:
            try:
                records.append(PiSessionRecord.model_validate(row))
            except Exception as exc:
                raise PiCopilotError(
                    "pi_session_store_invalid",
                    "The Pi Copilot metadata store contains an invalid session.",
                    status_code=500,
                ) from exc
        return records

    def _write_records(self, records: Iterable[PiSessionRecord]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "schema_version": "easyicu.pi-copilot-store/1",
            "updated_at": utc_now(),
            "sessions": [
                row.model_dump(mode="json") for row in list(records)[:MAX_SESSIONS]
            ],
        }
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.store_path.parent),
            prefix=".pi-copilot-",
            suffix=".tmp",
            delete=False,
        )
        tmp = Path(handle.name)
        try:
            with handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.chmod(0o600)
            tmp.replace(self.store_path)
            try:
                self.store_path.chmod(0o600)
            except OSError:
                pass
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _get_record(self, session_id: str) -> PiSessionRecord:
        clean = str(session_id or "").strip()
        with self._lock:
            record = next(
                (row for row in self._read_records() if row.session_id == clean),
                None,
            )
        if record is None:
            raise PiCopilotError(
                "pi_session_not_found",
                "The requested Pi Copilot session does not exist.",
                status_code=404,
            )
        return record

    def _save_record(self, record: PiSessionRecord) -> PiSessionRecord:
        record.updated_at = utc_now()
        with self._lock:
            rows = [
                row
                for row in self._read_records()
                if row.session_id != record.session_id
            ]
            rows.insert(0, record)
            overflow = max(0, len(rows) - MAX_SESSIONS)
            evicted: list[PiSessionRecord] = []
            if overflow:
                for candidate in reversed(rows):
                    if candidate.session_id in self._busy_sessions:
                        continue
                    evicted.append(candidate)
                    if len(evicted) == overflow:
                        break
                if len(evicted) != overflow:
                    raise PiCopilotError(
                        "pi_session_retention_busy",
                        "Pi session retention cannot evict an active conversation.",
                        status_code=409,
                    )
                evicted_ids = {row.session_id for row in evicted}
                rows = [row for row in rows if row.session_id not in evicted_ids]
            self._write_records(rows)
        for retired in evicted:
            self._retire_record(retired)
        return record

    def _retire_record(self, record: PiSessionRecord) -> None:
        """Dispose an evicted Pi session and remove only its private JSONL."""

        with self._lock:
            if record.session_id in self._busy_sessions:
                self._pending_retirements[record.session_id] = record
                return
        try:
            self.gateway.request(
                "session.dispose",
                {"session_id": record.session_id},
                timeout=5,
            )
        except (OSError, PiCopilotError):
            pass
        session_root = getattr(self.gateway, "session_dir", None)
        if session_root is None or not record.pi_session_file:
            return
        candidate = Path(record.pi_session_file).resolve()
        root = Path(session_root).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return
        if candidate.suffix != ".jsonl" or not candidate.is_file():
            return
        try:
            candidate.unlink()
        except OSError:
            pass

    def _flush_pending_retirement(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._busy_sessions:
                return
            record = self._pending_retirements.pop(session_id, None)
        if record is not None:
            self._retire_record(record)

    def _provider_gate(self, *, external_llm_opt_in: bool) -> Dict[str, Any]:
        current_settings = settings.load_settings()
        provider = str(
            getattr(self.gateway, "environ", {}).get("EASYICU_PI_PROVIDER")
            or "easyicu-local"
        )
        try:
            return provider_gate.resolve_provider_gate(
                run_type="full",
                llm_provider=provider,
                external_llm_opt_in=bool(external_llm_opt_in),
                ai_enabled=bool(current_settings.get("ai_enabled")),
                language=str(current_settings.get("language") or "en"),
            )
        except provider_gate.ProviderGateError as exc:
            raise PiCopilotError(
                str(exc.detail.get("error") or "external_llm_opt_in_required"),
                (
                    "Pi Copilot is an external shell-model call. Enable AI in "
                    "EasyICU Settings and explicitly opt in for this session."
                ),
                status_code=403,
                details={
                    key: exc.detail.get(key)
                    for key in (
                        "blocked_by",
                        "canonical_opt_in_passed",
                        "per_run_opt_in",
                    )
                    if exc.detail.get(key) is not None
                },
            ) from exc

    def runtime_status(self) -> Dict[str, Any]:
        install = self.gateway.installation_status()
        current_settings = settings.load_settings()
        blockers = []
        for key in (
            "node_available",
            "node_version_supported",
            "entrypoint_available",
            "dependency_installed",
            "lockfile_present",
            "runtime_integrity_verified",
            "api_key_configured",
            "base_url_configured",
        ):
            if not install.get(key):
                blockers.append(key)
        if install.get("api_key_configured") and not install.get(
            "provider_connection_verified"
        ):
            blockers.append("provider_connection_unverified")
        if not current_settings.get("ai_enabled"):
            blockers.append("easyicu_ai_opt_in_disabled")
        runtime_blockers = {
            "node_available",
            "node_version_supported",
            "entrypoint_available",
            "dependency_installed",
            "lockfile_present",
            "runtime_integrity_verified",
            "base_url_configured",
        }
        if not blockers:
            runtime_state = "ready"
        elif runtime_blockers.intersection(blockers):
            runtime_state = "unavailable"
        else:
            runtime_state = "setup_required"
        return {
            "ok": True,
            "runtime": {
                "status": runtime_state,
                "blockers": blockers,
                "pi_package_version": "0.84.1",
                "pi_source_commit": "9dd90a49711d088b86fdd9b4aea575913a8328a8",
                "provider": install["provider"],
                "model": install["model"],
                "api_transport": install["api_transport"],
                "local_openai_compatible_default": (
                    install["api_transport"]
                    in {"openai-completions", "openai-responses"}
                ),
                "built_in_tools_enabled": [],
                "agent_modes": ["research", "workspace"],
                "workspace_scope": "isolated_project_artifacts",
                "credential_values_exposed": False,
                "configuration": dict(install.get("provider_configuration") or {}),
            },
        }

    def configure_provider(
        self,
        *,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        api_transport: str,
        enable_ai: bool,
    ) -> Dict[str, Any]:
        """Verify and persist first-use Pi credentials before chat unlocks."""

        if not enable_ai:
            raise PiCopilotError(
                "external_llm_opt_in_required",
                "Confirm external AI use before verifying the model service.",
                status_code=403,
            )
        with self._lock:
            if self._busy_sessions:
                raise PiCopilotError(
                    "pi_provider_config_busy",
                    "Stop the active Pi response before changing model service settings.",
                    status_code=409,
                )
        apply_config = getattr(self.gateway, "apply_provider_config", None)
        if not callable(apply_config):
            raise PiCopilotError(
                "pi_provider_gateway_reconfigure_unsupported",
                "The Pi gateway cannot apply the verified configuration.",
                status_code=500,
            )
        # The explicit request is the canonical opt-in for this verification
        # call. Persist that choice before any network access.
        settings.update_settings({"ai_enabled": True})
        config, configuration = self.provider_store.verify_and_save(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            api_transport=api_transport,
        )
        apply_config(config)
        payload = self.runtime_status()
        return {
            "ok": True,
            "runtime": payload["runtime"],
            "configuration": configuration,
            "secrets_returned": False,
        }

    @staticmethod
    def _binding_for_context(
        context: Optional[Mapping[str, Any]], *, run_id: Optional[str] = None
    ) -> AuthorityBinding:
        if not context:
            return AuthorityBinding(run_id=run_id)
        return AuthorityBinding(
            study_context_id=str(context.get("id") or "") or None,
            study_revision=int(context.get("revision") or 0),
            run_id=run_id,
            active_job_id=(
                str(context.get("active_job_id"))
                if context.get("active_job_id")
                else None
            ),
        )

    @staticmethod
    def _latest_run_id(study_context_id: Optional[str]) -> Optional[str]:
        if not study_context_id:
            return None
        history = agent_runs.list_run_history(study_id=study_context_id, limit=1)
        rows = history.get("runs") or []
        if rows and isinstance(rows[0], dict):
            return str(rows[0].get("run_id") or "") or None
        return None

    def _resolve_project_context(
        self,
        *,
        project_id: str,
        title: str,
        requested_study_context_id: Optional[str] = None,
        confirm_initialization: bool = False,
    ) -> Dict[str, Any]:
        """Resolve only the Host-owned StudyContext for one research project."""

        mapped_id = self.project_store.resolve(project_id)
        context_id = mapped_id or str(requested_study_context_id or "").strip()
        context: Optional[Dict[str, Any]] = None
        if context_id:
            try:
                context = study_contexts.get_context(context_id)
            except study_contexts.StudyContextError as exc:
                raise PiCopilotError(
                    str(exc.detail.get("error") or "study_context_invalid"),
                    "The project's StudyContext could not be loaded.",
                    status_code=409,
                    details=exc.detail,
                ) from exc
            if context is None:
                raise PiCopilotError(
                    "pi_project_study_context_missing",
                    "The research project's authoritative StudyContext no longer exists.",
                    status_code=409,
                    details={"project_id": project_id},
                )
        else:
            setup = guided_sessions.read_project_study_setup(project_id)
            if setup is not None and not setup.missing_required:
                initial = setup.study_context_patch()
                receipt = ProjectStudyContextMigrationReceipt(
                    status="migrated",
                    source_schema=setup.schema_version,
                    source_digest=setup.source_digest,
                    migrated_fields=setup.migrated_fields,
                )
            elif not confirm_initialization:
                missing = (
                    list(setup.missing_required)
                    if setup is not None
                    else ["question", "cohort", "modules", "outcome", "time_window"]
                )
                raise PiCopilotError(
                    "pi_project_initialization_required",
                    "Confirm a new Pi study setup before opening this project.",
                    status_code=409,
                    details={
                        "project_id": project_id,
                        "missing_required": missing,
                        "saved_metadata_found": setup is not None,
                    },
                )
            else:
                initial = {
                    "title": str(title or "Pi Copilot").strip()[:160] or "Pi Copilot",
                    "current_stage": "study_setup",
                    "last_route": "guided",
                }
                source = json.dumps(
                    {
                        "schema_version": "easyicu.pi-project-initialization/1",
                        "project_id": project_id,
                        "title": initial["title"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                receipt = ProjectStudyContextMigrationReceipt(
                    status="initialized",
                    source_schema="easyicu.pi-project-initialization/1",
                    source_digest=hashlib.sha256(source).hexdigest(),
                    migrated_fields=["title"],
                )
            try:
                context = study_contexts.upsert_context(
                    initial,
                    active=True,
                    lifecycle_write=True,
                )
            except study_contexts.StudyContextError as exc:
                raise PiCopilotError(
                    str(exc.detail.get("error") or "study_context_invalid"),
                    "EasyICU could not initialize this project's StudyContext.",
                    status_code=409,
                    details=exc.detail,
                ) from exc
        assert context is not None and context.get("id")
        if mapped_id:
            self.project_store.bind(project_id, str(context["id"]))
        elif requested_study_context_id:
            requested_source = hashlib.sha256(
                str(requested_study_context_id).encode("utf-8")
            ).hexdigest()
            self.project_store.bind(
                project_id,
                str(context["id"]),
                migration_receipt=ProjectStudyContextMigrationReceipt(
                    status="initialized",
                    source_schema="easyicu.explicit-studycontext-binding/1",
                    source_digest=requested_source,
                    migrated_fields=["study_context_id"],
                ),
            )
        else:
            self.project_store.bind(
                project_id,
                str(context["id"]),
                migration_receipt=receipt,
            )
        return context

    def initialize_project(
        self,
        *,
        project_id: str,
        title: str,
        confirm_initialization: bool = False,
    ) -> Dict[str, Any]:
        """Explicitly migrate/bind Guided metadata before any session GET."""

        clean_project = str(project_id or "").strip()
        if not clean_project or len(clean_project) > 160:
            raise PiCopilotError(
                "pi_project_binding_required",
                "A research project is required for Pi initialization.",
                status_code=409,
            )
        with self._project_initialization_lock(clean_project):
            return self._initialize_project_locked(
                project_id=clean_project,
                title=title,
                confirm_initialization=confirm_initialization,
            )

    def _initialize_project_locked(
        self,
        *,
        project_id: str,
        title: str,
        confirm_initialization: bool,
    ) -> Dict[str, Any]:
        """Run resolve/create/bind as one per-project initialization transaction."""

        clean_project = project_id
        with self._lock:
            existing_rows = [
                row for row in self._read_records() if row.project_id == clean_project
            ]
        legacy_context_ids = {
            str(row.binding.study_context_id)
            for row in existing_rows
            if row.binding.study_context_id
        }
        if len(legacy_context_ids) > 1:
            raise PiCopilotError(
                "pi_project_study_context_mismatch",
                "Legacy Pi sessions disagree about this project's StudyContext.",
                status_code=409,
                details={"project_id": clean_project},
            )
        requested_context_id = next(iter(legacy_context_ids), None)
        context = self._resolve_project_context(
            project_id=clean_project,
            title=title,
            requested_study_context_id=requested_context_id,
            confirm_initialization=confirm_initialization,
        )
        with self._lock:
            rows = self._read_records()
            migrated = 0
            for record in rows:
                if record.project_id != clean_project:
                    continue
                if record.binding.study_context_id not in (
                    None,
                    str(context["id"]),
                ):
                    raise PiCopilotError(
                        "pi_session_project_authority_mismatch",
                        "A legacy Pi session belongs to another StudyContext.",
                        status_code=409,
                        details={"project_id": clean_project},
                    )
                record.binding = self._binding_for_context(
                    context,
                    run_id=self._latest_run_id(str(context["id"])),
                )
                record.updated_at = utc_now()
                migrated += 1
            if migrated:
                self._write_records(rows)
        binding = self.project_store.binding(clean_project)
        return {
            "ok": True,
            "status": "ready",
            "project_id": clean_project,
            "study_context_id": str(context["id"]),
            "migrated_sessions": migrated,
            "migration_receipt": (
                binding.migration_receipt.model_dump(mode="json")
                if binding and binding.migration_receipt
                else None
            ),
        }

    def _scoped_record(self, session_id: str, *, project_id: str) -> PiSessionRecord:
        clean_project = str(project_id or "").strip()
        if not clean_project:
            raise PiCopilotError(
                "pi_project_binding_required",
                "A research project is required for this Pi session operation.",
                status_code=409,
            )
        record = self._get_record(session_id)
        if record.project_id != clean_project:
            raise PiCopilotError(
                "pi_session_project_mismatch",
                "This Pi conversation belongs to a different EasyICU research project.",
                status_code=409,
                details={
                    "session_id": record.session_id,
                    "requested_project_id": clean_project,
                },
            )
        if not record.binding.study_context_id:
            raise PiCopilotError(
                "pi_project_initialization_required",
                "Initialize this project's StudyContext before reading its Pi sessions.",
                status_code=409,
                details={"project_id": clean_project},
            )
        self.project_store.assert_matches(
            clean_project,
            record.binding.study_context_id,
        )
        return record

    def create_session(
        self,
        *,
        project_id: str,
        title: str = "Pi Copilot",
        agent_mode: str = "research",
        language: str = "en",
        thinking_level: str = "off",
        study_context_id: Optional[str] = None,
        external_llm_opt_in: bool = False,
    ) -> Dict[str, Any]:
        clean_project_id = str(project_id or "").strip()
        if not clean_project_id:
            raise PiCopilotError(
                "pi_project_binding_required",
                "Select or create an EasyICU research project before starting a Pi conversation.",
                status_code=409,
            )
        self._provider_gate(external_llm_opt_in=external_llm_opt_in)
        resolved_mode = "workspace" if agent_mode == "workspace" else "research"
        install = self.gateway.installation_status()
        if not install.get("api_key_configured"):
            raise PiCopilotError(
                "pi_api_key_missing",
                "Set EASYICU_PI_API_KEY in the WebApp process environment.",
                status_code=503,
            )
        if not install.get("provider_connection_verified"):
            raise PiCopilotError(
                "pi_provider_setup_required",
                "Verify the Pi model service before starting a conversation.",
                status_code=503,
            )
        context = self._resolve_project_context(
            project_id=clean_project_id,
            title=title,
            requested_study_context_id=study_context_id,
            confirm_initialization=True,
        )
        resolved_language = "zh" if language == "zh" else "en"
        # Raw provider reasoning is neither streamed nor persisted by the
        # governed product shell. Historical values remain readable only for
        # metadata compatibility; all newly opened sessions run with it off.
        resolved_thinking = "off"
        session_id = f"pi_{secrets.token_hex(10)}"
        binding = self._binding_for_context(
            context,
            run_id=self._latest_run_id(str(context.get("id")) if context else None),
        )
        state = self.gateway.request(
            "session.create",
            {
                "session_id": session_id,
                "thinking_level": resolved_thinking,
                "agent_mode": resolved_mode,
            },
            timeout=30,
        )
        record = PiSessionRecord(
            session_id=session_id,
            project_id=clean_project_id,
            pi_session_id=str(state.get("pi_session_id") or "") or None,
            pi_session_file=str(state.get("session_file") or "") or None,
            title=str(title or "Pi Copilot").strip()[:160] or "Pi Copilot",
            agent_mode=resolved_mode,
            language=resolved_language,
            thinking_level=resolved_thinking,
            external_llm_opt_in=True,
            binding=binding,
        )
        self._save_record(record)
        return {
            "ok": True,
            "session": self._public_session(record, gateway_state=state),
        }

    def _ensure_open(self, record: PiSessionRecord) -> Dict[str, Any]:
        try:
            return self.gateway.request(
                "session.state",
                {"session_id": record.session_id},
                timeout=5,
            )
        except PiCopilotError as exc:
            if exc.code != "pi_session_not_open":
                raise
        return self.gateway.request(
            "session.create",
            {
                "session_id": record.session_id,
                "session_file": record.pi_session_file,
                "thinking_level": "off",
                "agent_mode": record.agent_mode,
            },
            timeout=30,
        )

    def _binding_stale_details(
        self,
        binding: AuthorityBinding,
        *,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if project_id:
            try:
                self.project_store.assert_matches(
                    project_id,
                    binding.study_context_id,
                )
            except PiCopilotError as exc:
                return {
                    "stale": True,
                    "reason": exc.code,
                    "project_id": project_id,
                }
        if not binding.study_context_id:
            try:
                active = study_contexts.get_active_context()
            except study_contexts.StudyContextError as exc:
                return {
                    "stale": True,
                    "reason": str(exc.detail.get("error") or "study_context_invalid"),
                }
            if active and active.get("id"):
                return {
                    "stale": True,
                    "reason": "authority_binding_available",
                    "mismatches": {
                        "study_context_id": {
                            "session": None,
                            "current": active.get("id"),
                        }
                    },
                }
            return {"stale": False}
        try:
            current = study_contexts.get_context(binding.study_context_id)
        except study_contexts.StudyContextError as exc:
            return {
                "stale": True,
                "reason": str(exc.detail.get("error") or "study_context_invalid"),
            }
        if current is None:
            return {"stale": True, "reason": "study_context_not_found"}
        current_revision = int(current.get("revision") or 0)
        current_job = (
            str(current.get("active_job_id")) if current.get("active_job_id") else None
        )
        current_run = self._latest_run_id(binding.study_context_id)
        mismatches = {}
        if current_revision != binding.study_revision:
            mismatches["study_revision"] = {
                "session": binding.study_revision,
                "current": current_revision,
            }
        if current_job != binding.active_job_id:
            mismatches["active_job_id"] = {
                "session": binding.active_job_id,
                "current": current_job,
            }
        if current_run != binding.run_id:
            mismatches["run_id"] = {
                "session": binding.run_id,
                "current": current_run,
            }
        return {
            "stale": bool(mismatches),
            "reason": "authority_binding_changed" if mismatches else None,
            "mismatches": mismatches,
        }

    def _stale_details(self, record: PiSessionRecord) -> Dict[str, Any]:
        return self._binding_stale_details(
            record.binding,
            project_id=record.project_id,
        )

    def rebind_session(self, session_id: str, *, project_id: str) -> Dict[str, Any]:
        record = self._scoped_record(session_id, project_id=project_id)
        try:
            context = study_contexts.get_context(
                self.project_store.assert_matches(
                    project_id,
                    record.binding.study_context_id,
                )
            )
        except study_contexts.StudyContextError as exc:
            raise PiCopilotError(
                str(exc.detail.get("error") or "study_context_invalid"),
                "The session cannot rebind because its StudyContext is invalid.",
                status_code=409,
                details=exc.detail,
            ) from exc
        if record.binding.study_context_id and context is None:
            raise PiCopilotError(
                "study_context_not_found",
                "The session cannot rebind because its StudyContext no longer exists.",
                status_code=404,
            )
        record.binding = self._binding_for_context(
            context,
            run_id=self._latest_run_id(str(context.get("id")) if context else None),
        )
        self._save_record(record)
        state = self._ensure_open(record)
        return {
            "ok": True,
            "session": self._public_session(record, gateway_state=state),
            "rebound": True,
        }

    def send_message(
        self,
        session_id: str,
        *,
        project_id: str,
        message: str,
        allowed_actions: Iterable[str] = (),
    ) -> Dict[str, Any]:
        record = self._scoped_record(session_id, project_id=project_id)
        self._provider_gate(external_llm_opt_in=record.external_llm_opt_in)
        text = str(message or "").strip()
        if not text:
            raise PiCopilotError(
                "pi_message_required",
                "A Pi Copilot message is required.",
            )
        if len(text) > MAX_MESSAGE_CHARS:
            raise PiCopilotError(
                "pi_message_too_long",
                "The Pi Copilot message exceeds its bounded contract.",
                details={"max_chars": MAX_MESSAGE_CHARS},
            )
        reject_sensitive_message(text)
        stale = self._stale_details(record)
        if stale.get("stale"):
            raise PiCopilotError(
                "pi_session_authority_stale",
                (
                    "The EasyICU study/run binding changed after this Pi session "
                    "was saved. Rebind before sending another message."
                ),
                status_code=409,
                details=stale,
            )
        requested_actions = frozenset(
            str(item).strip() for item in allowed_actions if str(item).strip()
        )
        unknown_actions = sorted(requested_actions - ALLOWED_TURN_ACTIONS)
        if unknown_actions:
            raise PiCopilotError(
                "pi_action_unknown",
                "The message requested an unknown host action capability.",
                details={"actions": unknown_actions},
            )
        self._ensure_open(record)
        tool_context = ToolExecutionContext(
            session=record.model_copy(deep=True),
            allowed_actions=requested_actions,
            authority_validator=lambda binding: self._binding_stale_details(
                binding,
                project_id=record.project_id,
            ),
            workspace_root=self.workspace_root,
        )
        with self._lock:
            if record.session_id in self._busy_sessions:
                raise PiCopilotError(
                    "pi_session_busy",
                    "This Pi Copilot session already has an active message.",
                    status_code=409,
                )
            self._busy_sessions.add(record.session_id)

        def runner(job: jobs.Job) -> Dict[str, Any]:
            with self._lock:
                self._active_message_jobs[record.session_id] = job.id

            def emit(event: Dict[str, Any]) -> None:
                if job.cancel_requested:
                    return
                job.emit({"type": "pi_event", "event": event})

            try:
                job.emit(
                    {
                        "type": "pi_session",
                        "session_id": record.session_id,
                        "allowed_actions": sorted(requested_actions),
                    }
                )
                state = self.gateway.request(
                    "session.prompt",
                    {"session_id": record.session_id, "message": text},
                    event_sink=emit,
                    tool_context=tool_context,
                )
                refreshed = self._get_record(record.session_id)
                refreshed.pi_session_id = (
                    str(state.get("pi_session_id") or "") or refreshed.pi_session_id
                )
                refreshed.pi_session_file = (
                    str(state.get("session_file") or "") or refreshed.pi_session_file
                )
                refreshed.last_message_job_id = job.id
                self._save_record(refreshed)
                return {
                    "session": self._public_session(refreshed, gateway_state=state),
                    "message_status": "aborted" if job.cancel_requested else "done",
                }
            finally:
                pending_retirement = False
                with self._lock:
                    if self._active_message_jobs.get(record.session_id) == job.id:
                        self._active_message_jobs.pop(record.session_id, None)
                    self._busy_sessions.discard(record.session_id)
                    pending_retirement = record.session_id in self._pending_retirements
                if pending_retirement:
                    self._flush_pending_retirement(record.session_id)

        try:
            job = jobs.MANAGER.submit("pi-copilot-message", runner)
        except jobs.JobCapacityError as exc:
            with self._lock:
                self._busy_sessions.discard(record.session_id)
            raise PiCopilotError(
                "job_capacity_reached",
                "EasyICU cannot start another Pi message job right now.",
                status_code=429,
                details={
                    "max_running": exc.max_running,
                    "running": exc.running,
                },
            ) from exc
        except Exception:
            with self._lock:
                self._busy_sessions.discard(record.session_id)
            raise
        return {
            "ok": True,
            "job_id": job.id,
            "kind": job.kind,
            "status": job.status,
            "session_id": record.session_id,
        }

    def abort_session(
        self,
        session_id: str,
        *,
        project_id: str,
        message_job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = self._scoped_record(session_id, project_id=project_id)
        with self._lock:
            active_job_id = self._active_message_jobs.get(record.session_id)
        requested_job_id = str(message_job_id or "").strip()
        if requested_job_id and requested_job_id != active_job_id:
            raise PiCopilotError(
                "pi_message_job_mismatch",
                "The abort request does not match this session's active message job.",
                status_code=409,
            )
        target_job_id = str(active_job_id or "").strip()
        job = jobs.MANAGER.get(target_job_id) if target_job_id else None
        cancel_requested = bool(job and job.request_cancel("pi_copilot_user_abort"))
        state = self.gateway.request(
            "session.abort",
            {"session_id": record.session_id},
            timeout=15,
        )
        return {
            "ok": True,
            "session_id": record.session_id,
            "message_job_id": target_job_id or None,
            "cancel_requested": cancel_requested,
            "pi_aborted": bool(state.get("aborted")),
        }

    def list_sessions(self, *, project_id: str, limit: int = 30) -> Dict[str, Any]:
        clean_project_id = str(project_id or "").strip()
        if not clean_project_id:
            raise PiCopilotError(
                "pi_project_binding_required",
                "A research project is required to list Pi conversations.",
                status_code=409,
            )
        max_items = max(1, min(100, int(limit or 30)))
        with self._lock:
            records = [
                row
                for row in self._read_records()
                if row.project_id == clean_project_id
            ][:max_items]
        records = [
            self._scoped_record(row.session_id, project_id=clean_project_id)
            for row in records
        ]
        return {
            "ok": True,
            "count": len(records),
            "sessions": [self._public_session(row) for row in records],
        }

    def get_session(self, session_id: str, *, project_id: str) -> Dict[str, Any]:
        record = self._scoped_record(session_id, project_id=project_id)
        state = self._ensure_open(record)
        return {
            "ok": True,
            "session": self._public_session(record, gateway_state=state),
        }

    def _assert_workspace_project(self, project_id: str) -> str:
        clean = str(project_id or "").strip()
        if not clean or not self.project_store.resolve(clean):
            raise PiCopilotError(
                "pi_workspace_project_not_initialized",
                "Initialize the EasyICU project before opening its artifact workspace.",
                status_code=409,
            )
        return clean

    def get_workspace_file(
        self,
        *,
        project_id: str,
        relative_file: str,
    ) -> Dict[str, Any]:
        clean = self._assert_workspace_project(project_id)
        return {
            "ok": True,
            "artifact": self.workspace.read_file(clean, relative_file),
        }

    def get_workspace_preview(
        self,
        *,
        project_id: str,
        relative_file: str,
    ) -> Dict[str, Any]:
        clean = self._assert_workspace_project(project_id)
        return {
            "ok": True,
            "artifact": self.workspace.preview_file(clean, relative_file),
        }

    def _public_session(
        self,
        record: PiSessionRecord,
        *,
        gateway_state: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = gateway_state or {}
        return {
            "session_id": record.session_id,
            "project_id": record.project_id,
            "title": record.title,
            "agent_mode": record.agent_mode,
            "language": record.language,
            "thinking_level": state.get("thinking_level") or record.thinking_level,
            "model": state.get("model"),
            "message_count": int(state.get("message_count") or 0),
            "streaming": bool(state.get("streaming")),
            "enabled_tools": (
                list(state.get("enabled_tools") or [])[:40]
                if isinstance(state.get("enabled_tools"), list)
                else []
            ),
            "transcript": (
                list(state.get("transcript") or [])[:100]
                if isinstance(state.get("transcript"), list)
                else []
            ),
            "shell_usage": (
                dict(state.get("shell_usage") or {})
                if isinstance(state.get("shell_usage"), Mapping)
                else {}
            ),
            "binding": record.binding.model_dump(mode="json"),
            "stale": self._stale_details(record),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "last_message_job_id": record.last_message_job_id,
            "session_storage": "private_local_jsonl",
            "scientific_authority": "EasyICU",
            "project_migration": (
                binding.migration_receipt.model_dump(mode="json")
                if (
                    (binding := self.project_store.binding(record.project_id))
                    and binding.migration_receipt
                )
                else None
            ),
        }

    def close(self) -> None:
        self.gateway.close()


_SERVICE: Optional[PiCopilotService] = None
_SERVICE_LOCK = threading.Lock()


def get_pi_copilot_service() -> PiCopilotService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = PiCopilotService()
        return _SERVICE


def reset_pi_copilot_service_for_tests() -> None:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is not None:
            _SERVICE.close()
        _SERVICE = None


__all__ = [
    "ALLOWED_TURN_ACTIONS",
    "PiCopilotService",
    "get_pi_copilot_service",
    "reset_pi_copilot_service_for_tests",
]
