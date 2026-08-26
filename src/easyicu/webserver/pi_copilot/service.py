"""Authoritative FastAPI-side owner for Pi Copilot sessions and turns."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from easyicu.databases.profiles import get_database_profile
from easyicu.extensions import (
    ExtensionActivationSnapshot,
    ExtensionRegistry,
    ExtensionRegistryError,
)

from easyicu.webserver import state_paths
from easyicu.webserver import (
    agent_pipeline_runs,
    agent_runs,
    guided_sessions,
    jobs,
    provider_gate,
    settings,
    sources,
    study_contexts,
)
from easyicu.webserver.data_package_review import DataPackageReviewSnapshotStore
from easyicu.webserver.copilot_data_workbench import (
    CopilotDataWorkbenchError,
    CopilotDataWorkbenchSnapshotStore,
)

from .contracts import (
    MAX_MESSAGE_CHARS,
    AuthorityBinding,
    PiCopilotError,
    PiProjectBindingHandoffReceipt,
    PiSessionDataSourceAuthorization,
    PiSessionDataSourceReference,
    PiSessionRecord,
    ResearchProviderBinding,
    ToolExecutionContext,
    utc_now,
)
from .codex_gateway import CodexPiGatewayPool
from .gateway import PiGatewayClient
from .project_authority import (
    ProjectAuthorityStore,
    ProjectStudyContextMigrationReceipt,
)
from .provider_config import PiProviderConfigStore
from .projections import project_job, project_pi_replay_event, reject_sensitive_message
from .replay_store import PiConversationReplayStore
from .run_authority import (
    latest_bound_run_id,
    list_bound_run_history,
    research_pipeline_project_root,
)
from .turn_authority import (
    explicitly_confirms_easyicu_registered_source,
    infer_explicit_turn_actions,
)
from .tools import extraction_workspace_resource
from .workspace import ProjectWorkspace
from .workflow import (
    build_research_workflow_snapshot,
    registered_export_matches_study,
)

MAX_SESSIONS = 100
COPILOT_MESSAGE_TIMEOUT_SECONDS = 90
MAX_RESEARCH_ARTIFACT_PREVIEW_BYTES = 2 * 1024 * 1024
ALLOWED_TURN_ACTIONS = frozenset(
    {
        "configure",
        "idea",
        "literature",
        "extract",
        "run",
        "provider_run",
        "cancel",
        "workspace_write",
        "mcp_read",
    }
)
_RETIRED_SESSION_METADATA_FIELDS = frozenset(
    {
        "canonical_task_id",
        "canonical_input_sha256",
        "canonical_job_id",
        "last_turn_events",
    }
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
        extension_registry: Optional[ExtensionRegistry] = None,
        replay_store: Optional[PiConversationReplayStore] = None,
        review_snapshot_store: Optional[DataPackageReviewSnapshotStore] = None,
        data_workbench_snapshot_store: Optional[
            CopilotDataWorkbenchSnapshotStore
        ] = None,
        codex_gateway_pool: Optional[CodexPiGatewayPool] = None,
    ) -> None:
        self.store_path = (
            Path(store_path)
            if store_path is not None
            else state_paths.state_root() / "pi_copilot_sessions.json"
        )
        self.gateway = gateway or PiGatewayClient()
        self.provider_store = (
            provider_store
            or getattr(self.gateway, "provider_store", None)
            or PiProviderConfigStore()
        )
        self.codex_gateway_pool = codex_gateway_pool or CodexPiGatewayPool(
            template_gateway=self.gateway,
        )
        self.project_store = project_store or ProjectAuthorityStore(
            None
            if store_path is None
            else self.store_path.with_name(f"{self.store_path.stem}.projects.json")
        )
        self.extension_registry = extension_registry or ExtensionRegistry()
        self.replay_store = replay_store or PiConversationReplayStore(
            None
            if store_path is None
            else self.store_path.with_name(f"{self.store_path.stem}.replay")
        )
        self.review_snapshot_store = (
            review_snapshot_store
            or DataPackageReviewSnapshotStore(
                None
                if store_path is None
                else self.store_path.with_name(
                    f"{self.store_path.stem}.data-package-reviews"
                )
            )
        )
        self.data_workbench_snapshot_store = (
            data_workbench_snapshot_store
            or CopilotDataWorkbenchSnapshotStore(
                None
                if store_path is None
                else self.store_path.with_name(
                    f"{self.store_path.stem}.copilot-data-workbench"
                )
            )
        )
        self._lock = threading.RLock()
        self._active_message_jobs: Dict[str, str] = {}
        self._busy_sessions: set[str] = set()
        self._pending_retirements: Dict[str, PiSessionRecord] = {}
        self._replay_child_watchers: set[tuple[str, str]] = set()
        self._project_initialization_locks: Dict[str, threading.RLock] = {}
        workspace_base = Path(
            getattr(
                self.gateway,
                "declared_cwd",
                self.store_path.parent / "pi_project_workspace",
            )
        )
        self.workspace_root = workspace_base.expanduser().absolute()
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
                    "The Copilot metadata store exceeds its bounded contract.",
                    status_code=500,
                )
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as exc:
            raise PiCopilotError(
                "pi_session_store_invalid",
                "The Copilot metadata store is invalid JSON.",
                status_code=500,
            ) from exc
        rows = raw.get("sessions") if isinstance(raw, dict) else None
        if not isinstance(rows, list):
            raise PiCopilotError(
                "pi_session_store_invalid",
                "The Copilot metadata store has an invalid shape.",
                status_code=500,
            )
        records = []
        for row in rows[:MAX_SESSIONS]:
            if isinstance(row, Mapping) and row.get("canonical_task_id"):
                # A short-lived development-only UI specialization wrote these
                # records before being retired. They never represented ordinary
                # user conversations, so do not surface them as such.
                continue
            try:
                migrated = (
                    {
                        key: value
                        for key, value in row.items()
                        if key not in _RETIRED_SESSION_METADATA_FIELDS
                    }
                    if isinstance(row, Mapping)
                    else row
                )
                records.append(PiSessionRecord.model_validate(migrated))
            except Exception as exc:
                raise PiCopilotError(
                    "pi_session_store_invalid",
                    "The Copilot metadata store contains an invalid session.",
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
                "The requested Copilot session does not exist.",
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
                    if (
                        candidate.session_id in self._busy_sessions
                        or candidate.pinned_for_presentation
                    ):
                        continue
                    evicted.append(candidate)
                    if len(evicted) == overflow:
                        break
                if len(evicted) != overflow:
                    raise PiCopilotError(
                        "pi_session_retention_protected",
                        "Copilot session retention cannot evict an active or presentation-pinned conversation.",
                        status_code=409,
                    )
                evicted_ids = {row.session_id for row in evicted}
                rows = [row for row in rows if row.session_id not in evicted_ids]
            self._write_records(rows)
        for retired in evicted:
            self._retire_record(retired)
        return record

    def _conversation_gateway_for_binding(
        self,
        binding: ResearchProviderBinding,
        *,
        unified: bool,
        refresh_account: bool = False,
    ) -> PiGatewayClient:
        if unified and binding.provider == "codex":
            return self.codex_gateway_pool.gateway_for(
                binding,
                refresh_account=refresh_account,
            )
        return self.gateway

    def _conversation_gateway(
        self,
        record: PiSessionRecord,
        *,
        refresh_account: bool = False,
    ) -> PiGatewayClient:
        return self._conversation_gateway_for_binding(
            record.research_provider,
            unified=record.uses_unified_model_connection,
            refresh_account=refresh_account,
        )

    def _retire_record(self, record: PiSessionRecord) -> None:
        """Dispose an evicted Pi session and remove only its private JSONL."""

        with self._lock:
            if record.session_id in self._busy_sessions:
                self._pending_retirements[record.session_id] = record
                return
        self.replay_store.retire(record.session_id)
        gateway: Any = self.gateway
        try:
            gateway = self._conversation_gateway(record)
            gateway.request(
                "session.dispose",
                {"session_id": record.session_id},
                timeout=5,
            )
        except (OSError, PiCopilotError):
            pass
        session_root = getattr(
            gateway,
            "session_dir",
            None,
        )
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

    def _provider_gate(
        self,
        *,
        external_llm_opt_in: bool,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        current_settings = settings.load_settings()
        selected_provider = str(
            provider
            or getattr(self.gateway, "environ", {}).get("EASYICU_PI_PROVIDER")
            or "easyicu-local"
        )
        try:
            return provider_gate.resolve_provider_gate(
                run_type="full",
                llm_provider=selected_provider,
                external_llm_opt_in=bool(external_llm_opt_in),
                ai_enabled=bool(current_settings.get("ai_enabled")),
                language=str(current_settings.get("language") or "en"),
            )
        except provider_gate.ProviderGateError as exc:
            raise PiCopilotError(
                str(exc.detail.get("error") or "external_llm_opt_in_required"),
                (
                    "EasyICU Copilot uses an external model call. Enable AI in "
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
        shell_ready = not runtime_blockers.intersection(blockers) and bool(
            current_settings.get("ai_enabled")
        )
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
                "shell_ready": shell_ready,
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

    def verified_api_research_provider_binding(self) -> ResearchProviderBinding:
        """Compile the verified Pi API config into a secret-free run binding."""

        self.provider_store.research_agent_environment(
            external_llm_opt_in=True,
        )
        status = self.provider_store.public_status()
        return ResearchProviderBinding(
            provider="openai",
            credential_source="pi_verified",
            authentication_mode="api_key",
            model=str(status.get("model") or "configured_provider_model"),
        )

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
    def _session_source_reference(
        context: Mapping[str, Any],
    ) -> Optional[PiSessionDataSourceReference]:
        source = context.get("data_source")
        if not isinstance(source, Mapping) or not any(
            str(source.get(key) or "").strip()
            for key in ("source_id", "id", "path", "path_hash", "database", "label")
        ):
            return None
        identity = {
            key: str(source.get(key) or "").strip()
            for key in (
                "source_id",
                "id",
                "path",
                "path_hash",
                "database",
                "label",
                "reference_release",
            )
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        database = str(source.get("database") or "").strip()
        profile = None
        if database:
            try:
                profile = get_database_profile(database)
            except KeyError:
                profile = None
        return PiSessionDataSourceReference(
            source_id=str(source.get("source_id") or source.get("id") or "").strip()
            or None,
            label=str(
                profile.display_name if profile is not None else source.get("label") or ""
            ).strip()[:160]
            or None,
            database=database[:80] or None,
            reference_release=str(
                source.get("reference_release")
                or (profile.reference_release if profile is not None else "")
            ).strip()[:80]
            or None,
            identity_sha256=digest,
            study_revision=int(context.get("revision") or 0),
        )

    @classmethod
    def _new_session_data_authorization(
        cls,
        context: Mapping[str, Any],
        *,
        agent_mode: str,
    ) -> PiSessionDataSourceAuthorization:
        if agent_mode == "workspace":
            return PiSessionDataSourceAuthorization(
                status="not_required",
                confirmation_mode=None,
            )
        source = cls._session_source_reference(context)
        return PiSessionDataSourceAuthorization(
            status="pending",
            reason=(
                "project_source_confirmation_required"
                if source is not None
                else "local_data_selection_required"
            ),
            confirmation_mode=None,
            source=source,
        )

    def _latest_run_id(
        self,
        study_context_id: Optional[str],
        *,
        project_id: Optional[str] = None,
    ) -> Optional[str]:
        project_root = research_pipeline_project_root(study_context_id)
        return latest_bound_run_id(
            study_context_id=study_context_id,
            project_root=project_root,
        )

    @staticmethod
    def _explicit_context_binding_receipt(
        study_context_id: str,
    ) -> ProjectStudyContextMigrationReceipt:
        requested_source = hashlib.sha256(
            str(study_context_id).encode("utf-8")
        ).hexdigest()
        return ProjectStudyContextMigrationReceipt(
            status="initialized",
            source_schema="easyicu.explicit-studycontext-binding/1",
            source_digest=requested_source,
            migrated_fields=["study_context_id"],
        )

    def _resolve_project_context(
        self,
        *,
        project_id: str,
        title: str,
        requested_study_context_id: Optional[str] = None,
        confirm_initialization: bool = False,
        defer_requested_binding: bool = False,
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
            try:
                setup = guided_sessions.read_project_study_setup(project_id)
            except guided_sessions.GuidedProjectMigrationError as exc:
                raise PiCopilotError(
                    exc.code,
                    (
                        "Existing Guided project metadata cannot be migrated "
                        "without changing scientific input identity."
                    ),
                    status_code=409,
                    details=exc.details,
                ) from exc
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
                    "Confirm a new Copilot study setup before opening this project.",
                    status_code=409,
                    details={
                        "project_id": project_id,
                        "missing_required": missing,
                        "saved_metadata_found": setup is not None,
                    },
                )
            else:
                initial = {
                    "title": str(title or "EasyICU Copilot").strip()[:160] or "EasyICU Copilot",
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
            if not defer_requested_binding:
                self.project_store.bind(
                    project_id,
                    str(context["id"]),
                    migration_receipt=self._explicit_context_binding_receipt(
                        str(requested_study_context_id)
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
        binding_receipt: Optional[PiProjectBindingHandoffReceipt] = None,
    ) -> Dict[str, Any]:
        """Explicitly migrate/bind Guided metadata before any session GET."""

        clean_project = str(project_id or "").strip()
        if not clean_project or len(clean_project) > 160:
            raise PiCopilotError(
                "pi_project_binding_required",
                "A research project is required for Copilot initialization.",
                status_code=409,
            )
        with self._project_initialization_lock(clean_project):
            return self._initialize_project_locked(
                project_id=clean_project,
                title=title,
                confirm_initialization=confirm_initialization,
                binding_receipt=binding_receipt,
            )

    def _initialize_project_locked(
        self,
        *,
        project_id: str,
        title: str,
        confirm_initialization: bool,
        binding_receipt: Optional[PiProjectBindingHandoffReceipt] = None,
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
                "Legacy Copilot sessions disagree about this project's StudyContext.",
                status_code=409,
                details={"project_id": clean_project},
            )
        requested_context_id = next(iter(legacy_context_ids), None)
        if binding_receipt is not None:
            if binding_receipt.project_id != clean_project:
                raise PiCopilotError(
                    "pi_project_handoff_mismatch",
                    "The Agent handoff belongs to another research project.",
                    status_code=409,
                )
            mapped_context_id = self.project_store.resolve(clean_project)
            if (
                mapped_context_id is not None
                and mapped_context_id != binding_receipt.study_context_id
            ):
                raise PiCopilotError(
                    "pi_project_study_context_mismatch",
                    "The project is already bound to another StudyContext.",
                    status_code=409,
                    details={
                        "project_id": clean_project,
                        "mapped_study_context_id": mapped_context_id,
                        "handoff_study_context_id": binding_receipt.study_context_id,
                    },
                )
            if (
                requested_context_id is not None
                and requested_context_id != binding_receipt.study_context_id
            ):
                raise PiCopilotError(
                    "pi_project_study_context_mismatch",
                    "Saved Copilot sessions disagree with the Agent handoff StudyContext.",
                    status_code=409,
                )
            requested_context_id = binding_receipt.study_context_id
            try:
                handed_off_context = study_contexts.get_context(requested_context_id)
            except study_contexts.StudyContextError as exc:
                raise PiCopilotError(
                    str(exc.detail.get("error") or "study_context_invalid"),
                    "The Agent handoff StudyContext could not be loaded.",
                    status_code=409,
                    details=exc.detail,
                ) from exc
            if handed_off_context is None:
                raise PiCopilotError(
                    "pi_project_study_context_missing",
                    "The Agent handoff StudyContext no longer exists.",
                    status_code=409,
                    details={"study_context_id": requested_context_id},
                )
            current_revision = int(handed_off_context.get("revision") or 0)
            if current_revision != binding_receipt.study_context_revision:
                raise PiCopilotError(
                    "pi_project_handoff_revision_conflict",
                    "The StudyContext changed after the Agent handoff was created.",
                    status_code=409,
                    details={
                        "study_context_id": binding_receipt.study_context_id,
                        "expected_revision": binding_receipt.study_context_revision,
                        "current_revision": current_revision,
                    },
                )
        context = self._resolve_project_context(
            project_id=clean_project,
            title=title,
            requested_study_context_id=requested_context_id,
            confirm_initialization=confirm_initialization,
            defer_requested_binding=binding_receipt is not None,
        )
        if binding_receipt is not None and (
            str(context.get("id") or "") != binding_receipt.study_context_id
            or int(context.get("revision") or 0)
            != binding_receipt.study_context_revision
        ):
            raise PiCopilotError(
                "pi_project_handoff_revision_conflict",
                "The StudyContext changed while the Agent handoff was being bound.",
                status_code=409,
                details={
                    "study_context_id": binding_receipt.study_context_id,
                    "expected_revision": binding_receipt.study_context_revision,
                    "current_study_context_id": str(context.get("id") or ""),
                    "current_revision": int(context.get("revision") or 0),
                },
            )
        if binding_receipt is not None:
            # The final revision check above must happen before the immutable
            # project mapping is published. Otherwise a concurrent StudyContext
            # update can reject the handoff while still leaving a new binding
            # behind.
            self.project_store.bind(
                clean_project,
                str(context["id"]),
                migration_receipt=self._explicit_context_binding_receipt(
                    binding_receipt.study_context_id
                ),
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
                        "A legacy Copilot session belongs to another StudyContext.",
                        status_code=409,
                        details={"project_id": clean_project},
                    )
                record.binding = self._binding_for_context(
                    context,
                    run_id=self._latest_run_id(
                        str(context["id"]),
                        project_id=clean_project,
                    ),
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
            "study_context_revision": int(context.get("revision") or 0),
            "binding_receipt": (
                binding_receipt.model_dump(mode="json")
                if binding_receipt is not None
                else None
            ),
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
                "A research project is required for this Copilot session operation.",
                status_code=409,
            )
        record = self._get_record(session_id)
        if record.project_id != clean_project:
            raise PiCopilotError(
                "pi_session_project_mismatch",
                "This Copilot conversation belongs to a different EasyICU research project.",
                status_code=409,
                details={
                    "session_id": record.session_id,
                    "requested_project_id": clean_project,
                },
            )
        if not record.binding.study_context_id:
            raise PiCopilotError(
                "pi_project_initialization_required",
                "Initialize this project's StudyContext before reading its Copilot sessions.",
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
        title: str = "EasyICU Copilot",
        agent_mode: str = "research",
        language: str = "en",
        thinking_level: str = "off",
        study_context_id: Optional[str] = None,
        external_llm_opt_in: bool = False,
        research_provider: Optional[ResearchProviderBinding] = None,
    ) -> Dict[str, Any]:
        clean_project_id = str(project_id or "").strip()
        if not clean_project_id:
            raise PiCopilotError(
                "pi_project_binding_required",
                "Select or create an EasyICU research project before starting a Copilot conversation.",
                status_code=409,
            )
        selected_model_connection = research_provider or ResearchProviderBinding()
        self._provider_gate(
            external_llm_opt_in=external_llm_opt_in,
            provider=selected_model_connection.provider,
        )
        resolved_mode = "workspace" if agent_mode == "workspace" else "research"
        conversation_gateway = self._conversation_gateway_for_binding(
            selected_model_connection,
            unified=True,
            refresh_account=True,
        )
        install = conversation_gateway.installation_status()
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
            run_id=self._latest_run_id(
                str(context.get("id")) if context else None,
                project_id=clean_project_id,
            ),
        )
        try:
            extension_activation = self.extension_registry.snapshot()
        except ExtensionRegistryError as exc:
            raise PiCopilotError(
                exc.code,
                exc.message,
                status_code=409,
                details=exc.details,
            ) from exc
        if not bool(settings.load_settings().get("mcp_tools_enabled", False)):
            extension_activation = ExtensionActivationSnapshot.build(
                revision=extension_activation.revision,
                skills=extension_activation.skills,
                mcp_servers=(),
            )
        state = conversation_gateway.request(
            "session.create",
            {
                "session_id": session_id,
                "thinking_level": resolved_thinking,
                "agent_mode": resolved_mode,
                "language": resolved_language,
                "extension_snapshot": extension_activation.model_dump(mode="json"),
            },
            timeout=30,
        )
        record = PiSessionRecord(
            session_id=session_id,
            project_id=clean_project_id,
            pi_session_id=str(state.get("pi_session_id") or "") or None,
            pi_session_file=str(state.get("session_file") or "") or None,
            title=str(title or "EasyICU Copilot").strip()[:160] or "EasyICU Copilot",
            agent_mode=resolved_mode,
            language=resolved_language,
            thinking_level=resolved_thinking,
            external_llm_opt_in=True,
            extension_activation=extension_activation,
            research_provider=selected_model_connection,
            binding=binding,
            data_source_authorization=self._new_session_data_authorization(
                context,
                agent_mode=resolved_mode,
            ),
        )
        self._save_record(record)
        return {
            "ok": True,
            "session": self._public_session(record, gateway_state=state),
        }

    def _ensure_open(
        self,
        record: PiSessionRecord,
        *,
        transcript_cursor: Optional[str] = None,
        transcript_limit: int = 100,
    ) -> Dict[str, Any]:
        gateway = self._conversation_gateway(record)
        state_params: Dict[str, Any] = {
            "session_id": record.session_id,
            "transcript_limit": max(1, min(200, int(transcript_limit))),
        }
        if transcript_cursor is not None:
            state_params["transcript_cursor"] = str(transcript_cursor)
        try:
            return gateway.request(
                "session.state",
                state_params,
                timeout=5,
            )
        except PiCopilotError as exc:
            if exc.code != "pi_session_not_open":
                raise
        opened = gateway.request(
            "session.create",
            {
                "session_id": record.session_id,
                "session_file": record.pi_session_file,
                "thinking_level": "off",
                "agent_mode": record.agent_mode,
                "language": record.language,
                "extension_snapshot": record.extension_activation.model_dump(
                    mode="json"
                ),
            },
            timeout=30,
        )
        if transcript_cursor is None and int(transcript_limit) == 100:
            return opened
        return gateway.request("session.state", state_params, timeout=5)

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
        current_run = self._latest_run_id(
            binding.study_context_id,
            project_id=project_id,
        )
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
            run_id=self._latest_run_id(
                str(context.get("id")) if context else None,
                project_id=record.project_id,
            ),
        )
        if record.agent_mode == "research":
            prior_authorization = record.data_source_authorization
            current_source = self._session_source_reference(context or {})
            if (
                prior_authorization.status == "confirmed"
                and prior_authorization.source is not None
                and current_source is not None
                and prior_authorization.source.identity_sha256
                == current_source.identity_sha256
            ):
                record.data_source_authorization = prior_authorization.model_copy(
                    update={"source": current_source}
                )
            elif prior_authorization.status != "selection_in_progress":
                record.data_source_authorization = self._new_session_data_authorization(
                    context or {},
                    agent_mode=record.agent_mode,
                )
        self._save_record(record)
        state = self._ensure_open(record)
        return {
            "ok": True,
            "session": self._public_session(record, gateway_state=state),
            "rebound": True,
        }

    def authorize_data_source(
        self,
        session_id: str,
        *,
        project_id: str,
        action: str,
    ) -> Dict[str, Any]:
        """Confirm one path-free source choice before a research conversation."""

        record = self._scoped_record(session_id, project_id=project_id)
        if record.agent_mode != "research":
            raise PiCopilotError(
                "pi_session_data_source_authorization_not_required",
                "Workspace conversations do not use the research data-source gate.",
                status_code=409,
            )
        clean_action = str(action or "").strip()
        if clean_action not in {
            "reuse_project_source",
            "begin_local_selection",
            "confirm_selected_source",
        }:
            raise PiCopilotError(
                "pi_session_data_source_action_invalid",
                "Choose a supported data-source confirmation action.",
                status_code=422,
            )
        context_id = self.project_store.assert_matches(
            project_id,
            record.binding.study_context_id,
        )
        try:
            context = study_contexts.get_context(context_id)
        except study_contexts.StudyContextError as exc:
            raise PiCopilotError(
                str(exc.detail.get("error") or "study_context_invalid"),
                "The project data source cannot be confirmed because its StudyContext is invalid.",
                status_code=409,
                details=exc.detail,
            ) from exc
        if context is None:
            raise PiCopilotError(
                "study_context_not_found",
                "The project data source cannot be confirmed because its StudyContext is missing.",
                status_code=404,
            )

        if clean_action == "begin_local_selection":
            record.data_source_authorization = PiSessionDataSourceAuthorization(
                status="selection_in_progress",
                reason="local_data_selection_required",
                confirmation_mode="select_local_source",
            )
            self._save_record(record)
            return {
                "ok": True,
                "session": self._public_session(record),
                "resource": extraction_workspace_resource(
                    context,
                    state="setup",
                    entry_mode="source_binding",
                ),
            }

        source = self._session_source_reference(context)
        if source is None:
            raise PiCopilotError(
                "pi_session_data_source_unavailable",
                "Select and validate a local data folder before confirming this conversation.",
                status_code=409,
            )
        if clean_action == "confirm_selected_source":
            if record.data_source_authorization.status != "selection_in_progress":
                raise PiCopilotError(
                    "pi_session_local_selection_not_started",
                    "Start the local folder workflow before confirming its selected source.",
                    status_code=409,
                )
            record.binding = self._binding_for_context(
                context,
                run_id=self._latest_run_id(
                    str(context.get("id") or ""),
                    project_id=record.project_id,
                ),
            )
            confirmation_mode = "select_local_source"
        else:
            stale = self._stale_details(record)
            if stale.get("stale"):
                raise PiCopilotError(
                    "pi_session_authority_stale",
                    "The project data source changed before confirmation. Refresh this conversation and choose again.",
                    status_code=409,
                    details=stale,
                )
            confirmation_mode = "reuse_project_source"
        record.data_source_authorization = PiSessionDataSourceAuthorization(
            status="confirmed",
            reason=None,
            confirmation_mode=confirmation_mode,
            source=source,
            confirmed_at=utc_now(),
        )
        self._save_record(record)
        return {
            "ok": True,
            "session": self._public_session(record),
            "resource": None,
        }

    def _confirm_registered_source_selected_in_turn(
        self,
        record: PiSessionRecord,
        *,
        user_message: str,
    ) -> None:
        """Project an explicit prepared-source choice into session authority.

        The StudyContext tool owns source binding.  This service owns the
        per-conversation data-access gate, so it reconciles the two only after
        the tool has persisted an exact path that still matches a validated
        registered export.
        """

        if (
            record.agent_mode != "research"
            or record.data_source_authorization.status == "confirmed"
            or not explicitly_confirms_easyicu_registered_source(user_message)
        ):
            return
        context_id = str(record.binding.study_context_id or "").strip()
        if not context_id:
            return
        try:
            context = study_contexts.get_context(context_id)
        except study_contexts.StudyContextError:
            return
        if not context:
            return
        source = context.get("data_source")
        source = source if isinstance(source, Mapping) else {}
        expected_path = str(source.get("path") or "").strip()
        registry = sources.load_registry()
        exact_registered_source = next(
            (
                row
                for row in (registry.get("sources") or [])
                if isinstance(row, Mapping)
                and bool(row.get("ok"))
                and str(row.get("path") or "").strip() == expected_path
            ),
            None,
        )
        if exact_registered_source is None:
            return
        source_reference = self._session_source_reference(context)
        if source_reference is None:
            return
        record.binding = self._binding_for_context(
            context,
            run_id=self._latest_run_id(
                context_id,
                project_id=record.project_id,
            ),
        )
        record.data_source_authorization = PiSessionDataSourceAuthorization(
            status="confirmed",
            reason=None,
            confirmation_mode="reuse_project_source",
            source=source_reference,
            confirmed_at=utc_now(),
        )

    def send_message(
        self,
        session_id: str,
        *,
        project_id: str,
        message: str,
        allowed_actions: Iterable[str] = (),
    ) -> Dict[str, Any]:
        record = self._scoped_record(session_id, project_id=project_id)
        self._provider_gate(
            external_llm_opt_in=record.external_llm_opt_in,
            provider=record.research_provider.provider,
        )
        text = str(message or "").strip()
        if not text:
            raise PiCopilotError(
                "pi_message_required",
                "An EasyICU Copilot message is required.",
            )
        if len(text) > MAX_MESSAGE_CHARS:
            raise PiCopilotError(
                "pi_message_too_long",
                "The EasyICU Copilot message exceeds its bounded contract.",
                details={"max_chars": MAX_MESSAGE_CHARS},
            )
        reject_sensitive_message(text)
        stale = self._stale_details(record)
        if stale.get("stale"):
            raise PiCopilotError(
                "pi_session_authority_stale",
                (
                    "The EasyICU study/run binding changed after this Copilot session "
                    "was saved. Rebind before sending another message."
                ),
                status_code=409,
                details=stale,
            )
        # A prepared source that is already bound to this project can be
        # confirmed before the provider turn.  This lets one explicit user
        # choice both unlock the data gate and advance to the requested data
        # action, instead of forcing a second identical confirmation click.
        # The helper still requires an exact validated registry-path match.
        prior_authorization = record.data_source_authorization.model_dump(mode="json")
        self._confirm_registered_source_selected_in_turn(
            record,
            user_message=text,
        )
        if (
            record.data_source_authorization.model_dump(mode="json")
            != prior_authorization
        ):
            self._save_record(record)
        requested_actions = frozenset(
            str(item).strip() for item in allowed_actions if str(item).strip()
        ) | infer_explicit_turn_actions(text)
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
            user_message=text,
            allowed_actions=requested_actions,
            authority_validator=lambda binding: self._binding_stale_details(
                binding,
                project_id=record.project_id,
            ),
            workspace=self.workspace,
            extension_registry=self.extension_registry,
        )
        with self._lock:
            if record.session_id in self._busy_sessions:
                raise PiCopilotError(
                    "pi_session_busy",
                    "This EasyICU Copilot session already has an active message.",
                    status_code=409,
                )
            self._busy_sessions.add(record.session_id)

        def runner(job: jobs.Job) -> Dict[str, Any]:
            conversation_gateway = self._conversation_gateway(
                record,
                refresh_account=True,
            )
            with self._lock:
                self._active_message_jobs[record.session_id] = job.id

            started = self._get_record(record.session_id)
            started.active_message_job_id = job.id
            started.last_turn_status = "running"
            started.last_turn_allowed_actions = sorted(requested_actions)
            self._save_record(started)
            self.replay_store.start_turn(
                session_id=record.session_id,
                project_id=str(record.project_id),
                job_id=job.id,
                allowed_actions=sorted(requested_actions),
            )

            def emit(event: Dict[str, Any]) -> None:
                if job.cancel_requested:
                    return
                job.emit({"type": "pi_event", "event": event})
                projected = project_pi_replay_event(event)
                if projected is None:
                    return
                self.replay_store.append_event(
                    session_id=record.session_id,
                    project_id=str(record.project_id),
                    job_id=job.id,
                    event=projected,
                )
                child_job_id = str(projected.get("job_id") or "").strip()
                if (
                    projected.get("type") == "tool_end"
                    and child_job_id
                    and projected.get("code")
                    in {
                        "easyicu_extraction_submitted",
                        "easyicu_demo_source_preparation_submitted",
                        "easyicu_run_submitted",
                        "easyicu_full_run_submitted",
                    }
                ):
                    self._watch_child_job_for_replay(
                        session_id=record.session_id,
                        project_id=str(record.project_id),
                        job_id=child_job_id,
                    )

            try:
                job.emit(
                    {
                        "type": "pi_session",
                        "session_id": record.session_id,
                        "allowed_actions": sorted(requested_actions),
                    }
                )
                state = conversation_gateway.request(
                    "session.prompt",
                    {"session_id": record.session_id, "message": text},
                    timeout=COPILOT_MESSAGE_TIMEOUT_SECONDS,
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
                self._confirm_registered_source_selected_in_turn(
                    refreshed,
                    user_message=text,
                )
                refreshed.last_message_job_id = job.id
                refreshed.active_message_job_id = job.id
                refreshed.last_turn_status = "running"
                self._save_record(refreshed)
                transcript = state.get("transcript")
                if isinstance(transcript, list):
                    terminal_assistant = next(
                        (
                            row
                            for row in reversed(transcript)
                            if isinstance(row, dict) and row.get("role") == "assistant"
                        ),
                        None,
                    )
                    if (
                        isinstance(terminal_assistant, dict)
                        and terminal_assistant.get("stop_reason") == "error"
                    ):
                        raise RuntimeError(
                            str(
                                terminal_assistant.get("error_code")
                                or "pi_model_provider_error"
                            )
                        )
                refreshed.active_message_job_id = None
                refreshed.last_turn_status = (
                    "cancelled" if job.cancel_requested else "done"
                )
                self._save_record(refreshed)
                self.replay_store.finish_turn(
                    session_id=record.session_id,
                    project_id=str(record.project_id),
                    job_id=job.id,
                    status="cancelled" if job.cancel_requested else "done",
                )
                return {
                    "session": self._public_session(refreshed, gateway_state=state),
                    "message_status": "aborted" if job.cancel_requested else "done",
                }
            finally:
                try:
                    terminal = self._get_record(record.session_id)
                    if terminal.active_message_job_id == job.id:
                        terminal.active_message_job_id = None
                        terminal.last_turn_status = (
                            "cancelled" if job.cancel_requested else "failed"
                        )
                        self._save_record(terminal)
                        self.replay_store.finish_turn(
                            session_id=record.session_id,
                            project_id=str(record.project_id),
                            job_id=job.id,
                            status=terminal.last_turn_status,
                        )
                except PiCopilotError:
                    pass
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

    def _watch_child_job_for_replay(
        self,
        *,
        session_id: str,
        project_id: str,
        job_id: str,
    ) -> None:
        """Archive one bounded child-job projection even if the tab closes."""

        watcher_key = (session_id, job_id)
        with self._lock:
            if watcher_key in self._replay_child_watchers:
                return
            self._replay_child_watchers.add(watcher_key)

        def watch() -> None:
            missing_deadline = time.monotonic() + 5
            try:
                while True:
                    child = jobs.MANAGER.get(job_id)
                    if child is None:
                        if time.monotonic() >= missing_deadline:
                            return
                        time.sleep(0.1)
                        continue
                    if child.status in {"done", "failed", "cancelled"}:
                        self.replay_store.archive_child_job(
                            session_id=session_id,
                            project_id=project_id,
                            job=project_job(child.snapshot()),
                        )
                        return
                    time.sleep(0.2)
            except PiCopilotError:
                return
            finally:
                with self._lock:
                    self._replay_child_watchers.discard(watcher_key)

        threading.Thread(
            target=watch,
            name=f"pi-replay-{job_id[:24]}",
            daemon=True,
        ).start()

    def archive_child_job(
        self,
        session_id: str,
        *,
        project_id: str,
        job_id: str,
    ) -> Dict[str, Any]:
        """Persist a safe job projection only for this conversation's child."""

        record = self._scoped_record(session_id, project_id=project_id)
        replay = self.replay_store.snapshot(
            session_id=record.session_id,
            project_id=str(record.project_id),
        )
        archived = next(
            (
                row
                for row in replay.get("child_jobs", [])
                if isinstance(row, Mapping) and row.get("job_id") == job_id
            ),
            None,
        )
        child = jobs.MANAGER.get(str(job_id or "").strip())
        if child is None:
            if archived is not None:
                return {"ok": True, "job": dict(archived), "already_archived": True}
            raise PiCopilotError(
                "pi_replay_child_job_unavailable",
                "The child job is no longer available to archive.",
                status_code=404,
            )
        projected = project_job(child.snapshot())
        saved = self.replay_store.archive_child_job(
            session_id=record.session_id,
            project_id=str(record.project_id),
            job=projected,
        )
        return {"ok": True, "job": saved, "already_archived": False}

    def set_presentation_pin(
        self,
        session_id: str,
        *,
        project_id: str,
        pinned: bool,
    ) -> Dict[str, Any]:
        """Protect or release one conversation from ordinary retention eviction."""

        record = self._scoped_record(session_id, project_id=project_id)
        record.pinned_for_presentation = bool(pinned)
        self._save_record(record)
        return {"ok": True, "session": self._public_session(record)}

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
        state = self._conversation_gateway(record).request(
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

    def list_sessions(
        self,
        *,
        project_id: str,
        limit: int = 30,
        agent_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        clean_project_id = str(project_id or "").strip()
        if not clean_project_id:
            raise PiCopilotError(
                "pi_project_binding_required",
                "A research project is required to list Copilot conversations.",
                status_code=409,
            )
        max_items = max(1, min(100, int(limit or 30)))
        clean_agent_mode = str(agent_mode or "").strip()
        if clean_agent_mode and clean_agent_mode not in {"research", "workspace"}:
            raise PiCopilotError(
                "pi_agent_mode_invalid",
                "Copilot conversation mode must be research or workspace.",
                status_code=422,
            )
        with self._lock:
            records = [
                row
                for row in self._read_records()
                if row.project_id == clean_project_id
                and (not clean_agent_mode or row.agent_mode == clean_agent_mode)
            ][:max_items]
        records = [
            self._reconcile_replay_execution(
                self._scoped_record(row.session_id, project_id=clean_project_id)
            )
            for row in records
        ]
        return {
            "ok": True,
            "count": len(records),
            "sessions": [
                self._public_session(row, include_replay=False) for row in records
            ],
        }

    def get_session(
        self,
        session_id: str,
        *,
        project_id: str,
        transcript_cursor: Optional[str] = None,
        transcript_limit: int = 100,
        replay_cursor: Optional[str] = None,
        replay_limit: int = 48,
    ) -> Dict[str, Any]:
        record = self._reconcile_replay_execution(
            self._scoped_record(session_id, project_id=project_id)
        )
        state = self._ensure_open(
            record,
            transcript_cursor=transcript_cursor,
            transcript_limit=transcript_limit,
        )
        return {
            "ok": True,
            "session": self._public_session(
                record,
                gateway_state=state,
                replay_cursor=replay_cursor,
                replay_limit=replay_limit,
            ),
        }

    def _assert_project_initialized(self, project_id: str) -> str:
        clean = str(project_id or "").strip()
        if not clean or not self.project_store.resolve(clean):
            raise PiCopilotError(
                "pi_project_not_initialized",
                "Initialize the EasyICU project before opening its governed resources.",
                status_code=409,
            )
        return clean

    def get_workspace_file(
        self,
        *,
        project_id: str,
        relative_file: str,
    ) -> Dict[str, Any]:
        clean = self._assert_project_initialized(project_id)
        return {
            "ok": True,
            "artifact": self.workspace.read_file(clean, relative_file),
        }

    def get_project_workflow(self, *, project_id: str) -> Dict[str, Any]:
        """Return the path-free project workflow compiled from owner receipts."""

        clean = self._assert_project_initialized(project_id)
        study_context_id = self.project_store.resolve(clean)
        study = (
            study_contexts.get_context(study_context_id) if study_context_id else None
        )
        registry = sources.load_registry()
        active_job = None
        if study and study.get("active_job_id"):
            job = jobs.MANAGER.get(str(study["active_job_id"]))
            active_job = job.snapshot() if job else None
        rows = list_bound_run_history(
            study_context_id=study_context_id,
            project_root=research_pipeline_project_root(study_context_id),
            limit=1,
        )
        latest_run = rows[0] if rows else None
        plan_review_authority = (
            agent_pipeline_runs.pending_review(latest_run.get("run_id"))
            if latest_run
            else None
        )
        snapshot = build_research_workflow_snapshot(
            study=study,
            active_export_present=registered_export_matches_study(study, registry),
            active_job=active_job,
            latest_run=latest_run,
            plan_review_authority=plan_review_authority,
        )
        return {
            "ok": True,
            "project_id": clean,
            "workflow": snapshot.model_dump(mode="json"),
            "active_job": project_job(active_job),
        }

    def get_workspace_preview(
        self,
        *,
        project_id: str,
        relative_file: str,
        checked_sha256: str,
    ) -> Dict[str, Any]:
        clean = self._assert_project_initialized(project_id)
        return {
            "ok": True,
            "artifact": self.workspace.preview_file(
                clean,
                relative_file,
                checked_sha256=checked_sha256,
            ),
        }

    def get_data_package_review(
        self,
        *,
        project_id: str,
        study_revision: int,
        review_sha256: str,
    ) -> Dict[str, Any]:
        """Open the exact immutable review, rebuilding only the current revision."""

        clean = self._assert_project_initialized(project_id)
        study_context_id = self.project_store.resolve(clean)
        study = (
            study_contexts.get_context(study_context_id) if study_context_id else None
        )
        if not study:
            raise PiCopilotError(
                "pi_data_package_study_not_found",
                "The research project has no bound StudyContext.",
                status_code=404,
            )
        expected_revision = int(study_revision)
        current_revision = int(study.get("revision") or 0)
        from easyicu.webserver.data_package_review import (
            DataPackageReviewError,
            build_registered_data_package_review,
        )

        expected_digest = str(review_sha256 or "").strip().lower()
        try:
            payload = self.review_snapshot_store.load(
                study_id=str(study_context_id),
                revision=expected_revision,
                digest=expected_digest,
            )
        except DataPackageReviewError as exc:
            if exc.code != "data_package_review_snapshot_not_found":
                raise PiCopilotError(
                    exc.code,
                    exc.message,
                    status_code=409,
                    details=exc.details,
                ) from exc
            if expected_revision != current_revision:
                raise PiCopilotError(
                    "pi_data_package_review_snapshot_missing",
                    "The historical data-package review snapshot is unavailable.",
                    status_code=404,
                    details={
                        "expected_revision": expected_revision,
                        "current_revision": current_revision,
                        "review_sha256": expected_digest,
                    },
                ) from exc
            try:
                payload = build_registered_data_package_review(study)
            except DataPackageReviewError as build_exc:
                raise PiCopilotError(
                    build_exc.code,
                    build_exc.message,
                    status_code=409,
                    details=build_exc.details,
                ) from build_exc
            actual_digest = str(payload.get("review_sha256") or "")
            if expected_digest != actual_digest:
                raise PiCopilotError(
                    "pi_data_package_review_digest_mismatch",
                    "The requested data-package review no longer matches its owner digest.",
                    status_code=409,
                )
            try:
                self.review_snapshot_store.persist(payload)
            except DataPackageReviewError as persist_exc:
                raise PiCopilotError(
                    persist_exc.code,
                    persist_exc.message,
                    status_code=409,
                    details=persist_exc.details,
                ) from persist_exc
        except (ValueError, TypeError) as exc:
            raise PiCopilotError(
                "pi_data_package_review_snapshot_coordinates_invalid",
                "The requested data-package review coordinates are invalid.",
                status_code=409,
            ) from exc
        actual_digest = str(payload.get("review_sha256") or "")
        if expected_digest != actual_digest:
            raise PiCopilotError(
                "pi_data_package_review_digest_mismatch",
                "The requested data-package review no longer matches its owner digest.",
                status_code=409,
            )
        return {
            "ok": True,
            "payload": self._browser_artifact_payload(payload),
            "privacy": dict(payload.get("privacy") or {}),
            "governance": {
                "claim_ceiling": "pre_analysis_review",
                "reportable": False,
                "human_signoff": "not_applicable",
                "analysis_results_withheld": True,
            },
        }

    def get_data_workbench_snapshot(
        self,
        *,
        project_id: str,
        snapshot_sha256: str,
    ) -> Dict[str, Any]:
        """Open one exact project-scoped browser-only Data Workbench snapshot."""

        clean = self._assert_project_initialized(project_id)
        try:
            snapshot = self.data_workbench_snapshot_store.load(
                project_id=clean,
                digest=str(snapshot_sha256 or ""),
            )
        except CopilotDataWorkbenchError as exc:
            status_code = (
                404
                if exc.code == "copilot_data_workbench_snapshot_not_found"
                else 409
            )
            raise PiCopilotError(
                exc.code,
                exc.message,
                status_code=status_code,
                details=exc.details,
            ) from exc
        return {
            "ok": True,
            "view": str(snapshot.get("view") or ""),
            "payload": self._browser_artifact_payload(snapshot.get("payload") or {}),
            "privacy": dict(snapshot.get("privacy") or {}),
            "governance": {
                "claim_ceiling": "descriptive_review",
                "reportable": False,
                "human_signoff": "not_applicable",
                "browser_only": True,
                "model_visible_patient_payload": False,
            },
        }

    @staticmethod
    def _browser_artifact_payload(value: Any) -> Any:
        """Remove host-only paths while retaining bounded structured results.

        ``relative_path`` is the only path-shaped browser field. Future artifact
        schemas therefore fail closed instead of relying on an ever-growing
        blacklist of host path key names.
        """

        if isinstance(value, Mapping):
            projected: Dict[str, Any] = {}
            for key, child in value.items():
                normalized = str(key).strip().lower().replace("-", "_")
                if normalized in {"project_dir", "source_path", "cwd"}:
                    continue
                path_shaped = normalized in {
                    "path",
                    "directory",
                    "dir",
                    "root",
                    "file",
                } or normalized.endswith(
                    ("_path", "_directory", "_dir", "_root", "_file")
                )
                if path_shaped and normalized != "relative_path":
                    continue
                projected[str(key)] = PiCopilotService._browser_artifact_payload(child)
            return projected
        if isinstance(value, list):
            return [PiCopilotService._browser_artifact_payload(item) for item in value]
        return value

    def get_research_artifact(
        self,
        *,
        project_id: str,
        run_id: str,
        artifact_name: str,
    ) -> Dict[str, Any]:
        """Resolve a run artefact through project authority without exposing paths."""

        clean_project = self._assert_project_initialized(project_id)
        study_context_id = self.project_store.resolve(clean_project)
        clean_run = str(run_id or "").strip()
        clean_artifact = str(artifact_name or "").strip()
        rows = list_bound_run_history(
            study_context_id=study_context_id,
            project_root=research_pipeline_project_root(study_context_id),
            limit=200,
        )
        row = next(
            (
                item
                for item in rows
                if isinstance(item, Mapping) and item.get("run_id") == clean_run
            ),
            None,
        )
        if row is None:
            raise PiCopilotError(
                "pi_research_run_not_found",
                "The requested EasyICU run does not belong to this research project.",
                status_code=404,
                details={"run_id": clean_run},
            )
        loaded = agent_runs.read_run_artifact(
            str(row.get("project_dir") or ""),
            clean_artifact,
        )
        if not loaded.get("ok"):
            code = str(loaded.get("error") or "pi_research_artifact_unavailable")
            raise PiCopilotError(
                code,
                "The requested EasyICU run artefact is unavailable.",
                status_code=404 if code == "artifact_not_found" else 400,
                details={"artifact": clean_artifact},
            )
        privacy_scan = loaded.get("privacy_scan") or {}
        if not privacy_scan.get("passed"):
            raise PiCopilotError(
                "pi_research_artifact_privacy_blocked",
                "The artefact preview was withheld by the EasyICU privacy scan.",
                status_code=409,
                details={"artifact": clean_artifact},
            )
        payload = self._browser_artifact_payload(loaded.get("payload") or {})
        encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        if len(encoded) > MAX_RESEARCH_ARTIFACT_PREVIEW_BYTES:
            raise PiCopilotError(
                "pi_research_artifact_preview_too_large",
                "The artefact exceeds the bounded browser preview contract.",
                status_code=413,
                details={
                    "artifact": clean_artifact,
                    "bytes": len(encoded),
                    "max_bytes": MAX_RESEARCH_ARTIFACT_PREVIEW_BYTES,
                },
            )
        review = agent_runs.read_run_review(str(row.get("project_dir") or ""))
        if not review.get("ok"):
            raise PiCopilotError(
                str(
                    review.get("error") or "pi_research_artifact_governance_unavailable"
                ),
                "The EasyICU run governance state is unavailable for this artefact.",
                status_code=409,
                details={"artifact": clean_artifact},
            )
        governance = agent_runs.project_artifact_governance(
            review,
            artifact=loaded.get("artifact") or {},
        )
        if not governance.get("ok"):
            raise PiCopilotError(
                str(
                    governance.get("error") or "pi_research_artifact_governance_invalid"
                ),
                "The EasyICU run governance state is invalid for this artefact.",
                status_code=409,
                details={"artifact": clean_artifact},
            )
        metadata = loaded.get("artifact") or {}
        return {
            "ok": True,
            "run_id": clean_run,
            "artifact": {
                "name": metadata.get("name") or clean_artifact,
                "bytes": metadata.get("bytes"),
                "sha256": metadata.get("sha256"),
                "kind": metadata.get("kind") or "json",
                "media_type": "application/json",
            },
            "payload": payload,
            "privacy": {"passed": True},
            "governance": {
                key: value for key, value in governance.items() if key != "ok"
            },
        }

    def get_research_document(
        self,
        *,
        project_id: str,
        run_id: str,
        document_name: str,
    ) -> Dict[str, Any]:
        """Return one fixed, receipt-bound manuscript document for preview."""

        clean_project = self._assert_project_initialized(project_id)
        study_context_id = self.project_store.resolve(clean_project)
        clean_run = str(run_id or "").strip()
        clean_name = str(document_name or "").strip()
        rows = list_bound_run_history(
            study_context_id=study_context_id,
            project_root=research_pipeline_project_root(study_context_id),
            limit=200,
        )
        row = next(
            (
                item
                for item in rows
                if isinstance(item, Mapping) and item.get("run_id") == clean_run
            ),
            None,
        )
        if row is None:
            raise PiCopilotError(
                "pi_research_run_not_found",
                "The requested EasyICU run does not belong to this research project.",
                status_code=404,
                details={"run_id": clean_run},
            )
        project_dir = str(row.get("project_dir") or "")
        loaded = agent_runs.read_run_artifact_bytes(project_dir, clean_name)
        if not loaded.get("ok"):
            code = str(loaded.get("error") or "pi_research_document_unavailable")
            raise PiCopilotError(
                code,
                "The requested EasyICU manuscript document is unavailable.",
                status_code=404 if code == "artifact_not_found" else 400,
                details={"document": clean_name},
            )
        review = agent_runs.read_run_review(project_dir)
        if not review.get("ok"):
            raise PiCopilotError(
                str(
                    review.get("error") or "pi_research_document_governance_unavailable"
                ),
                "The EasyICU run governance state is unavailable for this document.",
                status_code=409,
                details={"document": clean_name},
            )
        artifact = next(
            (
                item
                for item in (review.get("artifacts") or [])
                if isinstance(item, Mapping) and item.get("name") == clean_name
            ),
            None,
        )
        if artifact is None:
            raise PiCopilotError(
                "pi_research_document_unregistered",
                "The manuscript document is not registered in this run ledger.",
                status_code=409,
                details={"document": clean_name},
            )
        payloads = review.get("artifact_payloads")
        payloads = payloads if isinstance(payloads, Mapping) else {}
        ledger = payloads.get("evidence_ledger.json")
        ledger = ledger if isinstance(ledger, Mapping) else {}
        registered = next(
            (
                item
                for item in (ledger.get("artifacts") or [])
                if isinstance(item, Mapping) and item.get("name") == clean_name
            ),
            None,
        )
        expected_sha256 = (
            str(registered.get("sha256") or "").lower()
            if isinstance(registered, Mapping)
            else ""
        )
        current_sha256 = hashlib.sha256(loaded["content"]).hexdigest()
        if (
            not re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
            or current_sha256 != expected_sha256
        ):
            raise PiCopilotError(
                "pi_research_document_digest_mismatch",
                "The requested EasyICU document does not match its run ledger binding.",
                status_code=409,
                details={"document": clean_name},
            )
        governance = agent_runs.project_artifact_governance(
            review,
            artifact={**artifact, "sha256": current_sha256},
        )
        if not governance.get("ok"):
            raise PiCopilotError(
                str(
                    governance.get("error") or "pi_research_document_governance_invalid"
                ),
                "The EasyICU governance projection is invalid for this document.",
                status_code=409,
                details={"document": clean_name},
            )
        return {
            "content": loaded["content"],
            "media_type": loaded["media_type"],
            "claim_ceiling": governance.get("claim_ceiling") or "unsupported",
        }

    def _public_session(
        self,
        record: PiSessionRecord,
        *,
        gateway_state: Optional[Mapping[str, Any]] = None,
        include_replay: bool = True,
        replay_cursor: Optional[str] = None,
        replay_limit: int = 48,
    ) -> Dict[str, Any]:
        state = gateway_state or {}
        replay = (
            self.replay_store.snapshot(
                session_id=record.session_id,
                project_id=str(record.project_id),
                cursor=replay_cursor,
                limit=replay_limit,
            )
            if include_replay
            else {"turns": [], "turn_page": {}, "child_jobs": []}
        )
        replay_turns = replay.get("turns") or []
        latest_replay_turn = (
            replay_turns[-1]
            if replay_turns and isinstance(replay_turns[-1], Mapping)
            else {}
        )
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
                list(state.get("transcript") or [])[:200]
                if isinstance(state.get("transcript"), list)
                else []
            ),
            "transcript_page": (
                dict(state.get("transcript_page") or {})
                if isinstance(state.get("transcript_page"), Mapping)
                else {}
            ),
            "shell_usage": (
                dict(state.get("shell_usage") or {})
                if isinstance(state.get("shell_usage"), Mapping)
                else {}
            ),
            "binding": record.binding.model_dump(mode="json"),
            "data_source_authorization": record.data_source_authorization.model_dump(
                mode="json"
            ),
            "research_provider": record.research_provider.public_projection(),
            "model_connection": (
                record.research_provider.public_projection()
                if record.uses_unified_model_connection
                else None
            ),
            "stale": self._stale_details(record),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "last_message_job_id": record.last_message_job_id,
            "active_message_job_id": record.active_message_job_id,
            "last_turn_status": record.last_turn_status,
            "last_turn_allowed_actions": list(record.last_turn_allowed_actions),
            "last_turn_events": list(latest_replay_turn.get("events") or []),
            "conversation_replay": replay,
            "archived_child_jobs": list(replay.get("child_jobs") or []),
            "pinned_for_presentation": record.pinned_for_presentation,
            "session_storage": "private_local_jsonl",
            "scientific_authority": "EasyICU",
            "extension_activation": {
                "schema_version": record.extension_activation.schema_version,
                "activation_sha256": record.extension_activation.activation_sha256,
                "revision": record.extension_activation.revision,
                "skills": [
                    {
                        "name": item.name,
                        "description": item.description,
                        "digest": item.digest,
                        "stages": list(item.stages),
                    }
                    for item in record.extension_activation.skills
                ],
                "mcp_servers": [
                    {
                        "name": item.name,
                        "transport": item.transport,
                        "allowed_tools": list(item.allowed_tools),
                    }
                    for item in record.extension_activation.mcp_servers
                ],
            },
            "project_migration": (
                binding.migration_receipt.model_dump(mode="json")
                if (
                    (binding := self.project_store.binding(record.project_id))
                    and binding.migration_receipt
                )
                else None
            ),
        }

    def _reconcile_replay_execution(self, record: PiSessionRecord) -> PiSessionRecord:
        """Reconcile persisted UX continuity with the process-local JobManager."""

        active_job_id = str(record.active_message_job_id or "").strip()
        if not active_job_id or record.last_turn_status != "running":
            return record
        job = jobs.MANAGER.get(active_job_id)
        if job is not None and job.status == "running":
            return record
        record.active_message_job_id = None
        if job is not None and job.status in {"done", "failed", "cancelled"}:
            record.last_turn_status = job.status
        else:
            # The generic JobManager is intentionally process-local. A persisted
            # running pointer with no matching job therefore means the Web
            # process restarted; never leave the presentation showing "running".
            record.last_turn_status = "interrupted"
        saved = self._save_record(record)
        self.replay_store.finish_turn(
            session_id=record.session_id,
            project_id=str(record.project_id),
            job_id=active_job_id,
            status=str(record.last_turn_status),
        )
        return saved

    def close(self) -> None:
        self.gateway.close()
        self.codex_gateway_pool.close()


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
