"""Authoritative FastAPI-side owner for Pi Copilot sessions and turns."""

from __future__ import annotations

import hashlib
import hmac
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
    agent_runs,
    guided_sessions,
    jobs,
    primary_cohort,
    provider_gate,
    settings,
    source_identity_authority,
    sources,
    study_contexts,
)
from easyicu.webserver.data_package_review import DataPackageReviewSnapshotStore
from easyicu.webserver.copilot_data_workbench import (
    CopilotDataWorkbenchError,
    CopilotDataWorkbenchSnapshotStore,
    build_snapshot as build_data_workbench_snapshot,
)

from .plan_projection import project_plan_reader_fields
from . import cohort_eligibility, plan_decisions, plan_review_progress
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
from .message_input import prepare_user_message
from .project_authority import (
    ProjectAuthorityStore,
    ProjectStudyContextMigrationReceipt,
)
from .provider_config import PiProviderConfigStore
from .resource_lifecycle import WebMemoryAdmission
from .projections import (
    project_job,
    project_pi_replay_event,
    project_transcript,
)
from .user_visible_text import sanitize_user_visible_text
from .replay_store import PiConversationReplayStore
from .session_storage import SessionStorageMaintenance
from .run_authority import (
    latest_bound_run_id,
    list_bound_run_history,
    research_pipeline_project_root,
)
from .turn_authority import (
    explicitly_confirms_easyicu_registered_source,
    infer_explicit_turn_actions,
    infer_idea_mining_followup_intent,
    infer_research_entry_intent,
)
from .tools import extraction_workspace_resource
from .workspace import ProjectWorkspace
from .workflow import (
    build_project_workflow_projection,
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
HOST_ACTION_JOB_KINDS = {
    "auto_generate_plan": frozenset({"agent-run"}),
    "generate_plan": frozenset({"agent-run"}),
    "auto_revise_plan": frozenset({"agent-run"}),
    "prepare_analysis_data": frozenset({"agent-run"}),
    "execute_plan": frozenset({"agent-run"}),
    "retry_analysis": frozenset({"agent-run"}),
}
HOST_ACTIONS_WITHOUT_JOBS = frozenset(
    {
        "review_prepared_data",
        "review_results",
        "review_result_tables",
        "review_figures",
        "review_manuscript",
        "review_scientific_review",
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

_AUTO_SESSION_TITLE_SUFFIXES = (
    " · Research",
    " · 研究",
    " · Workspace",
    " · 工作区",
)


def _first_message_session_title(message: str) -> str:
    """Create a short local conversation label from the researcher's first turn."""

    compact = " ".join(str(message or "").split())
    if len(compact) <= 52:
        return compact
    return compact[:51].rstrip() + "…"


def _session_title_is_automatic(title: str) -> bool:
    value = str(title or "").strip()
    return value == "EasyICU Copilot" or value.endswith(_AUTO_SESSION_TITLE_SUFFIXES)


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
        memory_admission: Optional[WebMemoryAdmission] = None,
        storage_maintenance: Optional[SessionStorageMaintenance] = None,
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
        session_root = getattr(self.gateway, "session_dir", None)
        if session_root is None:
            session_root = self.store_path.parent / "pi-agent" / "sessions"
        self.memory_admission = memory_admission or WebMemoryAdmission()
        self.storage_maintenance = storage_maintenance or SessionStorageMaintenance(
            Path(session_root)
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
        self._opening_sessions = 0
        self._replay_child_watchers: set[tuple[str, str]] = set()
        self._project_initialization_locks: Dict[str, threading.RLock] = {}
        # Outstanding browser coordinates are deliberately process-local. A
        # restart invalidates every unconsumed option instead of accepting a
        # stale click against newly loaded scientific state.
        self._cohort_selection_secret = secrets.token_bytes(32)
        workspace_base = Path(
            getattr(
                self.gateway,
                "declared_cwd",
                self.store_path.parent / "pi_project_workspace",
            )
        )
        self.workspace_root = workspace_base.expanduser().absolute()
        self.workspace = ProjectWorkspace(self.workspace_root)

    def _admit_new_work(
        self,
        *,
        gateway: Optional[Any] = None,
        exclude_session_id: str = "",
    ) -> Dict[str, Any]:
        status = self.memory_admission.status()
        if status.get("pressure") in {"soft", "emergency"}:
            selected_gateway = gateway or self.gateway
            if selected_gateway is self.gateway:
                maintain = getattr(self.gateway, "maintain_sessions", None)
                if callable(maintain):
                    maintain(exclude_session_id=exclude_session_id)
            maintain_pool = getattr(self.codex_gateway_pool, "maintain_sessions", None)
            if callable(maintain_pool):
                maintain_pool(exclude_session_id=exclude_session_id)
        return self.memory_admission.require_capacity()

    def resource_status(self) -> Dict[str, Any]:
        """Return path-free process, hot-session, and transcript diagnostics."""

        with self._lock:
            records = self._read_records()
            opening = self._opening_sessions
            busy = len(self._busy_sessions)
        referenced = [
            row.pi_session_file for row in records if row.pi_session_file
        ]
        storage = self.storage_maintenance.inventory(referenced).public_projection()
        gateway_status = getattr(self.gateway, "memory_status", None)
        api_gateway = (
            gateway_status()
            if callable(gateway_status)
            else {"running": False, "diagnostics_available": False}
        )
        memory_statuses = getattr(self.codex_gateway_pool, "memory_statuses", None)
        return {
            "ok": True,
            "memory": self.memory_admission.status(),
            "sessions": {
                "retained": len(records),
                "busy": busy,
                "opening": opening,
                "pinned": sum(row.pinned_for_presentation for row in records),
            },
            "gateways": {
                "api": api_gateway,
                "codex": memory_statuses() if callable(memory_statuses) else [],
            },
            "storage": storage,
        }

    def maintain_session_storage(
        self,
        *,
        action: str,
        confirm: bool = False,
        quarantine_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Audit, quarantine, or restore private transcripts without deletion."""

        clean_action = str(action or "").strip()
        with self._lock:
            if self._opening_sessions or self._busy_sessions:
                raise PiCopilotError(
                    "pi_session_maintenance_busy",
                    "Transcript maintenance requires every Copilot turn to be idle.",
                    status_code=409,
                )
            records = self._read_records()
            referenced = [
                row.pi_session_file for row in records if row.pi_session_file
            ]
            if clean_action == "audit":
                result: Dict[str, Any] = {
                    "status": "audited",
                    "inventory": self.storage_maintenance.inventory(
                        referenced
                    ).public_projection(),
                }
            elif clean_action == "quarantine":
                result = self.storage_maintenance.quarantine(
                    referenced,
                    confirm=bool(confirm),
                )
            elif clean_action == "restore":
                result = self.storage_maintenance.restore(
                    str(quarantine_id or ""),
                    confirm=bool(confirm),
                )
            else:
                raise PiCopilotError(
                    "pi_session_maintenance_action_invalid",
                    "Transcript maintenance action must be audit, quarantine, or restore.",
                    status_code=422,
                )
        return {"ok": True, **result}

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
        gateway: Any = self.gateway
        try:
            gateway = self._conversation_gateway(record)
        except (OSError, PiCopilotError):
            pass
        self.replay_store.retire(record.session_id)
        self._dispose_gateway_session(
            gateway,
            session_id=record.session_id,
            session_file=record.pi_session_file,
        )

    @staticmethod
    def _dispose_gateway_session(
        gateway: Any,
        *,
        session_id: str,
        session_file: Optional[str],
    ) -> None:
        try:
            gateway.request(
                "session.dispose",
                {"session_id": session_id},
                timeout=5,
            )
        except (OSError, PiCopilotError):
            pass
        session_root = getattr(
            gateway,
            "session_dir",
            None,
        )
        if session_root is None or not session_file:
            return
        candidate = Path(session_file).resolve()
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

    def _cohort_selection_event_id(
        self,
        record: PiSessionRecord,
        context: Mapping[str, Any],
        option: Mapping[str, Any],
        *,
        purpose: str = "event",
    ) -> str:
        payload = {
            "schema_version": cohort_eligibility.SELECTION_EVENT_SCHEMA_VERSION,
            "purpose": str(purpose),
            "session_id": record.session_id,
            "study_context_id": str(context.get("id") or ""),
            "expected_revision": int(context.get("revision") or 0),
            "user_turn_id": self._cohort_selection_user_turn_id(record, context),
            "option_id": str(option.get("id") or ""),
            "primary_cohort_contract_sha256": str(
                option.get("primary_cohort_contract_sha256") or ""
            ),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hmac.new(
            self._cohort_selection_secret,
            encoded,
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _cohort_selection_user_turn_id(
        record: PiSessionRecord,
        context: Mapping[str, Any],
    ) -> str:
        return str(
            record.last_message_job_id
            or (
                f"host_revision_{record.session_id}_"
                f"{int(context.get('revision') or 0)}"
            )
        )

    def _cohort_eligibility_selection_projection(
        self,
        record: PiSessionRecord,
    ) -> Dict[str, Any]:
        """Return host-issued option coordinates; never project them to Pi."""

        context_id = str(record.binding.study_context_id or "").strip()
        if record.agent_mode == "workspace" or not context_id:
            return {"present": False}
        context = study_contexts.get_context(context_id)
        if context is None:
            return {"present": False}
        proposal = cohort_eligibility.eligibility_proposal(context)
        options = []
        for option in proposal.get("options") or []:
            projected = dict(option)
            projected["expected_revision"] = int(context.get("revision") or 0)
            projected["selection_event_id"] = self._cohort_selection_event_id(
                record,
                context,
                option,
            )
            options.append(projected)
        return {
            "present": True,
            "stated": bool(proposal.get("stated")),
            "selection_state": proposal.get("selection_state"),
            "authority_status": proposal.get("authority_status"),
            "blocker_code": proposal.get("blocker_code"),
            "primary_cohort_contract": proposal.get("primary_cohort_contract"),
            "primary_cohort_contract_sha256": proposal.get(
                "primary_cohort_contract_sha256"
            ),
            "options": options,
        }

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
        if source is not None:
            return PiSessionDataSourceAuthorization(
                status="confirmed",
                reason=None,
                confirmation_mode="agent_default_study_required",
                extraction_scope="study_required",
                source=source,
                confirmed_at=utc_now(),
            )
        return PiSessionDataSourceAuthorization(
            status="pending",
            reason="local_data_selection_required",
            confirmation_mode=None,
            source=None,
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
        authorization = record.data_source_authorization
        if (
            record.agent_mode == "research"
            and authorization.status == "pending"
            and authorization.reason == "project_source_confirmation_required"
            and not self._stale_details(record).get("stale")
        ):
            context = study_contexts.get_context(record.binding.study_context_id)
            source = self._session_source_reference(context or {})
            if source is not None:
                record.data_source_authorization = PiSessionDataSourceAuthorization(
                    status="confirmed",
                    reason=None,
                    confirmation_mode="agent_default_study_required",
                    extraction_scope="study_required",
                    source=source,
                    confirmed_at=utc_now(),
                )
                self._save_record(record)
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
        self._admit_new_work(gateway=conversation_gateway)
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
        state: Dict[str, Any] = {}
        record: Optional[PiSessionRecord] = None
        session_created = False
        with self._lock:
            self._opening_sessions += 1
        try:
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
            session_created = True
            record = PiSessionRecord(
                session_id=session_id,
                project_id=clean_project_id,
                pi_session_id=str(state.get("pi_session_id") or "") or None,
                pi_session_file=str(state.get("session_file") or "") or None,
                title=str(title or "EasyICU Copilot").strip()[:160]
                or "EasyICU Copilot",
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
        except Exception:
            if session_created:
                self._dispose_gateway_session(
                    conversation_gateway,
                    session_id=session_id,
                    session_file=str(state.get("session_file") or "") or None,
                )
            raise
        finally:
            with self._lock:
                self._opening_sessions = max(0, self._opening_sessions - 1)
        assert record is not None
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

    @staticmethod
    def _validated_regenerate_study_context_snapshot(
        record: PiSessionRecord,
        regenerate_target: Mapping[str, Any],
    ) -> tuple[str, Mapping[str, Any]]:
        """Validate one replay snapshot without changing persisted state."""

        snapshot = regenerate_target.get("study_context_snapshot")
        if not isinstance(snapshot, Mapping):
            raise PiCopilotError(
                "pi_regenerate_study_context_snapshot_missing",
                (
                    "This conversation turn has no authoritative StudyContext "
                    "snapshot, so EasyICU cannot safely replace its branch."
                ),
                status_code=409,
            )
        context_id = str(record.binding.study_context_id or "").strip()
        if not context_id:
            raise PiCopilotError(
                "pi_regenerate_study_context_snapshot_missing",
                "The conversation has no bound StudyContext to restore.",
                status_code=409,
            )
        return context_id, snapshot

    def _restore_regenerate_study_context(
        self,
        record: PiSessionRecord,
        regenerate_target: Mapping[str, Any],
    ) -> PiSessionRecord:
        """Rewind EasyICU scientific state before replacing a Pi branch."""

        context_id, snapshot = self._validated_regenerate_study_context_snapshot(
            record,
            regenerate_target,
        )
        try:
            restored = study_contexts.restore_turn_configuration_snapshot(
                context_id,
                snapshot,
                expected_revision=int(record.binding.study_revision),
            )
        except study_contexts.StudyContextError as exc:
            raise PiCopilotError(
                str(
                    exc.detail.get("error")
                    or "pi_regenerate_study_context_restore_failed"
                ),
                "EasyICU could not restore the scientific state for this conversation branch.",
                status_code=409,
                details=exc.detail,
            ) from exc

        record.binding = self._binding_for_context(
            restored,
            run_id=self._latest_run_id(
                context_id,
                project_id=record.project_id,
            ),
        )
        if record.agent_mode == "research":
            prior_authorization = record.data_source_authorization
            current_source = self._session_source_reference(restored)
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
                    restored,
                    agent_mode=record.agent_mode,
                )
        self._save_record(record)
        return record

    def authorize_data_source(
        self,
        session_id: str,
        *,
        project_id: str,
        action: str,
        database: Optional[str] = None,
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
            "use_study_required_data",
            "begin_local_selection",
            "begin_full_data_selection",
            "confirm_selected_source",
        }:
            raise PiCopilotError(
                "pi_session_data_source_action_invalid",
                "Choose a supported data-source confirmation action.",
                status_code=422,
            )
        clean_database = str(database or "").strip()
        expected_database = None
        if clean_database:
            if clean_action not in {
                "begin_local_selection",
                "begin_full_data_selection",
            }:
                raise PiCopilotError(
                    "pi_data_source_database_not_allowed",
                    "A database may be supplied only when local selection begins.",
                    status_code=422,
                )
            try:
                profile = get_database_profile(clean_database)
            except KeyError as exc:
                raise PiCopilotError(
                    "pi_database_not_supported",
                    "Choose one exact supported ICU database.",
                    status_code=422,
                ) from exc
            if not profile.is_public:
                raise PiCopilotError(
                    "pi_database_not_supported",
                    "Local selection requires a supported full-database family.",
                    status_code=422,
                )
            expected_database = profile.key
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

        if clean_action in {"begin_local_selection", "begin_full_data_selection"}:
            extraction_scope = (
                "all_supported"
                if clean_action == "begin_full_data_selection"
                else "study_required"
            )
            record.data_source_authorization = PiSessionDataSourceAuthorization(
                status="selection_in_progress",
                reason="local_data_selection_required",
                confirmation_mode="select_local_source",
                extraction_scope=extraction_scope,
            )
            self._save_record(record)
            return {
                "ok": True,
                "session": self._public_session(record),
                "resource": extraction_workspace_resource(
                    context,
                    state="setup",
                    expected_database=expected_database,
                    entry_mode="source_binding",
                    extraction_scope=extraction_scope,
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
            extraction_scope = record.data_source_authorization.extraction_scope
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
            extraction_scope = (
                "study_required"
                if clean_action == "use_study_required_data"
                else "reuse_prepared_full"
            )
        record.data_source_authorization = PiSessionDataSourceAuthorization(
            status="confirmed",
            reason=None,
            confirmation_mode=confirmation_mode,
            extraction_scope=extraction_scope,
            source=source,
            confirmed_at=utc_now(),
        )
        self._save_record(record)
        return {
            "ok": True,
            "session": self._public_session(record),
            "resource": (
                extraction_workspace_resource(
                    context,
                    state="setup",
                    expected_database=source.database,
                    extraction_scope="all_supported",
                )
                if extraction_scope == "all_supported"
                else None
            ),
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

    def _bind_registered_source_from_message(
        self,
        record: PiSessionRecord,
        *,
        source: Mapping[str, Any],
    ) -> PiSessionRecord:
        """Bind an exact registry path supplied by the user, before provider use."""

        context_id = str(record.binding.study_context_id or "").strip()
        if not context_id:
            raise PiCopilotError(
                "pi_message_study_context_required",
                "Create a research project before selecting its local data source.",
                status_code=409,
            )
        try:
            current = study_contexts.get_context(context_id)
        except study_contexts.StudyContextError as exc:
            raise PiCopilotError(
                str(exc.detail.get("error") or "study_context_invalid"),
                "The bound StudyContext could not be loaded.",
                status_code=409,
                details=exc.detail,
            ) from exc
        if not current:
            raise PiCopilotError(
                "study_context_not_found",
                "The bound StudyContext no longer exists.",
                status_code=409,
            )
        if current.get("active_job_id"):
            raise PiCopilotError(
                "study_context_active_job_conflict",
                "The local data source cannot change while its EasyICU job is active.",
                status_code=409,
            )
        source_path = str(source.get("path") or "").strip()
        database = str(source.get("database") or "").strip()
        if not source_path or not database or not bool(source.get("ok")):
            raise PiCopilotError(
                "pi_message_local_source_invalid",
                "The selected registry row is not a validated EasyICU data source.",
                status_code=409,
            )
        confirmations = dict(current.get("confirmations") or {})
        confirmations["extraction_completed"] = True
        try:
            updated = study_contexts.upsert_context(
                {
                    "id": context_id,
                    "data_source": {
                        "path": source_path,
                        "label": str(source.get("label") or "local EasyICU data")[:160],
                        "database": database,
                    },
                    "confirmations": confirmations,
                },
                active=True,
                expected_revision=int(current.get("revision") or 0),
                require_revision=True,
                lifecycle_write=False,
            )
        except study_contexts.StudyContextError as exc:
            raise PiCopilotError(
                str(exc.detail.get("error") or "study_context_update_blocked"),
                "The StudyContext owner rejected the local data-source binding.",
                status_code=409,
                details=exc.detail,
            ) from exc
        record.binding = self._binding_for_context(
            updated,
            run_id=self._latest_run_id(context_id, project_id=record.project_id),
        )
        source_reference = self._session_source_reference(updated)
        if source_reference is None:
            raise PiCopilotError(
                "pi_message_local_source_reference_unavailable",
                "The selected data source could not be projected into session authority.",
                status_code=409,
            )
        record.data_source_authorization = PiSessionDataSourceAuthorization(
            status="confirmed",
            reason=None,
            confirmation_mode="select_local_source",
            extraction_scope="reuse_prepared_full",
            source=source_reference,
            confirmed_at=utc_now(),
        )
        self._save_record(record)
        return record

    def send_message(
        self,
        session_id: str,
        *,
        project_id: str,
        message: str,
        allowed_actions: Iterable[str] = (),
        regenerate_user_entry_id: Optional[str] = None,
        regeneration_intent: Optional[str] = None,
        message_intent: Optional[str] = None,
        idea_source: Optional[Mapping[str, Any]] = None,
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
        prepared_input = prepare_user_message(
            text,
            registered_sources=sources.load_registry().get("sources") or [],
        )
        provider_text = prepared_input.provider_message
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
        if prepared_input.registered_source is not None:
            record = self._bind_registered_source_from_message(
                record,
                source=prepared_input.registered_source,
            )
        # A prepared source that is already bound to this project can be
        # confirmed before the provider turn.  This lets one explicit user
        # choice both unlock the data gate and advance to the requested data
        # action, instead of forcing a second identical confirmation click.
        # The helper still requires an exact validated registry-path match.
        prior_authorization = record.data_source_authorization.model_dump(mode="json")
        self._confirm_registered_source_selected_in_turn(
            record,
            user_message=provider_text,
        )
        if (
            record.data_source_authorization.model_dump(mode="json")
            != prior_authorization
        ):
            self._save_record(record)
        requested_actions = frozenset(
            str(item).strip() for item in allowed_actions if str(item).strip()
        ) | infer_explicit_turn_actions(provider_text)
        unknown_actions = sorted(requested_actions - ALLOWED_TURN_ACTIONS)
        if unknown_actions:
            raise PiCopilotError(
                "pi_action_unknown",
                "The message requested an unknown host action capability.",
                details={"actions": unknown_actions},
            )
        self._admit_new_work(
            gateway=self._conversation_gateway(record),
            exclude_session_id=record.session_id,
        )
        opened_state = self._ensure_open(record)
        if not regenerate_user_entry_id and not message_intent:
            workflow = build_project_workflow_projection(
                study_context_id=record.binding.study_context_id,
            )
            if workflow.workflow.current_stage == "idea":
                message_intent = infer_idea_mining_followup_intent(provider_text)
        if (
            not regenerate_user_entry_id
            and not message_intent
            and int(opened_state.get("message_count") or 0) == 0
        ):
            workflow = build_project_workflow_projection(
                study_context_id=record.binding.study_context_id,
            )
            if workflow.workflow.current_stage == "idea":
                message_intent = infer_research_entry_intent(provider_text)
        if (
            not regenerate_user_entry_id
            and int(opened_state.get("message_count") or 0) == 0
            and _session_title_is_automatic(record.title)
        ):
            record.title = _first_message_session_title(provider_text)
            self._save_record(record)
        regenerate_target: Optional[Dict[str, Any]] = None
        if regenerate_user_entry_id:
            regenerate_target = self._conversation_gateway(
                record,
                refresh_account=True,
            ).request(
                "session.regenerate.inspect",
                {
                    "session_id": record.session_id,
                    "user_entry_id": str(regenerate_user_entry_id),
                },
                timeout=30,
            )
            # Regenerating must re-run exactly what the user said, so a drifted
            # transcript is a conflict.  An explicit user edit is the opposite
            # operation: the text is meant to change, and the branch rewinds to
            # this turn so everything after it is replaced rather than appended.
            edited_regeneration_intents = {
                "user_edited_message",
                "replace_plan_response_preserve_study",
            }
            if (
                regeneration_intent not in edited_regeneration_intents
                and str(regenerate_target.get("message") or "").strip()
                != provider_text
            ):
                raise PiCopilotError(
                    "pi_regenerate_message_mismatch",
                    "The regenerate request no longer matches the active transcript.",
                    status_code=409,
                )
            self._validated_regenerate_study_context_snapshot(
                record,
                regenerate_target,
            )
        with self._lock:
            if record.session_id in self._busy_sessions:
                raise PiCopilotError(
                    "pi_session_busy",
                    "This EasyICU Copilot session already has an active message.",
                    status_code=409,
                )
            self._busy_sessions.add(record.session_id)

        start_gate = threading.Event()
        start_abort: Dict[str, str] = {}
        tool_context: Optional[ToolExecutionContext] = None

        def runner(job: jobs.Job) -> Dict[str, Any]:
            # JobManager starts its thread immediately.  Hold the accepted job
            # until the synchronous branch restore and tool-authority snapshot
            # are ready; rejected busy/capacity requests never reach either
            # mutation.
            start_gate.wait()
            if start_abort:
                raise RuntimeError(
                    str(start_abort.get("error") or "pi_message_setup_failed")
                )
            active_tool_context = tool_context
            if active_tool_context is None:
                raise RuntimeError("pi_message_tool_context_missing")
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
            if regenerate_target is not None:
                self.replay_store.supersede_from_turn_index(
                    session_id=record.session_id,
                    project_id=str(record.project_id),
                    turn_index=int(regenerate_target.get("turn_index") or 0),
                )
            self.replay_store.start_turn(
                session_id=record.session_id,
                project_id=str(record.project_id),
                job_id=job.id,
                allowed_actions=sorted(requested_actions),
            )

            def emit(event: Dict[str, Any]) -> None:
                if job.cancel_requested:
                    return
                # Best effort on the live stream: a token split across two
                # deltas still slips through, but the authoritative transcript
                # replaces this text at turn end and is sanitized in full. Done
                # here rather than in the shared job event transport, which
                # carries every kind of job and must not learn a Copilot rule.
                if event.get("type") == "text_delta" and isinstance(
                    event.get("delta"), str
                ):
                    event = {
                        **event,
                        "delta": sanitize_user_visible_text(event["delta"]),
                    }
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
                method = "session.regenerate" if regenerate_target is not None else "session.prompt"
                params = {"session_id": record.session_id, "message": provider_text}
                if idea_source:
                    params["idea_source"] = dict(idea_source)
                if message_intent:
                    params[
                        "turn_intent" if regenerate_target is not None else "intent"
                    ] = str(message_intent)
                if regenerate_target is not None:
                    params["user_entry_id"] = str(regenerate_user_entry_id)
                    if regeneration_intent:
                        params["intent"] = str(regeneration_intent)
                state = conversation_gateway.request(
                    method,
                    params,
                    timeout=COPILOT_MESSAGE_TIMEOUT_SECONDS,
                    event_sink=emit,
                    tool_context=active_tool_context,
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
                    user_message=provider_text,
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
        try:
            if (
                regenerate_target is not None
                and regeneration_intent != "replace_plan_response_preserve_study"
            ):
                record = self._restore_regenerate_study_context(
                    record,
                    regenerate_target,
                )
            tool_context = ToolExecutionContext(
                session=record.model_copy(deep=True),
                user_message=text,
                idea_source=idea_source,
                allowed_actions=requested_actions,
                authority_validator=lambda binding: self._binding_stale_details(
                    binding,
                    project_id=record.project_id,
                ),
                workspace=self.workspace,
                extension_registry=self.extension_registry,
            )
        except Exception as exc:
            start_abort["error"] = str(
                getattr(exc, "code", "") or "pi_message_setup_failed"
            )
            with self._lock:
                self._busy_sessions.discard(record.session_id)
            start_gate.set()
            raise
        start_gate.set()
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
                        self.replay_store.finish_host_actions_for_child_job(
                            session_id=session_id,
                            project_id=project_id,
                            child_job_id=job_id,
                            status=child.status,
                        )
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

    def record_host_action(
        self,
        session_id: str,
        *,
        project_id: str,
        action_code: str,
        action_key: str,
        child_job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bind one explicit host-UI action to the durable conversation replay."""

        record = self._scoped_record(session_id, project_id=project_id)
        action = str(action_code or "").strip()
        key = str(action_key or "").strip()
        child_id = str(child_job_id or "").strip()
        expected_kinds = HOST_ACTION_JOB_KINDS.get(action)
        if expected_kinds is None and action not in HOST_ACTIONS_WITHOUT_JOBS:
            raise PiCopilotError(
                "pi_host_action_unsupported",
                "This host conversation action is not supported.",
                status_code=400,
            )
        if not key:
            raise PiCopilotError(
                "pi_host_action_key_required",
                "The host conversation action is missing its stable key.",
                status_code=400,
            )
        child = None
        if expected_kinds is not None:
            if not child_id:
                raise PiCopilotError(
                    "pi_host_action_child_job_required",
                    "This host conversation action requires its child job.",
                    status_code=400,
                )
            child = jobs.MANAGER.get(child_id)
            if child is None or child.kind not in expected_kinds:
                raise PiCopilotError(
                    "pi_host_action_child_job_invalid",
                    "The host conversation action does not match this child job.",
                    status_code=409,
                )
        elif child_id:
            raise PiCopilotError(
                "pi_host_action_child_job_forbidden",
                "This host conversation action must not bind a child job.",
                status_code=400,
            )
        action_id = "host_" + hashlib.sha256(
            f"{record.session_id}\0{action}\0{key}".encode("utf-8")
        ).hexdigest()[:24]
        child_status = str(child.status) if child is not None else "done"
        status = child_status if child_status in {"done", "failed", "cancelled"} else "running"
        turn = self.replay_store.record_host_action(
            session_id=record.session_id,
            project_id=str(record.project_id),
            action_id=action_id,
            action_code=action,
            action_key=key,
            child_job_id=child_id or None,
            status=status,
        )
        if child is not None:
            if status == "running":
                self._watch_child_job_for_replay(
                    session_id=record.session_id,
                    project_id=str(record.project_id),
                    job_id=child_id,
                )
            else:
                self.replay_store.archive_child_job(
                    session_id=record.session_id,
                    project_id=str(record.project_id),
                    job=project_job(child.snapshot()),
                )
        return {"ok": True, "host_action": turn}

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
        if child.status in {"done", "failed", "cancelled"}:
            self.replay_store.finish_host_actions_for_child_job(
                session_id=record.session_id,
                project_id=str(record.project_id),
                child_job_id=job_id,
                status=child.status,
            )
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
        sessions: list[Dict[str, Any]] = []
        for row in records:
            public = self._public_session(row, include_replay=False)
            replay = self.replay_store.snapshot(
                session_id=row.session_id,
                project_id=clean_project_id,
                limit=1,
            )
            turn_page = replay.get("turn_page") or {}
            public["history_turn_count"] = int(turn_page.get("total") or 0)
            sessions.append(public)
        return {
            "ok": True,
            "count": len(sessions),
            "sessions": sessions,
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

    def confirm_cohort_eligibility(
        self,
        session_id: str,
        *,
        project_id: str,
        option_id: str,
        expected_revision: int,
        primary_cohort_contract_sha256: str,
        selection_event_id: str,
    ) -> Dict[str, Any]:
        """Consume one host-rendered cohort option without a provider turn."""

        record = self._scoped_record(session_id, project_id=project_id)
        stale = self._stale_details(record)
        if stale.get("stale"):
            raise PiCopilotError(
                "pi_session_authority_stale",
                "The StudyContext changed before this cohort option was selected.",
                status_code=409,
                details=stale,
            )
        context_id = str(record.binding.study_context_id or "").strip()
        context = study_contexts.get_context(context_id) if context_id else None
        if context is None:
            raise PiCopilotError(
                "cohort_eligibility_study_context_required",
                "The host must allocate and bind a StudyContext before eligibility confirmation.",
                status_code=409,
            )
        if context.get("active_job_id"):
            raise PiCopilotError(
                "study_context_active_job_conflict",
                "Cohort eligibility cannot change while an EasyICU job is active.",
                status_code=409,
            )
        current_revision = int(context.get("revision") or 0)
        if expected_revision != current_revision:
            raise PiCopilotError(
                "cohort_eligibility_selection_revision_conflict",
                "The displayed cohort option is stale; reload the current contract.",
                status_code=409,
                details={
                    "expected_revision": expected_revision,
                    "current_revision": current_revision,
                },
            )
        try:
            options = cohort_eligibility.selection_options_for_study(context)
        except primary_cohort.PrimaryCohortContractError as exc:
            raise PiCopilotError(
                exc.code,
                "The current primary-cohort contract is not executable.",
                status_code=409,
                details={"field": "cohort", **dict(exc.detail)},
            ) from exc
        selected = next(
            (
                option
                for option in options
                if str(option.get("id") or "") == str(option_id or "").strip()
            ),
            None,
        )
        if selected is None:
            raise PiCopilotError(
                "cohort_eligibility_option_unknown",
                "The selected cohort option is not available for the current contract.",
                status_code=409,
            )
        canonical_digest = str(
            selected.get("primary_cohort_contract_sha256") or ""
        )
        if canonical_digest != str(primary_cohort_contract_sha256 or ""):
            raise PiCopilotError(
                "cohort_eligibility_selection_scope_mismatch",
                "The selected cohort preview no longer matches current execution semantics.",
                status_code=409,
            )
        expected_event_id = self._cohort_selection_event_id(
            record,
            context,
            selected,
        )
        if not hmac.compare_digest(
            expected_event_id,
            str(selection_event_id or ""),
        ):
            raise PiCopilotError(
                "cohort_eligibility_selection_event_invalid",
                "The cohort confirmation was not issued by the current host session.",
                status_code=409,
            )
        one_use_grant_id = self._cohort_selection_event_id(
            record,
            context,
            selected,
            purpose="one_use_grant",
        )
        actor_id_sha256 = hashlib.sha256(
            f"local_interactive_user:{record.session_id}".encode("utf-8")
        ).hexdigest()
        event = cohort_eligibility.build_selection_event(
            option_id=option_id,
            study_context_id=context_id,
            expected_revision=current_revision,
            session_id=record.session_id,
            user_turn_id=self._cohort_selection_user_turn_id(record, context),
            event_id=expected_event_id,
            one_use_grant_id=one_use_grant_id,
            primary_cohort_contract_sha256=canonical_digest,
            actor_id_sha256=actor_id_sha256,
        )
        target_cohort = cohort_eligibility.selection_cohort_for_option(
            context,
            option_id,
        )
        authority = cohort_eligibility.confirmation_authority_for_option(
            option_id,
            study_context_id=context_id,
            study_context_revision=current_revision + 1,
            current_cohort=context.get("cohort") or {},
            selection_event=event,
        )
        try:
            updated = study_contexts.upsert_context(
                {
                    "id": context_id,
                    "cohort": target_cohort,
                    "cohort_eligibility_authority": authority,
                },
                expected_revision=current_revision,
                require_revision=True,
                lifecycle_write=False,
                _server_cohort_eligibility_authority_write=True,
            )
        except study_contexts.StudyContextError as exc:
            raise PiCopilotError(
                str(exc.detail.get("error") or "cohort_eligibility_update_blocked"),
                "The StudyContext owner rejected the cohort confirmation.",
                status_code=409,
                details=exc.detail,
            ) from exc
        record.binding = self._binding_for_context(
            updated,
            run_id=record.binding.run_id,
        )
        self._save_record(record)
        return {
            "ok": True,
            "status": "confirmed",
            "code": "cohort_eligibility_confirmed",
            "study_context_id": context_id,
            "study_context_revision": int(updated.get("revision") or 0),
            "receipt_id": authority["receipt_id"],
            "selection": self._cohort_eligibility_selection_projection(record),
        }

    def confirm_plan_decision(
        self,
        session_id: str,
        *,
        project_id: str,
        decision_code: str,
        option_id: str,
        expected_revision: int,
        run_id: str,
    ) -> Dict[str, Any]:
        """Persist one host-rendered scientific choice without a provider turn."""

        record = self._scoped_record(session_id, project_id=project_id)
        stale = self._stale_details(record)
        if stale.get("stale"):
            raise PiCopilotError(
                "pi_session_authority_stale",
                "The StudyContext or reviewed plan changed before this option was selected.",
                status_code=409,
                details=stale,
            )
        context_id = str(record.binding.study_context_id or "").strip()
        context = study_contexts.get_context(context_id) if context_id else None
        if context is None:
            raise PiCopilotError(
                "plan_decision_study_context_required",
                "The plan choice requires a bound StudyContext.",
                status_code=409,
            )
        if context.get("active_job_id"):
            raise PiCopilotError(
                "study_context_active_job_conflict",
                "The plan choice cannot change while an EasyICU job is active.",
                status_code=409,
            )
        current_revision = int(context.get("revision") or 0)
        if int(expected_revision) != current_revision:
            raise PiCopilotError(
                "plan_decision_revision_conflict",
                "The displayed plan choice is stale; reload the current review.",
                status_code=409,
                details={
                    "expected_revision": int(expected_revision),
                    "current_revision": current_revision,
                },
            )
        bound_run_id = str(record.binding.run_id or "").strip()
        if not bound_run_id or not hmac.compare_digest(
            bound_run_id,
            str(run_id or "").strip(),
        ):
            raise PiCopilotError(
                "plan_decision_run_mismatch",
                "The displayed plan choice does not belong to the bound review run.",
                status_code=409,
            )
        rows = list_bound_run_history(
            study_context_id=context_id,
            project_root=research_pipeline_project_root(context_id),
            limit=50,
        )
        row = next(
            (item for item in rows if str(item.get("run_id") or "") == bound_run_id),
            None,
        )
        if row is None:
            raise PiCopilotError(
                "plan_decision_run_not_found",
                "The reviewed plan is no longer available for this project.",
                status_code=409,
            )
        review = agent_runs.read_run_review(str(row.get("project_dir") or ""))
        payloads = review.get("artifact_payloads")
        payloads = payloads if isinstance(payloads, Mapping) else {}
        scientific_review = payloads.get("scientific_plan_review.json")
        findings = (
            scientific_review.get("findings")
            if isinstance(scientific_review, Mapping)
            else None
        )
        authorized_codes = {
            str(item.get("code") or "")
            for item in findings or []
            if isinstance(item, Mapping)
            and item.get("requires_user_authorization") is True
        }
        clean_code = str(decision_code or "").strip()
        if clean_code not in authorized_codes or plan_decisions.decision_is_resolved(context, clean_code):
            raise PiCopilotError(
                "plan_decision_not_required_by_review",
                "The selected decision is not an unresolved human choice in this review.",
                status_code=409,
                details={"decision_code": clean_code},
            )
        agent_plan = payloads.get("agent_plan.json")
        if not isinstance(agent_plan, Mapping):
            raise PiCopilotError(
                "plan_decision_plan_missing",
                "The reviewed plan payload is unavailable.",
                status_code=409,
            )
        try:
            plan_review_progress.validate_choice_source(context, row)
            compiled = plan_decisions.compile_plan_decision(
                decision_code=clean_code,
                option_id=option_id,
                study=context,
                agent_plan=agent_plan,
            )
            updated = study_contexts.upsert_context(
                {"id": context_id, **compiled.patch},
                expected_revision=current_revision,
                require_revision=True,
                lifecycle_write=False,
            )
            plan_review_progress.record_choice(
                before=context, after=updated, run=row,
                decision_code=clean_code, option_id=option_id,
            )
        except plan_decisions.PlanDecisionError as exc:
            raise PiCopilotError(
                exc.code,
                str(exc),
                status_code=409,
                details=exc.details,
            ) from exc
        except study_contexts.StudyContextError as exc:
            raise PiCopilotError(
                str(exc.detail.get("error") or "plan_decision_update_blocked"),
                "The StudyContext owner rejected the plan choice.",
                status_code=409,
                details=exc.detail,
            ) from exc
        record.binding = self._binding_for_context(updated, run_id=bound_run_id)
        self._save_record(record)
        remaining_decisions = [
            code
            for code in authorized_codes
            if not plan_decisions.decision_is_resolved(updated, code)
        ]
        next_action = (
            "continue_review" if remaining_decisions else compiled.next_action
        )
        return {
            "ok": True,
            "status": "confirmed",
            "code": "plan_decision_confirmed",
            "decision_code": clean_code,
            "option_id": str(option_id or "").strip(),
            "study_context_id": context_id,
            "study_context_revision": int(updated.get("revision") or 0),
            "display_label": {
                "en": compiled.display_label_en,
                "zh": compiled.display_label_zh,
            },
            "next_action": next_action,
            "remaining_decision_codes": sorted(remaining_decisions),
        }

    def apply_agent_plan_configuration(
        self,
        session_id: str,
        *,
        project_id: str,
        expected_revision: int,
        run_id: str,
    ) -> Dict[str, Any]:
        """Persist executable coordinates already selected by the Agent plan."""

        record = self._scoped_record(session_id, project_id=project_id)
        stale = self._stale_details(record)
        if stale.get("stale"):
            raise PiCopilotError(
                "pi_session_authority_stale",
                "The StudyContext or reviewed plan changed before Agent configuration compilation.",
                status_code=409,
                details=stale,
            )
        context_id = str(record.binding.study_context_id or "").strip()
        context = study_contexts.get_context(context_id) if context_id else None
        if context is None:
            raise PiCopilotError(
                "agent_plan_configuration_study_context_required",
                "Agent configuration compilation requires a bound StudyContext.",
                status_code=409,
            )
        if context.get("active_job_id"):
            raise PiCopilotError(
                "study_context_active_job_conflict",
                "Agent configuration cannot change while an EasyICU job is active.",
                status_code=409,
            )
        current_revision = int(context.get("revision") or 0)
        if int(expected_revision) != current_revision:
            raise PiCopilotError(
                "agent_plan_configuration_revision_conflict",
                "The displayed Agent plan is stale; reload the current review.",
                status_code=409,
                details={
                    "expected_revision": int(expected_revision),
                    "current_revision": current_revision,
                },
            )
        bound_run_id = str(record.binding.run_id or "").strip()
        if not bound_run_id or not hmac.compare_digest(
            bound_run_id, str(run_id or "").strip()
        ):
            raise PiCopilotError(
                "agent_plan_configuration_run_mismatch",
                "The displayed Agent plan does not belong to the bound review run.",
                status_code=409,
            )
        rows = list_bound_run_history(
            study_context_id=context_id,
            project_root=research_pipeline_project_root(context_id),
            limit=50,
        )
        row = next(
            (item for item in rows if str(item.get("run_id") or "") == bound_run_id),
            None,
        )
        if row is None:
            raise PiCopilotError(
                "agent_plan_configuration_run_not_found",
                "The reviewed Agent plan is no longer available for this project.",
                status_code=409,
            )
        review = agent_runs.read_run_review(str(row.get("project_dir") or ""))
        payloads = review.get("artifact_payloads")
        payloads = payloads if isinstance(payloads, Mapping) else {}
        scientific_review = payloads.get("scientific_plan_review.json")
        facts = (
            scientific_review.get("facts")
            if isinstance(scientific_review, Mapping)
            else None
        )
        buckets = facts.get("remediation_buckets") if isinstance(facts, Mapping) else None
        runtime_codes = (
            buckets.get("runtime_capability")
            if isinstance(buckets, Mapping)
            else None
        )
        runtime_codes = runtime_codes if isinstance(runtime_codes, list) else []
        agent_plan = payloads.get("agent_plan.json")
        if not isinstance(agent_plan, Mapping):
            raise PiCopilotError(
                "agent_plan_configuration_plan_missing",
                "The reviewed Agent plan payload is unavailable.",
                status_code=409,
            )
        source = context.get("data_source")
        source = source if isinstance(source, Mapping) else {}
        try:
            grouping = source_identity_authority.resolve_patient_grouping_authority(
                export_path=str(source.get("path") or ""),
                database=str(source.get("database") or ""),
            )
        except source_identity_authority.PatientGroupingAuthorityError:
            grouping = None
        try:
            plan_review_progress.validate_choice_source(context, row)
            compiled = plan_decisions.compile_agent_plan_configuration(
                study=context,
                agent_plan=agent_plan,
                runtime_finding_codes=runtime_codes,
                patient_cluster_available=grouping is not None,
            )
            updated = study_contexts.upsert_context(
                {"id": context_id, **compiled.patch},
                expected_revision=current_revision,
                require_revision=True,
                lifecycle_write=False,
            )
            plan_review_progress.record_choice(
                before=context,
                after=updated,
                run=row,
                decision_code="agent_plan_configuration",
                option_id="typed_runtime_projection",
            )
        except plan_decisions.PlanDecisionError as exc:
            raise PiCopilotError(
                exc.code,
                str(exc),
                status_code=409,
                details=exc.details,
            ) from exc
        except study_contexts.StudyContextError as exc:
            raise PiCopilotError(
                str(
                    exc.detail.get("error")
                    or "agent_plan_configuration_update_blocked"
                ),
                "The StudyContext owner rejected the Agent plan projection.",
                status_code=409,
                details=exc.detail,
            ) from exc
        record.binding = self._binding_for_context(updated, run_id=bound_run_id)
        self._save_record(record)
        return {
            "ok": True,
            "status": "compiled",
            "code": "agent_plan_configuration_compiled",
            "study_context_id": context_id,
            "study_context_revision": int(updated.get("revision") or 0),
            "source_run_id": bound_run_id,
            "runtime_finding_codes": list(compiled.runtime_finding_codes),
            "next_action": "fresh_plan",
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
        projection = build_project_workflow_projection(
            study_context_id=study_context_id,
        )
        return {
            "ok": True,
            "project_id": clean,
            **projection.model_dump(mode="json"),
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

    def prepare_data_package_review(self, *, project_id: str) -> Dict[str, Any]:
        """Seal a current aggregate-only package review for browser preview."""

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
        from easyicu.webserver.data_package_review import (
            DataPackageReviewError,
            build_plan_bound_data_package_review,
            build_registered_data_package_review,
        )

        try:
            payload = None
            run_id = self._latest_run_id(study_context_id, project_id=clean)
            if run_id:
                try:
                    row = self._research_run_row(clean, run_id)
                    wrapper = Path(str(row.get("project_dir") or "")).resolve()
                    plan_run = (wrapper / "pipeline" / run_id).resolve()
                    if plan_run.parent == (wrapper / "pipeline").resolve():
                        payload = build_plan_bound_data_package_review(
                            study,
                            cohort_file=plan_run / "cohort.parquet",
                            plan_file=plan_run / "analysis_plan.json",
                        )
                except (PiCopilotError, DataPackageReviewError):
                    payload = None
            if payload is None:
                payload = build_registered_data_package_review(study)
            self.review_snapshot_store.persist(payload)
        except DataPackageReviewError as exc:
            raise PiCopilotError(
                exc.code,
                exc.message,
                status_code=409,
                details=exc.details,
            ) from exc
        return {
            "ok": True,
            "resource": {
                "kind": "data_package_review",
                "study_context_id": str(study_context_id),
                "study_revision": int(study.get("revision") or 0),
                "review_sha256": str(payload.get("review_sha256") or ""),
                "label": "Analysis data preview",
                "media_type": "application/json",
            },
            "governance": {
                "claim_ceiling": "pre_analysis_review",
                "reportable": False,
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

    def prepare_data_workbench_snapshot(self, *, project_id: str) -> Dict[str, Any]:
        """Seal a native Cohort Workbench view for the project's bound source.

        Cohort Review remains the scientific/data owner.  This method only
        resolves plan-named columns against export metadata and persists the
        owner's aggregate payload behind browser-only immutable coordinates.
        """

        clean = self._assert_project_initialized(project_id)
        study_context_id = self.project_store.resolve(clean)
        study = (
            study_contexts.get_context(study_context_id) if study_context_id else None
        )
        if not study:
            raise PiCopilotError(
                "pi_data_workbench_study_not_found",
                "The research project has no bound StudyContext.",
                status_code=404,
            )
        source = study.get("data_source") if isinstance(study.get("data_source"), Mapping) else {}
        source_path = str(source.get("path") or "").strip()
        if not source_path:
            raise PiCopilotError(
                "pi_data_workbench_source_not_bound",
                "The research project has no bound EasyICU data source.",
                status_code=409,
            )

        from easyicu.webserver import cohort_review, dataio

        requested_variables: list[str] = []
        materialized_analysis_path: Path | None = None
        run_id = self._latest_run_id(study_context_id, project_id=clean)
        if run_id:
            try:
                row = self._research_run_row(clean, run_id)
                wrapper = Path(str(row.get("project_dir") or "")).resolve()
                plan_path = (wrapper / "pipeline" / run_id / "analysis_plan.json").resolve()
                if plan_path.parent.parent == (wrapper / "pipeline").resolve():
                    plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    candidates = ((plan.get("design_selection") or {}).get("candidates") or [])
                    selected = next(
                        (
                            candidate
                            for candidate in candidates
                            if isinstance(candidate, Mapping)
                            and candidate.get("disposition") == "selected"
                        ),
                        {},
                    )
                    requested_variables = [
                        str(value or "").strip().lower()
                        for value in (selected.get("required_variables") or [])
                        if str(value or "").strip()
                    ][:12]
                    run_cohort = (plan_path.parent / "cohort.parquet").resolve()
                    immutable_input = (
                        wrapper / "pipeline_input" / "web_research_universe.parquet"
                    ).resolve()
                    if run_cohort.parent == plan_path.parent and run_cohort.is_file():
                        materialized_analysis_path = run_cohort
                    elif (
                        immutable_input.parent == (wrapper / "pipeline_input").resolve()
                        and immutable_input.is_file()
                    ):
                        materialized_analysis_path = immutable_input
            except (OSError, ValueError, TypeError, json.JSONDecodeError, PiCopilotError):
                requested_variables = []
                materialized_analysis_path = None

        description = dataio.describe_export_source(source_path)
        files = [row for row in (description.get("files") or []) if isinstance(row, Mapping)]
        suffixes = ("_maximum", "_minimum", "_median", "_mean", "_max", "_min")
        identifier_columns = {
            "stay_id",
            "subject_id",
            "hadm_id",
            "icustay_id",
            "patientunitstayid",
        }
        feature_ids: list[str] = []
        for requested in requested_variables:
            if requested in identifier_columns:
                continue
            aliases = [requested]
            aliases.extend(
                requested[: -len(suffix)]
                for suffix in suffixes
                if requested.endswith(suffix) and len(requested) > len(suffix)
            )
            match = next(
                (
                    f"{row.get('module')}:{column}"
                    for alias in aliases
                    for row in files
                    for column in (row.get("columns") or [])
                    if str(column).strip().lower() == alias
                    and str(row.get("module") or "").strip()
                ),
                "",
            )
            if match and match not in feature_ids:
                feature_ids.append(match)
            if len(feature_ids) >= 4:
                break

        try:
            # A completed or reviewable run owns the exact cohort that its
            # plan and results consumed.  Prefer that immutable input even
            # while the broader export remains online; resolving plan tokens
            # back to similarly named source columns can visualize a different
            # feature (for example raw sepsis3_sofa1 instead of the analysed
            # sep3_sofa1_max).  The registered export is only the fallback
            # before a run-scoped cohort exists.
            if materialized_analysis_path is not None:
                owner_payload = cohort_review.materialized_analysis_review(
                    materialized_analysis_path,
                    requested_variables,
                )
            elif files:
                body: Dict[str, Any] = {"source_path": source_path}
                if feature_ids:
                    body["selected_features"] = feature_ids
                owner_payload = cohort_review.cohort_review_summary(body)
            else:
                reason = str(description.get("error") or "source_not_available")
                raise cohort_review.CohortReviewError({"error": reason})
            snapshot_payload = {
                key: owner_payload.get(key)
                for key in (
                    "source",
                    "summary",
                    "groups",
                    "feature_catalog",
                    "feature_selection",
                    "selected_feature_distributions",
                    "coverage",
                    "quality",
                    "survival_analysis",
                    "blocked_features",
                    "provenance",
                )
            }
            snapshot = build_data_workbench_snapshot(
                project_id=clean,
                view=(
                    "feature_distribution"
                    if feature_ids or requested_variables
                    else "cohort_summary"
                ),
                title=(
                    "Analysis data and feature distributions"
                    if feature_ids or requested_variables
                    else "Cohort review"
                ),
                payload=snapshot_payload,
                privacy=(
                    owner_payload.get("privacy")
                    if isinstance(owner_payload.get("privacy"), Mapping)
                    else {}
                ),
            )
            self.data_workbench_snapshot_store.persist(snapshot)
        except cohort_review.CohortReviewError as exc:
            reason = str((exc.detail or {}).get("error") or "cohort_review_blocked")
            raise PiCopilotError(
                reason,
                "The Cohort Review owner could not prepare this analysis-data view.",
                status_code=409,
                details={"owner": "easyicu.webserver.cohort_review"},
            ) from exc
        except CopilotDataWorkbenchError as exc:
            raise PiCopilotError(
                exc.code,
                exc.message,
                status_code=409,
                details=exc.details,
            ) from exc
        return {
            "ok": True,
            "resource": {
                "kind": "data_workbench_snapshot",
                "view": str(snapshot.get("view") or ""),
                "snapshot_sha256": str(snapshot.get("snapshot_sha256") or ""),
                "label": "Analysis data preview",
                "media_type": "application/json",
            },
            "governance": {
                "claim_ceiling": "descriptive_review",
                "reportable": False,
                "browser_only": True,
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

    def _research_run_row(self, project_id: str, run_id: str) -> Mapping[str, Any]:
        """Resolve one run through the project binding authority."""

        clean_project = self._assert_project_initialized(project_id)
        study_context_id = self.project_store.resolve(clean_project)
        clean_run = str(run_id or "").strip()
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
        return row

    def get_research_artifact(
        self,
        *,
        project_id: str,
        run_id: str,
        artifact_name: str,
        expected_sha256: str | None = None,
    ) -> Dict[str, Any]:
        """Resolve a run artefact through project authority without exposing paths."""

        clean_run = str(run_id or "").strip()
        clean_artifact = str(artifact_name or "").strip()
        row = self._research_run_row(project_id, clean_run)
        loaded = agent_runs.read_run_artifact(
            str(row.get("project_dir") or ""),
            clean_artifact,
        )
        if not loaded.get("ok"):
            code = str(loaded.get("error") or "pi_research_artifact_unavailable")
            if code == "artifact_privacy_scan_failed":
                raise PiCopilotError(
                    "pi_research_artifact_privacy_blocked",
                    "The artefact preview was withheld by the EasyICU privacy scan.",
                    status_code=409,
                    details={"artifact": clean_artifact},
                )
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
        metadata = loaded.get("artifact") or {}
        clean_expected = str(expected_sha256 or "").strip().lower()
        observed_sha256 = str(metadata.get("sha256") or "").strip().lower()
        if clean_expected and (
            re.fullmatch(r"[a-f0-9]{64}", clean_expected) is None
            or observed_sha256 != clean_expected
        ):
            raise PiCopilotError(
                "pi_research_artifact_digest_mismatch",
                "The EasyICU run artefact changed after its reference was issued.",
                status_code=409,
                details={"artifact": clean_artifact},
            )
        payload = self._browser_artifact_payload(loaded.get("payload") or {})
        # Stamp the research-agent's own compiled plan semantics so the reader
        # never re-derives them from free-text method names. Projection only:
        # the persisted artefact and its plan_sha256 are untouched.
        payload = project_plan_reader_fields(clean_artifact, payload)
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

    def get_research_evidence_preview(
        self,
        *,
        project_id: str,
        run_id: str,
        evidence_id: str,
        expected_sha256: str,
    ) -> Dict[str, Any]:
        """Return one digest-pinned evidence preview without exposing host paths."""

        clean_run = str(run_id or "").strip()
        clean_evidence = str(evidence_id or "").strip()
        clean_sha = str(expected_sha256 or "").strip().lower()
        row = self._research_run_row(project_id, clean_run)
        project_dir = str(row.get("project_dir") or "")
        wrapper = Path(project_dir).expanduser().resolve()
        pipeline_root = (wrapper / "pipeline").resolve()
        declared_pipeline_run = pipeline_root / clean_run
        evidence_run_dir = wrapper
        if (
            not declared_pipeline_run.is_symlink()
            and declared_pipeline_run.is_dir()
            and declared_pipeline_run.resolve().parent == pipeline_root
        ):
            # Wrapper-level artifacts are copied up for result browsing, while
            # the immutable evidence registry remains owned by the concrete
            # pipeline run. Keep governance at wrapper scope, but preview the
            # digest-pinned record from its actual run directory.
            evidence_run_dir = declared_pipeline_run.resolve()
        loaded = agent_runs.read_run_evidence_preview(
            str(evidence_run_dir), clean_evidence, clean_sha
        )
        if not loaded.get("ok"):
            code = str(loaded.get("error") or "evidence_preview_unavailable")
            if code == "evidence_preview_privacy_scan_failed":
                raise PiCopilotError(
                    "pi_research_evidence_privacy_blocked",
                    "The evidence preview was withheld by the EasyICU privacy scan.",
                    status_code=409,
                    details={"evidence_id": clean_evidence},
                )
            raise PiCopilotError(
                code,
                str(loaded.get("message") or "The evidence preview is unavailable."),
                status_code=404 if code == "evidence_preview_not_found" else 409,
                details={"evidence_id": clean_evidence},
            )
        privacy_scan = loaded.get("privacy_scan") or {}
        if not privacy_scan.get("passed"):
            raise PiCopilotError(
                "pi_research_evidence_privacy_blocked",
                "The evidence preview was withheld by the EasyICU privacy scan.",
                status_code=409,
                details={"evidence_id": clean_evidence},
            )
        payload = self._browser_artifact_payload(loaded.get("payload") or {})
        encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        if len(encoded) > MAX_RESEARCH_ARTIFACT_PREVIEW_BYTES:
            raise PiCopilotError(
                "pi_research_evidence_preview_too_large",
                "The evidence projection exceeds the bounded browser contract.",
                status_code=413,
                details={"evidence_id": clean_evidence, "bytes": len(encoded)},
            )
        review = agent_runs.read_run_review(project_dir)
        if not review.get("ok"):
            raise PiCopilotError(
                str(
                    review.get("error") or "pi_research_evidence_governance_unavailable"
                ),
                "The run governance state is unavailable for this evidence.",
                status_code=409,
                details={"evidence_id": clean_evidence},
            )
        governance = agent_runs.project_artifact_governance(review)
        if not governance.get("ok"):
            raise PiCopilotError(
                str(
                    governance.get("error") or "pi_research_evidence_governance_invalid"
                ),
                "The run governance state is invalid for this evidence.",
                status_code=409,
                details={"evidence_id": clean_evidence},
            )
        return {
            "ok": True,
            "run_id": clean_run,
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

        clean_run = str(run_id or "").strip()
        clean_name = str(document_name or "").strip()
        row = self._research_run_row(project_id, clean_run)
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
            # One boundary for every user-facing reply: identifiers that only
            # mean something to the model or a developer never reach the
            # browser, and therefore never reach the next-step buttons the
            # front end parses out of this same text.
            "transcript": project_transcript(state.get("transcript")),
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
            # Browser-only host coordinates. Pi/model projections never
            # receive the event ids that can mint eligibility authority.
            "cohort_eligibility_selection": (
                self._cohort_eligibility_selection_projection(record)
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
        if active_job_id and record.last_turn_status == "running":
            job = jobs.MANAGER.get(active_job_id)
            if job is None or job.status != "running":
                record.active_message_job_id = None
                if job is not None and job.status in {"done", "failed", "cancelled"}:
                    record.last_turn_status = job.status
                else:
                    # The generic JobManager is intentionally process-local. A
                    # persisted running pointer with no matching job therefore
                    # means the Web process restarted; never leave the
                    # presentation showing "running".
                    record.last_turn_status = "interrupted"
                record = self._save_record(record)
                self.replay_store.finish_turn(
                    session_id=record.session_id,
                    project_id=str(record.project_id),
                    job_id=active_job_id,
                    status=str(record.last_turn_status),
                )

        # Host-owned buttons launch separate child jobs and are intentionally
        # not represented by active_message_job_id. Reconcile them as well, or
        # a process restart leaves historical plan/extraction actions looking
        # live forever and blocks the next real action in the browser.
        for child_job_id in self.replay_store.running_host_action_child_job_ids(
            session_id=record.session_id,
            project_id=str(record.project_id),
        ):
            child = jobs.MANAGER.get(child_job_id)
            if child is not None and child.status == "running":
                self._watch_child_job_for_replay(
                    session_id=record.session_id,
                    project_id=str(record.project_id),
                    job_id=child_job_id,
                )
                continue
            terminal_status = (
                child.status
                if child is not None and child.status in {"done", "failed", "cancelled"}
                else "interrupted"
            )
            self.replay_store.finish_host_actions_for_child_job(
                session_id=record.session_id,
                project_id=str(record.project_id),
                child_job_id=child_job_id,
                status=terminal_status,
            )
            if child is not None:
                self.replay_store.archive_child_job(
                    session_id=record.session_id,
                    project_id=str(record.project_id),
                    job=project_job(child.snapshot()),
                )
        return record

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


def shutdown_pi_copilot_service() -> None:
    """Close the singleton without creating one during WebApp shutdown."""

    global _SERVICE
    with _SERVICE_LOCK:
        service = _SERVICE
        _SERVICE = None
    if service is not None:
        service.close()


def reset_pi_copilot_service_for_tests() -> None:
    shutdown_pi_copilot_service()


__all__ = [
    "ALLOWED_TURN_ACTIONS",
    "PiCopilotService",
    "get_pi_copilot_service",
    "shutdown_pi_copilot_service",
    "reset_pi_copilot_service_for_tests",
]
