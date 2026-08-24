"""Typed, dependency-neutral contracts for the Pi Copilot boundary."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from easyicu.extensions.contracts import (
    EMPTY_EXTENSION_ACTIVATION,
    ExtensionActivationSnapshot,
)

if TYPE_CHECKING:
    from easyicu.extensions import ExtensionRegistry
    from .workspace import ProjectWorkspace

PROTOCOL_VERSION = "easyicu.pi-copilot/1"
LEGACY_SESSION_SCHEMA_VERSION = "easyicu.pi-copilot-session/1"
SESSION_SCHEMA_VERSION = "easyicu.pi-copilot-session/2"
MAX_MESSAGE_CHARS = 12_000
AgentMode = Literal["research", "workspace"]
TURN_CAPABILITIES = frozenset({"workspace_write"})


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class PiCopilotError(RuntimeError):
    """Stable owner-attributable failure returned by the Copilot boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status_code = int(status_code)
        self.details = dict(details or {})

    @property
    def detail(self) -> Dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "reason": self.message,
            "owner": "easyicu.webserver.pi_copilot",
            **({"details": self.details} if self.details else {}),
        }


class AuthorityBinding(BaseModel):
    """Immutable scientific identity attached to one UX session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_context_id: Optional[str] = None
    study_revision: Optional[int] = Field(default=None, ge=0)
    run_id: Optional[str] = None
    active_job_id: Optional[str] = None


class ResearchProviderBinding(BaseModel):
    """Immutable model connection selected before a Copilot session starts.

    Session schema v2 uses this one binding for both the conversational shell
    and governed Research Agent calls.  The historical class/field name remains
    readable so v1 sessions keep their original two-provider semantics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.research-provider-binding/1"] = (
        "easyicu.research-provider-binding/1"
    )
    provider: Literal["openai", "codex"] = "openai"
    credential_source: Literal["pi_verified", "codex_user_auth"] = "pi_verified"
    authentication_mode: Literal["api_key", "chatgpt_account"] = "api_key"
    model: str = Field(default="configured_provider_model", min_length=1, max_length=256)
    account_session_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )

    @model_validator(mode="after")
    def _authority_is_coherent(self) -> "ResearchProviderBinding":
        if self.provider == "codex":
            if (
                self.credential_source != "codex_user_auth"
                or self.authentication_mode != "chatgpt_account"
                or not self.account_session_sha256
            ):
                raise ValueError("codex research provider binding is incomplete")
        elif (
            self.credential_source != "pi_verified"
            or self.authentication_mode != "api_key"
            or self.account_session_sha256 is not None
        ):
            raise ValueError("API research provider binding is inconsistent")
        return self

    def public_projection(self) -> Dict[str, Any]:
        """Return browser-safe identity without the server-side account locator."""

        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "credential_source": self.credential_source,
            "authentication_mode": self.authentication_mode,
            "model": self.model,
        }


class PiProjectBindingHandoffReceipt(BaseModel):
    """Path-free Agent/StudyContext handoff into one Pi research project."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.pi-project-binding-handoff/1"] = (
        "easyicu.pi-project-binding-handoff/1"
    )
    project_id: str = Field(min_length=1, max_length=160)
    project_title: str = Field(min_length=1, max_length=160)
    study_context_id: str = Field(min_length=1, max_length=160)
    study_context_revision: int = Field(ge=0)


class PiSessionRecord(BaseModel):
    """Bounded EasyICU metadata; Pi's JSONL remains separate UX state."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "easyicu.pi-copilot-session/1",
        "easyicu.pi-copilot-session/2",
    ] = SESSION_SCHEMA_VERSION
    session_id: str
    # Product/project ownership is fixed when the Pi AgentSession is created.
    # ``None`` is accepted only so metadata written before project scoping can
    # still be read and retired safely; new sessions must always set it.
    project_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=160,
        frozen=True,
    )
    pi_session_id: Optional[str] = None
    pi_session_file: Optional[str] = None
    title: str = "EasyICU Copilot"
    agent_mode: AgentMode = "research"
    language: Literal["en", "zh"] = "en"
    thinking_level: Literal["off", "minimal", "low", "medium", "high"] = "off"
    external_llm_opt_in: bool = False
    # Installed extensions are frozen when the Pi AgentSession is created.
    # Later registry edits affect only new sessions; old sessions retain this
    # path-free descriptor and load exact Skill objects by content digest.
    extension_activation: ExtensionActivationSnapshot = Field(
        default_factory=lambda: EMPTY_EXTENSION_ACTIVATION
    )
    research_provider: ResearchProviderBinding = Field(
        default_factory=ResearchProviderBinding,
        frozen=True,
    )
    binding: AuthorityBinding = Field(default_factory=AuthorityBinding)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    last_message_job_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("last_message_job_id", "last_job_id"),
    )
    # Process reconciliation and retention metadata only. Browser-safe lifecycle
    # replay lives in PiConversationReplayStore, outside this bounded index.
    active_message_job_id: Optional[str] = None
    last_turn_status: Optional[
        Literal["running", "done", "failed", "cancelled", "interrupted"]
    ] = None
    last_turn_allowed_actions: list[str] = Field(default_factory=list, max_length=16)
    pinned_for_presentation: bool = False

    @property
    def uses_unified_model_connection(self) -> bool:
        """Whether the frozen provider powers both Copilot and analysis."""

        return self.schema_version == SESSION_SCHEMA_VERSION


class PiToolResult(BaseModel):
    """Every model-visible tool result has summary and machine surfaces."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "blocked", "not_found", "error"]
    code: str
    summary: str
    owner: str
    details: Dict[str, Any] = Field(default_factory=dict)
    authority: Dict[str, Any] = Field(default_factory=dict)


class WorkspaceMutationReceipt(BaseModel):
    """Host-issued shared write/edit quota receipt for one model turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.workspace-mutation-receipt/1"] = (
        "easyicu.workspace-mutation-receipt/1"
    )
    tool: Literal["write", "edit"]
    ordinal: int = Field(ge=1)
    limit: int = Field(ge=1)


class WorkspaceMutationLimitError(RuntimeError):
    pass


class HostTurnGrant:
    """Host-held one-use actions and explicitly reusable turn capabilities."""

    def __init__(self, remaining: Optional[Dict[str, int]] = None) -> None:
        requested = {
            str(action): max(0, int(count))
            for action, count in dict(remaining or {}).items()
        }
        self._capabilities = frozenset(requested).intersection(TURN_CAPABILITIES)
        self._remaining = {
            action: count
            for action, count in requested.items()
            if action not in self._capabilities
        }
        self._provided = frozenset(requested)
        self._workspace_mutations = 0
        self._workspace_mutation_limit = 8
        self._lock = threading.Lock()

    @classmethod
    def from_actions(cls, actions: Iterable[str]) -> "HostTurnGrant":
        return cls({str(action): 1 for action in actions})

    def consume_once(
        self, action: str
    ) -> Literal["granted", "consumed", "missing", "capability"]:
        name = str(action)
        with self._lock:
            if name in self._capabilities:
                return "capability"
            if name not in self._provided:
                return "missing"
            if self._remaining.get(name, 0) <= 0:
                return "consumed"
            self._remaining[name] -= 1
            return "granted"

    def has_capability(self, action: str) -> bool:
        """Return whether a reusable capability was granted for this turn."""

        with self._lock:
            return str(action) in self._capabilities

    def reserve_workspace_mutation(
        self, tool: Literal["write", "edit"]
    ) -> WorkspaceMutationReceipt:
        """Reserve from one shared write/edit ceiling before filesystem access."""

        with self._lock:
            if "workspace_write" not in self._capabilities:
                raise WorkspaceMutationLimitError("workspace_write_not_granted")
            if self._workspace_mutations >= self._workspace_mutation_limit:
                raise WorkspaceMutationLimitError("workspace_mutation_limit_reached")
            self._workspace_mutations += 1
            return WorkspaceMutationReceipt(
                tool=tool,
                ordinal=self._workspace_mutations,
                limit=self._workspace_mutation_limit,
            )

    def was_provided(self, action: str) -> bool:
        """Return whether the user supplied this action for the turn.

        Unlike ``available_actions``, this remains true after a one-use grant
        has been consumed.  It is used only by the host to carry the user's
        explicit public-literature-search authorization into a subsequently
        submitted full pipeline; it is never exposed to the model.
        """

        with self._lock:
            return str(action) in self._provided

    @property
    def available_actions(self) -> frozenset[str]:
        with self._lock:
            one_use = frozenset(
                action for action, count in self._remaining.items() if count > 0
            )
            return one_use | self._capabilities


AuthorityValidator = Callable[[AuthorityBinding], Dict[str, Any]]


class ToolExecutionContext:
    """Host-held turn grants and authority freshness; never model supplied."""

    def __init__(
        self,
        *,
        session: PiSessionRecord,
        user_message: str = "",
        allowed_actions: Iterable[str] = (),
        grant: Optional[HostTurnGrant] = None,
        authority_validator: Optional[AuthorityValidator] = None,
        workspace_root: Optional[Path] = None,
        workspace: Optional["ProjectWorkspace"] = None,
        extension_registry: Optional["ExtensionRegistry"] = None,
    ) -> None:
        self.session = session
        # Host-captured text from the current user turn.  Tools may use this
        # bounded, PHI-screened value as user-intent authority; the model
        # cannot supply or rewrite it through tool arguments.
        self.user_message = str(user_message or "")
        self.grant = grant or HostTurnGrant.from_actions(allowed_actions)
        self.authority_validator = authority_validator
        self.workspace_root = (
            Path(workspace_root).expanduser().absolute() if workspace_root else None
        )
        self.workspace = workspace
        self.extension_registry = extension_registry
        self._authority_invalidated_reason: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def allowed_actions(self) -> frozenset[str]:
        return self.grant.available_actions

    def invalidate_authority(self, reason: str) -> None:
        with self._lock:
            self._authority_invalidated_reason = str(reason or "authority_mutated")

    def assert_authority_fresh(self) -> None:
        with self._lock:
            invalidated_reason = self._authority_invalidated_reason
        if invalidated_reason:
            raise PiCopilotError(
                "pi_session_authority_stale",
                "This Pi turn changed EasyICU authority and must stop until rebind.",
                status_code=409,
                details={"stale": True, "reason": invalidated_reason},
            )
        if self.authority_validator is None:
            return
        details = self.authority_validator(self.session.binding)
        if details.get("stale"):
            raise PiCopilotError(
                "pi_session_authority_stale",
                "EasyICU authority changed during this Pi turn; rebind before continuing.",
                status_code=409,
                details=details,
            )


__all__ = [
    "AgentMode",
    "AuthorityBinding",
    "HostTurnGrant",
    "LEGACY_SESSION_SCHEMA_VERSION",
    "MAX_MESSAGE_CHARS",
    "PROTOCOL_VERSION",
    "PiCopilotError",
    "PiProjectBindingHandoffReceipt",
    "ResearchProviderBinding",
    "PiSessionRecord",
    "PiToolResult",
    "SESSION_SCHEMA_VERSION",
    "ToolExecutionContext",
    "utc_now",
]
