"""Typed, dependency-neutral contracts for the Pi Copilot boundary."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from easyicu.extensions.contracts import (
    EMPTY_EXTENSION_ACTIVATION,
    ExtensionActivationSnapshot,
)

if TYPE_CHECKING:
    from easyicu.extensions import ExtensionRegistry
    from .workspace import ProjectWorkspace

PROTOCOL_VERSION = "easyicu.pi-copilot/1"
SESSION_SCHEMA_VERSION = "easyicu.pi-copilot-session/1"
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

    schema_version: Literal["easyicu.pi-copilot-session/1"] = SESSION_SCHEMA_VERSION
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
    title: str = "Pi Copilot"
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
    binding: AuthorityBinding = Field(default_factory=AuthorityBinding)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    last_message_job_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("last_message_job_id", "last_job_id"),
    )


class PiToolResult(BaseModel):
    """Every model-visible tool result has summary and machine surfaces."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "blocked", "not_found", "error"]
    code: str
    summary: str
    owner: str
    details: Dict[str, Any] = Field(default_factory=dict)
    authority: Dict[str, Any] = Field(default_factory=dict)


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
        allowed_actions: Iterable[str] = (),
        grant: Optional[HostTurnGrant] = None,
        authority_validator: Optional[AuthorityValidator] = None,
        workspace_root: Optional[Path] = None,
        workspace: Optional["ProjectWorkspace"] = None,
        extension_registry: Optional["ExtensionRegistry"] = None,
    ) -> None:
        self.session = session
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
    "MAX_MESSAGE_CHARS",
    "PROTOCOL_VERSION",
    "PiCopilotError",
    "PiProjectBindingHandoffReceipt",
    "PiSessionRecord",
    "PiToolResult",
    "SESSION_SCHEMA_VERSION",
    "ToolExecutionContext",
    "utc_now",
]
