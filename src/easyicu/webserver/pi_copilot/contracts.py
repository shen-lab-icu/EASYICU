"""Typed, dependency-neutral contracts for the Pi Copilot boundary."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

PROTOCOL_VERSION = "easyicu.pi-copilot/1"
SESSION_SCHEMA_VERSION = "easyicu.pi-copilot-session/1"
MAX_MESSAGE_CHARS = 12_000


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
    language: Literal["en", "zh"] = "en"
    thinking_level: Literal["off", "minimal", "low", "medium", "high"] = "off"
    external_llm_opt_in: bool = False
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
    """Atomically consumable one-use action grants held only by the host."""

    def __init__(self, remaining: Optional[Dict[str, int]] = None) -> None:
        self._remaining = {
            str(action): max(0, int(count))
            for action, count in dict(remaining or {}).items()
        }
        self._provided = frozenset(self._remaining)
        self._lock = threading.Lock()

    @classmethod
    def from_actions(cls, actions: Iterable[str]) -> "HostTurnGrant":
        return cls({str(action): 1 for action in actions})

    def consume(self, action: str) -> Literal["granted", "consumed", "missing"]:
        name = str(action)
        with self._lock:
            if name not in self._provided:
                return "missing"
            if self._remaining.get(name, 0) <= 0:
                return "consumed"
            self._remaining[name] -= 1
            return "granted"

    @property
    def available_actions(self) -> frozenset[str]:
        with self._lock:
            return frozenset(
                action for action, count in self._remaining.items() if count > 0
            )


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
    ) -> None:
        self.session = session
        self.grant = grant or HostTurnGrant.from_actions(allowed_actions)
        self.authority_validator = authority_validator
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
    "AuthorityBinding",
    "HostTurnGrant",
    "MAX_MESSAGE_CHARS",
    "PROTOCOL_VERSION",
    "PiCopilotError",
    "PiSessionRecord",
    "PiToolResult",
    "SESSION_SCHEMA_VERSION",
    "ToolExecutionContext",
    "utc_now",
]
