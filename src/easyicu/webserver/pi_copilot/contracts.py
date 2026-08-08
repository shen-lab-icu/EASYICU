"""Typed, dependency-neutral contracts for the Pi Copilot boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

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
    pi_session_id: Optional[str] = None
    pi_session_file: Optional[str] = None
    title: str = "Pi Copilot"
    language: Literal["en", "zh"] = "en"
    thinking_level: Literal["off", "minimal", "low", "medium", "high"] = "medium"
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


class ToolExecutionContext(BaseModel):
    """Host-held turn capability; it is never supplied by the model."""

    model_config = ConfigDict(extra="forbid")

    session: PiSessionRecord
    allowed_actions: frozenset[str] = frozenset()


__all__ = [
    "AuthorityBinding",
    "MAX_MESSAGE_CHARS",
    "PROTOCOL_VERSION",
    "PiCopilotError",
    "PiSessionRecord",
    "PiToolResult",
    "SESSION_SCHEMA_VERSION",
    "ToolExecutionContext",
    "utc_now",
]
