"""Typed HTTP adapters for the Pi-based Guided Copilot shell."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints

from easyicu.webserver.pi_copilot import get_pi_copilot_service
from easyicu.webserver.pi_copilot.contracts import PiCopilotError

router = APIRouter()

ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=160),
]
MessageText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=12_000),
]
ProviderText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
]
CredentialText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8192),
]
EndpointText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2048),
]
ModelText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]


class PiSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: ShortText
    title: str = "Pi Copilot"
    language: Literal["en", "zh"] = "en"
    thinking_level: Literal["off", "minimal", "low", "medium", "high"] = "off"
    study_context_id: ShortText | None = None
    external_llm_opt_in: StrictBool = False


class PiMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: MessageText
    allowed_actions: list[Literal["configure", "run", "cancel"]] = Field(
        default_factory=list,
        max_length=3,
    )


class PiAbortRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_job_id: ShortText | None = None


class PiProviderConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderText = "easyicu-local"
    api_key: CredentialText
    base_url: EndpointText = "http://127.0.0.1:8317/v1"
    model: ModelText = "gpt5.6 luna"
    api_transport: Literal[
        "anthropic-messages",
        "google-generative-ai",
        "openai-completions",
        "openai-responses",
    ] = "openai-completions"
    enable_ai: StrictBool = False


def _raise_http(error: PiCopilotError) -> None:
    raise HTTPException(status_code=error.status_code, detail=error.detail) from error


@router.get("/api/copilot/pi/status")
def get_pi_copilot_status() -> dict:
    return get_pi_copilot_service().runtime_status()


@router.post("/api/copilot/pi/provider-config")
def post_pi_copilot_provider_config(body: PiProviderConfigRequest) -> dict:
    try:
        return get_pi_copilot_service().configure_provider(
            provider=body.provider,
            api_key=body.api_key,
            base_url=body.base_url,
            model=body.model,
            api_transport=body.api_transport,
            enable_ai=body.enable_ai,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.post("/api/copilot/pi/sessions")
def post_pi_copilot_session(body: PiSessionCreateRequest) -> dict:
    try:
        return get_pi_copilot_service().create_session(
            project_id=body.project_id,
            title=body.title,
            language=body.language,
            thinking_level=body.thinking_level,
            study_context_id=body.study_context_id,
            external_llm_opt_in=body.external_llm_opt_in,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.get("/api/copilot/pi/sessions")
def get_pi_copilot_sessions(
    project_id: Annotated[str, Query(min_length=1, max_length=160)],
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
) -> dict:
    try:
        return get_pi_copilot_service().list_sessions(
            project_id=project_id,
            limit=limit,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.get("/api/copilot/pi/sessions/{session_id}")
def get_pi_copilot_session(
    session_id: ShortText,
    project_id: Annotated[str, Query(min_length=1, max_length=160)],
) -> dict:
    try:
        return get_pi_copilot_service().get_session(
            session_id,
            project_id=project_id,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.post("/api/copilot/pi/sessions/{session_id}/message")
def post_pi_copilot_message(
    session_id: ShortText, body: PiMessageRequest
) -> dict:
    try:
        return get_pi_copilot_service().send_message(
            session_id,
            message=body.message,
            allowed_actions=body.allowed_actions,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.post("/api/copilot/pi/sessions/{session_id}/rebind")
def post_pi_copilot_rebind(session_id: ShortText) -> dict:
    try:
        return get_pi_copilot_service().rebind_session(session_id)
    except PiCopilotError as exc:
        _raise_http(exc)


@router.post("/api/copilot/pi/sessions/{session_id}/abort")
def post_pi_copilot_abort(
    session_id: ShortText, body: PiAbortRequest
) -> dict:
    try:
        return get_pi_copilot_service().abort_session(
            session_id,
            message_job_id=body.message_job_id,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


__all__ = ["router"]
