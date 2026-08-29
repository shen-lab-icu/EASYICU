"""Typed HTTP adapters for the Pi-based Guided Copilot shell."""

from __future__ import annotations

from html import escape as html_escape
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints

from easyicu.webserver.pi_copilot import get_pi_copilot_service
from easyicu.webserver.pi_copilot.contracts import (
    PiCopilotError,
    PiProjectBindingHandoffReceipt,
    ResearchProviderBinding,
)
from easyicu.webserver import codex_account_sessions

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
WorkspaceFileText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=240),
]
RunIdText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,159}$",
    ),
]
ArtifactNameText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=6,
        max_length=160,
        pattern=r"^[A-Za-z0-9_.-]+\.json$",
    ),
]
EvidenceIdText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9_.-]{1,160}$",
    ),
]
ResearchDocumentNameText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=20,
        max_length=64,
        pattern=(
            r"^(?:manuscript_scaffold\.(?:pdf|tex|bib)|"
            r"system_validation_report\.(?:html|pdf))$"
        ),
    ),
]
Sha256Text = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=64,
        max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    ),
]


class PiSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: ShortText
    title: str = "EasyICU Copilot"
    agent_mode: Literal["research", "workspace"] = "research"
    language: Literal["en", "zh"] = "en"
    thinking_level: Literal["off", "minimal", "low", "medium", "high"] = "off"
    external_llm_opt_in: StrictBool = False
    research_provider: Literal["api", "codex"] = "api"
    research_model: ModelText | None = None


class CodexLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flow: Literal["browser", "device_code"] = "browser"


class PiMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: ShortText
    message: MessageText
    allowed_actions: list[
        Literal[
            "configure",
            "idea",
            "literature",
            "extract",
            "run",
            "provider_run",
            "cancel",
            "workspace_write",
            "mcp_read",
        ]
    ] = Field(
        default_factory=list,
        max_length=9,
    )
    turn_intent: (
        Literal[
            "confirm_formal_plan_generation",
            "confirm_planner_checkpoint_resume",
            "advance_after_data_source_confirmation",
        ]
        | None
    ) = None


class PiRegenerateRequest(PiMessageRequest):
    user_entry_id: ShortText
    regeneration_intent: Literal["user_edited_message"] | None = None


class PiDataSourceAuthorizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: ShortText
    action: Literal[
        "reuse_project_source",
        "use_study_required_data",
        "begin_local_selection",
        "begin_full_data_selection",
        "confirm_selected_source",
    ]
    database: Literal["miiv", "mimic", "eicu", "aumc", "hirid", "sic"] | None = None


class PiAbortRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: ShortText
    message_job_id: ShortText | None = None


class PiProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: ShortText


class PiPresentationPinRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: ShortText
    pinned: StrictBool = True


class PiProjectInitializeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: ShortText
    title: str = "EasyICU Copilot"
    confirm_initialization: StrictBool = False
    binding_receipt: PiProjectBindingHandoffReceipt | None = None


class PiProviderConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderText = "easyicu-local"
    api_key: CredentialText
    base_url: EndpointText = "http://127.0.0.1:8317/v1"
    model: ModelText = "gpt-5.6-luna"
    api_transport: Literal[
        "anthropic-messages",
        "google-generative-ai",
        "openai-completions",
        "openai-responses",
    ] = "openai-completions"
    enable_ai: StrictBool = False


def _raise_http(error: PiCopilotError) -> None:
    raise HTTPException(status_code=error.status_code, detail=error.detail) from error


def _raise_codex_http(error: codex_account_sessions.CodexAccountSessionError) -> None:
    raise HTTPException(
        status_code=400,
        detail={
            "error": error.code,
            "owner": "easyicu.webserver.codex_account_sessions",
        },
    ) from error


def _workspace_preview_document(*, file_name: str, artifact_html: str) -> str:
    """Keep Host provenance outside the sandboxed model-authored document."""

    safe_file = html_escape(file_name, quote=True)
    safe_srcdoc = html_escape(artifact_html, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>EasyICU workspace preview · {safe_file}</title>
  <style>
    *{{box-sizing:border-box}}
    html,body{{width:100%;height:100%;margin:0}}
    body{{display:grid;grid-template-rows:auto minmax(0,1fr);background:#fff;color:#202124;font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
    header{{display:flex;flex-wrap:wrap;gap:4px 10px;padding:10px 14px;border-bottom:1px solid #ead9aa;background:#fff9e9}}
    header strong{{font-weight:700}}
    header span{{color:#5f6368}}
    iframe{{width:100%;height:100%;border:0;background:#fff}}
  </style>
</head>
<body data-easyicu-workspace-preview="unvalidated">
  <header role="note" data-easyicu-workspace-provenance>
    <strong>Workspace artifact · Unvalidated / 工作区产物 · 未验证</strong>
    <span>Not scientific evidence; unsupported for clinical or manuscript claims. / 不是科学证据；不支持临床或论文结论。</span>
  </header>
  <iframe id="easyicu-workspace-preview-content" sandbox="allow-scripts" referrerpolicy="no-referrer" title="Unvalidated workspace artifact: {safe_file}" srcdoc="{safe_srcdoc}"></iframe>
</body>
</html>"""


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
def post_pi_copilot_session(body: PiSessionCreateRequest, request: Request) -> dict:
    try:
        service = get_pi_copilot_service()
        if body.research_provider == "codex":
            try:
                selected_model = codex_account_sessions.validated_model_for_request(
                    request,
                    str(body.research_model or ""),
                )
                research_provider = ResearchProviderBinding(
                    provider="codex",
                    credential_source="codex_user_auth",
                    authentication_mode="chatgpt_account",
                    model=selected_model,
                    account_session_sha256=(
                        codex_account_sessions.binding_for_request(request)
                    ),
                )
            except codex_account_sessions.CodexAccountSessionError as exc:
                _raise_codex_http(exc)
        else:
            research_provider = service.verified_api_research_provider_binding()
        return service.create_session(
            project_id=body.project_id,
            title=body.title,
            agent_mode=body.agent_mode,
            language=body.language,
            thinking_level=body.thinking_level,
            external_llm_opt_in=body.external_llm_opt_in,
            research_provider=research_provider,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.get("/api/copilot/pi/research-provider/codex/status")
def get_pi_copilot_codex_status(request: Request) -> dict:
    return {"ok": True, "auth": codex_account_sessions.status(request)}


@router.post("/api/copilot/pi/research-provider/codex/login")
def post_pi_copilot_codex_login(
    body: CodexLoginRequest,
    request: Request,
    response: Response,
) -> dict:
    try:
        return codex_account_sessions.start_login(
            request,
            response,
            flow=body.flow,
        )
    except codex_account_sessions.CodexAccountSessionError as exc:
        _raise_codex_http(exc)


@router.post("/api/copilot/pi/research-provider/codex/cancel")
def post_pi_copilot_codex_cancel(request: Request) -> dict:
    try:
        return codex_account_sessions.cancel_login(request)
    except codex_account_sessions.CodexAccountSessionError as exc:
        _raise_codex_http(exc)


@router.post("/api/copilot/pi/research-provider/codex/logout")
def post_pi_copilot_codex_logout(request: Request) -> dict:
    try:
        return codex_account_sessions.logout(request)
    except codex_account_sessions.CodexAccountSessionError as exc:
        _raise_codex_http(exc)


@router.get("/api/copilot/pi/research-provider/codex/models")
def get_pi_copilot_codex_models(request: Request) -> dict:
    try:
        return codex_account_sessions.models(request)
    except codex_account_sessions.CodexAccountSessionError as exc:
        _raise_codex_http(exc)


@router.post("/api/copilot/pi/projects/initialize")
def post_pi_copilot_project_initialize(
    body: PiProjectInitializeRequest,
) -> dict:
    try:
        return get_pi_copilot_service().initialize_project(
            project_id=body.project_id,
            title=body.title,
            confirm_initialization=body.confirm_initialization,
            binding_receipt=body.binding_receipt,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.get("/api/copilot/pi/projects/{project_id}/workflow")
def get_pi_copilot_project_workflow(project_id: ShortText) -> dict:
    try:
        return get_pi_copilot_service().get_project_workflow(
            project_id=project_id,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.get("/api/copilot/pi/projects/{project_id}/workspace/file")
def get_pi_copilot_workspace_file(
    project_id: ShortText,
    file: WorkspaceFileText,
) -> dict:
    try:
        return get_pi_copilot_service().get_workspace_file(
            project_id=project_id,
            relative_file=file,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.get(
    "/api/copilot/pi/projects/{project_id}/workspace/preview",
    response_class=HTMLResponse,
)
def get_pi_copilot_workspace_preview(
    project_id: ShortText,
    file: WorkspaceFileText,
    checked_sha256: Sha256Text,
) -> HTMLResponse:
    try:
        payload = get_pi_copilot_service().get_workspace_preview(
            project_id=project_id,
            relative_file=file,
            checked_sha256=checked_sha256,
        )
    except PiCopilotError as exc:
        _raise_http(exc)
    artifact = payload["artifact"]
    return HTMLResponse(
        content=_workspace_preview_document(
            file_name=str(artifact["file"]),
            artifact_html=str(artifact["text"]),
        ),
        headers={
            "Content-Security-Policy": (
                "sandbox allow-scripts; default-src 'none'; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; img-src data:; "
                "frame-src 'self'; "
                "connect-src 'none'; form-action 'none'; base-uri 'none'; "
                "frame-ancestors 'self'"
            ),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )


@router.get(
    "/api/copilot/pi/projects/{project_id}/runs/{run_id}/artifacts/{artifact_name}"
)
def get_pi_copilot_research_artifact(
    project_id: ShortText,
    run_id: RunIdText,
    artifact_name: ArtifactNameText,
) -> dict:
    try:
        return get_pi_copilot_service().get_research_artifact(
            project_id=project_id,
            run_id=run_id,
            artifact_name=artifact_name,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.get(
    "/api/copilot/pi/projects/{project_id}/runs/{run_id}/evidence/{evidence_id}"
)
def get_pi_copilot_research_evidence_preview(
    project_id: ShortText,
    run_id: RunIdText,
    evidence_id: EvidenceIdText,
    expected_sha256: Sha256Text,
) -> dict:
    try:
        return get_pi_copilot_service().get_research_evidence_preview(
            project_id=project_id,
            run_id=run_id,
            evidence_id=evidence_id,
            expected_sha256=expected_sha256,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.get("/api/copilot/pi/projects/{project_id}/data-package-review")
def get_pi_copilot_data_package_review(
    project_id: ShortText,
    study_revision: Annotated[int, Query(ge=0)],
    review_sha256: Sha256Text,
) -> dict:
    try:
        return get_pi_copilot_service().get_data_package_review(
            project_id=project_id,
            study_revision=study_revision,
            review_sha256=review_sha256,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.get("/api/copilot/pi/projects/{project_id}/data-workbench-snapshot")
def get_pi_copilot_data_workbench_snapshot(
    project_id: ShortText,
    snapshot_sha256: Sha256Text,
) -> dict:
    try:
        return get_pi_copilot_service().get_data_workbench_snapshot(
            project_id=project_id,
            snapshot_sha256=snapshot_sha256,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.get(
    "/api/copilot/pi/projects/{project_id}/runs/{run_id}/documents/{document_name}"
)
def get_pi_copilot_research_document(
    project_id: ShortText,
    run_id: RunIdText,
    document_name: ResearchDocumentNameText,
) -> Response:
    try:
        payload = get_pi_copilot_service().get_research_document(
            project_id=project_id,
            run_id=run_id,
            document_name=document_name,
        )
    except PiCopilotError as exc:
        _raise_http(exc)
    return Response(
        content=payload["content"],
        media_type=str(payload["media_type"]),
        headers={
            "Content-Disposition": f'inline; filename="{document_name}"',
            "Content-Security-Policy": (
                "sandbox; default-src 'none'; style-src 'unsafe-inline'; "
                "img-src data:; frame-ancestors 'self'"
            ),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "SAMEORIGIN",
            "Referrer-Policy": "no-referrer",
            "X-EasyICU-Claim-Ceiling": str(payload["claim_ceiling"]),
        },
    )


@router.get("/api/copilot/pi/sessions")
def get_pi_copilot_sessions(
    project_id: Annotated[str, Query(min_length=1, max_length=160)],
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
    agent_mode: Annotated[
        str | None, Query(pattern="^(research|workspace)$")
    ] = None,
) -> dict:
    try:
        return get_pi_copilot_service().list_sessions(
            project_id=project_id,
            limit=limit,
            agent_mode=agent_mode,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.get("/api/copilot/pi/sessions/{session_id}")
def get_pi_copilot_session(
    session_id: ShortText,
    project_id: Annotated[str, Query(min_length=1, max_length=160)],
    transcript_cursor: Annotated[str | None, Query(max_length=32)] = None,
    transcript_limit: Annotated[int, Query(ge=1, le=200)] = 100,
    replay_cursor: Annotated[str | None, Query(max_length=32)] = None,
    replay_limit: Annotated[int, Query(ge=1, le=100)] = 48,
) -> dict:
    try:
        return get_pi_copilot_service().get_session(
            session_id,
            project_id=project_id,
            transcript_cursor=transcript_cursor,
            transcript_limit=transcript_limit,
            replay_cursor=replay_cursor,
            replay_limit=replay_limit,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.post("/api/copilot/pi/sessions/{session_id}/message")
def post_pi_copilot_message(session_id: ShortText, body: PiMessageRequest) -> dict:
    try:
        return get_pi_copilot_service().send_message(
            session_id,
            project_id=body.project_id,
            message=body.message,
            allowed_actions=body.allowed_actions,
            message_intent=body.turn_intent,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.post("/api/copilot/pi/sessions/{session_id}/regenerate")
def post_pi_copilot_regenerate(
    session_id: ShortText,
    body: PiRegenerateRequest,
) -> dict:
    try:
        return get_pi_copilot_service().send_message(
            session_id,
            project_id=body.project_id,
            message=body.message,
            allowed_actions=body.allowed_actions,
            regenerate_user_entry_id=body.user_entry_id,
            regeneration_intent=body.regeneration_intent,
            message_intent=body.turn_intent,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.post("/api/copilot/pi/sessions/{session_id}/data-source-authorization")
def post_pi_copilot_data_source_authorization(
    session_id: ShortText,
    body: PiDataSourceAuthorizationRequest,
) -> dict:
    try:
        return get_pi_copilot_service().authorize_data_source(
            session_id,
            project_id=body.project_id,
            action=body.action,
            database=body.database,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.post("/api/copilot/pi/sessions/{session_id}/rebind")
def post_pi_copilot_rebind(session_id: ShortText, body: PiProjectRequest) -> dict:
    try:
        return get_pi_copilot_service().rebind_session(
            session_id,
            project_id=body.project_id,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.post("/api/copilot/pi/sessions/{session_id}/presentation")
def post_pi_copilot_presentation_pin(
    session_id: ShortText,
    body: PiPresentationPinRequest,
) -> dict:
    try:
        return get_pi_copilot_service().set_presentation_pin(
            session_id,
            project_id=body.project_id,
            pinned=body.pinned,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.post("/api/copilot/pi/sessions/{session_id}/child-jobs/{job_id}/archive")
def post_pi_copilot_child_job_archive(
    session_id: ShortText,
    job_id: ShortText,
    body: PiProjectRequest,
) -> dict:
    try:
        return get_pi_copilot_service().archive_child_job(
            session_id,
            project_id=body.project_id,
            job_id=job_id,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


@router.post("/api/copilot/pi/sessions/{session_id}/abort")
def post_pi_copilot_abort(session_id: ShortText, body: PiAbortRequest) -> dict:
    try:
        return get_pi_copilot_service().abort_session(
            session_id,
            project_id=body.project_id,
            message_job_id=body.message_job_id,
        )
    except PiCopilotError as exc:
        _raise_http(exc)


__all__ = ["router"]
