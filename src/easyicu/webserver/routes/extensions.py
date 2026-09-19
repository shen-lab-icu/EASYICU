"""HTTP adapters for the user Skill and MCP extension registry.

D-P2-7 exact-edit confirmation: installing, overwriting, or removing an
extension can point a later Copilot session or Agent run at an external MCP
endpoint, so install/overwrite/remove calls must prove they read the current
activation
state. Callers send ``expected_sha256`` equal to the ``activation_sha256``
returned by ``GET /api/extensions``; a missing value is rejected with 422
and a stale one with 409 (``extension_revision_mismatch``). Read the current
revision again after any 409 and ask the user to confirm before retrying.

Deployment note: these routes rely on the host loopback-only guard in
``app.local_clients_only``. ``EASYICU_WEB_TRUST_PROXY=1`` must only be set
when an authenticating reverse proxy in front of EasyICU verifies every
request itself — otherwise a remote client arrives as a loopback peer and
this confirmation is the only remaining barrier.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from easyicu.extensions import ExtensionRegistry, ExtensionRegistryError
from easyicu.extensions.mcp_client import list_mcp_tools
from easyicu.webserver import settings as settings_store

router = APIRouter()

#: Shared exact-edit confirmation field: the activation revision the caller
#: saw on GET /api/extensions. Required on install, overwrite, remove, and
#: enable/disable state changes.
ExpectedRevision = Field(
    min_length=64,
    max_length=64,
    pattern=r"^[0-9a-f]{64}$",
)


class SkillInstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_md: str = Field(min_length=1, max_length=12_000)
    stages: list[Literal["conversation", "writing"]] = Field(
        default_factory=lambda: ["conversation"], min_length=1, max_length=2
    )
    enabled: StrictBool = True
    expected_sha256: str = ExpectedRevision


class McpInstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    url: str = Field(min_length=1, max_length=2048)
    allowed_tools: list[str] = Field(min_length=1, max_length=32)
    enabled: StrictBool = False
    expected_sha256: str = ExpectedRevision


class ExtensionStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["skill", "mcp"]
    name: str = Field(min_length=1, max_length=64)
    enabled: StrictBool
    expected_sha256: str = ExpectedRevision


class ExtensionRemoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["skill", "mcp"]
    name: str = Field(min_length=1, max_length=64)
    expected_sha256: str = ExpectedRevision


class McpTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2048)


def _registry() -> ExtensionRegistry:
    return ExtensionRegistry()


def _raise_extension_error(exc: ExtensionRegistryError) -> None:
    if exc.code == "extension_revision_mismatch":
        detail = dict(exc.detail)
        current = exc.details.get("activation_sha256")
        detail["activation_sha256"] = current
        raise HTTPException(status_code=409, detail=detail) from exc
    raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.get("/api/extensions")
def get_extensions() -> Dict[str, Any]:
    """Return path-free installed state plus the current activation digest."""

    try:
        payload = _registry().list_public()
    except ExtensionRegistryError as exc:
        _raise_extension_error(exc)
    settings = settings_store.load_settings()
    payload["mcp_master_enabled"] = bool(settings.get("mcp_tools_enabled", False))
    return payload


@router.get("/api/extensions/skills/{name}")
def get_installed_skill(name: str) -> Dict[str, Any]:
    """Inspect the current local SKILL.md without exposing registry paths."""
    try:
        return _registry().current_skill(name)
    except ExtensionRegistryError as exc:
        _raise_extension_error(exc)


@router.post("/api/extensions/skills/install")
def post_install_skill(body: SkillInstallRequest) -> Dict[str, Any]:
    try:
        installed = _registry().install_skill(
            body.skill_md,
            stages=body.stages,
            enabled=body.enabled,
            expected_activation_sha256=body.expected_sha256,
        )
        return {"ok": True, "skill": installed, "extensions": get_extensions()}
    except ExtensionRegistryError as exc:
        _raise_extension_error(exc)


@router.post("/api/extensions/mcp/install")
def post_install_mcp(body: McpInstallRequest) -> Dict[str, Any]:
    try:
        installed = _registry().install_mcp_server(
            name=body.name,
            url=body.url,
            allowed_tools=body.allowed_tools,
            enabled=body.enabled,
            expected_activation_sha256=body.expected_sha256,
        )
        return {"ok": True, "mcp_server": installed, "extensions": get_extensions()}
    except ExtensionRegistryError as exc:
        _raise_extension_error(exc)


@router.post("/api/extensions/state")
def post_extension_state(body: ExtensionStateRequest) -> Dict[str, Any]:
    try:
        updated = _registry().set_enabled(
            kind=body.kind,
            name=body.name,
            enabled=body.enabled,
            expected_activation_sha256=body.expected_sha256,
        )
        return {"ok": True, "extension": updated, "extensions": get_extensions()}
    except ExtensionRegistryError as exc:
        _raise_extension_error(exc)


@router.post("/api/extensions/remove")
def post_remove_extension(body: ExtensionRemoveRequest) -> Dict[str, Any]:
    try:
        removed = _registry().remove(
            kind=body.kind,
            name=body.name,
            expected_activation_sha256=body.expected_sha256,
        )
        return {**removed, "extensions": get_extensions()}
    except ExtensionRegistryError as exc:
        _raise_extension_error(exc)


@router.post("/api/extensions/mcp/test")
def post_test_mcp(body: McpTestRequest) -> Dict[str, Any]:
    """Perform an explicit, bounded MCP handshake and list available tool names."""

    try:
        return list_mcp_tools(body.url)
    except ExtensionRegistryError as exc:
        _raise_extension_error(exc)


__all__ = ["router"]
