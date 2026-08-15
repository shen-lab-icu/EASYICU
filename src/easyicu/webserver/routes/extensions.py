"""HTTP adapters for the user Skill and MCP extension registry."""

from __future__ import annotations

from typing import Any, Dict, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from easyicu.extensions import ExtensionRegistry, ExtensionRegistryError
from easyicu.extensions.mcp_client import list_mcp_tools
from easyicu.webserver import settings as settings_store

router = APIRouter()


class SkillInstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_md: str = Field(min_length=1, max_length=12_000)
    stages: list[Literal["conversation", "writing"]] = Field(
        default_factory=lambda: ["conversation"], min_length=1, max_length=2
    )
    enabled: StrictBool = True


class McpInstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    url: str = Field(min_length=1, max_length=2048)
    allowed_tools: list[str] = Field(min_length=1, max_length=32)
    enabled: StrictBool = False


class ExtensionStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["skill", "mcp"]
    name: str = Field(min_length=1, max_length=64)
    enabled: StrictBool


class ExtensionRemoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["skill", "mcp"]
    name: str = Field(min_length=1, max_length=64)


class McpTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2048)


def _registry() -> ExtensionRegistry:
    return ExtensionRegistry()


def _raise_extension_error(exc: ExtensionRegistryError) -> None:
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


@router.post("/api/extensions/skills/install")
def post_install_skill(body: SkillInstallRequest) -> Dict[str, Any]:
    try:
        installed = _registry().install_skill(
            body.skill_md,
            stages=body.stages,
            enabled=body.enabled,
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
        )
        return {"ok": True, "extension": updated, "extensions": get_extensions()}
    except ExtensionRegistryError as exc:
        _raise_extension_error(exc)


@router.post("/api/extensions/remove")
def post_remove_extension(body: ExtensionRemoveRequest) -> Dict[str, Any]:
    try:
        removed = _registry().remove(kind=body.kind, name=body.name)
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
