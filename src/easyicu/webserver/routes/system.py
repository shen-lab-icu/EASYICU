"""System, settings, and capability HTTP adapters."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from easyicu.webserver import capabilities
from easyicu.webserver import settings as settings_store
from easyicu.webserver.catalog import build_catalog

router = APIRouter()


@router.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Avoid noisy browser 404s while the native UI has no branded icon asset."""
    return Response(status_code=204)


@router.get("/api/catalog")
def catalog() -> dict:
    """The concept catalog the Data Dictionary screen renders."""
    return build_catalog()


@router.get("/api/settings")
def get_settings() -> dict:
    """Local settings + read-only environment facts for the Settings screen."""
    return {**settings_store.load_settings(), "about": settings_store.about()}


@router.post("/api/settings")
def post_settings(patch: Dict[str, Any]) -> dict:
    """Merge-update known settings keys and persist locally.

    A patch naming an unknown, retired or invalid key is a 400. Answering 200
    and dropping the key left the client unable to tell a stored value from a
    discarded one.
    """
    try:
        updated = settings_store.update_settings(patch)
    except settings_store.SettingsValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc
    return {**updated, "about": settings_store.about()}


@router.post("/api/settings/reset")
def post_settings_reset() -> dict:
    """Reset local settings to backend defaults."""
    return {**settings_store.reset_settings(), "about": settings_store.about()}


@router.get("/api/capabilities")
def get_capabilities() -> dict:
    """Return backend capability state consumed by Settings and Agent Science."""
    return capabilities.capability_status()


@router.post("/api/capabilities/tool-check")
def post_capability_tool_check(body: Dict[str, Any]) -> dict:
    """Check whether an MCP-style tool is allowed under current Settings."""
    return capabilities.check_tool_allowed(str(body.get("tool_id") or ""))


@router.post("/api/capabilities/zotero/search")
def post_capability_zotero_search(body: Dict[str, Any]) -> dict:
    """Search Zotero through the local Zotero Desktop API when enabled."""
    return capabilities.search_zotero(
        str(body.get("query") or ""), limit=int(body.get("limit") or 5)
    )


@router.post("/api/capabilities/zotero/test")
def post_capability_zotero_test(body: Dict[str, Any] | None = None) -> dict:
    """Probe the local Zotero Desktop API and record the decision."""
    return capabilities.test_zotero_connection()


@router.post("/api/capabilities/zotero/source")
def post_capability_zotero_source(body: Dict[str, Any]) -> dict:
    """Convert a selected Zotero item into an Idea Mining source payload."""
    item = body.get("item") if isinstance(body.get("item"), dict) else None
    return capabilities.zotero_source(
        item=item,
        item_key=str(body.get("item_key") or body.get("key") or ""),
    )


@router.post("/api/capabilities/zotero/import")
def post_capability_zotero_import(body: Dict[str, Any]) -> dict:
    """Parse pasted DOI/BibTeX/RIS/title metadata into an Idea Mining source."""
    return capabilities.import_zotero_source(str(body.get("text") or ""))


@router.post("/api/capabilities/audit-events")
def post_capability_audit_events(body: Dict[str, Any] | None = None) -> dict:
    """Read the local capability/tool audit log."""
    return capabilities.audit_events(limit=int((body or {}).get("limit") or 20))


__all__ = ["router"]
