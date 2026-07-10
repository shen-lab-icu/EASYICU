"""Idea Mining HTTP route adapters."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from easyicu.webserver import capabilities
from easyicu.webserver import settings as settings_store
from easyicu.webserver.ideas import mining as idea_mining_web
from easyicu.webserver.routes.request_parsing import body_bool

router = APIRouter()


def _pubmed_connector_gate(
    body: Dict[str, Any],
) -> tuple[Dict[str, Any], Optional[str]]:
    """Apply the global PubMed connector switch before network-capable idea APIs.

    Returns the (possibly patched) request body plus the block reason when the
    connector is off. The reason must be surfaced on the RESPONSE by the
    route — writing it into the request body is invisible to the caller.
    """
    settings = settings_store.load_settings()
    if settings.get("connector_pubmed_enabled", True):
        return body, None
    patched = dict(body or {})
    patched["allow_network"] = False
    reason = "connector_pubmed_enabled_false"
    capabilities.record_tool_event(
        "pubmed_connector_blocked",
        {"reason": reason, "path": "ideas"},
    )
    return patched, reason


@router.post("/api/ideas/mine")
def post_ideas_mine(body: Dict[str, Any]) -> dict:
    """Run local-first idea mining from user-supplied metadata/excerpts."""
    try:
        return idea_mining_web.mine_ideas(body)
    except idea_mining_web.IdeaMiningWebError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/ideas/resolve-source")
def post_ideas_resolve_source(body: Dict[str, Any]) -> dict:
    """Resolve a paper/PDF/frontier source seed into bounded metadata."""
    try:
        return idea_mining_web.resolve_source(body)
    except idea_mining_web.IdeaMiningWebError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/ideas/discover")
def post_ideas_discover(body: Dict[str, Any]) -> dict:
    """Run or prepare opt-in PubMed/frontier literature discovery."""
    try:
        patched, connector_reason = _pubmed_connector_gate(body)
        payload = idea_mining_web.discover_literature(patched)
        if connector_reason:
            payload["connector_disabled_reason"] = connector_reason
        capabilities.record_tool_event(
            "pubmed_discovery",
            {
                "allow_network": body_bool(body, "allow_network"),
                "search_performed": bool(payload.get("search_performed")),
                "status": payload.get("status"),
            },
        )
        return payload
    except idea_mining_web.IdeaMiningWebError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/ideas/ingest-pdf")
def post_ideas_ingest_pdf(body: Dict[str, Any]) -> dict:
    """Parse a selected local PDF into bounded metadata and a short excerpt."""
    try:
        return idea_mining_web.ingest_pdf_source(body)
    except idea_mining_web.IdeaMiningWebError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/ideas/literature-folder")
def post_ideas_literature_folder(body: Dict[str, Any]) -> dict:
    """Scan a local literature folder for PDF metadata and bounded excerpts."""
    try:
        return idea_mining_web.scan_literature_folder(body)
    except idea_mining_web.IdeaMiningWebError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/ideas/prior-art")
def post_ideas_prior_art(body: Dict[str, Any]) -> dict:
    """Run or prepare an opt-in bounded prior-art check for an idea."""
    try:
        patched, connector_reason = _pubmed_connector_gate(body)
        payload = idea_mining_web.check_prior_art(patched)
        if connector_reason:
            payload["connector_disabled_reason"] = connector_reason
        capabilities.record_tool_event(
            "pubmed_prior_art",
            {
                "allow_network": body_bool(body, "allow_network"),
                "search_performed": bool(
                    (payload.get("prior_art") or {}).get("search_performed")
                ),
                "status": (payload.get("prior_art") or {}).get("status"),
            },
        )
        return payload
    except idea_mining_web.IdeaMiningWebError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/ideas/plan")
def post_ideas_plan(body: Dict[str, Any]) -> dict:
    """Create or revise the pre-Agent study plan for an idea."""
    try:
        return idea_mining_web.plan_idea(body)
    except idea_mining_web.IdeaMiningWebError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/ideas/bounded-feasibility")
def post_ideas_bounded_feasibility(body: Dict[str, Any]) -> dict:
    """Run a bounded sample feasibility check for a mined idea."""
    try:
        return idea_mining_web.bounded_sample_feasibility(body)
    except idea_mining_web.IdeaMiningWebError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/ideas/handoff")
def post_ideas_handoff(body: Dict[str, Any]) -> dict:
    """Freeze a selected idea into an Agent handoff plan."""
    try:
        return idea_mining_web.create_handoff(body)
    except idea_mining_web.IdeaMiningWebError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/ideas/create-agent-project")
def post_ideas_create_agent_project(body: Dict[str, Any]) -> dict:
    """Create a metadata-only Agent Projects seed from an idea handoff."""
    try:
        return idea_mining_web.create_agent_project(body)
    except idea_mining_web.IdeaMiningWebError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/ideas/agent-projects")
def post_ideas_agent_projects(body: Dict[str, Any] | None = None) -> dict:
    """List Agent project seeds created by Idea Mining."""
    return idea_mining_web.list_agent_projects(body or {})


@router.post("/api/ideas/history")
def post_ideas_history(body: Dict[str, Any] | None = None) -> dict:
    """List local metadata-only idea mining runs."""
    return idea_mining_web.list_runs(body or {})


@router.post("/api/ideas/run")
def post_ideas_run(body: Dict[str, Any] | None = None) -> dict:
    """Load one persisted metadata-only idea mining run."""
    try:
        return idea_mining_web.get_run(body or {})
    except idea_mining_web.IdeaMiningWebError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


__all__ = ["router"]
