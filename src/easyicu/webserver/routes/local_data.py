"""Local filesystem and single-workspace API routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from easyicu.webserver import dataio

router = APIRouter()


@router.get("/api/fs/list")
def fs_list(path: str | None = None) -> dict:
    """Server-side directory listing for the data-folder picker (local-first)."""
    return dataio.list_dir(path)


@router.post("/api/fs/mkdir")
def fs_mkdir(body: Dict[str, Any]) -> dict:
    """Create a local directory for picker destinations."""
    result = dataio.create_dir(body.get("path"))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/data/scan")
def data_scan(body: Dict[str, Any]) -> dict:
    """Inspect a folder: detect database, layout, and extraction readiness."""
    return dataio.scan_path(str(body.get("path", "")), body.get("source"))


@router.post("/api/workspace/summary")
def workspace_summary(body: Dict[str, Any]) -> dict:
    """Summarise an EasyICU export folder for Patient/Cohort review screens."""
    path = str(body.get("path", ""))
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    result = dataio.summarize_export_workspace(path)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


__all__ = ["router"]
