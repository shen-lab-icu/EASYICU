"""Cross-workspace and local export registry API routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from easyicu.webserver import dataio
from easyicu.webserver import export_download
from easyicu.webserver import sources as source_store
from easyicu.webserver.routes.request_parsing import body_bool

router = APIRouter()


@router.post("/api/workspaces/crossdb-summary")
def workspaces_crossdb_summary(body: Dict[str, Any]) -> dict:
    """Summarise two or more local EasyICU exports for Cross-DB preview."""
    paths = body.get("paths")
    if not isinstance(paths, list):
        raise HTTPException(status_code=400, detail="paths must be a list")
    result = dataio.summarize_crossdb_workspaces([str(p) for p in paths])
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.get("/api/workspaces/registry")
def workspaces_registry() -> dict:
    """Local export-source registry shared by Review, Cross-DB, Agent, Copilot."""
    return source_store.load_registry()


@router.post("/api/workspaces/registry")
def post_workspaces_registry(patch: Dict[str, Any]) -> dict:
    """Merge-update local export-source registry selections."""
    return source_store.save_registry(patch)


@router.post("/api/workspaces/register")
def post_workspaces_register(body: Dict[str, Any]) -> dict:
    """Validate and register one local EasyICU export folder."""
    path = str(body.get("path", ""))
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    result = source_store.register_source(
        path,
        label=body.get("label"),
        active=body_bool(body, "active", True),
        crossdb=body_bool(body, "crossdb", True),
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/workspaces/rename")
def post_workspaces_rename(body: Dict[str, Any]) -> dict:
    """Rename one registered local export source in registry metadata only."""
    path = str(body.get("path", ""))
    label = str(body.get("label", ""))
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    result = source_store.rename_source(path, label)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/workspaces/remove")
def post_workspaces_remove(body: Dict[str, Any]) -> dict:
    """Unregister one source. This never deletes export files from disk."""
    path = str(body.get("path", ""))
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    result = source_store.remove_source(path)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/api/workspaces/download")
def post_workspaces_download(body: Dict[str, Any]) -> StreamingResponse:
    """Download one manifest-bound registered export without accepting a path."""

    unexpected = sorted(set(body) - {"source_id"})
    if unexpected:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "registered_export_download_arguments_invalid",
                "reason": "Only an exact registered source id is accepted.",
            },
        )
    try:
        bundle = export_download.prepare_registered_export_bundle(
            str(body.get("source_id") or "")
        )
    except export_download.ExportDownloadError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error": exc.code, "reason": exc.message},
        ) from exc
    return StreamingResponse(
        export_download.iter_bundle_and_cleanup(bundle),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{bundle.filename}"',
            "X-EasyICU-Source-ID": bundle.source_id,
        },
    )


__all__ = ["router"]
