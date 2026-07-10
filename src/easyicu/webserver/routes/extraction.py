"""Data extraction filter API routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from easyicu.webserver import extraction_filters

router = APIRouter()


@router.post("/api/extraction/filter-options")
def extraction_filter_options(body: Dict[str, Any]) -> dict:
    """Return bounded real-source filter metadata for Data Extraction."""
    try:
        return extraction_filters.filter_options(body)
    except extraction_filters.ExtractionFilterError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/extraction/filter-preview")
def extraction_filter_preview(body: Dict[str, Any]) -> dict:
    """Apply supported extraction metadata filters; unsupported filters fail closed."""
    try:
        result = extraction_filters.filter_preview(body)
    except extraction_filters.ExtractionFilterError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


__all__ = ["router"]
