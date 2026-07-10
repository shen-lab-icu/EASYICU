"""Patient, cohort, and cross-database review API routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from easyicu.webserver import cohort_review
from easyicu.webserver import crossdb_review
from easyicu.webserver import patient_drilldown

router = APIRouter()


@router.post("/api/patient-review/drilldown")
def patient_review_drilldown(body: Dict[str, Any]) -> dict:
    """Return bounded real Patient Review aggregates plus one entity drilldown."""
    try:
        return patient_drilldown.patient_review_drilldown(body)
    except patient_drilldown.PatientReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/patient-review/sources")
def patient_review_sources(body: Dict[str, Any] | None = None) -> dict:
    """Return metadata-only local export candidates for Patient Review."""
    return patient_drilldown.patient_review_sources(body or {})


@router.post("/api/cohort-review/summary")
def cohort_review_summary(body: Dict[str, Any]) -> dict:
    """Return bounded real Cohort Review aggregates for the active export."""
    try:
        return cohort_review.cohort_review_summary(body)
    except cohort_review.CohortReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/crossdb-review/summary")
def crossdb_review_summary(body: Dict[str, Any]) -> dict:
    """Return bounded real Cross-DB descriptive aggregates for registered exports."""
    try:
        return crossdb_review.crossdb_review_summary(body)
    except crossdb_review.CrossdbReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/crossdb-review/raw-distribution")
def crossdb_raw_distribution(body: Dict[str, Any]) -> dict:
    """Return bounded real Cross-DB density aggregates from a local ICU data root."""
    try:
        return crossdb_review.crossdb_raw_distribution(body)
    except crossdb_review.CrossdbReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/crossdb-review/raw-root-scan")
def crossdb_raw_root_scan(body: Dict[str, Any]) -> dict:
    """Preflight a local raw ICU data root before launching Cross-DB loading."""
    try:
        return crossdb_review.crossdb_raw_root_scan(body)
    except crossdb_review.CrossdbReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


@router.post("/api/crossdb-review/demo-distribution")
def crossdb_demo_distribution(body: Dict[str, Any]) -> dict:
    """Return bounded legacy-seeded Cross-DB demo density aggregates."""
    try:
        return crossdb_review.crossdb_demo_distribution(body)
    except crossdb_review.CrossdbReviewError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


__all__ = ["router"]
