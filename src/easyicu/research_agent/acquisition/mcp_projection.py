"""Path-free MCP projections over EasyICU's prepared-export catalog.

Owner
-----
The acquisition layer owns the scientific meaning of "available concept" and
"coverage".  MCP is only an adapter: it may project those typed results, but
must not inspect Parquet files independently, infer aliases, or reproduce
coverage policy.

Public contract
---------------
``project_export_concepts`` and ``project_export_coverage`` accept an already
authorized export directory and return JSON-safe metadata.  They never return
the host path, patient rows, arbitrary table contents, or raw SQL.  Filesystem
confinement and MCP scopes remain the transport owner's responsibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from .catalog import CatalogConcept, assess_coverage, build_available_catalog


MCP_EXPORT_CATALOG_SCHEMA_VERSION = "easyicu.mcp-export-catalog/1"
MCP_EXPORT_COVERAGE_SCHEMA_VERSION = "easyicu.mcp-export-coverage/1"
MAX_MCP_EXPORT_CONCEPTS = 500
MAX_MCP_COVERAGE_CONCEPTS = 200


def _clean_sequence(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            text
            for value in values
            if (text := str(value or "").strip())
        )
    )


def _validated_limit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("limit must be an integer")
    if value < 1 or value > MAX_MCP_EXPORT_CONCEPTS:
        raise ValueError(
            f"limit must be between 1 and {MAX_MCP_EXPORT_CONCEPTS}"
        )
    return value


def _module_name(concept: CatalogConcept) -> str:
    name = Path(str(concept.file_name or "")).stem.strip().casefold()
    return name or "unknown"


def _concept_projection(concept: CatalogConcept) -> Dict[str, Any]:
    return {
        "concept_id": concept.concept_id,
        "description": concept.description,
        "category": concept.category,
        "module": _module_name(concept),
        "column_role": concept.column_role,
        "typed_metadata": bool(concept.typed_metadata),
        "methodology": concept.methodology,
        "selection_mode": concept.selection_mode,
        "selection_note": concept.selection_note,
        "canonical_alternative": concept.canonical_alternative,
    }


def _source_projection(source_ref: str) -> Dict[str, Any]:
    return {
        "source_ref": str(source_ref),
        "path_returned": False,
        "patient_rows_returned": False,
    }


def project_export_concepts(
    *,
    export_dir: Path,
    source_ref: str,
    query: str = "",
    modules: Sequence[str] = (),
    limit: int = 200,
) -> Dict[str, Any]:
    """Project the physical export catalog without returning host locations."""

    resolved_limit = _validated_limit(limit)
    requested_modules = {value.casefold() for value in _clean_sequence(modules)}
    query_text = str(query or "").strip().casefold()
    catalog = build_available_catalog(export_dir)
    candidates = []
    for concept in catalog.concepts:
        module = _module_name(concept)
        if requested_modules and module not in requested_modules:
            continue
        searchable = " ".join(
            (
                concept.concept_id,
                concept.description,
                concept.category,
                concept.methodology,
                concept.selection_note,
                concept.canonical_alternative,
            )
        ).casefold()
        if query_text and query_text not in searchable:
            continue
        candidates.append(concept)
    candidates.sort(key=lambda item: (item.concept_id.casefold(), item.concept_id))
    returned = candidates[:resolved_limit]
    module_counts: Dict[str, int] = {}
    for concept in catalog.concepts:
        module = _module_name(concept)
        module_counts[module] = module_counts.get(module, 0) + 1
    return {
        "schema_version": MCP_EXPORT_CATALOG_SCHEMA_VERSION,
        "source": _source_projection(source_ref),
        "catalog_concept_count": len(catalog.concepts),
        "matched_concept_count": len(candidates),
        "returned_concept_count": len(returned),
        "truncated": len(candidates) > len(returned),
        "filters": {
            "query": query_text,
            "modules": sorted(requested_modules),
            "limit": resolved_limit,
        },
        "module_counts": dict(sorted(module_counts.items())),
        "concepts": [_concept_projection(concept) for concept in returned],
        "privacy": {
            "patient_rows_returned": False,
            "host_path_returned": False,
            "raw_sql_returned": False,
        },
    }


def project_export_coverage(
    *,
    export_dir: Path,
    source_ref: str,
    concepts: Sequence[str],
) -> Dict[str, Any]:
    """Project owner-computed coverage for an explicit, non-empty concept set."""

    requested = _clean_sequence(concepts)
    if not requested:
        raise ValueError("concepts must contain at least one non-empty concept id")
    if len(requested) > MAX_MCP_COVERAGE_CONCEPTS:
        raise ValueError(
            "concepts may contain at most "
            f"{MAX_MCP_COVERAGE_CONCEPTS} unique concept ids"
        )
    catalog = build_available_catalog(export_dir)
    coverage = assess_coverage(requested, catalog)
    return {
        "schema_version": MCP_EXPORT_COVERAGE_SCHEMA_VERSION,
        "source": _source_projection(source_ref),
        "catalog_concept_count": len(catalog.concepts),
        "requested": list(coverage.requested),
        "available": list(coverage.available),
        "missing": list(coverage.missing),
        "resolved": dict(coverage.resolved),
        "sufficient": bool(coverage.sufficient),
        "advice": list(coverage.advice),
        "interpretation": (
            "All explicitly requested concepts resolved in this prepared export."
            if coverage.sufficient
            else "One or more requested concepts require re-extraction."
        ),
        "claim_boundary": (
            "Coverage describes only the explicit requested concept set; it "
            "does not establish that the data are scientifically sufficient "
            "for a research question."
        ),
        "privacy": {
            "patient_rows_returned": False,
            "host_path_returned": False,
            "raw_sql_returned": False,
        },
    }


__all__ = [
    "MAX_MCP_COVERAGE_CONCEPTS",
    "MAX_MCP_EXPORT_CONCEPTS",
    "MCP_EXPORT_CATALOG_SCHEMA_VERSION",
    "MCP_EXPORT_COVERAGE_SCHEMA_VERSION",
    "project_export_concepts",
    "project_export_coverage",
]
