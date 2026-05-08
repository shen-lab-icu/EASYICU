"""Cross-database EasyICU standardized extraction availability.

This module is intentionally built around EasyICU's standardized extraction
surface, not SQL-level access. It answers:
"Can this EasyICU concept be derived for this database, and if not, which
concept dependency blocks or degrades it?"

The implementation reads EasyICU's packaged concept dictionary and data-source
registry. Recursive and callback-derived concepts are resolved through their
declared dependencies so external agents can reason over the cross-database
standardization layer before calling ``easyicu.load_concepts``, without seeing
database-specific item ids or tables.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field


PUBLIC_DATABASES = ("mimic", "miiv", "eicu", "aumc", "hirid", "sic")

_DATABASE_ALIASES = {
    "miii": "mimic",
    "mimiciii": "mimic",
    "mimic-iii": "mimic",
    "mimic3": "mimic",
    "mimiciv": "miiv",
    "mimic-iv": "miiv",
    "miv": "miiv",
    "sicdb": "sic",
}

_CONCEPT_ALIASES = {
    "creatinine": "crea",
    "creat": "crea",
    "urine_output": "urine",
    "uo": "urine",
    "aki": "kdigo_aki",
    "kdigo": "kdigo_aki",
    "kdigo_stage": "kdigo_aki",
    "aki_stage": "kdigo_aki",
    "sofa-2": "sofa2",
    "mortality": "death",
    "icu_mortality": "death",
}


class ConceptDatabaseAvailability(BaseModel):
    """Availability of one EasyICU concept on one database."""

    model_config = ConfigDict(extra="forbid")

    concept: str
    requested_concept: str
    database: str
    status: str = Field(description="full, degraded, or blocked")
    available: bool = False
    direct_source: bool = False
    reason: Optional[str] = None
    available_dependencies: List[str] = Field(default_factory=list)
    degraded_dependencies: List[str] = Field(default_factory=list)
    missing_dependencies: List[str] = Field(default_factory=list)


def normalize_database_name(database: str) -> str:
    key = (database or "").strip().lower().replace("_", "-")
    return _DATABASE_ALIASES.get(key, key)


def normalize_concept_name(concept: str) -> str:
    key = (concept or "").strip().lower().replace(" ", "_")
    return _CONCEPT_ALIASES.get(key, key)


def default_public_databases() -> List[str]:
    return list(PUBLIC_DATABASES)


def cross_database_concept_availability(
    *,
    concepts: Sequence[str],
    databases: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return concept-level availability for concepts x databases.

    The returned structure is JSON-serialisable and stable for MCP clients:

    ``{requested_concept: {database: ConceptDatabaseAvailability.dict()}}``.
    """

    dbs = _normalise_database_list(databases)
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for requested in concepts:
        canonical = normalize_concept_name(requested)
        per_db: Dict[str, Dict[str, Any]] = {}
        for db in dbs:
            cell = explain_concept_availability(
                concept=canonical,
                database=db,
                requested_concept=requested,
            )
            per_db[db] = cell.model_dump(mode="json")
        out[str(requested)] = per_db
    return out


def explain_concept_availability(
    *,
    concept: str,
    database: str,
    requested_concept: Optional[str] = None,
) -> ConceptDatabaseAvailability:
    db = normalize_database_name(database)
    requested = requested_concept or concept
    canonical = normalize_concept_name(concept)
    return _explain_concept_availability_cached(canonical, db, str(requested))


def hypothesis_cross_database_feasibility(
    *,
    concepts: Sequence[str],
    databases: Sequence[str],
) -> Dict[str, Any]:
    """Summarise concept availability into hypothesis-level DB feasibility."""

    dbs = _normalise_database_list(databases)
    deps = _unique(normalize_concept_name(c) for c in concepts if c)
    availability = cross_database_concept_availability(
        concepts=deps,
        databases=dbs,
    )
    feasibility: Dict[str, str] = {}
    degraded_reason: Dict[str, str] = {}
    for db in dbs:
        cells = [availability[concept][db] for concept in deps if concept in availability]
        if not cells:
            feasibility[db] = "blocked"
            degraded_reason[db] = "No concept dependencies were available to assess."
            continue
        statuses = [str(cell.get("status")) for cell in cells]
        if all(status == "full" for status in statuses):
            feasibility[db] = "full"
            continue
        if any(status == "blocked" for status in statuses):
            feasibility[db] = "blocked"
        else:
            feasibility[db] = "degraded"
        degraded_reason[db] = "; ".join(
            _reason_for_cell(cell)
            for cell in cells
            if str(cell.get("status")) != "full"
        )
    return {
        "concept_dependencies": deps,
        "cross_database_feasibility": feasibility,
        "degraded_reason": degraded_reason,
        "availability": availability,
    }


@lru_cache(maxsize=4096)
def _explain_concept_availability_cached(
    concept: str,
    database: str,
    requested_concept: str,
) -> ConceptDatabaseAvailability:
    from easyicu.resources import load_data_sources, load_dictionary

    db = normalize_database_name(database)
    dictionary = load_dictionary(include_sofa2=True)
    registry = load_data_sources()
    canonical = normalize_concept_name(concept)
    definition = dictionary.get(canonical)
    if definition is None:
        return ConceptDatabaseAvailability(
            concept=canonical,
            requested_concept=requested_concept,
            database=db,
            status="blocked",
            available=False,
            reason="concept_not_found",
        )

    try:
        config = registry.get(db)
    except KeyError:
        config = None

    direct_source = False
    if config is not None:
        try:
            direct_source = bool(definition.for_data_source(config))
        except Exception:
            direct_source = False
    else:
        direct_source = db in getattr(definition, "sources", {})

    if direct_source:
        return ConceptDatabaseAvailability(
            concept=canonical,
            requested_concept=requested_concept,
            database=db,
            status="full",
            available=True,
            direct_source=True,
            reason="direct_source_available",
        )

    dependencies = _concept_dependencies(definition)
    if not dependencies:
        return ConceptDatabaseAvailability(
            concept=canonical,
            requested_concept=requested_concept,
            database=db,
            status="blocked",
            available=False,
            reason="no_direct_source_or_dependencies",
        )

    dep_cells = [
        _explain_dependency(dep, db)
        for dep in dependencies
        if normalize_concept_name(dep) != canonical
    ]
    if not dep_cells:
        return ConceptDatabaseAvailability(
            concept=canonical,
            requested_concept=requested_concept,
            database=db,
            status="blocked",
            available=False,
            reason="recursive_dependency_cycle_or_empty_dependency_set",
        )

    available_dependencies = [
        cell.concept for cell in dep_cells if cell.status == "full"
    ]
    degraded_dependencies = [
        cell.concept for cell in dep_cells if cell.status == "degraded"
    ]
    missing_dependencies = [
        cell.concept for cell in dep_cells if cell.status == "blocked"
    ]
    if not missing_dependencies and not degraded_dependencies:
        status = "full"
        reason = "all_dependencies_available"
    elif available_dependencies or degraded_dependencies:
        status = "degraded"
        reason = "partial_dependency_availability"
    else:
        status = "blocked"
        reason = "all_dependencies_blocked"

    return ConceptDatabaseAvailability(
        concept=canonical,
        requested_concept=requested_concept,
        database=db,
        status=status,
        available=status != "blocked",
        direct_source=False,
        reason=reason,
        available_dependencies=available_dependencies,
        degraded_dependencies=degraded_dependencies,
        missing_dependencies=missing_dependencies,
    )


def _explain_dependency(dep: str, database: str) -> ConceptDatabaseAvailability:
    canonical = normalize_concept_name(dep)
    return _explain_concept_availability_cached(canonical, database, dep)


def _concept_dependencies(definition: Any) -> List[str]:
    deps: List[str] = []
    for attr in ("sub_concepts", "depends_on"):
        values = getattr(definition, attr, None) or []
        if isinstance(values, str):
            deps.append(values)
        elif isinstance(values, Iterable):
            deps.extend(str(v) for v in values)
    return _unique(deps)


def _normalise_database_list(databases: Optional[Sequence[str]]) -> List[str]:
    if not databases:
        return default_public_databases()
    return _unique(normalize_database_name(db) for db in databases if db)


def _unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = str(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _reason_for_cell(cell: Mapping[str, Any]) -> str:
    concept = str(cell.get("concept") or cell.get("requested_concept") or "concept")
    status = str(cell.get("status") or "unknown")
    missing = cell.get("missing_dependencies") or []
    degraded = cell.get("degraded_dependencies") or []
    reason = str(cell.get("reason") or "")
    parts = [f"{concept}={status}"]
    if missing:
        parts.append("missing=" + ",".join(map(str, missing[:6])))
    if degraded:
        parts.append("degraded=" + ",".join(map(str, degraded[:6])))
    if reason and reason not in {"direct_source_available", "all_dependencies_available"}:
        parts.append(reason)
    return " ".join(parts)


__all__ = [
    "ConceptDatabaseAvailability",
    "PUBLIC_DATABASES",
    "cross_database_concept_availability",
    "default_public_databases",
    "explain_concept_availability",
    "hypothesis_cross_database_feasibility",
    "normalize_concept_name",
    "normalize_database_name",
]
