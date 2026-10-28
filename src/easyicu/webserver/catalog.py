"""Serialise the concept catalog into the shape the frontend expects.

The frontend (``static/js/screens-dict.js`` and friends) reads a global
``window.EU_CATALOG`` object:

    { groups, groupConcepts, dict, cov, desc, totalConcepts }

Labels and legacy database counts come from ``easyicu.concept_catalog``.
Display audience uses cached dictionary metadata. Instance coverage uses native
export manifests, with declared support as the legacy-source fallback;
active-export coverage shares the same source-registry snapshot.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from easyicu.concept import catalog as cc
from easyicu.outcome_availability import (
    OUTCOME_CONCEPT_SUPPORTED_DATABASES,
    structural_outcome_unavailability,
)
from easyicu.webserver import entity_ids as entity_id_contract
from easyicu.webserver.catalog_metadata import (
    CONCEPT_VARIANTS,
    catalog_definitions,
    concept_audience,
    concept_databases,
)

# Load packaged facts during startup, before the first page request.
catalog_definitions()

_ACTIVE_COVERAGE_FULL_READ_ROW_LIMIT = 1_000_000
#: Formats pandas reads too slowly to scan on every catalog request (a
#: 270k-row Excel extraction took 26-96 s and held every page at its pre-data
#: render); their coverage is the export schema, as for a large export.
_SCHEMA_ONLY_SUFFIXES = (".xlsx",)
#: Row-read coverage of recent exports, keyed by path, file sizes/mtimes and
#: the concept keys: the catalog is fetched on every page load.
_ROW_COVERAGE_CACHE: Dict[tuple, Dict[str, Any]] = {}
_ROW_COVERAGE_CACHE_LIMIT = 4
_ROW_COVERAGE_CACHE_LOCK = threading.Lock()
# Immutable per-file concept sets; None means no native-v2 manifest. Module
# existence is checked on use so a removed file cannot keep claiming coverage.
_INSTANCE_COVERAGE_CACHE: Dict[tuple, Optional[tuple]] = {}
_INSTANCE_COVERAGE_CACHE_LIMIT = 64
_INSTANCE_COVERAGE_CACHE_LOCK = threading.Lock()


def _registered_source_concepts(source: Dict[str, Any]) -> Optional[frozenset[str]]:
    """Read native availability through dataio, never module rows or schemas.

    None permits the declared-mapping fallback; an empty set is an authoritative
    native export with no provided concepts. Cache both by registry entry and
    manifest size/mtime, including both markers the manifest reader accepts.
    """
    from easyicu.webserver import dataio

    if not source.get("path"):
        return None
    path = Path(str(source["path"])).expanduser()
    signatures = []
    for name in dataio._MODULE_MANIFESTS:
        try:
            stat = (path / name).stat()
            signatures.append((name, stat.st_size, stat.st_mtime_ns))
        except OSError:
            signatures.append((name, None, None))
    key = (json.dumps(source, sort_keys=True), tuple(signatures))
    with _INSTANCE_COVERAGE_CACHE_LOCK:
        hit = key in _INSTANCE_COVERAGE_CACHE
        entries = _INSTANCE_COVERAGE_CACHE.get(key)
    if not hit:
        manifest = dataio.read_prepared_export_manifest(str(path))
        entries = None
        if manifest and manifest.get("schema_version") == dataio._NATIVE_EXPORT_SCHEMA_V2:
            provided = []
            for entry in dataio._manifest_file_entries(manifest):
                member = Path(str(entry["file"]))
                if member.is_absolute() or ".." in member.parts:
                    continue
                if entry.get("rows") == 0 or entry.get("availability") == "structurally_unavailable":
                    continue
                columns = entry.get("columns")
                if columns is None:
                    columns = entry.get("physical_schema")
                if columns is None:
                    columns = entry.get("column_metadata_columns") or []
                statuses = entry.get("concept_status") or {}
                concepts = frozenset(
                    str(column) for column in columns
                    if (statuses.get(column) or {}).get("availability") not in {
                        "produced_all_null", "structurally_unavailable_placeholder",
                    }
                )
                provided.append((str(member), concepts))
            entries = tuple(provided)
        with _INSTANCE_COVERAGE_CACHE_LOCK:
            _INSTANCE_COVERAGE_CACHE[key] = entries
            while len(_INSTANCE_COVERAGE_CACHE) > _INSTANCE_COVERAGE_CACHE_LIMIT:
                _INSTANCE_COVERAGE_CACHE.pop(next(iter(_INSTANCE_COVERAGE_CACHE)))
    if entries is None:
        return None
    return frozenset(
        concept for member, concepts in entries
        if (path / member).is_file()
        for concept in concepts
    )


def _declared_databases(key: str, definitions: Dict[str, Any]) -> list[str]:
    """Supported databases the concept dictionary maps ``key`` to, demos folded in."""
    declared = concept_databases(key, definitions)
    return [
        database for database in cc.SUPPORTED_DB_KEYS
        if database in declared or f"{database}_demo" in declared
    ]


def _coverage_metadata(
    concept_dict: Dict[str, Any],
    definitions: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Name the databases the concept dictionary maps each concept to.

    ``declared_databases`` is dictionary provenance (a declared source mapping,
    or for a derived concept its inputs' mappings), not observed coverage; the
    active export's coverage is reported separately. ``data_audit_databases``
    keeps the count an earlier data audit recorded (CONCEPT_DB_COVERAGE) where
    one exists; it names no database, so it never replaces the declared list.
    An outcome the outcome owner supports in no database is ``unsupported``,
    with that owner's reason code, rather than an unmapped concept.
    """
    definitions = catalog_definitions() if definitions is None else definitions
    audit = dict(cc.CONCEPT_DB_COVERAGE)
    supported_count = len(cc.SUPPORTED_DB_KEYS)
    metadata: Dict[str, Dict[str, Any]] = {}
    summary = {
        "supportedDatabases": supported_count,
        "mappedAll": 0,
        "mappedPartial": 0,
        "unmapped": 0,
        "unsupported": 0,
        "derived": 0,
        "dataAudited": 0,
    }

    for key in concept_dict:
        declared = _declared_databases(key, definitions)
        definition = definitions.get(key) or {}
        direct = any(mappings for mappings in (definition.get("sources") or {}).values())
        unsupported = not declared and key in OUTCOME_CONCEPT_SUPPORTED_DATABASES
        if direct:
            kind = "mapped"
        elif unsupported:
            kind = "unsupported"
        else:
            kind = "derived" if declared or key not in definitions else "unmapped"
        entry: Dict[str, Any] = {
            "kind": kind,
            "databases": len(declared),
            "declared_databases": declared,
            "basis": "concept_dictionary",
        }
        if not direct and key in cc.COMPOSITE_CONCEPT_OUTPUT_SOURCES:
            entry["source"] = cc.COMPOSITE_CONCEPT_OUTPUT_SOURCES[key]
        if unsupported:
            receipt = structural_outcome_unavailability(key, cc.SUPPORTED_DB_KEYS[0])
            entry["unavailable_reason"] = receipt.reason_code
        if key in audit:
            entry["data_audit_databases"] = int(audit[key])
            summary["dataAudited"] += 1
        metadata[key] = entry
        if entry["kind"] == "derived":
            summary["derived"] += 1
        if unsupported:
            summary["unsupported"] += 1
        elif not declared:
            summary["unmapped"] += 1
        elif len(declared) == supported_count:
            summary["mappedAll"] += 1
        else:
            summary["mappedPartial"] += 1

    return metadata, summary


def _active_export_coverage(
    concept_dict: Dict[str, Any], registry: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Summarise concept coverage in the active registered export.

    The Data Dictionary should not imply that static catalog mapping counts are
    current-export missingness. This pass reads only the active export's schemas
    and the needed ``stay_id + concept`` columns, then returns aggregate counts.
    It never returns row-level values or identifiers.
    """
    try:
        from easyicu.webserver import dataio
        from easyicu.webserver import sources
    except Exception as exc:  # noqa: BLE001 - catalog must still render.
        return {
            "status": "unavailable",
            "reason": str(exc),
            "concepts": {},
            "summary": {},
        }

    if registry is None:
        registry = sources.load_registry()
    active_path = registry.get("active_path")
    if not active_path:
        return {
            "status": "no_active_source",
            "concepts": {},
            "summary": {"included": 0, "notInExport": len(concept_dict)},
            "payload_scope": "aggregate_only_no_rows",
        }

    desc = dataio.describe_export_source(str(active_path))
    if not desc.get("ok"):
        return {
            "status": "invalid_active_source",
            "reason": desc.get("error", "invalid_export"),
            "concepts": {},
            "summary": {"included": 0, "notInExport": len(concept_dict)},
            "payload_scope": "aggregate_only_no_rows",
        }

    files = [f for f in desc.get("files", []) if f.get("file")]
    total_rows = int((desc.get("summary") or {}).get("total_rows") or 0)
    if total_rows > _ACTIVE_COVERAGE_FULL_READ_ROW_LIMIT or any(
        str(f["file"]).lower().endswith(_SCHEMA_ONLY_SUFFIXES) for f in files
    ):
        return _active_export_schema_coverage(desc, concept_dict)

    path = Path(str(active_path)).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    key = _row_coverage_cache_key(path, files, concept_dict)
    if key is not None:
        with _ROW_COVERAGE_CACHE_LOCK:
            cached = _ROW_COVERAGE_CACHE.get(key)
        if cached is not None:
            return copy.deepcopy(cached)
    result = _active_export_row_coverage(dataio, desc, path, files, concept_dict)
    if key is not None:
        with _ROW_COVERAGE_CACHE_LOCK:
            _ROW_COVERAGE_CACHE[key] = copy.deepcopy(result)
            while len(_ROW_COVERAGE_CACHE) > _ROW_COVERAGE_CACHE_LIMIT:
                _ROW_COVERAGE_CACHE.pop(next(iter(_ROW_COVERAGE_CACHE)))
    return result


def _row_coverage_cache_key(
    path: Path, files: list[Dict[str, Any]], concept_dict: Dict[str, Any]
) -> Optional[tuple]:
    """Identity of the files a row-read coverage pass reads, or ``None``."""
    try:
        stats = tuple(
            (str(f["file"]), st.st_size, st.st_mtime_ns)
            for f in files
            for st in [(path / str(f["file"])).stat()]
        )
    except OSError:
        return None
    return str(path), stats, tuple(sorted(concept_dict))


def _active_export_row_coverage(
    dataio: Any,
    desc: Dict[str, Any],
    path: Path,
    files: list[Dict[str, Any]],
    concept_dict: Dict[str, Any],
) -> Dict[str, Any]:
    stay_ids = dataio._fast_stay_ids(path, files)
    denominator = len(stay_ids) if stay_ids is not None else None
    concept_keys = set(concept_dict)
    coverage: Dict[str, Dict[str, Any]] = {}

    for file_meta in files:
        columns = [str(c) for c in file_meta.get("columns") or []]
        present = [c for c in columns if c in concept_keys]
        if not present:
            continue
        file_path = path / str(file_meta["file"])
        file_coverage = _file_concept_coverage(
            file_path=file_path,
            module=str(file_meta.get("module") or ""),
            concepts=present,
            stay_ids=stay_ids,
            denominator=denominator,
        )
        for concept, meta in file_coverage.items():
            prev = coverage.get(concept)
            if prev is None or _coverage_sort_key(meta) > _coverage_sort_key(prev):
                coverage[concept] = meta

    high = sum(1 for v in coverage.values() if _coverage_band(v) == "high")
    medium = sum(1 for v in coverage.values() if _coverage_band(v) == "medium")
    low = sum(1 for v in coverage.values() if _coverage_band(v) == "low")
    return {
        "status": "ready",
        "source": {
            "label": desc.get("label"),
            "database": desc.get("database"),
            "path_hash": hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12],
        },
        "denominator": denominator,
        "concepts": coverage,
        "summary": {
            "included": len(coverage),
            "notInExport": max(0, len(concept_dict) - len(coverage)),
            "high": high,
            "medium": medium,
            "low": low,
        },
        "payload_scope": "aggregate_only_no_rows",
        "coverage_basis": "non_null_unique_stay_intersection",
    }


def _active_export_schema_coverage(
    desc: Dict[str, Any], concept_dict: Dict[str, Any]
) -> Dict[str, Any]:
    path = Path(str(desc.get("path") or "local")).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    denominator = (desc.get("summary") or {}).get("stays")
    concept_keys = set(concept_dict)
    coverage: Dict[str, Dict[str, Any]] = {}
    for file_meta in desc.get("files", []) or []:
        module = str(file_meta.get("module") or "")
        file_name = str(file_meta.get("file") or "")
        for concept in [
            str(c) for c in file_meta.get("columns") or [] if str(c) in concept_keys
        ]:
            coverage.setdefault(
                concept,
                {
                    "kind": (
                        "active_event_schema"
                        if _event_like_concept(module, concept)
                        else "active_export_schema"
                    ),
                    "module": module,
                    "file": file_name,
                    "coverage_pct": None,
                    "observed_entities": None,
                    "denominator": denominator,
                    "basis": "column_present_in_export_schema",
                },
            )
    return {
        "status": "ready",
        "mode": "schema_only",
        "source": {
            "label": desc.get("label"),
            "database": desc.get("database"),
            "path_hash": hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12],
        },
        "denominator": denominator,
        "concepts": coverage,
        "summary": {
            "included": len(coverage),
            "notInExport": max(0, len(concept_dict) - len(coverage)),
            "high": 0,
            "medium": 0,
            "low": 0,
            "schemaOnly": len(coverage),
        },
        "payload_scope": "aggregate_only_no_rows",
        "coverage_basis": "column_present_in_export_schema",
    }


def _file_concept_coverage(
    *,
    file_path: Path,
    module: str,
    concepts: list[str],
    stay_ids: set[str] | None,
    denominator: int | None,
) -> Dict[str, Dict[str, Any]]:
    try:
        import pandas as pd
    except Exception:
        return {}

    usecols = ["stay_id", *concepts]
    try:
        if file_path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(file_path, columns=usecols)
        elif file_path.suffix.lower() == ".xlsx":
            frame = pd.read_excel(file_path, usecols=usecols)
        else:
            frame = pd.read_csv(file_path, usecols=usecols)
    except Exception:
        return {
            concept: {
                "kind": "active_unreadable",
                "module": module,
                "file": file_path.name,
                "coverage_pct": None,
                "observed_entities": None,
                "denominator": denominator,
                "basis": "column_present_but_aggregate_read_failed",
            }
            for concept in concepts
        }

    if "stay_id" not in frame.columns:
        return {}
    norm_stays = frame["stay_id"].map(entity_id_contract.normalize_entity_id).astype(str)
    valid_frame = frame.assign(_easyicu_stay_id=norm_stays)
    valid_frame = valid_frame[valid_frame["_easyicu_stay_id"].astype(bool)]
    if stay_ids is not None:
        valid_frame = valid_frame[valid_frame["_easyicu_stay_id"].isin(stay_ids)]
        denom = denominator or len(stay_ids)
    else:
        denom = valid_frame["_easyicu_stay_id"].nunique()

    out: Dict[str, Dict[str, Any]] = {}
    for concept in concepts:
        if concept not in valid_frame.columns:
            continue
        series = valid_frame[concept]
        present_mask = series.notna()
        if series.dtype == object:
            present_mask = present_mask & series.astype(str).str.strip().ne("")
        observed = int(valid_frame.loc[present_mask, "_easyicu_stay_id"].nunique())
        pct = round(observed / denom * 100, 1) if denom else None
        kind = (
            "active_event" if _event_like_concept(module, concept) else "active_export"
        )
        out[concept] = {
            "kind": kind,
            "module": module,
            "file": file_path.name,
            "coverage_pct": pct,
            "observed_entities": observed,
            "denominator": denom,
            "basis": "non_null_unique_stay_intersection",
        }
    return out


def _event_like_concept(module: str, concept: str) -> bool:
    if module.startswith("sepsis3") or concept.startswith("sep3_"):
        return True
    return concept in {
        "death",
        "mort_28d",
        "mort_90d",
        "mort_365d",
        "icu_readmission",
        "persistent_critical_illness",
        "mech_vent",
        "vent_ind",
        "aki",
        "rrt",
        "vaso_ind",
    }


def _coverage_sort_key(meta: Dict[str, Any]) -> tuple[int, float]:
    pct = meta.get("coverage_pct")
    value = float(pct) if isinstance(pct, (int, float)) else -1.0
    ready = 1 if meta.get("kind") in {"active_export", "active_event"} else 0
    return ready, value


def _coverage_band(meta: Dict[str, Any]) -> str:
    pct = meta.get("coverage_pct")
    if not isinstance(pct, (int, float)):
        return "low"
    if pct >= 80:
        return "high"
    if pct >= 50:
        return "medium"
    return "low"


def build_catalog() -> Dict[str, Any]:
    """Keep legacy labels/counts and enrich each ``conceptCoverage`` entry."""
    from easyicu.webserver import sources

    # groups: ordered [key, name_en, name_zh] mirroring the curated order.
    groups = [
        [gk, *cc.CONCEPT_GROUP_NAMES.get(gk, (gk, gk))]
        for gk in cc.CONCEPT_GROUPS_INTERNAL
    ]
    group_concepts = {
        gk: list(members) for gk, members in cc.CONCEPT_GROUPS_INTERNAL.items()
    }

    # dict[k] = [name_en, name_zh, unit]; tuples -> lists for JSON.
    concept_dict = {k: list(v) for k, v in cc.CONCEPT_DICTIONARY.items()}
    desc = {k: list(v) for k, v in cc.CONCEPT_DESCRIPTIONS.items()}
    cov = dict(cc.CONCEPT_DB_COVERAGE)
    definitions = catalog_definitions()
    concept_coverage, coverage_summary = _coverage_metadata(concept_dict, definitions)
    registry = sources.load_registry()
    registered_sources = [
        source
        for source in registry.get("sources") or []
        if source.get("ok") and source.get("database")
    ]
    instance_databases = sorted({source["database"] for source in registered_sources})
    native_support: Dict[str, set[str]] = {}
    fallback_databases = set()
    for source in registered_sources:
        concepts = _registered_source_concepts(source)
        if concepts is None:
            fallback_databases.add(source["database"])
        else:
            for concept in concepts:
                native_support.setdefault(concept, set()).add(source["database"])
    # A mixed registry is labelled conservatively: some of its availability
    # decisions still rely on declarations, including negative decisions.
    instance_basis = "declared_mapping" if fallback_databases else "export_manifest"
    for key, coverage in concept_coverage.items():
        coverage["audience"] = concept_audience(key, definitions.get(key, {}))
        supported = native_support.get(key, set())
        if fallback_databases:
            supported = supported | (concept_databases(key, definitions) & fallback_databases)
        coverage["instance_databases_covered"] = sorted(supported)
        coverage["instance_coverage_basis"] = instance_basis
    # A database that holds a variable only as a substitute (age in bands, ICU
    # death for death) says so in the dictionary, from the data map's facts.
    from easyicu.webserver.data_map import concept_substitutes

    for key, by_database in concept_substitutes().items():
        if key in concept_coverage:
            concept_coverage[key]["substitutes"] = by_database
    active_export_coverage = _active_export_coverage(concept_dict, registry)

    return {
        "groups": groups,
        "groupConcepts": group_concepts,
        "dict": concept_dict,
        "desc": desc,
        "variants": {key: list(members) for key, members in CONCEPT_VARIANTS.items()},
        "cov": cov,
        "conceptCoverage": concept_coverage,
        "coverageSummary": coverage_summary,
        "activeExportCoverage": active_export_coverage,
        "supportedDbs": list(cc.SUPPORTED_DB_KEYS),
        "instance_databases": instance_databases,
        "totalConcepts": len(concept_dict),
    }


def build_concept_lineage(concept_id: str) -> Dict[str, Any] | None:
    """Return declared cross-database lineage for one catalog concept.

    This is dictionary provenance, not an observed-data claim. Missing fields
    remain null so the UI never invents a source column or transformation.
    """
    from easyicu.concept.loader import load_concept_dict_cached

    raw = load_concept_dict_cached().get(concept_id)
    if not isinstance(raw, dict):
        return None
    data_sources_path = Path(__file__).resolve().parents[1] / "data" / "data-sources.json"
    with data_sources_path.open(encoding="utf-8") as handle:
        source_catalog = json.load(handle)
    source_profiles = {
        str(item.get("name")): item for item in source_catalog if isinstance(item, dict)
    }
    lanes = []
    raw_sources = raw.get("sources") if isinstance(raw.get("sources"), dict) else {}
    for database in cc.SUPPORTED_DB_KEYS:
        mappings = raw_sources.get(database)
        if not isinstance(mappings, list) or not mappings:
            continue
        profile = source_profiles.get(database, {})
        table_catalog = profile.get("tables") if isinstance(profile.get("tables"), dict) else {}
        serialized = []
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            table = mapping.get("table")
            table_meta = table_catalog.get(table, {}) if table else {}
            defaults = table_meta.get("defaults") if isinstance(table_meta, dict) else {}
            defaults = defaults if isinstance(defaults, dict) else {}
            ids = mapping.get("ids")
            if ids is not None and not isinstance(ids, list):
                ids = [ids]
            serialized.append(
                {
                    "table": table,
                    "selector_field": mapping.get("sub_var"),
                    "selector_values": ids,
                    "value_field": mapping.get("value_var")
                    or mapping.get("val_var")
                    or defaults.get("val_var"),
                    "unit_field": mapping.get("unit_var") or defaults.get("unit_var"),
                    "time_field": mapping.get("index_var")
                    or mapping.get("time_var")
                    or defaults.get("index_var"),
                    "callback": mapping.get("callback") or raw.get("callback"),
                    "class_name": mapping.get("class") or mapping.get("class_name"),
                }
            )
        if serialized:
            display_name = (
                (profile.get("profile") or {}).get("display_name")
                if isinstance(profile.get("profile"), dict)
                else None
            )
            lanes.append(
                {
                    "database": database,
                    "label": display_name or database.upper(),
                    "mappings": serialized,
                }
            )
    units = raw.get("unit")
    if isinstance(units, list):
        canonical_unit = str(units[0]) if units else None
        unit_aliases = [str(value) for value in units[1:]]
    else:
        canonical_unit = units
        unit_aliases = []
    return {
        "concept": concept_id,
        "name": (cc.CONCEPT_DICTIONARY.get(concept_id) or [concept_id])[0],
        "canonical": {
            "unit": canonical_unit,
            "unit_aliases": unit_aliases,
            "minimum": raw.get("min"),
            "maximum": raw.get("max"),
            "aggregate": raw.get("aggregate"),
        },
        "lanes": lanes,
        "scope": "declared_dictionary_lineage_not_observed_data",
    }
