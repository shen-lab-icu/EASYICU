"""Declared table → field → variable map for one database (Data Dictionary).

The Data Dictionary's "Source tables" view reads this. It is dictionary
provenance, not observed data: every row comes from the same per-database
dictionary the extraction resolves (``resources.load_dictionary`` with the
SOFA-2 overlay), and the table columns come from the packaged
``data-sources.json``. No database file is opened.

A variable with source rows in this database is *mapped*: each row names the
table, the selector (for example ``itemid``) and its declared values, and the
value, unit and time fields. A variable without source rows but with input
variables is *derived*: it is computed from those inputs by its callback. A
*function* item has a source row without a table: its callback reads the stay
record itself (for example the length of stay).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from easyicu.concept import catalog as cc

SCOPE = "declared_dictionary_lineage_not_observed_data"


@functools.lru_cache(maxsize=1)
def _source_profiles() -> Dict[str, Dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / "data" / "data-sources.json"
    with path.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    return {str(item.get("name")): item for item in catalog if isinstance(item, dict)}


def database_label(database: str) -> str:
    profile = _source_profiles().get(database, {})
    meta = profile.get("profile") if isinstance(profile.get("profile"), dict) else {}
    return str(meta.get("display_name") or database.upper())


def mapped_databases() -> List[str]:
    """Supported databases with a packaged table profile, in catalog order."""
    profiles = _source_profiles()
    return [db for db in cc.SUPPORTED_DB_KEYS if isinstance(profiles.get(db, {}).get("tables"), dict)]


def _resolver_dictionary(database: str):
    from easyicu.resources import load_dictionary

    return load_dictionary(include_sofa2=True, source_name=database)


def _as_list(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    return list(value) if isinstance(value, (list, tuple)) else [value]


def serialize_source(source: Any, table_meta: Dict[str, Any], concept_callback: Any = None) -> Dict[str, Any]:
    """One declared source row; a field the dictionary leaves out stays None."""
    defaults = table_meta.get("defaults") if isinstance(table_meta.get("defaults"), dict) else {}
    get = (lambda key: source.get(key)) if isinstance(source, dict) else (lambda key: getattr(source, key, None))
    return {
        "table": get("table"),
        "selector_field": get("sub_var"),
        "selector_values": _as_list(get("ids")),
        "value_field": get("value_var") or get("val_var") or defaults.get("val_var"),
        "unit_field": get("unit_var") or defaults.get("unit_var"),
        "time_field": get("index_var") or get("time_var") or defaults.get("index_var"),
        "callback": get("callback") or concept_callback,
        "class_name": get("class_name") or get("class"),
    }


def resolver_definition(concept_id: str, database: Optional[str] = None) -> Any:
    """The definition extraction resolves (concept dictionary + SOFA-2 overlay)."""
    from easyicu.resources import load_dictionary

    dictionary = _resolver_dictionary(database) if database else load_dictionary(include_sofa2=True)
    return dictionary.get(concept_id)


def concept_inputs(concept_id: str, database: Optional[str] = None) -> List[str]:
    """The input variables a derived variable is computed from (declared)."""
    definition = resolver_definition(concept_id, database)
    return [str(name) for name in (_as_list(getattr(definition, "sub_concepts", None)) or [])]


def build_source_map(database: str) -> Optional[Dict[str, Any]]:
    """Every declared table/field/variable row of one database, grouped by table."""
    if database not in mapped_databases():
        return None
    profile = _source_profiles()[database]
    table_catalog: Dict[str, Any] = profile["tables"]
    tables: Dict[str, Dict[str, Any]] = {}
    derived: List[Dict[str, Any]] = []
    functions: List[Dict[str, Any]] = []
    for concept_id, definition in _resolver_dictionary(database).items():
        sources = (getattr(definition, "sources", None) or {}).get(database) or []
        if sources:
            for source in sources:
                table_meta = table_catalog.get(getattr(source, "table", None)) or {}
                row = serialize_source(source, table_meta if isinstance(table_meta, dict) else {},
                                       getattr(definition, "callback", None))
                table = str(row.pop("table") or "")
                if not table:
                    # A function item reads the stay record through its callback;
                    # the dictionary names no single table for it.
                    functions.append({"concept": concept_id, "callback": row["callback"]})
                    continue
                entry = tables.setdefault(table, {"table": table, "rows": []})
                entry["rows"].append({"concept": concept_id, **row})
            continue
        inputs = _as_list(getattr(definition, "sub_concepts", None)) or []
        if inputs:
            derived.append({
                "concept": concept_id,
                "inputs": [str(name) for name in inputs],
                "callback": getattr(definition, "callback", None),
            })
    serialized = []
    for name, entry in tables.items():
        meta = table_catalog.get(name) if isinstance(table_catalog.get(name), dict) else {}
        columns = list((meta.get("cols") or {}).keys()) if isinstance(meta.get("cols"), dict) else []
        used: Dict[str, List[str]] = {}
        for row in entry["rows"]:
            for role in ("selector_field", "value_field", "unit_field", "time_field"):
                field = row.get(role)
                if field:
                    used.setdefault(str(field), [])
                    if role not in used[str(field)]:
                        used[str(field)].append(role)
        serialized.append({
            "table": name,
            "declared_in_profile": bool(meta),
            "columns": columns,
            "used_fields": [{"field": field, "roles": roles} for field, roles in used.items()],
            "concepts": sorted({row["concept"] for row in entry["rows"]}),
            "rows": sorted(entry["rows"], key=lambda row: row["concept"]),
        })
    serialized.sort(key=lambda item: (-len(item["concepts"]), item["table"]))
    return {
        "database": database,
        "label": database_label(database),
        "databases": [{"database": db, "label": database_label(db)} for db in mapped_databases()],
        "tables": serialized,
        "derived": sorted(derived, key=lambda item: item["concept"]),
        "functions": sorted(functions, key=lambda item: item["concept"]),
        "scope": SCOPE,
    }
