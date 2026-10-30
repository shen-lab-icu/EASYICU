"""Project dictionary mappings as reference metadata, never as consumed lineage."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib import resources
import json
from typing import Any

from easyicu.databases.profiles import get_database_profile
from easyicu.config import load_src_cfg
from easyicu.research_agent.canonical_json import canonical_sha256


@lru_cache(maxsize=1)
def _dictionary() -> tuple[dict[str, Any], str]:
    data = resources.files("easyicu.data").joinpath("concept-dict.json").read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("concept dictionary must be an object")
    return value, hashlib.sha256(data).hexdigest()


def source_concept_reference(concept_id: str, database: str) -> dict[str, Any] | None:
    """Return the declared mapping for one database; never inspect patient data.

    The mapping describes the packaged dictionary, not the inputs consumed by
    a particular run or the correctness of each measurement's unit.
    """
    try:
        key = get_database_profile(str(database or "")).key
    except (KeyError, ValueError):
        return None
    try:
        payload, digest = _dictionary()
    except (OSError, ValueError, UnicodeDecodeError):
        # A missing/broken optional dictionary must not make an otherwise
        # valid source catalog unusable or manufacture a mapping.
        return None
    item = payload.get(str(concept_id))
    if not isinstance(item, dict) or not isinstance(item.get("sources"), dict):
        return None
    raw = item["sources"].get(key)
    if not isinstance(raw, list) or not raw:
        return None
    try:
        config = load_src_cfg(key)
        config_digest = canonical_sha256(config.model_dump(mode="json"))
    except (KeyError, OSError, ValueError, TypeError):
        config = None
        config_digest = None
    mappings = []
    for entry in raw[:20]:
        if not isinstance(entry, dict):
            continue
        mapping: dict[str, Any] = {}
        for name in ("table", "sub_var", "value_var", "unit_var", "index_var", "dur_var"):
            if isinstance(entry.get(name), str) and entry[name].strip():
                mapping[name] = entry[name][:120]
        ids = entry.get("ids")
        if isinstance(ids, (str, int, float)) and not isinstance(ids, bool):
            ids = [ids]
        if isinstance(ids, list):
            mapping["ids"] = [v for v in ids[:100] if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
            mapping["ids_truncated"] = len(ids) > 100
        if entry.get("callback") or entry.get("regex") or entry.get("params"):
            mapping["additional_rules_present"] = True
        table = config.tables.get(entry.get("table")) if config and isinstance(entry.get("table"), str) else None
        defaults = getattr(table, "defaults", None)
        if defaults is not None:
            names = {name: getattr(defaults, name, None) for name in ("id_var", "index_var", "val_var", "unit_var")}
            mapping["source_table_defaults"] = {
                name: value[:120] for name, value in names.items()
                if isinstance(value, str) and value.strip()
            }
        if mapping:
            mappings.append(mapping)
    if not mappings:
        return None
    unit = item.get("unit")
    units = [unit] if isinstance(unit, str) and unit.strip() else []
    if isinstance(unit, list):
        units = [u[:80] for u in unit[:20] if isinstance(u, str) and u.strip()]
    return {"database": key, "dictionary_sha256": digest, "source_config_sha256": config_digest, "declared_units": units,
            "declared_mappings": mappings, "scope": "packaged_dictionary_reference_only",
            "actual_source_reads_verified": False, "record_units_verified": False}
