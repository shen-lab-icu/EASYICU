"""Serialise the concept catalog into the shape the frontend expects.

The frontend (``static/js/screens-dict.js`` and friends) reads a global
``window.EU_CATALOG`` object:

    { groups, groupConcepts, dict, cov, desc, totalConcepts }

All of that data already exists, hand-curated for the UI, in
``easyicu.concept_catalog``. This module is the single source of
truth for the migration's first read-only endpoint — it just reshapes
those dicts; it does not recompute anything.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

from easyicu.concept import catalog as cc

_ACTIVE_COVERAGE_FULL_READ_ROW_LIMIT = 1_000_000


def _coverage_metadata(
    concept_dict: Dict[str, Any],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Classify catalog coverage without inventing database support counts."""
    cov = dict(cc.CONCEPT_DB_COVERAGE)
    supported_count = len(cc.SUPPORTED_DB_KEYS)
    derived_rule_prefixes = ("sofa_", "sofa2_", "sep3_")
    derived_rule_concepts = {
        "susp_inf",
        "culture_positive",
        "bld_culture_positive",
    }
    metadata: Dict[str, Dict[str, Any]] = {}
    summary = {
        "supportedDatabases": supported_count,
        "audited": 0,
        "auditedAll": 0,
        "auditedFive": 0,
        "auditedPartial": 0,
        "derived": 0,
        "notAudited": 0,
    }

    for key in concept_dict:
        if key in cov:
            databases = int(cov[key])
            metadata[key] = {
                "kind": "audited",
                "databases": databases,
                "basis": "CONCEPT_DB_COVERAGE",
            }
            summary["audited"] += 1
            if databases == supported_count:
                summary["auditedAll"] += 1
            elif databases == supported_count - 1:
                summary["auditedFive"] += 1
            else:
                summary["auditedPartial"] += 1
            continue

        if key in cc.COMPOSITE_CONCEPT_OUTPUT_SOURCES:
            metadata[key] = {
                "kind": "derived",
                "databases": None,
                "basis": "COMPOSITE_CONCEPT_OUTPUT_SOURCES",
                "source": cc.COMPOSITE_CONCEPT_OUTPUT_SOURCES[key],
            }
            summary["derived"] += 1
            continue

        if key.startswith(derived_rule_prefixes) or key in derived_rule_concepts:
            metadata[key] = {
                "kind": "derived",
                "databases": None,
                "basis": "score_or_rule_component",
                "source": "rule_based_output",
            }
            summary["derived"] += 1
            continue

        metadata[key] = {
            "kind": "not_audited",
            "databases": None,
            "basis": "missing_from_CONCEPT_DB_COVERAGE",
        }
        summary["notAudited"] += 1

    return metadata, summary


def _active_export_coverage(concept_dict: Dict[str, Any]) -> Dict[str, Any]:
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
    if total_rows > _ACTIVE_COVERAGE_FULL_READ_ROW_LIMIT:
        return _active_export_schema_coverage(desc, concept_dict)

    path = Path(str(active_path)).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
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
    from easyicu.webserver import dataio

    norm_stays = frame["stay_id"].map(dataio._norm_id).astype(str)
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
    concept_coverage, coverage_summary = _coverage_metadata(concept_dict)
    active_export_coverage = _active_export_coverage(concept_dict)

    return {
        "groups": groups,
        "groupConcepts": group_concepts,
        "dict": concept_dict,
        "desc": desc,
        "cov": cov,
        "conceptCoverage": concept_coverage,
        "coverageSummary": coverage_summary,
        "activeExportCoverage": active_export_coverage,
        "supportedDbs": list(cc.SUPPORTED_DB_KEYS),
        "totalConcepts": len(concept_dict),
    }
