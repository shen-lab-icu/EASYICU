"""Bounded Patient Review drilldown payloads for the native FastAPI UI.

This module is intentionally separate from ``workspace/summary``. The older
summary endpoint returns preview row structures used by early migration stages;
Patient Review now needs a real, fail-closed path that exposes only aggregate
metadata plus one pseudonymous entity drilldown.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store

_MAX_ENTITIES = 5
_MAX_SIGNAL_POINTS = 12
_READ_MODULES = ("demographics", "outcome", "sofa2_score", "sepsis3_sofa2", "vitals")
_SIGNAL_SPECS = (
    ("hr", "Heart rate", "bpm"),
    ("map", "MAP", "mmHg"),
    ("spo2", "SpO2", "%"),
    ("temp", "Temp", "deg C"),
)
_MODULE_COLUMNS = {
    "demographics": ("stay_id", "age", "sex"),
    "outcome": ("stay_id", "death", "los_icu"),
    "sofa2_score": ("stay_id", "charttime", "sofa2"),
    "sepsis3_sofa2": ("stay_id", "sep3_sofa2"),
    "vitals": ("stay_id", "charttime", "hr", "map", "spo2", "temp"),
}


def patient_review_drilldown(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return a real, bounded Patient Review payload for one registered export."""
    source, desc = _resolve_registered_source(body)
    path = Path(str(desc.get("path") or source.get("path") or "")).expanduser()
    frames = {
        module: _read_module_frame(path, desc, module)
        for module in _READ_MODULES
    }

    demo = frames.get("demographics")
    if demo is None or getattr(demo, "empty", True):
        fallback = _fallback_entity_frame(path, desc)
        if fallback is None or getattr(fallback, "empty", True):
            raise PatientReviewError({"error": "no_entity_denominator"})
        demo = fallback

    demo = demo.copy()
    demo["stay_id"] = demo["stay_id"].map(dataio._norm_id)
    demo = demo[demo["stay_id"].astype(bool)].drop_duplicates("stay_id")
    if demo.empty:
        raise PatientReviewError({"error": "no_entity_denominator"})

    entity_ids = [str(value) for value in demo["stay_id"].tolist()]
    entity_set = set(entity_ids)
    outcome = dataio._filter_by_stay(frames.get("outcome"), entity_set)
    sofa2 = dataio._filter_by_stay(frames.get("sofa2_score"), entity_set)
    sepsis = dataio._filter_by_stay(frames.get("sepsis3_sofa2"), entity_set)
    vitals = dataio._filter_by_stay(frames.get("vitals"), entity_set)

    death_by_entity = dataio._stay_bool(outcome, "death", missing_false=True)
    los_by_entity = dataio._stay_numeric(outcome, "los_icu", "median")
    sofa_by_entity = dataio._stay_numeric(sofa2, "sofa2", "max")
    sepsis_by_entity = dataio._stay_bool(sepsis, "sep3_sofa2", missing_false=True)
    for entity_id in entity_ids:
        if outcome is not None and not outcome.empty:
            death_by_entity.setdefault(entity_id, False)
        if sepsis is not None and not sepsis.empty:
            sepsis_by_entity.setdefault(entity_id, False)

    requested_ref = str(body.get("entity_ref") or body.get("selected_ref") or "")
    ref_to_id = {_entity_ref(path, entity_id): entity_id for entity_id in entity_ids}
    selected_id = ref_to_id.get(requested_ref) if requested_ref else None
    if selected_id is None:
        selected_id = entity_ids[0]

    demo_by_id = demo.set_index("stay_id", drop=False)
    selected_row = demo_by_id.loc[selected_id] if selected_id in demo_by_id.index else {}
    entities = [
        _entity_option(path, entity_id, ordinal, death_by_entity, sofa_by_entity)
        for ordinal, entity_id in enumerate(entity_ids[:_MAX_ENTITIES], start=1)
    ]
    selected = _selected_payload(
        path=path,
        entity_id=selected_id,
        ordinal=entity_ids.index(selected_id) + 1,
        row=selected_row,
        death_by_entity=death_by_entity,
        los_by_entity=los_by_entity,
        sofa_by_entity=sofa_by_entity,
        sepsis_by_entity=sepsis_by_entity,
        vitals=vitals,
    )

    summary = {
        "entities": int((desc.get("summary") or {}).get("stays") or len(entity_ids)),
        "modules": int((desc.get("summary") or {}).get("modules") or 0),
        "file_count": int((desc.get("summary") or {}).get("file_count") or 0),
        "total_rows": int((desc.get("summary") or {}).get("total_rows") or 0),
        "mean_age": dataio._series_mean(demo.get("age")) if "age" in demo.columns else None,
        "female_pct": dataio._sex_pct(demo.get("sex"), "female") if "sex" in demo.columns else None,
        "mortality": dataio._bool_pct(list(death_by_entity.values())),
        "median_los_icu": dataio._median(list(los_by_entity.values())),
        "median_sofa2": dataio._median(list(sofa_by_entity.values())),
        "sepsis_pct": dataio._bool_pct(list(sepsis_by_entity.values())),
    }

    return {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source": _source_provenance(source, desc),
        "provenance": {
            "computed_from": [
                "source_registry",
                "export_manifest",
                "bounded_column_reads",
                "pseudonymous_entity_reference",
            ],
            "payload_scope": "aggregate_plus_one_entity",
            "signals": "selected_entity_only_capped",
        },
        "privacy": {
            "raw_rows_returned": False,
            "direct_identifiers_returned": False,
            "max_entity_options": _MAX_ENTITIES,
            "max_points_per_signal": _MAX_SIGNAL_POINTS,
        },
        "summary": summary,
        "entities": entities,
        "selected": selected,
        "quality": _quality_payload(path, desc),
        "blocked_features": [
            {
                "id": "raw_identifier_table",
                "status": "blocked",
                "reason": "Patient Review returns aggregates and one pseudonymous entity only.",
            },
            {
                "id": "arbitrary_entity_search",
                "status": "blocked",
                "reason": "Search by direct clinical identifiers is not exposed in the native API.",
            },
            {
                "id": "full_timeline_export",
                "status": "blocked",
                "reason": "Signal values are capped for browser review; formal export stays in the evidence-bound agent path.",
            },
        ],
    }


def _resolve_registered_source(body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    registry = source_store.load_registry()
    sources = [s for s in registry.get("sources") or [] if isinstance(s, dict)]
    requested = body.get("source_path") or body.get("path")
    if requested:
        norm = _norm_path(str(requested))
        source = next((s for s in sources if _norm_path(str(s.get("path") or "")) == norm), None)
        if source is None:
            raise PatientReviewError({"error": "source_not_registered", "path_hash": _hash(norm)})
    else:
        active = registry.get("active_path")
        if not active:
            raise PatientReviewError({"error": "no_active_export"})
        active_norm = _norm_path(str(active))
        source = next((s for s in sources if _norm_path(str(s.get("path") or "")) == active_norm), None)
        if source is None:
            raise PatientReviewError({"error": "active_source_not_registered", "path_hash": _hash(active_norm)})

    desc = dataio.describe_export_source(str(source.get("path") or ""))
    if not desc.get("ok"):
        raise PatientReviewError({"error": "invalid_export", "detail": desc.get("error")})
    return source, desc


def _read_module_frame(path: Path, desc: Dict[str, Any], module: str) -> Any:
    file_meta = next((f for f in desc.get("files") or [] if f.get("module") == module), None)
    if not file_meta:
        return None
    file_name = str(file_meta.get("file") or "")
    columns = [c for c in _MODULE_COLUMNS[module] if c in (file_meta.get("columns") or [])]
    if "stay_id" not in columns:
        return None
    return _read_selected_columns(path / file_name, columns)


def _fallback_entity_frame(path: Path, desc: Dict[str, Any]) -> Any:
    file_meta = next((f for f in desc.get("files") or [] if "stay_id" in (f.get("columns") or [])), None)
    if not file_meta:
        return None
    return _read_selected_columns(path / str(file_meta.get("file") or ""), ["stay_id"])


def _read_selected_columns(path: Path, columns: List[str]) -> Any:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix == ".xlsx":
        return pd.read_excel(path, usecols=columns)
    return pd.read_csv(path, usecols=columns)


def _entity_option(
    path: Path,
    entity_id: str,
    ordinal: int,
    death_by_entity: Dict[str, bool],
    sofa_by_entity: Dict[str, float],
) -> Dict[str, Any]:
    dead = death_by_entity.get(entity_id)
    sofa = dataio._num(sofa_by_entity.get(entity_id))
    return {
        "ref": _entity_ref(path, entity_id),
        "label": f"Entity {ordinal}",
        "ordinal": ordinal,
        "outcome": "Deceased" if dead is True else ("Survived" if dead is False else "Unknown"),
        "severity": None if sofa is None else f"SOFA-2 {sofa:g}",
    }


def _selected_payload(
    path: Path,
    entity_id: str,
    ordinal: int,
    row: Any,
    death_by_entity: Dict[str, bool],
    los_by_entity: Dict[str, float],
    sofa_by_entity: Dict[str, float],
    sepsis_by_entity: Dict[str, bool],
    vitals: Any,
) -> Dict[str, Any]:
    dead = death_by_entity.get(entity_id)
    sepsis = sepsis_by_entity.get(entity_id)
    return {
        "ref": _entity_ref(path, entity_id),
        "label": f"Entity {ordinal}",
        "ordinal": ordinal,
        "demographics": {
            "age": dataio._num(_row_value(row, "age")),
            "sex": dataio._clean(_row_value(row, "sex")),
        },
        "scores": {
            "sofa2_max": dataio._num(sofa_by_entity.get(entity_id)),
            "sepsis3_sofa2": bool(sepsis) if sepsis is not None else None,
        },
        "outcomes": {
            "status": "Deceased" if dead is True else ("Survived" if dead is False else "Unknown"),
            "icu_los_days": dataio._num(los_by_entity.get(entity_id)),
        },
        "signals": _signals_payload(vitals, entity_id),
    }


def _row_value(row: Any, key: str) -> Any:
    try:
        return row.get(key)
    except AttributeError:
        return None


def _signals_payload(vitals: Any, entity_id: str) -> List[Dict[str, Any]]:
    if vitals is None or vitals.empty or "stay_id" not in vitals.columns:
        return []
    one = vitals[vitals["stay_id"].map(dataio._norm_id) == str(entity_id)].copy()
    if one.empty:
        return []
    if "charttime" in one.columns:
        one = one.sort_values("charttime")
    out: List[Dict[str, Any]] = []
    for key, name, unit in _SIGNAL_SPECS:
        if key not in one.columns:
            continue
        values = dataio._numeric_values(one[key])
        bounded = values[:_MAX_SIGNAL_POINTS]
        if bounded:
            out.append({
                "key": key,
                "name": name,
                "unit": unit,
                "current": bounded[-1],
                "values": bounded,
                "point_count": len(values),
                "bounded": True,
                "max_points": _MAX_SIGNAL_POINTS,
            })
    return out


def _quality_payload(path: Path, desc: Dict[str, Any]) -> List[Dict[str, Any]]:
    cohort_size = (desc.get("summary") or {}).get("stays")
    out: List[Dict[str, Any]] = []
    for item in desc.get("files") or []:
        module = str(item.get("module") or "")
        if not module:
            continue
        columns = item.get("columns") or []
        rows = int(item.get("rows") or 0)
        covered = _bounded_covered_entities(path, item, cohort_size)
        coverage = round(covered / cohort_size * 100, 1) if isinstance(cohort_size, int) and cohort_size else None
        out.append({
            "module": module,
            "rows": rows,
            "column_count": len(columns),
            "covered_entities": covered,
            "coverage_pct": coverage,
            "quality_status": _quality_status(module, coverage),
        })
    return out


def _bounded_covered_entities(path: Path, item: Dict[str, Any], cohort_size: Any) -> int | None:
    if not isinstance(cohort_size, int) or cohort_size <= 0:
        return None
    file_name = str(item.get("file") or "")
    if not file_name:
        return None
    ids = dataio._read_stay_ids(path / file_name)
    if ids is None:
        return None
    return min(len(ids), cohort_size)


def _quality_status(module: str, coverage_pct: float | None) -> str:
    if coverage_pct is None:
        return "unknown"
    if module in dataio._EVENT_PRESENCE_MODULES:
        return "neutral"
    if coverage_pct >= 80:
        return "ok"
    if coverage_pct >= 50:
        return "warn"
    return "bad"


def _source_provenance(source: Dict[str, Any], desc: Dict[str, Any]) -> Dict[str, Any]:
    path = str(desc.get("path") or source.get("path") or "")
    return {
        "id": source.get("id"),
        "label": source.get("label") or Path(path).name or "local",
        "path_hash": _hash(path),
        "database": desc.get("database") or source.get("database"),
        "generated": desc.get("generated") or source.get("generated"),
    }


def _entity_ref(path: Path, entity_id: str) -> str:
    token = f"{path.resolve()}::{entity_id}"
    return "ent_" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def _norm_path(raw: str) -> str:
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    return str(path)


def _hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


class PatientReviewError(Exception):
    def __init__(self, detail: Dict[str, Any]):
        super().__init__(str(detail.get("error") or "patient_review_error"))
        self.detail = detail
