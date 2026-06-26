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

from easyicu.concept import catalog as concept_catalog
from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store

_MAX_ENTITIES = 5
_MAX_REVIEW_ENTITIES = 500
_MAX_SIGNAL_POINTS = 12
_MAX_REVIEW_SIGNALS = 24
_MAX_TABLE_PREVIEW_ROWS = 24
_MAX_TABLE_PREVIEW_COLUMNS = 14
_MAX_TABLE_PREVIEW_MODULES = 32
_READ_MODULES = ("demographics", "outcome", "sofa2_score", "sepsis3_sofa2", "vitals")
_SIGNAL_SPECS = (
    ("hr", "Heart rate", "bpm"),
    ("map", "MAP", "mmHg"),
    ("spo2", "SpO2", "%"),
    ("temp", "Temp", "deg C"),
)
_TIME_COLUMNS = (
    "charttime",
    "time",
    "datetime",
    "timestamp",
    "starttime",
    "endtime",
    "hour",
    "measuredat_minutes",
    "observationoffset",
)
_ID_COLUMNS = {"stay_id", "subject_id", "hadm_id"}
_DIRECT_IDENTIFIER_COLUMN_KEYS = {
    "stayid",
    "icustayid",
    "patientunitstayid",
    "patienthealthsystemstayid",
    "subjectid",
    "hadmid",
    "patientid",
    "admissionid",
    "caseid",
    "uniquepid",
}
_METADATA_COLUMNS = {
    "valueuom",
    "unit",
    "units",
    "category",
    "type",
    "dur_var",
    "entertime",
    "intakeoutputentryoffset",
}
_MODULE_COLUMNS = {
    "demographics": ("stay_id", "age", "sex"),
    "outcome": ("stay_id", "death", "los_icu"),
    "sofa2_score": ("stay_id", "charttime", "sofa2"),
    "sepsis3_sofa2": ("stay_id", "sep3_sofa2"),
    "vitals": ("stay_id", "charttime", "hr", "map", "spo2", "temp"),
}


def patient_review_sources(body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return registered local exports that can back Patient Review.

    This is a source-selection contract, not a data preview: it reads registry
    and export metadata only, and never returns patient rows or row identifiers.
    """
    del body
    registry = source_store.load_registry()
    active_path = str(registry.get("active_path") or "")
    active_norm = _norm_path(active_path) if active_path else ""
    sources: List[Dict[str, Any]] = []

    for source in registry.get("sources") or []:
        if not isinstance(source, dict):
            continue
        raw_path = str(source.get("path") or "")
        if not raw_path or not source.get("ok", True):
            continue
        desc = dataio.describe_export_source(raw_path)
        if not desc.get("ok"):
            continue
        summary = desc.get("summary") or source.get("summary") or {}
        source_norm = _norm_path(raw_path)
        public = _source_provenance(source, desc)
        public.update({
            "active": source_norm == active_norm,
            "path": raw_path,
            "path_hash": _hash(source_norm),
            "patient_ready": _is_patient_ready(desc),
            "summary": {
                "entities": _int_or_none(summary.get("stays")),
                "modules": _int_or_none(summary.get("modules")),
                "file_count": _int_or_none(summary.get("file_count")),
                "total_rows": _int_or_none(summary.get("total_rows")),
            },
        })
        sources.append(public)

    active_source = next((item for item in sources if item.get("active")), None)
    return {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source_count": len(sources),
        "active_path_hash": _hash(active_norm) if active_norm else None,
        "active_source": active_source,
        "can_load": bool(active_source and active_source.get("patient_ready")),
        "sources": sources,
        "provenance": {
            "computed_from": [
                "source_registry",
                "autodiscovered_export_folders",
                "export_manifest_or_file_schema",
            ],
            "payload_scope": "local_export_source_metadata_only",
        },
        "privacy": {
            "raw_rows_returned": False,
            "direct_identifiers_returned": False,
            "patient_rows_returned": False,
        },
        "blocked_features": [
            {
                "id": "unregistered_path_review",
                "status": "blocked",
                "reason": "Patient Review only loads registered local exports.",
            },
            {
                "id": "row_preview_in_source_picker",
                "status": "blocked",
                "reason": "The source picker shows metadata only; drilldown remains bounded and pseudonymous.",
            },
        ],
    }


def patient_review_drilldown(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return a real, bounded Patient Review payload for one registered export."""
    source, desc = _resolve_registered_source(body)
    path = Path(str(desc.get("path") or source.get("path") or "")).expanduser()
    demo = _read_module_frame(path, desc, "demographics")
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
    review_entity_ids = entity_ids[:_MAX_REVIEW_ENTITIES]
    entity_set = set(review_entity_ids)
    frames = {
        "demographics": demo,
        "outcome": _read_module_frame(path, desc, "outcome"),
        "sepsis3_sofa2": _read_module_frame(path, desc, "sepsis3_sofa2"),
        "sofa2_score": _read_module_frame(path, desc, "sofa2_score", entity_set),
        "vitals": _read_module_frame(path, desc, "vitals", entity_set),
    }
    outcome_all = frames.get("outcome")
    sepsis_all = frames.get("sepsis3_sofa2")
    outcome = dataio._filter_by_stay(outcome_all, entity_set)
    sofa2 = dataio._filter_by_stay(frames.get("sofa2_score"), entity_set)
    sepsis = dataio._filter_by_stay(sepsis_all, entity_set)
    vitals = dataio._filter_by_stay(frames.get("vitals"), entity_set)
    review_frames = _read_review_frames(path, desc, entity_set)

    death_by_entity = dataio._stay_bool(outcome, "death", missing_false=True)
    los_by_entity = dataio._stay_numeric(outcome, "los_icu", "median")
    sofa_by_entity = dataio._stay_numeric(sofa2, "sofa2", "max")
    sepsis_by_entity = dataio._stay_bool(sepsis, "sep3_sofa2", missing_false=True)
    for entity_id in review_entity_ids:
        if outcome is not None and not outcome.empty:
            death_by_entity.setdefault(entity_id, False)
        if sepsis is not None and not sepsis.empty:
            sepsis_by_entity.setdefault(entity_id, False)
    death_by_entity_all = dataio._stay_bool(outcome_all, "death", missing_false=True)
    los_by_entity_all = dataio._stay_numeric(outcome_all, "los_icu", "median")
    sepsis_by_entity_all = dataio._stay_bool(sepsis_all, "sep3_sofa2", missing_false=True)
    for entity_id in entity_ids:
        if outcome_all is not None and not outcome_all.empty:
            death_by_entity_all.setdefault(entity_id, False)
        if sepsis_all is not None and not sepsis_all.empty:
            sepsis_by_entity_all.setdefault(entity_id, False)

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
        "review_entities": len(review_entity_ids),
        "review_entity_cap": _MAX_REVIEW_ENTITIES,
        "review_scope": "browser_bounded_entity_sample" if len(review_entity_ids) < len(entity_ids) else "full_entity_set",
        "static_aggregate_scope": "full_entity_set",
        "dynamic_aggregate_scope": "browser_bounded_entity_sample" if len(review_entity_ids) < len(entity_ids) else "full_entity_set",
        "mean_age": dataio._series_mean(demo.get("age")) if "age" in demo.columns else None,
        "female_pct": dataio._sex_pct(demo.get("sex"), "female") if "sex" in demo.columns else None,
        "mortality": dataio._bool_pct(list((death_by_entity_all or death_by_entity).values())),
        "median_los_icu": dataio._median(list((los_by_entity_all or los_by_entity).values())),
        "median_sofa2": dataio._median(list(sofa_by_entity.values())),
        "sepsis_pct": dataio._bool_pct(list((sepsis_by_entity_all or sepsis_by_entity).values())),
    }
    module_profiles = _module_profiles(desc, review_frames, entity_set)
    time_lanes = _time_lane_payloads(review_frames, selected_id)
    quality_metrics = _quality_metrics_payload(review_frames, entity_set)
    quality = _quality_from_module_profiles(module_profiles)
    data_tables = _data_table_review_payload(path, desc, module_profiles, summary)
    trajectory_review = _trajectory_review_payload(time_lanes, selected, entities, quality_metrics)
    patient_overview = _patient_overview_payload(selected, entities, time_lanes, quality_metrics)
    quality_review = _quality_review_payload(quality, quality_metrics)

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
            "max_table_preview_rows": _MAX_TABLE_PREVIEW_ROWS,
            "max_table_preview_columns": _MAX_TABLE_PREVIEW_COLUMNS,
            "bounded_table_previews": True,
            "payload_tables_are_aggregated": False,
            "payload_tables_are_bounded": True,
        },
        "summary": summary,
        "module_profiles": module_profiles,
        "entities": entities,
        "selected": selected,
        "time_lanes": time_lanes,
        "quality": quality,
        "quality_metrics": quality_metrics,
        "data_tables": data_tables,
        "trajectory_review": trajectory_review,
        "patient_overview": patient_overview,
        "quality_review": quality_review,
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


def _read_module_frame(path: Path, desc: Dict[str, Any], module: str, stay_ids: set[str] | None = None) -> Any:
    file_meta = next((f for f in desc.get("files") or [] if f.get("module") == module), None)
    if not file_meta:
        return None
    file_name = str(file_meta.get("file") or "")
    columns = [c for c in _MODULE_COLUMNS[module] if c in (file_meta.get("columns") or [])]
    if "stay_id" not in columns:
        return None
    return _read_selected_columns(path / file_name, columns, stay_ids=stay_ids)


def _fallback_entity_frame(path: Path, desc: Dict[str, Any]) -> Any:
    file_meta = next((f for f in desc.get("files") or [] if "stay_id" in (f.get("columns") or [])), None)
    if not file_meta:
        return None
    return _read_selected_columns(path / str(file_meta.get("file") or ""), ["stay_id"])


def _read_selected_columns(path: Path, columns: List[str], stay_ids: set[str] | None = None) -> Any:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        filters = _stay_id_filters(path, stay_ids) if stay_ids and "stay_id" in columns else None
        if filters:
            return pd.read_parquet(path, columns=columns, filters=filters)
        return pd.read_parquet(path, columns=columns)
    if suffix == ".xlsx":
        frame = pd.read_excel(path, usecols=columns)
    else:
        frame = pd.read_csv(path, usecols=columns)
    if stay_ids and "stay_id" in frame.columns:
        frame = frame.copy()
        frame["stay_id"] = frame["stay_id"].map(dataio._norm_id)
        frame = frame[frame["stay_id"].isin(stay_ids)]
    return frame


def _stay_id_filters(path: Path, stay_ids: set[str]) -> List[Tuple[str, str, List[Any]]] | None:
    values: List[Any]
    try:
        import pyarrow.parquet as pq
        import pyarrow.types as pat

        field = pq.ParquetFile(path).schema_arrow.field("stay_id")
        if pat.is_integer(field.type):
            values = [int(value) for value in stay_ids if str(value).isdigit()]
        elif pat.is_floating(field.type):
            values = [float(value) for value in stay_ids if _is_number_like(value)]
        else:
            values = [str(value) for value in stay_ids if str(value)]
    except Exception:
        values = [str(value) for value in stay_ids if str(value)]
    if not values:
        return None
    return [("stay_id", "in", values)]


def _is_number_like(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _read_review_frames(path: Path, desc: Dict[str, Any], entity_set: set[str]) -> List[Dict[str, Any]]:
    """Read bounded columns needed for Patient Review charts.

    The old Streamlit Patient Review computed module previews, clinical-lane
    trends, and data-quality profiles from loaded concept frames. The native API
    mirrors those calculations but only reads selected non-identifier columns
    and returns aggregates plus the chosen pseudonymous entity signals.
    """
    best_by_module: Dict[str, Dict[str, Any]] = {}
    for item in desc.get("files") or []:
        module = str(item.get("module") or "")
        columns = [str(col) for col in (item.get("columns") or [])]
        if not module or module not in _READ_MODULES or "stay_id" not in columns:
            continue
        feature_cols = _feature_columns(columns)
        time_cols = [col for col in _TIME_COLUMNS if col in columns]
        selected_columns = _ordered_unique(["stay_id", *time_cols, *feature_cols])
        if len(selected_columns) <= 1:
            continue
        try:
            frame = _read_selected_columns(path / str(item.get("file") or ""), selected_columns, stay_ids=entity_set)
        except Exception:
            continue
        if frame is None or getattr(frame, "empty", True) or "stay_id" not in frame.columns:
            continue
        frame = frame.copy()
        frame["stay_id"] = frame["stay_id"].map(dataio._norm_id)
        frame = frame[frame["stay_id"].isin(entity_set)]
        if frame.empty:
            continue
        candidate = {
            "module": module,
            "file": item.get("file"),
            "rows": int(item.get("rows") or len(frame)),
            "columns": columns,
            "features": [col for col in feature_cols if col in frame.columns],
            "time_col": _detect_time_col(frame),
            "frame": frame,
            "entity_overlap": int(frame["stay_id"].nunique()),
        }
        current = best_by_module.get(module)
        if current is None or candidate["entity_overlap"] > int(current.get("entity_overlap") or 0):
            best_by_module[module] = candidate
    return list(best_by_module.values())


def _feature_columns(columns: List[str]) -> List[str]:
    concepts = set(concept_catalog.CONCEPT_DICTIONARY)
    out: List[str] = []
    for col in columns:
        if col in concepts and col not in _ID_COLUMNS and col not in _METADATA_COLUMNS:
            out.append(col)
    return out


def _ordered_unique(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _detect_time_col(frame: Any) -> str | None:
    for col in _TIME_COLUMNS:
        if col in getattr(frame, "columns", []):
            return col
    return None


def _module_profiles(
    desc: Dict[str, Any],
    review_frames: List[Dict[str, Any]],
    entity_set: set[str],
) -> List[Dict[str, Any]]:
    frame_by_module = {row["module"]: row for row in review_frames}
    order: List[str] = []
    item_by_module: Dict[str, Dict[str, Any]] = {}
    for item in desc.get("files") or []:
        module = str(item.get("module") or "")
        if not module:
            continue
        if module not in item_by_module:
            order.append(module)
            item_by_module[module] = item
        frame_item = frame_by_module.get(module)
        if frame_item and item.get("file") == frame_item.get("file"):
            item_by_module[module] = item

    profiles: List[Dict[str, Any]] = []
    for module in order:
        item = item_by_module[module]
        frame_item = frame_by_module.get(module)
        columns = [str(col) for col in ((frame_item or item).get("columns") or [])]
        features = [str(col) for col in ((frame_item or {}).get("features") or _feature_columns(columns))]
        frame = frame_item.get("frame") if frame_item else None
        time_col = frame_item.get("time_col") if frame_item else None
        rows = int((frame_item or item).get("rows") or 0)
        entities = None
        dynamic_features = 0
        static_features = 0
        observed_features = 0
        if frame is not None and not frame.empty:
            entities = int(frame["stay_id"].nunique()) if "stay_id" in frame.columns else None
            for feature in [f for f in features if f in frame.columns]:
                if frame[feature].notna().any():
                    observed_features += 1
                    if time_col:
                        dynamic_features += 1
                    else:
                        static_features += 1
        coverage = round(entities / len(entity_set) * 100, 1) if entities is not None and entity_set else None
        profiles.append({
            "module": module,
            "label": _module_label(module),
            "rows": rows,
            "feature_count": len(features),
            "observed_features": observed_features,
            "entities": entities,
            "coverage_pct": coverage,
            "time_indexed": bool(time_col),
            "dynamic_features": dynamic_features,
            "static_features": static_features,
            "preview_features": features[:6],
        })
    return profiles


def _module_label(module: str) -> str:
    label = concept_catalog.CONCEPT_GROUP_NAMES.get(module, (module, module))[0]
    return _plain_label(label)


def _plain_label(label: str) -> str:
    text = str(label or "").strip()
    while text and not (text[0].isalnum() or "\u4e00" <= text[0] <= "\u9fff"):
        text = text[1:].lstrip()
    return text or str(label or "")


def _time_lane_payloads(review_frames: List[Dict[str, Any]], entity_id: str) -> List[Dict[str, Any]]:
    by_feature: Dict[str, Dict[str, Any]] = {}
    for item in review_frames:
        frame = item.get("frame")
        if frame is None or frame.empty or "stay_id" not in frame.columns:
            continue
        one = frame[frame["stay_id"] == str(entity_id)].copy()
        if one.empty:
            continue
        time_col = item.get("time_col")
        if not time_col or time_col not in one.columns:
            continue
        if time_col and time_col in one.columns:
            one = one.sort_values(time_col)
        for feature in item.get("features") or []:
            if feature not in one.columns:
                continue
            values = dataio._numeric_values(one[feature])
            if not values:
                continue
            by_feature.setdefault(feature, {
                "feature": feature,
                "name": _concept_name(feature),
                "unit": _concept_unit(feature),
                "module": item.get("module"),
                "time_indexed": bool(time_col),
                "values": values[:_MAX_SIGNAL_POINTS],
                "point_count": len(values),
                "current": values[min(len(values), _MAX_SIGNAL_POINTS) - 1],
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "mean": round(sum(values) / len(values), 3),
                "thresholds": _threshold_payload(feature),
            })
            if len(by_feature) >= _MAX_REVIEW_SIGNALS:
                break
        if len(by_feature) >= _MAX_REVIEW_SIGNALS:
            break

    lanes: List[Dict[str, Any]] = []
    used: set[str] = set()
    for lane, features in concept_catalog.CLINICAL_LANES.items():
        lane_signals = [by_feature[f] for f in features if f in by_feature]
        used.update(row["feature"] for row in lane_signals)
        lanes.append({
            "lane": lane,
            "label": lane.replace("_", " ").title(),
            "signal_count": len(lane_signals),
            "signals": lane_signals,
            "status": "ready" if lane_signals else "unavailable",
        })
    other = [row for key, row in by_feature.items() if key not in used]
    if other:
        lanes.append({
            "lane": "other",
            "label": "Other signals",
            "signal_count": len(other),
            "signals": other,
            "status": "ready",
        })
    return lanes


def _concept_name(feature: str) -> str:
    entry = concept_catalog.CONCEPT_DICTIONARY.get(feature)
    if entry:
        return str(entry[0])
    return feature


def _concept_unit(feature: str) -> str:
    threshold_unit = concept_catalog.CLINICAL_THRESHOLDS.get(feature, {}).get("unit")
    if threshold_unit:
        return str(threshold_unit)
    entry = concept_catalog.CONCEPT_DICTIONARY.get(feature)
    if entry and len(entry) > 2:
        return str(entry[2] or "")
    return ""


def _threshold_payload(feature: str) -> List[Dict[str, Any]]:
    thresholds = concept_catalog.CLINICAL_THRESHOLDS.get(feature) or {}
    lines = thresholds.get("lines") or []
    labels = thresholds.get("labels") or []
    out = []
    for idx, value in enumerate(lines):
        out.append({
            "value": dataio._num(value),
            "label": labels[idx] if idx < len(labels) else "clinical threshold",
        })
    return out


def _quality_metrics_payload(review_frames: List[Dict[str, Any]], entity_set: set[str]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    total_records = 0
    missing_weight = 0.0
    outlier_weight = 0.0
    duplicate_weight = 0.0
    denominator = len(entity_set)

    for item in review_frames:
        frame = item.get("frame")
        if frame is None or frame.empty or "stay_id" not in frame.columns:
            continue
        module = str(item.get("module") or "")
        time_col = item.get("time_col")
        for feature in item.get("features") or []:
            if feature not in frame.columns:
                continue
            observed = frame[["stay_id", feature] + ([time_col] if time_col and time_col in frame.columns else [])].copy()
            non_null = observed[observed[feature].notna()]
            records = int(len(non_null))
            entities = int(non_null["stay_id"].nunique()) if records else 0
            missing_pct = round((1 - entities / denominator) * 100, 1) if denominator else None
            outlier_pct = _out_of_physio_pct(feature, non_null[feature])
            duplicate_pct = _duplicate_time_pct(non_null, time_col)
            density = round(records / max(denominator, 1), 3) if denominator else None
            total_records += records
            if missing_pct is not None:
                missing_weight += missing_pct * max(records, 1)
            outlier_weight += outlier_pct * max(records, 1)
            duplicate_weight += duplicate_pct * max(records, 1)
            rows.append({
                "feature": feature,
                "name": _concept_name(feature),
                "module": module,
                "records": records,
                "entities": entities,
                "coverage_pct": round(entities / denominator * 100, 1) if denominator else None,
                "missing_pct": missing_pct,
                "out_of_physio_pct": outlier_pct,
                "duplicate_time_pct": duplicate_pct,
                "density_per_entity": density,
                "time_indexed": bool(time_col),
                "status": _quality_feature_status(missing_pct, outlier_pct, duplicate_pct),
            })

    weight_denominator = sum(max(int(row.get("records") or 0), 1) for row in rows)
    summary = {
        "concept_count": len(rows),
        "total_records": total_records,
        "weighted_missing_pct": round(missing_weight / weight_denominator, 1) if weight_denominator else None,
        "weighted_out_of_physio_pct": round(outlier_weight / weight_denominator, 1) if weight_denominator else None,
        "weighted_duplicate_time_pct": round(duplicate_weight / weight_denominator, 1) if weight_denominator else None,
        "denominator_entities": denominator,
    }
    top_issues = sorted(
        rows,
        key=lambda row: (
            float(row.get("missing_pct") or 0),
            float(row.get("out_of_physio_pct") or 0),
            float(row.get("duplicate_time_pct") or 0),
            int(row.get("records") or 0),
        ),
        reverse=True,
    )[:5]
    return {
        "summary": summary,
        "features": rows[:80],
        "top_issues": top_issues,
        "payload_scope": "aggregate_quality_metrics_no_row_payload",
    }


def _data_table_review_payload(
    path: Path,
    desc: Dict[str, Any],
    module_profiles: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Mirror the old Data Tables review contract with bounded local previews."""
    module_count = len([row for row in module_profiles if int(row.get("feature_count") or 0) > 0])
    feature_count = sum(int(row.get("feature_count") or 0) for row in module_profiles)
    selected_count = sum(int(row.get("observed_features") or 0) for row in module_profiles)
    modules = []
    for row in module_profiles:
        feature_count_row = int(row.get("feature_count") or 0)
        if feature_count_row <= 0:
            continue
        coverage = row.get("coverage_pct")
        modules.append({
            "module": row.get("module"),
            "label": row.get("label"),
            "review_features": feature_count_row,
            "observed_features": int(row.get("observed_features") or 0),
            "rows": int(row.get("rows") or 0),
            "entities": row.get("entities"),
            "coverage_pct": coverage,
            "share_pct": round(feature_count_row / feature_count * 100, 1) if feature_count else None,
            "shape": "time_indexed" if row.get("time_indexed") else "static",
            "dynamic_features": int(row.get("dynamic_features") or 0),
            "static_features": int(row.get("static_features") or 0),
            "preview_features": [
                {
                    "feature": feature,
                    "name": _concept_name(str(feature)),
                    "unit": _concept_unit(str(feature)),
                    "group": _concept_group_label(str(feature)),
                }
                for feature in (row.get("preview_features") or [])[:6]
            ],
            "status": _module_review_status(coverage, feature_count_row),
        })
    return {
        "loaded_summary": {
            "entities": summary.get("entities"),
            "review_features": feature_count,
            "observed_features": selected_count,
            "module_count": module_count,
            "source_count": 1,
        },
        "module_picker": {
            "default_module": modules[0]["module"] if modules else None,
            "module_count": len(modules),
            "selection_mode": "module_then_feature",
        },
        "detail_gate": {
            "title": "Bounded local table previews",
            "default_open": False,
            "reason": "The browser renders capped module table previews with pseudonymous entity tokens; direct identifiers and full tables stay on disk.",
            "available_detail_modes": ["module_table_preview", "module_glance", "single_feature_metadata"],
        },
        "modules": modules,
        "table_previews": _table_preview_payloads(path, desc, module_profiles),
        "payload_scope": "old_data_tables_semantics_with_bounded_pseudonymous_table_previews",
    }


def _table_preview_payloads(
    path: Path,
    desc: Dict[str, Any],
    module_profiles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    profile_by_module = {str(row.get("module") or ""): row for row in module_profiles}
    previews: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in desc.get("files") or []:
        module = str(item.get("module") or "")
        if not module or module in seen:
            continue
        seen.add(module)
        if len(previews) >= _MAX_TABLE_PREVIEW_MODULES:
            break
        columns = [str(col) for col in (item.get("columns") or [])]
        id_col, read_columns, display_columns, hidden_count = _table_preview_columns(columns)
        profile = profile_by_module.get(module) or {}
        base = {
            "module": module,
            "label": profile.get("label") or _module_label(module),
            "file": item.get("file"),
            "rows_total": int(item.get("rows") or profile.get("rows") or 0),
            "columns_total": len(columns),
            "display_columns": display_columns,
            "hidden_columns": hidden_count,
            "row_cap": _MAX_TABLE_PREVIEW_ROWS,
            "column_cap": _MAX_TABLE_PREVIEW_COLUMNS,
            "pseudonymous_entity_column": bool(id_col),
        }
        if not read_columns:
            previews.append({
                **base,
                "status": "unavailable",
                "rows": [],
                "row_count": 0,
                "reason": "No displayable columns after direct identifiers are removed.",
            })
            continue
        try:
            frame = _read_table_preview(path / str(item.get("file") or ""), read_columns, _MAX_TABLE_PREVIEW_ROWS)
        except Exception as exc:
            previews.append({
                **base,
                "status": "unavailable",
                "rows": [],
                "row_count": 0,
                "reason": str(exc)[:160],
            })
            continue
        rows = _public_preview_rows(path, frame, id_col, display_columns)
        previews.append({
            **base,
            "status": "ready" if rows else "empty",
            "rows": rows,
            "row_count": len(rows),
            "truncated_rows": int(base["rows_total"]) > len(rows),
            "truncated_columns": hidden_count > 0,
            "payload_scope": "bounded_pseudonymous_module_table_preview",
        })
    return previews


def _table_preview_columns(columns: List[str]) -> Tuple[str | None, List[str], List[str], int]:
    id_col = next((col for col in columns if _is_direct_identifier_column(col)), None)
    non_id = [col for col in columns if not _is_direct_identifier_column(col)]
    time_cols = [col for col in _TIME_COLUMNS if col in non_id]
    feature_cols = [col for col in _feature_columns(non_id) if col not in time_cols]
    other_cols = [col for col in non_id if col not in time_cols and col not in feature_cols and col not in _METADATA_COLUMNS]
    source_display = _ordered_unique([*time_cols, *feature_cols, *other_cols])
    source_display = source_display[: max(0, _MAX_TABLE_PREVIEW_COLUMNS - (1 if id_col else 0))]
    read_columns = _ordered_unique(([id_col] if id_col else []) + source_display)
    display_columns = (["entity"] if id_col else []) + source_display
    hidden_count = max(0, len([col for col in non_id if col not in source_display]))
    return id_col, read_columns, display_columns, hidden_count


def _is_direct_identifier_column(column: str) -> bool:
    key = str(column or "").strip().lower().replace("_", "").replace("-", "")
    return key in _DIRECT_IDENTIFIER_COLUMN_KEYS


def _read_table_preview(path: Path, columns: List[str], nrows: int) -> Any:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=max(nrows, 1), columns=columns):
                return batch.to_pandas().head(nrows)
            return pd.DataFrame(columns=columns)
        except Exception:
            return pd.read_parquet(path, columns=columns).head(nrows)
    if suffix == ".xlsx":
        return pd.read_excel(path, usecols=columns, nrows=nrows)
    return pd.read_csv(path, usecols=columns, nrows=nrows)


def _public_preview_rows(
    path: Path,
    frame: Any,
    id_col: str | None,
    display_columns: List[str],
) -> List[Dict[str, Any]]:
    if frame is None or getattr(frame, "empty", True):
        return []
    rows: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        public: Dict[str, Any] = {}
        if id_col:
            entity_id = dataio._norm_id(row.get(id_col))
            public["entity"] = _entity_ref(path, entity_id) if entity_id else None
        for col in display_columns:
            if col == "entity":
                continue
            public[col] = _json_cell(row.get(col))
        rows.append(public)
        if len(rows) >= _MAX_TABLE_PREVIEW_ROWS:
            break
    return rows


def _json_cell(value: Any) -> Any:
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _trajectory_review_payload(
    time_lanes: List[Dict[str, Any]],
    selected: Dict[str, Any],
    entities: List[Dict[str, Any]],
    quality_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Build bounded time-window feature-matrix review metadata."""
    ready_lanes = [row for row in time_lanes if row.get("status") == "ready" and row.get("signals")]
    signal_count = sum(int(row.get("signal_count") or 0) for row in ready_lanes)
    selected_signals = []
    for lane in ready_lanes:
        for signal in lane.get("signals") or []:
            selected_signals.append(signal)
    selected_signals = selected_signals[:_MAX_REVIEW_SIGNALS]
    comparison_features = _comparison_feature_payload(quality_metrics)
    return {
        "contract": [
            {
                "index": "01",
                "label": "Entity scope",
                "detail": f"{len(entities)} pseudonymous options exposed",
                "status": "ready" if entities else "warn",
            },
            {
                "index": "02",
                "label": "Loaded signals",
                "detail": f"{signal_count} selected-entity signals",
                "status": "ready" if signal_count else "neutral",
            },
            {
                "index": "03",
                "label": "Feature matrices",
                "detail": f"{len(ready_lanes)} matrix groups available",
                "status": "ready" if ready_lanes else "neutral",
            },
            {
                "index": "04",
                "label": "Review mode",
                "detail": "time windows x features / single entity / aggregate comparison",
                "status": "ready",
            },
        ],
        "modes": [
            {
                "id": "feature_matrix",
                "label": "Feature Matrix",
                "status": "ready" if ready_lanes else "unavailable",
                "description": "Bounded time-window by feature matrices for grouped longitudinal signals.",
            },
            {
                "id": "single_entity",
                "label": "Single Patient",
                "status": "ready" if selected_signals else "unavailable",
                "description": "Selected pseudonymous entity trends and latest values.",
            },
            {
                "id": "multi_entity_comparison",
                "label": "Multi-Patient Comparison",
                "status": "aggregate_only" if comparison_features else "unavailable",
                "description": "Cohort-level feature summaries replace raw multi-entity traces in the native browser payload.",
            },
        ],
        "lanes": ready_lanes,
        "single_entity": {
            "selected_ref": selected.get("ref"),
            "selected_label": selected.get("label"),
            "signals": selected_signals[:12],
        },
        "multi_entity_comparison": {
            "selection_cap": _MAX_ENTITIES,
            "normalization_available": True,
            "features": comparison_features,
            "payload_scope": "aggregate_comparison_no_multi_entity_rows",
        },
        "payload_scope": "feature_matrix_semantics_bounded",
    }


def _patient_overview_payload(
    selected: Dict[str, Any],
    entities: List[Dict[str, Any]],
    time_lanes: List[Dict[str, Any]],
    quality_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Mirror old Patient Overview dashboard/category/table modes."""
    signal_index = _selected_signal_index(time_lanes)
    category_sections = [
        _category_section("vitals", "Vital Signs Snapshot", ("hr", "map", "sbp", "dbp", "resp", "temp", "spo2"), signal_index),
        _category_section("labs", "Key Laboratory Snapshot", ("lact", "lac", "crea", "plt", "wbc", "hgb", "bili"), signal_index),
        _category_section("scores", "Scores and sepsis flags", ("sofa", "sofa2", "qsofa", "sirs", "gcs", "sep3_sofa1", "sep3_sofa2"), signal_index),
        _category_section("support", "Support and therapies", ("mech_vent", "vent_ind", "rrt", "vaso_ind", "norepi_rate", "epi_rate"), signal_index),
    ]
    available_features = {
        str(row.get("feature"))
        for row in (quality_metrics.get("features") or [])
        if row.get("feature")
    }
    return {
        "navigator": {
            "current": selected.get("label"),
            "ordinal": selected.get("ordinal"),
            "options": [
                {
                    "ref": item.get("ref"),
                    "label": item.get("label"),
                    "outcome": item.get("outcome"),
                    "severity": item.get("severity"),
                }
                for item in entities
            ],
            "actions": ["first", "previous", "next", "last", "random"],
        },
        "dashboard": {
            "mode": "Dashboard",
            "summary_cards": _patient_summary_cards(selected),
            "trend_panels": [
                section for section in category_sections
                if section.get("available_count")
            ][:3],
            "sofa_comparator": _sofa_comparator(signal_index),
        },
        "category_view": {
            "mode": "Category View",
            "sections": category_sections,
        },
        "data_table": {
            "mode": "Data Table",
            "available_features": len(available_features),
            "row_preview": "available_in_data_tables",
            "reason": "Use the Data Tables tab for bounded pseudonymous module table previews.",
        },
        "payload_scope": "old_patient_overview_semantics_pseudonymous",
    }


def _quality_review_payload(
    quality: List[Dict[str, Any]],
    quality_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Mirror old Quality page summary, contract and three panels."""
    summary = quality_metrics.get("summary") or {}
    features = quality_metrics.get("features") or []
    missing = sorted(features, key=lambda row: float(row.get("missing_pct") or 0), reverse=True)[:10]
    outliers = sorted(features, key=lambda row: float(row.get("out_of_physio_pct") or 0), reverse=True)[:10]
    temporal = sorted(features, key=lambda row: float(row.get("duplicate_time_pct") or 0), reverse=True)[:10]
    return {
        "summary_cards": [
            {"label": "QC concepts", "value": summary.get("concept_count"), "tone": "ok"},
            {"label": "Records", "value": summary.get("total_records"), "tone": "accent"},
            {"label": "Weighted missing", "value": summary.get("weighted_missing_pct"), "unit": "%", "tone": _rate_tone(summary.get("weighted_missing_pct"), warn=5, danger=20)},
            {"label": "Out-of-physio", "value": summary.get("weighted_out_of_physio_pct"), "unit": "%", "tone": _rate_tone(summary.get("weighted_out_of_physio_pct"), warn=1, danger=5)},
            {"label": "Duplicate TS", "value": summary.get("weighted_duplicate_time_pct"), "unit": "%", "tone": _rate_tone(summary.get("weighted_duplicate_time_pct"), warn=0.5, danger=2)},
        ],
        "contract": [
            {
                "index": "01",
                "label": "Local concept scope",
                "detail": f"{summary.get('concept_count') or 0} concepts · {summary.get('denominator_entities') or 0} entities · {summary.get('total_records') or 0} records",
                "status": "ready" if summary.get("concept_count") else "neutral",
            },
            {
                "index": "02",
                "label": "Missingness gate",
                "detail": f"{summary.get('weighted_missing_pct')}% weighted missing",
                "status": _rate_tone(summary.get("weighted_missing_pct"), warn=5, danger=20),
            },
            {
                "index": "03",
                "label": "Physiologic range",
                "detail": f"{summary.get('weighted_out_of_physio_pct')}% out-of-range values",
                "status": _rate_tone(summary.get("weighted_out_of_physio_pct"), warn=1, danger=5),
            },
            {
                "index": "04",
                "label": "Temporal integrity",
                "detail": f"{summary.get('weighted_duplicate_time_pct')}% duplicate time rows",
                "status": _rate_tone(summary.get("weighted_duplicate_time_pct"), warn=0.5, danger=2),
            },
        ],
        "panels": [
            {"id": "missingness", "label": "Missingness", "rows": _quality_panel_rows(missing, "missing_pct")},
            {"id": "outliers", "label": "Out-of-Physio", "rows": _quality_panel_rows(outliers, "out_of_physio_pct")},
            {"id": "temporal", "label": "Temporal Integrity", "rows": _quality_panel_rows(temporal, "duplicate_time_pct")},
        ],
        "top_issues": quality_metrics.get("top_issues") or [],
        "module_coverage": quality,
        "payload_scope": "old_quality_semantics_aggregate_only",
    }


def _comparison_feature_payload(quality_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    features = [
        row for row in (quality_metrics.get("features") or [])
        if row.get("time_indexed") and int(row.get("records") or 0) > 0
    ]
    features = sorted(
        features,
        key=lambda row: (
            int(row.get("entities") or 0),
            int(row.get("records") or 0),
        ),
        reverse=True,
    )
    return [
        {
            "feature": row.get("feature"),
            "name": row.get("name"),
            "module": row.get("module"),
            "records": row.get("records"),
            "entities": row.get("entities"),
            "coverage_pct": row.get("coverage_pct"),
            "density_per_entity": row.get("density_per_entity"),
        }
        for row in features[:8]
    ]


def _selected_signal_index(time_lanes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for lane in time_lanes:
        for signal in lane.get("signals") or []:
            feature = str(signal.get("feature") or "")
            if feature and feature not in index:
                index[feature] = signal
    return index


def _category_section(
    section_id: str,
    title: str,
    features: Tuple[str, ...],
    signal_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    cards = []
    for feature in features:
        signal = signal_index.get(feature)
        if not signal:
            continue
        current = dataio._num(signal.get("current"))
        cards.append({
            "feature": feature,
            "label": signal.get("name") or _concept_name(feature),
            "unit": signal.get("unit") or _concept_unit(feature),
            "current": current,
            "delta": _signal_delta(signal.get("values") or []),
            "tone": _patient_feature_tone(feature, current),
            "values": (signal.get("values") or [])[:_MAX_SIGNAL_POINTS],
            "thresholds": signal.get("thresholds") or [],
        })
    return {
        "id": section_id,
        "title": title,
        "available_count": len(cards),
        "cards": cards,
    }


def _patient_summary_cards(selected: Dict[str, Any]) -> List[Dict[str, Any]]:
    demo = selected.get("demographics") or {}
    scores = selected.get("scores") or {}
    outcomes = selected.get("outcomes") or {}
    return [
        {
            "label": "Age / sex",
            "value": f"{_display_value(dataio._num(demo.get('age')), decimals=0)} / {dataio._clean(demo.get('sex')) or 'unknown'}",
            "tone": "neutral",
        },
        {
            "label": "SOFA-2 max",
            "value": _display_value(dataio._num(scores.get("sofa2_max")), decimals=1),
            "tone": _score_tone(dataio._num(scores.get("sofa2_max"))),
        },
        {
            "label": "Sepsis-3",
            "value": "Positive" if scores.get("sepsis3_sofa2") is True else ("Negative" if scores.get("sepsis3_sofa2") is False else "unknown"),
            "tone": "warn" if scores.get("sepsis3_sofa2") is True else "ok",
        },
        {
            "label": "Outcome",
            "value": outcomes.get("status") or "Unknown",
            "tone": "bad" if outcomes.get("status") == "Deceased" else "ok",
        },
        {
            "label": "ICU LOS",
            "value": f"{_display_value(dataio._num(outcomes.get('icu_los_days')), decimals=1)} d",
            "tone": "neutral",
        },
    ]


def _sofa_comparator(signal_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    sofa1 = signal_index.get("sofa")
    sofa2 = signal_index.get("sofa2")
    if not sofa1 or not sofa2:
        return {
            "status": "unavailable",
            "reason": "SOFA-1 and SOFA-2 signals are both required.",
        }
    return {
        "status": "ready",
        "features": [
            {
                "feature": "sofa",
                "label": "SOFA-1",
                "current": sofa1.get("current"),
                "values": sofa1.get("values") or [],
            },
            {
                "feature": "sofa2",
                "label": "SOFA-2",
                "current": sofa2.get("current"),
                "values": sofa2.get("values") or [],
            },
        ],
    }


def _quality_panel_rows(rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    return [
        {
            "feature": row.get("feature"),
            "name": row.get("name"),
            "module": row.get("module"),
            "value": row.get(metric),
            "records": row.get("records"),
            "entities": row.get("entities"),
            "status": row.get("status"),
        }
        for row in rows[:8]
    ]


def _concept_group_label(feature: str) -> str:
    for group, features in concept_catalog.CONCEPT_GROUPS_INTERNAL.items():
        if feature in features:
            return _module_label(group)
    return "Other"


def _module_review_status(coverage: Any, feature_count: int) -> str:
    if feature_count <= 0:
        return "empty"
    if not isinstance(coverage, (int, float)):
        return "unknown"
    if coverage >= 80:
        return "ready"
    if coverage >= 50:
        return "partial"
    return "sparse"


def _signal_delta(values: List[Any]) -> float | None:
    nums = [dataio._num(value) for value in values]
    nums = [value for value in nums if value is not None]
    if len(nums) < 2:
        return None
    return round(float(nums[-1] - nums[0]), 3)


def _patient_feature_tone(feature: str, value: float | None) -> str:
    if value is None:
        return "neutral"
    if feature in {"sep3_sofa1", "sep3_sofa2", "susp_inf", "infection_icd"}:
        return "bad" if value >= 1 else "ok"
    if feature in {"mech_vent", "vent_ind", "rrt", "vaso_ind"}:
        return "warn" if value >= 1 else "ok"
    if feature in {"sofa", "sofa2", "qsofa", "sirs", "mews", "news"}:
        return _score_tone(value)
    bounds = concept_catalog.PHYSIOLOGIC_RANGES.get(feature)
    if bounds:
        low, high = bounds
        if value < low or value > high:
            return "warn"
    return "neutral"


def _score_tone(value: float | None) -> str:
    if value is None:
        return "neutral"
    if value < 6:
        return "ok"
    if value < 10:
        return "warn"
    return "bad"


def _rate_tone(value: Any, *, warn: float, danger: float) -> str:
    number = dataio._num(value)
    if number is None:
        return "neutral"
    if number >= danger:
        return "bad"
    if number >= warn:
        return "warn"
    return "ok"


def _display_value(value: float | None, *, decimals: int) -> str:
    if value is None:
        return "unknown"
    return f"{value:.{decimals}f}"


def _out_of_physio_pct(feature: str, values: Any) -> float:
    bounds = concept_catalog.PHYSIOLOGIC_RANGES.get(feature)
    if not bounds:
        return 0.0
    numeric = dataio._numeric_values(values)
    if not numeric:
        return 0.0
    low, high = bounds
    hits = sum(1 for value in numeric if value < low or value > high)
    return round(hits / len(numeric) * 100, 1)


def _duplicate_time_pct(frame: Any, time_col: str | None) -> float:
    if not time_col or time_col not in getattr(frame, "columns", []) or "stay_id" not in getattr(frame, "columns", []):
        return 0.0
    observed = frame.dropna(subset=[time_col])
    if observed.empty:
        return 0.0
    duplicates = int(observed.duplicated(subset=["stay_id", time_col], keep="first").sum())
    return round(duplicates / len(observed) * 100, 1)


def _quality_feature_status(missing_pct: float | None, outlier_pct: float, duplicate_pct: float) -> str:
    missing = float(missing_pct or 0.0)
    if missing >= 50 or outlier_pct >= 5 or duplicate_pct >= 2:
        return "bad"
    if missing >= 20 or outlier_pct >= 1 or duplicate_pct >= 0.5:
        return "warn"
    return "ok"


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


def _quality_from_module_profiles(module_profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in module_profiles:
        module = str(item.get("module") or "")
        coverage = item.get("coverage_pct")
        out.append({
            "module": module,
            "rows": item.get("rows"),
            "column_count": item.get("feature_count"),
            "covered_entities": item.get("entities"),
            "coverage_pct": coverage,
            "quality_status": _quality_status(module, coverage if isinstance(coverage, (int, float)) else None),
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


def _is_patient_ready(desc: Dict[str, Any]) -> bool:
    modules = {str(item.get("module") or "") for item in desc.get("files") or []}
    if "demographics" in modules:
        return True
    return any("stay_id" in (item.get("columns") or []) for item in desc.get("files") or [])


def _source_provenance(source: Dict[str, Any], desc: Dict[str, Any]) -> Dict[str, Any]:
    path = str(desc.get("path") or source.get("path") or "")
    return {
        "id": source.get("id"),
        "label": source.get("label") or Path(path).name or "local",
        "path_hash": _hash(path),
        "database": desc.get("database") or source.get("database"),
        "generated": desc.get("generated") or source.get("generated"),
    }


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


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
