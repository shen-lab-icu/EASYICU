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
from easyicu.webserver.patient_drilldown import coverage as _feature_coverage
from easyicu.webserver.patient_drilldown import eligibility as _eligibility
from easyicu.webserver import entity_ids as _entity_ids
from easyicu.webserver.patient_drilldown import feature_detail as _feature_detail
from easyicu.webserver.patient_drilldown import navigation as _navigation
from easyicu.webserver import review_labels
from easyicu.webserver import prepared_frames

_eligibility_flow_payload = _eligibility._eligibility_flow_payload
_first_int = _eligibility._first_int
_demographic_flow_label = _eligibility._demographic_flow_label
_demographic_flow_note = _eligibility._demographic_flow_note
_target_clinical_flow_preset = _eligibility._target_clinical_flow_preset
_target_clinical_flow_label = _eligibility._target_clinical_flow_label
_target_clinical_flow_note = _eligibility._target_clinical_flow_note
_int_or_none = _eligibility._int_or_none
_entity_ref = _navigation.entity_ref

_MAX_ENTITIES = 5
_MAX_REVIEW_ENTITIES = 500
_MAX_SIGNAL_POINTS = 12
_MAX_REVIEW_SIGNALS = 24
_MAX_TABLE_PREVIEW_ROWS = 24
_MAX_TABLE_PREVIEW_COLUMNS = 14
_MAX_TABLE_PREVIEW_MODULES = 32
_MAX_TABLE_PAGE_SIZE = 100
_READ_MODULES = ("demographics", "outcome", "sofa2_score", "sepsis3_sofa2", "vitals")
_SIGNAL_SPECS = (
    ("hr", "Heart rate", "bpm"),
    ("map", "MAP", "mmHg"),
    ("spo2", "SpO2", "%"),
    ("temp", "Temp", "deg C"),
)
_TIME_COLUMNS = _feature_coverage.TIME_COLUMNS
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
        public.update(
            {
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
            }
        )
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


def patient_review_entity_page(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return one bounded page of pseudonymous entity navigation options."""
    source, desc = _resolve_registered_source(body)
    path = Path(str(desc.get("path") or source.get("path") or "")).expanduser()
    item = _entity_index_item(desc)
    total_entities = _entity_index_total(desc, item)
    request = _navigation.entity_page_request(body, total_entities)
    rows = _read_entity_index_rows(
        path,
        item,
        offset=int(request["offset"]),
        nrows=int(request["page_size"]),
    )
    if total_entities and not rows:
        raise PatientReviewError({"error": "entity_page_unavailable"})
    navigation = _navigation.entity_navigation_payload(
        path,
        [(ordinal, entity_id) for ordinal, entity_id, _row in rows],
        total_entities=total_entities,
        page=int(request["page"]),
        page_size=int(request["page_size"]),
        page_count=int(request["page_count"]),
        selected_ref=str(body.get("selected_ref") or "") or None,
        selected_ordinal=_strict_positive_int(body.get("selected_ordinal")),
        randomized=bool(request["randomized"]),
    )
    return {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source": _source_provenance(source, desc),
        "navigation": navigation,
        "privacy": {
            "direct_identifiers_returned": False,
            "raw_rows_returned": False,
            "max_entity_page_size": _navigation.MAX_ENTITY_PAGE_SIZE,
            "payload_scope": "bounded_pseudonymous_entity_navigation_page",
        },
    }


def patient_review_entity(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return one verified pseudonymous entity plus a five-entity comparison."""
    source, desc = _resolve_registered_source(body)
    path = Path(str(desc.get("path") or source.get("path") or "")).expanduser()
    item = _entity_index_item(desc)
    total_entities = _entity_index_total(desc, item)
    ordinal = _strict_positive_int(body.get("entity_ordinal"))
    requested_ref = str(body.get("entity_ref") or "").strip()
    if ordinal is None or ordinal > total_entities or not requested_ref:
        raise PatientReviewError({"error": "entity_ref_and_ordinal_required"})
    selected_rows = _read_entity_index_rows(
        path,
        item,
        offset=ordinal - 1,
        nrows=1,
        include_demographics=True,
    )
    if not selected_rows:
        raise PatientReviewError({"error": "unknown_entity_ordinal"})
    selected_ordinal, selected_id, selected_row = selected_rows[0]
    expected_ref = _entity_ref(path, selected_id)
    if selected_ordinal != ordinal or expected_ref != requested_ref:
        raise PatientReviewError({"error": "entity_ref_ordinal_mismatch"})

    comparison_rows = _read_entity_index_rows(
        path,
        item,
        offset=0,
        nrows=_MAX_ENTITIES,
        include_demographics=True,
    )
    if all(row[1] != selected_id for row in comparison_rows):
        comparison_rows = comparison_rows[: max(0, _MAX_ENTITIES - 1)] + [
            (selected_ordinal, selected_id, selected_row)
        ]
    comparison_ids = [row[1] for row in comparison_rows]
    entity_set = set(comparison_ids)
    outcome = _read_module_frame(path, desc, "outcome", entity_set)
    sepsis = _read_module_frame(path, desc, "sepsis3_sofa2", entity_set)
    sofa2 = _read_module_frame(path, desc, "sofa2_score", entity_set)
    vitals = _read_module_frame(path, desc, "vitals", entity_set)
    review_frames = _read_review_frames(path, desc, entity_set)
    death_by_entity = dataio._stay_bool(outcome, "death", missing_false=True)
    los_by_entity = dataio._stay_numeric(outcome, "los_icu", "median")
    sofa_by_entity = dataio._stay_numeric(sofa2, "sofa2", "max")
    sepsis_by_entity = dataio._stay_bool(sepsis, "sep3_sofa2", missing_false=True)
    for entity_id in comparison_ids:
        if outcome is not None and not outcome.empty:
            death_by_entity.setdefault(entity_id, False)
        if sepsis is not None and not sepsis.empty:
            sepsis_by_entity.setdefault(entity_id, False)
    entities = [
        _entity_option(path, entity_id, row_ordinal, death_by_entity, sofa_by_entity)
        for row_ordinal, entity_id, _row in comparison_rows
    ]
    selected = _selected_payload(
        path=path,
        entity_id=selected_id,
        ordinal=selected_ordinal,
        row=selected_row,
        death_by_entity=death_by_entity,
        los_by_entity=los_by_entity,
        sofa_by_entity=sofa_by_entity,
        sepsis_by_entity=sepsis_by_entity,
        vitals=vitals,
    )
    time_lanes = _time_lane_payloads(review_frames, selected_id)
    detail_quality = _quality_metrics_payload(review_frames, entity_set)
    trajectory_review = _trajectory_review_payload(
        time_lanes,
        selected,
        entities,
        detail_quality,
        review_frames,
        path,
        comparison_ids,
    )
    return {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source": _source_provenance(source, desc),
        "selected": selected,
        "entities": entities,
        "time_lanes": time_lanes,
        "trajectory_review": trajectory_review,
        "patient_overview": _patient_overview_payload(
            selected, entities, time_lanes, detail_quality
        ),
        "privacy": {
            "direct_identifiers_returned": False,
            "raw_rows_returned": False,
            "max_comparison_entities": _MAX_ENTITIES,
            "max_points_per_signal": _MAX_SIGNAL_POINTS,
            "payload_scope": "one_verified_pseudonymous_entity_plus_bounded_comparison",
        },
    }


def patient_review_table_preview(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return exactly one bounded module table page without recomputing drilldown."""
    source, desc = _resolve_registered_source(body)
    path = Path(str(desc.get("path") or source.get("path") or "")).expanduser()
    table_paging = _table_preview_paging(body)
    module = str(table_paging.get("module") or "")
    if not module:
        raise PatientReviewError({"error": "table_module_required"})
    profiles = _module_profiles(desc, [], set())
    previews = _table_preview_payloads(
        path,
        desc,
        profiles,
        table_paging,
        module,
    )
    if not previews:
        raise PatientReviewError({"error": "unknown_table_module", "module": module})
    preview = previews[0]
    if preview.get("error_code"):
        raise PatientReviewError({"error": preview["error_code"], "module": module})
    return {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source": _source_provenance(source, desc),
        "module_preview": preview,
        "privacy": {
            "direct_identifiers_returned": False,
            "raw_source_rows_returned": False,
            "bounded_pseudonymous_preview_rows_returned": True,
            "max_table_page_size": _MAX_TABLE_PAGE_SIZE,
            "max_table_preview_columns": _MAX_TABLE_PREVIEW_COLUMNS,
            "payload_scope": "one_bounded_pseudonymous_module_table_page",
        },
    }


def patient_review_feature(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return one lazy feature for one verified pseudonymous entity."""

    _source, desc = _resolve_registered_source(body)
    path = Path(str(desc.get("path") or "")).expanduser()
    item = _entity_index_item(desc)
    total_entities = _entity_index_total(desc, item)
    ordinal = _strict_positive_int(body.get("entity_ordinal"))
    requested_ref = str(body.get("entity_ref") or "").strip()
    if ordinal is None or ordinal > total_entities or not requested_ref:
        raise PatientReviewError({"error": "entity_ref_and_ordinal_required"})
    rows = _read_entity_index_rows(path, item, offset=ordinal - 1, nrows=1)
    if not rows:
        raise PatientReviewError({"error": "unknown_entity_ordinal"})
    selected_ordinal, entity_id, _row = rows[0]
    if selected_ordinal != ordinal or _entity_ref(path, entity_id) != requested_ref:
        raise PatientReviewError({"error": "entity_ref_ordinal_mismatch"})
    try:
        return _feature_detail.load_feature_detail(
            export_path=path,
            description=desc,
            entity_id=entity_id,
            entity_ref=requested_ref,
            entity_ordinal=selected_ordinal,
            feature=str(body.get("feature") or ""),
        )
    except _feature_detail.FeatureDetailError as exc:
        raise PatientReviewError(exc.detail) from exc


def patient_review_drilldown(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return a real, bounded Patient Review payload for one registered export."""
    source, desc = _resolve_registered_source(body)
    table_paging = _table_preview_paging(body)
    path = Path(str(desc.get("path") or source.get("path") or "")).expanduser()
    demo = _read_module_frame(path, desc, "demographics")
    if demo is None or getattr(demo, "empty", True):
        fallback = _fallback_entity_frame(path, desc)
        if fallback is None or getattr(fallback, "empty", True):
            raise PatientReviewError({"error": "no_entity_denominator"})
        demo = fallback

    demo = demo.copy()
    demo["stay_id"] = demo["stay_id"].map(_entity_ids.normalize_entity_id)
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
    sepsis_by_entity_all = dataio._stay_bool(
        sepsis_all, "sep3_sofa2", missing_false=True
    )
    for entity_id in entity_ids:
        if outcome_all is not None and not outcome_all.empty:
            death_by_entity_all.setdefault(entity_id, False)
        if sepsis_all is not None and not sepsis_all.empty:
            sepsis_by_entity_all.setdefault(entity_id, False)

    requested_ref = str(body.get("entity_ref") or body.get("selected_ref") or "")
    ref_to_id = {
        _entity_ref(path, entity_id): entity_id for entity_id in review_entity_ids
    }
    selected_id = ref_to_id.get(requested_ref) if requested_ref else None
    if requested_ref and selected_id is None:
        raise PatientReviewError({"error": "unknown_bounded_entity_ref"})
    if selected_id is None:
        selected_id = entity_ids[0]

    demo_by_id = demo.set_index("stay_id", drop=False)
    selected_row = (
        demo_by_id.loc[selected_id] if selected_id in demo_by_id.index else {}
    )
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
        "review_scope": (
            "browser_bounded_entity_sample"
            if len(review_entity_ids) < len(entity_ids)
            else "full_entity_set"
        ),
        "static_aggregate_scope": "full_entity_set",
        "dynamic_aggregate_scope": (
            "browser_bounded_entity_sample"
            if len(review_entity_ids) < len(entity_ids)
            else "full_entity_set"
        ),
        "mean_age": (
            dataio._series_mean(demo.get("age")) if "age" in demo.columns else None
        ),
        "female_pct": (
            dataio._sex_pct(demo.get("sex"), "female")
            if "sex" in demo.columns
            else None
        ),
        "mortality": dataio._bool_pct(
            list((death_by_entity_all or death_by_entity).values())
        ),
        "median_los_icu": dataio._median(
            list((los_by_entity_all or los_by_entity).values())
        ),
        "median_sofa2": dataio._median(list(sofa_by_entity.values())),
        "sepsis_pct": dataio._bool_pct(
            list((sepsis_by_entity_all or sepsis_by_entity).values())
        ),
    }
    navigation_request = _navigation.entity_page_request(body, len(entity_ids))
    navigation_offset = int(navigation_request["offset"])
    navigation_ids = entity_ids[
        navigation_offset : navigation_offset + int(navigation_request["page_size"])
    ]
    entity_navigation = _navigation.entity_navigation_payload(
        path,
        [
            (navigation_offset + index, entity_id)
            for index, entity_id in enumerate(navigation_ids, start=1)
        ],
        total_entities=len(entity_ids),
        page=int(navigation_request["page"]),
        page_size=int(navigation_request["page_size"]),
        page_count=int(navigation_request["page_count"]),
        selected_ref=_entity_ref(path, selected_id),
        selected_ordinal=entity_ids.index(selected_id) + 1,
        randomized=bool(navigation_request["randomized"]),
    )
    eligibility_flow = _eligibility_flow_payload(path, desc, summary)
    feature_coverage = _feature_coverage.build_feature_coverage(path, desc)
    module_profiles = _feature_coverage.apply_to_module_profiles(
        _module_profiles(desc, review_frames, entity_set),
        feature_coverage,
    )
    time_lanes = _time_lane_payloads(review_frames, selected_id)
    quality_metrics = _quality_metrics_payload(review_frames, entity_set)
    quality = _quality_from_module_profiles(module_profiles)
    data_tables = _data_table_review_payload(
        path,
        desc,
        module_profiles,
        summary,
        table_paging,
        feature_coverage,
    )
    trajectory_review = _trajectory_review_payload(
        time_lanes, selected, entities, quality_metrics, review_frames, path, entity_ids
    )
    patient_overview = _patient_overview_payload(
        selected, entities, time_lanes, quality_metrics
    )
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
            "raw_source_rows_returned": False,
            "direct_identifiers_returned": False,
            "max_entity_options": _MAX_ENTITIES,
            "max_entity_page_size": _navigation.MAX_ENTITY_PAGE_SIZE,
            "max_points_per_signal": _MAX_SIGNAL_POINTS,
            "max_table_preview_rows": _MAX_TABLE_PREVIEW_ROWS,
            "max_table_preview_columns": _MAX_TABLE_PREVIEW_COLUMNS,
            "max_table_page_size": _MAX_TABLE_PAGE_SIZE,
            "bounded_table_previews": True,
            "bounded_pseudonymous_preview_rows_returned": True,
            "row_payload_scope": "bounded_pseudonymous_table_previews",
            "payload_tables_are_aggregated": False,
            "payload_tables_are_bounded": True,
        },
        "summary": summary,
        "eligibility_flow": eligibility_flow,
        "module_profiles": module_profiles,
        "feature_coverage": feature_coverage,
        "entities": entities,
        "entity_navigation": entity_navigation,
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


def _resolve_registered_source(
    body: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    registry = source_store.load_registry()
    sources = [s for s in registry.get("sources") or [] if isinstance(s, dict)]
    requested = body.get("source_path") or body.get("path")
    if requested:
        norm = _norm_path(str(requested))
        source = next(
            (s for s in sources if _norm_path(str(s.get("path") or "")) == norm), None
        )
        if source is None:
            raise PatientReviewError(
                {"error": "source_not_registered", "path_hash": _hash(norm)}
            )
    else:
        active = registry.get("active_path")
        if not active:
            raise PatientReviewError({"error": "no_active_export"})
        active_norm = _norm_path(str(active))
        source = next(
            (s for s in sources if _norm_path(str(s.get("path") or "")) == active_norm),
            None,
        )
        if source is None:
            raise PatientReviewError(
                {
                    "error": "active_source_not_registered",
                    "path_hash": _hash(active_norm),
                }
            )

    desc = dataio.describe_export_source(str(source.get("path") or ""))
    if not desc.get("ok"):
        raise PatientReviewError(
            {"error": "invalid_export", "detail": desc.get("error")}
        )
    return source, desc


def _read_module_frame(
    path: Path, desc: Dict[str, Any], module: str, stay_ids: set[str] | None = None
) -> Any:
    return prepared_frames.read_module_frame(
        path, desc, module, _MODULE_COLUMNS[module], stay_ids=stay_ids
    )


def _fallback_entity_frame(path: Path, desc: Dict[str, Any]) -> Any:
    return prepared_frames.fallback_entity_frame(path, desc)


def _entity_index_item(desc: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a stable one-row-per-entity navigation table."""
    item = next(
        (
            row
            for row in desc.get("files") or []
            if row.get("module") == "demographics"
            and _entity_ids.resolve_entity_id_column(row.get("columns") or [])
        ),
        None,
    )
    if not item:
        raise PatientReviewError({"error": "stable_entity_index_unavailable"})
    return item


def _entity_index_total(desc: Dict[str, Any], item: Dict[str, Any]) -> int:
    total = int((desc.get("summary") or {}).get("stays") or item.get("rows") or 0)
    if total <= 0:
        raise PatientReviewError({"error": "no_entity_denominator"})
    return total


def _read_entity_index_rows(
    path: Path,
    item: Dict[str, Any],
    *,
    offset: int,
    nrows: int,
    include_demographics: bool = False,
) -> List[Tuple[int, str, Any]]:
    entity_column = _entity_ids.resolve_entity_id_column(item.get("columns") or [])
    if not entity_column:
        raise PatientReviewError({"error": "stable_entity_index_unavailable"})
    columns = [entity_column]
    if include_demographics:
        columns.extend(
            column for column in ("age", "sex") if column in (item.get("columns") or [])
        )
    try:
        frame = _read_table_preview(
            path / str(item.get("file") or ""),
            columns,
            nrows,
            offset,
        )
    except Exception as exc:
        raise PatientReviewError({"error": "entity_index_read_failed"}) from exc
    if frame is None or getattr(frame, "empty", True):
        return []
    rows: List[Tuple[int, str, Any]] = []
    seen: set[str] = set()
    for position, (_index, row) in enumerate(frame.iterrows(), start=1):
        entity_id = _entity_ids.normalize_entity_id(row.get(entity_column))
        if not entity_id or entity_id in seen:
            raise PatientReviewError({"error": "unstable_entity_index"})
        seen.add(entity_id)
        rows.append((offset + position, entity_id, row))
    return rows


def _strict_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _read_selected_columns(
    path: Path,
    columns: List[str],
    stay_ids: set[str] | None = None,
    *,
    entity_column: str = "stay_id",
) -> Any:
    return prepared_frames.read_selected_columns(
        path, columns, stay_ids, entity_column=entity_column
    )


def _table_preview_paging(body: Dict[str, Any]) -> Dict[str, Any]:
    module = str(body.get("table_module") or "").strip()
    return {
        "module": module,
        "page": _bounded_int(body.get("table_page"), 1, 1, 100000),
        "page_size": _bounded_int(
            body.get("table_page_size"),
            _MAX_TABLE_PREVIEW_ROWS,
            1,
            _MAX_TABLE_PAGE_SIZE,
        ),
    }


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _read_review_frames(
    path: Path, desc: Dict[str, Any], entity_set: set[str]
) -> List[Dict[str, Any]]:
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
        entity_column = _entity_ids.resolve_entity_id_column(columns)
        if not module or module not in _READ_MODULES or not entity_column:
            continue
        feature_cols = _feature_columns(columns)
        time_cols = [col for col in _TIME_COLUMNS if col in columns]
        selected_columns = _ordered_unique(
            [entity_column, *time_cols, *feature_cols]
        )
        if len(selected_columns) <= 1:
            continue
        try:
            frame = _read_selected_columns(
                path / str(item.get("file") or ""),
                selected_columns,
                stay_ids=entity_set,
                entity_column=entity_column,
            )
        except Exception:
            continue
        if (
            frame is None
            or getattr(frame, "empty", True)
            or entity_column not in frame.columns
        ):
            continue
        frame = _entity_ids.canonicalize_entity_frame(frame, entity_column)
        frame = frame.copy()
        frame["stay_id"] = frame["stay_id"].map(_entity_ids.normalize_entity_id)
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
        if current is None or candidate["entity_overlap"] > int(
            current.get("entity_overlap") or 0
        ):
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
        features = [
            str(col)
            for col in ((frame_item or {}).get("features") or _feature_columns(columns))
        ]
        frame = frame_item.get("frame") if frame_item else None
        time_col = frame_item.get("time_col") if frame_item else None
        rows = int((frame_item or item).get("rows") or 0)
        entities = None
        dynamic_features = 0
        static_features = 0
        observed_features = 0
        if frame is not None and not frame.empty:
            entities = (
                int(frame["stay_id"].nunique()) if "stay_id" in frame.columns else None
            )
            for feature in [f for f in features if f in frame.columns]:
                if frame[feature].notna().any():
                    observed_features += 1
                    if time_col:
                        dynamic_features += 1
                    else:
                        static_features += 1
        coverage = (
            round(entities / len(entity_set) * 100, 1)
            if entities is not None and entity_set
            else None
        )
        profiles.append(
            {
                "module": module,
                "label": _module_label(module),
                "label_i18n": _module_label_i18n(module),
                "rows": rows,
                "feature_count": len(features),
                "observed_features": observed_features,
                "entities": entities,
                "coverage_pct": coverage,
                "time_indexed": bool(time_col),
                "dynamic_features": dynamic_features,
                "static_features": static_features,
                "preview_features": features[:6],
            }
        )
    return profiles


def _module_label(module: str) -> str:
    return review_labels.module_label(module)


def _module_label_i18n(module: str) -> Dict[str, str]:
    return review_labels.module_label_i18n(module)


def _plain_label(label: str) -> str:
    return review_labels.plain_label(label)


def _concept_label_i18n(feature: str) -> Dict[str, str]:
    entry = concept_catalog.CONCEPT_DICTIONARY.get(feature)
    if entry:
        return {
            "en": str(entry[0] or feature),
            "zh": (
                str(entry[1] or entry[0] or feature)
                if len(entry) > 1
                else str(entry[0] or feature)
            ),
        }
    fallback = _human_column_label(feature)
    return {"en": fallback, "zh": fallback}


def _column_label_i18n(column: str) -> Dict[str, Any]:
    key = str(column or "")
    lowered = key.lower()
    if key == "entity":
        return {
            "column": key,
            "label_en": "Pseudonymous entity",
            "label_zh": "伪匿名实体",
            "unit": "",
        }
    time_labels = {
        "charttime": ("Chart time", "记录时间"),
        "time": ("Time", "时间"),
        "datetime": ("Date time", "日期时间"),
        "timestamp": ("Timestamp", "时间戳"),
        "starttime": ("Start time", "开始时间"),
        "endtime": ("End time", "结束时间"),
        "storetime": ("Stored time", "存储时间"),
    }
    if lowered in time_labels:
        en, zh = time_labels[lowered]
        return {"column": key, "label_en": en, "label_zh": zh, "unit": ""}
    concept = _concept_label_i18n(lowered)
    unit = _concept_unit(lowered)
    if lowered in concept_catalog.CONCEPT_DICTIONARY:
        return {
            "column": key,
            "label_en": concept["en"],
            "label_zh": concept["zh"],
            "unit": unit,
        }
    fallback = _human_column_label(key)
    return {"column": key, "label_en": fallback, "label_zh": fallback, "unit": unit}


def _human_column_label(column: str) -> str:
    text = str(column or "").replace("_", " ").replace("-", " ").strip()
    return " ".join(part.capitalize() for part in text.split()) or str(column or "")


def _paired_numeric_time_values(
    frame: Any, time_col: str, feature: str
) -> Tuple[List[Any], List[float], int, List[float]]:
    """Pair chart times with numeric values after applying the same row filter."""
    times: List[Any] = []
    values: List[float] = []
    if (
        frame is None
        or getattr(frame, "empty", True)
        or time_col not in frame.columns
        or feature not in frame.columns
    ):
        return times, values, 0, values
    one = frame.sort_values(time_col)
    for raw_time, raw_value in zip(one[time_col], one[feature]):
        num = dataio._num(raw_value)
        if num is None:
            continue
        values.append(float(num))
        times.append(_json_cell(raw_time))
    point_count = len(values)
    sampled = _bounded_signal_indices(point_count)
    return (
        [times[index] for index in sampled],
        [values[index] for index in sampled],
        point_count,
        values,
    )


def _bounded_signal_indices(point_count: int) -> List[int]:
    """Select at most 12 ordered points while preserving the full time window."""
    if point_count <= _MAX_SIGNAL_POINTS:
        return list(range(point_count))
    last = point_count - 1
    intervals = _MAX_SIGNAL_POINTS - 1
    return [(index * last) // intervals for index in range(_MAX_SIGNAL_POINTS)]


def _time_axis_payload(time_col: str, times: List[Any]) -> Dict[str, str]:
    """Describe the source time coordinate without inventing ICU-relative units."""
    lowered = str(time_col or "").strip().lower()
    numeric = bool(times) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in times
    )
    if lowered == "hour" or (lowered == "charttime" and numeric):
        return {
            "kind": "relative_hours",
            "label_en": "ICU hour",
            "label_zh": "ICU 入科后小时",
            "unit": "hour",
            "source_column": lowered,
        }
    if lowered in {"measuredat_minutes", "observationoffset"}:
        return {
            "kind": "relative_minutes",
            "label_en": "ICU hour",
            "label_zh": "ICU 入科后小时",
            "unit": "minute",
            "source_column": lowered,
        }
    if numeric:
        return {
            "kind": "recorded_offset",
            "label_en": "Recorded offset",
            "label_zh": "源记录偏移",
            "unit": "",
            "source_column": lowered,
        }
    return {
        "kind": "datetime",
        "label_en": "Recorded time",
        "label_zh": "源记录时间",
        "unit": "",
        "source_column": lowered,
    }


def _time_lane_payloads(
    review_frames: List[Dict[str, Any]], entity_id: str
) -> List[Dict[str, Any]]:
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
        for feature in item.get("features") or []:
            if feature not in one.columns:
                continue
            times, values, point_count, all_values = _paired_numeric_time_values(
                one, str(time_col), str(feature)
            )
            if not values:
                continue
            by_feature.setdefault(
                feature,
                {
                    "feature": feature,
                    "name": _concept_name(feature),
                    "unit": _concept_unit(feature),
                    "module": item.get("module"),
                    "time_indexed": bool(time_col),
                    "values": values,
                    "times": times,
                    "time_axis": _time_axis_payload(str(time_col), times),
                    "point_count": point_count,
                    "current": all_values[-1],
                    "min": round(min(all_values), 3),
                    "max": round(max(all_values), 3),
                    "mean": round(sum(all_values) / len(all_values), 3),
                    "thresholds": _threshold_payload(feature),
                },
            )
            if len(by_feature) >= _MAX_REVIEW_SIGNALS:
                break
        if len(by_feature) >= _MAX_REVIEW_SIGNALS:
            break

    lanes: List[Dict[str, Any]] = []
    used: set[str] = set()
    for lane, features in concept_catalog.CLINICAL_LANES.items():
        lane_signals = [by_feature[f] for f in features if f in by_feature]
        used.update(row["feature"] for row in lane_signals)
        lanes.append(
            {
                "lane": lane,
                "label": lane.replace("_", " ").title(),
                "signal_count": len(lane_signals),
                "signals": lane_signals,
                "status": "ready" if lane_signals else "unavailable",
            }
        )
    other = [row for key, row in by_feature.items() if key not in used]
    if other:
        lanes.append(
            {
                "lane": "other",
                "label": "Other signals",
                "signal_count": len(other),
                "signals": other,
                "status": "ready",
            }
        )
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
        out.append(
            {
                "value": dataio._num(value),
                "label": labels[idx] if idx < len(labels) else "clinical threshold",
            }
        )
    return out


def _quality_metrics_payload(
    review_frames: List[Dict[str, Any]], entity_set: set[str]
) -> Dict[str, Any]:
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
            observed = frame[
                ["stay_id", feature]
                + ([time_col] if time_col and time_col in frame.columns else [])
            ].copy()
            non_null = observed[observed[feature].notna()]
            records = int(len(non_null))
            entities = int(non_null["stay_id"].nunique()) if records else 0
            missing_pct = (
                round((1 - entities / denominator) * 100, 1) if denominator else None
            )
            outlier_pct = _out_of_physio_pct(feature, non_null[feature])
            duplicate_pct = _duplicate_time_pct(non_null, time_col)
            density = round(records / max(denominator, 1), 3) if denominator else None
            total_records += records
            if missing_pct is not None:
                missing_weight += missing_pct * max(records, 1)
            outlier_weight += outlier_pct * max(records, 1)
            duplicate_weight += duplicate_pct * max(records, 1)
            rows.append(
                {
                    "feature": feature,
                    "name": _concept_name(feature),
                    "module": module,
                    "records": records,
                    "entities": entities,
                    "coverage_pct": (
                        round(entities / denominator * 100, 1) if denominator else None
                    ),
                    "missing_pct": missing_pct,
                    "out_of_physio_pct": outlier_pct,
                    "duplicate_time_pct": duplicate_pct,
                    "density_per_entity": density,
                    "time_indexed": bool(time_col),
                    "status": _quality_feature_status(
                        missing_pct, outlier_pct, duplicate_pct
                    ),
                }
            )

    weight_denominator = sum(max(int(row.get("records") or 0), 1) for row in rows)
    summary = {
        "concept_count": len(rows),
        "total_records": total_records,
        "weighted_missing_pct": (
            round(missing_weight / weight_denominator, 1)
            if weight_denominator
            else None
        ),
        "weighted_out_of_physio_pct": (
            round(outlier_weight / weight_denominator, 1)
            if weight_denominator
            else None
        ),
        "weighted_duplicate_time_pct": (
            round(duplicate_weight / weight_denominator, 1)
            if weight_denominator
            else None
        ),
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
    table_paging: Dict[str, Any],
    feature_coverage: Dict[str, Any],
) -> Dict[str, Any]:
    """Mirror the old Data Tables review contract with bounded local previews."""
    module_count = len(
        [
            row
            for row in module_profiles
            if int(row.get("catalog_feature_count") or row.get("feature_count") or 0)
            > 0
        ]
    )
    feature_count = sum(
        int(row.get("catalog_feature_count") or row.get("feature_count") or 0)
        for row in module_profiles
    )
    selected_count = sum(
        int(row.get("export_observed_features") or row.get("observed_features") or 0)
        for row in module_profiles
    )
    modules = []
    for row in module_profiles:
        feature_count_row = int(
            row.get("catalog_feature_count") or row.get("feature_count") or 0
        )
        if feature_count_row <= 0:
            continue
        coverage = row.get("coverage_pct")
        modules.append(
            {
                "module": row.get("module"),
                "label": row.get("label"),
                "label_i18n": row.get("label_i18n")
                or _module_label_i18n(str(row.get("module") or "")),
                "review_features": feature_count_row,
                "observed_features": int(
                    row.get("export_observed_features")
                    or row.get("observed_features")
                    or 0
                ),
                "rows": int(row.get("rows") or 0),
                "entities": row.get("entities"),
                "coverage_pct": coverage,
                "share_pct": (
                    round(feature_count_row / feature_count * 100, 1)
                    if feature_count
                    else None
                ),
                "shape": "time_indexed" if row.get("time_indexed") else "static",
                "dynamic_features": int(
                    row.get("trajectory_candidate_features")
                    or row.get("dynamic_features")
                    or 0
                ),
                "static_features": int(
                    row.get("export_static_observed_features")
                    or row.get("static_features")
                    or 0
                ),
                "preview_features": [
                    {
                        "feature": feature,
                        "name": _concept_name(str(feature)),
                        "name_i18n": _concept_label_i18n(str(feature)),
                        "unit": _concept_unit(str(feature)),
                        "group": _concept_group_label(str(feature)),
                        "group_i18n": _module_label_i18n(
                            _concept_group_key(str(feature))
                        ),
                    }
                    for feature in (row.get("preview_features") or [])[:6]
                ],
                "status": _module_review_status(coverage, feature_count_row),
                "review_status": (
                    "computed" if row.get("entities") is not None else "inventory_only"
                ),
                "coverage_basis": (
                    "bounded_review_entity_intersection"
                    if row.get("entities") is not None
                    else "not_computed"
                ),
                "preview_status": "available_on_demand",
            }
        )
    requested_module = str(table_paging.get("module") or "")
    if requested_module and not any(
        row.get("module") == requested_module for row in modules
    ):
        raise PatientReviewError(
            {"error": "unknown_table_module", "module": requested_module}
        )
    default_module = (
        requested_module
        if requested_module
        and any(row.get("module") == requested_module for row in modules)
        else (modules[0]["module"] if modules else None)
    )
    coverage_summary = feature_coverage.get("summary") or {}
    return {
        "loaded_summary": {
            "entities": summary.get("entities"),
            "review_features": int(
                coverage_summary.get("definitions") or feature_count
            ),
            "observed_features": int(
                coverage_summary.get("observed") or selected_count
            ),
            "module_count": int(coverage_summary.get("modules") or module_count),
            "source_count": 1,
        },
        "module_picker": {
            "default_module": default_module,
            "module_count": len(modules),
            "selection_mode": "module_then_feature",
        },
        "detail_gate": {
            "title": "Bounded local table previews",
            "default_open": False,
            "reason": "The browser renders capped module table previews with pseudonymous entity tokens; direct identifiers and full tables stay on disk.",
            "available_detail_modes": [
                "module_table_preview",
                "module_glance",
                "single_feature_metadata",
            ],
        },
        "modules": modules,
        "table_previews": _table_preview_payloads(
            path, desc, module_profiles, table_paging, default_module
        ),
        "payload_scope": "module_inventory_plus_one_bounded_pseudonymous_preview",
    }


def _table_preview_payloads(
    path: Path,
    desc: Dict[str, Any],
    module_profiles: List[Dict[str, Any]],
    table_paging: Dict[str, Any],
    default_module: str | None,
) -> List[Dict[str, Any]]:
    profile_by_module = {str(row.get("module") or ""): row for row in module_profiles}
    previews: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in desc.get("files") or []:
        module = str(item.get("module") or "")
        if not module or module in seen:
            continue
        seen.add(module)
        if default_module and module != default_module:
            continue
        if len(previews) >= _MAX_TABLE_PREVIEW_MODULES:
            break
        columns = [str(col) for col in (item.get("columns") or [])]
        id_col, read_columns, display_columns, hidden_count = _table_preview_columns(
            columns
        )
        profile = profile_by_module.get(module) or {}
        selected_for_page = module == default_module
        page_size = int(table_paging.get("page_size") or _MAX_TABLE_PREVIEW_ROWS)
        rows_total = int(item.get("rows") or profile.get("rows") or 0)
        page_count = max(1, (rows_total + page_size - 1) // page_size)
        requested_page = int(table_paging.get("page") or 1) if selected_for_page else 1
        page = max(1, min(page_count, requested_page))
        offset = (page - 1) * page_size
        base = {
            "module": module,
            "label": profile.get("label") or _module_label(module),
            "label_i18n": profile.get("label_i18n") or _module_label_i18n(module),
            "file": item.get("file"),
            "rows_total": rows_total,
            "columns_total": len(columns),
            "display_columns": display_columns,
            "display_column_labels": [
                _column_label_i18n(column) for column in display_columns
            ],
            "hidden_columns": hidden_count,
            "row_cap": _MAX_TABLE_PREVIEW_ROWS,
            "page_size_cap": _MAX_TABLE_PAGE_SIZE,
            "column_cap": _MAX_TABLE_PREVIEW_COLUMNS,
            "pseudonymous_entity_column": bool(id_col),
            "identifier_policy": "pseudonymous_entity_token",
            "page": page,
            "page_size": page_size,
            "page_count": page_count,
            "row_offset": offset,
        }
        if not read_columns:
            previews.append(
                {
                    **base,
                    "status": "unavailable",
                    "rows": [],
                    "row_count": 0,
                    "reason": "No displayable columns after direct identifiers are removed.",
                }
            )
            continue
        try:
            frame = _read_table_preview(
                path / str(item.get("file") or ""),
                read_columns,
                page_size,
                offset,
            )
        except Exception:
            previews.append(
                {
                    **base,
                    "status": "unavailable",
                    "rows": [],
                    "row_count": 0,
                    "reason": "Bounded table preview could not be read.",
                    "error_code": "bounded_table_preview_read_failed",
                }
            )
            continue
        rows = _public_preview_rows(path, frame, id_col, display_columns)
        row_start = offset + 1 if rows else 0
        row_end = offset + len(rows) if rows else 0
        previews.append(
            {
                **base,
                "status": "ready" if rows else "empty",
                "rows": rows,
                "row_count": len(rows),
                "row_start": row_start,
                "row_end": row_end,
                "has_previous": page > 1,
                "has_next": row_end < rows_total,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "page_count": page_count,
                    "row_start": row_start,
                    "row_end": row_end,
                    "rows_total": rows_total,
                    "has_previous": page > 1,
                    "has_next": row_end < rows_total,
                },
                "truncated_rows": row_end < rows_total,
                "truncated_columns": hidden_count > 0,
                "payload_scope": "bounded_pseudonymous_module_table_preview",
            }
        )
    return previews


def _table_preview_columns(
    columns: List[str],
) -> Tuple[str | None, List[str], List[str], int]:
    id_col = next((col for col in columns if _is_direct_identifier_column(col)), None)
    non_id = [col for col in columns if not _is_direct_identifier_column(col)]
    time_cols = [col for col in _TIME_COLUMNS if col in non_id]
    feature_cols = [col for col in _feature_columns(non_id) if col not in time_cols]
    other_cols = [
        col
        for col in non_id
        if col not in time_cols
        and col not in feature_cols
        and col not in _METADATA_COLUMNS
    ]
    source_display = _ordered_unique([*time_cols, *feature_cols, *other_cols])
    source_display = source_display[
        : max(0, _MAX_TABLE_PREVIEW_COLUMNS - (1 if id_col else 0))
    ]
    read_columns = _ordered_unique(([id_col] if id_col else []) + source_display)
    display_columns = (["entity"] if id_col else []) + source_display
    hidden_count = max(0, len([col for col in non_id if col not in source_display]))
    return id_col, read_columns, display_columns, hidden_count


def _is_direct_identifier_column(column: str) -> bool:
    key = str(column or "").strip().lower().replace("_", "").replace("-", "")
    return key in _DIRECT_IDENTIFIER_COLUMN_KEYS


def _read_table_preview(
    path: Path, columns: List[str], nrows: int, offset: int = 0
) -> Any:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        remaining_skip = max(0, offset)
        remaining_take = max(0, nrows)
        frames = []
        for batch in parquet.iter_batches(
            batch_size=max(nrows, min(1024, remaining_skip + remaining_take), 1),
            columns=columns,
        ):
            frame = batch.to_pandas()
            if remaining_skip >= len(frame):
                remaining_skip -= len(frame)
                continue
            if remaining_skip:
                frame = frame.iloc[remaining_skip:]
                remaining_skip = 0
            piece = frame.head(remaining_take)
            frames.append(piece)
            remaining_take -= len(piece)
            if remaining_take <= 0:
                break
        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)
    if suffix == ".xlsx":
        skiprows = range(1, offset + 1) if offset else None
        return pd.read_excel(path, usecols=columns, nrows=nrows, skiprows=skiprows)
    skiprows = range(1, offset + 1) if offset else None
    return pd.read_csv(path, usecols=columns, nrows=nrows, skiprows=skiprows)


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
            entity_id = _entity_ids.normalize_entity_id(row.get(id_col))
            public["entity"] = _entity_ref(path, entity_id) if entity_id else None
        for col in display_columns:
            if col == "entity":
                continue
            public[col] = _json_cell(row.get(col))
        rows.append(public)
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
    review_frames: List[Dict[str, Any]],
    path: Path,
    entity_ids: List[str],
) -> Dict[str, Any]:
    """Build bounded time-window feature-matrix review metadata."""
    ready_lanes = [
        row for row in time_lanes if row.get("status") == "ready" and row.get("signals")
    ]
    signal_count = sum(int(row.get("signal_count") or 0) for row in ready_lanes)
    selected_signals = []
    for lane in ready_lanes:
        for signal in lane.get("signals") or []:
            selected_signals.append(signal)
    selected_signals = selected_signals[:_MAX_REVIEW_SIGNALS]
    comparison_payload = _multi_entity_comparison_payload(
        review_frames,
        path,
        entity_ids[:_MAX_ENTITIES],
        selected_signals,
        quality_metrics,
    )
    comparison_features = comparison_payload.get("features") or []
    has_multi_traces = bool(comparison_payload.get("traces"))
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
                "detail": "clinical lanes / single entity / multi-entity same-feature traces",
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
                "status": "ready"
                if has_multi_traces
                else ("aggregate_only" if comparison_features else "unavailable"),
                "description": "Same feature across a bounded set of pseudonymous entities.",
            },
        ],
        "lanes": ready_lanes,
        "single_entity": {
            "selected_ref": selected.get("ref"),
            "selected_label": selected.get("label"),
            "signals": selected_signals[:12],
        },
        "multi_entity_comparison": comparison_payload,
        "payload_scope": "feature_matrix_semantics_bounded",
    }


def _multi_entity_comparison_payload(
    review_frames: List[Dict[str, Any]],
    path: Path,
    entity_ids: List[str],
    selected_signals: List[Dict[str, Any]],
    quality_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Return old Patient Review-style same-feature traces across entities.

    The payload remains bounded and pseudonymous: it exposes at most five entity
    tokens and at most ``_MAX_SIGNAL_POINTS`` values per entity, never stay_id.
    """
    aggregate_features = _comparison_feature_payload(quality_metrics)
    candidates = _comparison_feature_candidates(review_frames, selected_signals)
    for feature in candidates:
        for item in review_frames:
            frame = item.get("frame")
            time_col = item.get("time_col")
            if (
                frame is None
                or frame.empty
                or "stay_id" not in frame.columns
                or not time_col
                or time_col not in frame.columns
                or feature not in frame.columns
            ):
                continue
            traces = _feature_traces_for_entities(
                frame, path, entity_ids, time_col, feature
            )
            if len(traces) >= 2:
                return {
                    "selection_cap": _MAX_ENTITIES,
                    "normalization_available": True,
                    "feature": feature,
                    "label": _concept_name(feature),
                    "unit": _concept_unit(feature),
                    "module": item.get("module"),
                    "module_label": _module_label(str(item.get("module") or "")),
                    "traces": traces,
                    "time_axis": traces[0].get("time_axis") or {},
                    "compared_entities": len(traces),
                    "features": aggregate_features[:8],
                    "payload_scope": "bounded_pseudonymous_multi_entity_same_feature_traces",
                }
    return {
        "selection_cap": _MAX_ENTITIES,
        "normalization_available": True,
        "features": aggregate_features,
        "traces": [],
        "payload_scope": "aggregate_comparison_no_multi_entity_traces_available",
    }


def _comparison_feature_candidates(
    review_frames: List[Dict[str, Any]], selected_signals: List[Dict[str, Any]]
) -> List[str]:
    preferred = [
        "hr",
        "map",
        "sbp",
        "dbp",
        "spo2",
        "resp",
        "temp",
        "lact",
        "lac",
        "sofa2",
        "sofa",
    ]
    out: List[str] = []
    for signal in selected_signals:
        feature = str(signal.get("feature") or signal.get("key") or "").strip()
        if feature and feature not in out:
            out.append(feature)
    for feature in preferred:
        if feature not in out:
            out.append(feature)
    for item in review_frames:
        if not item.get("time_col"):
            continue
        for feature in item.get("features") or []:
            if feature and feature not in out:
                out.append(str(feature))
    return out


def _feature_traces_for_entities(
    frame: Any,
    path: Path,
    entity_ids: List[str],
    time_col: str,
    feature: str,
) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    subset = frame[frame["stay_id"].isin(entity_ids)].copy()
    if subset.empty:
        return traces
    subset = subset.dropna(subset=[feature])
    if subset.empty:
        return traces
    for ordinal, entity_id in enumerate(entity_ids, start=1):
        one = subset[subset["stay_id"] == entity_id].copy()
        if one.empty:
            continue
        times, values, point_count, _all_values = _paired_numeric_time_values(
            one, str(time_col), str(feature)
        )
        if point_count < 2:
            continue
        traces.append(
            {
                "ref": _entity_ref(path, entity_id),
                "label": f"Entity {ordinal}",
                "values": values,
                "times": times,
                "time_axis": _time_axis_payload(str(time_col), times),
                "point_count": point_count,
                "bounded": True,
                "max_points": _MAX_SIGNAL_POINTS,
            }
        )
        if len(traces) >= _MAX_ENTITIES:
            break
    return traces


def _patient_overview_payload(
    selected: Dict[str, Any],
    entities: List[Dict[str, Any]],
    time_lanes: List[Dict[str, Any]],
    quality_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Mirror old Patient Overview dashboard/category/table modes."""
    signal_index = _selected_signal_index(time_lanes)
    category_sections = [
        _category_section(
            "vitals",
            "Vital Signs Snapshot",
            ("hr", "map", "sbp", "dbp", "resp", "temp", "spo2"),
            signal_index,
        ),
        _category_section(
            "labs",
            "Key Laboratory Snapshot",
            ("lact", "lac", "crea", "plt", "wbc", "hgb", "bili"),
            signal_index,
        ),
        _category_section(
            "scores",
            "Scores and sepsis flags",
            ("sofa", "sofa2", "qsofa", "sirs", "gcs", "sep3_sofa1", "sep3_sofa2"),
            signal_index,
        ),
        _category_section(
            "support",
            "Support and therapies",
            ("mech_vent", "vent_ind", "rrt", "vaso_ind", "norepi_rate", "epi_rate"),
            signal_index,
        ),
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
            "actions": [
                "previous_group",
                "next_group",
                "random_group",
                "select_entity",
            ],
        },
        "dashboard": {
            "mode": "Dashboard",
            "summary_cards": _patient_summary_cards(selected),
            "trend_panels": [
                section
                for section in category_sections
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
    missing = sorted(
        features, key=lambda row: float(row.get("missing_pct") or 0), reverse=True
    )[:10]
    outliers = sorted(
        features, key=lambda row: float(row.get("out_of_physio_pct") or 0), reverse=True
    )[:10]
    temporal = sorted(
        features,
        key=lambda row: float(row.get("duplicate_time_pct") or 0),
        reverse=True,
    )[:10]
    return {
        "summary_cards": [
            {
                "label": "QC concepts",
                "value": summary.get("concept_count"),
                "tone": "ok",
            },
            {
                "label": "Records",
                "value": summary.get("total_records"),
                "tone": "accent",
            },
            {
                "label": "Weighted missing",
                "value": summary.get("weighted_missing_pct"),
                "unit": "%",
                "tone": _rate_tone(
                    summary.get("weighted_missing_pct"), warn=5, danger=20
                ),
            },
            {
                "label": "Out-of-physio",
                "value": summary.get("weighted_out_of_physio_pct"),
                "unit": "%",
                "tone": _rate_tone(
                    summary.get("weighted_out_of_physio_pct"), warn=1, danger=5
                ),
            },
            {
                "label": "Duplicate TS",
                "value": summary.get("weighted_duplicate_time_pct"),
                "unit": "%",
                "tone": _rate_tone(
                    summary.get("weighted_duplicate_time_pct"), warn=0.5, danger=2
                ),
            },
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
                "status": _rate_tone(
                    summary.get("weighted_missing_pct"), warn=5, danger=20
                ),
            },
            {
                "index": "03",
                "label": "Physiologic range",
                "detail": f"{summary.get('weighted_out_of_physio_pct')}% out-of-range values",
                "status": _rate_tone(
                    summary.get("weighted_out_of_physio_pct"), warn=1, danger=5
                ),
            },
            {
                "index": "04",
                "label": "Temporal integrity",
                "detail": f"{summary.get('weighted_duplicate_time_pct')}% duplicate time rows",
                "status": _rate_tone(
                    summary.get("weighted_duplicate_time_pct"), warn=0.5, danger=2
                ),
            },
        ],
        "panels": [
            {
                "id": "missingness",
                "label": "Missingness",
                "rows": _quality_panel_rows(missing, "missing_pct"),
            },
            {
                "id": "outliers",
                "label": "Out-of-Physio",
                "rows": _quality_panel_rows(outliers, "out_of_physio_pct"),
            },
            {
                "id": "temporal",
                "label": "Temporal Integrity",
                "rows": _quality_panel_rows(temporal, "duplicate_time_pct"),
            },
        ],
        "top_issues": quality_metrics.get("top_issues") or [],
        "module_coverage": quality,
        "payload_scope": "old_quality_semantics_aggregate_only",
    }


def _comparison_feature_payload(
    quality_metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    features = [
        row
        for row in (quality_metrics.get("features") or [])
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


def _selected_signal_index(
    time_lanes: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
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
        cards.append(
            {
                "feature": feature,
                "label": signal.get("name") or _concept_name(feature),
                "unit": signal.get("unit") or _concept_unit(feature),
                "current": current,
                "delta": _signal_delta(signal.get("values") or []),
                "tone": _patient_feature_tone(feature, current),
                "values": (signal.get("values") or [])[:_MAX_SIGNAL_POINTS],
                "thresholds": signal.get("thresholds") or [],
            }
        )
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
    los_value = _display_value(dataio._num(outcomes.get("icu_los_days")), decimals=1)
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
            "value": (
                "Positive"
                if scores.get("sepsis3_sofa2") is True
                else ("Negative" if scores.get("sepsis3_sofa2") is False else "unknown")
            ),
            "tone": "warn" if scores.get("sepsis3_sofa2") is True else "ok",
        },
        {
            "label": "Outcome",
            "value": outcomes.get("status") or "Unknown",
            "tone": "bad" if outcomes.get("status") == "Deceased" else "ok",
        },
        {
            "label": "ICU LOS",
            "value": f"{los_value} d" if los_value != "unknown" else los_value,
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


def _quality_panel_rows(
    rows: List[Dict[str, Any]], metric: str
) -> List[Dict[str, Any]]:
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
    group = _concept_group_key(feature)
    if group != "other":
        return _module_label(group)
    return "Other"


def _concept_group_key(feature: str) -> str:
    for group, features in concept_catalog.CONCEPT_GROUPS_INTERNAL.items():
        if feature in features:
            return group
    return "other"


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
    if (
        not time_col
        or time_col not in getattr(frame, "columns", [])
        or "stay_id" not in getattr(frame, "columns", [])
    ):
        return 0.0
    observed = frame.dropna(subset=[time_col])
    if observed.empty:
        return 0.0
    duplicates = int(
        observed.duplicated(subset=["stay_id", time_col], keep="first").sum()
    )
    return round(duplicates / len(observed) * 100, 1)


def _quality_feature_status(
    missing_pct: float | None, outlier_pct: float, duplicate_pct: float
) -> str:
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
        "outcome": (
            "Deceased" if dead is True else ("Survived" if dead is False else "Unknown")
        ),
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
            "status": (
                "Deceased"
                if dead is True
                else ("Survived" if dead is False else "Unknown")
            ),
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
    one = vitals[vitals["stay_id"].map(_entity_ids.normalize_entity_id) == str(entity_id)].copy()
    if one.empty:
        return []
    if "charttime" in one.columns:
        one = one.sort_values("charttime")
    out: List[Dict[str, Any]] = []
    for key, name, unit in _SIGNAL_SPECS:
        if key not in one.columns:
            continue
        values = dataio._numeric_values(one[key])
        bounded = [values[index] for index in _bounded_signal_indices(len(values))]
        if bounded:
            out.append(
                {
                    "key": key,
                    "name": name,
                    "unit": unit,
                    "current": values[-1],
                    "values": bounded,
                    "point_count": len(values),
                    "bounded": True,
                    "max_points": _MAX_SIGNAL_POINTS,
                }
            )
    return out


def _quality_from_module_profiles(
    module_profiles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in module_profiles:
        module = str(item.get("module") or "")
        coverage = item.get("coverage_pct")
        out.append(
            {
                "module": module,
                "metric_kind": dataio._presence_rate_kind(module) or "coverage",
                "rows": item.get("rows"),
                "column_count": item.get("feature_count"),
                "covered_entities": item.get("entities"),
                "coverage_pct": coverage,
                "quality_status": _quality_status(
                    module, coverage if isinstance(coverage, (int, float)) else None
                ),
            }
        )
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
        coverage = (
            round(covered / cohort_size * 100, 1)
            if isinstance(cohort_size, int) and cohort_size
            else None
        )
        out.append(
            {
                "module": module,
                "metric_kind": dataio._presence_rate_kind(module) or "coverage",
                "rows": rows,
                "column_count": len(columns),
                "covered_entities": covered,
                "coverage_pct": coverage,
                "quality_status": _quality_status(module, coverage),
            }
        )
    return out


def _bounded_covered_entities(
    path: Path, item: Dict[str, Any], cohort_size: Any
) -> int | None:
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
    if dataio._is_presence_rate_module(module):
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
    return any(
        "stay_id" in (item.get("columns") or []) for item in desc.get("files") or []
    )


def _source_provenance(source: Dict[str, Any], desc: Dict[str, Any]) -> Dict[str, Any]:
    path = str(desc.get("path") or source.get("path") or "")
    return {
        "id": source.get("id"),
        "label": source.get("label") or Path(path).name or "local",
        "path_hash": _hash(path),
        "database": desc.get("database") or source.get("database"),
        "generated": desc.get("generated") or source.get("generated"),
    }


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


__all__ = [
    "PatientReviewError",
    "patient_review_drilldown",
    "patient_review_entity",
    "patient_review_entity_page",
    "patient_review_feature",
    "patient_review_sources",
    "patient_review_table_preview",
]
