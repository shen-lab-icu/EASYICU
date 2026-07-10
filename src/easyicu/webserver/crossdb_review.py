"""Bounded Cross-DB Review aggregates for the native FastAPI UI.

Stage18 compares two or more registered EasyICU exports using cohort-level
aggregates only. Matched cohorts, row-level filters, p-values/SMDs, and formal
cross-database claims remain fail-closed until the numeric evidence audit gate.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List

from easyicu.concept import catalog as concept_catalog
from easyicu.databases.profiles import (
    DATABASE_LABELS,
    normalize_database_key as canonical_database_key,
    public_database_keys,
)
from easyicu.io.data_paths import (
    DATABASE_ALIASES,
    _path_looks_like_database,
    find_database_path,
)
from easyicu.webserver import cohort_review
from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store

_REQUIRED_CORE_MODULES = {"demographics", "outcome"}
_OPTIONAL_COMPARISON_MODULES = {"sepsis3_sofa2", "sofa2_score", "vitals"}
_UNSUPPORTED_FILTERS = {
    "row_level_filters": "Cross-DB Stage18 accepts registered-source aggregates only.",
    "age_at_admission": "Age cuts need audited row-level cohort construction.",
    "minimum_icu_los": "LOS filters must recompute denominators before comparison.",
    "sepsis3_positive_only": "Event-restricted cohorts need a dedicated cohort-builder pass.",
    "matched_cohort": "Matched cohorts are blocked until the numeric evidence audit gate.",
    "propensity_match": "Matched analyses are not implemented in the native Cross-DB aggregate path.",
    "patient_list": "Identifier-list filtering is intentionally not accepted by this endpoint.",
}
_UNSUPPORTED_STATISTICS = {
    "p_value": "Inferential statistics are withheld until the numeric evidence audit gate.",
    "p-values": "Inferential statistics are withheld until the numeric evidence audit gate.",
    "smd": "Standardized mean differences need audited row-level group construction.",
    "confidence_interval": "Confidence intervals need an audited statistical backend.",
    "matched_analysis": "Matched analyses are outside Stage18 Cross-DB Review.",
    "paired_sofa_reclassification": "Paired SOFA reclassification is blocked before the numeric audit gate.",
}
_NON_FEATURE_COLUMNS = {
    "stay_id",
    "subject_id",
    "hadm_id",
    "icustay_id",
    "patient_id",
    "patient",
    "charttime",
    "time",
    "timestamp",
    "starttime",
    "endtime",
    "admittime",
    "dischtime",
    "intime",
    "outtime",
    "row_id",
    "database",
    "source",
}
_RAW_DB_LABELS = DATABASE_LABELS
_COMMON_RAW_ROOT_CANDIDATES: List[str] = []
_DEMO_MULTIDB_DATABASES = public_database_keys()
_DEMO_RECORDS_PER_FEATURE = 192
_DEMO_MULTIDB_FEATURE_SPECS = {
    "miiv": {
        "hr": (80, 15),
        "sbp": (120, 20),
        "dbp": (70, 12),
        "map": (85, 15),
        "temp": (37.2, 0.5),
        "resp": (18, 4),
        "spo2": (96, 3),
        "glu": (140, 50),
        "na": (140, 5),
        "k": (4.2, 0.6),
        "crea": (1.2, 0.8),
        "bili": (1.5, 1.2),
        "lact": (2.2, 1.5),
        "hgb": (11, 2),
        "plt": (200, 80),
        "wbc": (12, 5),
        "ph": (7.38, 0.08),
        "po2": (90, 20),
        "pco2": (40, 8),
        "fio2": (45, 20),
        "sofa2": (5.2, 3.8),
        "sofa2_resp": (1.2, 1.1),
        "sofa2_coag": (0.8, 0.9),
        "sofa2_liver": (0.6, 0.8),
        "sofa2_cardio": (1.0, 1.2),
        "sofa2_cns": (0.8, 1.0),
        "sofa2_renal": (0.8, 1.0),
    },
    "eicu": {
        "hr": (85, 18),
        "sbp": (125, 25),
        "dbp": (72, 14),
        "map": (88, 18),
        "temp": (37.0, 0.6),
        "resp": (20, 5),
        "spo2": (95, 4),
        "glu": (150, 60),
        "na": (139, 6),
        "k": (4.0, 0.7),
        "crea": (1.4, 1.0),
        "bili": (1.8, 1.5),
        "lact": (2.5, 1.8),
        "hgb": (10.5, 2.2),
        "plt": (180, 90),
        "wbc": (13, 6),
        "ph": (7.36, 0.09),
        "po2": (85, 22),
        "pco2": (42, 10),
        "fio2": (50, 25),
        "sofa2": (6.0, 4.2),
        "sofa2_resp": (1.4, 1.2),
        "sofa2_coag": (0.9, 1.0),
        "sofa2_liver": (0.7, 0.9),
        "sofa2_cardio": (1.2, 1.3),
        "sofa2_cns": (0.9, 1.1),
        "sofa2_renal": (0.9, 1.1),
    },
    "aumc": {
        "hr": (75, 12),
        "sbp": (115, 18),
        "dbp": (65, 10),
        "map": (80, 12),
        "temp": (37.4, 0.4),
        "resp": (16, 3),
        "spo2": (97, 2),
        "glu": (130, 45),
        "na": (141, 4),
        "k": (4.3, 0.5),
        "crea": (1.0, 0.6),
        "bili": (1.2, 1.0),
        "lact": (1.8, 1.2),
        "hgb": (11.5, 1.8),
        "plt": (220, 70),
        "wbc": (11, 4),
        "ph": (7.40, 0.06),
        "po2": (95, 18),
        "pco2": (38, 6),
        "fio2": (40, 18),
        "sofa2": (4.5, 3.5),
        "sofa2_resp": (1.0, 1.0),
        "sofa2_coag": (0.7, 0.8),
        "sofa2_liver": (0.5, 0.7),
        "sofa2_cardio": (0.9, 1.1),
        "sofa2_cns": (0.7, 0.9),
        "sofa2_renal": (0.7, 0.9),
    },
    "hirid": {
        "hr": (78, 14),
        "sbp": (118, 22),
        "dbp": (68, 11),
        "map": (83, 14),
        "temp": (37.3, 0.5),
        "resp": (17, 4),
        "spo2": (96, 3),
        "glu": (135, 48),
        "na": (140, 5),
        "k": (4.1, 0.6),
        "crea": (1.1, 0.7),
        "bili": (1.4, 1.1),
        "lact": (2.0, 1.4),
        "hgb": (11.2, 2.0),
        "plt": (210, 75),
        "wbc": (11.5, 4.5),
        "ph": (7.39, 0.07),
        "po2": (92, 19),
        "pco2": (39, 7),
        "fio2": (42, 19),
        "sofa2": (4.8, 3.6),
        "sofa2_resp": (1.1, 1.0),
        "sofa2_coag": (0.7, 0.9),
        "sofa2_liver": (0.5, 0.7),
        "sofa2_cardio": (1.0, 1.1),
        "sofa2_cns": (0.7, 0.9),
        "sofa2_renal": (0.8, 1.0),
    },
    "mimic": {
        "hr": (82, 16),
        "sbp": (122, 21),
        "dbp": (71, 13),
        "map": (86, 16),
        "temp": (37.1, 0.5),
        "resp": (19, 4),
        "spo2": (95, 3),
        "glu": (145, 55),
        "na": (139, 5),
        "k": (4.1, 0.6),
        "crea": (1.3, 0.9),
        "bili": (1.6, 1.3),
        "lact": (2.3, 1.6),
        "hgb": (10.8, 2.1),
        "plt": (190, 85),
        "wbc": (12.5, 5.5),
        "ph": (7.37, 0.08),
        "po2": (88, 21),
        "pco2": (41, 9),
        "fio2": (48, 22),
        "sofa2": (5.5, 4.0),
        "sofa2_resp": (1.3, 1.1),
        "sofa2_coag": (0.8, 0.9),
        "sofa2_liver": (0.6, 0.8),
        "sofa2_cardio": (1.1, 1.2),
        "sofa2_cns": (0.8, 1.0),
        "sofa2_renal": (0.9, 1.0),
    },
    "sic": {
        "hr": (77, 13),
        "sbp": (116, 19),
        "dbp": (67, 11),
        "map": (82, 13),
        "temp": (37.3, 0.4),
        "resp": (17, 3),
        "spo2": (97, 2),
        "glu": (132, 46),
        "na": (141, 4),
        "k": (4.2, 0.5),
        "crea": (1.05, 0.65),
        "bili": (1.3, 1.0),
        "lact": (1.9, 1.3),
        "hgb": (11.3, 1.9),
        "plt": (215, 72),
        "wbc": (11.2, 4.2),
        "ph": (7.40, 0.06),
        "po2": (93, 18),
        "pco2": (38, 6),
        "fio2": (41, 18),
        "sofa2": (4.2, 3.3),
        "sofa2_resp": (1.0, 1.0),
        "sofa2_coag": (0.6, 0.8),
        "sofa2_liver": (0.5, 0.7),
        "sofa2_cardio": (0.8, 1.0),
        "sofa2_cns": (0.6, 0.8),
        "sofa2_renal": (0.7, 0.9),
    },
}
_DEMO_FEATURE_BOUNDS = {
    "hr": (35, 180),
    "sbp": (60, 230),
    "dbp": (25, 140),
    "map": (35, 170),
    "temp": (34.0, 42.0),
    "resp": (5, 50),
    "spo2": (60, 100),
    "glu": (25, 600),
    "na": (110, 170),
    "k": (1.8, 8.5),
    "crea": (0.1, 12.0),
    "bili": (0.1, 30.0),
    "lact": (0.2, 18.0),
    "hgb": (3.5, 22.0),
    "plt": (5, 900),
    "wbc": (0.1, 80),
    "ph": (6.8, 7.8),
    "po2": (20, 320),
    "pco2": (10, 120),
    "fio2": (21, 100),
    "sofa2": (0, 24),
    "sofa2_resp": (0, 4),
    "sofa2_coag": (0, 4),
    "sofa2_liver": (0, 4),
    "sofa2_cardio": (0, 4),
    "sofa2_cns": (0, 4),
    "sofa2_renal": (0, 4),
}


def crossdb_review_summary(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return native Cross-DB descriptive aggregates for registered exports."""
    _reject_unsupported_request(body)
    requested_sources = _resolve_registered_sources(body)

    cohort_payloads: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for source in requested_sources:
        safe_source = _safe_registered_source(source)
        try:
            cohort_payloads.append(
                cohort_review.cohort_review_summary({"source_path": source["path"]})
            )
        except cohort_review.CohortReviewError as exc:
            errors.append(
                {
                    "source": safe_source,
                    "error": (exc.detail or {}).get("error") or "cohort_summary_failed",
                    "detail": _safe_error_detail(exc.detail),
                }
            )
    if errors:
        raise CrossdbReviewError(
            {
                "error": "invalid_export",
                "sources": [
                    _safe_registered_source(source) for source in requested_sources
                ],
                "errors": errors,
                "privacy": _privacy_payload(),
            }
        )

    sources = [
        _source_aggregate(source, payload)
        for source, payload in zip(requested_sources, cohort_payloads)
    ]
    module_sets = [set(source.get("modules") or []) for source in sources]
    shared_modules = sorted(set.intersection(*module_sets)) if module_sets else []
    all_modules = sorted(set.union(*module_sets)) if module_sets else []
    compatibility_gate = _compatibility_gate(sources, shared_modules, all_modules)
    blocked_features = _blocked_features()

    if compatibility_gate["status"] != "compatible":
        raise CrossdbReviewError(
            {
                "error": "crossdb_incompatible",
                "mode": "real",
                "demo": False,
                "source_count": len(sources),
                "sources": _public_sources(sources),
                "shared_modules": shared_modules,
                "all_modules": all_modules,
                "compatibility_gate": compatibility_gate,
                "blocked_features": blocked_features,
                "privacy": _privacy_payload(),
            }
        )

    return {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source_count": len(sources),
        "sources": _public_sources(sources),
        "rows": _comparison_rows(sources, compatibility_gate),
        "availability": _module_availability(sources, all_modules),
        "feature_density": _feature_density_payload(sources),
        "feature_distributions": _feature_distribution_payload(sources),
        "shared_modules": shared_modules,
        "all_modules": all_modules,
        "compatibility_gate": compatibility_gate,
        "provenance": {
            "computed_from": [
                "source_registry",
                "export_manifest",
                "export_header_schema",
                "manifest_row_counts",
                "bounded_column_reads",
                "bounded_feature_distribution_aggregates",
                "cohort_level_aggregates",
            ],
            "payload_scope": "cross_database_aggregate_only",
            "inference": "blocked_until_numeric_evidence_gate",
        },
        "privacy": _privacy_payload(),
        "blocked_features": blocked_features,
    }


def crossdb_raw_distribution(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return real raw-database Cross-DB feature density aggregates.

    This is the native FastAPI equivalent of the legacy Streamlit
    ``cohort_multidb_page`` operational loader: a local ICU data root plus two
    or more database selections are passed through ``MultiDatabaseDistribution``
    and ``load_concepts``. The response is aggregate-only and contains no
    patient rows or identifiers.
    """
    _reject_unsupported_request(body)
    data_root = _resolve_raw_data_root(body)
    databases = _resolve_raw_databases(data_root, body)
    features = _resolve_raw_features(body)
    max_patients = _bounded_int(
        body.get("max_patients"), default=300, minimum=20, maximum=2000
    )
    sample_size = _bounded_int(
        body.get("sample_size"), default=1500, minimum=100, maximum=10000
    )

    if len(databases) < 2:
        raise CrossdbReviewError(
            {
                "error": "need_two_raw_databases",
                "mode": "real",
                "source_type": "raw_database_root",
                "root_hash": _hash(str(data_root)),
                "detected_databases": databases,
                "privacy": _privacy_payload(),
            }
        )

    try:
        frames = _load_raw_feature_data(
            data_root=str(data_root),
            concepts=features,
            databases=databases,
            max_patients=max_patients,
            sample_size=sample_size,
        )
    except Exception as exc:
        raise CrossdbReviewError(
            {
                "error": "raw_distribution_load_failed",
                "mode": "real",
                "source_type": "raw_database_root",
                "root_hash": _hash(str(data_root)),
                "detail": str(exc),
                "privacy": _privacy_payload(),
            }
        ) from exc

    loaded = {
        str(db): frame
        for db, frame in (frames or {}).items()
        if _raw_frame_has_values(frame)
    }
    missing_loaded = [db for db in databases if db not in loaded]
    if missing_loaded:
        raise CrossdbReviewError(
            {
                "error": "loaded_fewer_than_requested_raw_databases",
                "mode": "real",
                "source_type": "raw_database_root",
                "root_hash": _hash(str(data_root)),
                "requested_databases": databases,
                "loaded_databases": sorted(loaded),
                "missing_databases": missing_loaded,
                "feature_count": len(features),
                "privacy": _privacy_payload(),
            }
        )

    sources = [
        _raw_source_summary(db, frame, data_root) for db, frame in loaded.items()
    ]
    feature_distributions = _raw_feature_distribution_payload(loaded, features)
    shared_modules = [
        row["module"]
        for row in feature_distributions
        if row.get("shared_feature_count")
    ]
    return {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source_type": "raw_database_root",
        "source_count": len(sources),
        "sources": sources,
        "rows": _raw_comparison_rows(loaded, features),
        "availability": _raw_module_availability(feature_distributions, sources),
        "feature_density": [],
        "feature_distributions": feature_distributions,
        "shared_modules": shared_modules,
        "all_modules": [row["module"] for row in feature_distributions],
        "compatibility_gate": {
            "status": "compatible",
            "comparison_mode": "raw_feature_distribution_only",
            "matched_cohort": False,
            "matched_cohort_ready": False,
            "descriptive_only": True,
            "inferential_statistics_allowed": False,
            "claim_level": "preview_not_reportable",
            "checks": [
                {
                    "id": "raw_database_count",
                    "passed": len(sources) >= 2,
                    "value": len(sources),
                    "minimum": 2,
                },
                {
                    "id": "feature_density_available",
                    "passed": bool(feature_distributions),
                    "feature_count": len(features),
                },
                {
                    "id": "no_patient_rows_returned",
                    "passed": True,
                    "basis": "aggregate_density_payload",
                },
            ],
            "reasons": [],
            "warnings": [],
            "comparable_metrics": ["feature_rows", "concepts_present"],
        },
        "provenance": {
            "computed_from": [
                "raw_icu_data_root",
                "easyicu.load_concepts",
                "MultiDatabaseDistribution",
                "bounded_feature_distribution_aggregates",
            ],
            "payload_scope": "raw_database_feature_density_aggregate_only",
            "data_root_hash": _hash(str(data_root)),
            "feature_scope": body.get("feature_scope") or "all_catalog",
            "max_patients": max_patients,
            "sample_size": sample_size,
            "inference": "blocked_until_numeric_evidence_gate",
        },
        "privacy": _privacy_payload(),
        "blocked_features": _blocked_features(),
    }


def crossdb_raw_root_scan(body: Dict[str, Any]) -> Dict[str, Any]:
    """Preflight a local raw ICU root without loading patient rows.

    The UI asks the user for one parent folder, but that folder still has to
    contain recognizable database subfolders. This scan makes the implicit
    alias matching visible before an expensive Cross-DB density job can start.
    """
    raw = str(body.get("data_root") or body.get("root") or "").strip()
    requested = _requested_raw_database_keys(body) or list(_RAW_DB_LABELS)
    base = {
        "mode": "real",
        "source_type": "raw_database_root",
        "selected_databases": requested,
        "selected_count": len(requested),
        "minimum_required": 2,
        "aliases": _raw_database_aliases_payload(),
        "privacy": _privacy_payload(),
    }
    if not raw:
        return {
            **base,
            "ok": False,
            "error": "raw_data_root_required",
            "hint": "Choose a local ICU data root containing database subfolders before running Cross-DB.",
            "detected": [],
            "missing_selected": _raw_missing_database_payload(requested),
            "unrecognized_folders": [],
            "detected_selected_count": 0,
            "runnable": False,
        }

    path = Path(raw).expanduser()
    if not path.exists() or not path.is_dir():
        return {
            **base,
            "ok": False,
            "error": "raw_data_root_not_found",
            "root_hash": _hash(raw),
            "hint": "The requested ICU data root does not exist or is not a directory.",
            "detected": [],
            "missing_selected": _raw_missing_database_payload(requested),
            "unrecognized_folders": [],
            "detected_selected_count": 0,
            "runnable": False,
        }

    root = _normalize_raw_root(path)
    detected = []
    recognized_top_folders = set()
    for db in _RAW_DB_LABELS:
        entry = _raw_database_scan_entry(root, db)
        if entry is None:
            continue
        entry["selected"] = db in requested
        detected.append(entry)
        top = str(entry.get("folder_name") or "").lower()
        if top:
            recognized_top_folders.add(top)

    detected_keys = {entry["key"] for entry in detected}
    detected_selected = [entry for entry in detected if entry["key"] in requested]
    missing = _raw_missing_database_payload(
        [db for db in requested if db not in detected_keys]
    )
    unrecognized = _unrecognized_raw_child_folders(root, recognized_top_folders)
    direct_unrecognized = (
        bool(_path_looks_like_database(str(root))) and not detected and root.name
    )
    if direct_unrecognized and root.name not in unrecognized:
        unrecognized.insert(0, root.name)
    runnable = len(detected_selected) >= 2
    return {
        **base,
        "ok": True,
        "root_hash": _hash(str(root)),
        "detected": detected,
        "detected_databases": [entry["key"] for entry in detected],
        "detected_selected_count": len(detected_selected),
        "missing_selected": missing,
        "unrecognized_folders": unrecognized[:20],
        "unrecognized_count": len(unrecognized),
        "runnable": runnable,
        "hint": (
            "Run is available after at least two selected database folders are recognized."
            if not runnable
            else "At least two selected database folders were recognized; the loader will validate files during run."
        ),
    }


def make_crossdb_raw_distribution_runner(body: Dict[str, Any]):
    """Build a background-job runner for raw Cross-DB density aggregation.

    The raw loader can touch multiple large local databases, so the native UI
    must not run it as a foreground request. Cancellation is cooperative: the
    runner checks before and after the expensive load phase, but it cannot
    forcibly interrupt a currently executing database read.
    """
    request_body = dict(body or {})

    def runner(job) -> Dict[str, Any]:
        def _cancelled(phase: str) -> Dict[str, Any] | None:
            if not getattr(job, "cancel_requested", False):
                return None
            return {
                "ok": False,
                "cancelled": True,
                "cancelled_at": phase,
                "cancel_reason": getattr(job, "cancel_reason", None)
                or "user_requested",
                "privacy": _privacy_payload(),
            }

        job.emit(
            {
                "type": "progress",
                "phase": "resolving",
                "message": "Resolving local raw database root and feature catalog.",
            }
        )
        _reject_unsupported_request(request_body)
        data_root = _resolve_raw_data_root(request_body)
        databases = _resolve_raw_databases(data_root, request_body)
        features = _resolve_raw_features(request_body)
        max_patients = _bounded_int(
            request_body.get("max_patients"), default=300, minimum=20, maximum=2000
        )
        sample_size = _bounded_int(
            request_body.get("sample_size"), default=1500, minimum=100, maximum=10000
        )
        cancelled = _cancelled("resolving")
        if cancelled is not None:
            return cancelled
        job.emit(
            {
                "type": "progress",
                "phase": "loading",
                "current": 0,
                "total": len(databases),
                "databases": databases,
                "feature_count": len(features),
                "max_patients": max_patients,
                "sample_size": sample_size,
                "message": (
                    f"Loading sampled aggregate feature distributions for "
                    f"{len(databases)} local databases and {len(features)} concepts "
                    f"(max {max_patients} entities/database, max {sample_size} "
                    "values/feature)."
                ),
            }
        )
        payload = crossdb_raw_distribution(request_body)
        cancelled = _cancelled("loading")
        if cancelled is not None:
            return cancelled
        job.emit(
            {
                "type": "progress",
                "phase": "finalizing",
                "current": len(payload.get("sources") or []),
                "total": len(databases),
                "feature_count": sum(
                    len(row.get("features") or [])
                    for row in payload.get("feature_distributions") or []
                ),
                "message": "Finalizing aggregate-only Cross-DB density payload.",
            }
        )
        return payload

    return runner


def crossdb_demo_distribution(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return legacy-seeded Cross-DB demo density aggregates.

    The old Streamlit Cross-DB demo used clinically-shaped synthetic feature
    frames per database before rendering the N×N distribution grid. This keeps
    that useful data contract but returns only aggregate density points, so the
    native UI does not fabricate curve shapes in JavaScript and does not expose
    row-level demo records as if they were real data.
    """
    _reject_unsupported_request(body)
    databases = _resolve_demo_databases(body)
    features = _resolve_demo_features(body)
    records_per_feature = _bounded_int(
        body.get("records_per_feature"),
        default=_DEMO_RECORDS_PER_FEATURE,
        minimum=24,
        maximum=1000,
    )
    if len(databases) < 2:
        raise CrossdbReviewError(
            {
                "error": "need_two_demo_databases",
                "mode": "demo",
                "source_type": "legacy_simulated_multidb_feature_frames",
                "requested_databases": databases,
                "privacy": _privacy_payload(),
            }
        )

    frames = _generate_demo_multidb_feature_frames(
        databases=databases,
        features=features,
        records_per_feature=records_per_feature,
    )
    loaded = {db: frame for db, frame in frames.items() if _raw_frame_has_values(frame)}
    if len(loaded) < 2:
        raise CrossdbReviewError(
            {
                "error": "loaded_fewer_than_two_demo_databases",
                "mode": "demo",
                "source_type": "legacy_simulated_multidb_feature_frames",
                "requested_databases": databases,
                "loaded_databases": sorted(loaded),
                "feature_count": len(features),
                "privacy": _privacy_payload(),
            }
        )

    sources = [_demo_source_summary(db, frame) for db, frame in loaded.items()]
    feature_distributions = _raw_feature_distribution_payload(loaded, features)
    shared_modules = [
        row["module"]
        for row in feature_distributions
        if row.get("shared_feature_count")
    ]
    return {
        "ok": True,
        "mode": "demo",
        "demo": True,
        "source_type": "legacy_simulated_multidb_feature_frames",
        "source_count": len(sources),
        "sources": sources,
        "rows": _raw_comparison_rows(loaded, features),
        "availability": _raw_module_availability(feature_distributions, sources),
        "feature_density": [],
        "feature_distributions": feature_distributions,
        "shared_modules": shared_modules,
        "all_modules": [row["module"] for row in feature_distributions],
        "compatibility_gate": {
            "status": "compatible",
            "comparison_mode": "seeded_demo_distribution_only",
            "matched_cohort": False,
            "matched_cohort_ready": False,
            "descriptive_only": True,
            "inferential_statistics_allowed": False,
            "claim_level": "demo_not_reportable",
            "checks": [
                {
                    "id": "demo_database_count",
                    "passed": len(sources) >= 2,
                    "value": len(sources),
                    "minimum": 2,
                },
                {
                    "id": "legacy_demo_feature_specs",
                    "passed": bool(feature_distributions),
                    "feature_count": len(features),
                },
                {
                    "id": "no_row_payload_returned",
                    "passed": True,
                    "basis": "aggregate_density_payload",
                },
            ],
            "reasons": [],
            "warnings": [
                {
                    "id": "simulated_not_real_database",
                    "message": "Demo curves come from legacy seeded feature frames, not user ICU databases.",
                }
            ],
            "comparable_metrics": ["feature_rows", "concepts_present"],
        },
        "provenance": {
            "computed_from": [
                "legacy_streamlit_generate_mock_multidb_data",
                "seeded_clinical_feature_specs",
                "bounded_feature_distribution_aggregates",
            ],
            "payload_scope": "simulated_multidb_density_aggregate_only",
            "feature_scope": "legacy_demo_supported_features",
            "records_per_feature": records_per_feature,
            "inference": "demo_not_reportable",
        },
        "privacy": _privacy_payload(),
        "blocked_features": _blocked_features(),
    }


def _resolve_demo_databases(body: Dict[str, Any]) -> List[str]:
    requested = body.get("databases") or body.get("database_keys") or []
    if isinstance(requested, str):
        requested = [item.strip() for item in requested.split(",")]
    if not requested:
        return list(_DEMO_MULTIDB_DATABASES)
    out = []
    for item in requested:
        db = _normalize_database_key(str(item))
        if db in _DEMO_MULTIDB_DATABASES and db not in out:
            out.append(db)
    return out


def _resolve_demo_features(body: Dict[str, Any]) -> List[str]:
    requested = body.get("features") or body.get("concepts")
    if isinstance(requested, str):
        requested = [item.strip() for item in requested.split(",")]
    catalog_features = _catalog_features_in_order()
    catalog_set = set(catalog_features)
    if isinstance(requested, list) and requested:
        candidates = [str(item).strip() for item in requested if str(item).strip()]
    elif (
        str(body.get("feature_scope") or "").strip() == "legacy_demo_supported_features"
    ):
        supported = _legacy_demo_supported_features()
        candidates = [feature for feature in catalog_features if feature in supported]
    else:
        candidates = catalog_features

    max_features = _bounded_int(
        body.get("max_features"),
        default=len(candidates) or len(catalog_features),
        minimum=1,
        maximum=len(catalog_features),
    )
    out = []
    seen = set()
    for feature in candidates:
        if feature in catalog_set and feature not in seen:
            seen.add(feature)
            out.append(feature)
        if len(out) >= max_features:
            break
    if not out:
        raise CrossdbReviewError(
            {
                "error": "no_supported_demo_features_requested",
                "supported_feature_count": len(catalog_features),
                "privacy": _privacy_payload(),
            }
        )
    return out


def _generate_demo_multidb_feature_frames(
    *,
    databases: List[str],
    features: List[str],
    records_per_feature: int,
) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    result: Dict[str, Any] = {}
    for db in databases:
        specs = _DEMO_MULTIDB_FEATURE_SPECS.get(db) or {}
        rows = []
        for feature in features:
            mean, std, low, high, integer_like = _demo_feature_profile(
                db, feature, specs.get(feature)
            )
            if integer_like and low == 0 and high == 1:
                p = max(0.02, min(0.98, float(mean)))
                values = rng.binomial(1, p, int(records_per_feature))
            else:
                values = rng.normal(float(mean), float(std), int(records_per_feature))
                values = np.clip(values, low, high)
            if integer_like:
                values = np.rint(values).astype(int)
            rows.extend({"concept": feature, "value": float(value)} for value in values)
        result[db] = pd.DataFrame(rows, columns=["concept", "value"])
    return result


def _legacy_demo_supported_features() -> set:
    return set().union(*[set(specs) for specs in _DEMO_MULTIDB_FEATURE_SPECS.values()])


def _catalog_features_in_order() -> List[str]:
    out: List[str] = []
    seen = set()
    for features in concept_catalog.CONCEPT_GROUPS_INTERNAL.values():
        for feature in features:
            if feature in concept_catalog.CONCEPT_DICTIONARY and feature not in seen:
                seen.add(feature)
                out.append(feature)
    for feature in concept_catalog.CONCEPT_DICTIONARY:
        if feature not in seen:
            seen.add(feature)
            out.append(feature)
    return out


def _demo_feature_profile(db: str, feature: str, explicit: Any = None) -> tuple:
    """Return deterministic demo mean/std/bounds for any catalog concept."""
    if explicit:
        mean, std = explicit
        low, high = _DEMO_FEATURE_BOUNDS.get(
            feature, (float(mean) - 4 * float(std), float(mean) + 4 * float(std))
        )
        integer_like = (
            feature in {"sex", "death", "adm"}
            or feature.startswith(("sofa", "aki_stage", "sep3_"))
            or _catalog_unit(feature).lower() == "boolean"
        )
        return float(mean), float(std), float(low), float(high), integer_like

    low, high, integer_like = _demo_feature_bounds(feature)
    span = max(float(high) - float(low), 1.0)
    digest = hashlib.sha256(f"{db}:{feature}".encode("utf-8")).digest()
    feature_offset = int.from_bytes(digest[:2], "big") / 65535.0
    db_index = _DEMO_MULTIDB_DATABASES.index(db) if db in _DEMO_MULTIDB_DATABASES else 0
    db_offset = (db_index - (len(_DEMO_MULTIDB_DATABASES) - 1) / 2) * 0.035

    if integer_like and low == 0 and high == 1:
        mean = min(0.92, max(0.03, 0.08 + feature_offset * 0.68 + db_offset))
        return mean, 0.35, low, high, True

    center = 0.28 + feature_offset * 0.48 + db_offset
    mean = float(low) + span * min(0.82, max(0.12, center))
    std = max(span * (0.055 + (digest[2] / 255.0) * 0.055), 0.05)
    return mean, std, float(low), float(high), integer_like


def _demo_feature_bounds(feature: str) -> tuple:
    if feature in _DEMO_FEATURE_BOUNDS:
        low, high = _DEMO_FEATURE_BOUNDS[feature]
        return (
            float(low),
            float(high),
            feature.startswith(("sofa", "aki_stage", "sep3_")),
        )

    unit = _catalog_unit(feature).lower()
    if (
        unit == "boolean"
        or feature.endswith("_ind")
        or feature.endswith("60")
        or feature.startswith(("sep3_", "infection_"))
    ):
        return 0.0, 1.0, True
    if unit.startswith("0-"):
        try:
            return 0.0, float(unit.split("-", 1)[1]), True
        except ValueError:
            pass
    if "datetime" in unit or "time" in feature:
        return 0.0, 168.0, False
    if "years" in unit or feature == "age":
        return 18.0, 95.0, False
    if "kg/m" in unit or feature == "bmi":
        return 12.0, 60.0, False
    if unit in {"", "score"} and feature in {"sex", "adm", "avpu"}:
        return 0.0, 3.0, True
    if "hours" in unit or feature.endswith("_dur"):
        return 0.0, 96.0, False
    if "days" in unit or feature.startswith("los_"):
        return 0.0, 30.0, False
    if "ml/kg/h" in unit:
        return 0.0, 4.0, False
    if "ml" in unit:
        return 0.0, 5000.0, False
    if "mcg" in unit or "units" in unit:
        return 0.0, 3.0, False
    if "%" in unit:
        return 0.0, 100.0, False
    return 0.0, 100.0, False


def _catalog_unit(feature: str) -> str:
    meta = concept_catalog.CONCEPT_DICTIONARY.get(feature)
    if isinstance(meta, (list, tuple)) and len(meta) >= 3:
        return str(meta[2] or "")
    return ""


def _resolve_raw_data_root(body: Dict[str, Any]) -> Path:
    raw = str(body.get("data_root") or body.get("root") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if path.exists() and path.is_dir():
            return _normalize_raw_root(path)
        raise CrossdbReviewError(
            {
                "error": "raw_data_root_not_found",
                "source_type": "raw_database_root",
                "root_hash": _hash(raw),
                "hint": "The requested ICU data root does not exist or is not a directory.",
                "privacy": _privacy_payload(),
            }
        )

    candidates = []
    env_root = str(__import__("os").environ.get("EASYICU_DATA_PATH") or "").strip()
    if env_root:
        candidates.append(env_root)
    candidates.extend(_COMMON_RAW_ROOT_CANDIDATES)

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.exists() or not path.is_dir():
            continue
        normalized = _normalize_raw_root(path)
        if len(_detect_raw_databases(normalized)) >= 2:
            return normalized
        if len(_detect_raw_databases(path)) >= 2:
            return path
    raise CrossdbReviewError(
        {
            "error": "raw_data_root_not_found",
            "source_type": "raw_database_root",
            "hint": "Provide a local ICU data root containing at least two database folders.",
            "privacy": _privacy_payload(),
        }
    )


def _normalize_raw_root(path: Path) -> Path:
    aliases = {alias for names in DATABASE_ALIASES.values() for alias in names}
    name = path.name.lower()
    if name in aliases or any(alias in name for alias in aliases):
        parent = path.parent
        if parent != path and len(_detect_raw_databases(parent)) >= 2:
            return parent
    return path


def _resolve_raw_databases(data_root: Path, body: Dict[str, Any]) -> List[str]:
    requested = _requested_raw_database_keys(body)
    if requested:
        return [db for db in requested if _raw_database_exists(data_root, db)]
    return _detect_raw_databases(data_root)


def _requested_raw_database_keys(body: Dict[str, Any]) -> List[str]:
    requested = body.get("databases") or body.get("database_keys") or []
    if isinstance(requested, str):
        requested = [item.strip() for item in requested.split(",")]
    allowed = set(_RAW_DB_LABELS)
    if requested:
        normalized = []
        for item in requested:
            db = _normalize_database_key(str(item))
            if db in allowed and db not in normalized:
                normalized.append(db)
        return normalized
    return []


def _raw_database_aliases_payload() -> Dict[str, Dict[str, Any]]:
    return {
        db: {
            "label": label,
            "aliases": [str(alias) for alias in DATABASE_ALIASES.get(db, [db])],
        }
        for db, label in _RAW_DB_LABELS.items()
    }


def _raw_missing_database_payload(databases: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "key": db,
            "label": _RAW_DB_LABELS.get(db, db),
            "aliases": [str(alias) for alias in DATABASE_ALIASES.get(db, [db])],
        }
        for db in databases
    ]


def _normalize_database_key(value: str) -> str:
    raw = value.strip().lower()
    try:
        return canonical_database_key(raw)
    except KeyError:
        return raw


def _detect_raw_databases(root: Path) -> List[str]:
    found = []
    for db in _RAW_DB_LABELS:
        if _raw_database_exists(root, db):
            found.append(db)
    return found


def _raw_database_exists(root: Path, db: str) -> bool:
    return _raw_database_resolved_path(root, db) is not None


def _raw_database_resolved_path(root: Path, db: str) -> Path | None:
    try:
        resolved = Path(find_database_path(str(root), db)).expanduser()
    except Exception:
        return None
    if not resolved.exists() or not resolved.is_dir():
        return None
    try:
        same = resolved.resolve() == root.resolve()
    except OSError:
        same = resolved == root
    if not same:
        return resolved
    # A direct database path is valid only for the matching database key; do
    # not let a single raw DB folder masquerade as every supported database.
    name = root.name.lower()
    aliases = DATABASE_ALIASES.get(db, [db])
    if name in aliases or any(alias in name for alias in aliases):
        return resolved
    return None


def _raw_database_scan_entry(root: Path, db: str) -> Dict[str, Any] | None:
    resolved = _raw_database_resolved_path(root, db)
    if resolved is None:
        return None
    folder_name = resolved.name
    version_folder = None
    try:
        rel = resolved.resolve().relative_to(root.resolve())
        if rel.parts:
            folder_name = rel.parts[0]
            if len(rel.parts) > 1:
                version_folder = rel.parts[-1]
    except (OSError, ValueError):
        pass
    payload = {
        "key": db,
        "label": _RAW_DB_LABELS.get(db, db),
        "folder_name": folder_name,
        "path_hash": _hash(str(resolved)),
        "aliases": [str(alias) for alias in DATABASE_ALIASES.get(db, [db])],
    }
    if version_folder and version_folder != folder_name:
        payload["version_folder"] = version_folder
    return payload


def _unrecognized_raw_child_folders(
    root: Path, recognized_top_folders: set[str]
) -> List[str]:
    try:
        children = sorted(
            [child for child in root.iterdir() if child.is_dir()],
            key=lambda child: child.name.lower(),
        )
    except OSError:
        return []
    unrecognized = []
    for child in children:
        name = child.name
        if name.lower() in recognized_top_folders:
            continue
        if _raw_child_name_matches_known_alias(name):
            continue
        unrecognized.append(name)
    return unrecognized


def _raw_child_name_matches_known_alias(name: str) -> bool:
    lower = name.lower()
    normalized = lower.replace("_", "-")
    for aliases in DATABASE_ALIASES.values():
        for alias in aliases:
            alias_norm = str(alias).lower().replace("_", "-")
            if lower == alias_norm or normalized == alias_norm:
                return True
            if alias_norm and (
                alias_norm in normalized or normalized.startswith(alias_norm)
            ):
                return True
    return False


def _resolve_raw_features(body: Dict[str, Any]) -> List[str]:
    requested = body.get("features") or body.get("concepts")
    if isinstance(requested, str):
        requested = [item.strip() for item in requested.split(",")]
    if isinstance(requested, list) and requested:
        features = [str(item).strip() for item in requested if str(item).strip()]
    elif str(body.get("feature_scope") or "").strip() == "all_catalog":
        features = _catalog_features_in_order()
    else:
        min_coverage = _bounded_int(
            body.get("coverage_min"), default=2, minimum=1, maximum=6
        )
        features = [
            concept
            for concept in concept_catalog.CONCEPT_DICTIONARY
            if int(concept_catalog.CONCEPT_DB_COVERAGE.get(concept, 0)) >= min_coverage
        ]
    max_features = _bounded_int(
        body.get("max_features"),
        default=len(features) or len(concept_catalog.CONCEPT_DICTIONARY),
        minimum=1,
        maximum=len(concept_catalog.CONCEPT_DICTIONARY),
    )
    seen = set()
    out = []
    for feature in features:
        if feature in concept_catalog.CONCEPT_DICTIONARY and feature not in seen:
            seen.add(feature)
            out.append(feature)
        if len(out) >= max_features:
            break
    if not out:
        raise CrossdbReviewError(
            {"error": "no_supported_features_requested", "privacy": _privacy_payload()}
        )
    return out


def _load_raw_feature_data(
    *,
    data_root: str,
    concepts: List[str],
    databases: List[str],
    max_patients: int,
    sample_size: int,
) -> Dict[str, Any]:
    import pandas as pd
    from easyicu import load_concepts
    from easyicu.cohort_visualization import MultiDatabaseDistribution

    mdd = MultiDatabaseDistribution(data_root=data_root, language="en")
    result: Dict[str, Any] = {}
    chunk_size = 24
    for db in databases:
        db_path = mdd._get_db_path(db)
        if not db_path.exists():
            continue
        chunks = [
            concepts[i : i + chunk_size] for i in range(0, len(concepts), chunk_size)
        ]
        all_data = []
        for chunk in chunks:
            try:
                frame = load_concepts(
                    concepts=chunk,
                    database=db,
                    data_path=str(db_path),
                    max_patients=max_patients,
                    require_bounded_sample=True,
                    verbose=False,
                )
                all_data.extend(_wide_concepts_to_long(frame, chunk, sample_size))
            except Exception:
                for concept in chunk:
                    try:
                        frame = load_concepts(
                            concepts=[concept],
                            database=db,
                            data_path=str(db_path),
                            max_patients=max_patients,
                            require_bounded_sample=True,
                            verbose=False,
                        )
                        all_data.extend(
                            _wide_concepts_to_long(frame, [concept], sample_size)
                        )
                    except Exception:
                        continue
        if all_data:
            result[db] = pd.concat(all_data, ignore_index=True)
    return result


def _wide_concepts_to_long(
    frame: Any, concepts: List[str], sample_size: int
) -> List[Any]:
    import pandas as pd

    if frame is None or getattr(frame, "empty", True):
        return []
    out = []
    for concept in concepts:
        if concept not in getattr(frame, "columns", []):
            continue
        values = frame[concept].dropna()
        if len(values) > sample_size:
            values = values.sample(n=sample_size, random_state=42)
        if len(values) > 0:
            out.append(pd.DataFrame({"concept": concept, "value": values.to_numpy()}))
    return out


def _raw_frame_has_values(frame: Any) -> bool:
    return (
        hasattr(frame, "columns")
        and "concept" in frame.columns
        and "value" in frame.columns
        and not frame.empty
    )


def _raw_source_summary(db: str, frame: Any, data_root: Path) -> Dict[str, Any]:
    concepts = sorted(str(item) for item in frame["concept"].dropna().unique().tolist())
    return {
        "id": db,
        "label": _RAW_DB_LABELS.get(db, db),
        "database": db,
        "path_hash": _hash(str(data_root / db)),
        "summary": {
            "feature_rows": int(len(frame)),
            "concepts_present": len(concepts),
            "modules": len(_modules_for_features(concepts)),
            "total_records": int(len(frame)),
        },
    }


def _demo_source_summary(db: str, frame: Any) -> Dict[str, Any]:
    concepts = sorted(str(item) for item in frame["concept"].dropna().unique().tolist())
    return {
        "id": db,
        "label": _RAW_DB_LABELS.get(db, db),
        "database": db,
        "path_hash": _hash(f"legacy-demo:{db}"),
        "summary": {
            "feature_rows": int(len(frame)),
            "concepts_present": len(concepts),
            "modules": len(_modules_for_features(concepts)),
            "total_records": int(len(frame)),
        },
    }


def _raw_comparison_rows(
    frames: Dict[str, Any], features: List[str]
) -> List[Dict[str, Any]]:
    dbs = list(frames)
    rows: List[Dict[str, Any]] = []
    feature_counts = [
        int(frames[db]["concept"].nunique()) if _raw_frame_has_values(frames[db]) else 0
        for db in dbs
    ]
    record_counts = [int(len(frames[db])) for db in dbs]
    rows.append(
        {
            "key": "feature_rows",
            "label": "Feature rows",
            "values": record_counts,
            "delta": (
                max(record_counts) - min(record_counts)
                if len(record_counts) >= 2
                else None
            ),
            "comparison": "descriptive_range",
        }
    )
    rows.append(
        {
            "key": "concepts_present",
            "label": "Concepts present",
            "values": feature_counts,
            "delta": (
                max(feature_counts) - min(feature_counts)
                if len(feature_counts) >= 2
                else None
            ),
            "comparison": "descriptive_range",
        }
    )
    for feature in features[:8]:
        values = []
        for db in dbs:
            subset = frames[db].loc[frames[db]["concept"] == feature, "value"]
            summary = _summarize_feature_distribution(subset)
            if summary.get("kind") == "numeric" and summary.get("non_null"):
                import pandas as pd

                numeric = pd.to_numeric(subset, errors="coerce").dropna()
                values.append(
                    round(float(numeric.median()), 3) if not numeric.empty else None
                )
            else:
                values.append(None)
        numeric_values = [
            float(value) for value in values if isinstance(value, (int, float))
        ]
        if numeric_values:
            rows.append(
                {
                    "key": f"{feature}_median",
                    "label": f"{feature} median",
                    "values": values,
                    "delta": (
                        round(max(numeric_values) - min(numeric_values), 3)
                        if len(numeric_values) >= 2
                        else None
                    ),
                    "comparison": "descriptive_range",
                }
            )
    return rows


def _raw_feature_distribution_payload(
    frames: Dict[str, Any], features: List[str]
) -> List[Dict[str, Any]]:
    module_map = _feature_module_map()
    dbs = list(frames)
    by_module: Dict[str, List[str]] = {}
    for feature in features:
        by_module.setdefault(module_map.get(feature, "other"), []).append(feature)

    out: List[Dict[str, Any]] = []
    for module, module_features in by_module.items():
        feature_rows = []
        for feature in module_features:
            values = []
            present_count = 0
            for db in dbs:
                subset = frames[db].loc[frames[db]["concept"] == feature, "value"]
                summary = _summarize_feature_distribution(subset)
                present = (
                    summary.get("kind") not in {"empty", "missing"}
                    and int(summary.get("non_null") or 0) > 0
                )
                if present:
                    present_count += 1
                values.append(
                    {
                        "source": _RAW_DB_LABELS.get(db, db),
                        "database": db,
                        "present": present,
                        **summary,
                    }
                )
            if present_count:
                feature_rows.append(
                    {
                        "feature": feature,
                        "label": concept_catalog.CONCEPT_DICTIONARY.get(
                            feature, (feature, feature, "")
                        )[0],
                        "shared": present_count == len(dbs),
                        "present_count": present_count,
                        "values": values,
                    }
                )
        if feature_rows:
            out.append(
                {
                    "module": module,
                    "source_count": len(dbs),
                    "feature_count": len(feature_rows),
                    "shared_feature_count": sum(
                        1 for row in feature_rows if row["shared"]
                    ),
                    "features": feature_rows,
                }
            )
    return out


def _raw_module_availability(
    feature_distributions: List[Dict[str, Any]], sources: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    labels = [str(source.get("label") or source.get("database")) for source in sources]
    out = []
    for module in feature_distributions:
        out.append(
            {
                "module": module.get("module"),
                "present_count": sum(1 for _ in labels),
                "source_count": len(labels),
                "shared": module.get("shared_feature_count", 0) > 0,
                "median_coverage_pct": None,
                "values": [
                    {
                        "source": label,
                        "present": True,
                        "coverage_pct": None,
                        "quality_status": "aggregate_density_only",
                    }
                    for label in labels
                ],
            }
        )
    return out


def _feature_module_map() -> Dict[str, str]:
    out = {}
    for module, features in concept_catalog.CONCEPT_GROUPS_INTERNAL.items():
        for feature in features:
            out.setdefault(feature, module)
    return out


def _modules_for_features(features: List[str]) -> List[str]:
    module_map = _feature_module_map()
    return sorted({module_map.get(feature, "other") for feature in features})


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    return max(minimum, min(maximum, out))


def _reject_unsupported_request(body: Dict[str, Any]) -> None:
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else {}
    requested_filters = [
        {
            "id": str(key),
            "reason": _UNSUPPORTED_FILTERS.get(
                str(key), "Cross-DB Stage18 does not accept row-level filters."
            ),
        }
        for key, value in filters.items()
        if _truthy_request(value)
    ]
    if _truthy_request(body.get("matched_cohort")):
        requested_filters.append(
            {"id": "matched_cohort", "reason": _UNSUPPORTED_FILTERS["matched_cohort"]}
        )
    comparison = body.get("comparison")
    if isinstance(comparison, dict) and _truthy_request(
        comparison.get("matched_cohort")
    ):
        requested_filters.append(
            {"id": "matched_cohort", "reason": _UNSUPPORTED_FILTERS["matched_cohort"]}
        )
    if requested_filters:
        raise CrossdbReviewError(
            {
                "error": "unsupported_filter",
                "unsupported": requested_filters,
                "supported_scope": "registered_source_crossdb_aggregates_only",
            }
        )

    stats = body.get("statistics") or body.get("stats") or []
    if isinstance(stats, str):
        stats = [stats]
    requested_stats = [
        {
            "id": str(item),
            "reason": _UNSUPPORTED_STATISTICS.get(
                str(item),
                "Requested statistic is not supported by the Stage18 aggregate endpoint.",
            ),
        }
        for item in stats
        if _truthy_request(item)
    ]
    if requested_stats:
        raise CrossdbReviewError(
            {
                "error": "unsupported_statistic",
                "unsupported": requested_stats,
                "supported_scope": "descriptive_crossdb_aggregate_only",
            }
        )


def _resolve_registered_sources(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    registry = source_store.load_registry()
    sources = [
        s for s in registry.get("sources") or [] if isinstance(s, dict) and s.get("ok")
    ]
    by_path = {
        _norm_path(str(s.get("path") or "")): s for s in sources if s.get("path")
    }
    requested = body.get("paths") or body.get("source_paths")

    if requested is not None:
        if not isinstance(requested, list):
            raise CrossdbReviewError({"error": "paths_must_be_list"})
        selected: List[Dict[str, Any]] = []
        seen = set()
        for raw in requested:
            norm = _norm_path(str(raw or ""))
            if not norm or norm in seen:
                continue
            seen.add(norm)
            source = by_path.get(norm)
            if source is None:
                raise CrossdbReviewError(
                    {"error": "source_not_registered", "path_hash": _hash(norm)}
                )
            selected.append(source)
    else:
        selected = []
        seen = set()
        for raw in registry.get("crossdb_paths") or []:
            norm = _norm_path(str(raw or ""))
            if not norm or norm in seen:
                continue
            seen.add(norm)
            source = by_path.get(norm)
            if source is not None:
                selected.append(source)

    if len(selected) < 2:
        raise CrossdbReviewError(
            {
                "error": "need_two_exports",
                "source_count": len(selected),
                "sources": [_safe_registered_source(source) for source in selected],
                "privacy": _privacy_payload(),
            }
        )
    return selected


def _source_aggregate(
    source: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    safe_source = dict(payload.get("source") or _safe_registered_source(source))
    summary = payload.get("summary") or {}
    coverage = payload.get("coverage") or []
    quality = payload.get("quality") or {}
    desc = dataio.describe_export_source(str(source.get("path") or ""))
    modules = sorted({str(row.get("module")) for row in coverage if row.get("module")})
    if not modules:
        modules = sorted(
            str(module) for module in (source.get("modules") or []) if module
        )
    module_coverage = {
        str(row.get("module")): {
            "rows": int(row.get("rows") or 0),
            "coverage_pct": row.get("coverage_pct"),
            "quality_status": row.get("quality_status") or "unknown",
        }
        for row in coverage
        if row.get("module")
    }
    return {
        "id": safe_source.get("id"),
        "label": safe_source.get("label"),
        "path_hash": safe_source.get("path_hash")
        or _hash(str(source.get("path") or "")),
        "database": safe_source.get("database"),
        "generated": safe_source.get("generated"),
        "summary": {
            "cohort_size": summary.get("cohort_size"),
            "modules": summary.get("modules"),
            "file_count": summary.get("file_count"),
            "total_records": summary.get("total_records"),
            "mortality_pct": summary.get("mortality_pct"),
            "age_mean": (summary.get("age") or {}).get("mean"),
            "age_median": (summary.get("age") or {}).get("median"),
            "female_pct": (summary.get("sex") or {}).get("female_pct"),
            "sofa2_median": (summary.get("sofa2") or {}).get("median"),
            "los_median": (summary.get("los_icu_days") or {}).get("median"),
            "sepsis_pct": summary.get("sepsis_pct"),
            "coverage_median_pct": quality.get("median_coverage_pct"),
            "quality_watchlist_count": quality.get("watchlist_count"),
        },
        "modules": modules,
        "module_coverage": module_coverage,
        "feature_density": _source_feature_density(
            desc, summary.get("cohort_size"), module_coverage
        ),
        "feature_distributions": _source_feature_distributions(desc),
        "quality": {
            "modules_ok": quality.get("modules_ok"),
            "modules_warn": quality.get("modules_warn"),
            "modules_bad": quality.get("modules_bad"),
            "modules_neutral": quality.get("modules_neutral"),
            "modules_unknown": quality.get("modules_unknown"),
            "watchlist_count": quality.get("watchlist_count"),
            "median_coverage_pct": quality.get("median_coverage_pct"),
        },
    }


def _compatibility_gate(
    sources: List[Dict[str, Any]],
    shared_modules: List[str],
    all_modules: List[str],
) -> Dict[str, Any]:
    shared = set(shared_modules)
    checks: List[Dict[str, Any]] = []
    reasons: List[Dict[str, Any]] = []

    enough_sources = len(sources) >= 2
    checks.append(
        {
            "id": "source_count",
            "passed": enough_sources,
            "value": len(sources),
            "minimum": 2,
        }
    )
    if not enough_sources:
        reasons.append(
            {
                "id": "need_two_exports",
                "detail": "At least two registered exports are required.",
            }
        )

    denominators = [
        {
            "label": source.get("label"),
            "cohort_size": (source.get("summary") or {}).get("cohort_size"),
        }
        for source in sources
    ]
    denominator_ok = all(
        isinstance(row["cohort_size"], (int, float)) and row["cohort_size"] > 0
        for row in denominators
    )
    checks.append(
        {"id": "denominator_present", "passed": denominator_ok, "sources": denominators}
    )
    if not denominator_ok:
        reasons.append({"id": "missing_denominator", "sources": denominators})

    missing_core = sorted(_REQUIRED_CORE_MODULES - shared)
    checks.append(
        {
            "id": "core_modules_shared",
            "passed": not missing_core,
            "required_modules": sorted(_REQUIRED_CORE_MODULES),
            "shared_modules": shared_modules,
            "missing_modules": missing_core,
        }
    )
    if missing_core:
        reasons.append(
            {
                "id": "core_modules_not_shared",
                "missing_shared_modules": missing_core,
                "sources": [
                    {
                        "label": source.get("label"),
                        "missing_core_modules": sorted(
                            _REQUIRED_CORE_MODULES - set(source.get("modules") or [])
                        ),
                    }
                    for source in sources
                ],
            }
        )

    comparable_metrics = [
        "cohort_size",
        "modules",
        "total_records",
        "age_mean",
        "age_median",
        "female_pct",
        "mortality_pct",
        "coverage_median_pct",
    ]
    if "sepsis3_sofa2" in shared:
        comparable_metrics.append("sepsis_pct")
    if "sofa2_score" in shared:
        comparable_metrics.append("sofa2_median")
    if "outcome" in shared:
        comparable_metrics.append("los_median")

    warnings: List[Dict[str, Any]] = []
    missing_optional = sorted(
        (_OPTIONAL_COMPARISON_MODULES & set(all_modules)) - shared
    )
    if missing_optional:
        warnings.append(
            {
                "id": "optional_modules_not_shared",
                "modules": missing_optional,
                "effect": "dependent descriptive metrics are omitted from comparison rows",
            }
        )

    compatible = all(check["passed"] for check in checks)
    return {
        "status": "compatible" if compatible else "incompatible",
        "comparison_mode": "descriptive_only",
        "matched_cohort": False,
        "matched_cohort_ready": False,
        "descriptive_only": True,
        "inferential_statistics_allowed": False,
        "claim_level": "preview_not_reportable",
        "checks": checks,
        "reasons": reasons,
        "warnings": warnings,
        "comparable_metrics": comparable_metrics if compatible else [],
    }


def _comparison_rows(
    sources: List[Dict[str, Any]], gate: Dict[str, Any]
) -> List[Dict[str, Any]]:
    comparable = set(gate.get("comparable_metrics") or [])
    rows: List[Dict[str, Any]] = []
    for key, label, digits, dependency in [
        ("cohort_size", "Cohort size", 0, "demographics"),
        ("modules", "Modules", 0, None),
        ("total_records", "Records", 0, None),
        ("age_mean", "Mean age", 1, "demographics"),
        ("age_median", "Median age", 1, "demographics"),
        ("female_pct", "Female %", 1, "demographics"),
        ("mortality_pct", "Mortality %", 1, "outcome"),
        ("los_median", "Median ICU LOS", 1, "outcome"),
        ("sepsis_pct", "Sepsis-3 %", 1, "sepsis3_sofa2"),
        ("sofa2_median", "Median SOFA-2", 1, "sofa2_score"),
        ("coverage_median_pct", "Median module coverage %", 1, None),
    ]:
        if dependency and key not in comparable:
            continue
        values = [(source.get("summary") or {}).get(key) for source in sources]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        delta = (
            round(max(numeric) - min(numeric), digits) if len(numeric) >= 2 else None
        )
        rows.append(
            {
                "key": key,
                "label": label,
                "values": values,
                "delta": delta,
                "comparison": "descriptive_range",
            }
        )
    return rows


def _module_availability(
    sources: List[Dict[str, Any]], all_modules: List[str]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for module in all_modules:
        values = []
        present_count = 0
        coverage_values: List[float] = []
        for source in sources:
            module_info = (source.get("module_coverage") or {}).get(module)
            present = module_info is not None
            if present:
                present_count += 1
                coverage = module_info.get("coverage_pct")
                if isinstance(coverage, (int, float)):
                    coverage_values.append(float(coverage))
            values.append(
                {
                    "source": source.get("label"),
                    "present": present,
                    "coverage_pct": (
                        module_info.get("coverage_pct") if module_info else None
                    ),
                    "quality_status": (
                        module_info.get("quality_status") if module_info else "missing"
                    ),
                }
            )
        out.append(
            {
                "module": module,
                "present_count": present_count,
                "source_count": len(sources),
                "shared": present_count == len(sources),
                "median_coverage_pct": dataio._median(coverage_values),
                "values": values,
            }
        )
    return out


def _source_feature_density(
    desc: Dict[str, Any],
    cohort_size: Any,
    module_coverage: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build manifest-level feature density without returning patient rows.

    ``density_per_100_entities`` is based on module file row counts divided by
    the cohort denominator. It intentionally does not claim feature-level
    non-null coverage, because that would require a separate bounded feature
    scan. The frontend labels this as record density.
    """
    out: Dict[str, Dict[str, Any]] = {}
    denominator = (
        int(cohort_size) if isinstance(cohort_size, int) and cohort_size > 0 else None
    )
    for item in desc.get("files") or []:
        module = str(item.get("module") or "")
        if not module:
            continue
        rows = int(item.get("rows") or 0)
        columns = [str(col) for col in (item.get("columns") or [])]
        features = [col for col in columns if _is_feature_column(col)]
        if not features:
            continue
        module_info = module_coverage.get(module) or {}
        density = round(rows / denominator * 100, 1) if denominator else None
        out[module] = {
            "module": module,
            "row_count": rows,
            "cohort_size": denominator,
            "coverage_pct": module_info.get("coverage_pct"),
            "quality_status": module_info.get("quality_status") or "unknown",
            "feature_count": len(features),
            "features": [
                {
                    "feature": feature,
                    "records": rows,
                    "density_per_100_entities": density,
                    "coverage_pct": module_info.get("coverage_pct"),
                }
                for feature in features
            ],
        }
    return out


def _feature_density_payload(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    modules = sorted(
        {
            module
            for source in sources
            for module in (source.get("feature_density") or {}).keys()
        }
    )
    out: List[Dict[str, Any]] = []
    for module in modules:
        features = sorted(
            {
                str(feature.get("feature"))
                for source in sources
                for feature in (
                    (source.get("feature_density") or {}).get(module) or {}
                ).get("features", [])
                if feature.get("feature")
            }
        )
        if not features:
            continue
        feature_rows: List[Dict[str, Any]] = []
        for feature in features:
            values = []
            present_count = 0
            for source in sources:
                module_payload = (source.get("feature_density") or {}).get(module) or {}
                hit = next(
                    (
                        item
                        for item in module_payload.get("features") or []
                        if item.get("feature") == feature
                    ),
                    None,
                )
                present = hit is not None
                if present:
                    present_count += 1
                values.append(
                    {
                        "source": source.get("label"),
                        "present": present,
                        "records": hit.get("records") if hit else None,
                        "density_per_100_entities": (
                            hit.get("density_per_100_entities") if hit else None
                        ),
                        "coverage_pct": hit.get("coverage_pct") if hit else None,
                    }
                )
            feature_rows.append(
                {
                    "feature": feature,
                    "present_count": present_count,
                    "shared": present_count == len(sources),
                    "values": values,
                }
            )
        out.append(
            {
                "module": module,
                "source_count": len(sources),
                "feature_count": len(feature_rows),
                "shared_feature_count": sum(1 for row in feature_rows if row["shared"]),
                "features": feature_rows,
            }
        )
    return out


def _source_feature_distributions(desc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    path = Path(str(desc.get("path") or "")).expanduser()
    for item in desc.get("files") or []:
        module = str(item.get("module") or "")
        file_name = str(item.get("file") or "")
        if not module or not file_name:
            continue
        columns = [str(col) for col in (item.get("columns") or [])]
        features = [col for col in columns if _is_feature_column(col)]
        if not features:
            continue
        frame = _read_feature_columns(path / file_name, features)
        feature_payloads = []
        for feature in features:
            if feature not in frame:
                continue
            feature_payloads.append(
                {"feature": feature, **_summarize_feature_distribution(frame[feature])}
            )
        if feature_payloads:
            out[module] = {
                "module": module,
                "feature_count": len(feature_payloads),
                "features": feature_payloads,
            }
    return out


def _feature_distribution_payload(
    sources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    modules = sorted(
        {
            module
            for source in sources
            for module in (source.get("feature_distributions") or {}).keys()
        }
    )
    out: List[Dict[str, Any]] = []
    for module in modules:
        features = sorted(
            {
                str(feature.get("feature"))
                for source in sources
                for feature in (
                    (source.get("feature_distributions") or {}).get(module) or {}
                ).get("features", [])
                if feature.get("feature")
            }
        )
        if not features:
            continue
        rows = []
        for feature in features:
            values = []
            present_count = 0
            for source in sources:
                module_payload = (source.get("feature_distributions") or {}).get(
                    module
                ) or {}
                hit = next(
                    (
                        item
                        for item in module_payload.get("features") or []
                        if item.get("feature") == feature
                    ),
                    None,
                )
                present = hit is not None and hit.get("kind") not in {
                    "empty",
                    "missing",
                }
                if present:
                    present_count += 1
                values.append(
                    {
                        "source": source.get("label"),
                        "present": present,
                        "kind": hit.get("kind") if hit else "missing",
                        "n": hit.get("n") if hit else 0,
                        "non_null": hit.get("non_null") if hit else 0,
                        "min": hit.get("min") if hit else None,
                        "max": hit.get("max") if hit else None,
                        "points": hit.get("points") if hit else [],
                        "categories": hit.get("categories") if hit else [],
                    }
                )
            rows.append(
                {
                    "feature": feature,
                    "present_count": present_count,
                    "shared": present_count == len(sources),
                    "values": values,
                }
            )
        out.append(
            {
                "module": module,
                "source_count": len(sources),
                "feature_count": len(rows),
                "shared_feature_count": sum(1 for row in rows if row["shared"]),
                "features": rows,
            }
        )
    return out


def _read_feature_columns(path: Path, features: List[str]) -> Any:
    import pandas as pd

    lower = str(path).lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path, columns=features)
    if lower.endswith(".xlsx"):
        return pd.read_excel(path, usecols=features)
    return pd.read_csv(path, usecols=features)


def _summarize_feature_distribution(series: Any) -> Dict[str, Any]:
    import pandas as pd

    total = int(len(series))
    clean = series.dropna()
    if clean.empty:
        return {
            "kind": "empty",
            "n": total,
            "non_null": 0,
            "points": [],
            "categories": [],
        }

    bool_numeric = _bool_like_numeric(clean)
    numeric = (
        bool_numeric
        if bool_numeric is not None
        else pd.to_numeric(clean, errors="coerce")
    )
    numeric = numeric.dropna()
    if len(numeric) >= max(2, int(len(clean) * 0.65)):
        values = [float(v) for v in numeric.tolist()]
        return {
            "kind": "numeric",
            "n": total,
            "non_null": len(values),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "points": _density_points(values),
            "categories": [],
        }

    counts = clean.astype(str).str.strip().replace("", "missing").value_counts().head(8)
    categories = [
        {
            "label": str(label),
            "count": int(count),
            "pct": round(int(count) / total * 100, 1) if total else None,
        }
        for label, count in counts.items()
    ]
    return {
        "kind": "categorical",
        "n": total,
        "non_null": int(len(clean)),
        "points": [],
        "categories": categories,
    }


def _bool_like_numeric(series: Any) -> Any:
    import pandas as pd

    mapping = {
        "true": 1.0,
        "t": 1.0,
        "yes": 1.0,
        "y": 1.0,
        "1": 1.0,
        "false": 0.0,
        "f": 0.0,
        "no": 0.0,
        "n": 0.0,
        "0": 0.0,
    }
    text = series.astype(str).str.strip().str.lower()
    unique = {value for value in text.unique() if value}
    if unique and unique <= set(mapping):
        return pd.Series([mapping[value] for value in text], index=series.index)
    return None


def _density_points(values: List[float], max_bins: int = 32) -> List[Dict[str, float]]:
    import numpy as np

    if not values:
        return []
    if len(values) == 1 or min(values) == max(values):
        center = float(values[0])
        spread = max(abs(center) * 0.05, 0.5)
        return [
            {"x": round(center - spread, 6), "density": 0.0},
            {"x": round(center, 6), "density": 1.0},
            {"x": round(center + spread, 6), "density": 0.0},
        ]
    arr = np.asarray(values, dtype=float)
    if len(arr) >= 10:
        q1, q99 = np.percentile(arr, [1, 99])
        trimmed = arr[(arr >= q1) & (arr <= q99)]
        if len(trimmed) >= 6 and float(trimmed.min()) != float(trimmed.max()):
            arr = trimmed
    bins = max(6, min(max_bins, int(math.sqrt(len(values))) + 2))
    counts, edges = np.histogram(arr, bins=bins, density=True)
    if len(counts) >= 3:
        counts = np.convolve(counts, np.array([0.25, 0.5, 0.25]), mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    return [
        {"x": round(float(x), 6), "density": round(float(y), 8)}
        for x, y in zip(centers, counts)
        if np.isfinite(x) and np.isfinite(y)
    ]


def _public_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            key: value
            for key, value in source.items()
            if key not in {"feature_density", "feature_distributions"}
        }
        for source in sources
    ]


def _is_feature_column(name: str) -> bool:
    key = str(name or "").strip().lower()
    if not key:
        return False
    if key in _NON_FEATURE_COLUMNS:
        return False
    if key.endswith("_id") or key.endswith("time"):
        return False
    return True


def _safe_registered_source(source: Dict[str, Any]) -> Dict[str, Any]:
    raw_path = str(source.get("path") or "")
    return {
        "id": source.get("id"),
        "label": source.get("label") or Path(raw_path).name or "local",
        "path_hash": _hash(_norm_path(raw_path)),
        "database": source.get("database"),
        "generated": source.get("generated"),
    }


def _safe_error_detail(detail: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(detail, dict):
        return {}
    return {
        key: value
        for key, value in detail.items()
        if key
        not in {"source", "selected", "entities", "groups", "coverage", "summary"}
    }


def _privacy_payload() -> Dict[str, Any]:
    return {
        "raw_rows_returned": False,
        "direct_identifiers_returned": False,
        "patient_rows_persisted": False,
        "secrets_returned": False,
    }


def _blocked_features() -> List[Dict[str, Any]]:
    return [
        {
            "id": "row_level_filters",
            "status": "blocked",
            "reason": "Native Cross-DB Stage18 compares registered export aggregates only.",
        },
        {
            "id": "inferential_statistics",
            "status": "blocked",
            "reason": "No p-values, SMDs, or confidence intervals before the numeric evidence audit gate.",
        },
        {
            "id": "matched_cohort",
            "status": "blocked",
            "reason": "Matched cohorts need audited row-level construction and are not implemented here.",
        },
        {
            "id": "reportable_claims",
            "status": "blocked",
            "reason": "Cross-DB preview remains analysis-only until Stage19 numeric evidence audit.",
        },
    ]


def _truthy_request(value: Any) -> bool:
    if value in (None, False, "", [], {}):
        return False
    return True


def _norm_path(raw: str) -> str:
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    return str(path)


def _hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


class CrossdbReviewError(Exception):
    def __init__(self, detail: Dict[str, Any]):
        super().__init__(str(detail.get("error") or "crossdb_review_error"))
        self.detail = detail
