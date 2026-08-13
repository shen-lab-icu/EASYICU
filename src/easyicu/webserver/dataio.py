"""Local filesystem + data-folder inspection for the Data Extraction screen.

Two net-new capabilities the design mock did not have (it used hardcoded
``DETECTED`` data and a no-op Browse button):

- :func:`list_dir` — a server-side directory browser. A browser ``<input
  type=file>`` can only upload files, never enumerate the user's folders, so
  the local-first FastAPI process lists directories on demand for the picker.
- :func:`create_dir` — a local mkdir endpoint for picker destinations. Export
  destinations are folders on this machine; the UI should not require users to
  leave EasyICU just to create the parent folder.
- :func:`scan_path` — points the existing extraction logic at a folder and
  reports the database / layout / readiness. Uses the same pure readiness rules
  plus a light database heuristic (mirrors ``DataConverter._detect_database``
  without constructing one).

Everything runs locally; nothing is uploaded. Conversion is an SSE job that
drives ``DataConverter.convert_all(progress_callback=...)`` directly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from easyicu.concept.export_metadata import (
    ExportMetadataError,
    build_export_file_metadata_binding,
    missing_primary_metadata_concepts,
)
from easyicu.concept_output_sources import (
    ConceptLoadPlan,
    ConceptLoadPlanError,
    compile_concept_load_plan,
)
from easyicu.outcome_availability import structural_outcome_unavailability
from easyicu.webserver.input_validation import parse_bool

# Core metadata tables per database — a folder that holds these (as parquet or
# csv) is recognised as that database. Mirrors check_data_status' core_tables.
_CORE_TABLES = {
    "miiv": ["icustays", "patients", "admissions"],
    "miii": ["icustays", "patients", "admissions"],
    "eicu": ["patient", "apachepatientresult"],
    "aumc": ["admissions", "drugitems"],
    "hirid": ["general_table", "observations"],
    "sicdb": ["cases", "data_float_h"],
}

_DB_LABELS = {
    "miiv": "MIMIC-IV",
    "eicu": "eICU-CRD",
    "aumc": "AmsterdamUMCdb",
    "hirid": "HiRID",
    "sicdb": "SICdb",
    "miii": "MIMIC-III",
}

_MODULE_MANIFESTS = ("easyicu_export_manifest.json", "_manifest.json")
_EXPORT_METADATA_FILES = {
    "feature_definitions.csv",
    "feature_definitions.json",
}
_NATIVE_EXPORT_SCHEMA_V2 = "easyicu_native_export_v2"
DEFAULT_OBSERVATION_WINDOW_HOURS = 24 * 30
_WORKSPACE_SAMPLE_LIMIT = 500

_COHORT_PROGRESS_MESSAGES = {
    "normalizing": "Preparing cohort contract",
    "all_icu_ids": "Listing ICU stays",
    "all_icu_selected": "ICU denominator selected",
    "demographics_filter": "Applying demographic and stay filters",
    "demographics_selected": "Base cohort selected",
    "concept_prefilter": "Applying clinical concept prefilter",
    "concept_prefilter_selected": "Clinical concept prefilter complete",
    "icd_filter": "Applying ICD diagnosis filters",
    "icd_selected": "ICD diagnosis filters complete",
    "cohort_selected": "Cohort selected",
    "ready": "Cohort resolved",
}


def _safe_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def list_dir(raw_path: Optional[str]) -> Dict[str, Any]:
    """List immediate sub-directories of ``raw_path`` for the folder picker.

    When ``raw_path`` is empty/None we start from the user's home and add
    common OS shortcuts that exist on the current machine.
    """
    home = Path.home()

    if not raw_path:
        start = home
    else:
        start = Path(raw_path).expanduser()

    # Resolve symlinks where possible but never crash on a bad path.
    try:
        start = start.resolve()
    except OSError:
        pass

    if not _safe_dir(start):
        return {
            "ok": False,
            "error": "not_a_directory",
            "path": str(start),
            "parent": str(start.parent) if str(start) != str(start.parent) else None,
            "entries": [],
            "shortcuts": _shortcuts(home),
        }

    entries: List[Dict[str, Any]] = []
    try:
        for child in sorted(start.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            if not _safe_dir(child):
                continue
            # Cheap hints so the picker can flag likely data folders.
            has_csv = _has_glob(child, "*.csv") or _has_glob(child, "*.csv.gz")
            has_parquet = _has_glob(child, "*.parquet")
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "hint": "parquet" if has_parquet else ("csv" if has_csv else None),
                }
            )
    except PermissionError:
        return {
            "ok": False,
            "error": "permission_denied",
            "path": str(start),
            "parent": str(start.parent) if str(start) != str(start.parent) else None,
            "entries": [],
            "shortcuts": _shortcuts(home),
        }

    parent = str(start.parent) if str(start) != str(start.parent) else None
    return {
        "ok": True,
        "path": str(start),
        "parent": parent,
        "entries": entries,
        "shortcuts": _shortcuts(home),
    }


def create_dir(raw_path: Optional[str]) -> Dict[str, Any]:
    """Create a local directory for a folder picker destination.

    The operation is intentionally small and local-only: it never deletes or
    renames anything, and it fails if the requested path already exists as a
    file. ``parents=True`` lets the user create a nested export parent folder
    from the Web UI without switching to Finder or a shell.
    """
    if not raw_path or not str(raw_path).strip():
        return {"ok": False, "error": "path_required"}

    target = Path(str(raw_path).strip()).expanduser()
    try:
        target = target.resolve(strict=False)
    except OSError:
        pass

    if target.exists() and not _safe_dir(target):
        return {
            "ok": False,
            "error": "path_exists_not_directory",
            "path": str(target),
        }

    created = not target.exists()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return {
            "ok": False,
            "error": "permission_denied",
            "path": str(target),
            "parent": str(target.parent),
        }
    except OSError as exc:
        return {
            "ok": False,
            "error": "mkdir_failed",
            "message": str(exc),
            "path": str(target),
            "parent": str(target.parent),
        }

    return {
        "ok": True,
        "created": created,
        "path": str(target),
        "parent": str(target.parent) if str(target) != str(target.parent) else None,
        "shortcuts": _shortcuts(Path.home()),
    }


def _shortcuts(home: Path) -> List[Dict[str, str]]:
    out = [{"name": "Home", "path": str(home)}]
    volumes = Path("/Volumes")
    if _safe_dir(volumes):
        out.append({"name": "Volumes", "path": str(volumes)})
    return out


def _has_glob(path: Path, pattern: str) -> bool:
    try:
        return next(path.glob(pattern), None) is not None
    except OSError:
        return False


def _detect_database(path: Path) -> str:
    """Map the canonical schema-first identity to Web surface aliases."""
    from easyicu.databases.detection import (
        DatabaseDetectionError,
        detect_database_identity,
    )

    detected = detect_database_identity(path, strict=False)
    manifest_candidates: Set[str] = set()
    for name in _MODULE_MANIFESTS:
        manifest_path = path / name
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            continue
        declared = str(payload.get("database") or "").strip()
        if not declared:
            continue
        try:
            manifest_candidates.add(detect_database_identity(database=declared))
        except ValueError as exc:
            raise DatabaseDetectionError(
                "database_detection_manifest_invalid",
                f"Unsupported database identity {declared!r} in {manifest_path}.",
                data_path=path,
            ) from exc
    candidates = set(manifest_candidates)
    if detected != "unknown":
        candidates.add(detected)
    if len(candidates) > 1:
        raise DatabaseDetectionError(
            "database_detection_ambiguous",
            f"Conflicting database evidence in {path}: {sorted(candidates)}.",
            data_path=path,
            candidates=candidates,
        )
    if candidates:
        detected = next(iter(candidates))
    return {"mimic": "miii", "sic": "sicdb"}.get(detected, detected)


#: Columns that appear in exactly one MIMIC generation.
_MIMIC_VERSION_MARKERS = {
    "miiv": ("stay_id", "anchor_year", "anchor_age"),
    "miii": ("icustay_id", "dob", "hadm_id_seq"),
}


def _detect_mimic_version_by_schema(path: Path) -> Optional[str]:
    """Tell MIMIC-III from MIMIC-IV by reading a table's column names.

    Filenames cannot do this — both generations ship ``icustays``,
    ``patients`` and ``admissions``. ``stay_id`` (MIMIC-IV) vs ``icustay_id``
    (MIMIC-III) can, and reading a parquet footer costs no row scan.
    """

    for table in ("icustays", "patients", "admissions"):
        columns = _peek_columns(path, table)
        if not columns:
            continue
        lowered = {str(col).lower() for col in columns}
        for db_key, markers in _MIMIC_VERSION_MARKERS.items():
            if lowered.intersection(markers):
                return db_key
    return None


def _peek_columns(path: Path, table: str) -> List[str]:
    """Read a table's column names without loading any rows."""

    for candidate in (path / f"{table}.parquet", path / table):
        try:
            if candidate.is_file():
                import pyarrow.parquet as pq

                return list(pq.read_schema(candidate).names)
            if candidate.is_dir():
                import pyarrow.parquet as pq

                shard = next(candidate.glob("*.parquet"), None)
                if shard is not None:
                    return list(pq.read_schema(shard).names)
        except Exception:
            continue

    csv_candidate = next(path.glob(f"{table}.csv*"), None)
    if csv_candidate is not None:
        try:
            import pandas as pd

            header = pd.read_csv(csv_candidate, nrows=0)
            return list(header.columns)
        except Exception:
            return []
    return []


def scan_path(raw_path: str, source_hint: Optional[str] = None) -> Dict[str, Any]:
    """Inspect a folder and report what extraction can do with it.

    Returns the shape the ``scanResultState`` screen renders:
    ``{ok, path, db, db_key, layout, source, tables, modules, ready, size_hint}``.
    ``source`` is one of ``prepared`` | ``module`` | ``raw`` (or ``unknown``).
    """
    path = Path(raw_path).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass

    if not _safe_dir(path):
        return {"ok": False, "error": "not_a_directory", "path": str(path)}

    from easyicu.databases.detection import DatabaseDetectionError

    try:
        db_key = _detect_database(path)
    except DatabaseDetectionError as exc:
        return {
            "ok": False,
            "error": exc.code,
            "path": str(path),
            "candidates": list(exc.candidates),
            "ready": False,
            "privacy": {
                "raw_rows_read": False,
                "patient_identifiers_returned": False,
            },
        }
    db_label = _DB_LABELS.get(db_key, "Unknown")

    status = _check_data_status(path, db_key)
    parquet_count = status["parquet_count"]
    csv_count = status["csv_count"]
    ready = status["ready"]
    missing_tables = status["missing_tables"]

    is_module = any((path / name).exists() for name in _MODULE_MANIFESTS)

    if is_module:
        source = "module"
        layout = ["EasyICU module export", "EasyICU 模块导出"]
        ready = True
        tables = parquet_count + csv_count
    elif parquet_count > 0:
        source = "prepared"
        layout = ["Prepared (Parquet)", "已转换 (Parquet)"]
        tables = parquet_count
    elif csv_count > 0:
        source = "raw"
        layout = ["Raw CSV / CSV.GZ", "原始 CSV / CSV.GZ"]
        ready = False
        tables = csv_count
    else:
        return {
            "ok": False,
            "error": "unrecognized_folder",
            "path": str(path),
            "db": db_label,
            "db_key": db_key,
            "layout": ["No recognized ICU tables", "未识别到 ICU 数据表"],
            "source": "unknown",
            "tables": 0,
            "modules": 0,
            "ready": False,
            "missing_tables": missing_tables,
            "privacy": {
                "raw_rows_read": False,
                "patient_identifiers_returned": False,
            },
        }

    if db_key == "unknown":
        return {
            "ok": False,
            "error": "database_detection_unavailable",
            "path": str(path),
            "db": db_label,
            "db_key": db_key,
            "layout": layout,
            "source": source,
            "tables": tables,
            "modules": 0,
            "ready": False,
            "missing_tables": missing_tables,
            "privacy": {
                "raw_rows_read": False,
                "patient_identifiers_returned": False,
            },
        }

    # Honor an explicit user hint only when it does not contradict readiness.
    if source_hint in {"prepared", "module", "raw"} and source != "unknown":
        if source_hint == "raw" and parquet_count == 0:
            source = "raw"
            ready = False

    result: Dict[str, Any] = {
        "ok": True,
        "path": str(path),
        "db": db_label,
        "db_key": db_key,
        "layout": layout,
        "source": source,
        "tables": tables,
        "modules": _mappable_modules(),
        "ready": ready,
        "missing_tables": missing_tables,
        "privacy": {
            "raw_rows_read": False,
            "patient_identifiers_returned": False,
        },
    }
    if source == "raw":
        result["size_hint"] = _estimate_size(path)
    return result


def make_convert_runner(raw_path: str, database: str) -> Any:
    """Build a job runner that converts a raw folder to Parquet, emitting one
    progress event per file. Drives ``DataConverter.convert_all`` directly.
    convert_all is idempotent: already-converted files are skipped, so a re-run
    finishes fast."""

    def runner(job: Any) -> Dict[str, Any]:
        from easyicu.io.data_converter import ConversionStatus, DataConverter

        # Construction can raise ValueError on a bad path/database — let it
        # propagate so JobManager marks the job failed with the message.
        converter = DataConverter(data_path=raw_path, database=database, verbose=False)
        counts = {"converted": 0, "failed": 0, "skipped": 0}

        job.emit({"type": "start", "path": raw_path, "database": database})

        def cb(info: Dict[str, Any]) -> None:
            st = info.get("status")
            if st == ConversionStatus.FAILED:
                counts["failed"] += 1
            elif st == ConversionStatus.SKIPPED:
                counts["skipped"] += 1
            else:
                counts["converted"] += 1
            res = info.get("result") or {}
            job.emit(
                {
                    "type": "progress",
                    "current": info.get("current"),
                    "total": info.get("total"),
                    "file": info.get("file"),
                    "status": st,
                    "rows": res.get("row_count"),
                    "shards": res.get("shards"),
                    "error": res.get("error"),
                    "counts": dict(counts),
                }
            )

        results = converter.convert_all(force=False, progress_callback=cb)
        nothing = counts["converted"] == 0 and counts["failed"] == 0
        return {
            "converted": counts["converted"],
            "failed": counts["failed"],
            "skipped": counts["skipped"],
            "total_files": len(results),
            "nothing_to_do": nothing,
        }

    return runner


_EXPORT_EXT = {"csv": "csv", "excel": "xlsx", "parquet": "parquet"}
_EVENT_PRESENCE_MODULES = {"sepsis3_sofa1", "sepsis3_sofa2"}
_EXPOSURE_PRESENCE_MODULES = {
    "vasopressor",
    "vasopressors",
    "ventilation",
    "ventilator",
}
_PRESENCE_RATE_MODULES = _EVENT_PRESENCE_MODULES | _EXPOSURE_PRESENCE_MODULES


def _presence_rate_kind(module: object) -> str | None:
    normalized = str(module or "").strip().lower()
    if normalized in _EVENT_PRESENCE_MODULES:
        return "event_rate"
    if normalized in _EXPOSURE_PRESENCE_MODULES:
        return "exposure_rate"
    return None


def _is_presence_rate_module(module: object) -> bool:
    return _presence_rate_kind(module) is not None


_SUPPORTED_COHORT_PRESETS = {
    "all_icu",
    "adult_first",
    "adult_all",
    "sepsis3",
    "aki",
    "ventilation",
    "vasopressor",
    "respiratory",
    "icd",
}
_ICD_SUPPORTED_DATABASES = {"miiv", "mimic", "miii", "eicu"}
_CONCEPT_DERIVED_COHORTS = {
    "sepsis3": {
        "concepts": ["sep3_sofa2"],
        "positive": ["sep3_sofa2", "sep3", "sepsis3_sofa2"],
    },
    "aki": {
        "concepts": ["aki"],
        "positive": ["aki", "aki_stage"],
    },
    "ventilation": {
        "concepts": ["mech_vent", "vent_ind"],
        "positive": ["mech_vent", "vent_ind"],
    },
    "vasopressor": {
        "concepts": ["vaso_ind"],
        "positive": ["vaso_ind", "norepi60", "epi60", "dopa60", "dobu60"],
        "numeric_positive": [
            "norepi_rate",
            "norepi_equiv",
            "epi_rate",
            "dopa_rate",
            "dobu_rate",
        ],
    },
    "respiratory": {
        "concepts": ["adv_resp", "mech_vent", "vent_ind", "pafi", "safi"],
        "positive": ["adv_resp", "mech_vent", "vent_ind"],
        "thresholds": {"pafi": ("le", 300.0), "safi": ("le", 315.0)},
    },
}


def _choice_str(value: Any, allowed: Set[str], default: str) -> str:
    choice = str(value or "").strip()
    return choice if choice in allowed else default


def _choice_int(value: Any, allowed: Set[int], default: int) -> int:
    try:
        choice = int(float(value))
    except (TypeError, ValueError):
        return default
    return choice if choice in allowed else default


def _bool_choice(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _normalize_delta_function(value: Any) -> str:
    aliases = {
        "delta_cummin": "delta_cummin",
        "cumulative_minimum": "delta_cummin",
        "cumulative_minimum_within_si_window": "delta_cummin",
        "delta_start": "delta_start",
        "first_observed": "delta_start",
        "start_value": "delta_start",
        "delta_min": "delta_min",
        "sliding_minimum": "delta_min",
        "windowed_minimum": "delta_min",
    }
    return aliases.get(str(value or "").strip(), "delta_cummin")


def _normalize_sepsis_definition(value: Any) -> Dict[str, Any]:
    """Record Sepsis/SOFA detection parameters and runtime callback kwargs.

    The Web extraction contract is intentionally narrower than the lower-level
    callback signatures.  It locks the Sepsis-3 core definition to suspected
    infection plus a >=2 SOFA increase in the standard window.  The SOFA score
    source is owned by the selected modules (sep3_sofa2 / sep3_sofa1), not by a
    user-facing toggle.  The suspected-infection strategy is recorded as runtime
    metadata, but is not exposed as a review option: MIMIC-style sources have one
    ABX+sample timing chain, and the `icd_abx` value is accepted only for
    legacy/eICU-specific metadata.  The remaining user-facing audit choice is
    repeated-SI event selection.
    """
    raw = value if isinstance(value, dict) else {}
    profiles = {
        "selected_module_defaults": "module-specific SOFA source",
        "sofa2_primary": "SOFA-2",
        "sofa1_sensitivity": "SOFA-1",
        "dual_audit": "SOFA-2 + SOFA-1",
    }
    score_to_profile = {score: profile for profile, score in profiles.items()}
    implementation_profile = str(raw.get("implementation_profile") or "").strip()
    if implementation_profile not in profiles:
        implementation_profile = score_to_profile.get(
            str(raw.get("score_family") or "").strip(), "selected_module_defaults"
        )
    score_family = profiles[implementation_profile]

    raw_si = raw.get("suspected_infection")
    raw_si = raw_si if isinstance(raw_si, dict) else {}
    raw_sofa = raw.get("sofa_increase")
    raw_sofa = raw_sofa if isinstance(raw_sofa, dict) else {}

    si_aliases = {"antibiotic_and_sample": "and", "abx_sample": "and"}
    si_mode = str(raw_si.get("mode") or "auto").strip()
    si_mode = si_aliases.get(si_mode, si_mode)
    si_mode = _choice_str(si_mode, {"auto", "and", "icd_abx"}, "auto")
    abx_win_hours = 24
    samp_win_hours = 72
    abx_count_win_hours = 24
    abx_min_count = 1
    positive_cultures = False

    si_window_value = raw_sofa.get("si_window")
    if si_window_value is None:
        si_window_value = raw_sofa.get("si_event")
    si_window = _choice_str(str(si_window_value or "first"), {"first", "any"}, "first")
    window_before = 48
    window_after = 24
    delta_function = "delta_cummin"
    threshold = 2
    keep_components = False

    si_def = {
        "mode": si_mode,
        "abx_win_hours": abx_win_hours,
        "samp_win_hours": samp_win_hours,
        "abx_count_win_hours": abx_count_win_hours,
        "abx_min_count": abx_min_count,
        "positive_cultures_required": positive_cultures,
    }
    sofa_def = {
        "si_window": si_window,
        "window_before_si_hours": window_before,
        "window_after_si_hours": window_after,
        "delta_function": delta_function,
        "threshold": threshold,
        "keep_components": keep_components,
    }
    runtime_kwargs = {
        "si_mode": si_mode,
        "abx_win": f"{abx_win_hours}h",
        "samp_win": f"{samp_win_hours}h",
        "abx_count_win": f"{abx_count_win_hours}h",
        "abx_min_count": abx_min_count,
        "positive_cultures": positive_cultures,
        "si_window": si_window,
        "delta_fun": delta_function,
        "sofa_thresh": threshold,
        "si_lwr": f"{window_before}h",
        "si_upr": f"{window_after}h",
        "keep_components": keep_components,
    }

    return {
        "record_scope": str(
            raw.get("record_scope") or "metadata_current_runtime_defaults"
        )[:80],
        "runtime_profile": str(raw.get("runtime_profile") or "easyicu_ricu_default_v1")[
            :80
        ],
        "implementation_profile": implementation_profile,
        "score_family": score_family,
        "definition_locked": True,
        "suspected_infection": si_def,
        "sofa_increase": sofa_def,
        "runtime_kwargs": runtime_kwargs,
        "review_options": {
            "si_window": ["first", "any"],
        },
        "locked_core": {
            "suspected_infection_windows": "ABX->sample 24h; sample->ABX 72h",
            "sofa_window": "-48h/+24h",
            "delta_rule": "cumulative minimum within SI window",
            "sofa_threshold": "delta >= 2",
        },
    }


def _sepsis_runtime_kwargs(sepsis_definition: Any) -> Dict[str, Any]:
    if not isinstance(sepsis_definition, dict):
        return {}
    runtime_kwargs = sepsis_definition.get("runtime_kwargs")
    if isinstance(runtime_kwargs, dict):
        return dict(runtime_kwargs)
    normalized = _normalize_sepsis_definition(sepsis_definition)
    return dict(normalized.get("runtime_kwargs") or {})


class ExportCohortError(ValueError):
    """Raised when a user-requested extraction cohort cannot be applied honestly."""

    def __init__(self, error: str, detail: Optional[Dict[str, Any]] = None):
        self.error = error
        self.detail = {"error": error, **(detail or {})}
        super().__init__(error)


def _write_frame(df: Any, dest: Path, export_format: str) -> int:
    """Write a DataFrame to ``dest`` in the requested format; return row count."""
    rows = int(getattr(df, "shape", (0,))[0]) if df is not None else 0
    if export_format == "parquet":
        df.to_parquet(dest, index=False)
    elif export_format == "excel":
        df.to_excel(dest, index=False)  # needs openpyxl
    else:
        df.to_csv(dest, index=False)
    return rows


def _safe_slug(value: Any, fallback: str = "export") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip().lower()).strip(
        "-._"
    )
    return slug[:64] or fallback


def _unique_child_dir(root: Path, name: str) -> Path:
    candidate = root / name
    if not candidate.exists():
        return candidate
    for i in range(2, 1000):
        alt = root / f"{name}-{i:03d}"
        if not alt.exists():
            return alt
    raise FileExistsError(f"could not create unique export folder under {root}")


def _resolve_export_out_dir(
    *,
    out_dir: Optional[str],
    database: str,
    export_format: str,
    create_run_subdir: bool,
) -> Path:
    root = (
        Path(out_dir).expanduser()
        if out_dir
        else (Path.home() / ".easyicu" / "exports")
    )
    if not create_run_subdir:
        return root
    import time

    label = "easyicu_export_{stamp}_{database}_{fmt}".format(
        stamp=time.strftime("%Y%m%d_%H%M%S"),
        database=_safe_slug(database, "database"),
        fmt=_safe_slug(export_format, "format"),
    )
    return _unique_child_dir(root, label)


def _render_export_readme(
    manifest: Dict[str, Any],
    *,
    files: List[Dict[str, Any]],
    definition_files: Optional[List[Dict[str, Any]]] = None,
) -> str:
    cohort = manifest.get("cohort_contract") or {}
    report = manifest.get("cohort_report") or {}
    sepsis_def = cohort.get("sepsis_definition") if isinstance(cohort, dict) else None
    si_def = (
        sepsis_def.get("suspected_infection", {})
        if isinstance(sepsis_def, dict)
        else {}
    )
    sofa_def = (
        sepsis_def.get("sofa_increase", {}) if isinstance(sepsis_def, dict) else {}
    )
    modules = [f.get("module") for f in files if f.get("module")]
    concept_availability = manifest.get("concept_availability") or {}
    structurally_unavailable_count = int(
        concept_availability.get("structurally_unavailable_count") or 0
    )
    unique_modules = []
    for module in modules:
        if module not in unique_modules:
            unique_modules.append(module)
    definition_lines = []
    if isinstance(sepsis_def, dict):
        definition_lines = [
            f"- Sepsis runtime profile: `{sepsis_def.get('runtime_profile', '')}`",
            f"- Sepsis implementation profile: `{sepsis_def.get('implementation_profile', '')}`",
            f"- Sepsis score family: `{sepsis_def.get('score_family', '')}`",
            (
                "- Suspected infection: "
                f"`{si_def.get('mode', '')}`, ABX->sample `{si_def.get('abx_win_hours', '')}h`, "
                f"sample->ABX `{si_def.get('samp_win_hours', '')}h`, "
                f"ABX count `≥{si_def.get('abx_min_count', '')}/{si_def.get('abx_count_win_hours', '')}h`, "
                f"positive cultures `{si_def.get('positive_cultures_required', '')}`"
            ),
            (
                "- SOFA increase rule: "
                f"SI event `{sofa_def.get('si_window', '')}`, window `-{sofa_def.get('window_before_si_hours', '')}h/+{sofa_def.get('window_after_si_hours', '')}h`, "
                f"delta `{sofa_def.get('delta_function', '')}`, threshold `{sofa_def.get('threshold', '')}`, "
                f"keep components `{sofa_def.get('keep_components', '')}`"
            ),
            f"- Sepsis runtime kwargs: `{sepsis_def.get('runtime_kwargs', {})}`",
            f"- Definition note scope: `{sepsis_def.get('record_scope', '')}`",
        ]

    lines = [
        "# EasyICU Export",
        "",
        "This folder was generated locally by the EasyICU FastAPI web app.",
        "No patient rows are included in this README; row-level data are only in the exported module files in this folder.",
        "",
        "## Extraction summary",
        "",
        f"- Generated: `{manifest.get('generated', '')}`",
        f"- Database: `{manifest.get('database', '')}`",
        f"- Source path: `{manifest.get('data_path', '')}`",
        f"- Export format: `{manifest.get('format', '')}`",
        f"- Max patients requested: `{manifest.get('max_patients')}`",
        f"- Cohort preset: `{cohort.get('preset', '')}`",
        f"- Cohort selected: `{report.get('selected', report.get('cohort_size', ''))}`",
        f"- Observation window: `{cohort.get('observation_window_hours', '')} hours`",
        f"- Modules: `{', '.join(unique_modules)}`",
        f"- Concepts selected: `{sum(int(f.get('concepts') or 0) for f in files)}`",
        f"- Structurally unavailable for this database: `{structurally_unavailable_count}` (listed with reason codes in `_manifest.json`)",
        *definition_lines,
        "",
        "## Reproducibility files",
        "",
        "- `_manifest.json` contains the machine-readable extraction contract, module files, row counts, and cohort report.",
        *(
            [
                "- The content-addressed column metadata sidecar binds each authorized physical output to its source concept, role, units, ranges, lineage, and derivation window."
            ]
            if isinstance(manifest.get("column_metadata"), dict)
            else []
        ),
        "- Each module file contains the extracted concept table for the same resolved cohort.",
        *(
            [
                "- `feature_definitions.json` and `feature_definitions.csv` contain the selected concept IDs, names, units, exported module files, and callback provenance.",
                "- Example definition row: `concept_id=age`, `module=demographics`, `unit=years`, `export_files=demographics.parquet`, `callback_import_path=easyicu.api.load_concepts`, `callback_project_ref=<local path omitted>`.",
                "- Raw table/column lineage is included only when declared by the catalog; otherwise `raw_metadata_status=not_declared_in_current_catalog` is used instead of guessing.",
            ]
            if definition_files
            else []
        ),
        "- This `README.md` is a human-readable summary of the same extraction contract.",
        "",
        "## Files",
        "",
    ]
    for f in files:
        lines.append(
            f"- `{f.get('file')}` — module `{f.get('module', '')}`, rows `{f.get('rows', '')}`"
        )
    for f in definition_files or []:
        lines.append(
            f"- `{f.get('file')}` — feature definition manifest, records `{f.get('records', '')}`"
        )
    metadata_descriptor = manifest.get("column_metadata")
    if isinstance(metadata_descriptor, dict) and metadata_descriptor.get("file"):
        lines.append(
            f"- `{metadata_descriptor.get('file')}` — typed physical-column metadata, records `{metadata_descriptor.get('record_count', '')}`"
        )
    lines.append("")
    return "\n".join(lines)


def _normalize_export_concepts(
    selected_modules: List[str],
    concepts: Any,
    catalog: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Resolve an optional UI concept selection to a module -> concept list.

    ``concepts`` may be omitted (all concepts in selected modules), a mapping
    keyed by module name, or a flat concept-id list. Invalid concept/module
    requests fail closed before any export files are written.
    """
    modules = [module for module in selected_modules if catalog.get(module)]
    module_set = set(modules)
    if concepts is None:
        return {module: list(catalog[module]) for module in modules}

    selected: Dict[str, List[str]] = {module: [] for module in modules}
    invalid: List[str] = []

    if isinstance(concepts, dict):
        for module, raw_values in concepts.items():
            module_key = str(module)
            if module_key not in module_set:
                invalid.append(module_key)
                continue
            allowed = set(catalog[module_key])
            if raw_values is None:
                selected[module_key] = list(catalog[module_key])
                continue
            if not isinstance(raw_values, list):
                invalid.append(f"{module_key}:not_a_list")
                continue
            seen: Set[str] = set()
            for value in raw_values:
                concept = str(value)
                if concept not in allowed:
                    invalid.append(f"{module_key}:{concept}")
                    continue
                if concept not in seen:
                    selected[module_key].append(concept)
                    seen.add(concept)
    elif isinstance(concepts, list):
        concept_to_module = {
            concept: module for module in modules for concept in catalog.get(module, [])
        }
        for value in concepts:
            concept = str(value)
            module = concept_to_module.get(concept)
            if not module:
                invalid.append(concept)
                continue
            if concept not in selected[module]:
                selected[module].append(concept)
    else:
        raise ExportCohortError(
            "invalid_concept_selection",
            {"detail": "concepts must be a module mapping or a flat concept list"},
        )

    if invalid:
        raise ExportCohortError(
            "invalid_selected_concepts",
            {"invalid": invalid[:20], "invalid_count": len(invalid)},
        )

    selected = {module: ids for module, ids in selected.items() if ids}
    if not selected:
        raise ExportCohortError(
            "no_selected_concepts",
            {"modules": modules},
        )
    return selected


def _feature_definition_payload(
    *,
    database: str,
    data_path: str,
    export_path: Path,
    concept_plan: Dict[str, List[str]],
    files: List[Dict[str, Any]],
    api_module: Any,
    unavailable_concepts: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a metadata-only definition manifest for selected concepts."""
    from easyicu.concept.catalog import (
        COMPOSITE_CONCEPT_OUTPUT_SOURCES,
        CONCEPT_DESCRIPTIONS,
        CONCEPT_DICTIONARY,
        CONCEPT_GROUP_NAMES,
    )

    project_path = Path(__file__).resolve().parents[3]
    callback = getattr(api_module, "load_concepts", None)
    callback_module = sys.modules.get(getattr(callback, "__module__", ""))
    callback_source = Path(
        getattr(callback_module, "__file__", getattr(api_module, "__file__", ""))
    ).resolve()
    data_source_ref = _shareable_path_reference(data_path)
    export_ref = _shareable_path_reference(export_path)
    callback_source_ref = _shareable_path_reference(
        callback_source, relative_to=project_path
    )
    project_ref = _shareable_project_reference(project_path)
    file_by_module: Dict[str, List[str]] = {}
    for item in files:
        module = str(item.get("module") or "")
        file_name = str(item.get("file") or "")
        if module and file_name:
            file_by_module.setdefault(module, []).append(file_name)

    unavailable_by_concept = {
        str(item.get("concept_id")): dict(item)
        for item in (unavailable_concepts or ())
        if item.get("concept_id")
    }
    records: List[Dict[str, Any]] = []
    for module, concept_ids in concept_plan.items():
        group_en, group_zh = CONCEPT_GROUP_NAMES.get(module, (module, module))
        for concept_id in concept_ids:
            name_en, name_zh, unit = CONCEPT_DICTIONARY.get(
                concept_id, (concept_id, concept_id, "")
            )
            desc_en, desc_zh = CONCEPT_DESCRIPTIONS.get(concept_id, ("", ""))
            derived_output_source = COMPOSITE_CONCEPT_OUTPUT_SOURCES.get(concept_id)
            unavailable = unavailable_by_concept.get(concept_id)
            records.append(
                {
                    "database": database,
                    "concept_id": concept_id,
                    "name_en": name_en,
                    "name_zh": name_zh,
                    "unit": unit,
                    "module": module,
                    "module_name_en": group_en,
                    "module_name_zh": group_zh,
                    "description_en": desc_en,
                    "description_zh": desc_zh,
                    "availability": (
                        unavailable
                        if unavailable is not None
                        else {
                            "concept_id": concept_id,
                            "module": module,
                            "database": database,
                            "status": "selected_for_export",
                            "reason_code": None,
                            "supported_databases": [],
                        }
                    ),
                    "source": {
                        "data_source_ref": data_source_ref,
                        "export_ref": export_ref,
                        "export_files": file_by_module.get(module, []),
                        "raw_tables": [],
                        "raw_columns": [],
                        "raw_metadata_status": "not_declared_in_current_catalog",
                        "local_path_policy": "absolute_paths_omitted_from_shareable_manifest",
                        "note": (
                            "EasyICU resolves raw database tables inside concept "
                            "callbacks. This manifest records the selected concept "
                            "metadata, exported module files, and callback "
                            "provenance; raw table/column lineage is only populated "
                            "when declared by the catalog."
                        ),
                    },
                    "callback": {
                        "import_path": "easyicu.api.load_concepts",
                        "function": "load_concepts",
                        "source_module_file": callback_source_ref["hint"],
                        "source_file_ref": callback_source_ref,
                        "project_ref": project_ref,
                        "module_callback": derived_output_source,
                        "call_signature": (
                            "load_concepts(concept_ids, patient_ids=resolved_cohort, "
                            "database=database, data_path=data_path, use_sofa2=..., "
                            "merge=True, **cohort_runtime_kwargs)"
                        ),
                    },
                }
            )

    return {
        "schema_version": "easyicu_feature_definitions_v1",
        "database": database,
        "data_source_ref": data_source_ref,
        "export_ref": export_ref,
        "record_count": len(records),
        "raw_lineage_scope": "catalog_metadata_plus_callback_provenance",
        "local_path_policy": "absolute_paths_omitted_from_shareable_feature_definitions",
        "records": records,
    }


def _shareable_path_reference(
    path_like: str | Path, *, relative_to: Path | None = None
) -> Dict[str, Any]:
    raw = str(path_like or "")
    path = Path(raw) if raw else Path("")
    hint = path.name if raw else ""
    if raw and relative_to is not None:
        try:
            hint = str(path.resolve().relative_to(relative_to.resolve()))
        except Exception:
            pass
    return {
        "hint": hint,
        "sha256_12": (
            hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12] if raw else ""
        ),
        "absolute_path_omitted": bool(raw and path.is_absolute()),
    }


def _shareable_project_reference(project_path: Path) -> Dict[str, Any]:
    """Return stable public provenance for an EasyICU source checkout."""
    ref = _shareable_path_reference(project_path)
    # Checkouts and worktrees may have arbitrary directory names. Do not leak
    # that local name into an otherwise shareable manifest.
    ref["hint"] = "EASYICU"
    return ref


def _format_shareable_path_reference(ref: Any) -> str:
    if not isinstance(ref, dict):
        return ""
    hint = str(ref.get("hint") or "")
    digest = str(ref.get("sha256_12") or "")
    if hint and digest:
        return f"{hint}#{digest}"
    return hint or digest


def _write_feature_definition_files(
    out: Path, payload: Dict[str, Any]
) -> List[Dict[str, Any]]:
    json_name = "feature_definitions.json"
    csv_name = "feature_definitions.csv"
    (out / json_name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fieldnames = [
        "database",
        "module",
        "module_name_en",
        "module_name_zh",
        "concept_id",
        "name_en",
        "name_zh",
        "unit",
        "export_files",
        "raw_tables",
        "raw_columns",
        "raw_metadata_status",
        "data_source_ref",
        "export_ref",
        "local_path_policy",
        "callback_import_path",
        "callback_function",
        "callback_source_module_file",
        "callback_source_file_ref",
        "callback_project_ref",
        "module_callback",
        "description_en",
        "description_zh",
    ]
    with (out / csv_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in payload.get("records", []):
            source = record.get("source") or {}
            callback = record.get("callback") or {}
            writer.writerow(
                {
                    "database": record.get("database", ""),
                    "module": record.get("module", ""),
                    "module_name_en": record.get("module_name_en", ""),
                    "module_name_zh": record.get("module_name_zh", ""),
                    "concept_id": record.get("concept_id", ""),
                    "name_en": record.get("name_en", ""),
                    "name_zh": record.get("name_zh", ""),
                    "unit": record.get("unit", ""),
                    "export_files": ";".join(source.get("export_files") or []),
                    "raw_tables": ";".join(source.get("raw_tables") or []),
                    "raw_columns": ";".join(source.get("raw_columns") or []),
                    "raw_metadata_status": source.get("raw_metadata_status", ""),
                    "data_source_ref": _format_shareable_path_reference(
                        source.get("data_source_ref")
                    ),
                    "export_ref": _format_shareable_path_reference(
                        source.get("export_ref")
                    ),
                    "local_path_policy": source.get("local_path_policy", ""),
                    "callback_import_path": callback.get("import_path", ""),
                    "callback_function": callback.get("function", ""),
                    "callback_source_module_file": callback.get(
                        "source_module_file", ""
                    ),
                    "callback_source_file_ref": _format_shareable_path_reference(
                        callback.get("source_file_ref")
                    ),
                    "callback_project_ref": _format_shareable_path_reference(
                        callback.get("project_ref")
                    ),
                    "module_callback": callback.get("module_callback") or "",
                    "description_en": record.get("description_en", ""),
                    "description_zh": record.get("description_zh", ""),
                }
            )

    return [
        {
            "file": json_name,
            "kind": "feature_definitions",
            "records": payload.get("record_count", 0),
        },
        {
            "file": csv_name,
            "kind": "feature_definitions_csv",
            "records": payload.get("record_count", 0),
        },
    ]


def _build_export_file_metadata_binding(
    *,
    relative_path: str,
    module: str,
    frame: Any,
    concept_ids: Sequence[str],
    database: str,
    database_class_prefixes: Sequence[str],
    dictionary: Any,
):
    """Bind producer-owned physical outputs to typed metadata exactly once."""

    try:
        return build_export_file_metadata_binding(
            relative_path=relative_path,
            module=module,
            frame=frame,
            concept_ids=concept_ids,
            database=database,
            database_class_prefixes=database_class_prefixes,
            dictionary=dictionary,
        )
    except ExportMetadataError as exc:
        raise ExportCohortError(exc.error, exc.detail) from exc


def _missing_primary_metadata_concepts(
    *,
    concept_plan: Dict[str, List[str]],
    file_bindings: Sequence[Any],
) -> List[str]:
    """Return selected concepts without one unambiguous typed primary column."""

    return missing_primary_metadata_concepts(
        concept_plan=concept_plan,
        file_bindings=file_bindings,
    )


def _classify_structurally_unavailable_concepts(
    *,
    concepts: Sequence[str],
    concept_plan: Dict[str, List[str]],
    database: str,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Separate owner-confirmed cross-database gaps from unexplained outputs."""

    module_by_concept = {
        concept: module
        for module, module_concepts in concept_plan.items()
        for concept in module_concepts
    }
    unavailable: List[Dict[str, Any]] = []
    unexplained: List[str] = []
    for concept in concepts:
        receipt = structural_outcome_unavailability(concept, database)
        if receipt is None:
            unexplained.append(concept)
            continue
        unavailable.append(
            {
                "concept_id": receipt.concept_id,
                "module": module_by_concept.get(receipt.concept_id),
                "database": receipt.database,
                "status": "structurally_unavailable",
                "reason_code": receipt.reason_code,
                "supported_databases": list(receipt.supported_databases),
            }
        )
    return unavailable, unexplained


def _coerce_int(
    value: Any,
    default: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        out = default
    if min_value is not None:
        out = max(min_value, out)
    if max_value is not None:
        out = min(max_value, out)
    return out


def _split_icd_tokens(raw: Any) -> List[str]:
    import re

    text = str(raw or "").upper().replace("，", ",").replace("；", ";")
    tokens: List[str] = []
    for part in re.split(r"[\s,;]+", text):
        token = part.strip().replace(".", "")
        if not token:
            continue
        if "-" in token:
            start, end = [p.strip() for p in token.split("-", 1)]
            if (
                len(start) == len(end)
                and len(start) >= 2
                and start[:-2] == end[:-2]
                and start[-2:].isdigit()
                and end[-2:].isdigit()
            ):
                prefix = start[:-2]
                lo, hi = int(start[-2:]), int(end[-2:])
                if 0 <= hi - lo <= 50:
                    tokens.extend(f"{prefix}{i:02d}" for i in range(lo, hi + 1))
                    continue
        tokens.append(token)
    seen: Set[str] = set()
    out: List[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out[:64]


def _normalize_export_cohort(cohort: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = cohort if isinstance(cohort, dict) else {}
    has_explicit_contract = isinstance(cohort, dict) and bool(cohort)
    preset = (
        str(
            raw.get("preset") or ("adult_first" if has_explicit_contract else "all_icu")
        )
        .strip()
        .lower()
    )
    if preset not in _SUPPORTED_COHORT_PRESETS:
        raise ExportCohortError(
            "unsupported_cohort_preset",
            {"preset": preset, "supported": sorted(_SUPPORTED_COHORT_PRESETS)},
        )

    include = _split_icd_tokens(
        raw.get("icd_include")
        or raw.get("include_diagnoses")
        or raw.get("include")
        or ""
    )
    exclude = _split_icd_tokens(
        raw.get("icd_exclude")
        or raw.get("exclude_diagnoses")
        or raw.get("exclude")
        or ""
    )
    try:
        icd_enabled = (
            parse_bool(raw.get("icd_enabled"), default=False) or preset == "icd"
        )
        exclude_readmissions = parse_bool(
            raw.get("exclude_readmissions"), default=preset == "adult_first"
        )
    except ValueError as exc:
        raise ExportCohortError(
            "invalid_cohort_boolean",
            {"fields": ["icd_enabled", "exclude_readmissions"]},
        ) from exc
    if not icd_enabled:
        include = []
        exclude = []
    if preset == "icd" and not include and not exclude:
        raise ExportCohortError("empty_icd_filter", {"preset": preset})

    age_min = _coerce_int(
        raw.get("age_min"), 18 if preset == "adult_first" else 0, 0, 120
    )
    age_max = _coerce_int(raw.get("age_max"), 100, 0, 120)
    if age_min > age_max:
        age_min, age_max = age_max, age_min
    if preset == "adult_first":
        age_min = max(18, age_min)
    min_los = _coerce_int(raw.get("min_icu_los_hours"), 0, 0, 24 * 30)
    window = _coerce_int(
        raw.get("observation_window_hours"),
        DEFAULT_OBSERVATION_WINDOW_HOURS,
        1,
        DEFAULT_OBSERVATION_WINDOW_HOURS,
    )

    return {
        "preset": preset,
        "age_min": age_min,
        "age_max": age_max,
        "min_icu_los_hours": min_los,
        "observation_window_hours": window,
        "exclude_readmissions": exclude_readmissions,
        "icd_enabled": icd_enabled,
        "icd_include": include,
        "icd_exclude": exclude,
        "sepsis_definition": _normalize_sepsis_definition(raw.get("sepsis_definition")),
    }


def _database_for_patient_filter(database: str) -> str:
    if database == "sicdb":
        return "sic"
    if database == "miii":
        return "mimic"
    return database


def _find_table_file(root: Path, table_names: List[str]) -> Optional[Path]:
    suffixes = (".parquet", ".csv", ".csv.gz")
    for table in table_names:
        for suffix in suffixes:
            direct = root / f"{table}{suffix}"
            if direct.exists():
                return direct
            try:
                for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                    if not child.is_dir():
                        continue
                    nested = child / f"{table}{suffix}"
                    if nested.exists():
                        return nested
            except OSError:
                continue
    return None


def _read_table_columns(path: Path, columns: List[str]) -> Any:
    import pandas as pd

    present = set(_read_columns(path))
    wanted = [c for c in columns if c in present]
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=wanted or None)
    if wanted:
        return pd.read_csv(path, usecols=wanted)
    return pd.read_csv(path)


def _normal_text_series(series: Any) -> Any:
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(".", "", regex=False)
        .str.strip()
    )


def _match_mimic_icd_ids(
    data_path: Path, stay_id_col: str, include: List[str], exclude: List[str]
) -> Tuple[Set[Any], Set[Any]]:
    import pandas as pd

    stays_path = _find_table_file(data_path, ["icustays"])
    diag_path = _find_table_file(data_path, ["diagnoses_icd", "diagnoses"])
    if not stays_path or not diag_path:
        raise ExportCohortError(
            "icd_tables_missing", {"required_tables": ["icustays", "diagnoses_icd"]}
        )

    stays = _read_table_columns(
        stays_path, [stay_id_col, "stay_id", "icustay_id", "hadm_id"]
    )
    diag = _read_table_columns(
        diag_path, ["hadm_id", "icd_code", "icd9_code", "diagnosis"]
    )
    id_col = (
        stay_id_col
        if stay_id_col in stays.columns
        else ("stay_id" if "stay_id" in stays.columns else "icustay_id")
    )
    if (
        "hadm_id" not in stays.columns
        or "hadm_id" not in diag.columns
        or id_col not in stays.columns
    ):
        raise ExportCohortError("icd_join_columns_missing", {"database": "miiv"})

    code_col = next(
        (c for c in ["icd_code", "icd9_code", "diagnosis"] if c in diag.columns), None
    )
    if not code_col:
        raise ExportCohortError("icd_code_column_missing", {"database": "miiv"})

    joined = diag[["hadm_id", code_col]].merge(
        stays[["hadm_id", id_col]], on="hadm_id", how="inner"
    )
    codes = _normal_text_series(joined[code_col])

    def select(tokens: List[str]) -> Set[Any]:
        if not tokens:
            return set()
        mask = pd.Series(False, index=joined.index)
        for token in tokens:
            mask = mask | codes.str.startswith(token)
        return set(joined.loc[mask, id_col].dropna().tolist())

    return select(include), select(exclude)


def _match_eicu_icd_ids(
    data_path: Path, include: List[str], exclude: List[str]
) -> Tuple[Set[Any], Set[Any]]:
    import pandas as pd

    diag_path = _find_table_file(data_path, ["diagnosis"])
    if not diag_path:
        raise ExportCohortError(
            "icd_tables_missing", {"required_tables": ["diagnosis"]}
        )
    diag = _read_table_columns(
        diag_path, ["patientunitstayid", "icd9code", "diagnosisstring"]
    )
    if "patientunitstayid" not in diag.columns:
        raise ExportCohortError("icd_join_columns_missing", {"database": "eicu"})
    search_cols = [c for c in ["icd9code", "diagnosisstring"] if c in diag.columns]
    if not search_cols:
        raise ExportCohortError("icd_code_column_missing", {"database": "eicu"})

    normalized = [_normal_text_series(diag[c]) for c in search_cols]

    def select(tokens: List[str]) -> Set[Any]:
        if not tokens:
            return set()
        mask = pd.Series(False, index=diag.index)
        for token in tokens:
            for values in normalized:
                mask = mask | values.str.contains(token, regex=False)
        return set(diag.loc[mask, "patientunitstayid"].dropna().tolist())

    return select(include), select(exclude)


def _match_icd_ids(
    data_path: Path, database: str, id_col: str, include: List[str], exclude: List[str]
) -> Tuple[Set[Any], Set[Any]]:
    db = _database_for_patient_filter(database)
    if db not in _ICD_SUPPORTED_DATABASES:
        raise ExportCohortError(
            "icd_filter_unsupported_database", {"database": database}
        )
    if db in {"miiv", "mimic", "miii"}:
        return _match_mimic_icd_ids(data_path, id_col, include, exclude)
    return _match_eicu_icd_ids(data_path, include, exclude)


def _cohort_id_column(frame: Any, preferred: str) -> Optional[str]:
    candidates = [
        preferred,
        "stay_id",
        "icustay_id",
        "patientunitstayid",
        "admissionid",
        "patientid",
        "CaseID",
    ]
    return next(
        (col for col in candidates if col in getattr(frame, "columns", [])), None
    )


def _truthy_mask(values: Any) -> Any:
    import pandas as pd

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0) > 0
    lowered = values.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y", "positive", "present"})


def _positive_ids_from_concept_payload(
    payload: Any, id_col: str, spec: Dict[str, Any]
) -> Set[Any]:
    import pandas as pd

    frames = list(payload.values()) if isinstance(payload, dict) else [payload]
    matched: Set[Any] = set()
    for frame in frames:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        frame_id = _cohort_id_column(frame, id_col)
        if not frame_id:
            continue

        mask = pd.Series(False, index=frame.index)
        for col in spec.get("positive", []):
            if col in frame.columns:
                mask = mask | _truthy_mask(frame[col])
        for col in spec.get("numeric_positive", []):
            if col in frame.columns:
                numeric = pd.to_numeric(frame[col], errors="coerce")
                mask = mask | (numeric.fillna(0) > 0)
        for col, (op, threshold) in (spec.get("thresholds") or {}).items():
            if col in frame.columns:
                numeric = pd.to_numeric(frame[col], errors="coerce")
                if op == "le":
                    mask = mask | ((numeric > 0) & (numeric <= threshold))
                elif op == "ge":
                    mask = mask | (numeric >= threshold)
        matched.update(frame.loc[mask, frame_id].dropna().tolist())
    return matched


def _match_concept_derived_cohort_ids(
    api: Any,
    data_path: str,
    database: str,
    id_col: str,
    base_ids: Set[Any],
    preset: str,
    window_hours: int,
    sepsis_load_kwargs: Optional[Dict[str, Any]] = None,
) -> Set[Any]:
    spec = _CONCEPT_DERIVED_COHORTS.get(preset)
    if not spec:
        return set(base_ids)
    load_kwargs = {"win_length": f"{window_hours}h"}
    if _module_uses_sepsis_kwargs(spec["concepts"]):
        load_kwargs.update(sepsis_load_kwargs or {})
    try:
        payload = api.load_concepts(
            spec["concepts"],
            patient_ids={id_col: sorted(base_ids, key=lambda value: str(value))},
            database=database,
            data_path=str(data_path),
            merge=True,
            verbose=False,
            **load_kwargs,
        )
    except Exception as exc:
        raise ExportCohortError(
            "concept_cohort_unavailable",
            {"preset": preset, "concepts": spec["concepts"], "detail": str(exc)},
        ) from exc

    matched = _positive_ids_from_concept_payload(payload, id_col, spec)
    if not matched and base_ids:
        return set()
    return set(base_ids) & matched


def _resolve_export_cohort(
    data_path: str,
    database: str,
    cohort: Optional[Dict[str, Any]],
    max_patients: Optional[int],
    api: Any,
    progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    normalized = _normalize_export_cohort(cohort)
    sepsis_load_kwargs = _sepsis_runtime_kwargs(normalized.get("sepsis_definition"))
    max_n = _coerce_int(max_patients, 0, 0, None) if max_patients is not None else 0
    id_col = api.get_id_col_for_database(_database_for_patient_filter(database))
    patient_filter_db = _database_for_patient_filter(database)

    def emit(stage: str, **extra: Any) -> None:
        if progress is not None:
            progress(stage, extra)

    emit(
        "normalizing",
        preset=normalized["preset"],
        window_hours=normalized["observation_window_hours"],
    )

    filters_active = (
        normalized["preset"] != "all_icu"
        or normalized["age_min"] > 0
        or normalized["age_max"] < 100
        or normalized["min_icu_los_hours"] > 0
        or normalized["exclude_readmissions"]
        or normalized["icd_enabled"]
    )

    if not filters_active:
        emit("all_icu_ids", preset=normalized["preset"])
        ids_list, id_col = api.get_all_patient_ids(
            str(data_path),
            database=_database_for_patient_filter(database),
            max_patients=max_n or None,
        )
        emit(
            "all_icu_selected", selected=len(ids_list), max_patients_applied=bool(max_n)
        )
        return {
            "patient_ids": {id_col: list(ids_list)} if ids_list else None,
            "cohort_size": len(ids_list),
            "id_col": id_col,
            "cohort_contract": normalized,
            "cohort_report": {
                "mode": "all_icu",
                "selected": len(ids_list),
                "max_patients_applied": bool(max_n),
                "applied_filters": [],
            },
            "load_kwargs": {"win_length": f"{normalized['observation_window_hours']}h"},
            "sepsis_load_kwargs": sepsis_load_kwargs,
        }

    from easyicu.patient_filter import PatientFilter

    emit(
        "demographics_filter",
        preset=normalized["preset"],
        age_min=normalized["age_min"],
        age_max=normalized["age_max"],
        min_icu_los_hours=normalized["min_icu_los_hours"],
        exclude_readmissions=normalized["exclude_readmissions"],
    )
    pf = PatientFilter(database=patient_filter_db, data_path=data_path, verbose=False)
    filtered = pf.filter(
        age_min=normalized["age_min"] if normalized["age_min"] > 0 else None,
        age_max=normalized["age_max"] if normalized["age_max"] < 100 else None,
        first_icu_stay=normalized["exclude_readmissions"]
        or normalized["preset"] == "adult_first",
        los_min=(
            normalized["min_icu_los_hours"]
            if normalized["min_icu_los_hours"] > 0
            else None
        ),
        los_max=None,
        has_sepsis=None,
        return_dataframe=True,
    )
    source_total = _positive_int(getattr(pf, "_last_original_count", None))
    if "patient_id" not in filtered.columns:
        raise ExportCohortError(
            "cohort_filter_missing_patient_id", {"database": database}
        )
    selected: Set[Any] = set(filtered["patient_id"].dropna().tolist())
    before_concept = len(selected)
    concept_matches: Optional[int] = None
    before_icd = len(selected)
    applied = ["demographics"]
    emit("demographics_selected", selected=before_concept)

    if normalized["preset"] in _CONCEPT_DERIVED_COHORTS:
        spec = _CONCEPT_DERIVED_COHORTS[normalized["preset"]]
        emit(
            "concept_prefilter",
            preset=normalized["preset"],
            concepts=spec["concepts"],
            candidates=before_concept,
            window_hours=normalized["observation_window_hours"],
        )
        selected = _match_concept_derived_cohort_ids(
            api,
            str(data_path),
            database,
            id_col,
            selected,
            normalized["preset"],
            normalized["observation_window_hours"],
            sepsis_load_kwargs,
        )
        concept_matches = len(selected)
        before_icd = len(selected)
        applied.append("concept_prefilter")
        emit(
            "concept_prefilter_selected",
            preset=normalized["preset"],
            selected=concept_matches,
        )

    include_ids: Set[Any] = set()
    exclude_ids: Set[Any] = set()
    if normalized["icd_enabled"]:
        emit(
            "icd_filter",
            include_tokens=len(normalized["icd_include"]),
            exclude_tokens=len(normalized["icd_exclude"]),
        )
        include_ids, exclude_ids = _match_icd_ids(
            Path(data_path).expanduser(),
            database,
            id_col,
            normalized["icd_include"],
            normalized["icd_exclude"],
        )
        if normalized["icd_include"]:
            selected = selected & include_ids
        if normalized["icd_exclude"]:
            selected = selected - exclude_ids
        applied.append("icd")
        emit(
            "icd_selected",
            selected=len(selected),
            include_matches=len(include_ids),
            exclude_matches=len(exclude_ids),
        )

    ids = sorted(selected, key=lambda value: str(value))
    uncapped = len(ids)
    if max_n and len(ids) > max_n:
        ids = ids[:max_n]
    emit(
        "cohort_selected",
        selected=len(ids),
        selected_before_cap=uncapped,
        max_patients_applied=bool(max_n and uncapped > max_n),
    )

    return {
        "patient_ids": {id_col: ids} if ids else {id_col: []},
        "cohort_size": len(ids),
        "id_col": id_col,
        "cohort_contract": normalized,
        "cohort_report": {
            "mode": normalized["preset"],
            "source_total": source_total,
            "selected": len(ids),
            "selected_before_cap": uncapped,
            "selected_before_concept_prefilter": before_concept,
            "concept_matches": concept_matches,
            "selected_before_icd": before_icd,
            "max_patients_applied": bool(max_n and uncapped > max_n),
            "applied_filters": applied,
            "icd": {
                "enabled": normalized["icd_enabled"],
                "include_tokens": normalized["icd_include"],
                "exclude_tokens": normalized["icd_exclude"],
                "include_matches": len(include_ids),
                "exclude_matches": len(exclude_ids),
            },
        },
        "load_kwargs": {"win_length": f"{normalized['observation_window_hours']}h"},
        "sepsis_load_kwargs": sepsis_load_kwargs,
    }


def _module_uses_sepsis_kwargs(concepts: List[str]) -> bool:
    sepsis_concepts = {"susp_inf", "sep3", "sep3_sofa2"}
    return any(str(concept) in sepsis_concepts for concept in concepts)


def _materialize_concept_load_plan(frame: Any, plan: ConceptLoadPlan) -> Any:
    """Project loader source columns back onto the requested public outputs."""

    if not plan.materializations or frame is None or not hasattr(frame, "columns"):
        return frame

    projected = frame
    copied = False
    for binding in plan.materializations:
        output = binding.output_concept
        source = binding.source_concept
        if output in projected.columns or source not in projected.columns:
            continue
        if not copied:
            projected = projected.copy()
            copied = True
        projected[output] = projected[source]

    removable_sources = {
        binding.source_concept
        for binding in plan.materializations
        if binding.source_concept not in plan.output_concepts
        and binding.output_concept in projected.columns
        and binding.source_concept in projected.columns
    }
    if removable_sources:
        if not copied:
            projected = projected.copy()
        projected = projected.drop(columns=sorted(removable_sources))
    return projected


def make_export_runner(
    data_path: str,
    database: str,
    modules: Optional[List[str]] = None,
    concepts: Any = None,
    export_format: str = "csv",
    merge: bool = False,
    out_dir: Optional[str] = None,
    create_run_subdir: bool = False,
    max_patients: Optional[int] = None,
    cohort: Optional[Dict[str, Any]] = None,
    include_feature_definitions: bool = True,
) -> Any:
    """Build a job runner that extracts the selected feature modules to disk.

    Reuses the public extraction API (``easyicu.api.load_concepts``) one module
    at a time, sharing raw reads across modules via ``api.keep_cache`` (the bulk
    rule from CLAUDE.md). Each module is written as ``{module}.{ext}`` plus a
    ``_manifest.json``. ``max_patients`` caps the cohort (None = all). Emits one
    ``progress`` event per module.
    """
    import os

    def runner(job: Any) -> Dict[str, Any]:
        import json
        import time

        # On this memory-tight machine the batch estimator over-predicts ~5x and
        # trips the low-mem path; force the fast in-process path (see CLAUDE.md).
        os.environ.setdefault("EASYICU_FORCE_INPROCESS_BATCH", "1")
        import easyicu.api as api
        from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
        from easyicu.concept.metadata_sidecar import (
            EXPORT_PHYSICAL_SCOPE,
            ColumnMetadataSidecar,
            write_content_addressed_sidecar,
        )
        from easyicu.config import load_src_cfg
        from easyicu.resources import load_dictionary

        sel_modules = [
            m
            for m in (modules or list(CONCEPT_GROUPS_INTERNAL.keys()))
            if CONCEPT_GROUPS_INTERNAL.get(m)
        ]
        concept_plan = _normalize_export_concepts(
            sel_modules, concepts, CONCEPT_GROUPS_INTERNAL
        )
        sel = [m for m in sel_modules if concept_plan.get(m)]
        ext = _EXPORT_EXT.get(export_format, "csv")
        out = _resolve_export_out_dir(
            out_dir=out_dir,
            database=database,
            export_format=export_format,
            create_run_subdir=create_run_subdir,
        )
        out.mkdir(parents=True, exist_ok=True)

        # Select ONE cohort up front and pass the same patient_ids to every
        # module, so all files share a consistent cohort. The native UI sends a
        # cohort contract; unsupported filters fail closed instead of silently
        # exporting the full database.
        cohort_progress_total = max(1, len(sel) + 1)

        def emit_cohort_progress(stage: str, extra: Dict[str, Any]) -> None:
            # Only aggregate counts and configuration labels are emitted; never
            # patient/stay identifiers or row-level concept payloads.
            job.emit(
                {
                    "type": "progress",
                    "phase": "cohort",
                    "current": 0,
                    "total": cohort_progress_total,
                    "module": "cohort",
                    "stage": stage,
                    "message": _COHORT_PROGRESS_MESSAGES.get(
                        stage, stage.replace("_", " ")
                    ),
                    **extra,
                }
            )

        cohort_info = _resolve_export_cohort(
            str(data_path),
            database,
            cohort,
            max_patients,
            api,
            progress=emit_cohort_progress,
        )
        patient_ids = cohort_info["patient_ids"]
        cohort_size = cohort_info["cohort_size"]
        load_kwargs = dict(cohort_info.get("load_kwargs") or {})
        sepsis_load_kwargs = dict(cohort_info.get("sepsis_load_kwargs") or {})
        if getattr(job, "cancel_requested", False):
            return {
                "out_dir": str(out),
                "files": [],
                "file_count": 0,
                "total_rows": 0,
                "manifest": None,
                "cancelled_at": "cohort",
            }

        job.emit(
            {
                "type": "start",
                "modules": sel,
                "out_dir": str(out),
                "concepts": {module: len(concept_plan[module]) for module in sel},
                "format": export_format,
                "max_patients": max_patients,
                "cohort_size": cohort_size,
                "cohort": cohort_info.get("cohort_report"),
            }
        )
        job.emit(
            {
                "type": "progress",
                "phase": "cohort",
                "current": 1,
                "total": cohort_progress_total,
                "module": "cohort",
                "stage": "ready",
                "message": "Cohort resolved",
                "cohort_size": cohort_size,
            }
        )

        current_manifest = out / "_manifest.json"
        if current_manifest.exists() or current_manifest.is_symlink():
            if current_manifest.is_symlink() or not current_manifest.is_file():
                raise ExportCohortError(
                    "existing_export_manifest_invalid",
                    {"path": str(current_manifest)},
                )
            current_manifest.unlink()

        files: List[Dict[str, Any]] = []
        metadata_file_bindings: List[Any] = []
        definition_files: List[Dict[str, Any]] = []
        definition_payload: Optional[Dict[str, Any]] = None
        metadata_database = str(database).strip().lower()
        metadata_dictionary = load_dictionary(include_sofa2=True)
        metadata_source_config = load_src_cfg(metadata_database)
        metadata_class_prefixes = tuple(
            str(value).strip().lower()
            for value in metadata_source_config.class_prefix
            if str(value).strip()
        )
        total = len(sel)
        with api.keep_cache(database=database, data_path=str(data_path)):
            for i, mod in enumerate(sel, start=1):
                if getattr(job, "cancel_requested", False):
                    break
                module_concepts = concept_plan[mod]
                try:
                    load_plan = compile_concept_load_plan(module_concepts)
                except ConceptLoadPlanError as exc:
                    raise ExportCohortError(
                        "concept_output_load_plan_invalid",
                        {
                            "module": mod,
                            "reason": exc.reason_code,
                            "position": exc.position,
                        },
                    ) from exc
                use_sofa2 = any(
                    c.startswith("sofa2") or c == "sep3_sofa2" for c in module_concepts
                )
                module_kwargs = dict(load_kwargs)
                if _module_uses_sepsis_kwargs(list(load_plan.source_concepts)):
                    module_kwargs.update(sepsis_load_kwargs)
                df = api.load_concepts(
                    list(load_plan.source_concepts),
                    patient_ids=patient_ids,
                    database=database,
                    data_path=str(data_path),
                    use_sofa2=use_sofa2,
                    merge=True,
                    verbose=False,
                    **module_kwargs,
                )
                if isinstance(df, dict):
                    df = {
                        key: _materialize_concept_load_plan(sub, load_plan)
                        for key, sub in df.items()
                    }
                else:
                    df = _materialize_concept_load_plan(df, load_plan)
                written: List[Dict[str, Any]] = []
                if isinstance(df, dict):
                    for key, sub in df.items():
                        fname = f"{mod}__{key}.{ext}"
                        binding = _build_export_file_metadata_binding(
                            relative_path=fname,
                            module=mod,
                            frame=sub,
                            concept_ids=module_concepts,
                            database=metadata_database,
                            database_class_prefixes=metadata_class_prefixes,
                            dictionary=metadata_dictionary,
                        )
                        rows = _write_frame(sub, out / fname, export_format)
                        written.append(
                            {
                                "file": fname,
                                "module": mod,
                                "concepts": len(module_concepts),
                                "concept_ids": list(module_concepts),
                                "rows": rows,
                            }
                        )
                        written[-1]["column_metadata_columns"] = list(binding.columns)
                        metadata_file_bindings.append(binding)
                else:
                    fname = f"{mod}.{ext}"
                    binding = _build_export_file_metadata_binding(
                        relative_path=fname,
                        module=mod,
                        frame=df,
                        concept_ids=module_concepts,
                        database=metadata_database,
                        database_class_prefixes=metadata_class_prefixes,
                        dictionary=metadata_dictionary,
                    )
                    rows = _write_frame(df, out / fname, export_format)
                    written.append(
                        {
                            "file": fname,
                            "module": mod,
                            "concepts": len(module_concepts),
                            "concept_ids": list(module_concepts),
                            "rows": rows,
                        }
                    )
                    written[-1]["column_metadata_columns"] = list(binding.columns)
                    metadata_file_bindings.append(binding)
                files.extend(written)
                job.emit(
                    {
                        "type": "progress",
                        "current": i,
                        "total": total,
                        "module": mod,
                        "file": written[0]["file"],
                        "rows": sum(w["rows"] for w in written),
                    }
                )

        if getattr(job, "cancel_requested", False):
            return {
                "out_dir": str(out),
                "files": files,
                "file_count": len(files),
                "total_rows": sum(f["rows"] for f in files),
                "manifest": None,
                "cancelled_at": "modules",
            }

        missing_primary_metadata = _missing_primary_metadata_concepts(
            concept_plan={module: concept_plan[module] for module in sel},
            file_bindings=metadata_file_bindings,
        )
        unavailable_concepts, unexplained_missing = (
            _classify_structurally_unavailable_concepts(
                concepts=missing_primary_metadata,
                concept_plan={module: concept_plan[module] for module in sel},
                database=database,
            )
        )
        if unexplained_missing:
            raise ExportCohortError(
                "column_metadata_primary_binding_missing",
                {"concepts": unexplained_missing},
            )

        if include_feature_definitions:
            definition_payload = _feature_definition_payload(
                database=database,
                data_path=str(data_path),
                export_path=out,
                concept_plan={module: concept_plan[module] for module in sel},
                files=files,
                api_module=api,
                unavailable_concepts=unavailable_concepts,
            )
            definition_files = _write_feature_definition_files(out, definition_payload)

        metadata_sidecar = ColumnMetadataSidecar(
            source_database=metadata_database,
            source_database_class_prefixes=metadata_class_prefixes,
            scope=EXPORT_PHYSICAL_SCOPE,
            files=tuple(metadata_file_bindings),
        )
        metadata_ref = write_content_addressed_sidecar(out, metadata_sidecar)

        manifest = {
            "schema_version": _NATIVE_EXPORT_SCHEMA_V2,
            "database": database,
            "data_path": str(data_path),
            "format": export_format,
            "max_patients": max_patients,
            "export_folder": {
                "path": str(out),
                "run_subdir": bool(create_run_subdir),
                "label": out.name,
            },
            "cohort_contract": cohort_info.get("cohort_contract"),
            "cohort_report": cohort_info.get("cohort_report"),
            "concept_selection": {
                "mode": (
                    "explicit" if concepts is not None else "all_in_selected_modules"
                ),
                "modules": {module: concept_plan[module] for module in sel},
            },
            "concept_availability": {
                "structurally_unavailable_count": len(unavailable_concepts),
                "structurally_unavailable": unavailable_concepts,
            },
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "files": files,
            "feature_definitions": (
                {
                    "included": True,
                    "schema_version": definition_payload.get("schema_version"),
                    "record_count": definition_payload.get("record_count"),
                    "raw_lineage_scope": definition_payload.get("raw_lineage_scope"),
                    "files": definition_files,
                }
                if definition_payload
                else {"included": False}
            ),
            "column_metadata": metadata_ref.to_dict(),
        }
        (out / "_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )
        (out / "README.md").write_text(
            _render_export_readme(
                manifest, files=files, definition_files=definition_files
            ),
            encoding="utf-8",
        )
        return {
            "out_dir": str(out),
            "files": files,
            "definition_files": definition_files,
            "file_count": len(files),
            "total_rows": sum(f["rows"] for f in files),
            "manifest": "_manifest.json",
            "readme": "README.md",
            "feature_definitions": (
                "feature_definitions.json" if definition_files else None
            ),
            "feature_definitions_csv": (
                "feature_definitions.csv" if definition_files else None
            ),
            "column_metadata": metadata_ref.file,
            "column_metadata_sha256": metadata_ref.sha256,
        }

    return runner


def summarize_export_workspace(raw_path: str) -> Dict[str, Any]:
    """Summarise an EasyICU module-export folder for visualization screens.

    This is the Stage-4 bridge: patient/cohort/cross-db views should consume a
    bounded, UI-safe snapshot instead of static demo numbers. The snapshot is
    intentionally compact and local-only; it reads exported module files from
    disk and returns summary tables, not raw full frames.
    """
    import json

    path = Path(raw_path).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    if not _safe_dir(path):
        return {"ok": False, "error": "not_a_directory", "path": str(path)}

    manifest_path = path / "_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    files = _export_file_inventory(path, manifest)
    module_files = {
        module: next((f for f in files if f.get("module") == module), None)
        for module in (
            "demographics",
            "outcome",
            "vitals",
            "sofa2_score",
            "sepsis3_sofa2",
        )
    }
    id_file = module_files.get("demographics") or next(
        (f for f in files if "stay_id" in (f.get("columns") or [])),
        None,
    )
    if id_file is None:
        return {
            "ok": False,
            "error": "no_stay_id",
            "path": str(path),
            "files": files,
        }
    id_frame = _read_stay_id_frame(path / str(id_file["file"]))
    if id_frame is None or id_frame.empty or "stay_id" not in id_frame.columns:
        return {
            "ok": False,
            "error": "no_stay_id",
            "path": str(path),
            "files": files,
        }
    all_stay_ids = [
        stay_id
        for stay_id in id_frame["stay_id"].map(_norm_id).drop_duplicates().tolist()
        if stay_id
    ]
    total_stays = len(all_stay_ids)
    sampled_id_order = all_stay_ids[:_WORKSPACE_SAMPLE_LIMIT]
    stay_ids = set(sampled_id_order)

    selected_columns = {
        "demographics": ["stay_id", "age", "sex"],
        "outcome": ["stay_id", "death", "los_icu"],
        "vitals": ["stay_id", "charttime", "hr", "map", "spo2", "temp"],
        "sofa2_score": ["stay_id", "sofa2"],
        "sepsis3_sofa2": ["stay_id", "sep3_sofa2"],
    }
    frames: Dict[str, Any] = {}
    for module, hit in module_files.items():
        if hit:
            frames[module] = _read_export_frame(
                path / str(hit["file"]),
                columns=selected_columns[module],
                stay_ids=stay_ids,
            )

    demo = frames.get("demographics")
    if demo is None or demo.empty:
        import pandas as pd

        demo = pd.DataFrame({"stay_id": sampled_id_order})

    demo = demo.copy()
    demo["stay_id"] = demo["stay_id"].map(_norm_id)
    demo = demo[demo["stay_id"].astype(str).str.len() > 0].drop_duplicates("stay_id")
    demo = demo.head(_WORKSPACE_SAMPLE_LIMIT)
    sampled_stays = len(stay_ids)
    snapshot_basis = (
        f"bounded_first_{_WORKSPACE_SAMPLE_LIMIT}_stays"
        if sampled_stays < total_stays
        else "complete_export"
    )

    outcome = _filter_by_stay(frames.get("outcome"), stay_ids)
    sofa2 = _filter_by_stay(frames.get("sofa2_score"), stay_ids)
    sepsis = _filter_by_stay(frames.get("sepsis3_sofa2"), stay_ids)
    vitals = _filter_by_stay(frames.get("vitals"), stay_ids)

    death_by_stay = _stay_bool(outcome, "death", missing_false=True)
    los_by_stay = _stay_numeric(outcome, "los_icu", "median")
    sofa_by_stay = _stay_numeric(sofa2, "sofa2", "max")
    sep_by_stay = _stay_bool(sepsis, "sep3_sofa2", missing_false=True)
    for sid in stay_ids:
        if outcome is not None and not outcome.empty:
            death_by_stay.setdefault(sid, False)
        if sepsis is not None and not sepsis.empty:
            sep_by_stay.setdefault(sid, False)

    cohort_rows = []
    for _, row in demo.iterrows():
        sid = str(row.get("stay_id", ""))
        dead = death_by_stay.get(sid)
        cohort_rows.append(
            {
                "stay_id": sid,
                "age": _num(row.get("age")),
                "sex": _clean(row.get("sex")),
                "sofa2": _num(sofa_by_stay.get(sid)),
                "los_icu": _num(los_by_stay.get(sid)),
                "outcome": (
                    "Deceased"
                    if dead is True
                    else ("Survived" if dead is False else "Unknown")
                ),
            }
        )
    table_rows = cohort_rows[:12]

    first_id = table_rows[0]["stay_id"] if table_rows else next(iter(stay_ids), "")
    patient = (
        next((r for r in table_rows if r["stay_id"] == first_id), {})
        if table_rows
        else {}
    )
    patient = {
        **patient,
        "sepsis3": bool(sep_by_stay.get(first_id)) if first_id in sep_by_stay else None,
    }

    sampled_rows = _cohort_total_rows(path, files, stay_ids)
    manifest_rows = sum(
        int(file_meta.get("rows") or 0) for file_meta in files if file_meta.get("rows")
    )
    summary = {
        "stays": total_stays,
        "total_stays": total_stays,
        "sampled_stays": sampled_stays,
        "sample_limit": _WORKSPACE_SAMPLE_LIMIT,
        "snapshot_basis": snapshot_basis,
        "modules": len({f.get("module") for f in files if f.get("module")}),
        "file_count": len(files),
        "total_rows": manifest_rows or sampled_rows,
        "sampled_rows": sampled_rows,
        "mean_age": _series_mean(demo.get("age")),
        "female_pct": _sex_pct(demo.get("sex"), "female"),
        "mortality": _bool_pct(list(death_by_stay.values())),
        "median_los_icu": _median(list(los_by_stay.values())),
        "median_sofa2": _median(list(sofa_by_stay.values())),
        "sepsis_pct": _bool_pct(list(sep_by_stay.values())),
    }

    cohort = _cohort_summary(cohort_rows)
    cohort.update(
        {
            "total_stays": total_stays,
            "sampled_stays": sampled_stays,
            "snapshot_basis": snapshot_basis,
        }
    )
    quality = [
        _quality_row(path / str(f["file"]), f, stay_ids) for f in files if f.get("file")
    ]

    return {
        "ok": True,
        "path": str(path),
        "database": manifest.get("database"),
        "generated": manifest.get("generated"),
        "files": files,
        "summary": summary,
        "tableRows": table_rows,
        "series": _series_payload(vitals, first_id),
        "patient": patient,
        "quality": quality,
        "cohort": cohort,
    }


def summarize_crossdb_workspaces(raw_paths: List[str]) -> Dict[str, Any]:
    """Summarise two or more local export folders for Cross-DB preview.

    This is deliberately descriptive. It compares already-exported module
    snapshots and their denominators; it does not assert matched cohorts,
    formal cross-database transportability, or inferential statistics.
    """
    paths = []
    seen = set()
    for raw in raw_paths or []:
        path = str(raw or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    if len(paths) < 2:
        return {"ok": False, "error": "need_two_exports", "sources": []}

    sources: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    label_counts: Dict[str, int] = {}
    for path in paths:
        result = summarize_export_workspace(path)
        if not result.get("ok"):
            errors.append(
                {"path": path, "error": result.get("error"), "detail": result}
            )
            continue
        label = _crossdb_label(result)
        count = label_counts.get(label, 0) + 1
        label_counts[label] = count
        if count > 1:
            label = f"{label}-{count}"
        modules = sorted(
            {f.get("module") for f in result.get("files", []) if f.get("module")}
        )
        sources.append(
            {
                "label": label,
                "database": result.get("database"),
                "path": result.get("path"),
                "summary": result.get("summary", {}),
                "modules": modules,
                "files": len(result.get("files", [])),
            }
        )

    if errors:
        return {
            "ok": False,
            "error": "invalid_export",
            "sources": sources,
            "errors": errors,
        }
    if len(sources) < 2:
        return {"ok": False, "error": "need_two_exports", "sources": sources}

    module_sets = [set(s["modules"]) for s in sources]
    shared_modules = sorted(set.intersection(*module_sets)) if module_sets else []
    all_modules = sorted(set.union(*module_sets)) if module_sets else []
    compatibility_gate = _crossdb_compatibility_gate(
        sources, shared_modules, all_modules
    )
    if compatibility_gate["status"] != "compatible":
        return {
            "ok": False,
            "error": "crossdb_incompatible",
            "source_count": len(sources),
            "sources": sources,
            "shared_modules": shared_modules,
            "all_modules": all_modules,
            "compatibility_gate": compatibility_gate,
        }

    rows = []
    comparable_metrics = set(compatibility_gate["comparable_metrics"])
    for key, label, digits, dependency in [
        ("stays", "Stays", 0, "demographics"),
        ("modules", "Modules", 0, None),
        ("total_rows", "Rows", 0, None),
        ("mean_age", "Mean age", 1, "demographics"),
        ("female_pct", "Female %", 1, "demographics"),
        ("mortality", "Mortality %", 1, "outcome"),
        ("sepsis_pct", "Sepsis-3 %", 1, "sepsis3_sofa2"),
        ("median_sofa2", "Median SOFA-2", 1, "sofa2_score"),
    ]:
        if dependency and key not in comparable_metrics:
            continue
        values = [s["summary"].get(key) for s in sources]
        numeric = [float(v) for v in values if isinstance(v, (int, float))]
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

    return {
        "ok": True,
        "source_count": len(sources),
        "sources": sources,
        "rows": rows,
        "shared_modules": shared_modules,
        "all_modules": all_modules,
        "compatibility_gate": compatibility_gate,
    }


def _crossdb_compatibility_gate(
    sources: List[Dict[str, Any]],
    shared_modules: List[str],
    all_modules: List[str],
) -> Dict[str, Any]:
    shared = set(shared_modules)
    required_core = {"demographics", "outcome"}
    optional_modules = {"sepsis3_sofa2", "sofa2_score", "vitals"}
    reasons: List[Dict[str, Any]] = []
    checks: List[Dict[str, Any]] = []

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
                "detail": "At least two valid exports are required.",
            }
        )

    denominators = [
        {"label": s.get("label"), "stays": (s.get("summary") or {}).get("stays")}
        for s in sources
    ]
    denominator_ok = all(
        isinstance(row["stays"], (int, float)) and row["stays"] > 0
        for row in denominators
    )
    checks.append(
        {"id": "denominator_present", "passed": denominator_ok, "sources": denominators}
    )
    if not denominator_ok:
        reasons.append({"id": "missing_denominator", "sources": denominators})

    missing_core = sorted(required_core - shared)
    checks.append(
        {
            "id": "core_modules_shared",
            "passed": not missing_core,
            "required_modules": sorted(required_core),
            "shared_modules": shared_modules,
            "missing_modules": missing_core,
        }
    )
    if missing_core:
        per_source = [
            {
                "label": s.get("label"),
                "missing_core_modules": sorted(
                    required_core - set(s.get("modules") or [])
                ),
            }
            for s in sources
        ]
        reasons.append(
            {
                "id": "core_modules_not_shared",
                "missing_shared_modules": missing_core,
                "sources": per_source,
            }
        )

    comparable_metrics = ["stays", "mean_age", "female_pct", "mortality"]
    if "sepsis3_sofa2" in shared:
        comparable_metrics.append("sepsis_pct")
    if "sofa2_score" in shared:
        comparable_metrics.append("median_sofa2")

    warnings = []
    missing_optional = sorted((optional_modules & set(all_modules)) - shared)
    if missing_optional:
        warnings.append(
            {
                "id": "optional_modules_not_shared",
                "modules": missing_optional,
                "effect": "dependent metrics are omitted from descriptive rows",
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


def describe_export_source(raw_path: str) -> Dict[str, Any]:
    """Return a compact registry-safe description of an export folder.

    Unlike :func:`summarize_export_workspace`, this reads only manifest metadata,
    parquet/csv schemas, and a stay-id column only when no manifest denominator
    is available. It does not load full export tables for registry rendering.
    """
    path = Path(raw_path).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    if not _safe_dir(path):
        return {"ok": False, "error": "not_a_directory", "path": str(path)}

    manifest = _read_export_manifest(path)

    files = _export_file_inventory(path, manifest)
    if not files:
        return {"ok": False, "error": "no_export_files", "path": str(path), "files": []}

    modules = sorted({f.get("module") for f in files if f.get("module")})
    label = str(manifest.get("database") or path.name or "local").upper()
    manifest_stays = _manifest_stay_count(manifest)
    stay_ids = None if manifest_stays is not None else _fast_stay_ids(path, files)
    summary = {
        "stays": (
            manifest_stays
            if manifest_stays is not None
            else (len(stay_ids) if stay_ids is not None else None)
        ),
        "modules": len(modules),
        "file_count": len(files),
        "total_rows": sum(int(f.get("rows") or 0) for f in files),
    }
    return {
        "ok": True,
        "path": str(path),
        "label": label,
        "database": manifest.get("database"),
        "generated": manifest.get("generated") or manifest.get("exported_at"),
        "modules": modules,
        "files": files,
        "summary": summary,
    }


def _read_export_manifest(path: Path) -> Dict[str, Any]:
    import json

    for name in ("_manifest.json", "easyicu_export_manifest.json"):
        manifest_path = path / name
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if isinstance(manifest, dict):
            manifest["_manifest_file"] = name
            return manifest
    return {}


def _manifest_file_entries(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    files = manifest.get("files")
    if isinstance(files, list):
        return [dict(row) for row in files if isinstance(row, dict) and row.get("file")]

    # Older full exports wrote ``easyicu_export_manifest.json`` with a short
    # ``modules`` list. Treat those rows as hints, but still scan the folder
    # because the manifest may be incomplete.
    modules = manifest.get("modules")
    if not isinstance(modules, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in modules:
        if not isinstance(row, dict) or not row.get("file"):
            continue
        entry = dict(row)
        if "module" not in entry and row.get("group"):
            entry["module"] = row.get("group")
        out.append(entry)
    return out


def _manifest_stay_count(manifest: Dict[str, Any]) -> Optional[int]:
    cohort_report = manifest.get("cohort_report")
    if isinstance(cohort_report, dict):
        for key in ("selected", "cohort_size", "stays"):
            value = _positive_int(cohort_report.get(key))
            if value is not None:
                return value
    for key in ("patient_count", "cohort_size", "stays"):
        value = _positive_int(manifest.get(key))
        if value is not None:
            return value
    return None


def _positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _export_file_inventory(
    path: Path, manifest: Dict[str, Any]
) -> List[Dict[str, Any]]:
    manifest_files = {
        f.get("file"): f for f in _manifest_file_entries(manifest) if f.get("file")
    }
    out: List[Dict[str, Any]] = []
    for f in sorted(path.iterdir(), key=lambda p: p.name):
        if (
            f.name.startswith(".")
            or f.name in _MODULE_MANIFESTS
            or f.name in _EXPORT_METADATA_FILES
            or not f.is_file()
        ):
            continue
        if f.suffix.lower() not in {".csv", ".parquet", ".xlsx"}:
            continue
        meta = dict(manifest_files.get(f.name, {}))
        if "file" not in meta:
            meta["file"] = f.name
        if "module" not in meta:
            meta["module"] = _infer_export_module(f)
        if "rows" not in meta:
            meta["rows"] = _count_rows(f)
        meta["columns"] = _read_columns(f)
        out.append(meta)
    return out


def _infer_export_module(path: Path) -> str:
    stem = path.stem
    try:
        from easyicu.concept import catalog as concept_catalog

        groups = sorted(concept_catalog.CONCEPT_GROUPS_INTERNAL, key=len, reverse=True)
    except Exception:
        groups = []
    for group in groups:
        if (
            stem == group
            or stem.startswith(group + "_")
            or stem.startswith(group + "__")
        ):
            return group
    return stem.split("__", 1)[0]


def _fast_stay_count(path: Path, files: List[Dict[str, Any]]) -> Optional[int]:
    stay_ids = _fast_stay_ids(path, files)
    return len(stay_ids) if stay_ids is not None else None


def _fast_stay_ids(path: Path, files: List[Dict[str, Any]]) -> Optional[set[str]]:
    hit = next((f for f in files if f.get("module") == "demographics"), None)
    if hit is None:
        hit = next((f for f in files if "stay_id" in (f.get("columns") or [])), None)
    if hit is None:
        return None
    return _read_stay_ids(path / str(hit["file"]))


def _crossdb_label(result: Dict[str, Any]) -> str:
    db = str(result.get("database") or "").strip()
    if db:
        return db.upper()
    path = Path(str(result.get("path") or "local"))
    return path.name or "local"


def _read_export_frame(
    path: Path,
    *,
    columns: Optional[List[str]] = None,
    stay_ids: Optional[set[str]] = None,
) -> Any:
    return _read_export_projection(path, columns=columns, stay_ids=stay_ids)


def _read_stay_id_frame(
    path: Path,
    *,
    stay_ids: Optional[set[str]] = None,
) -> Any:
    return _read_export_projection(
        path,
        columns=["stay_id"],
        stay_ids=stay_ids,
    )


def _read_export_projection(
    path: Path,
    *,
    columns: Optional[List[str]],
    stay_ids: Optional[set[str]],
    entity_column: str = "stay_id",
) -> Any:
    """Read projected export columns, pushing a bounded stay filter when set."""
    import pandas as pd

    suffix = path.suffix.lower()
    available = _read_columns(path)
    selected = (
        [column for column in columns if column in available]
        if columns is not None
        else list(available)
    )
    if (
        stay_ids is not None
        and entity_column in available
        and entity_column not in selected
    ):
        selected.insert(0, entity_column)
    if not selected:
        return pd.DataFrame()
    if stay_ids is not None and not stay_ids:
        return pd.DataFrame(columns=selected)

    if suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.dataset as ds

            dataset = ds.dataset(path, format="parquet")
            filter_expression = None
            if stay_ids is not None:
                if entity_column not in dataset.schema.names:
                    return pd.DataFrame(columns=selected)
                field_type = dataset.schema.field(entity_column).type
                values = pc.cast(pa.array(sorted(stay_ids)), field_type)
                filter_expression = ds.field(entity_column).isin(values)
            return dataset.to_table(
                columns=selected,
                filter=filter_expression,
            ).to_pandas()
        except Exception:
            return _read_export_frame_duckdb(
                path,
                selected,
                stay_ids=stay_ids,
                source="parquet",
                entity_column=entity_column,
            )
    if suffix == ".csv":
        return _read_export_frame_duckdb(
            path,
            selected,
            stay_ids=stay_ids,
            source="csv",
            entity_column=entity_column,
        )
    frame = pd.read_excel(path, usecols=selected)
    if stay_ids is not None and entity_column in frame.columns:
        frame = frame[frame[entity_column].map(_norm_id).isin(stay_ids)].copy()
    return frame


def read_export_projection(
    path: Path,
    *,
    columns: Optional[List[str]] = None,
    stay_ids: Optional[set[str]] = None,
    entity_column: str = "stay_id",
) -> Any:
    """Public, projected export reader for bounded webserver owners."""

    return _read_export_projection(
        path,
        columns=columns,
        stay_ids=stay_ids,
        entity_column=entity_column,
    )


def _read_export_frame_duckdb(
    path: Path,
    columns: List[str],
    *,
    stay_ids: Optional[set[str]],
    source: str,
    entity_column: str = "stay_id",
) -> Any:
    import duckdb

    projection = ", ".join(
        f'"{column.replace(chr(34), chr(34) * 2)}"' for column in columns
    )
    reader = (
        "read_parquet(?)" if source == "parquet" else "read_csv_auto(?, header=true)"
    )
    query = f"SELECT {projection} FROM {reader}"
    params: List[Any] = [str(path)]
    if stay_ids is not None:
        escaped_entity = entity_column.replace('"', '""')
        query += f' WHERE CAST("{escaped_entity}" AS VARCHAR) IN (SELECT UNNEST(?))'
        params.append(sorted(stay_ids))
    connection = duckdb.connect(database=":memory:")
    try:
        return connection.execute(query, params).fetch_df()
    finally:
        connection.close()


def _read_columns(path: Path) -> List[str]:
    try:
        import pandas as pd

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return list(pd.read_csv(path, nrows=0).columns)
        if suffix == ".xlsx":
            return list(pd.read_excel(path, nrows=0).columns)
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq

                return list(pq.ParquetFile(path).schema_arrow.names)
            except Exception:
                return list(pd.read_parquet(path).columns)
    except Exception:
        return []
    return []


def _count_rows(path: Path) -> int:
    try:
        if path.suffix.lower() == ".csv":
            with path.open("rb") as fh:
                return max(sum(1 for _ in fh) - 1, 0)
        if path.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq

                return int(pq.ParquetFile(path).metadata.num_rows)
            except Exception:
                pass
        return int(_read_export_frame(path).shape[0])
    except Exception:
        return 0


def _read_stay_ids(
    path: Path,
    stay_ids: Optional[set[str]] = None,
) -> Optional[set[str]]:
    try:
        frame = _read_stay_id_frame(path, stay_ids=stay_ids)
    except Exception:
        return None
    if "stay_id" not in frame.columns:
        return None
    return {sid for sid in frame["stay_id"].map(_norm_id).dropna().astype(str) if sid}


def _norm_id(value: Any) -> str:
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return "" if value is None else str(value)


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _num(value: Any) -> Optional[float]:
    try:
        import pandas as pd

        if pd.isna(value):
            return None
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _filter_by_stay(frame: Any, stay_ids: set[str]) -> Any:
    if frame is None or frame.empty or "stay_id" not in frame.columns:
        return frame
    tmp = frame.copy()
    tmp["stay_id"] = tmp["stay_id"].map(_norm_id)
    return tmp[tmp["stay_id"].isin(stay_ids)]


def _truthy(value: Any) -> Optional[bool]:
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric == 1:
        return True
    if numeric == 0:
        return False
    return None


def _stay_bool(frame: Any, column: str, missing_false: bool = False) -> Dict[str, bool]:
    if (
        frame is None
        or frame.empty
        or "stay_id" not in frame.columns
        or column not in frame.columns
    ):
        return {}
    out: Dict[str, bool] = {}
    for sid, vals in frame.groupby("stay_id")[column]:
        flags = [_truthy(v) for v in vals]
        if missing_false:
            flags = [False if v is None else v for v in flags]
        flags = [v for v in flags if v is not None]
        if flags:
            out[str(sid)] = any(flags)
    return out


def _stay_numeric(frame: Any, column: str, mode: str) -> Dict[str, float]:
    if (
        frame is None
        or frame.empty
        or "stay_id" not in frame.columns
        or column not in frame.columns
    ):
        return {}
    numeric = frame[["stay_id", column]].copy()
    import pandas as pd

    numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric = numeric.dropna(subset=[column])
    if numeric.empty:
        return {}
    grouped = numeric.groupby("stay_id")[column]
    values = grouped.max() if mode == "max" else grouped.median()
    return {str(k): float(v) for k, v in values.items()}


def _series_mean(series: Any) -> Optional[float]:
    vals = _numeric_values(series)
    return round(sum(vals) / len(vals), 2) if vals else None


def _median(vals: List[Any]) -> Optional[float]:
    xs = sorted(v for v in (_num(v) for v in vals) if v is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    val = xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2
    return round(val, 2)


def _numeric_values(series: Any) -> List[float]:
    if series is None:
        return []
    vals = []
    for v in series:
        n = _num(v)
        if n is not None:
            vals.append(float(n))
    return vals


def _sex_pct(series: Any, target: str) -> Optional[float]:
    if series is None:
        return None
    vals = [str(v).lower() for v in series if _clean(v)]
    if not vals:
        return None
    hits = sum(1 for v in vals if target in v or (target == "female" and v == "f"))
    return round(hits / len(vals) * 100, 1)


def _bool_pct(vals: List[Any]) -> Optional[float]:
    flags = [v for v in vals if v is not None]
    if not flags:
        return None
    return round(sum(1 for v in flags if v) / len(flags) * 100, 1)


def _cohort_summary(table_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    survived = [r for r in table_rows if r.get("outcome") == "Survived"]
    deceased = [r for r in table_rows if r.get("outcome") == "Deceased"]

    def mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    characteristics = [
        [
            "Age, mean",
            mean(table_rows, "age"),
            mean(survived, "age"),
            mean(deceased, "age"),
        ],
        [
            "SOFA-2, mean",
            mean(table_rows, "sofa2"),
            mean(survived, "sofa2"),
            mean(deceased, "sofa2"),
        ],
        [
            "ICU LOS, mean",
            mean(table_rows, "los_icu"),
            mean(survived, "los_icu"),
            mean(deceased, "los_icu"),
        ],
    ]
    return {
        "survived": len(survived),
        "deceased": len(deceased),
        "characteristics": characteristics,
    }


def _cohort_total_rows(
    path: Path, files: List[Dict[str, Any]], stay_ids: set[str]
) -> int:
    total = 0
    for file_meta in files:
        file_name = file_meta.get("file")
        if not file_name:
            continue
        total += _count_matching_stay_rows(path / str(file_name), stay_ids)
    return total


def _count_matching_stay_rows(path: Path, stay_ids: set[str]) -> int:
    try:
        frame = _read_stay_id_frame(path, stay_ids=stay_ids)
    except Exception:
        return 0
    if "stay_id" not in frame.columns:
        return 0
    return int(frame["stay_id"].map(_norm_id).isin(stay_ids).sum())


def _quality_row(
    path: Path, file_meta: Dict[str, Any], stay_ids: set[str]
) -> Dict[str, Any]:
    module = file_meta.get("module")
    rows = int(file_meta.get("rows") or 0)
    denominator = len(stay_ids)
    file_stays = _read_stay_ids(path, stay_ids=stay_ids)
    unique_stays = len(file_stays & stay_ids) if file_stays is not None else None
    coverage = (
        round(unique_stays / denominator * 100, 1)
        if denominator and unique_stays is not None
        else None
    )
    if coverage is None:
        status = "unknown"
    elif _is_presence_rate_module(module):
        status = "neutral"
    else:
        status = "ok" if coverage >= 80 else ("warn" if coverage >= 50 else "bad")
    return {
        "module": module,
        "metric_kind": _presence_rate_kind(module) or "coverage",
        "file": file_meta.get("file"),
        "rows": rows,
        "columns": len(file_meta.get("columns") or []),
        "unique_stays": unique_stays,
        "coverage_pct": coverage,
        "coverage_basis": "unique_stay_id_intersection",
        "denominator": denominator,
        "status": status,
    }


def _series_payload(vitals: Any, stay_id: str) -> List[Dict[str, Any]]:
    if vitals is None or vitals.empty or "stay_id" not in vitals.columns:
        return []
    one = vitals[vitals["stay_id"].astype(str) == str(stay_id)].copy()
    if "charttime" in one.columns:
        one = one.sort_values("charttime")
    specs = [
        ("hr", "Heart rate", "bpm"),
        ("map", "MAP", "mmHg"),
        ("spo2", "SpO2", "%"),
        ("temp", "Temp", "deg C"),
    ]
    out = []
    for col, name, unit in specs:
        if col not in one.columns:
            continue
        vals = _numeric_values(one[col].head(12))
        if vals:
            out.append(
                {
                    "key": col,
                    "name": name,
                    "unit": unit,
                    "current": vals[-1],
                    "values": vals,
                }
            )
    return out


def _check_data_status(path: Path, db_key: str) -> Dict[str, Any]:
    """Count prepared parquet (flat + shard dirs) vs raw csv, and judge ready.

    Pure pathlib implementation kept here so the backend does not transitively
    import a UI runtime.
    """
    parquet_files = [f for f in path.glob("*.parquet") if not f.name.startswith(".")]
    parquet_names = [f.stem for f in parquet_files]
    parquet_count = len(parquet_files)

    # Shard directories (e.g. chartevents/1.parquet) count as one table each.
    try:
        for sub in path.iterdir():
            if sub.is_dir() and next(sub.glob("[0-9]*.parquet"), None) is not None:
                parquet_count += 1
                parquet_names.append(sub.name)
    except OSError:
        pass

    csv_files = [
        f
        for f in list(path.glob("*.csv")) + list(path.glob("*.csv.gz"))
        if not f.name.startswith(".")
    ]

    required = _CORE_TABLES.get(db_key, [])
    found = {n.lower() for n in parquet_names}
    ready = False
    missing: List[str] = []
    if parquet_count > 0:
        missing = [t for t in required if t not in found]
        ready = len(missing) <= 1  # tolerate one missing core table
    return {
        "parquet_count": parquet_count,
        "csv_count": len(csv_files),
        "ready": ready,
        "missing_tables": [] if ready else missing,
    }


def _mappable_modules() -> int:
    """Number of feature-module groups EasyICU can map (the 19 catalog groups)."""
    try:
        from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL

        return len(CONCEPT_GROUPS_INTERNAL)
    except Exception:
        return 19


def _estimate_size(path: Path) -> Optional[str]:
    """Rough total size of raw csv(.gz) files, formatted for display."""
    total = 0
    try:
        for f in list(path.glob("*.csv")) + list(path.glob("*.csv.gz")):
            try:
                total += f.stat().st_size
            except OSError:
                continue
    except OSError:
        return None
    if total <= 0:
        return None
    gb = total / (1024**3)
    if gb >= 1:
        return f"~{gb:.1f} GB"
    mb = total / (1024**2)
    return f"~{mb:.0f} MB"
