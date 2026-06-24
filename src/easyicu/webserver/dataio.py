"""Local filesystem + data-folder inspection for the Data Extraction screen.

Two net-new capabilities the design mock did not have (it used hardcoded
``DETECTED`` data and a no-op Browse button):

- :func:`list_dir` — a server-side directory browser. A browser ``<input
  type=file>`` can only upload files, never enumerate the user's folders, so
  the local-first FastAPI process lists directories on demand for the picker.
- :func:`scan_path` — points the existing extraction logic at a folder and
  reports the database / layout / readiness. Wraps the *pure* parts of
  ``webapp.data_workflows.check_data_status`` plus a light database heuristic
  (mirrors ``DataConverter._detect_database`` without constructing one).

Everything runs locally; nothing is uploaded. The Streamlit converter wrapper
``convert_data_with_progress`` is intentionally NOT reused here — it renders
``st.progress``/``st.balloons``. Conversion is a later SSE job that drives
``DataConverter.convert_all(progress_callback=...)`` directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# Core metadata tables per database — a folder that holds these (as parquet or
# csv) is recognised as that database. Mirrors check_data_status' core_tables.
_CORE_TABLES = {
    "miiv": ["icustays", "patients", "admissions"],
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


def _safe_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def list_dir(raw_path: Optional[str]) -> Dict[str, Any]:
    """List immediate sub-directories of ``raw_path`` for the folder picker.

    When ``raw_path`` is empty/None we start from the user's home and surface
    mounted volumes (macOS ``/Volumes``) so external drives are reachable —
    that is where ICU dumps usually live on this machine.
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
    """Light DB heuristic — mirrors DataConverter._detect_database without
    constructing a converter (which would validate/scan the folder)."""
    s = str(path).lower()
    if "eicu" in s:
        return "eicu"
    if "miiv" in s or "mimic" in s:
        return "miiv"
    if "aumc" in s:
        return "aumc"
    if "hirid" in s:
        return "hirid"
    if "sicdb" in s or "sic" in s:
        return "sicdb"

    names = []
    try:
        names = [p.name.lower() for p in list(path.glob("*.csv*")) + list(path.glob("*.parquet"))]
        for sub in path.iterdir():
            if sub.is_dir():
                names.append(sub.name.lower() + "/")
    except OSError:
        pass
    if any(n.startswith("patient.") or n == "patient/" for n in names):
        return "eicu"
    if any(n.startswith("admissions.") or n == "admissions/" for n in names):
        return "miiv"
    if any(n.startswith("general_table") or n.startswith("general/") for n in names):
        return "hirid"
    return "unknown"


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

    db_key = _detect_database(path)
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
        tables = parquet_count
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
        source = "unknown"
        layout = ["No recognized tables", "未识别到数据表"]
        ready = False
        tables = 0

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
    }
    if source == "raw":
        result["size_hint"] = _estimate_size(path)
    return result


def make_convert_runner(raw_path: str, database: str) -> Any:
    """Build a job runner that converts a raw folder to Parquet, emitting one
    progress event per file. Drives ``DataConverter.convert_all`` directly —
    NOT the Streamlit ``convert_data_with_progress`` wrapper (which renders
    ``st.progress``). convert_all is idempotent: already-converted files are
    skipped, so a re-run finishes fast."""

    def runner(job: Any) -> Dict[str, Any]:
        from easyicu.data_converter import ConversionStatus, DataConverter

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
            job.emit({
                "type": "progress",
                "current": info.get("current"),
                "total": info.get("total"),
                "file": info.get("file"),
                "status": st,
                "rows": res.get("row_count"),
                "shards": res.get("shards"),
                "error": res.get("error"),
                "counts": dict(counts),
            })

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
_EVENT_PRESENCE_MODULES = {"sepsis3_sofa2"}


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


def make_export_runner(
    data_path: str,
    database: str,
    modules: Optional[List[str]] = None,
    export_format: str = "csv",
    merge: bool = False,
    out_dir: Optional[str] = None,
    max_patients: Optional[int] = None,
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
        from easyicu.concept_catalog import CONCEPT_GROUPS_INTERNAL

        sel = [m for m in (modules or list(CONCEPT_GROUPS_INTERNAL.keys()))
               if CONCEPT_GROUPS_INTERNAL.get(m)]
        ext = _EXPORT_EXT.get(export_format, "csv")
        out = Path(out_dir).expanduser() if out_dir else (Path.home() / "easyicu" / "exports" / database)
        out.mkdir(parents=True, exist_ok=True)

        # Select ONE cohort up front and pass the same patient_ids to every
        # module, so all files share a consistent cohort. (max_patients alone is
        # a per-module sampling hint some concept loaders ignore — e.g. outcome
        # returned the full DB — which would desync the export.)
        patient_ids = None
        cohort_size = None
        if max_patients:
            # get_all_patient_ids -> (ids_list, id_column_name)
            ids_list, id_col = api.get_all_patient_ids(
                str(data_path), database=database, max_patients=max_patients)
            patient_ids = {id_col: list(ids_list)}
            cohort_size = len(ids_list)

        job.emit({"type": "start", "modules": sel, "out_dir": str(out),
                  "format": export_format, "max_patients": max_patients,
                  "cohort_size": cohort_size})

        files: List[Dict[str, Any]] = []
        total = len(sel)
        with api.keep_cache(database=database, data_path=str(data_path)):
            for i, mod in enumerate(sel, start=1):
                concepts = CONCEPT_GROUPS_INTERNAL[mod]
                use_sofa2 = any(c.startswith("sofa2") or c == "sep3_sofa2" for c in concepts)
                df = api.load_concepts(
                    concepts, patient_ids=patient_ids, database=database,
                    data_path=str(data_path), use_sofa2=use_sofa2,
                    merge=True, verbose=False,
                )
                written: List[Dict[str, Any]] = []
                if isinstance(df, dict):
                    for key, sub in df.items():
                        fname = f"{mod}__{key}.{ext}"
                        rows = _write_frame(sub, out / fname, export_format)
                        written.append({"file": fname, "module": mod, "rows": rows})
                else:
                    fname = f"{mod}.{ext}"
                    rows = _write_frame(df, out / fname, export_format)
                    written.append({"file": fname, "module": mod,
                                    "concepts": len(concepts), "rows": rows})
                files.extend(written)
                job.emit({"type": "progress", "current": i, "total": total, "module": mod,
                          "file": written[0]["file"], "rows": sum(w["rows"] for w in written)})

        manifest = {
            "database": database,
            "data_path": str(data_path),
            "format": export_format,
            "max_patients": max_patients,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "files": files,
        }
        (out / "_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        return {
            "out_dir": str(out),
            "files": files,
            "file_count": len(files),
            "total_rows": sum(f["rows"] for f in files),
            "manifest": "_manifest.json",
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
    frames: Dict[str, Any] = {}
    for module in ("demographics", "outcome", "vitals", "sofa2_score", "sepsis3_sofa2"):
        hit = next((f for f in files if f.get("module") == module), None)
        if hit:
            frames[module] = _read_export_frame(path / str(hit["file"]))

    demo = frames.get("demographics")
    if demo is None or demo.empty:
        first = next((f for f in frames.values() if f is not None and not f.empty), None)
        if first is None or "stay_id" not in first.columns:
            return {"ok": False, "error": "no_stay_id", "path": str(path), "files": files}
        demo = first[["stay_id"]].drop_duplicates().head(100).copy()

    demo = demo.copy()
    demo["stay_id"] = demo["stay_id"].map(_norm_id)
    demo = demo.drop_duplicates("stay_id").head(500)
    stay_ids = {sid for sid in demo["stay_id"].dropna().astype(str) if sid}

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
        cohort_rows.append({
            "stay_id": sid,
            "age": _num(row.get("age")),
            "sex": _clean(row.get("sex")),
            "sofa2": _num(sofa_by_stay.get(sid)),
            "los_icu": _num(los_by_stay.get(sid)),
            "outcome": "Deceased" if dead is True else ("Survived" if dead is False else "Unknown"),
        })
    table_rows = cohort_rows[:12]

    first_id = table_rows[0]["stay_id"] if table_rows else next(iter(stay_ids), "")
    patient = next((r for r in table_rows if r["stay_id"] == first_id), {}) if table_rows else {}
    patient = {
        **patient,
        "sepsis3": bool(sep_by_stay.get(first_id)) if first_id in sep_by_stay else None,
    }

    summary = {
        "stays": len(stay_ids),
        "modules": len({f.get("module") for f in files if f.get("module")}),
        "file_count": len(files),
        "total_rows": _cohort_total_rows(path, files, stay_ids),
        "mean_age": _series_mean(demo.get("age")),
        "female_pct": _sex_pct(demo.get("sex"), "female"),
        "mortality": _bool_pct(list(death_by_stay.values())),
        "median_los_icu": _median(list(los_by_stay.values())),
        "median_sofa2": _median(list(sofa_by_stay.values())),
        "sepsis_pct": _bool_pct(list(sep_by_stay.values())),
    }

    cohort = _cohort_summary(cohort_rows)
    quality = [_quality_row(path / str(f["file"]), f, stay_ids) for f in files if f.get("file")]

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
            errors.append({"path": path, "error": result.get("error"), "detail": result})
            continue
        label = _crossdb_label(result)
        count = label_counts.get(label, 0) + 1
        label_counts[label] = count
        if count > 1:
            label = f"{label}-{count}"
        modules = sorted({f.get("module") for f in result.get("files", []) if f.get("module")})
        sources.append({
            "label": label,
            "database": result.get("database"),
            "path": result.get("path"),
            "summary": result.get("summary", {}),
            "modules": modules,
            "files": len(result.get("files", [])),
        })

    if errors:
        return {"ok": False, "error": "invalid_export", "sources": sources, "errors": errors}
    if len(sources) < 2:
        return {"ok": False, "error": "need_two_exports", "sources": sources}

    module_sets = [set(s["modules"]) for s in sources]
    shared_modules = sorted(set.intersection(*module_sets)) if module_sets else []
    all_modules = sorted(set.union(*module_sets)) if module_sets else []
    compatibility_gate = _crossdb_compatibility_gate(sources, shared_modules, all_modules)
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
        delta = round(max(numeric) - min(numeric), digits) if len(numeric) >= 2 else None
        rows.append({
            "key": key,
            "label": label,
            "values": values,
            "delta": delta,
            "comparison": "descriptive_range",
        })

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
    checks.append({"id": "source_count", "passed": enough_sources, "value": len(sources), "minimum": 2})
    if not enough_sources:
        reasons.append({"id": "need_two_exports", "detail": "At least two valid exports are required."})

    denominators = [
        {"label": s.get("label"), "stays": (s.get("summary") or {}).get("stays")}
        for s in sources
    ]
    denominator_ok = all(isinstance(row["stays"], (int, float)) and row["stays"] > 0 for row in denominators)
    checks.append({"id": "denominator_present", "passed": denominator_ok, "sources": denominators})
    if not denominator_ok:
        reasons.append({"id": "missing_denominator", "sources": denominators})

    missing_core = sorted(required_core - shared)
    checks.append({
        "id": "core_modules_shared",
        "passed": not missing_core,
        "required_modules": sorted(required_core),
        "shared_modules": shared_modules,
        "missing_modules": missing_core,
    })
    if missing_core:
        per_source = [
            {
                "label": s.get("label"),
                "missing_core_modules": sorted(required_core - set(s.get("modules") or [])),
            }
            for s in sources
        ]
        reasons.append({
            "id": "core_modules_not_shared",
            "missing_shared_modules": missing_core,
            "sources": per_source,
        })

    comparable_metrics = ["stays", "mean_age", "female_pct", "mortality"]
    if "sepsis3_sofa2" in shared:
        comparable_metrics.append("sepsis_pct")
    if "sofa2_score" in shared:
        comparable_metrics.append("median_sofa2")

    warnings = []
    missing_optional = sorted((optional_modules & set(all_modules)) - shared)
    if missing_optional:
        warnings.append({
            "id": "optional_modules_not_shared",
            "modules": missing_optional,
            "effect": "dependent metrics are omitted from descriptive rows",
        })

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
    file schemas, and a stay-id column when available. It does not load full
    export tables for registry rendering.
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
    if not files:
        return {"ok": False, "error": "no_export_files", "path": str(path), "files": []}

    modules = sorted({f.get("module") for f in files if f.get("module")})
    label = str(manifest.get("database") or path.name or "local").upper()
    stay_ids = _fast_stay_ids(path, files)
    summary = {
        "stays": len(stay_ids) if stay_ids is not None else None,
        "modules": len(modules),
        "file_count": len(files),
        "total_rows": (
            _cohort_total_rows(path, files, stay_ids)
            if stay_ids is not None
            else sum(int(f.get("rows") or 0) for f in files)
        ),
    }
    return {
        "ok": True,
        "path": str(path),
        "label": label,
        "database": manifest.get("database"),
        "generated": manifest.get("generated"),
        "modules": modules,
        "files": files,
        "summary": summary,
    }


def _export_file_inventory(path: Path, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    manifest_files = {f.get("file"): f for f in manifest.get("files", []) if f.get("file")}
    out: List[Dict[str, Any]] = []
    for f in sorted(path.iterdir(), key=lambda p: p.name):
        if f.name.startswith(".") or f.name == "_manifest.json" or not f.is_file():
            continue
        if f.suffix.lower() not in {".csv", ".parquet", ".xlsx"}:
            continue
        meta = dict(manifest_files.get(f.name, {}))
        if "file" not in meta:
            meta["file"] = f.name
        if "module" not in meta:
            meta["module"] = f.stem.split("__", 1)[0]
        if "rows" not in meta:
            meta["rows"] = _count_rows(f)
        meta["columns"] = _read_columns(f)
        out.append(meta)
    return out


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


def _read_export_frame(path: Path) -> Any:
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


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


def _read_stay_ids(path: Path) -> Optional[set[str]]:
    try:
        import pandas as pd

        suffix = path.suffix.lower()
        if suffix == ".parquet":
            frame = pd.read_parquet(path, columns=["stay_id"])
        elif suffix == ".xlsx":
            frame = pd.read_excel(path, usecols=["stay_id"])
        else:
            frame = pd.read_csv(path, usecols=["stay_id"])
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
    if frame is None or frame.empty or "stay_id" not in frame.columns or column not in frame.columns:
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
    if frame is None or frame.empty or "stay_id" not in frame.columns or column not in frame.columns:
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
        ["Age, mean", mean(table_rows, "age"), mean(survived, "age"), mean(deceased, "age")],
        ["SOFA-2, mean", mean(table_rows, "sofa2"), mean(survived, "sofa2"), mean(deceased, "sofa2")],
        ["ICU LOS, mean", mean(table_rows, "los_icu"), mean(survived, "los_icu"), mean(deceased, "los_icu")],
    ]
    return {
        "survived": len(survived),
        "deceased": len(deceased),
        "characteristics": characteristics,
    }


def _cohort_total_rows(path: Path, files: List[Dict[str, Any]], stay_ids: set[str]) -> int:
    total = 0
    for file_meta in files:
        file_name = file_meta.get("file")
        if not file_name:
            continue
        total += _count_matching_stay_rows(path / str(file_name), stay_ids)
    return total


def _count_matching_stay_rows(path: Path, stay_ids: set[str]) -> int:
    try:
        import pandas as pd

        suffix = path.suffix.lower()
        if suffix == ".parquet":
            frame = pd.read_parquet(path, columns=["stay_id"])
        elif suffix == ".xlsx":
            frame = pd.read_excel(path, usecols=["stay_id"])
        else:
            frame = pd.read_csv(path, usecols=["stay_id"])
    except Exception:
        return 0
    if "stay_id" not in frame.columns:
        return 0
    return int(frame["stay_id"].map(_norm_id).isin(stay_ids).sum())


def _quality_row(path: Path, file_meta: Dict[str, Any], stay_ids: set[str]) -> Dict[str, Any]:
    module = file_meta.get("module")
    rows = int(file_meta.get("rows") or 0)
    denominator = len(stay_ids)
    file_stays = _read_stay_ids(path)
    unique_stays = len(file_stays & stay_ids) if file_stays is not None else None
    coverage = round(unique_stays / denominator * 100, 1) if denominator and unique_stays is not None else None
    if coverage is None:
        status = "unknown"
    elif module in _EVENT_PRESENCE_MODULES:
        status = "neutral"
    else:
        status = "ok" if coverage >= 80 else ("warn" if coverage >= 50 else "bad")
    return {
        "module": module,
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
            out.append({"key": col, "name": name, "unit": unit, "current": vals[-1], "values": vals})
    return out


def _check_data_status(path: Path, db_key: str) -> Dict[str, Any]:
    """Count prepared parquet (flat + shard dirs) vs raw csv, and judge ready.

    Pure pathlib port of ``webapp.data_workflows.check_data_status`` — inlined
    so the new backend never transitively imports Streamlit.
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
        f for f in list(path.glob("*.csv")) + list(path.glob("*.csv.gz"))
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
        from easyicu.concept_catalog import CONCEPT_GROUPS_INTERNAL

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
    gb = total / (1024 ** 3)
    if gb >= 1:
        return f"~{gb:.1f} GB"
    mb = total / (1024 ** 2)
    return f"~{mb:.0f} MB"
