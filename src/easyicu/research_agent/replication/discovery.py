"""Deterministic replication helpers for EasyICU research-agent cases.

This module is intentionally not an agent. It turns one or more
EasyICU concept-export packages into the same analysis cohort and the
same compact result tables. The LLM can plan and write prose around
these outputs, but the cross-database comparison itself stays
deterministic and auditable.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..easyicu_case_builder import EasyICUCasePackage, build_lactate_map_vaso_cohort_from_export


PathLike = Union[str, Path]


LACTATE_MAP_VASO_EXPORT_GROUPS: Mapping[str, Sequence[str]] = {
    "demographics": ("age", "sex", "adm", "bmi", "weight"),
    "outcome": ("death", "los_icu", "los_hosp"),
    "blood_gas": ("lact",),
    "vitals": ("map",),
    "vasopressors": ("vaso_ind", "norepi_equiv"),
    "circulatory": ("circ_event", "circ_failure"),
    "sepsis3": ("sep3_sofa2",),
}

LACTATE_MAP_VASO_MINIMAL_EXPORT_GROUPS: Mapping[str, Sequence[str]] = {
    key: value
    for key, value in LACTATE_MAP_VASO_EXPORT_GROUPS.items()
    if key in {"demographics", "outcome", "blood_gas", "vitals", "vasopressors"}
}


@dataclass(frozen=True)
class ReplicationTarget:
    """One database/export to include in a replication run."""

    database: str
    export_dir: Optional[Path] = None
    label: Optional[str] = None


def _rate_ci(events: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float, float]:
    """Wilson binomial interval returned as (rate, lower, upper)."""
    if n <= 0:
        return (math.nan, math.nan, math.nan)
    p = events / n
    denom = 1.0 + (z * z / n)
    center = (p + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z * z / (4 * n * n)))
    return (p, max(0.0, center - half), min(1.0, center + half))


def _status_row(
    *,
    database: str,
    status: str,
    export_dir: Optional[Path],
    message: str,
) -> Dict[str, object]:
    return {
        "database": database,
        "status": status,
        "export_dir": str(export_dir) if export_dir is not None else "",
        "message": message,
        "n_stays": 0,
        "n_deaths": 0,
        "mortality_rate": math.nan,
        "mortality_ci_lower": math.nan,
        "mortality_ci_upper": math.nan,
        "lactate_measured_n": 0,
        "lactate_measured_pct": math.nan,
        "lactate_measured_mortality": math.nan,
        "lactate_unmeasured_mortality": math.nan,
        "occult_no_vaso_mortality": math.nan,
        "occult_vaso_mortality": math.nan,
        "map_adequate_low_lactate_mortality": math.nan,
        "map_low_mortality": math.nan,
    }


def _safe_slug(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        elif char in {"_", "-", " "}:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "group"


def _guess_database_from_path(path: Path) -> Optional[str]:
    text = str(path).lower()
    for database in ["miiv", "mimic", "eicu", "hirid", "aumc", "sic"]:
        if database in text:
            return database
    return None


def discover_easyicu_exports(
    roots: Sequence[PathLike],
    *,
    required_concepts: Sequence[str] = ("death", "lact", "map", "vaso_ind"),
) -> Dict[str, Path]:
    """Find EasyICU concept-export directories under one or more roots.

    Discovery prefers directories with ``easyicu_export_manifest.json``
    and verifies that required shock-case concepts are present in the
    parquet files before returning a candidate.
    """
    found: Dict[str, Path] = {}
    for root_like in roots:
        root = Path(root_like).expanduser()
        if not root.exists():
            continue
        manifests = list(root.rglob("easyicu_export_manifest.json"))
        if root.name == "easyicu_export_manifest.json":
            manifests.append(root)
        for manifest_path in sorted(set(manifests)):
            export_dir = manifest_path.parent
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                database = str(manifest.get("database") or _guess_database_from_path(export_dir) or "")
                index = build_export_index(export_dir)
                if not set(required_concepts).issubset(index):
                    continue
            except Exception:
                continue
            if database and database not in found:
                found[database] = export_dir
    return found


def build_export_index(export_dir: PathLike) -> Dict[str, Dict[str, object]]:
    """Thin wrapper used by discovery to avoid importing case-builder names elsewhere."""
    from ..easyicu_case_builder import index_export_package

    return index_export_package(export_dir)


def _normalise_easyicu_frame(
    frame: pd.DataFrame,
    *,
    database: str,
    concepts: Sequence[str],
) -> pd.DataFrame:
    """Normalise an EasyICU API frame into export-package column names."""
    try:
        from easyicu.api import get_id_col_for_database

        id_col = get_id_col_for_database(database)
    except Exception:
        id_col = {
            "miiv": "stay_id",
            "mimic": "icustay_id",
            "eicu": "patientunitstayid",
            "hirid": "patientid",
            "aumc": "admissionid",
            "sic": "CaseID",
        }.get(database, "stay_id")

    work = frame.copy()
    if id_col in work.columns and id_col != "stay_id":
        work = work.rename(columns={id_col: "stay_id"})
    elif "stay_id" not in work.columns:
        for candidate in ["icustay_id", "patientunitstayid", "admissionid", "patientid", "CaseID"]:
            if candidate in work.columns:
                work = work.rename(columns={candidate: "stay_id"})
                break

    if "stay_id" not in work.columns:
        work["stay_id"] = pd.Series(dtype="object")

    for concept in concepts:
        if concept not in work.columns:
            work[concept] = pd.NA

    preferred = ["stay_id"]
    if "charttime" in work.columns:
        preferred.append("charttime")
    elif "time" in work.columns:
        work = work.rename(columns={"time": "charttime"})
        preferred.append("charttime")
    return work[[c for c in [*preferred, *concepts] if c in work.columns]].copy()


def _load_easyicu_group(
    *,
    database: str,
    data_path: Path,
    concepts: Sequence[str],
    max_patients: Optional[int],
    concept_workers: int,
    parallel_workers: int,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load a group through EasyICU, falling back to concept-by-concept loading."""
    from easyicu import load_concepts

    unavailable: List[str] = []
    empty: List[str] = []
    try:
        frame = load_concepts(
            list(concepts),
            database=database,
            data_path=data_path,
            max_patients=max_patients,
            merge=True,
            verbose=False,
            concept_workers=concept_workers,
            parallel_workers=parallel_workers,
        )
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame()
    except Exception:
        pieces: List[pd.DataFrame] = []
        for concept in concepts:
            try:
                part = load_concepts(
                    [concept],
                    database=database,
                    data_path=data_path,
                    max_patients=max_patients,
                    merge=True,
                    verbose=False,
                    concept_workers=1,
                    parallel_workers=1,
                )
                if isinstance(part, pd.DataFrame) and not part.empty:
                    pieces.append(part)
                else:
                    empty.append(concept)
            except Exception:
                unavailable.append(concept)
        frame = pieces[0] if pieces else pd.DataFrame()
        for part in pieces[1:]:
            id_candidates = [
                c for c in ["stay_id", "icustay_id", "patientunitstayid", "admissionid", "patientid", "CaseID", "charttime"]
                if c in frame.columns and c in part.columns
            ]
            frame = frame.merge(part, on=id_candidates, how="outer") if id_candidates else pd.concat([frame, part], ignore_index=True)

    normalised = _normalise_easyicu_frame(frame, database=database, concepts=concepts)
    for concept in concepts:
        if concept in unavailable:
            continue
        if concept not in frame.columns or normalised[concept].dropna().empty:
            empty.append(concept)
    return normalised, sorted(set(unavailable)), sorted(set(empty))


def export_lactate_map_vaso_concepts_from_easyicu(
    *,
    database: str,
    data_path: PathLike,
    output_dir: PathLike,
    max_patients: Optional[int] = None,
    groups: Optional[Mapping[str, Sequence[str]]] = None,
    concept_workers: int = 1,
    parallel_workers: int = 1,
) -> Path:
    """Create a minimal EasyICU concept-export package for the shock case.

    The output directory follows the same shape as the web export:
    grouped parquet files plus ``easyicu_export_manifest.json``. It can
    then be consumed by ``build_lactate_map_vaso_cohort_from_export``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(data_path)
    groups = groups or LACTATE_MAP_VASO_EXPORT_GROUPS

    exported_files: List[str] = []
    selected_concepts: List[str] = []
    unavailable_concepts: List[str] = []
    empty_data_concepts: List[str] = []

    for group_name, concepts in groups.items():
        frame, unavailable, empty = _load_easyicu_group(
            database=database,
            data_path=path,
            concepts=concepts,
            max_patients=max_patients,
            concept_workers=concept_workers,
            parallel_workers=parallel_workers,
        )
        file_name = f"{_safe_slug(group_name)}_{'_'.join(_safe_slug(c) for c in concepts[:5])}.parquet"
        frame.to_parquet(out / file_name, index=False)
        exported_files.append(file_name)
        selected_concepts.extend(concepts)
        unavailable_concepts.extend(unavailable)
        empty_data_concepts.extend(empty)

    manifest = {
        "easyicu_version": "1.0.0",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "database": database,
        "entry_mode": "real",
        "export_dir": str(out),
        "export_format": "parquet",
        "patient_count": max_patients,
        "concept_count": len(set(selected_concepts)),
        "selected_concepts": sorted(set(selected_concepts)),
        "selected_groups": list(groups),
        "exported_files": exported_files,
        "unavailable_concepts": sorted(set(unavailable_concepts)),
        "empty_data_concepts": sorted(set(empty_data_concepts)),
        "failed_concepts": [],
        "note": "Generated by easyicu.research_agent.replication.export_lactate_map_vaso_concepts_from_easyicu",
    }
    (out / "easyicu_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def _mortality_for(frame: pd.DataFrame) -> Tuple[int, int, float, float, float]:
    n = int(len(frame))
    deaths = int(pd.to_numeric(frame["death"], errors="coerce").fillna(0).sum()) if n else 0
    rate, lo, hi = _rate_ci(deaths, n)
    return n, deaths, rate, lo, hi


def shock_strata(cohort: pd.DataFrame, *, database: str) -> pd.DataFrame:
    """Return lactate-MAP-vasopressor strata for one cohort."""
    required = {"death", "lactate_measured_24h", "lactate_max_24h", "map_min_24h", "vaso_any_24h"}
    missing = sorted(required.difference(cohort.columns))
    if missing:
        raise KeyError(f"cohort is missing required shock-strata columns: {missing}")

    work = cohort.copy()
    work["death"] = pd.to_numeric(work["death"], errors="coerce").fillna(0)
    work["lactate_measured_24h"] = pd.to_numeric(
        work["lactate_measured_24h"], errors="coerce"
    ).fillna(0)
    work["lactate_max_24h"] = pd.to_numeric(work["lactate_max_24h"], errors="coerce")
    work["map_min_24h"] = pd.to_numeric(work["map_min_24h"], errors="coerce")
    work["vaso_any_24h"] = pd.to_numeric(work["vaso_any_24h"], errors="coerce").fillna(0)

    measured = work[work["lactate_measured_24h"] == 1].copy()
    strata = [
        ("MAP<65", measured["map_min_24h"] < 65),
        (
            "MAP>=65 & Lactate>2 & NoVaso",
            (measured["map_min_24h"] >= 65)
            & (measured["lactate_max_24h"] > 2)
            & (measured["vaso_any_24h"] == 0),
        ),
        (
            "MAP>=65 & Lactate>2 & Vaso",
            (measured["map_min_24h"] >= 65)
            & (measured["lactate_max_24h"] > 2)
            & (measured["vaso_any_24h"] > 0),
        ),
        (
            "MAP>=65 & Lactate<=2",
            (measured["map_min_24h"] >= 65) & (measured["lactate_max_24h"] <= 2),
        ),
    ]
    rows: List[Dict[str, object]] = []
    for stratum, mask in strata:
        n, deaths, rate, lo, hi = _mortality_for(measured[mask])
        rows.append({
            "database": database,
            "stratum": stratum,
            "n": n,
            "deaths": deaths,
            "mortality_rate": rate,
            "ci_lower": lo,
            "ci_upper": hi,
        })
    return pd.DataFrame(rows)


def summarize_lactate_map_vaso_cohort(cohort: pd.DataFrame, *, database: str) -> Dict[str, object]:
    """Create one manuscript-facing summary row for a shock cohort."""
    n, deaths, mortality, mortality_lo, mortality_hi = _mortality_for(cohort)
    measured = cohort[pd.to_numeric(cohort["lactate_measured_24h"], errors="coerce").fillna(0) == 1]
    unmeasured = cohort[pd.to_numeric(cohort["lactate_measured_24h"], errors="coerce").fillna(0) == 0]
    strata = shock_strata(cohort, database=database).set_index("stratum")

    def _stratum_rate(name: str) -> float:
        value = strata.loc[name, "mortality_rate"]
        return float(value) if not pd.isna(value) else math.nan

    return {
        "database": database,
        "status": "ok",
        "export_dir": "",
        "message": "cohort built from EasyICU concept export",
        "n_stays": n,
        "n_deaths": deaths,
        "mortality_rate": mortality,
        "mortality_ci_lower": mortality_lo,
        "mortality_ci_upper": mortality_hi,
        "lactate_measured_n": int(len(measured)),
        "lactate_measured_pct": float(len(measured) / n) if n else math.nan,
        "lactate_measured_mortality": _mortality_for(measured)[2],
        "lactate_unmeasured_mortality": _mortality_for(unmeasured)[2],
        "occult_no_vaso_mortality": _stratum_rate("MAP>=65 & Lactate>2 & NoVaso"),
        "occult_vaso_mortality": _stratum_rate("MAP>=65 & Lactate>2 & Vaso"),
        "map_adequate_low_lactate_mortality": _stratum_rate("MAP>=65 & Lactate<=2"),
        "map_low_mortality": _stratum_rate("MAP<65"),
    }


def _normalise_targets(
    targets: Mapping[str, Optional[PathLike]] | Sequence[ReplicationTarget],
) -> List[ReplicationTarget]:
    if isinstance(targets, Mapping):
        return [
            ReplicationTarget(database=str(database), export_dir=Path(path) if path else None)
            for database, path in targets.items()
        ]
    return [
        ReplicationTarget(
            database=t.database,
            export_dir=Path(t.export_dir) if t.export_dir is not None else None,
            label=t.label,
        )
        for t in targets
    ]


def _write_appendix(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    strata: pd.DataFrame,
    manifest: Dict[str, object],
) -> Path:
    ok = summary[summary["status"] == "ok"]
    pending = summary[summary["status"] != "ok"]
    lines = [
        "# Lactate-MAP-Vasopressor Replication Appendix",
        "",
        "This appendix is generated deterministically from EasyICU concept-export packages. "
        "It does not call an LLM and does not use ad hoc SQL.",
        "",
        "## Replication Targets",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.database}: {row.status}"
            + (f" ({row.message})" if getattr(row, "message", "") else "")
        )
    if len(ok):
        lines.extend(["", "## Completed Cohorts", ""])
        for row in ok.itertuples(index=False):
            lines.append(
                f"- {row.database}: n={int(row.n_stays)}, deaths={int(row.n_deaths)}, "
                f"mortality={100 * row.mortality_rate:.1f}%, "
                f"lactate measured={100 * row.lactate_measured_pct:.1f}%."
            )
    if len(pending):
        lines.extend(["", "## Pending Cohorts", ""])
        lines.append(
            "Rows marked as pending/error keep the protocol honest: they are part of the "
            "planned cross-database design, but no compatible EasyICU export was available "
            "or buildable in this local run."
        )
    lines.extend([
        "",
        "## Output Files",
        "",
        "- `tables/replication_summary.csv`",
        "- `tables/shock_strata_by_database.csv`",
        "- `replication_manifest.json`",
        "",
        "## Traceability",
        "",
        f"Run manifest records {len(manifest.get('targets', []))} target(s) and "
        f"{len(manifest.get('cohort_outputs', {}))} completed cohort package(s).",
    ])
    path = output_dir / "replication_appendix.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_lactate_map_vaso_replication(
    targets: Mapping[str, Optional[PathLike]] | Sequence[ReplicationTarget],
    output_dir: PathLike,
    *,
    window: Tuple[float, float] = (0.0, 24.0),
    include_unmeasured_lactate: bool = True,
) -> Dict[str, Path]:
    """Build a deterministic cross-export replication package.

    Parameters
    ----------
    targets:
        Either ``{"miiv": "/path/to/export", "eicu": None}`` or a
        sequence of :class:`ReplicationTarget` objects. ``None`` marks
        planned but unavailable exports.
    output_dir:
        Directory where cohorts, tables and manifest should be written.
    window:
        Time window in hours relative to ICU admission.
    include_unmeasured_lactate:
        Preserve unmeasured lactate rows so missingness remains visible.
    """
    out = Path(output_dir)
    cohort_dir = out / "cohorts"
    table_dir = out / "tables"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    strata_frames: List[pd.DataFrame] = []
    manifest: Dict[str, object] = {
        "builder": "easyicu.research_agent.replication.run_lactate_map_vaso_replication",
        "case": "lactate_map_vaso_shock_mortality",
        "window_hours": {"start": window[0], "end": window[1], "anchor": "icu_admission"},
        "targets": [],
        "cohort_outputs": {},
    }

    for target in _normalise_targets(targets):
        db = target.database
        export_dir = target.export_dir
        manifest["targets"].append({
            "database": db,
            "export_dir": str(export_dir) if export_dir is not None else None,
        })

        if export_dir is None:
            summary_rows.append(_status_row(
                database=db,
                status="pending",
                export_dir=None,
                message="no EasyICU export directory supplied",
            ))
            continue
        if not export_dir.exists():
            summary_rows.append(_status_row(
                database=db,
                status="pending",
                export_dir=export_dir,
                message="EasyICU export directory does not exist",
            ))
            continue

        try:
            package: EasyICUCasePackage = build_lactate_map_vaso_cohort_from_export(
                export_dir,
                window=window,
                include_unmeasured_lactate=include_unmeasured_lactate,
            )
            db_dir = cohort_dir / db
            written = package.write(db_dir, stem=f"{db}_lactate_map_vaso_24h")
            source_manifest = db_dir / "source_manifest.json"
            shutil.copy2(written["manifest"], source_manifest)

            row = summarize_lactate_map_vaso_cohort(package.cohort, database=db)
            row["export_dir"] = str(export_dir)
            summary_rows.append(row)
            strata_frames.append(shock_strata(package.cohort, database=db))
            manifest["cohort_outputs"][db] = {k: str(v) for k, v in written.items()}
            manifest["cohort_outputs"][db]["source_manifest"] = str(source_manifest)
        except Exception as exc:  # pragma: no cover - exercised by integration use
            summary_rows.append(_status_row(
                database=db,
                status="error",
                export_dir=export_dir,
                message=f"{type(exc).__name__}: {exc}",
            ))

    summary = pd.DataFrame(summary_rows)
    strata = pd.concat(strata_frames, ignore_index=True) if strata_frames else pd.DataFrame(
        columns=["database", "stratum", "n", "deaths", "mortality_rate", "ci_lower", "ci_upper"]
    )
    summary_path = table_dir / "replication_summary.csv"
    strata_path = table_dir / "shock_strata_by_database.csv"
    manifest_path = out / "replication_manifest.json"
    summary.to_csv(summary_path, index=False)
    strata.to_csv(strata_path, index=False)
    manifest["summary_table"] = str(summary_path)
    manifest["shock_strata_table"] = str(strata_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    appendix_path = _write_appendix(output_dir=out, summary=summary, strata=strata, manifest=manifest)

    return {
        "summary": summary_path,
        "shock_strata": strata_path,
        "manifest": manifest_path,
        "appendix": appendix_path,
    }


__all__ = [
    "ReplicationTarget",
    "LACTATE_MAP_VASO_EXPORT_GROUPS",
    "LACTATE_MAP_VASO_MINIMAL_EXPORT_GROUPS",
    "discover_easyicu_exports",
    "export_lactate_map_vaso_concepts_from_easyicu",
    "shock_strata",
    "summarize_lactate_map_vaso_cohort",
    "run_lactate_map_vaso_replication",
]
