"""EasyICU-native cohort builders for research-agent case studies.

The functions in this module consume an EasyICU concept export package:
a directory of parquet files produced by the extraction UI / worker
where each file contains one or more EasyICU concepts. They convert
that export into one-row-per-stay analysis cohorts plus a machine
readable source manifest.

This is intentionally separate from the agent loop. The builder is
deterministic data plumbing; ``ResearchAgentPipeline`` then handles
planning, code execution, validation, evidence binding and manuscript
scaffolding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from easyicu.io.parquet_reader import parquet_metadata, read_parquet


ID_COL = "stay_id"
TIME_COL = "charttime"


@dataclass(frozen=True)
class EasyICUCasePackage:
    """A deterministic cohort and its EasyICU concept provenance."""

    cohort: pd.DataFrame
    source_manifest: Dict[str, object]
    concept_sources: Dict[str, str]

    def write(self, output_dir: Union[str, Path], *, stem: str) -> Dict[str, Path]:
        """Write cohort CSV/parquet and manifest JSON into ``output_dir``."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / f"{stem}.csv"
        parquet_path = out / f"{stem}.parquet"
        manifest_path = out / f"{stem}_source_manifest.json"
        self.cohort.to_csv(csv_path, index=False)
        self.cohort.to_parquet(parquet_path, index=False)
        manifest_path.write_text(
            json.dumps(self.source_manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return {"csv": csv_path, "parquet": parquet_path, "manifest": manifest_path}


def index_export_package(export_dir: Union[str, Path]) -> Dict[str, Dict[str, object]]:
    """Return concept-name -> parquet metadata for an EasyICU export dir."""
    root = Path(export_dir)
    if not root.exists():
        raise FileNotFoundError(f"EasyICU export directory not found: {root}")

    index: Dict[str, Dict[str, object]] = {}
    for path in sorted(root.glob("*.parquet")):
        meta = parquet_metadata(path)
        for col in meta.get("columns", []):
            if col in {ID_COL, TIME_COL, "time"}:
                continue
            index[str(col)] = {
                "file": str(path),
                "file_name": path.name,
                "rows": int(meta.get("nrow", 0)),
                "columns": list(meta.get("columns", [])),
            }
    return index


def resolve_exported_concept(
    index: Mapping[str, Dict[str, object]], concept: str
) -> Optional[str]:
    """Resolve ``concept`` to an actual export column name, or ``None``.

    Exact match wins. Otherwise a **conservative** alias is allowed: if the
    export stored the concept under a single score-variant column name (e.g.
    ``sep3`` exported as ``sep3_sofa2``), and there is *exactly one* column of
    the form ``<concept>_<suffix>``, that column is used. Ambiguous cases
    (e.g. ``los`` -> ``los_icu``/``los_hosp``) deliberately do NOT resolve, so
    the caller must name the concept explicitly rather than risk a silent
    wrong match.
    """
    if concept in index:
        return concept
    candidates = sorted(c for c in index if c.startswith(concept + "_"))
    return candidates[0] if len(candidates) == 1 else None


def read_exported_concept(
    export_dir: Union[str, Path],
    concept: str,
    *,
    extra_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Read one concept from an EasyICU export package.

    ``extra_columns`` lets callers request sibling concepts from the same
    parquet file, for example ``norepi_equiv`` next to ``vaso_ind``.

    A conservative alias is applied (see :func:`resolve_exported_concept`): if
    ``concept`` is not an exact column but maps unambiguously to a single
    score-variant column (e.g. ``sep3`` -> ``sep3_sofa2``), that column is read
    and renamed to ``concept`` so downstream code sees the requested name.
    """
    index = index_export_package(export_dir)
    resolved = resolve_exported_concept(index, concept)
    if resolved is None:
        raise KeyError(
            f"Concept {concept!r} is not present in {Path(export_dir)}. "
            f"Available concepts include: {sorted(index)[:20]}"
        )
    path = Path(index[resolved]["file"])
    available = set(index[resolved]["columns"])
    columns = [
        c for c in [ID_COL, TIME_COL, resolved, *(extra_columns or [])] if c in available
    ]
    df = read_parquet(path, columns=columns)
    if resolved != concept:
        df = df.rename(columns={resolved: concept})
    return df


def _window(df: pd.DataFrame, start_hour: float, end_hour: float) -> pd.DataFrame:
    if TIME_COL not in df.columns:
        return df.copy()
    out = df.copy()
    out[TIME_COL] = pd.to_numeric(out[TIME_COL], errors="coerce")
    return out[(out[TIME_COL] >= start_hour) & (out[TIME_COL] <= end_hour)].copy()


def _first_nonnull(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else pd.NA


def _aggregate_lactate(df: pd.DataFrame, start_hour: float, end_hour: float) -> pd.DataFrame:
    work = _window(df, start_hour, end_hour)
    if work.empty:
        return pd.DataFrame(columns=[ID_COL])
    work["lact"] = pd.to_numeric(work["lact"], errors="coerce")
    work = work.dropna(subset=[ID_COL])
    grouped = work.groupby(ID_COL, dropna=True)
    out = grouped.agg(
        lactate_max_24h=("lact", "max"),
        lactate_median_24h=("lact", "median"),
        lactate_first_24h=("lact", _first_nonnull),
        lactate_n_24h=("lact", "count"),
    ).reset_index()
    out["lactate_measured_24h"] = (out["lactate_n_24h"] > 0).astype(int)
    out["hyperlactatemia_24h"] = (out["lactate_max_24h"] > 2.0).astype("Int64")
    out["lactate_gt4_24h"] = (out["lactate_max_24h"] > 4.0).astype("Int64")
    return out


def _aggregate_map(df: pd.DataFrame, start_hour: float, end_hour: float) -> pd.DataFrame:
    work = _window(df, start_hour, end_hour)
    if work.empty:
        return pd.DataFrame(columns=[ID_COL])
    work["map"] = pd.to_numeric(work["map"], errors="coerce")
    work = work.dropna(subset=[ID_COL])
    grouped = work.groupby(ID_COL, dropna=True)
    out = grouped.agg(
        map_min_24h=("map", "min"),
        map_median_24h=("map", "median"),
        map_mean_24h=("map", "mean"),
        map_n_24h=("map", "count"),
    ).reset_index()
    out["map_low_any_24h"] = (out["map_min_24h"] <= 65.0).astype("Int64")
    out["map_ge65_all_24h"] = (out["map_min_24h"] >= 65.0).astype("Int64")
    return out


def _aggregate_vaso(df: pd.DataFrame, start_hour: float, end_hour: float) -> pd.DataFrame:
    work = _window(df, start_hour, end_hour)
    if work.empty:
        return pd.DataFrame(columns=[ID_COL])
    for col in ["vaso_ind", "norepi_equiv"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[ID_COL])
    grouped = work.groupby(ID_COL, dropna=True)
    specs = {
        "vaso_any_24h": ("vaso_ind", "max"),
        "vaso_hours_24h": ("vaso_ind", "sum"),
    }
    if "norepi_equiv" in work.columns:
        specs.update({
            "norepi_equiv_max_24h": ("norepi_equiv", "max"),
            "norepi_equiv_median_24h": ("norepi_equiv", "median"),
        })
    out = grouped.agg(**specs).reset_index()
    out["vaso_any_24h"] = out["vaso_any_24h"].fillna(0).astype(int)
    out["vaso_hours_24h"] = out["vaso_hours_24h"].fillna(0)
    return out


def _aggregate_circ(df: pd.DataFrame, start_hour: float, end_hour: float) -> pd.DataFrame:
    work = _window(df, start_hour, end_hour)
    if work.empty:
        return pd.DataFrame(columns=[ID_COL])
    for col in ["circ_event", "circ_failure"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    grouped = work.groupby(ID_COL, dropna=True)
    out = grouped.agg(
        circ_event_max_24h=("circ_event", "max"),
        circ_failure_any_24h=("circ_failure", "max"),
        circ_observed_hours_24h=("circ_event", "count"),
    ).reset_index()
    out["circ_failure_any_24h"] = out["circ_failure_any_24h"].fillna(0).astype(int)
    return out


def _aggregate_sep3(df: pd.DataFrame, start_hour: float, end_hour: float) -> pd.DataFrame:
    work = _window(df, start_hour, end_hour)
    if work.empty or "sep3_sofa2" not in work.columns:
        return pd.DataFrame(columns=[ID_COL])
    work["sep3_sofa2"] = work["sep3_sofa2"].fillna(False).astype(bool)
    out = (
        work.groupby(ID_COL, dropna=True)
        .agg(sep3_sofa2_any_24h=("sep3_sofa2", "max"))
        .reset_index()
    )
    out["sep3_sofa2_any_24h"] = out["sep3_sofa2_any_24h"].astype(int)
    return out


def _merge_left(base: pd.DataFrame, frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    for frame in frames:
        if frame is None or frame.empty or ID_COL not in frame.columns:
            continue
        out = out.merge(frame, on=ID_COL, how="left")
    return out


def build_lactate_map_vaso_cohort_from_export(
    export_dir: Union[str, Path],
    *,
    window: Tuple[float, float] = (0.0, 24.0),
    include_unmeasured_lactate: bool = True,
) -> EasyICUCasePackage:
    """Build the shock physiology case cohort from an EasyICU export.

    The resulting cohort asks: among first ICU stays, does early lactate
    identify mortality risk beyond MAP and vasopressor exposure, including
    patients with apparently adequate MAP?
    """
    start_hour, end_hour = window
    root = Path(export_dir)
    index = index_export_package(root)
    required = ["age", "sex", "death", "los_icu", "lact", "map", "vaso_ind"]
    missing = [c for c in required if c not in index]
    if missing:
        raise KeyError(f"EasyICU export is missing required concepts: {missing}")

    demo = read_exported_concept(root, "age", extra_columns=["sex", "adm", "bmi", "weight"])
    outcome = read_exported_concept(root, "death", extra_columns=["los_icu", "los_hosp"])
    lact = read_exported_concept(root, "lact")
    vitals = read_exported_concept(root, "map")
    vaso = read_exported_concept(root, "vaso_ind", extra_columns=["norepi_equiv"])

    base_cols = [c for c in [ID_COL, "age", "sex", "adm", "bmi", "weight"] if c in demo.columns]
    base = demo[base_cols].drop_duplicates(subset=[ID_COL]).copy()
    out_cols = [c for c in [ID_COL, "death", "los_icu", "los_hosp"] if c in outcome.columns]
    out = outcome[out_cols].drop_duplicates(subset=[ID_COL]).copy()
    if "death" in out.columns:
        out["death"] = out["death"].fillna(False).astype(bool).astype(int)

    agg_frames: List[pd.DataFrame] = [
        out,
        _aggregate_lactate(lact, start_hour, end_hour),
        _aggregate_map(vitals, start_hour, end_hour),
        _aggregate_vaso(vaso, start_hour, end_hour),
    ]

    if "circ_failure" in index:
        circ = read_exported_concept(root, "circ_failure", extra_columns=["circ_event"])
        agg_frames.append(_aggregate_circ(circ, start_hour, end_hour))
    if "sep3_sofa2" in index:
        sep3 = read_exported_concept(root, "sep3_sofa2")
        agg_frames.append(_aggregate_sep3(sep3, start_hour, end_hour))

    cohort = _merge_left(base, agg_frames)

    default_zero = {
        "vaso_any_24h": 0,
        "vaso_hours_24h": 0,
        "circ_failure_any_24h": 0,
        "sep3_sofa2_any_24h": 0,
    }
    for col, default in default_zero.items():
        if col in cohort.columns:
            cohort[col] = cohort[col].fillna(default)
        else:
            cohort[col] = default
    if "lactate_measured_24h" not in cohort.columns:
        cohort["lactate_measured_24h"] = 0
    else:
        cohort["lactate_measured_24h"] = cohort["lactate_measured_24h"].fillna(0).astype(int)

    if not include_unmeasured_lactate:
        cohort = cohort[cohort["lactate_measured_24h"] == 1].copy()

    if "death" not in cohort.columns:
        raise KeyError("Outcome concept 'death' did not produce a death column.")

    concept_sources = {
        "age": index["age"]["file_name"],
        "sex": index["sex"]["file_name"],
        "death": index["death"]["file_name"],
        "los_icu": index["los_icu"]["file_name"],
        "lactate_max_24h": index["lact"]["file_name"],
        "map_min_24h": index["map"]["file_name"],
        "vaso_any_24h": index["vaso_ind"]["file_name"],
    }
    if "circ_failure" in index:
        concept_sources["circ_failure_any_24h"] = index["circ_failure"]["file_name"]
    if "sep3_sofa2" in index:
        concept_sources["sep3_sofa2_any_24h"] = index["sep3_sofa2"]["file_name"]

    manifest = {
        "builder": "easyicu.research_agent.easyicu_case_builder.build_lactate_map_vaso_cohort_from_export",
        "export_dir": str(root),
        "window_hours": {"start": start_hour, "end": end_hour, "anchor": "icu_admission"},
        "unit_of_analysis": "one row per ICU stay",
        "n_stays": int(len(cohort)),
        "n_deaths": int(cohort["death"].sum()),
        "mortality": float(cohort["death"].mean()) if len(cohort) else None,
        "concept_sources": concept_sources,
        "available_export_concepts": sorted(index),
        "notes": [
            "death is stored by EasyICU as a logical event and converted to 0/1 hospital mortality.",
            "lactate is right-skewed and aggregated as first/median/max, not mean.",
            "vasopressor exposure is represented as any exposure and norepinephrine-equivalent dose within the first 24 h.",
            "MAP is summarized by minimum/median/mean; the clinical discordance analysis uses minimum MAP >= 65 mmHg as apparent hemodynamic adequacy.",
        ],
    }
    return EasyICUCasePackage(
        cohort=cohort.reset_index(drop=True),
        source_manifest=manifest,
        concept_sources=concept_sources,
    )


__all__ = [
    "EasyICUCasePackage",
    "index_export_package",
    "read_exported_concept",
    "build_lactate_map_vaso_cohort_from_export",
]
