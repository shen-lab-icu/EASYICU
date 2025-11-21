#!/usr/bin/env python3
"""Reusable feature comparison utilities between ricu CSV outputs and pyricu.

This module hosts the heavy lifting that used to live in the legacy
``compare_ricu_pyricu.py`` script so that other tools (and downstream users)
can leverage the same logic programmatically.  The thin CLI wrapper now just
parses arguments and delegates to :func:`main`.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import load_concepts
from .project_config import (
    DEFAULT_PATIENTS_AUMC,
    DEFAULT_PATIENTS_EICU,
    DEFAULT_PATIENTS_HIRID,
    DEFAULT_PATIENTS_MIIV,
    PRODUCTION_DATA_AUMC,
    PRODUCTION_DATA_EICU,
    PRODUCTION_DATA_HIRID,
    PRODUCTION_DATA_PATH,
)
from .runtime_defaults import resolve_loader_defaults


@dataclass
class FeatureModule:
    """Metadata describing a ricu CSV <-> pyricu concept family."""

    name: str
    ricu_file: str
    concepts: List[str]
    id_column: str = "stay_id"
    time_column: Optional[str] = "charttime"
    description: str = ""


MODULES: List[FeatureModule] = [
    FeatureModule(
        name="outcome",
        ricu_file="{db}_outcome.csv",
        concepts=[
            "sofa",
            "sofa_resp",
            "sofa_coag",
            "sofa_liver",
            "sofa_cardio",
            "sofa_cns",
            "sofa_renal",
            "qsofa",
            "sirs",
            "death",
            "los_icu",
        ],
        time_column="index_var",
        description="SOFA outcome timeline",
    ),
    FeatureModule(
        name="resp",
        ricu_file="{db}_resp.csv",
        concepts=[
            "pafi",
            "safi",
            "resp",
            "supp_o2",
            "vent_ind",
            "o2sat",
            "sao2",
            "mech_vent",
            "ett_gcs",
            "fio2",
        ],
        description="Respiratory chain (PaFi/SaFi/Vent indicators)",
    ),
    FeatureModule(
        name="blood",
        ricu_file="{db}_blood.csv",
        concepts=["be", "cai", "fio2", "hbco", "lact", "methb", "pco2", "ph", "po2", "tco2"],
        description="Arterial blood gas",
    ),
    FeatureModule(
        name="lab",
        ricu_file="{db}_lab.csv",
        concepts=[
            "alb",
            "alp",
            "alt",
            "ast",
            "bicar",
            "bili",
            "bili_dir",
            "bun",
            "ca",
            "ck",
            "ckmb",
            "cl",
            "crea",
            "crp",
            "glu",
            "k",
            "mg",
            "na",
            "phos",
            "tnt",
        ],
        description="General chemistry",
    ),
    FeatureModule(
        name="hematology",
        ricu_file="{db}_hematology.csv",
        concepts=["bnd", "esr", "fgn", "hgb", "inr_pt", "lymph", "mch", "mchc", "mcv", "neut", "plt", "ptt", "wbc"],
        description="Heme/Coag profile",
    ),
    FeatureModule(
        name="med",
        ricu_file="{db}_med.csv",
        concepts=[
            "abx",
            "adh_rate",
            "cort",
            "dex",
            "dobu_dur",
            "dobu_rate",
            "dobu60",
            "epi_dur",
            "epi_rate",
            "ins",
            "norepi_dur",
            "norepi_equiv",
            "norepi_rate",
            "vaso_ind",
        ],
        time_column="starttime",
        description="Vasopressors / medications",
    ),
    FeatureModule(
        name="vital",
        ricu_file="{db}_vital.csv",
        concepts=["dbp", "etco2", "hr", "map", "sbp", "temp"],
        description="Bedside vitals",
    ),
    FeatureModule(
        name="output",
        ricu_file="{db}_output.csv",
        concepts=["urine", "urine24"],
        description="Urine output",
    ),
    FeatureModule(
        name="neu",
        ricu_file="{db}_neu.csv",
        concepts=["avpu", "egcs", "gcs", "mgcs", "rass", "vgcs"],
        description="Neurologic assessments",
    ),
    FeatureModule(
        name="demo",
        ricu_file="{db}_demo.csv",
        concepts=["age", "bmi", "height", "sex", "weight"],
        time_column=None,
        description="Demographics",
    ),
]


SOFA_COMPONENT_DEPENDENCIES: Dict[str, List[str]] = {
    "sofa_resp": ["pafi", "safi", "vent_ind", "supp_o2", "fio2", "resp", "o2sat", "sao2"],
    "sofa_coag": ["plt", "inr_pt", "ptt"],
    "sofa_liver": ["bili", "bili_dir", "alp", "ast", "alt"],
    "sofa_cardio": ["map", "vaso_ind", "norepi_rate", "dobu_rate", "epi_rate", "adh_rate"],
    "sofa_cns": ["gcs", "mgcs", "rass", "avpu"],
    "sofa_renal": ["crea", "bun", "urine", "urine24"],
}

ID_CANDIDATES = ["stay_id", "subject_id", "patientunitstayid", "admissionid", "patientid"]
TIME_CANDIDATES = [
    "charttime",
    "index_var",
    "starttime",
    "time",
    "measuredat",
    "registeredat",
    "nursingchartoffset",
    "labresultoffset",
    "observationoffset",
    "chartoffset",
    "offset",
]
FIO2_ITEMIDS = [223835, 50816]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RICU_ROOT = REPO_ROOT / "ricu_data"
SUPPORTED_DATABASES = ("miiv", "eicu", "aumc", "hirid")

ALIGNMENT_ENV_KEYS = {
    "chunk": ("RICU_ALIGN_CHUNK", "PYRICU_CHUNK_SIZE"),
    "parallel": ("RICU_ALIGN_PARALLEL", "PYRICU_PARALLEL_WORKERS"),
    "concept": ("RICU_ALIGN_CONCEPT", "PYRICU_CONCEPT_WORKERS"),
    "backend": ("RICU_ALIGN_BACKEND", "PYRICU_PARALLEL_BACKEND"),
}

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATHS: Dict[str, Path] = {
    "miiv": PRODUCTION_DATA_PATH,
    "eicu": PRODUCTION_DATA_EICU,
    "aumc": PRODUCTION_DATA_AUMC,
    "hirid": PRODUCTION_DATA_HIRID,
}

DEFAULT_PATIENTS: Dict[str, List[int]] = {
    "miiv": DEFAULT_PATIENTS_MIIV,
    "eicu": DEFAULT_PATIENTS_EICU,
    "aumc": DEFAULT_PATIENTS_AUMC,
    "hirid": DEFAULT_PATIENTS_HIRID or [],
}

PATIENT_ID_SOURCES: Dict[str, tuple[str, str]] = {
    "miiv": ("icustays.parquet", "stay_id"),
    "eicu": ("patient.parquet", "patientunitstayid"),
    "aumc": ("admissions.parquet", "admissionid"),
    "hirid": ("general.parquet", "patientid"),
}


@dataclass
class SeriesStats:
    rows: int
    non_null: int
    mean: Optional[float]
    maximum: Optional[float]
    first_signal: Optional[float]

    @property
    def coverage(self) -> float:
        return (self.non_null / self.rows) if self.rows else 0.0


class FeatureComparison:
    """Holds normalised ricu/pyricu time series for a module."""

    def __init__(
        self,
        module: FeatureModule,
        ricu_series: Dict[str, pd.DataFrame],
        pyricu_series: Dict[str, pd.DataFrame],
    ):
        self.module = module
        self.ricu_series = ricu_series
        self.pyricu_series = pyricu_series

    def available_concepts(self, source: str) -> Iterable[str]:
        mapping = self.pyricu_series if source == "pyricu" else self.ricu_series
        return mapping.keys()

    def compute_metrics(self) -> Dict[str, Dict[str, object]]:
        metrics: Dict[str, Dict[str, object]] = {}
        names = sorted(set(self.ricu_series.keys()) | set(self.pyricu_series.keys()))
        for name in names:
            ricu_stats = self._stats(self.ricu_series.get(name))
            py_stats = self._stats(self.pyricu_series.get(name))
            entry = {
                "ricu_rows": ricu_stats.rows,
                "pyricu_rows": py_stats.rows,
                "ricu_coverage": ricu_stats.coverage,
                "pyricu_coverage": py_stats.coverage,
                "coverage_gap": ricu_stats.coverage - py_stats.coverage,
                "ricu_mean": ricu_stats.mean,
                "pyricu_mean": py_stats.mean,
                "mean_diff": (py_stats.mean - ricu_stats.mean)
                if (ricu_stats.mean is not None and py_stats.mean is not None)
                else None,
                "ricu_max": ricu_stats.maximum,
                "pyricu_max": py_stats.maximum,
                "ricu_first": ricu_stats.first_signal,
                "pyricu_first": py_stats.first_signal,
            }
            if py_stats.rows == 0 and ricu_stats.rows > 0:
                entry["status"] = "missing:pyricu"
            elif py_stats.rows > 0 and ricu_stats.rows == 0:
                entry["status"] = "missing:ricu"
            elif ricu_stats.non_null > 0 and py_stats.non_null == 0:
                entry["status"] = "missing:pyricu"
            else:
                entry["status"] = "ok"
            metrics[name] = entry
        return metrics

    @staticmethod
    def _stats(df: Optional[pd.DataFrame]) -> SeriesStats:
        if df is None or df.empty:
            return SeriesStats(rows=0, non_null=0, mean=None, maximum=None, first_signal=None)

        values = df["value"]
        if isinstance(values, pd.DataFrame) or (hasattr(values, "ndim") and getattr(values, "ndim") > 1):
            return SeriesStats(rows=len(df), non_null=0, mean=None, maximum=None, first_signal=None)
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        try:
            numeric = pd.to_numeric(values, errors="coerce")
        except Exception:
            numeric = pd.Series(np.nan, index=values.index)
        numeric_valid = numeric.notna().sum() > 0
        mean = float(numeric.mean()) if numeric_valid else None
        maximum = float(numeric.max()) if numeric_valid else None

        first_signal = None
        if "time" in df.columns:
            if numeric_valid:
                mask = numeric > 0
            else:
                mask = values.notna()
            signal_times = df.loc[mask & df["time"].notna(), "time"]
            if not signal_times.empty:
                first_signal = float(signal_times.min())

        return SeriesStats(rows=len(df), non_null=int(values.notna().sum()), mean=mean, maximum=maximum, first_signal=first_signal)


class RicuPyricuComparator:
    """Loads ricu CSVs and pyricu concepts and produces coverage comparisons."""

    def __init__(
        self,
        database: str = "miiv",
        data_path: Optional[Path | str] = None,
        ricu_data_path: Optional[Path | str] = None,
        patients: Optional[Sequence[int]] = None,
        max_patients: Optional[int] = None,
    ) -> None:
        if database not in SUPPORTED_DATABASES:
            raise ValueError(f"Unsupported database '{database}'. Expected one of {SUPPORTED_DATABASES}.")
        self.database = database
        self.data_path = self._resolve_data_path(database, data_path)
        self.ricu_data_path = self._resolve_ricu_path(database, ricu_data_path)
        self.test_patients = self._resolve_patients(patients, max_patients)
        self._fio2_override: Optional[pd.DataFrame] = None
        self._icustay_times: Optional[pd.DataFrame] = None
        dict_path = REPO_ROOT / "ricu" / "inst" / "extdata" / "config" / "concept-dict.json"
        self._loader_kwargs: Dict[str, object] = {}
        if dict_path.exists():
            self._loader_kwargs["dict_path"] = str(dict_path)

        self._interval_cap_hours = 24 * 365  # prevent runaway window expansion
        patient_goal = (
            len(self.test_patients)
            if isinstance(self.test_patients, list)
            else (max_patients if max_patients and max_patients > 0 else None)
        )
        self._loader_defaults = resolve_loader_defaults(
            patient_goal,
            env_keys=ALIGNMENT_ENV_KEYS,
        )
        loader_kwargs = self._loader_defaults.as_loader_kwargs(progress=False)
        for key, value in loader_kwargs.items():
            if value is None:
                continue
            self._loader_kwargs[key] = value

        self._concept_cache: Dict[Tuple[str, bool], pd.DataFrame] = {}

    def run(self) -> Dict[str, FeatureComparison]:
        self._ensure_fio2_patch()
        results: Dict[str, FeatureComparison] = {}
        print(
            f"⚙️  Loader profile [{self.database}]: {self._loader_defaults.summary()}"
        )
        for module in MODULES:
            comparison = self._compare_module(module)
            results[module.name] = comparison
            self._print_module_summary(module, comparison)
        self._print_dependency_report(results)
        return results

    def _compare_module(self, module: FeatureModule) -> FeatureComparison:
        ricu_series = self._load_ricu_series(module)
        reference_grid = self._build_time_grid(ricu_series, module)
        pyricu_series = self._load_pyricu_series(module, reference_grid, ricu_series)
        return FeatureComparison(module, ricu_series, pyricu_series)

    def _load_ricu_series(self, module: FeatureModule) -> Dict[str, pd.DataFrame]:
        file_name = module.ricu_file.format(db=self.database)
        file_path = self.ricu_data_path / file_name
        if not file_path.exists():
            print(f"⚠️  Missing ricu file: {file_path}")
            return {}
        
        # 🚀 OPTIMIZATION: Load only header first to detect columns, then filter during read
        sample_df = pd.read_csv(file_path, nrows=0, low_memory=False)
        id_column = module.id_column if module.id_column in sample_df.columns else self._detect_column(sample_df, module.id_column, ID_CANDIDATES)
        time_column = None
        if module.time_column:
            if module.time_column in sample_df.columns:
                time_column = module.time_column
            else:
                time_column = self._detect_column(sample_df, module.time_column, TIME_CANDIDATES)
        
        # 🚀 OPTIMIZATION: Filter patients during CSV parsing using chunked reader
        if id_column and self.test_patients:
            chunks = []
            chunksize = 50000  # Read in chunks to save memory
            for chunk in pd.read_csv(file_path, low_memory=False, chunksize=chunksize):
                filtered = chunk[chunk[id_column].isin(self.test_patients)]
                if not filtered.empty:
                    chunks.append(filtered)
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=sample_df.columns)
        else:
            df = pd.read_csv(file_path, low_memory=False)
        
        return self._split_wide_table(df, module, detected_id_column=id_column, detected_time_column=time_column)

    def _load_pyricu_series(
        self,
        module: FeatureModule,
        reference_grid: Optional[pd.DataFrame] = None,
        reference_series: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, pd.DataFrame]:
        frames: Dict[str, pd.DataFrame] = {}
        for name in module.concepts:
            try:
                frame = self._load_concept_frame(name, self.test_patients or None, module)
            except Exception as exc:
                print(f"   ⚠️  concept {name} failed in module {module.name}: {exc}")
                continue
            if name == "fio2" and self._fio2_override is not None:
                frame = self._fio2_override.copy()
            series = self._normalize_concept_frame(frame, module, name)
            if self._should_retry_per_patient(series, frame):
                retry = self._reload_concept_per_patient(name, module)
                if retry is not None:
                    series = retry
            if series is not None:
                frames[name] = series

        align_grid: Optional[pd.DataFrame] = None
        if module.time_column:
            if reference_grid is not None and not reference_grid.empty:
                align_grid = reference_grid.copy()
            elif frames:
                align_grid = self._build_time_grid(frames, module)

        if align_grid is not None and not align_grid.empty:
            grid = align_grid.copy()
            grid["id"] = pd.to_numeric(grid["id"], errors="coerce")
            grid["time"] = pd.to_numeric(grid["time"], errors="coerce")
            grid = grid.dropna(subset=["id", "time"]).drop_duplicates().reset_index(drop=True)
            if not grid.empty:
                for name, df in list(frames.items()):
                    if "time" not in df.columns:
                        continue
                    df = df.copy()
                    df["id"] = pd.to_numeric(df["id"], errors="coerce")
                    df["time"] = pd.to_numeric(df["time"], errors="coerce")
                    df = df.dropna(subset=["id", "time"])
                    aligned = grid.merge(df, on=["id", "time"], how="left")
                    frames[name] = aligned
                align_grid = grid
            else:
                align_grid = None

        base_placeholder = align_grid.copy() if module.time_column and align_grid is not None and not align_grid.empty else None

        for concept in module.concepts:
            if concept in frames:
                continue
            placeholder: Optional[pd.DataFrame] = None
            if base_placeholder is not None:
                placeholder = base_placeholder.assign(value=np.nan)
            elif reference_series and concept in reference_series:
                ref_df = reference_series[concept]
                if not ref_df.empty and "id" in ref_df.columns:
                    cols = ["id"]
                    if "time" in ref_df.columns:
                        cols.append("time")
                    placeholder = ref_df[cols].copy()
                    placeholder["value"] = np.nan
            if placeholder is not None:
                frames[concept] = placeholder.reset_index(drop=True)

        return frames

    def _split_wide_table(
        self,
        df: pd.DataFrame,
        module: FeatureModule,
        detected_id_column: Optional[str] = None,
        detected_time_column: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        id_column = detected_id_column
        if not id_column or id_column not in df.columns:
            id_column = self._detect_column(df, module.id_column, ID_CANDIDATES)
        if id_column is None or id_column not in df.columns:
            return result
        time_column = detected_time_column
        if module.time_column:
            if not time_column or time_column not in df.columns:
                time_column = self._detect_column(df, module.time_column, TIME_CANDIDATES)
        base_cols = [id_column]
        if time_column and time_column in df.columns:
            base_cols.append(time_column)
        for column in df.columns:
            if column in base_cols:
                continue
            subset_cols = base_cols + [column]
            series = df[subset_cols].copy()
            rename_map = {column: "value", id_column: "id"}
            if time_column and time_column in series.columns:
                rename_map[time_column] = "time"
            series = series.rename(columns=rename_map)
            if "time" in series.columns and not pd.api.types.is_numeric_dtype(series["time"]):
                id_ref = series["id"] if "id" in series.columns else None
                series["time"] = self._time_to_hours(series["time"], id_ref)
            series = series.dropna(subset=["id"])
            result[column] = series.reset_index(drop=True)
        return result

    def _normalize_concept_frame(
        self,
        frame: pd.DataFrame,
        module: FeatureModule,
        concept: str,
        *,
        force_id: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        if frame is None or (hasattr(frame, "empty") and frame.empty):
            return None
        if isinstance(frame, pd.Series):
            df = frame.to_frame(name="value").reset_index()
        else:
            df = frame.copy()
        if df.empty:
            return None
        
        # Enhanced ID column detection: try preferred column, then fallbacks
        id_col = self._detect_column(df, module.id_column, ID_CANDIDATES)
        
        # Special handling: if we got subject_id but need stay_id (or vice versa),
        # convert it now using icustays mapping BEFORE renaming to "id"
        needs_conversion = False
        if id_col == "subject_id" and module.id_column == "stay_id":
            needs_conversion = True
        elif id_col == "stay_id" and module.id_column == "subject_id":
            needs_conversion = True
        
        # Convert BEFORE renaming to preserve original column name
        if needs_conversion:
            df = self._convert_id_column(df, from_col=id_col, to_col=module.id_column)
            # Update id_col to reflect the converted column
            id_col = module.id_column
        
        if id_col is None:
            return None
        rename_map = {id_col: "id"}
        cols = [id_col]
        time_col = None
        if module.time_column:
            time_col = self._detect_column(df, module.time_column, TIME_CANDIDATES)
            if time_col:
                rename_map[time_col] = "time"
                cols.append(time_col)
        value_col = concept if concept in df.columns else self._detect_value_column(df, cols)
        if value_col is None:
            return None
        rename_map[value_col] = "value"
        cols.append(value_col)

        # keep auxiliary columns so we can expand windows or repair ids later
        end_col = self._detect_column(df, "endtime", ["endtime", "end_time", "stop", f"{concept}_end", f"{concept}_endtime"])
        duration_col = self._detect_column(df, f"{concept}_dur", [f"{concept}_dur", f"{concept}_duration", "duration", "dur", "hours", "length"])
        backup_ids = [candidate for candidate in ("subject_id", "hadm_id", "patientunitstayid") if candidate in df.columns]
        for extra in filter(None, [end_col, duration_col]):
            if extra not in cols and extra in df.columns:
                cols.append(extra)
        for backup in backup_ids:
            if backup not in cols:
                cols.append(backup)
                rename_map.setdefault(backup, backup)
        if end_col:
            rename_map[end_col] = "endtime"
        if duration_col:
            rename_map[duration_col] = "duration"
        df = df[cols].rename(columns=rename_map)
        if "value" in df.columns and pd.api.types.is_timedelta64_dtype(df["value"]):
            df["value"] = df["value"].dt.total_seconds() / 3600.0
        # Drop duplicate columns (e.g., repeated component outputs) while keeping first values
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        if "time" in df.columns and not pd.api.types.is_numeric_dtype(df["time"]):
            id_ref = df["id"] if "id" in df.columns else None
            df["time"] = self._time_to_hours(df["time"], id_ref)
        if "time" in df.columns:
            df = df.dropna(subset=["time"])
        
        # Fill any remaining missing IDs
        df = self._fill_missing_ids(df)
        
        if force_id is not None and "id" in df.columns:
            df["id"] = df["id"].fillna(force_id)
        df = df.dropna(subset=["id"]).reset_index(drop=True)
        df = self._expand_interval_rows(df, module, concept)
        return df

    def _load_concept_frame(
        self,
        concept: str,
        patient_ids: Optional[Sequence[int]],
        module: FeatureModule,
    ) -> pd.DataFrame:
        cache_key = (concept, bool(module.time_column))
        cached = self._concept_cache.get(cache_key)
        if cached is not None:
            return cached

        frame = load_concepts(
            concept,
            patient_ids=patient_ids,
            database=self.database,
            data_path=str(self.data_path),
            interval="1h" if module.time_column else None,
            merge=True,
            verbose=False,
            _allow_missing_concept=True,
            **self._loader_kwargs,
        )
        self._concept_cache[cache_key] = frame
        return frame

    def _should_retry_per_patient(
        self,
        series: Optional[pd.DataFrame],
        raw_frame: Optional[pd.DataFrame],
    ) -> bool:
        if not self.test_patients:
            return False
        if raw_frame is None or not isinstance(raw_frame, pd.DataFrame) or raw_frame.empty:
            return False
        if series is None or series.empty:
            return True
        if "id" not in series.columns:
            return True
        return not series["id"].notna().any()

    def _reload_concept_per_patient(self, concept: str, module: FeatureModule) -> Optional[pd.DataFrame]:
        if not self.test_patients:
            return None
        per_patient: List[pd.DataFrame] = []
        for stay_id in self.test_patients:
            try:
                frame = self._load_concept_frame(concept, [stay_id], module)
            except Exception as exc:
                logger.warning("Failed reloading %s for stay %s: %s", concept, stay_id, exc)
                continue
            normalized = self._normalize_concept_frame(frame, module, concept, force_id=stay_id)
            if normalized is not None and not normalized.empty:
                per_patient.append(normalized)
        if not per_patient:
            return None
        combined = pd.concat(per_patient, ignore_index=True)
        sort_cols = ["id"] + (["time"] if "time" in combined.columns else [])
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
        return combined

    def _convert_id_column(self, df: pd.DataFrame, from_col: str, to_col: str) -> pd.DataFrame:
        """Convert between ID column types (e.g., subject_id <-> stay_id).
        
        This function expects the ORIGINAL column name (e.g., 'subject_id')
        and replaces it with the target column (e.g., 'stay_id').
        """
        if from_col not in df.columns:
            logger.warning(f"Cannot convert {from_col} to {to_col}: {from_col} not in columns")
            return df
        
        lookup = self._load_icustay_times()
        if lookup.empty or from_col not in lookup.columns or to_col not in lookup.columns:
            logger.warning(f"Cannot convert {from_col} to {to_col}: missing in lookup table")
            return df
        
        # Build mapping table: from_col → to_col
        bridge = lookup[[from_col, to_col]].dropna().drop_duplicates()
        if bridge.empty:
            logger.warning(f"Cannot convert {from_col} to {to_col}: no valid mappings")
            return df
        
        # Merge directly on original column name
        merged = df.merge(bridge, on=from_col, how="left")
        
        # Drop the old column, keep the new one
        if from_col in merged.columns:
            merged = merged.drop(columns=[from_col])
        
        return merged
    
    def _fill_missing_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing ID values using backup ID columns."""
        if "id" not in df.columns:
            return df
        
        # If all IDs are present, nothing to do
        if df["id"].notna().all():
            return df
        
        lookup = self._load_icustay_times()
        if lookup.empty:
            return df

        merged = df.copy()
        for column in ("hadm_id", "subject_id"):
            if column not in merged.columns or column not in lookup.columns:
                continue
            bridge = lookup[["stay_id", column]].dropna().drop_duplicates()
            if bridge.empty:
                continue
            merged = merged.merge(bridge, on=column, how="left", suffixes=("", "_mapped"))
            mapped_col = "stay_id"
            if mapped_col in merged.columns:
                merged["id"] = merged["id"].fillna(merged[mapped_col])
                merged = merged.drop(columns=[mapped_col])
        
        # Clean up backup ID columns
        for cleanup in ("subject_id", "hadm_id", "patientunitstayid"):
            if cleanup in merged.columns:
                merged = merged.drop(columns=[cleanup])
        return merged

    def _expand_interval_rows(self, df: pd.DataFrame, module: FeatureModule, concept: str) -> pd.DataFrame:
        # Check if we have time column (required for expansion)
        if "time" not in df.columns:
            return df.drop(columns=["endtime", "duration"], errors="ignore")

        has_end = "endtime" in df.columns and df["endtime"].notna().any()
        has_duration = "duration" in df.columns and df["duration"].notna().any()

        # Duration-style concepts (e.g., *_dur) already encode exposure length and
        # should not be expanded into per-hour series.
        concept_lower = concept.lower()
        if concept_lower.endswith("_dur"):
            return df.drop(columns=["endtime", "duration"], errors="ignore")
        
        # Logical/boolean concepts (abx, samp, etc.) are event indicators and
        # should NOT be expanded into continuous time series. They represent
        # discrete events (e.g., antibiotic administration started at time X),
        # not continuous states. Expanding them would incorrectly fill gaps.
        logical_concepts = {"abx", "samp", "cort"}  # Add other event-based concepts as needed
        if concept_lower in logical_concepts:
            return df.drop(columns=["endtime", "duration"], errors="ignore")

        # Do not fabricate synthetic durations when the upstream concept already
        # provides discrete time points (ricu treats those as point events).  The
        # previous heuristic expanded every measurement until the next one,
        # drastically overestimating vasopressor exposure compared to ricu.
        if not has_end and not has_duration:
            return df.drop(columns=["endtime", "duration"], errors="ignore")
        
        # Expand window-based concepts (mech_vent, etc.) from all modules, not just med
        # This includes respiratory concepts like mech_vent that have start/end times

        working = df.copy()
        
        # Debug
        if has_end:
            import sys
            print(f"DEBUG _expand_interval_rows: concept={concept}, has_end={has_end}", file=sys.stderr)
            print(f"DEBUG endtime dtype: {working['endtime'].dtype}", file=sys.stderr)
            print(f"DEBUG endtime is_numeric: {pd.api.types.is_numeric_dtype(working['endtime'])}", file=sys.stderr)
            if len(working) > 0:
                print(f"DEBUG endtime first value: {working['endtime'].iloc[0]}", file=sys.stderr)
            print(f"DEBUG working has 'id' column: {'id' in working.columns}", file=sys.stderr)
            print(f"DEBUG working columns: {working.columns.tolist()}", file=sys.stderr)

        if has_end and not pd.api.types.is_numeric_dtype(working["endtime"]):
            import sys
            icu = self._load_icustay_times()
            print(f"DEBUG: ICU times loaded, empty={icu.empty}, has_intime={'intime' in icu.columns if not icu.empty else False}", file=sys.stderr)
            
            if not icu.empty and "intime" in icu.columns:
                # Determine the ID column in working DataFrame
                id_col_in_working = "id" if "id" in working.columns else "stay_id" if "stay_id" in working.columns else None
                print(f"DEBUG: id_col_in_working={id_col_in_working}", file=sys.stderr)
                
                if id_col_in_working:
                    icu_map = icu[["stay_id", "intime"]].dropna().drop_duplicates()
                    icu_map["intime"] = pd.to_datetime(icu_map["intime"], errors="coerce").dt.tz_localize(None)
                    print(f"DEBUG: icu_map shape={icu_map.shape}, first intime={icu_map['intime'].iloc[0] if len(icu_map) > 0 else 'N/A'}", file=sys.stderr)
                    
                    # Rename to match working's ID column
                    if id_col_in_working != "stay_id":
                        icu_map = icu_map.rename(columns={"stay_id": id_col_in_working})
                    
                    print(f"DEBUG: Before merge, working shape={working.shape}", file=sys.stderr)
                    working = working.merge(icu_map, on=id_col_in_working, how="left")
                    print(f"DEBUG: After merge, working shape={working.shape}, has_intime={'intime' in working.columns}", file=sys.stderr)
                    
                    end_dt = pd.to_datetime(working["endtime"], errors="coerce").dt.tz_localize(None)
                    print(f"DEBUG: end_dt first value={end_dt.iloc[0] if len(end_dt) > 0 else 'N/A'}", file=sys.stderr)
                    
                    working["endtime"] = (
                        (end_dt - working["intime"]).dt.total_seconds() / 3600.0
                    )
                    print(f"DEBUG: After conversion, endtime first value={working['endtime'].iloc[0] if len(working) > 0 else 'N/A'}", file=sys.stderr)
                    working = working.drop(columns=["intime"], errors="ignore")
                else:
                    # Fallback: try to parse as timestamp
                    working["endtime"] = self._time_to_hours(working["endtime"], None)
            else:
                working["endtime"] = self._time_to_hours(working["endtime"], working.get("id"))
        if has_duration:
            if pd.api.types.is_timedelta64_dtype(working["duration"]):
                working["duration"] = working["duration"].dt.total_seconds() / 3600.0
            working["duration"] = pd.to_numeric(working["duration"], errors="coerce")

        starts = pd.to_numeric(working["time"], errors="coerce")
        if has_end:
            ends = pd.to_numeric(working["endtime"], errors="coerce")
        elif has_duration:
            ends = starts + working["duration"].fillna(0)
        else:
            ends = starts

        records: List[dict] = []
        skipped = 0
        for idx, (start, end, value, stay_id) in enumerate(zip(starts, ends, working.get("value"), working.get("id"))):
            if pd.isna(start) or pd.isna(end) or pd.isna(stay_id) or pd.isna(value):
                continue
            if end < start:
                end = start
            span = min(end - start, self._interval_cap_hours)
            if span <= 0:
                records.append({"id": stay_id, "time": float(start), "value": value})
                continue
            if span >= self._interval_cap_hours:
                skipped += 1
            start_hour = int(math.floor(start))
            end_hour = int(math.ceil(min(end, start + self._interval_cap_hours)))
            for hour in range(start_hour, end_hour + 1):
                records.append({"id": stay_id, "time": float(hour), "value": value})
        if not records:
            return df.drop(columns=["endtime", "duration"], errors="ignore")
        expanded = pd.DataFrame.from_records(records)
        expanded = expanded.drop_duplicates(subset=["id", "time", "value"]).reset_index(drop=True)
        return expanded

    def _detect_column(self, df: pd.DataFrame, preferred: Optional[str], fallbacks: Iterable[str]) -> Optional[str]:
        if preferred and preferred in df.columns:
            return preferred
        for candidate in fallbacks:
            if candidate in df.columns:
                return candidate
        return None

    def _detect_value_column(self, df: pd.DataFrame, exclude: Iterable[str]) -> Optional[str]:
        exclude_set = set(exclude)
        numeric_cols = [c for c in df.columns if c not in exclude_set]
        return numeric_cols[0] if numeric_cols else None

    def _time_to_hours(self, series: pd.Series, id_series: Optional[pd.Series]) -> pd.Series:
        if series.empty:
            return series
        if pd.api.types.is_datetime64_any_dtype(series):
            clean = series.dt.tz_localize(None)
            if id_series is not None:
                return clean.groupby(id_series).transform(lambda s: (s - s.min()).dt.total_seconds() / 3600.0)
            return (clean - clean.min()).dt.total_seconds() / 3600.0
        if pd.api.types.is_timedelta64_dtype(series):
            return series.dt.total_seconds() / 3600.0
        return pd.to_numeric(series, errors="coerce")

    def _build_time_grid(
        self,
        series_dict: Dict[str, pd.DataFrame],
        module: FeatureModule,
    ) -> Optional[pd.DataFrame]:
        if not module.time_column:
            return None
        frames = [
            df[["id", "time"]]
            for df in series_dict.values()
            if isinstance(df, pd.DataFrame) and {"id", "time"}.issubset(df.columns)
        ]
        if not frames:
            return None
        grid = (
            pd.concat(frames, ignore_index=True)
            .dropna(subset=["id", "time"])
            .drop_duplicates()
            .sort_values(["id", "time"])
            .reset_index(drop=True)
        )
        return grid if not grid.empty else None

    def _ensure_fio2_patch(self) -> None:
        """Apply fio2 patch only for MIMIC-IV database."""
        if self._fio2_override is not None:
            return
        # Only apply this patch for miiv database (chartevents.parquet is miiv-specific)
        if self.database not in ("miiv", "mimiciv"):
            return
        base = self._load_pyricu_fio2()
        base_has_values = (
            isinstance(base, pd.DataFrame)
            and not base.empty
            and "fio2" in base.columns
            and base["fio2"].notna().any()
        )
        if base_has_values:
            # 现在pyricu的fio2加载已经可以解析百分号字符串，直接使用标准结果即可
            return
        raw = self._load_raw_fio2()
        frames = [df for df in (base, raw) if isinstance(df, pd.DataFrame) and not df.empty]
        if not frames:
            return
        merged = pd.concat(frames, ignore_index=True)
        # Check if stay_id column exists before using it
        if "stay_id" not in merged.columns:
            return
        merged = merged.dropna(subset=["stay_id"])
        if "charttime" not in merged.columns:
            return
        merged["charttime"] = pd.to_numeric(merged["charttime"], errors="coerce")
        merged["charttime"] = np.floor(merged["charttime"].astype(float))
        merged = merged.sort_values(["stay_id", "charttime"])
        merged = merged.drop_duplicates(subset=["stay_id", "charttime"], keep="first")
        self._fio2_override = merged.reset_index(drop=True)

    def _load_pyricu_fio2(self) -> pd.DataFrame:
        try:
            df = load_concepts(
                "fio2",
                patient_ids=self.test_patients or None,
                database=self.database,
                data_path=str(self.data_path),
                merge=True,
                verbose=False,
                **self._loader_kwargs,
            )
        except Exception:
            return pd.DataFrame(columns=["stay_id", "charttime", "fio2"])
        keep = [c for c in ["stay_id", "charttime", "fio2"] if c in df.columns]
        return df[keep].copy() if keep else pd.DataFrame(columns=["stay_id", "charttime", "fio2"])

    def _load_raw_fio2(self) -> pd.DataFrame:
        chart_path = Path(self.data_path) / "chartevents.parquet"
        if not chart_path.exists():
            return pd.DataFrame(columns=["stay_id", "charttime", "fio2"])
        chart = pd.read_parquet(chart_path, columns=["stay_id", "charttime", "itemid", "valuenum"])
        if self.test_patients:
            chart = chart[chart["stay_id"].isin(self.test_patients)]
        chart = chart[chart["itemid"].isin(FIO2_ITEMIDS)]
        chart = chart.dropna(subset=["valuenum", "charttime"])
        if chart.empty:
            return pd.DataFrame(columns=["stay_id", "charttime", "fio2"])
        times = self._load_icustay_times()
        chart = chart.merge(times, on="stay_id", how="left")
        chart = chart.dropna(subset=["intime"])
        chart["charttime"] = (
            pd.to_datetime(chart["charttime"]) - pd.to_datetime(chart["intime"])
        ).dt.total_seconds() / 3600.0
        chart["charttime"] = np.floor(chart["charttime"].astype(float))
        chart = chart.dropna(subset=["charttime"])
        chart["fio2"] = pd.to_numeric(chart["valuenum"], errors="coerce")
        return chart[["stay_id", "charttime", "fio2"]].dropna(subset=["fio2"])

    def _load_icustay_times(self) -> pd.DataFrame:
        if self._icustay_times is not None:
            return self._icustay_times
        icu_path = Path(self.data_path) / "icustays.parquet"
        if icu_path.exists():
            desired = ["stay_id", "subject_id", "hadm_id", "intime", "outtime"]
            try:
                df = pd.read_parquet(icu_path, columns=desired)
            except Exception:
                df = pd.read_parquet(icu_path)
                keep = [col for col in desired if col in df.columns]
                df = df[keep] if keep else df
            for time_col in ("intime", "outtime"):
                if time_col in df.columns:
                    df[time_col] = pd.to_datetime(df[time_col])
        else:
            df = pd.DataFrame(columns=["stay_id", "subject_id", "hadm_id", "intime", "outtime"])
        self._icustay_times = df
        return df

    def _print_module_summary(self, module: FeatureModule, comparison: FeatureComparison) -> None:
        print(f"\n[{module.name.upper()}] {module.description}")
        metrics = comparison.compute_metrics()
        if not metrics:
            print("   ⚠️  No comparable concepts loaded.")
            return
        for name, stats in metrics.items():
            coverage_gap = stats["coverage_gap"]
            status = stats["status"]
            mean_diff = stats["mean_diff"]
            mean_repr = "  N/A" if mean_diff is None else f"{mean_diff:6.2f}"
            print(
                f"   - {name:12s} "
                f"ricu_rows={stats['ricu_rows']:4d} "
                f"py_rows={stats['pyricu_rows']:4d} "
                f"cov_gap={coverage_gap:6.2f} "
                f"mean_diff={mean_repr} "
                f"status={status}"
            )

    def _print_dependency_report(self, results: Dict[str, FeatureComparison]) -> None:
        print("\nSOFA dependency coverage:")
        for component, deps in SOFA_COMPONENT_DEPENDENCIES.items():
            missing = [dep for dep in deps if not self._has_concept(results, dep)]
            if missing:
                print(f" - {component}: missing {', '.join(missing)}")
            else:
                print(f" - {component}: OK")

    def _has_concept(self, results: Dict[str, FeatureComparison], concept: str) -> bool:
        for comparison in results.values():
            series = comparison.pyricu_series.get(concept)
            if series is None or series.empty:
                continue
            if "value" not in series.columns:
                continue
            value_col = series["value"]
            if isinstance(value_col, pd.DataFrame):
                has_values = bool(value_col.stack().notna().any())
            else:
                has_values = bool(pd.Series(value_col).notna().any())
            if has_values:
                return True
        return False

    def _resolve_data_path(self, database: str, override: Optional[Path | str]) -> Path:
        if override is not None:
            path = Path(override).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Data path {path} does not exist")
            return path
        default = DEFAULT_DATA_PATHS.get(database)
        if default is None:
            raise ValueError(f"No default data path for database '{database}'. Please pass --data-path.")
        path = Path(default).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Default data path for '{database}' not found at {path}")
        return path

    def _resolve_ricu_path(self, database: str, override: Optional[Path | str]) -> Path:
        if override is not None:
            candidate = Path(override).expanduser().resolve()
            if candidate.is_dir():
                return candidate
            base = candidate / database
            if base.is_dir():
                return base
            raise FileNotFoundError(f"Cannot locate ricu data for '{database}' under {candidate}")
        candidate = (DEFAULT_RICU_ROOT / database).resolve()
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Cannot locate default ricu data for '{database}' (looked in {candidate})")

    def _resolve_patients(
        self,
        override: Optional[Sequence[int]],
        max_patients: Optional[int],
    ) -> Optional[List[int]]:
        if override:
            ids = [int(pid) for pid in override]
        else:
            ids = list(DEFAULT_PATIENTS.get(self.database, []))
        if not ids:
            ids = self._discover_patient_ids(max_patients or 5)
        if max_patients is not None and ids:
            ids = ids[:max_patients]
        return ids or None

    def _discover_patient_ids(self, limit: int) -> List[int]:
        table, column = PATIENT_ID_SOURCES.get(self.database, (None, None))
        if table is None or column is None:
            return []
        path = Path(self.data_path) / table
        if not path.exists():
            return []
        try:
            df = pd.read_parquet(path, columns=[column])
        except Exception:
            return []
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        unique_ids: List[int] = []
        for value in series:
            candidate = int(value)
            if candidate not in unique_ids:
                unique_ids.append(candidate)
            if len(unique_ids) >= limit:
                break
        return unique_ids


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ricu CSV exports with pyricu outputs.")
    parser.add_argument(
        "-d",
        "--database",
        choices=SUPPORTED_DATABASES,
        nargs="+",
        default=["miiv"],
        help="Database(s) to evaluate (default: miiv).",
    )
    parser.add_argument(
        "--data-path",
        help="Override pyricu data path template. Use {db} to substitute the database name when running multiple targets.",
    )
    parser.add_argument(
        "--ricu-data",
        help="Override ricu CSV path template. Use {db} placeholder for multi-database runs. Defaults to <repo>/ricu_data/<db>.",
    )
    parser.add_argument(
        "-p",
        "--patients",
        nargs="+",
        type=int,
        help="Explicit patient IDs to test. Applies to all databases in this run.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        help="Limit the number of patients pulled from the default/discovered lists.",
    )
    return parser.parse_args(argv)


def _format_path_template(template: Optional[str], database: str, fallback: Optional[Path]) -> Path:
    if not template:
        if fallback is None:
            raise ValueError(f"No default path available for database '{database}'. Please supply --data-path/--ricu-data.")
        return fallback
    value = template.format(db=database)
    return Path(value).expanduser().resolve()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    for index, database in enumerate(args.database):
        data_default = DEFAULT_DATA_PATHS.get(database)
        data_path = _format_path_template(args.data_path, database, data_default)
        ricu_default = (DEFAULT_RICU_ROOT / database).resolve()
        ricu_path = _format_path_template(args.ricu_data, database, ricu_default)
        comparator = RicuPyricuComparator(
            database=database,
            data_path=data_path,
            ricu_data_path=ricu_path,
            patients=args.patients,
            max_patients=args.max_patients,
        )
        header = f"\n{'=' * 30} Database: {database} {'=' * 30}"
        if index == 0:
            header = header.lstrip("\n")
        print(header)
        comparator.run()


__all__ = [
    "FeatureModule",
    "FeatureComparison",
    "RicuPyricuComparator",
    "MODULES",
    "SOFA_COMPONENT_DEPENDENCIES",
    "main",
]
