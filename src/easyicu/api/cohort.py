"""Cohort-domain services behind :mod:`easyicu.api`."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Union

import pandas as pd

from ..patient_filter import PatientFilter
from ..patient_filter import get_cohort_stats as _get_cohort_stats


class PatientIdDiscoveryError(RuntimeError):
    """Raised when the patient/stay universe cannot be established safely."""


def get_id_col_for_database(
    database: str,
    *,
    database_id_config: Mapping[str, Mapping[str, str]],
) -> str:
    """Return the configured stay-identifier column, rejecting unknown keys."""
    try:
        return database_id_config[database]["id_col"]
    except KeyError as exc:
        raise ValueError(f"Unsupported database for patient ID lookup: {database!r}") from exc


def get_patient_table_for_database(
    database: str,
    *,
    database_id_config: Mapping[str, Mapping[str, str]],
) -> str:
    """Return the configured patient table, rejecting unknown keys."""
    try:
        return database_id_config[database]["table"]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported database for patient table lookup: {database!r}"
        ) from exc


def get_all_patient_ids_impl(
    data_path: Union[str, Path],
    *,
    database_id_config: Mapping[str, Mapping[str, str]],
    detect_database_type_fn: Callable[..., str],
    base_loader_cls: Callable[..., object],
    sample_patient_ids_fn: Callable[..., Optional[List]],
    database: Optional[str] = None,
    max_patients: Optional[int] = None,
) -> tuple[List, str]:
    """Discover the stay universe without turning read failures into emptiness."""
    resolved_database = database or detect_database_type_fn(data_path)
    id_col = get_id_col_for_database(
        resolved_database,
        database_id_config=database_id_config,
    )
    table_name = get_patient_table_for_database(
        resolved_database,
        database_id_config=database_id_config,
    )
    root = Path(data_path)
    aliases = {"general": ("general", "general_table")}
    name_candidates = aliases.get(table_name, (table_name,))

    try:
        all_ids: Optional[List] = None
        parquet_file = None
        for name in name_candidates:
            flat = root / f"{name}.parquet"
            if flat.exists():
                parquet_file = flat
                break
            nested = next(iter(sorted(root.glob(f"*/{name}.parquet"))), None)
            if nested is not None:
                parquet_file = nested
                break
        if parquet_file is not None:
            try:
                frame = pd.read_parquet(parquet_file, columns=[id_col])
                all_ids = frame[id_col].dropna().unique().tolist()
            except Exception:
                # A present but unreadable Parquet file may have a CSV sibling.
                all_ids = None

        if all_ids is None:
            for suffix in (".csv", ".csv.gz"):
                csv_file = root / f"{table_name}{suffix}"
                if not csv_file.exists():
                    continue
                try:
                    frame = pd.read_csv(csv_file, usecols=[id_col])
                    all_ids = frame[id_col].dropna().unique().tolist()
                    break
                except Exception:
                    continue

        if all_ids is None:
            shard_dir = root / table_name
            if shard_dir.is_dir():
                all_ids = []
                for shard in sorted(shard_dir.glob("*.parquet")):
                    frame = pd.read_parquet(shard, columns=[id_col])
                    all_ids.extend(frame[id_col].dropna().unique().tolist())
                all_ids = list(dict.fromkeys(all_ids))
            else:
                loader = base_loader_cls(
                    database=resolved_database,
                    data_path=root,
                    verbose=False,
                )
                sampled = sample_patient_ids_fn(
                    loader,
                    max_patients or 999_999_999,
                    verbose=False,
                )
                return list(sampled or []), id_col

        if max_patients and len(all_ids) > max_patients:
            all_ids = all_ids[:max_patients]
        return all_ids, id_col
    except Exception as exc:
        raise PatientIdDiscoveryError(
            "Unable to establish the patient ID universe "
            f"for {resolved_database!r} ({type(exc).__name__})."
        ) from exc


def _resolve_source(
    *,
    database: Optional[str],
    data_path: Optional[Union[str, Path]],
    detect_database_type_fn: Callable[..., str],
    get_default_data_path_fn: Callable[..., Union[str, Path]],
) -> tuple[str, Union[str, Path]]:
    resolved_database = database or detect_database_type_fn(data_path)
    resolved_path = data_path or get_default_data_path_fn(resolved_database)
    return resolved_database, resolved_path


def filter_patients_impl(
    *,
    detect_database_type_fn: Callable[..., str],
    get_default_data_path_fn: Callable[..., Union[str, Path]],
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
    first_icu_stay: Optional[bool] = None,
    los_min: Optional[float] = None,
    los_max: Optional[float] = None,
    gender: Optional[str] = None,
    survived: Optional[bool] = None,
    has_sepsis: Optional[bool] = None,
    return_dataframe: bool = False,
    verbose: bool = False,
) -> Union[List[int], pd.DataFrame]:
    """Filter an ICU cohort through the canonical patient-filter service."""
    resolved_database, resolved_path = _resolve_source(
        database=database,
        data_path=data_path,
        detect_database_type_fn=detect_database_type_fn,
        get_default_data_path_fn=get_default_data_path_fn,
    )
    patient_filter = PatientFilter(
        database=resolved_database,
        data_path=resolved_path,
        verbose=verbose,
    )
    return patient_filter.filter(
        age_min=age_min,
        age_max=age_max,
        first_icu_stay=first_icu_stay,
        los_min=los_min,
        los_max=los_max,
        gender=gender,
        survived=survived,
        has_sepsis=has_sepsis,
        return_dataframe=return_dataframe,
    )


def load_concepts_filtered_impl(
    concepts: Union[str, List[str]],
    *,
    filter_patients_fn: Callable[..., Union[List[int], pd.DataFrame]],
    load_concepts_fn: Callable[..., pd.DataFrame],
    detect_database_type_fn: Callable[..., str],
    get_default_data_path_fn: Callable[..., Union[str, Path]],
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
    first_icu_stay: Optional[bool] = None,
    los_min: Optional[float] = None,
    los_max: Optional[float] = None,
    gender: Optional[str] = None,
    survived: Optional[bool] = None,
    has_sepsis: Optional[bool] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Optional[Union[str, pd.Timedelta]] = "1h",
    win_length: Optional[Union[str, pd.Timedelta]] = None,
    aggregate: Optional[Union[str, Dict]] = None,
    keep_components: bool = False,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Filter a cohort, then load concepts for exactly that cohort."""
    resolved_database, resolved_path = _resolve_source(
        database=database,
        data_path=data_path,
        detect_database_type_fn=detect_database_type_fn,
        get_default_data_path_fn=get_default_data_path_fn,
    )
    filter_values = (
        age_min,
        age_max,
        first_icu_stay,
        los_min,
        los_max,
        gender,
        survived,
        has_sepsis,
    )
    patient_ids = None
    if any(value is not None for value in filter_values):
        if verbose:
            print("🔍 第1步：筛选患者队列...")
        patient_ids = filter_patients_fn(
            database=resolved_database,
            data_path=resolved_path,
            age_min=age_min,
            age_max=age_max,
            first_icu_stay=first_icu_stay,
            los_min=los_min,
            los_max=los_max,
            gender=gender,
            survived=survived,
            has_sepsis=has_sepsis,
            verbose=verbose,
        )
        if verbose:
            print(f"   ✓ 筛选到 {len(patient_ids)} 名患者")
        if len(patient_ids) == 0:
            if verbose:
                print("   ❌ 没有符合条件的患者")
            return pd.DataFrame()

    if verbose:
        print("📊 第2步：加载概念数据...")
    return load_concepts_fn(
        concepts=concepts,
        patient_ids=patient_ids,
        database=resolved_database,
        data_path=resolved_path,
        interval=interval,
        win_length=win_length,
        aggregate=aggregate,
        keep_components=keep_components,
        verbose=verbose,
        **kwargs,
    )


def get_cohort_comparison_impl(
    *,
    detect_database_type_fn: Callable[..., str],
    get_default_data_path_fn: Callable[..., Union[str, Path]],
    patient_ids: Optional[List[int]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    group_by: str = "survived",
    custom_groups: Optional[Dict[str, List[int]]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Return grouped cohort statistics from PatientFilter."""
    resolved_database, resolved_path = _resolve_source(
        database=database,
        data_path=data_path,
        detect_database_type_fn=detect_database_type_fn,
        get_default_data_path_fn=get_default_data_path_fn,
    )
    patient_filter = PatientFilter(
        database=resolved_database,
        data_path=resolved_path,
        verbose=verbose,
    )
    patient_filter.filter(return_dataframe=True)
    if patient_ids is not None:
        patient_filter._last_result = patient_filter._last_result[
            patient_filter._last_result["patient_id"].isin(patient_ids)
        ]
    return patient_filter.get_cohort_comparison(
        group_by=group_by,
        custom_groups=custom_groups,
    )


def get_cohort_stats_impl(
    patient_ids: List[int],
    *,
    detect_database_type_fn: Callable[..., str],
    get_default_data_path_fn: Callable[..., Union[str, Path]],
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """Return a cohort summary through the canonical statistics service."""
    resolved_database, resolved_path = _resolve_source(
        database=database,
        data_path=data_path,
        detect_database_type_fn=detect_database_type_fn,
        get_default_data_path_fn=get_default_data_path_fn,
    )
    return _get_cohort_stats(
        patient_ids,
        database=resolved_database,
        data_path=resolved_path,
    )


__all__ = [
    "PatientIdDiscoveryError",
    "filter_patients_impl",
    "get_all_patient_ids_impl",
    "get_cohort_comparison_impl",
    "get_cohort_stats_impl",
    "get_id_col_for_database",
    "get_patient_table_for_database",
    "load_concepts_filtered_impl",
]
