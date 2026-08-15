"""Stable public facade for EasyICU data APIs.

Domain implementations live in sibling modules. This facade preserves the
EasyICU 1.x import surface while keeping orchestration out of domain owners.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from ..base import (
    BaseICULoader,
    DatabaseDetectionError,
    detect_database_type,
    get_default_data_path,
)
from ..config import DATABASE_ID_CONFIG
from .cache import (
    data_path_fingerprint as _data_path_fingerprint_impl,
    get_cache_key as _get_cache_key_impl,
    load_concept_cached_impl as _load_concept_cached_impl,
)
from .compat import align_to_icu_admission
from .concepts import (
    MAX_WINDOW_EXPANSION_POINTS,
    SOFA_FIXED_CHUNK_SIZE,
    WindowExpansionError,
    _compress_dtypes as _compress_dtypes,
    _database_profile_or_default as _database_profile_or_default,
    _expand_public_numeric_win_tbl_output as _expand_public_numeric_win_tbl_output,
    _get_auto_chunk_strategy as _get_auto_chunk_strategy,
    _get_global_loader as _get_global_loader,
    _get_patient_id_source as _get_patient_id_source,
    _get_smart_workers,
    _get_total_patient_count as _get_total_patient_count,
    _guard_window_expansion as _guard_window_expansion,
    _iter_patient_id_batches as _iter_patient_id_batches,
    _normalize_patient_ids_for_db as _normalize_patient_ids_for_db,
    _sample_patient_ids as _sample_patient_ids,
    clear_global_loader,
    keep_cache,
    load_concept,
    load_concepts,
)

from .convenience import (
    load_blood_gas,
    load_demographics,
    load_hematology,
    load_lab_comprehensive,
    load_labs,
    load_neurological,
    load_outcomes,
    load_output,
    load_respiratory,
    load_sepsis3,
    load_sofa,
    load_sofa2,
    load_vitals,
    load_vitals_detailed,
)
from .convenience import load_sofa_with_score_impl as _load_sofa_with_score_impl
from .cohort import (
    PatientIdDiscoveryError,
    filter_patients_impl as _filter_patients_impl,
    get_all_patient_ids_impl as _get_all_patient_ids_impl,
    get_cohort_comparison_impl as _get_cohort_comparison_impl,
    get_cohort_stats_impl as _get_cohort_stats_impl,
    get_id_col_for_database as _get_id_col_for_database,
    get_patient_table_for_database as _get_patient_table_for_database,
    load_concepts_filtered_impl as _load_concepts_filtered_impl,
)
from .extraction import (
    EXTRACT_MODULES,
    EXTRACT_MODULE_ORDER,
    extract_all_databases,
    extract_database,
)
from .medications import (
    MedicationLoadError,
    MedicationMergeError,
    load_medications_impl as _load_medications_impl,
)
from .special_concepts import (
    _validate_concepts,
    get_concept_info,
    list_available_concepts,
    list_available_sources,
)


def load_medications(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    verbose: bool = False,
    groups: Optional[Union[str, List[str]]] = None,
    include_new: bool = True,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Load the medication domain through the stable public facade."""
    return _load_medications_impl(
        load_concepts_fn=load_concepts,
        validate_concepts_fn=_validate_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        verbose=verbose,
        groups=groups,
        include_new=include_new,
        allow_partial=allow_partial,
    )


# 为了兼容性，也导出原始的类和函数
__all__ = [
    # 主要API
    "load_concepts",  # 主API（智能默认值）
    "load_concept",  # 别名（向后兼容）
    # Easy API（便捷函数）
    "load_sofa",
    "load_sofa2",
    "load_sepsis3",
    "load_vitals",
    "load_labs",
    # 新增模块函数（参考ricu.R）
    "load_demographics",  # 基础人口统计学
    "load_outcomes",  # 结局指标
    "load_vitals_detailed",  # 详细生命体征
    "load_neurological",  # 神经系统评估
    "load_output",  # 输出量
    "load_respiratory",  # 呼吸系统
    "load_lab_comprehensive",  # 全面实验室检查
    "load_blood_gas",  # 血气分析
    "load_hematology",  # 血液学检查
    "load_medications",  # 药物治疗
    "MedicationLoadError",
    "MedicationMergeError",
    "MAX_WINDOW_EXPANSION_POINTS",
    "SOFA_FIXED_CHUNK_SIZE",
    "WindowExpansionError",
    # 工具函数
    "list_available_concepts",
    "list_available_sources",
    "get_concept_info",
    # 缓存管理
    "keep_cache",
    "clear_global_loader",
    # 增强功能（从api_enhanced.py合并）
    "load_concept_cached",
    "align_to_icu_admission",
    "load_sofa_with_score",
    # Cohort domain
    "filter_patients",
    "load_concepts_filtered",
    "get_cohort_comparison",
    "get_cohort_stats",
    "get_id_col_for_database",
    "get_patient_table_for_database",
    "get_all_patient_ids",
    "DatabaseDetectionError",
    "PatientIdDiscoveryError",
    # Full-database extraction
    "get_smart_parallel_config",
    "extract_database",
    "extract_all_databases",
    "EXTRACT_MODULES",
    "EXTRACT_MODULE_ORDER",
]


# ============================================================================
# 增强功能 - 缓存和时间对齐 (从api_enhanced.py合并)
# ============================================================================

def _get_cache_key(concepts: List[str], source: str, **kwargs) -> str:
    """Generate cache key from parameters."""
    return _get_cache_key_impl(concepts, source, **kwargs)


def _data_path_fingerprint(
    data_path: Union[str, Path],
    *,
    exclude_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Fingerprint dataset identity and file metadata for cache isolation."""
    return _data_path_fingerprint_impl(data_path, exclude_dir=exclude_dir)


def load_concept_cached(
    concepts: Union[str, List[str]],
    source: str,
    data_path: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
    patient_ids: Optional[List] = None,
    merge: bool = True,
    align_time: bool = False,
    verbose: bool = True,
    use_pickle: bool = False,
    n_patients: Optional[int] = None,
    **kwargs,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load ICU concept data with a safe Parquet cache by default.

    ``use_pickle=True`` is a trusted-local compatibility opt-in and must not be
    used with cache files supplied by another user or process boundary.
    """
    return _load_concept_cached_impl(
        concepts,
        source,
        data_path,
        get_cache_key_fn=_get_cache_key,
        data_path_fingerprint_fn=_data_path_fingerprint,
        load_concepts_fn=load_concepts,
        align_time_fn=align_to_icu_admission,
        cache_dir=cache_dir,
        force_reload=force_reload,
        patient_ids=patient_ids,
        merge=merge,
        align_time=align_time,
        verbose=verbose,
        use_pickle=use_pickle,
        n_patients=n_patients,
        **kwargs,
    )


def load_sofa_with_score(
    patient_ids: Optional[List] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: str = "1h",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Load SOFA total and component scores through the public facade."""
    return _load_sofa_with_score_impl(
        load_concepts_fn=load_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        verbose=verbose,
        **kwargs,
    )


# ==============================================================================
# 患者队列筛选 API
# ==============================================================================


def filter_patients(
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
    """Filter an ICU cohort through the cohort-domain service."""
    return _filter_patients_impl(
        detect_database_type_fn=detect_database_type,
        get_default_data_path_fn=get_default_data_path,
        database=database,
        data_path=data_path,
        age_min=age_min,
        age_max=age_max,
        first_icu_stay=first_icu_stay,
        los_min=los_min,
        los_max=los_max,
        gender=gender,
        survived=survived,
        has_sepsis=has_sepsis,
        return_dataframe=return_dataframe,
        verbose=verbose,
    )


def load_concepts_filtered(
    concepts: Union[str, List[str]],
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
    return _load_concepts_filtered_impl(
        concepts,
        filter_patients_fn=filter_patients,
        load_concepts_fn=load_concepts,
        detect_database_type_fn=detect_database_type,
        get_default_data_path_fn=get_default_data_path,
        age_min=age_min,
        age_max=age_max,
        first_icu_stay=first_icu_stay,
        los_min=los_min,
        los_max=los_max,
        gender=gender,
        survived=survived,
        has_sepsis=has_sepsis,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        aggregate=aggregate,
        keep_components=keep_components,
        verbose=verbose,
        **kwargs,
    )


def get_cohort_comparison(
    patient_ids: Optional[List[int]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    group_by: str = "survived",
    custom_groups: Optional[Dict[str, List[int]]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Return grouped cohort statistics through the cohort-domain service."""
    return _get_cohort_comparison_impl(
        detect_database_type_fn=detect_database_type,
        get_default_data_path_fn=get_default_data_path,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        group_by=group_by,
        custom_groups=custom_groups,
        verbose=verbose,
    )


def get_cohort_stats(
    patient_ids: List[int],
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """Return a cohort summary through the cohort-domain service."""
    return _get_cohort_stats_impl(
        patient_ids,
        detect_database_type_fn=detect_database_type,
        get_default_data_path_fn=get_default_data_path,
        database=database,
        data_path=data_path,
    )


# =============================================================================
# 工具函数导出 - 供 webapp 和外部使用
# =============================================================================


def get_id_col_for_database(database: str) -> str:
    """Return the configured stay-identifier column for a database."""
    return _get_id_col_for_database(
        database,
        database_id_config=DATABASE_ID_CONFIG,
    )


def get_patient_table_for_database(database: str) -> str:
    """Return the configured patient table for a database."""
    return _get_patient_table_for_database(
        database,
        database_id_config=DATABASE_ID_CONFIG,
    )


def get_all_patient_ids(
    data_path: Union[str, Path],
    database: Optional[str] = None,
    max_patients: Optional[int] = None,
) -> tuple:
    """Return patient IDs, failing closed when discovery cannot be verified."""
    return _get_all_patient_ids_impl(
        data_path,
        database_id_config=DATABASE_ID_CONFIG,
        detect_database_type_fn=detect_database_type,
        base_loader_cls=BaseICULoader,
        sample_patient_ids_fn=_sample_patient_ids,
        database=database,
        max_patients=max_patients,
    )


def get_smart_parallel_config(
    num_concepts: int = 1,
    num_patients: Optional[int] = None,
) -> tuple:
    """智能计算最佳并行配置

    根据概念数量和患者数量自动选择最优的并行策略。

    Args:
        num_concepts: 要加载的概念数量
        num_patients: 患者数量（如果已知）

    Returns:
        (concept_workers, parallel_workers): 概念并行数和患者批次并行数

    Examples:
        >>> concept_workers, parallel_workers = get_smart_parallel_config(5, 10000)
        >>> print(f"概念并行: {concept_workers}, 患者批次并行: {parallel_workers}")
    """
    return _get_smart_workers(num_concepts, num_patients)
