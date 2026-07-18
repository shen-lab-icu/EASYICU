"""
easyicu 高层API - 提供简单易用的接口，同时支持高级自定义

重构后的统一API，整合了多个模块的功能:
- api.py: 原始高层API
- api_enhanced.py: 缓存功能
- api_unified.py: 统一加载器
- load_concepts.py: 加载逻辑

两层设计:
1. Easy API - 预定义的便捷函数 (load_vitals, load_sofa等)
2. Concept API - 灵活的主API (load_concepts) 带智能默认值

使用示例:
    >>> from easyicu import load_concepts, load_sofa, load_vitals
    >>>
    >>> # 简单用法 - 自动检测数据库
    >>> hr = load_concepts('hr', patient_ids=[123, 456])
    >>>
    >>> # 完全自定义
    >>> sofa = load_concepts('sofa', patient_ids=[123, 456],
    ...                      database='miiv', data_path='/path/to/data',
    ...                      interval='6h', win_length='24h', aggregate='max')
    >>>
    >>> # Easy API - 开箱即用
    >>> vitals = load_vitals(patient_ids=[123, 456])
"""

from typing import List, Union, Optional, Dict
from pathlib import Path
import os
import numpy as np
import pandas as pd
import logging

from .base import BaseICULoader, get_default_data_path, detect_database_type
from .resources import load_dictionary, load_data_sources as load_packaged_data_sources
from .config import DATABASE_ID_CONFIG, load_data_sources as load_user_data_sources
from .databases.profiles import get_database_profile
from .concept.catalog import CONCEPT_GROUPS_INTERNAL

logger = logging.getLogger(__name__)

# 全局加载器实例，用于复用初始化开销
_global_loader = None
_loader_config = None


def _normalize_patient_ids_for_db(database_name: str, patient_ids):
    """Normalize patient IDs to the canonical ID column for each database."""
    if patient_ids is None or isinstance(patient_ids, dict):
        return patient_ids

    return {_database_profile_or_default(database_name).stay_id_col: patient_ids}


def _database_profile_or_default(database_name: str):
    """Resolve database metadata while preserving the legacy MIIV fallback."""

    try:
        return get_database_profile(database_name)
    except KeyError:
        return get_database_profile("miiv")


def _patient_filter_values(patient_ids):
    """Return concrete IDs from either public patient-filter representation."""
    if patient_ids is None:
        return None
    if isinstance(patient_ids, dict):
        if not patient_ids:
            return []
        values = next(iter(patient_ids.values()))
    else:
        values = patient_ids
    return list(values)
def _expand_public_numeric_win_tbl_output(
    result: pd.DataFrame,
    concept_name: str,
    interval: Optional[Union[str, pd.Timedelta]],
) -> pd.DataFrame:
    """Expand single-concept numeric win_tbl output to ricu-compatible rows."""
    if not isinstance(result, pd.DataFrame) or result.empty:
        return result
    if concept_name not in result.columns or 'dur_var' not in result.columns:
        return result

    numeric_values = pd.to_numeric(result[concept_name], errors='coerce')
    if numeric_values.notna().sum() == 0:
        return result

    index_candidates = [
        'charttime', 'starttime', 'start', 'datetime', 'measuredat', 'measuredat_minutes',
        'givenat', 'infusionoffset', 'observationoffset', 'labresultoffset',
    ]
    index_column = next((col for col in index_candidates if col in result.columns), None)
    if index_column is None:
        return result

    id_priority = ['stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID', 'subject_id']
    id_columns = [col for col in id_priority if col in result.columns]
    if not id_columns:
        id_columns = [col for col in result.columns if col.lower().endswith('id') and col not in {index_column, 'dur_var'}]
    if not id_columns:
        return result

    interval_td = pd.to_timedelta(interval or '1h')
    if pd.isna(interval_td) or interval_td <= pd.Timedelta(0):
        interval_td = pd.Timedelta(hours=1)

    work = result[id_columns + [index_column, 'dur_var', concept_name]].copy()
    work[concept_name] = numeric_values
    work = work.dropna(subset=[index_column, concept_name])
    if work.empty:
        return result

    expanded_rows = []
    is_datetime_index = pd.api.types.is_datetime64_any_dtype(work[index_column])

    if is_datetime_index:
        work[index_column] = pd.to_datetime(work[index_column], errors='coerce')
        # 🔧 FIX: dur_var may already be pd.Timedelta (set by ts_to_win_tbl for datetime indices).
        # pd.to_numeric on Timedelta returns nanoseconds, so pd.to_timedelta(..., unit='m')
        # would treat those nanoseconds as minutes → duration of ~114,000 years → infinite loop.
        if pd.api.types.is_timedelta64_dtype(work['dur_var']):
            duration_values = work['dur_var'].fillna(pd.Timedelta(0))
        else:
            dur_numeric = pd.to_numeric(work['dur_var'], errors='coerce').fillna(0.0)
            duration_values = pd.to_timedelta(dur_numeric, unit='m')
        epsilon = pd.Timedelta(microseconds=1)

        for row, duration in zip(work.itertuples(index=False), duration_values):
            row_dict = row._asdict()
            start = row_dict[index_column]
            if pd.isna(start):
                continue
            end = start + duration
            current = start
            while current <= end + epsilon:
                expanded_rows.append({
                    **{col: row_dict[col] for col in id_columns},
                    index_column: current,
                    concept_name: row_dict[concept_name],
                })
                current = current + interval_td
    else:
        work[index_column] = pd.to_numeric(work[index_column], errors='coerce')
        work = work.dropna(subset=[index_column])
        if work.empty:
            return result

        # 🔧 FIX: dur_var may be pd.Timedelta (set by ts_to_win_tbl for datetime indices).
        # After time alignment, the index becomes numeric (hours) but dur_var stays as Timedelta.
        # pd.to_numeric on Timedelta returns nanoseconds → wildly wrong duration → infinite loop.
        if pd.api.types.is_timedelta64_dtype(work['dur_var']):
            # Convert Timedelta to minutes (the standard ricu unit for dur_var)
            dur_numeric = work['dur_var'].dt.total_seconds().div(60.0).fillna(0.0)
        else:
            dur_numeric = pd.to_numeric(work['dur_var'], errors='coerce').fillna(0.0)
        interval_hours = interval_td.total_seconds() / 3600.0
        if interval_hours <= 0:
            interval_hours = 1.0

        # 兼容多库 win_tbl：有的 `dur_var` 是分钟（如 eICU medication），有的是小时
        # （如 HiRID grp_mount_to_rate 之后的 `givenat` + `dur_var`）。
        dur_sample = dur_numeric.loc[work.index].dropna()
        duration_is_hours = False
        if not dur_sample.empty:
            q95 = float(dur_sample.quantile(0.95))
            median = float(dur_sample.median())
            duration_is_hours = q95 <= 48.0 and median <= 24.0

        duration_hours = dur_numeric.loc[work.index] if duration_is_hours else (dur_numeric.loc[work.index] / 60.0)
        epsilon = 1e-9

        for row, duration_hour in zip(work.itertuples(index=False), duration_hours):
            row_dict = row._asdict()
            start = row_dict[index_column]
            if pd.isna(start):
                continue
            end = start + max(float(duration_hour), 0.0)
            current = float(start)
            while current <= end + epsilon:
                expanded_rows.append({
                    **{col: row_dict[col] for col in id_columns},
                    index_column: current,
                    concept_name: row_dict[concept_name],
                })
                current += interval_hours

    if not expanded_rows:
        return result

    expanded = pd.DataFrame(expanded_rows)
    expanded = (
        expanded
        .groupby(id_columns + [index_column], as_index=False)[concept_name]
        .median()
        .sort_values(id_columns + [index_column], kind='mergesort')
        .reset_index(drop=True)
    )
    return expanded


def _build_fast_scan_expr(loader: 'BaseICULoader', table_name: str) -> Optional[str]:
    """Build a DuckDB scan expression for a table without materializing it in pandas."""
    data_source = getattr(loader, 'datasource', None)
    if data_source is None or not hasattr(data_source, '_resolve_loader_from_disk'):
        return None

    source = data_source._resolve_loader_from_disk(table_name)
    if not isinstance(source, Path):
        return None

    def _escape(path: str) -> str:
        return path.replace("'", "''").replace('\\', '/')

    if source.is_dir():
        # 显式文件列表，过滤 AppleDouble (._*.parquet) — 见 datasource._enumerate_bucket_parquet_files
        try:
            from .datasource import _enumerate_bucket_parquet_files as _enum
        except Exception:
            _enum = None
        if _enum is not None:
            files = _enum(source)
            if files:
                files_sql = '[' + ', '.join(f"'{_escape(f)}'" for f in files) + ']'
                return f"read_parquet({files_sql}, union_by_name=true)"
        # Fallback: 旧的 glob 路径（仅在 helper 不可用或空目录时）
        bucket_dirs = list(source.glob('bucket_id=*'))
        if bucket_dirs:
            pattern = str(source / 'bucket_id=*' / '*.parquet').replace('\\', '/')
        else:
            parquet_files = list(source.glob('*.parquet')) + list(source.glob('*.pq'))
            if not parquet_files:
                return None
            pattern = (str(source / '*.parquet') if list(source.glob('*.parquet')) else str(source / '*.pq')).replace('\\', '/')
        return f"read_parquet('{_escape(pattern)}', union_by_name=true)"

    suffixes = [s.lower() for s in source.suffixes]
    source_str = _escape(str(source))
    if source.suffix.lower() in {'.parquet', '.pq'}:
        return f"read_parquet('{source_str}', union_by_name=true)"
    if '.csv' in suffixes or source.suffix.lower() == '.csv':
        return f"read_csv_auto('{source_str}')"
    return None


def _query_patient_ids_fast(
    loader: 'BaseICULoader',
    table_name: str,
    id_col: str,
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    sample_strategy: str = 'sorted',
) -> Optional[List]:
    """Fetch distinct patient IDs via DuckDB, avoiding full-table pandas loads."""
    scan_expr = _build_fast_scan_expr(loader, table_name)
    if not scan_expr:
        return None

    try:
        import duckdb
    except ImportError:
        return None

    order_expr = f'"{id_col}"'
    if sample_strategy == 'random':
        order_expr = f'hash("{id_col}")'

    limit_clause = f' LIMIT {int(limit)}' if limit and limit > 0 else ''
    offset_clause = f' OFFSET {int(offset)}' if offset and offset > 0 else ''
    query = (
        f'SELECT DISTINCT "{id_col}" AS patient_id '
        f'FROM {scan_expr} '
        f'WHERE "{id_col}" IS NOT NULL '
        f'ORDER BY {order_expr}{limit_clause}{offset_clause}'
    )

    conn = duckdb.connect()
    try:
        conn.execute("SET timezone='UTC'")
        conn.execute("SET enable_progress_bar = false")
        conn.execute("SET enable_progress_bar_print = false")
        conn.execute("SET memory_limit = '2GB'")
        return conn.execute(query).fetchnumpy()['patient_id'].tolist()
    finally:
        conn.close()


def _count_patient_ids_fast(loader: 'BaseICULoader', table_name: str, id_col: str) -> Optional[int]:
    """Count distinct patient IDs via DuckDB without loading the ID table into pandas."""
    scan_expr = _build_fast_scan_expr(loader, table_name)
    if not scan_expr:
        return None

    try:
        import duckdb
    except ImportError:
        return None

    query = (
        f'SELECT COUNT(DISTINCT "{id_col}") AS n '
        f'FROM {scan_expr} '
        f'WHERE "{id_col}" IS NOT NULL'
    )

    conn = duckdb.connect()
    try:
        conn.execute("SET timezone='UTC'")
        conn.execute("SET enable_progress_bar = false")
        conn.execute("SET enable_progress_bar_print = false")
        conn.execute("SET memory_limit = '2GB'")
        result = conn.execute(query).fetchone()
        return int(result[0]) if result and result[0] is not None else None
    finally:
        conn.close()

def clear_global_loader():
    """清除全局加载器，强制下一次调用重新创建"""
    global _global_loader, _loader_config
    if _global_loader is not None:
        if hasattr(_global_loader, 'clear_cache'):
            _global_loader.clear_cache()
        else:
            # 清理加载器内部缓存
            if hasattr(_global_loader, 'concept_resolver'):
                _global_loader.concept_resolver.clear()
            for attr in ('datasource', 'data_source'):
                data_source = getattr(_global_loader, attr, None)
                if data_source is not None and hasattr(data_source, 'clear'):
                    data_source.clear()
    _global_loader = None
    _loader_config = None


from contextlib import contextmanager

@contextmanager
def keep_cache(database=None, data_path=None, dict_path=None, use_sofa2=False, verbose=False):
    """Context manager: keep raw/table cache between sequential load_concepts calls.
    
    Usage::
    
        with keep_cache(database='miiv'):
            df1 = load_concepts(['hr', 'sbp'], database='miiv', max_patients=1000)
            df2 = load_concepts(['sofa'], database='miiv', max_patients=1000)
            # sofa reuses cached hr/sbp/map/etc. from df1's sub-concept loads
    """
    loader = _get_global_loader(database=database, data_path=data_path,
                                dict_path=dict_path, use_sofa2=use_sofa2, verbose=verbose)
    resolver = loader.concept_resolver
    resolver._keep_cache_between_calls = True
    try:
        yield loader
    finally:
        resolver._keep_cache_between_calls = False
        if hasattr(resolver, 'drop_source_caches'):
            resolver.drop_source_caches()
        else:  # pragma: no cover - legacy resolver without cache accounting
            with resolver._cache_lock:
                resolver._raw_concept_cache.clear()
                resolver._table_cache.clear()

import numpy as np

def _sample_patient_ids(loader: 'BaseICULoader', max_patients: int, verbose: bool = False,
                        sample_strategy: str = 'sorted') -> List:
    """
    从数据库中采样患者ID（用于 max_patients 参数）
    
    根据数据库类型，从对应的住院/ICU表中获取患者ID。
    
    Args:
        loader: BaseICULoader 实例
        max_patients: 最大患者数量
        verbose: 是否输出调试信息
        sample_strategy: 采样策略（仅在取「子集」即 max_patients < 全库时影响代表性；
            全量加载时两者只是排序差异、不影响覆盖）
            - 'random': seeded 随机采样 N 个（默认，seed=42 可复现）。多中心库(eICU 等)
              患者 id 按医院/批次聚簇，sorted 前缀会得到非代表性子群、使有覆盖的概念
              在子集里假性为空，故默认用随机保证代表性。
            - 'sorted': 按 ID 排序取前 N 个。用于与 ricu 金标准 fixture 对齐(parity/
              fixture 生成时显式传 'sorted')。
    """
    profile = _database_profile_or_default(loader.database)
    table_name, id_col = profile.stay_table, profile.stay_id_col
    
    try:
        fast_ids = _query_patient_ids_fast(
            loader,
            table_name,
            id_col,
            limit=max_patients if sample_strategy != 'random' else max_patients,
            sample_strategy=sample_strategy,
        )
        if fast_ids is not None:
            if sample_strategy == 'random':
                sampled_ids = sorted(fast_ids[:max_patients])
                strategy_label = "随机采样"
            else:
                sampled_ids = fast_ids[:max_patients]
                strategy_label = "已排序"

            if verbose:
                print(f"🎯 max_patients={max_patients}: DuckDB 快速采样 {len(sampled_ids)} 个患者 ({strategy_label})")
            return sampled_ids

        # 只加载ID列，限制行数
        id_table = loader.datasource.load_table(table_name, columns=[id_col], verbose=False)
        all_ids = id_table.data[id_col].dropna().unique()
        
        if sample_strategy == 'random' and len(all_ids) > max_patients:
            import numpy as np
            rng = np.random.default_rng(seed=42)  # 固定种子保证可复现
            sampled_ids = sorted(rng.choice(all_ids, size=max_patients, replace=False).tolist())
            strategy_label = "随机采样"
        else:
            # 🔧 按ID排序后再采样，确保与 RICU 金标准生成脚本一致
            all_ids = sorted(all_ids)
            sampled_ids = list(all_ids[:max_patients])
            strategy_label = "已排序"
        
        if verbose:
            print(f"🎯 max_patients={max_patients}: 从 {table_name}.{id_col} 采样 {len(sampled_ids)} 个患者 ({strategy_label})")
        
        return sampled_ids
    except Exception as e:
        if verbose:
            print(f"⚠️ 采样患者ID失败: {e}，将加载所有患者")
        return None


def _get_patient_id_source(loader: 'BaseICULoader') -> tuple[str, str]:
    """Return the canonical (table_name, id_col) pair for a database."""
    profile = _database_profile_or_default(loader.database)
    return profile.stay_table, profile.stay_id_col


def _iter_patient_id_batches(
    loader: 'BaseICULoader',
    batch_size: int,
    *,
    total_patients: Optional[int] = None,
    sample_strategy: str = 'sorted',
):
    """Yield patient-id batches directly from storage without materializing the full ID list."""
    table_name, id_col = _get_patient_id_source(loader)
    remaining = total_patients
    offset = 0

    while remaining is None or remaining > 0:
        limit = batch_size if remaining is None else min(batch_size, remaining)
        batch_ids = _query_patient_ids_fast(
            loader,
            table_name,
            id_col,
            limit=limit,
            offset=offset,
            sample_strategy=sample_strategy,
        )

        if batch_ids is None:
            all_ids = _sample_patient_ids(loader, total_patients or 999999999, verbose=False, sample_strategy=sample_strategy)
            if not all_ids:
                return
            for start in range(0, len(all_ids), batch_size):
                yield {id_col: all_ids[start:start + batch_size]}
            return

        if not batch_ids:
            return

        yield {id_col: batch_ids}
        offset += len(batch_ids)
        if remaining is not None:
            remaining -= len(batch_ids)


def _get_total_patient_count(loader: 'BaseICULoader') -> Optional[int]:
    """
    快速获取数据库中的总患者数（用于自动分批决策）。
    使用 DuckDB COUNT(DISTINCT) 避免加载全部数据。
    """
    profile = _database_profile_or_default(loader.database)
    table_name, id_col = profile.stay_table, profile.stay_id_col
    
    try:
        fast_count = _count_patient_ids_fast(loader, table_name, id_col)
        if fast_count is not None:
            return fast_count
        id_table = loader.datasource.load_table(table_name, columns=[id_col], verbose=False)
        return id_table.data[id_col].nunique()
    except Exception:
        return None


def _compress_dtypes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    压缩 DataFrame 的数据类型以减少内存使用
    
    - int64 -> int32 (如果值范围允许)
    - float64 -> float32 (对于非精确值)
    - 保持 datetime64 不变
    
    可以节省约 50-60% 的内存
    """
    if hasattr(df, 'data') and isinstance(getattr(df, 'data'), pd.DataFrame):
        df.data = _compress_dtypes(df.data, verbose=verbose)
        return df

    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    
    original_mem = df.memory_usage(deep=True).sum()
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # 整数类型压缩
        if col_type == np.int64:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
        
        # 浮点类型压缩 - SOFA 分数等小整数可以用 int8
        elif col_type == np.float64:
            # 检查是否都是整数值
            if df[col].dropna().apply(lambda x: x == int(x)).all():
                col_min, col_max = df[col].min(), df[col].max()
                if not np.isnan(col_min) and col_min >= -128 and col_max <= 127:
                    # 小整数用 Int8 (可空整数)
                    df[col] = df[col].astype('Int8')
                elif not np.isnan(col_min) and col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype('Int16')
            else:
                # 一般浮点数用 float32
                df[col] = df[col].astype(np.float32)
    
    if verbose:
        new_mem = df.memory_usage(deep=True).sum()
        saved = (original_mem - new_mem) / original_mem * 100
        print(f"💾 内存压缩: {original_mem/1024/1024:.1f}MB → {new_mem/1024/1024:.1f}MB (节省 {saved:.0f}%)")
    
    return df


def _get_global_loader(
    database: Optional[str] = None,
    data_path: Optional[Path] = None,
    dict_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    **kwargs,
) -> BaseICULoader:
    """获取或创建全局加载器实例（减少重复初始化）"""
    global _global_loader, _loader_config

    if dict_path is None:
        dict_key = None
    elif isinstance(dict_path, (list, tuple)):
        dict_key = tuple(map(str, dict_path))
    else:
        dict_key = str(dict_path)

    # 🚀 只比较影响加载器初始化的关键参数，忽略运行时参数（如 verbose）
    # 这允许在多次调用之间复用加载器，共享缓存
    config_kwargs = {k: v for k, v in kwargs.items() if k in ('use_sofa2',)}
    current_config = (database, str(data_path) if data_path else None, dict_key, frozenset(config_kwargs.items()))

    if _global_loader is None or _loader_config != current_config:
        _global_loader = BaseICULoader(
            database=database,
            data_path=data_path,
            dict_path=dict_path,
            **kwargs,
        )
        _loader_config = current_config

    return _global_loader

def _get_smart_workers(num_concepts: int, num_patients: Optional[int] = None) -> tuple:
    """
    智能计算最佳并行配置
    
    使用 parallel_config 模块根据系统资源自动调整。
    
    Args:
        num_concepts: 要加载的概念数量
        num_patients: 患者数量（如果已知）
    
    Returns:
        (concept_workers, parallel_workers): 概念并行数和患者批次并行数
    """
    # 检查是否禁用自动优化
    if os.getenv('EASYICU_NO_AUTO_PARALLEL'):
        return 1, None
    
    from .runtime.parallel_config import get_global_config, get_runtime_load_strategy

    strategy = get_runtime_load_strategy(
        [f"concept_{i}" for i in range(num_concepts)],
        num_patients=num_patients,
        config=get_global_config(),
    )
    concept_workers = int(strategy["concept_workers"])
    parallel_workers = int(strategy["parallel_workers"])
    return concept_workers, (parallel_workers if parallel_workers > 1 else None)


def _is_low_memory_chunk_candidate(
    concepts_list: List[str],
    *,
    merge: bool,
    chunk_size: Optional[int],
    batch_size: Optional[int],
) -> bool:
    """Whether the request should prefer the validated low-memory chunk path."""
    if os.getenv('EASYICU_DISABLE_AUTO_CHUNK'):
        return False
    if not merge:
        return False
    if chunk_size is not None or batch_size is not None:
        return False

    normalized = {str(name).lower() for name in concepts_list}
    heavy_concepts = {
        'sofa',
        'sofa2',
        'kdigo_aki',
        'aki',
        'sep3',
        'sep3_sofa2',
    }
    return bool(normalized.intersection(heavy_concepts))


def _get_auto_chunk_strategy(
    concepts_list: List[str],
    num_patients: Optional[int],
    *,
    merge: bool,
    chunk_size: Optional[int],
    batch_size: Optional[int],
    parallel_workers: Optional[int],
    concept_workers: Optional[int],
) -> Optional[Dict[str, int]]:
    """Return an auto-tuned chunk strategy for heavy large-scale extraction.

    The policy balances throughput and memory while preserving deterministic
    score-window expansion for clinical scores. SOFA-family concepts currently
    stay on the validated 2000-stay chunk profile even when more memory is
    available, because larger chunks can change large-cohort window expansion
    results.
    """
    if not _is_low_memory_chunk_candidate(
        concepts_list,
        merge=merge,
        chunk_size=chunk_size,
        batch_size=batch_size,
    ):
        return None
    if num_patients is None or num_patients < 2000:
        return None

    from .runtime.parallel_config import get_global_config, get_runtime_load_strategy
    from .runtime.memory_manager import get_available_memory_mb

    config = get_global_config()
    available_memory_mb = get_available_memory_mb()
    normalized = {str(name).lower() for name in concepts_list}

    sepsis_heavy_concepts = {'sep3', 'sep3_sofa2'}
    renal_heavy_concepts = {'kdigo_aki', 'aki'}
    sofa_heavy_concepts = {'sofa', 'sofa2'}

    if 'EASYICU_AUTO_CHUNK_SIZE' in os.environ:
        auto_chunk_size = max(250, int(os.getenv('EASYICU_AUTO_CHUNK_SIZE', '1000')))
        if normalized.intersection(sofa_heavy_concepts) and auto_chunk_size > 2000:
            logger.warning(
                "Capping SOFA auto chunk size at 2000 because larger chunks can "
                "change SOFA window expansion results in current large-cohort mode."
            )
            auto_chunk_size = 2000
    elif normalized.intersection(sepsis_heavy_concepts):
        # 复合 sepsis 链路更依赖批次数带来的并行度；过大 chunk 会明显拖慢速度
        auto_chunk_size = 1000
    elif normalized.intersection(renal_heavy_concepts):
        if available_memory_mb >= 10 * 1024:
            auto_chunk_size = 4000
        elif available_memory_mb >= 6 * 1024:
            auto_chunk_size = 2000
        else:
            auto_chunk_size = 1000
    elif normalized.intersection(sofa_heavy_concepts):
        if available_memory_mb >= 10 * 1024:
            auto_chunk_size = 2000
        elif available_memory_mb >= 6 * 1024:
            auto_chunk_size = 2000
        elif available_memory_mb >= 3 * 1024:
            auto_chunk_size = 2000
        else:
            auto_chunk_size = 1000
    elif available_memory_mb >= 10 * 1024:
        auto_chunk_size = 8000
    elif available_memory_mb >= 6 * 1024:
        auto_chunk_size = 4000
    elif available_memory_mb >= 3 * 1024:
        auto_chunk_size = 2000
    else:
        auto_chunk_size = 1000

    if num_patients is not None:
        auto_chunk_size = min(auto_chunk_size, max(250, int(num_patients)))

    batches = max(1, (num_patients + auto_chunk_size - 1) // auto_chunk_size)
    runtime_strategy = get_runtime_load_strategy(
        concepts_list,
        num_patients=num_patients,
        chunk_size=auto_chunk_size,
        requested_concept_workers=concept_workers,
        requested_parallel_workers=parallel_workers,
        config=config,
    )
    tuned_parallel_workers = min(int(runtime_strategy['parallel_workers']), batches)
    tuned_concept_workers = int(runtime_strategy['concept_workers'])

    return {
        'chunk_size': auto_chunk_size,
        'parallel_workers': max(1, tuned_parallel_workers),
        'concept_workers': max(1, tuned_concept_workers),
    }


# SOFA2 相关概念集合（需要加载 sofa2-dict）。load_concepts 的自动检测和
# extract_database 的分组 worker 共用这一份定义，保证两边判定一致。
_SOFA2_TRIGGER_CONCEPTS = frozenset({
    'sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver',
    'sofa2_cardio', 'sofa2_cns', 'sofa2_renal',
    'uo_6h', 'uo_12h', 'uo_24h', 'rrt_criteria', 'rrt',
    'adv_resp', 'ecmo', 'ecmo_indication', 'sedated_gcs',
    'mech_circ_support', 'other_vaso', 'delirium_tx',
    'motor_response', 'delirium_positive',
})


def _concepts_need_sofa2(concepts) -> bool:
    """True when any concept requires the sofa2-dict overlay."""
    return any(
        c in _SOFA2_TRIGGER_CONCEPTS or 'sofa2' in str(c).lower()
        for c in concepts
    )


def load_concepts(
    concepts: Union[str, List[str]],
    patient_ids: Optional[Union[List, Dict]] = None,
    # 数据源参数 - 智能默认值
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    # 时间参数 - 默认与ricu一致 (interval=hours(1L))
    interval: Optional[Union[str, pd.Timedelta]] = '1h',  # ricu默认: hours(1L)
    win_length: Optional[Union[str, pd.Timedelta]] = None,
    # 聚合参数
    aggregate: Optional[Union[str, Dict]] = None,
    # SOFA相关
    keep_components: bool = False,
    # 其他
    verbose: bool = False,
    use_sofa2: bool = False,  # 新增：是否使用SOFA2字典
    merge: bool = True,       # 新增：是否合并结果
    r_compatible: bool = True,  # 默认启用ricu.R兼容格式
    dict_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    chunk_size: Optional[int] = None,
    progress: bool = False,
    parallel_workers: Optional[int] = None,
    concept_workers: Optional[int] = None,  # 改为Optional，支持自动检测
    parallel_backend: str = 'auto',
    max_patients: Optional[int] = None,  # 限制加载的患者数量（自动采样）
    limit: Optional[int] = None,  # max_patients 的别名（兼容 extract_sofa_data.py）
    sample_strategy: str = 'random',  # 采样策略: 'random'=seeded 随机(默认,代表性);'sorted'=按ID排序前N个(ricu-parity 用)
    batch_size: Optional[int] = None,  # 🆕 分批处理大小（默认30000，适合12GB内存）
    memory_efficient: bool = False,  # 🆕 内存优化模式（压缩数据类型）
    require_bounded_sample: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    加载ICU概念数据 - easyicu的主要API (重构版本)

    这个函数使用统一的BaseICULoader，整合了多个模块的功能：
    - 原api.py的所有功能
    - api_enhanced.py的缓存支持
    - api_unified.py的统一逻辑
    - load_concepts.py的加载实现

    Args:
        concepts: 概念名称或概念名称列表
            例如: 'hr', ['hr', 'sbp', 'temp'], 'sofa', 'sofa2'
        patient_ids: 可选的患者ID列表或字典
            - List: [123, 456] (自动转换为正确的ID列)
            - Dict: {'stay_id': [123, 456]} (显式指定ID列)
            - None: 加载所有患者

        # === 数据源参数 (可选，有智能默认值) ===
        database: 数据库类型
            - None: 自动检测（从环境变量）
            - 'miiv', 'mimic', 'eicu', 'hirid', 'aumc'
        data_path: 数据路径
            - None: 从环境变量或常见路径自动查找
            - str/Path: 显式指定路径

        # === 时间参数 (默认与ricu一致) ===
        interval: 时间对齐间隔 (默认'1h'，与ricu的hours(1L)一致)
            - '1h': 默认值，与ricu R包一致
            - '6h', '12h': 其他时间间隔
            - None: 使用原始时间点（不对齐）
            - pd.Timedelta(hours=1): Timedelta对象
        win_length: 滑动窗口长度（用于SOFA等评分）
            - None: 点数据（不使用窗口）
            - '24h': 字符串格式
            - pd.Timedelta(hours=24): Timedelta对象

        # === 聚合参数 (可选) ===
        aggregate: 聚合方式
            - None: 使用默认聚合（通常是'mean'）
            - 'mean', 'max', 'min', 'median': 单一聚合函数
            - {'hr': 'mean', 'sbp': 'max'}: 每个概念指定聚合

        # === SOFA相关 ===
        keep_components: 是否保留SOFA组件列
            - False: 只返回总分
            - True: 返回 sofa + sofa_resp + sofa_coag + ...
        use_sofa2: 是否加载SOFA2字典（自动检测SOFA2概念时启用）

        # === 其他 ===
        merge: 是否合并多个概念到一个DataFrame
        verbose: 是否显示详细信息
        max_patients: 自动采样的患者上限
        require_bounded_sample: 若为 True，则在 max_patients 采样失败时立即报错，
            不允许回退到全库加载。适用于 Web 等必须硬性有界的调用路径。
        sample_strategy: 取子集时的采样策略。默认 'random'(seeded, 可复现, 代表性);
            ricu fixture parity 需显式传 'sorted'。见 _sample_patient_ids 文档。
        limit: max_patients 的别名
        n_patients: max_patients 的兼容别名（可通过 kwargs 传入）
        **kwargs: 其他参数传递给底层API

    Returns:
        DataFrame 或 dict of DataFrames

    Examples:
        >>> # 最简单的用法 - 自动检测所有参数
        >>> hr = load_concepts('hr')
        >>>
        >>> # 指定患者ID
        >>> hr = load_concepts('hr', patient_ids=[123, 456, 789])
        >>>
        >>> # 加载多个概念并对齐到1小时间隔
        >>> vitals = load_concepts(['hr', 'sbp', 'temp'],
        ...                        patient_ids=[123, 456],
        ...                        interval='1h')
        >>>
        >>> # SOFA评分 - 24小时窗口，保留组件
        >>> sofa = load_concepts('sofa',
        ...                      patient_ids=[123, 456],
        ...                      interval='6h',
        ...                      win_length='24h',
        ...                      keep_components=True)
        >>>
        >>> # SOFA2评分 (2025标准)
        >>> sofa2 = load_concepts('sofa2',
        ...                       patient_ids=[123, 456],
        ...                       use_sofa2=True)
        >>>
        >>> # 完全自定义
        >>> data = load_concepts('sofa2',
        ...                      patient_ids={'stay_id': [123, 456]},
        ...                      database='miiv',
        ...                      data_path='/custom/path',
        ...                      interval=pd.Timedelta(hours=6),
        ...                      win_length=pd.Timedelta(hours=24),
        ...                      aggregate='max',
        ...                      verbose=True)
    """
    # 自动检测SOFA2需求
    if isinstance(concepts, str):
        concepts_list = [concepts]
    else:
        concepts_list = list(concepts)

    # SOFA2 相关概念集合（需要加载 sofa2-dict）— 共享定义见 _SOFA2_TRIGGER_CONCEPTS
    requested_concepts = list(concepts_list)
    if _concepts_need_sofa2(concepts_list):
        use_sofa2 = True

    # 2026-05-20: SPECIAL_CONCEPTS dispatch — these concepts are NOT in
    # concept-dict.json. The webapp routes them through dedicated loader
    # functions (`kdigo_aki.load_kdigo_aki`, `circ_failure.load_circ_failure`).
    # Previously `load_concepts(['aki'])` would raise
    #   KeyError: "Concept 'aki' not present in dictionary"
    # forcing API users to know about the side-channel modules. Detect
    # them up front, peel them off, run the standard path on the rest,
    # then re-attach the special results.
    _KDIGO_OUTPUTS = {'aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo',
                      'aki_stage_rrt', 'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
                      'creat_low_past_48hr', 'creat_low_past_7day'}
    _CIRC_OUTPUTS = {'circ_failure', 'circ_event'}
    # Comorbidity indices live in comorbidity.py (ICD code-set matching over
    # the diagnosis table), not concept-dict.json — route like kdigo/circ.
    _COMORB_OUTPUTS = {'charlson', 'elixhauser'}
    # Composite outcome endpoints (outcomes.py) — fixed-horizon mortality etc.
    _OUTCOME_OUTPUTS = {'mort_28d', 'mort_90d', 'mort_365d',
                        'icu_free_days_28', 'icu_readmission',
                        'vent_free_days_28'}
    # Microbiology culture-positivity (microbiology.py).
    _MICRO_OUTPUTS = {'culture_positive', 'bld_culture_positive'}
    _requested = set(concepts_list)
    _need_kdigo = _requested & _KDIGO_OUTPUTS
    _need_circ = _requested & _CIRC_OUTPUTS
    _need_comorb = _requested & _COMORB_OUTPUTS
    _need_outcome = _requested & _OUTCOME_OUTPUTS
    _need_micro = _requested & _MICRO_OUTPUTS
    _special = (_need_kdigo | _need_circ | _need_comorb
                | _need_outcome | _need_micro)
    # Keep the FULL requested list (incl. special concepts) for the batched path.
    # When batching triggers, the code returns before the special re-attach (~L1480),
    # so the batch loader must receive the specials and re-run their loaders per batch
    # (each patient-id batch carries complete per-patient histories -> baselines compute).
    _concepts_all = list(concepts_list)
    if _special:
        # Pull special concepts out of the list passed to the standard resolver.
        concepts_list = [c for c in concepts_list if c not in _special]

    # 防御性检查: 检测常见的位置参数误用 (load_concepts(['hr'], 'miiv') 应为 database='miiv')
    if isinstance(patient_ids, str):
        try:
            positional_database = get_database_profile(patient_ids)
        except KeyError:
            positional_database = None
        if positional_database is not None:
            if database is None:
                database = positional_database.key
                patient_ids = None
            else:
                raise TypeError(
                    f"patient_ids 收到字符串 '{patient_ids}'，看起来是数据库名。"
                    f"请使用关键字参数: load_concepts(concepts, database='{patient_ids}')")

    if patient_ids is None:
        id_kwargs = [
            "patientunitstayid",
            "admissionid",
            "stay_id",
            "subject_id",
            "patientid",
        ]
        for id_key in id_kwargs:
            if id_key in kwargs:
                patient_ids = {id_key: kwargs.pop(id_key)}
                break

    # An explicitly empty cohort means "no patients", never "all patients".
    # Return before constructing a loader so every downstream fast path is
    # fail-closed, including dedicated outcome/comorbidity loaders.
    if patient_ids is not None and len(_patient_filter_values(patient_ids)) == 0:
        if merge:
            return pd.DataFrame()
        return {name: pd.DataFrame() for name in requested_concepts}
    if verbose:
        print(f"📊 使用统一API加载 {len(concepts_list)} 个概念...")
        print(f"   概念: {', '.join(concepts_list)}")

    # 创建或获取全局加载器
    loader = _get_global_loader(
        database=database,
        data_path=data_path,
        dict_path=dict_path,
        use_sofa2=use_sofa2,
        verbose=verbose
    )

    # 🚀 从 kwargs 中提取患者 ID（支持通过 patientunitstayid=, admissionid=, stay_id= 等传入）

    # 🚀 处理患者数量别名（兼容旧测试/benchmark）
    n_patients_alias = kwargs.pop('n_patients', None)
    if (
        n_patients_alias is not None
        and max_patients is not None
        and int(n_patients_alias) != int(max_patients)
    ):
        raise ValueError(
            f"收到冲突的患者上限参数: n_patients={n_patients_alias}, "
            f"max_patients={max_patients}"
        )
    if (
        n_patients_alias is not None
        and limit is not None
        and int(n_patients_alias) != int(limit)
        and max_patients is None
    ):
        raise ValueError(
            f"收到冲突的患者上限参数: n_patients={n_patients_alias}, "
            f"limit={limit}"
        )

    # 🚀 处理 limit 别名（兼容性）
    effective_max_patients = max_patients
    if effective_max_patients is None and limit is not None:
        effective_max_patients = limit
    if effective_max_patients is None and n_patients_alias is not None:
        effective_max_patients = int(n_patients_alias)

    # 🚀 max_patients 支持：自动从数据库采样患者ID
    if effective_max_patients == 0 and patient_ids is None:
        if merge:
            return pd.DataFrame()
        return {name: pd.DataFrame() for name in requested_concepts}
    if effective_max_patients is not None and patient_ids is None:
        patient_ids = _sample_patient_ids(loader, effective_max_patients, verbose,
                                          sample_strategy=sample_strategy)
        if require_bounded_sample and (
            patient_ids is None or len(patient_ids) == 0
        ):
            raise RuntimeError(
                "Unable to build the required bounded patient sample; refusing "
                "to fall back to an unbounded database load."
            )

    # 规范化患者ID
    if patient_ids is not None and not isinstance(patient_ids, dict):
        patient_ids = _normalize_patient_ids_for_db(loader.database, patient_ids)

    # 🚀 智能并行配置：根据概念数量和患者数量自动优化
    special_patient_ids = _patient_filter_values(patient_ids)
    num_patients = None
    if patient_ids is not None:
        if isinstance(patient_ids, dict):
            for v in patient_ids.values():
                if isinstance(v, (list, tuple)):
                    num_patients = len(v)
                    break
        elif isinstance(patient_ids, (list, tuple)):
            num_patients = len(patient_ids)

    prefer_low_memory_chunk = _is_low_memory_chunk_candidate(
        concepts_list,
        merge=merge,
        chunk_size=chunk_size,
        batch_size=batch_size,
    )
    inferred_total_patients = num_patients
    if inferred_total_patients is None and patient_ids is None and prefer_low_memory_chunk:
        try:
            inferred_total_patients = _get_total_patient_count(loader)
        except Exception as e:
            logger.debug(f"低内存 chunk 总患者数检测失败: {e}")
    
    # 只有当用户没有指定时才使用智能配置
    effective_concept_workers = concept_workers
    effective_parallel_workers = parallel_workers
    
    if concept_workers is None or parallel_workers is None:
        from .runtime.parallel_config import get_global_config, get_runtime_load_strategy

        runtime_strategy = get_runtime_load_strategy(
            concepts_list,
            num_patients=inferred_total_patients,
            chunk_size=chunk_size,
            requested_concept_workers=concept_workers,
            requested_parallel_workers=parallel_workers,
            requested_backend=parallel_backend if parallel_backend != 'auto' else None,
            config=get_global_config(),
        )
        if concept_workers is None:
            effective_concept_workers = int(runtime_strategy['concept_workers'])
        if parallel_workers is None:
            auto_parallel = int(runtime_strategy['parallel_workers'])
            effective_parallel_workers = auto_parallel if auto_parallel > 1 else None
        
        if verbose and (effective_concept_workers > 1 or effective_parallel_workers):
            print(f"   ⚡ 智能优化: concept_workers={effective_concept_workers}, "
                  f"parallel_workers={effective_parallel_workers or '不分批'}")

    effective_chunk_size = chunk_size
    auto_chunk_strategy = _get_auto_chunk_strategy(
        concepts_list,
        inferred_total_patients,
        merge=merge,
        chunk_size=chunk_size,
        batch_size=batch_size,
        parallel_workers=parallel_workers,
        concept_workers=concept_workers,
    )
    if auto_chunk_strategy:
        effective_chunk_size = auto_chunk_strategy['chunk_size']
        effective_parallel_workers = auto_chunk_strategy['parallel_workers']
        effective_concept_workers = auto_chunk_strategy['concept_workers']
        if verbose:
            print(
                f"   🚀 大样本复合概念优先使用平衡分块: chunk_size={effective_chunk_size}, "
                f"parallel_workers={effective_parallel_workers}, "
                f"concept_workers={effective_concept_workers}"
            )

        if patient_ids is None and inferred_total_patients:
            patient_ids = _sample_patient_ids(
                loader,
                inferred_total_patients,
                verbose=verbose,
                sample_strategy='sorted',
            )
            if patient_ids is not None and not isinstance(patient_ids, dict):
                patient_ids = _normalize_patient_ids_for_db(loader.database, patient_ids)
            if isinstance(patient_ids, dict):
                for v in patient_ids.values():
                    if isinstance(v, (list, tuple)):
                        num_patients = len(v)
                        break
            elif isinstance(patient_ids, (list, tuple)):
                num_patients = len(patient_ids)
            else:
                num_patients = inferred_total_patients

            if verbose and patient_ids is not None:
                print(f"   📦 已为平衡分块准备全量患者ID: {num_patients} patients")

    # ====================================================================
    # 🆕 内存感知自动分批处理
    # ====================================================================
    # 策略：
    # 1. 用户指定了 batch_size → 使用用户指定值
    # 2. patient_ids=None（全量加载）→ 自动检测总患者数并估算内存
    # 3. 内存充足 → 不分批直接加载
    # 4. 内存不足 → 自动计算 batch_size 并分批
    # 5. 可用内存 < 16GB → 使用子进程隔离（内存完全归还）
    # 6. 可用内存 >= 16GB → 进程内分批 + malloc_trim（更快）
    # ====================================================================
    
    from .runtime.memory_manager import (
        auto_batch_size, estimate_memory_mb,
        get_available_memory_mb, inprocess_batch_load,
        inprocess_batch_load_streaming, subprocess_batch_load,
    )
    
    effective_batch_size = batch_size
    use_subprocess = False
    use_streaming_patient_batches = False
    
    # 提取患者ID信息
    _id_col = None
    _all_ids = None
    if patient_ids is not None and isinstance(patient_ids, dict):
        _id_col = list(patient_ids.keys())[0]
        _all_ids = list(patient_ids.values())[0]
        _total_patients = len(_all_ids)
    elif patient_ids is not None and isinstance(patient_ids, (list, tuple)):
        _total_patients = len(patient_ids)
    else:
        _total_patients = None
    
    # 🔧 2026-05-11: 默认不分批，追求合理内存下的最优速度。
    # 实测（MIMIC-IV 94k 患者 167 特征）：单模块 peak ≤ 8 GB（vitals 最高 ~5GB，
    # sofa/sep3 系列 peak 几乎不随 N 增长——DuckDB 内部 hash 工作集主导）。
    # 触发自动分批的条件（同时满足）：
    #   1. 可用内存 < 6 GB（低内存系统才需要保守路径）
    #   2. 估算峰值 > 可用内存（否则一次跑完最快）
    # 12 GB+ 系统：默认全跑，速度最优。
    LOW_MEM_THRESHOLD_MB = 6 * 1024
    
    # 自动检测全量加载场景
    if (not auto_chunk_strategy) and _total_patients is None and patient_ids is None and effective_batch_size is None:
        # 全量加载：查询总患者数来决定是否需要分批
        try:
            _total_patients_in_db = _get_total_patient_count(loader)
            if _total_patients_in_db and _total_patients_in_db > 1000:
                # 估算内存需求
                est_mem = estimate_memory_mb(concepts_list, loader.database, _total_patients_in_db)
                avail_mem = get_available_memory_mb()

                # 🚀 稳定预算分批判定（16GB 可用性 + 确定性）：用**物理总内存**这个稳定值
                # 判定是否分批，而不是波动的"当前可用"。否则 (a) 空闲机(avail 大)时宽模块
                # 一次性→放不下→静默返回空概念(截断)，(b) 繁忙机(avail 小)时连窄模块也被
                # 误判分批→过度分批变慢。只有估算峰值 > 0.6×总内存的模块才分批，并用同一
                # 稳定预算反算 batch_size(大批少次)。实测 16GB 上只有 49-概念宽模块
                # (medications/chemistry)在大队列分批，其余 17 个模块保持一次性(最快)。
                # EASYICU_ONESHOT_BUDGET_MB 可覆盖每模块一次性内存上限(MB)。
                try:
                    _env_b = os.environ.get('EASYICU_ONESHOT_BUDGET_MB')
                    if _env_b:
                        _oneshot_budget_mb = float(_env_b)
                    else:
                        import psutil as _psb
                        _oneshot_budget_mb = (_psb.virtual_memory().total / (1024 * 1024)) * 0.6
                except Exception:
                    _oneshot_budget_mb = 9830.0  # 16GB*0.6 回退
                if est_mem > _oneshot_budget_mb:
                    _total_patients = _total_patients_in_db
                    effective_batch_size = auto_batch_size(
                        concepts_list, loader.database, _total_patients,
                        available_memory_mb=_oneshot_budget_mb / 0.6,
                    )

                    if verbose and effective_batch_size:
                        print(f"⚠️  稳定预算分批 (估算 {est_mem:.0f}MB > 预算 {_oneshot_budget_mb:.0f}MB), "
                              f"全量加载 {_total_patients} patients 分批 (batch_size={effective_batch_size})")

                    # 分批时仍用子进程隔离；进程内路径优先走流式 patient batch
                    use_subprocess = True
                    use_streaming_patient_batches = effective_batch_size is not None
                elif verbose:
                    print(f"🚀 全量加载 {_total_patients_in_db} patients, "
                          f"估算 {est_mem:.0f}MB ≤ 预算 {_oneshot_budget_mb:.0f}MB, 不分批（最优速度）")
        except Exception as e:
            logger.debug(f"自动分批检测失败: {e}")
    
    # 用户显式指定了 batch_size
    if effective_batch_size is None and batch_size is not None:
        effective_batch_size = batch_size
    
    # 🔧 FIX: 当 batch_size 已指定但 _id_col/_all_ids 未设置时（patient_ids=None 全量加载），
    # 从数据库查询所有患者 ID 以启用分批。之前 batch_size 在此场景下被静默忽略，
    # 导致 34K HiRID 患者在单次 DuckDB 查询中加载，32GB PC 上 OOM。
    if effective_batch_size is not None and _id_col is None:
        try:
            _total_patients_in_db = _get_total_patient_count(loader)
            if _total_patients_in_db and _total_patients_in_db > effective_batch_size:
                _fetched_ids = _sample_patient_ids(loader, _total_patients_in_db, verbose=False, sample_strategy='sorted')
                if _fetched_ids:
                    _id_col = _database_profile_or_default(loader.database).stay_id_col
                    _all_ids = list(_fetched_ids)
                    _total_patients = len(_all_ids)
                    if verbose:
                        print(f"📊 分批启用: 获取 {_total_patients} 患者ID, batch_size={effective_batch_size}")
        except Exception as e:
            logger.debug(f"获取患者ID以启用分批失败: {e}")
    
    # 自动检测：用户指定了 patient_ids 但未指定 batch_size
    # 🔧 2026-05-11: 同样默认不分批，仅低内存系统才触发
    if (
        not auto_chunk_strategy
        and merge
        and effective_chunk_size is None
        and effective_batch_size is None
        and _total_patients is not None
        and _total_patients > 5000
    ):
        avail_mem = get_available_memory_mb()
        est_mem = estimate_memory_mb(concepts_list, loader.database, _total_patients)
        # 🚀 稳定预算判定（同上）：用物理总内存而非波动可用，确定性地只对真正放不下的
        # 宽模块分批（medications/chemistry），窄模块保持一次性(快)；空闲机也不会因
        # "avail 大"而漏判导致宽模块一次性→静默截断。EASYICU_ONESHOT_BUDGET_MB 可覆盖。
        try:
            _env_b = os.environ.get('EASYICU_ONESHOT_BUDGET_MB')
            if _env_b:
                _oneshot_budget_mb = float(_env_b)
            else:
                import psutil as _psb
                _oneshot_budget_mb = (_psb.virtual_memory().total / (1024 * 1024)) * 0.6
        except Exception:
            _oneshot_budget_mb = 9830.0
        if est_mem > _oneshot_budget_mb:
            effective_batch_size = auto_batch_size(
                concepts_list, loader.database, _total_patients,
                available_memory_mb=_oneshot_budget_mb / 0.6,
            )
            if verbose and effective_batch_size:
                print(f"⚠️  稳定预算分批 (估算 {est_mem:.0f}MB > 预算 {_oneshot_budget_mb:.0f}MB): "
                      f"{_total_patients} patients, batch_size={effective_batch_size}")
            use_subprocess = True

    if auto_chunk_strategy and verbose and effective_batch_size is None:
        print("   🧠 已跳过自动 batch 分批，优先采用已验证的平衡 chunk 路径")
    
    # 大量患者时自动启用子进程隔离，避免 Python pymalloc 内存碎片
    # inprocess_batch_load 每批次泄漏 0.5-1.5GB 碎片（pymalloc arena 不归还 OS），
    # N 批次后 RSS = N * 碎片 + 结果数据。MIIV 94K patients: 15G RSS for 1.4G data.
    # subprocess 隔离: 每批在子进程中运行，子进程退出后 OS 完整回收内存，零碎片。
    #
    # 🔧 2026-05-16: 阈值改基于 **物理总内存** (psutil.virtual_memory().total)，
    # 不再用 available_mb。available 在 macOS/16GB 机上常驻 4-8GB（别的应用占了
    # 一部分），导致几乎所有 16GB+ 机器都走 subprocess 路径，付出 N×5s fork+import
    # 开销（eicu 33 batch ≈ 165s）。
    # 决策表（按 total RAM）：
    #   <12GB: 始终 subprocess（小内存，pymalloc 碎片致命）
    #   12-32GB: 仅在 cohort > 60K 时 subprocess（典型场景仍走 inprocess）
    #   ≥32GB: 仅在 cohort > 120K 时 subprocess
    # 显式覆盖：EASYICU_FORCE_INPROCESS_BATCH=1 强制 inprocess；
    #          EASYICU_FORCE_SUBPROCESS_BATCH=1 强制 subprocess。
    if not use_subprocess and effective_batch_size is not None:
        if os.environ.get('EASYICU_FORCE_INPROCESS_BATCH'):
            pass  # 用户显式禁用 subprocess，保持 inprocess
        elif os.environ.get('EASYICU_FORCE_SUBPROCESS_BATCH'):
            use_subprocess = True
        else:
            try:
                import psutil
                _total_mb = psutil.virtual_memory().total / (1024 * 1024)
            except Exception:
                _total_mb = get_available_memory_mb()  # 降级
            if _total_mb < 12 * 1024:
                use_subprocess = True
            elif _total_mb < 32 * 1024 and _total_patients is not None and _total_patients > 60000:
                use_subprocess = True
            elif _total_patients is not None and _total_patients > 120000:
                use_subprocess = True
    
    # 🔧 FIX Bug 54/63: daemon 子进程的分批隔离
    # Webapp 用 daemon=True 启动模块子进程以隔离内存碎片。
    # subprocess_batch_load 已支持三种隔离方式：
    #   - Linux/macOS daemon: os.fork()（_fork_and_run，不受 daemon 限制）
    #   - Windows daemon: subprocess.Popen（_popen_and_run，CreateProcess 不受 daemon 限制）
    #   - 非 daemon: multiprocessing.Process
    # 因此不再需要在 Windows daemon 中禁用 subprocess 模式。
    
    # 🔧 FIX: special concepts (KDIGO/circ/comorb/outcome/micro) were stripped from
    # concepts_list above and are only re-attached on the non-batched path (~L1480).
    # When batching triggers we route through subprocess_batch_load with the FULL list
    # (_concepts_all) so each per-batch api.load_concepts re-runs the special loaders.
    # inprocess_batch_load calls the *base* loader (no special routing), so specials
    # MUST take the subprocess path here.
    if _special and effective_batch_size is not None:
        use_subprocess = True

    # 执行分批处理
    if effective_batch_size is not None and _id_col is not None and _all_ids is not None:
        if _total_patients > effective_batch_size:
            load_kwargs = dict(
                interval=interval,
                win_length=win_length,
                aggregate=aggregate,
                keep_components=keep_components,
                merge=merge,
                r_compatible=r_compatible,
                chunk_size=effective_chunk_size,
                progress=progress,
                parallel_workers=effective_parallel_workers,
                concept_workers=effective_concept_workers,
                parallel_backend=parallel_backend,
                **kwargs,
            )
            
            if use_subprocess:
                # 子进程隔离（内存 < 16GB 或患者数 > 30K — 避免 pymalloc 碎片）
                if verbose:
                    print(f"🔒 使用子进程隔离模式 ({_total_patients} patients)")
                # 排除已显式传递的参数，避免重复
                _explicit_keys = {'merge', 'r_compatible', 'verbose'}
                subprocess_kwargs = {k: v for k, v in load_kwargs.items() if k not in _explicit_keys}
                final_result = subprocess_batch_load(
                    concepts=(_concepts_all if _special else concepts_list),
                    database=loader.database,
                    all_patient_ids={_id_col: _all_ids},
                    batch_size=effective_batch_size,
                    data_path=str(loader.data_path) if hasattr(loader, 'data_path') else None,
                    verbose=verbose,
                    merge=merge,
                    r_compatible=r_compatible,
                    dict_path=dict_path,
                    use_sofa2=use_sofa2,
                    **subprocess_kwargs,
                )
            else:
                # 进程内分批 + malloc_trim（内存 >= 16GB）
                final_result = inprocess_batch_load(
                    loader=loader,
                    concepts=concepts_list,
                    patient_ids={_id_col: _all_ids},
                    batch_size=effective_batch_size,
                    verbose=verbose,
                    memory_efficient=memory_efficient,
                    **load_kwargs,
                )
            
            if memory_efficient and isinstance(final_result, pd.DataFrame):
                final_result = _compress_dtypes(final_result, verbose=verbose)
            
            return final_result

    if (
        effective_batch_size is not None
        and use_streaming_patient_batches
        and _total_patients is not None
        and patient_ids is None
    ):
        load_kwargs = dict(
            interval=interval,
            win_length=win_length,
            aggregate=aggregate,
            keep_components=keep_components,
            merge=merge,
            r_compatible=r_compatible,
            chunk_size=effective_chunk_size,
            progress=progress,
            parallel_workers=effective_parallel_workers,
            concept_workers=effective_concept_workers,
            parallel_backend=parallel_backend,
            **kwargs,
        )
        if use_subprocess:
            patient_ids = _sample_patient_ids(loader, _total_patients, verbose, sample_strategy='sorted')
            if patient_ids is not None:
                patient_ids = _normalize_patient_ids_for_db(loader.database, patient_ids)
                _id_col = list(patient_ids.keys())[0]
                _all_ids = list(patient_ids.values())[0]
                return subprocess_batch_load(
                    concepts=(_concepts_all if _special else concepts_list),
                    database=loader.database,
                    all_patient_ids={_id_col: _all_ids},
                    batch_size=effective_batch_size,
                    data_path=str(loader.data_path) if hasattr(loader, 'data_path') else None,
                    verbose=verbose,
                    merge=merge,
                    r_compatible=r_compatible,
                    dict_path=dict_path,
                    use_sofa2=use_sofa2,
                    **load_kwargs,
                )
        return inprocess_batch_load_streaming(
            loader=loader,
            concepts=concepts_list,
            patient_batches=_iter_patient_id_batches(
                loader,
                effective_batch_size,
                total_patients=_total_patients,
                sample_strategy='sorted',
            ),
            total_patients=_total_patients,
            batch_size=effective_batch_size,
            verbose=verbose,
            memory_efficient=memory_efficient,
            **load_kwargs,
        )

    # 使用统一加载器加载概念
    if concepts_list:
        result = loader.load_concepts(
            concepts=concepts_list,
            patient_ids=patient_ids,
            interval=interval,
            win_length=win_length,
            aggregate=aggregate,
            keep_components=keep_components,
            merge=merge,
            r_compatible=r_compatible,
            chunk_size=effective_chunk_size,
            progress=progress,
            parallel_workers=effective_parallel_workers,
            concept_workers=effective_concept_workers,
            parallel_backend=parallel_backend,
            **kwargs
        )
    else:
        # All requested concepts were special; standard loader has nothing to do.
        result = {} if not merge else pd.DataFrame()

    # 2026-05-20: load any SPECIAL_CONCEPTS that were peeled off above
    # (`aki*`, `circ_failure`, `circ_event`) via their dedicated loaders,
    # then splice the resulting columns into the standard result so the
    # API surface looks uniform.
    if _special:
        # Build preloaded_data dict from the standard result so the
        # special loaders don't re-fetch crea / urine / lact / map etc.
        # — that would re-trigger the memory_manager auto-batch
        # subprocess pool (see Bug E6 in 2026-05-19 audit).
        def _frame_of(v):
            if v is None:
                return None
            if isinstance(v, pd.DataFrame):
                return v if not v.empty else None
            data_attr = getattr(v, "data", None)
            if isinstance(data_attr, pd.DataFrame) and not data_attr.empty:
                return data_attr
            return None

        _pre: Dict[str, pd.DataFrame] = {}
        if isinstance(result, dict):
            for name in (
                "crea",
                "urine",
                "weight",
                "rrt",
                "lact",
                "map",
                "norepi_rate",
                "epi_rate",
                "dobu_rate",
                "dopa_rate",
            ):
                df = _frame_of(result.get(name))
                if df is not None:
                    _pre[name] = df

        special_dict: Dict[str, pd.DataFrame] = {}
        if _need_kdigo:
            try:
                from .scores.kdigo_aki import load_kdigo_aki

                aki_df = load_kdigo_aki(
                    database=loader.database,
                    data_path=str(loader.data_path),
                    patient_ids=special_patient_ids,
                    max_patients=effective_max_patients,
                    verbose=verbose,
                    preloaded_data=_pre or None,
                )
                if isinstance(aki_df, pd.DataFrame) and not aki_df.empty:
                    id_time = [
                        c
                        for c in (
                            "stay_id",
                            "icustay_id",
                            "patientunitstayid",
                            "admissionid",
                            "patientid",
                            "CaseID",
                            "charttime",
                            "datetime",
                            "observationoffset",
                        )
                        if c in aki_df.columns
                    ]
                    for c in _need_kdigo:
                        if c in aki_df.columns:
                            special_dict[c] = aki_df[id_time + [c]].copy()
            except Exception as e:
                logger.warning(f"load_kdigo_aki failed: {e}")
        if _need_circ:
            try:
                from .scores.circ_failure import load_circ_failure

                cf_df = load_circ_failure(
                    database=loader.database,
                    data_path=str(loader.data_path),
                    max_patients=effective_max_patients,
                    patient_ids=special_patient_ids,
                    verbose=verbose,
                    preloaded_data=_pre or None,
                )
                if isinstance(cf_df, pd.DataFrame) and not cf_df.empty:
                    id_time = [
                        c
                        for c in (
                            "stay_id",
                            "icustay_id",
                            "patientunitstayid",
                            "admissionid",
                            "patientid",
                            "CaseID",
                            "charttime",
                            "datetime",
                            "observationoffset",
                        )
                        if c in cf_df.columns
                    ]
                    for c in _need_circ:
                        if c in cf_df.columns:
                            special_dict[c] = cf_df[id_time + [c]].copy()
            except Exception as e:
                logger.warning(f"load_circ_failure failed: {e}")
        if _need_comorb:
            # 'charlson' -> charlson_index, 'elixhauser' -> elixhauser_vw.
            # The full per-condition flags remain available via
            # easyicu.comorbidity.load_comorbidity(...).
            from .scores.comorbidity import load_comorbidity

            _comorb_index_col = {
                "charlson": "charlson_index",
                "elixhauser": "elixhauser_vw",
            }
            for c in _need_comorb:
                try:
                    como = load_comorbidity(
                        loader.database,
                        data_path=str(loader.data_path),
                        system=c,
                        patient_ids=special_patient_ids,
                        verbose=verbose,
                    )
                except Exception as e:
                    logger.warning(f"load_comorbidity({c}) failed: {e}")
                    continue
                if not isinstance(como, pd.DataFrame) or como.empty:
                    continue
                id_cols = [
                    col
                    for col in ("stay_id", "icustay_id", "patientunitstayid", "CaseID")
                    if col in como.columns
                ]
                src_col = _comorb_index_col[c]
                if src_col in como.columns and id_cols:
                    col = como[id_cols + [src_col]].rename(columns={src_col: c})
                    special_dict[c] = col.copy()
        if _need_outcome:
            from .scores.outcomes import load_outcomes

            try:
                oc = load_outcomes(
                    loader.database,
                    data_path=str(loader.data_path),
                    patient_ids=special_patient_ids,
                    verbose=verbose,
                )
            except Exception as e:
                logger.warning(f"load_outcomes failed: {e}")
                oc = None
            if isinstance(oc, pd.DataFrame) and not oc.empty:
                id_cols = [
                    col
                    for col in (
                        "stay_id",
                        "icustay_id",
                        "patientunitstayid",
                        "CaseID",
                        "admissionid",
                    )
                    if col in oc.columns
                ]
                for c in _need_outcome:
                    if c in oc.columns and id_cols:
                        special_dict[c] = oc[id_cols + [c]].copy()
        if _need_micro:
            from .scores.microbiology import load_microbiology

            try:
                mic = load_microbiology(
                    loader.database,
                    data_path=str(loader.data_path),
                    patient_ids=special_patient_ids,
                    verbose=verbose,
                )
            except Exception as e:
                logger.warning(f"load_microbiology failed: {e}")
                mic = None
            if isinstance(mic, pd.DataFrame) and not mic.empty:
                id_cols = [
                    col
                    for col in ("stay_id", "icustay_id", "patientunitstayid")
                    if col in mic.columns
                ]
                for c in _need_micro:
                    if c in mic.columns and id_cols:
                        special_dict[c] = mic[id_cols + [c]].copy()

        if special_dict:
            if not merge:
                # User asked for dict. Standard loader returns dict for
                # >1 concepts but a bare DataFrame for a single concept
                # (a quirk of the public API). Normalise to dict so the
                # special concepts can be attached uniformly.
                if isinstance(result, pd.DataFrame):
                    if concepts_list:
                        result = {concepts_list[0]: result}
                    else:
                        result = {}
                elif not isinstance(result, dict):
                    result = {}
                result.update(special_dict)
            else:
                # merge=True: outer-join special columns onto the standard
                # result by shared id/time columns.
                if isinstance(result, dict):
                    # If standard path somehow returned dict despite
                    # merge=True (edge case when concepts_list was empty),
                    # bring it back to DataFrame form.
                    if result:
                        from functools import reduce

                        frames = list(result.values())
                        if frames and all(isinstance(f, pd.DataFrame) for f in frames):
                            result = reduce(
                                lambda a, b: a.merge(
                                    b,
                                    on=[c for c in a.columns if c in b.columns],
                                    how="outer",
                                ),
                                frames,
                            )
                        else:
                            result = pd.DataFrame()
                    else:
                        result = pd.DataFrame()
                if isinstance(result, pd.DataFrame):
                    for name, sdf in special_dict.items():
                        join_cols = [
                            c for c in sdf.columns if c in result.columns and c != name
                        ]
                        if join_cols:
                            try:
                                result = result.merge(sdf, on=join_cols, how="outer")
                            except Exception as e:
                                logger.warning(
                                    f"failed to merge special concept {name!r}: {e}"
                                )
                        elif result.empty:
                            # No standard concepts at all — return the
                            # special frame as-is for this single-concept
                            # case.
                            result = sdf

    # 🆕 内存优化模式：压缩数据类型
    if memory_efficient:
        if isinstance(result, pd.DataFrame):
            result = _compress_dtypes(result, verbose=verbose)
        elif isinstance(result, dict):
            result = {k: _compress_dtypes(v, verbose=verbose) for k, v in result.items()}

    if r_compatible and merge and len(concepts_list) == 1 and isinstance(result, pd.DataFrame):
        result = _expand_public_numeric_win_tbl_output(result, concepts_list[0], interval)
    
    return result

# 为了兼容旧代码，保留旧的函数名
def load_concept(*args, **kwargs):
    """load_concepts的别名（向后兼容）"""
    return load_concepts(*args, **kwargs)

def load_sofa(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    keep_components: bool = True,
    verbose: bool = False,
    **kwargs  # 允许传递额外参数如align_to_admission
) -> pd.DataFrame:
    """
    加载SOFA评分（便捷函数）- 重构版本

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        keep_components: 是否保留组件（默认True）
        verbose: 是否显示详细信息
        **kwargs: 额外参数传递给load_concepts（如align_to_admission）

    Returns:
        SOFA评分DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> sofa = load_sofa(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> sofa = load_sofa(patient_ids=[123, 456],
        ...                  database='miiv', data_path='/data/miiv',
        ...                  win_length='12h', interval='6h')
        >>>
        >>> # 使用时间对齐
        >>> sofa = load_sofa(patient_ids=[123, 456],
        ...                  align_to_admission=True)
    """
    if verbose:
        print("🏥 加载SOFA评分...")

    return load_concepts(
        'sofa',
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        keep_components=keep_components,
        verbose=verbose,
        **kwargs  # 传递额外参数
    )

def load_sofa2(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    keep_components: bool = True,
    verbose: bool = False,
    **kwargs  # 允许传递额外参数如align_to_admission
) -> pd.DataFrame:
    """
    加载SOFA-2评分（2025年新标准）- 重构版本

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        keep_components: 是否保留组件（默认True）
        verbose: 是否显示详细信息
        **kwargs: 额外参数传递给load_concepts（如align_to_admission）

    Returns:
        SOFA-2评分DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> sofa2 = load_sofa2(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> sofa2 = load_sofa2(patient_ids=[123, 456],
        ...                   database='miiv', data_path='/data/miiv')
    """
    if verbose:
        print("🏥 加载SOFA-2评分（2025标准）...")

    return load_concepts(
        'sofa2',
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        keep_components=keep_components,
        verbose=verbose,
        use_sofa2=True,  # 强制使用SOFA2字典
        **kwargs  # 传递额外参数
    )

def load_sepsis3(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载Sepsis-3诊断相关数据 - 重构版本

    包含: SOFA, abx, samp, susp_inf, sep3

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        verbose: 是否显示详细信息

    Returns:
        Sepsis-3数据DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> sep3 = load_sepsis3(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> sep3 = load_sepsis3(patient_ids=[123, 456],
        ...                     database='miiv', data_path='/data/miiv')
    """
    if verbose:
        print("🦠 加载Sepsis-3相关数据...")

    # 只加载sep3概念，它已经包含了所有必需的诊断信息
    # 如果需要详细的组件（SOFA, abx等），用户可以分别加载
    return load_concepts(
        'sep3',
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        verbose=verbose
    )

def load_vitals(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载生命体征数据（便捷函数）- 重构版本

    包含: hr, sbp, dbp, temp, resp, spo2

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        verbose: 是否显示详细信息

    Returns:
        生命体征DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> vitals = load_vitals(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> vitals = load_vitals(patient_ids=[123, 456],
        ...                      database='miiv', data_path='/data/miiv',
        ...                      interval='30m')
    """
    vital_concepts = ['hr', 'sbp', 'dbp', 'temp', 'resp', 'spo2']

    if verbose:
        print("❤️  加载生命体征...")

    return load_concepts(
        vital_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        verbose=verbose
    )

def load_labs(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '6h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载实验室检查数据（便捷函数）- 重构版本

    包含: wbc, plt, crea, bili, lact, ph

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认6小时，实验室检查频率较低）
        verbose: 是否显示详细信息

    Returns:
        实验室检查DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> labs = load_labs(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> labs = load_labs(patient_ids=[123, 456],
        ...                   database='miiv', data_path='/data/miiv',
        ...                   interval='12h')
    """
    lab_concepts = ['wbc', 'plt', 'crea', 'bili', 'lact', 'ph']

    if verbose:
        print("🔬 加载实验室检查...")

    return load_concepts(
        lab_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        verbose=verbose
    )

def list_available_concepts(source: Optional[str] = None) -> List[str]:
    """
    列出可用的概念
    
    Args:
        source: 如果指定，只列出该数据源支持的概念
        
    Returns:
        概念名称列表
        
    Examples:
        >>> # 列出所有概念
        >>> all_concepts = list_available_concepts()
        >>> 
        >>> # 列出MIMIC支持的概念
        >>> mimic_concepts = list_available_concepts('mimic')
    """
    dict_obj = load_dictionary()
    
    if source is None:
        # 返回所有概念 (使用 _concepts 属性)
        return list(dict_obj._concepts.keys())
    
    # 返回特定数据源支持的概念
    supported = []
    for name, concept in dict_obj._concepts.items():
        if hasattr(concept, 'sources') and source in concept.sources:
            supported.append(name)
    
    return sorted(supported)

def list_available_sources(use_user_config: bool = False) -> List[str]:
    """
    列出可用的数据源

    Args:
        use_user_config: If True, read the legacy user configuration registry.
            By default this reports packaged sources shipped with EasyICU.
    
    Returns:
        数据源名称列表
        
    Examples:
        >>> sources = list_available_sources()
        >>> print(sources)
        ['mimic', 'hirid', 'eicu', 'aumc']
    """
    registry = load_user_data_sources() if use_user_config else load_packaged_data_sources()
    return [cfg.name for cfg in registry]

def get_concept_info(concept_name: str) -> Dict:
    """
    获取概念的详细信息
    
    Args:
        concept_name: 概念名称
        
    Returns:
        包含概念信息的字典
        
    Examples:
        >>> info = get_concept_info('hr')
        >>> print(info['description'])
        'heart rate'
    """
    dict_obj = load_dictionary()
    concept = dict_obj.get(concept_name)

    if concept is None:
        raise ValueError(f"未知概念: {concept_name}")

    units = list(getattr(concept, 'units', None) or [])
    sources = getattr(concept, 'sources', {}) or {}

    info = {
        'name': concept_name,
        'description': getattr(concept, 'description', ''),
        'category': getattr(concept, 'category', ''),
        'units': units,
        'unit': units[0] if units else '',
        'sources': sorted(sources.keys()),
        'class_name': getattr(concept, 'class_name', None),
        'callback': getattr(concept, 'callback', None),
        'sub_concepts': list(getattr(concept, 'sub_concepts', None) or []),
        'depends_on': list(getattr(concept, 'depends_on', None) or []),
    }
    
    return info

# === 新增模块函数（参考ricu.R） ===

def _validate_concepts(concepts: List[str], verbose: bool = False) -> List[str]:
    """
    验证概念是否存在于字典中，返回可用的概念列表

    Args:
        concepts: 要验证的概念列表
        verbose: 是否显示详细信息

    Returns:
        可用的概念列表
    """
    try:
        dict_obj = load_dictionary()
        # 使用 _concepts 属性 (ConceptDictionary 内部存储)
        all_concepts = set(dict_obj._concepts.keys())
        available_concepts = [c for c in concepts if c in all_concepts]
        missing_concepts = [c for c in concepts if c not in all_concepts]

        if verbose and missing_concepts:
            print(f"  ⚠️  以下概念在字典中不存在，将被跳过: {missing_concepts}")

        return available_concepts
    except Exception:
        return concepts  # 如果验证失败，返回原列表

def load_demographics(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载基础人口统计学数据（参考ricu.R的data_demo）

    包含: age, bmi, height, sex, weight

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        人口统计学DataFrame

    Examples:
        >>> demo = load_demographics(patient_ids=[123, 456])
    """
    if verbose:
        print("👥 加载基础人口统计学数据...")

    demo_concepts = ['age', 'bmi', 'height', 'sex', 'weight']

    try:
        result = load_concepts(
            concepts=demo_concepts,
            patient_ids=patient_ids,
            database=database,
            data_path=data_path,
            merge=True,
            verbose=verbose
        )
        if result is None:
            return pd.DataFrame()
        return result

    except Exception as e:
        if verbose:
            print(f"  ❌ 人口统计学数据加载失败: {e}")
        return pd.DataFrame()

def load_outcomes(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    keep_components: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载结局指标数据（参考ricu.R的data_outcome）

    包含: death, los_icu, qsofa, sirs

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        keep_components: 是否保留组件（默认True）
        verbose: 是否显示详细信息

    Returns:
        结局指标DataFrame

    Examples:
        >>> outcomes = load_outcomes(patient_ids=[123, 456])
    """
    if verbose:
        print("📊 加载结局指标数据...")

    concepts = ['death', 'los_icu', 'qsofa', 'sirs']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        keep_components=keep_components,
        merge=True,
        verbose=verbose
    )

def load_vitals_detailed(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载详细生命体征数据（参考ricu.R的data_vital）

    包含: dbp, etco2, hr, map, sbp, temp

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        详细生命体征DataFrame

    Examples:
        >>> vitals = load_vitals_detailed(patient_ids=[123, 456])
    """
    if verbose:
        print("❤️ 加载详细生命体征数据...")

    concepts = ['dbp', 'etco2', 'hr', 'map', 'sbp', 'temp']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )

def load_neurological(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载神经系统评估数据（参考ricu.R的data_neu）

    包含: avpu, egcs, gcs, mgcs, rass, vgcs

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        神经系统评估DataFrame

    Examples:
        >>> neuro = load_neurological(patient_ids=[123, 456])
    """
    if verbose:
        print("🧠 加载神经系统评估数据...")

    concepts = ['avpu', 'egcs', 'gcs', 'mgcs', 'rass', 'vgcs']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )

def load_output(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载输出量数据（参考ricu.R的data_output）

    包含: urine, urine24, total_input_ml, fluid_balance, fluid_balance_cumulative

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        输出量DataFrame

    Examples:
        >>> output = load_output(patient_ids=[123, 456])
    """
    if verbose:
        print("💧 加载输出量数据...")

    concepts = ['urine', 'urine24', 'total_input_ml',
                'fluid_balance', 'fluid_balance_cumulative']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )

def load_respiratory(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载呼吸系统数据（参考ricu.R的data_resp）

    包含: ett_gcs, mech_vent, o2sat, sao2, pafi, resp, safi, supp_o2, vent_ind

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        呼吸系统DataFrame

    Examples:
        >>> resp = load_respiratory(patient_ids=[123, 456])
    """
    if verbose:
        print("🫁 加载呼吸系统数据...")

    concepts = ['ett_gcs', 'mech_vent', 'o2sat', 'sao2', 'pafi', 'resp', 'safi', 'supp_o2', 'vent_ind']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )

def load_lab_comprehensive(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载全面的实验室检查数据（参考ricu.R的data_lab）

    包含: alb, alp, alt, ast, bicar, bili, bili_dir, bun, ca, ck, ckmb,
          cl, crea, crp, glu, k, mg, na, phos, tnt

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        实验室检查DataFrame

    Examples:
        >>> labs = load_lab_comprehensive(patient_ids=[123, 456])
    """
    if verbose:
        print("🧪 加载全面的实验室检查数据...")

    concepts = ['alb', 'alp', 'alt', 'ast', 'bicar', 'bili', 'bili_dir', 'bun',
               'ca', 'ck', 'ckmb', 'cl', 'crea', 'crp', 'glu', 'k', 'mg', 'na', 'phos', 'tnt']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )

def load_blood_gas(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载血气分析数据（参考ricu.R的data_blood）

    包含: be, cai, fio2, hbco, lact, methb, pco2, ph, po2, tco2

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        血气分析DataFrame

    Examples:
        >>> blood_gas = load_blood_gas(patient_ids=[123, 456])
    """
    if verbose:
        print("🩸 加载血气分析数据...")

    concepts = ['be', 'cai', 'fio2', 'hbco', 'lact', 'methb', 'pco2', 'ph', 'po2', 'tco2']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    # 逐个尝试加载，跳过无法加载的概念（某些概念可能在特定数据库中没有配置）
    results = []
    loaded_concepts = []
    for concept in available_concepts:
        try:
            df = load_concepts(
                concepts=[concept],
                patient_ids=patient_ids,
                database=database,
                data_path=data_path,
                interval=interval,
                win_length=win_length,
                merge=True,
                verbose=False
            )
            if df is not None and not df.empty:
                results.append(df)
                loaded_concepts.append(concept)
        except Exception:
            pass  # 跳过无法加载的概念
    
    if not results:
        if verbose:
            print("  ❌ 没有成功加载的概念")
        return pd.DataFrame()
    
    if verbose:
        print(f"  ✅ 成功加载 {len(loaded_concepts)} 个概念: {loaded_concepts}")
    
    # 合并结果
    if len(results) == 1:
        return results[0]
    
    # 多个结果需要合并
    merged = results[0]
    for df in results[1:]:
        # 找到共同的 ID 和时间列进行合并
        id_cols = [c for c in merged.columns if 'id' in c.lower() or c in ['stay_id', 'subject_id', 'patientunitstayid', 'admissionid', 'patientid']]
        time_cols = [c for c in merged.columns if 'time' in c.lower() or c == 'charttime']
        merge_cols = list(set(id_cols + time_cols) & set(df.columns))
        if merge_cols:
            merged = pd.merge(merged, df, on=merge_cols, how='outer')
        else:
            merged = pd.concat([merged, df], ignore_index=True)
    
    return merged

def load_hematology(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载血液学检查数据（参考ricu.R的data_hematology）

    包含: bnd, esr, fgn, hgb, inr_pt, lymph, mch, mchc, mcv, neut, plt, ptt, wbc

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        血液学DataFrame

    Examples:
        >>> hematology = load_hematology(patient_ids=[123, 456])
    """
    if verbose:
        print("🩸 加载血液学检查数据...")

    concepts = ['bnd', 'esr', 'fgn', 'hgb', 'inr_pt', 'lymph', 'mch', 'mchc',
               'mcv', 'neut', 'plt', 'ptt', 'wbc']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )

def load_medications(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
    groups: Optional[Union[str, List[str]]] = None,
    include_new: bool = True,
) -> pd.DataFrame:
    """
    加载药物治疗数据（参考ricu.R的data_med + EasyICU 扩展药物集）

    默认包含（如 include_new=True，共 48 个概念）:
      - Vasopressors/inotropes: adh_rate, dobu_rate/dur/60, dopa_rate/dur,
        epi_rate/dur, norepi_rate/dur/equiv, phn_rate, vaso_ind
      - Sedation: propofol, propofol_rate, midazolam, midazolam_rate,
        dexmedetomidine, lorazepam, ketamine
      - Analgesia: fentanyl, fentanyl_rate, morphine
      - Neuromuscular blockers: rocuronium, vecuronium, cisatracurium
      - Antibiotics: abx, vancomycin, meropenem
      - GI prophylaxis: pantoprazole
      - GI: octreotide
      - Neurology: levetiracetam
      - Corticosteroids: cort, dexamethasone
      - Reversal: neostigmine
      - Electrolytes: calcium_iv, potassium_iv, magnesium_iv, bicarbonate
      - Colloids / blood products: albumin_iv, packed_rbc, ffp, platelets
      - Other: dex, dextrose50, ins, furosemide, heparin, mannitol,
        amiodarone, milrinone, nitroglycerin

    如 include_new=False, 回落到最初的 14 个 ricu 概念（向后兼容）:
      abx, adh_rate, cort, dex, dobu_dur, dobu_rate, dobu60,
      epi_dur, epi_rate, ins, norepi_dur, norepi_equiv, norepi_rate, vaso_ind

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息
        groups: 只加载指定的药理学分组。支持单个字符串或列表，可选值：
            'vasopressors', 'sedation', 'analgesia', 'neuromuscular',
            'antibiotics', 'cardiac', 'diuretics', 'anticoagulation',
            'endocrine', 'vasodilators', 'gi_prophylaxis', 'electrolytes',
            'colloids_blood', 'neurology', 'gi', 'reversal',
            'corticosteroids', 'other'
            例如 groups='sedation' 只加载镇静类药物。
        include_new: 是否包含 EasyICU 扩展药物。默认 True。
            设为 False 仅加载原 ricu 14 个概念（向后兼容老脚本）。

    Returns:
        药物治疗DataFrame

    Examples:
        >>> # 全部药物（默认）
        >>> meds = load_medications(patient_ids=[123, 456])

        >>> # 只要镇静药物
        >>> seda = load_medications(groups='sedation', database='miiv')

        >>> # 镇静 + 镇痛
        >>> analg = load_medications(groups=['sedation', 'analgesia'])

        >>> # 向后兼容模式（仅原 ricu 14 概念）
        >>> ricu_compat = load_medications(include_new=False)
    """
    if verbose:
        print("💊 加载药物治疗数据...")

    # 药理学分组 - single source of truth
    MEDICATION_GROUPS = {
        'vasopressors': [
            'adh_rate', 'dobu_dur', 'dobu_rate', 'dobu60',
            'dopa_dur', 'dopa_rate', 'epi_dur', 'epi_rate',
            'norepi_dur', 'norepi_equiv', 'norepi_rate',
            'phn_rate', 'vaso_ind',
        ],
        'sedation': [
            'propofol', 'propofol_rate',
            'midazolam', 'midazolam_rate',
            'dexmedetomidine', 'lorazepam', 'ketamine',
        ],
        'analgesia': [
            'fentanyl', 'fentanyl_rate', 'morphine',
        ],
        'neuromuscular': [
            'rocuronium', 'vecuronium', 'cisatracurium',
        ],
        'antibiotics': ['abx', 'vancomycin', 'meropenem'],
        'cardiac': ['amiodarone', 'milrinone'],
        'diuretics': ['furosemide', 'mannitol'],
        'anticoagulation': ['heparin', 'warfarin', 'apixaban', 'enoxaparin'],
        'antiplatelet': ['aspirin'],
        'endocrine': ['cort', 'ins', 'insulin'],
        'vasodilators': ['nitroglycerin'],
        'gi_prophylaxis': ['pantoprazole'],
        'electrolytes': ['calcium_iv', 'potassium_iv', 'magnesium_iv', 'bicarbonate'],
        'colloids_blood': ['albumin_iv', 'packed_rbc', 'ffp', 'platelets'],
        'neurology': ['levetiracetam'],
        'gi': ['octreotide'],
        'reversal': ['neostigmine'],
        'corticosteroids': ['cort', 'dexamethasone'],
        'other': ['dex', 'dextrose50'],  # dextrose as fluid/med
    }

    # 构建目标概念列表
    legacy_concepts = [
        'abx', 'adh_rate', 'cort', 'dex', 'dobu_dur', 'dobu_rate', 'dobu60',
        'epi_dur', 'epi_rate', 'ins', 'norepi_dur', 'norepi_equiv',
        'norepi_rate', 'vaso_ind',
    ]

    if groups is not None:
        # Explicit group selection overrides include_new
        if isinstance(groups, str):
            groups = [groups]
        unknown = set(groups) - set(MEDICATION_GROUPS)
        if unknown:
            raise ValueError(
                f"Unknown medication group(s): {sorted(unknown)}. "
                f"Valid groups: {sorted(MEDICATION_GROUPS)}"
            )
        concepts: List[str] = []
        seen: set = set()
        for g in groups:
            for c in MEDICATION_GROUPS[g]:
                if c not in seen:
                    concepts.append(c)
                    seen.add(c)
    elif include_new:
        # Full catalog: union of all groups
        concepts = []
        seen = set()
        for group_concepts in MEDICATION_GROUPS.values():
            for c in group_concepts:
                if c not in seen:
                    concepts.append(c)
                    seen.add(c)
    else:
        # Backward-compatible ricu subset
        concepts = legacy_concepts

    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    # 逐个尝试加载，跳过无法加载的概念（某些概念可能在特定数据库中没有配置）
    results = []
    loaded_concepts = []
    for concept in available_concepts:
        try:
            df = load_concepts(
                concepts=[concept],
                patient_ids=patient_ids,
                database=database,
                data_path=data_path,
                interval=interval,
                win_length=win_length,
                merge=True,
                verbose=False
            )
            if df is not None and not df.empty:
                results.append(df)
                loaded_concepts.append(concept)
        except Exception:
            pass  # 跳过无法加载的概念
    
    if not results:
        if verbose:
            print("  ❌ 没有成功加载的概念")
        return pd.DataFrame()
    
    if verbose:
        print(f"  ✅ 成功加载 {len(loaded_concepts)} 个概念: {loaded_concepts}")
    
    # 合并结果
    if len(results) == 1:
        return results[0]
    
    # 多个结果需要合并
    merged = results[0]
    for df in results[1:]:
        # 找到共同的 ID 和时间列进行合并
        id_cols = [c for c in merged.columns if 'id' in c.lower() or c in ['stay_id', 'subject_id', 'patientunitstayid', 'admissionid', 'patientid']]
        time_cols = [c for c in merged.columns if 'time' in c.lower() or c == 'charttime']
        merge_cols = list(set(id_cols + time_cols) & set(df.columns))
        if merge_cols:
            merged = pd.merge(merged, df, on=merge_cols, how='outer')
        else:
            merged = pd.concat([merged, df], ignore_index=True)
    
    return merged

# 为了兼容性，也导出原始的类和函数
__all__ = [
    # 主要API
    'load_concepts',      # 主API（智能默认值）
    'load_concept',       # 别名（向后兼容）

    # Easy API（便捷函数）
    'load_sofa',
    'load_sofa2',
    'load_sepsis3',
    'load_vitals',
    'load_labs',

    # 新增模块函数（参考ricu.R）
    'load_demographics',     # 基础人口统计学
    'load_outcomes',         # 结局指标
    'load_vitals_detailed',   # 详细生命体征
    'load_neurological',     # 神经系统评估
    'load_output',           # 输出量
    'load_respiratory',      # 呼吸系统
    'load_lab_comprehensive', # 全面实验室检查
    'load_blood_gas',        # 血气分析
    'load_hematology',       # 血液学检查
    'load_medications',      # 药物治疗

    # 工具函数
    'list_available_concepts',
    'list_available_sources',
    'get_concept_info',
    
    # 缓存管理
    'keep_cache',
    'clear_global_loader',
    
    # 增强功能（从api_enhanced.py合并）
    'load_concept_cached',
    'align_to_icu_admission',
    'load_sofa_with_score',
]


# ============================================================================
# 增强功能 - 缓存和时间对齐 (从api_enhanced.py合并)
# ============================================================================

import pickle
import hashlib

import json
def _get_cache_key(concepts: List[str], source: str, **kwargs) -> str:
    """Generate cache key from parameters."""
    payload = {
        "source": source,
        "concepts": sorted(concepts),
        "parameters": kwargs,
    }
    key_str = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(key_str.encode()).hexdigest()

def _data_path_fingerprint(
    data_path: Union[str, Path],
    *,
    exclude_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Fingerprint dataset identity and file metadata for cache isolation."""
    root = Path(data_path).expanduser().resolve()
    excluded = Path(exclude_dir).expanduser().resolve() if exclude_dir else None
    digest = hashlib.sha256(str(root).encode())
    if root.is_file():
        stat = root.stat()
        digest.update(f"{root.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return digest.hexdigest()

    suffixes = {".parquet", ".csv", ".gz", ".json"}
    for path in sorted(
        (
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in suffixes
            and (excluded is None or not p.is_relative_to(excluded))
        ),
        key=lambda p: str(p.relative_to(root)),
    ):
        stat = path.stat()
        rel = path.relative_to(root)
        digest.update(f"{rel}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()
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
    use_pickle: bool = True,
    n_patients: Optional[int] = None,
    **kwargs,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load ICU concept data with caching support.

    Args:
        concepts: Concept name(s) to load
        source: Data source name ('mimic', 'miiv', etc.)
        data_path: Path to data source files
        cache_dir: Directory for cache files (default: data_path/cache)
        force_reload: If True, ignore cache and reload from source
        patient_ids: Optional patient ID filter
        merge: If True, merge concepts into wide format
        align_time: If True, align charttime to ICU admission (hours since admission)
        verbose: Show progress messages
        use_pickle: If True, cache as pickle; if False, use CSV
        n_patients: If provided, randomly sample N patients (for testing)
        **kwargs: Additional parameters for concept resolver

    Returns:
        DataFrame with concept data (and optionally time-aligned)
    """
    # Setup cache directory
    if cache_dir is None:
        cache_dir = Path(data_path) / "cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare concept list
    if isinstance(concepts, str):
        concept_list = [concepts]
    else:
        concept_list = list(concepts)
    
    # Generate cache key
    cache_params = {
        "merge": merge,
        "align_time": align_time,
        "patient_ids": patient_ids,
        "n_patients": n_patients,
        "data_path": str(Path(data_path).expanduser().resolve()),
        "data_fingerprint": _data_path_fingerprint(data_path, exclude_dir=cache_dir),
        **kwargs,
    }
    cache_key = _get_cache_key(concept_list, source, **cache_params)
    cache_ext = 'pkl' if use_pickle else 'csv'
    cache_file = (
        cache_dir
        / f"{source}_{'_'.join(concept_list[:3])}_{cache_key[:32]}.{cache_ext}"
    )
    
    # Try to load from cache
    if not force_reload and cache_file.exists():
        if verbose:
            print(f"📦 从缓存加载: {cache_file.name}")
        try:
            if use_pickle:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
            else:
                result = pd.read_csv(cache_file, parse_dates=['charttime'])
            
            if verbose:
                if isinstance(result, pd.DataFrame):
                    print(f"✅ 成功加载 {len(result):,} 行缓存数据")
                else:
                    print(f"✅ 成功加载 {len(result)} 个概念的缓存数据")
            return result
        except Exception as e:
            if verbose:
                print(f"⚠️  缓存加载失败: {e}，重新提取...")
    
    # Load from source using load_concepts
    result = load_concepts(
        concepts=concept_list,
        patient_ids=patient_ids,
        database=source,
        data_path=data_path,
        merge=merge,
        verbose=verbose,
        n_patients=n_patients,
        **kwargs,
    )
    
    # Save to cache
    if align_time:
        result = align_to_icu_admission(
            result,
            database=source,
            data_path=data_path,
            verbose=verbose,
        )
    try:
        if use_pickle:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        else:
            if isinstance(result, pd.DataFrame):
                result.to_csv(cache_file, index=False)
        if verbose:
            print(f"💾 缓存已保存: {cache_file.name}")
    except Exception as e:
        if verbose:
            print(f"⚠️  缓存保存失败: {e}")
    
    return result

def align_to_icu_admission(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    aggregate_hourly: bool = True,
    agg_func: str = 'median',
    filter_icu_window: bool = True,
    before_icu_hours: int = 0,
    after_icu_hours: int = 0,
    verbose: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Align charttime to ICU admission time and aggregate to hourly intervals.
    根据ricu的stay_windows逻辑，默认只保留ICU住院期间的数据。
    
    Args:
        data: Concept data with charttime
        database: Database name ('miiv', 'eicu', etc.)
        data_path: Path to data source files
        aggregate_hourly: If True, aggregate multiple measurements per hour
        agg_func: Aggregation function ('median', 'mean', 'min', 'max')
        filter_icu_window: If True, filter to ICU stay window (default: True)
        before_icu_hours: Hours before ICU admission to include (default: 0)
        after_icu_hours: Hours after ICU discharge to include (default: 0)
        verbose: Show progress
        
    Returns:
        Data with charttime as integer hours since ICU admission, one row per hour
    """
    if verbose:
        print("⏰ 对齐时间到ICU入院时间...")
    
    # Handle dict of DataFrames
    if isinstance(data, dict):
        return {
            name: align_to_icu_admission(df, database, data_path, aggregate_hourly, agg_func, 
                                        filter_icu_window, before_icu_hours, after_icu_hours, verbose=False)
            for name, df in data.items()
        }
    
    # Simplified implementation - users can extend with full logic from api_enhanced.py if needed
    if verbose:
        print("⚠️  完整的时间对齐功能需要从load_concepts返回的数据包含charttime列")
    
    return data

def load_sofa_with_score(
    patient_ids: Optional[List] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: str = '1h',
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Load SOFA score with all components in a single DataFrame.
    
    Args:
        patient_ids: Patient ID filter
        database: Database name
        data_path: Path to data source
        interval: Time interval
        verbose: Show progress
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with SOFA scores and components
    """
    sofa_concepts = ['sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 
                     'sofa_cardio', 'sofa_cns', 'sofa_renal']
    
    result = load_concepts(
        concepts=sofa_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        merge=True,
        verbose=verbose,
        **kwargs
    )
    
    return result


# ==============================================================================
# 患者队列筛选 API
# ==============================================================================

def filter_patients(
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    # 筛选条件
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
    first_icu_stay: Optional[bool] = None,
    los_min: Optional[float] = None,
    los_max: Optional[float] = None,
    gender: Optional[str] = None,
    survived: Optional[bool] = None,
    has_sepsis: Optional[bool] = None,
    # 输出控制
    return_dataframe: bool = False,
    verbose: bool = False,
) -> Union[List[int], pd.DataFrame]:
    """
    根据人口统计学和临床条件筛选ICU患者队列
    
    支持的筛选条件:
    - 年龄范围 (age_min, age_max)
    - 是否首次入ICU (first_icu_stay)
    - ICU住院时长 (los_min, los_max，单位：小时)
    - 性别 (gender: 'M' 或 'F')
    - 是否存活出院 (survived)
    - 是否有Sepsis诊断 (has_sepsis)
    
    Args:
        database: 数据库类型 ('miiv', 'eicu', 'aumc', 'hirid')
        data_path: 数据路径
        age_min: 最小年龄
        age_max: 最大年龄
        first_icu_stay: 是否仅首次入ICU
        los_min: 最短住院时长（小时）
        los_max: 最长住院时长（小时）
        gender: 性别 ('M' 男 / 'F' 女)
        survived: 是否存活出院
        has_sepsis: 是否有Sepsis诊断
        return_dataframe: 是否返回完整DataFrame（包含人口统计学信息）
        verbose: 显示详细信息
    
    Returns:
        患者ID列表，或人口统计学DataFrame（如果return_dataframe=True）
    
    Examples:
        >>> # 筛选18-80岁首次入ICU的成人患者
        >>> adult_first_icu = filter_patients(
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     age_min=18, age_max=80,
        ...     first_icu_stay=True
        ... )
        >>> print(f"筛选到 {len(adult_first_icu)} 名患者")
        >>>
        >>> # 筛选Sepsis存活患者
        >>> sepsis_survivors = filter_patients(
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     has_sepsis=True,
        ...     survived=True
        ... )
        >>>
        >>> # 获取完整人口统计学信息
        >>> cohort_df = filter_patients(
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     age_min=18,
        ...     return_dataframe=True
        ... )
    """
    from .patient_filter import PatientFilter
    
    # 自动检测数据库和路径
    if database is None:
        database = detect_database_type(data_path)
    if data_path is None:
        data_path = get_default_data_path(database)
    
    pf = PatientFilter(database=database, data_path=data_path, verbose=verbose)
    
    return pf.filter(
        age_min=age_min, age_max=age_max,
        first_icu_stay=first_icu_stay,
        los_min=los_min, los_max=los_max,
        gender=gender, survived=survived,
        has_sepsis=has_sepsis,
        return_dataframe=return_dataframe
    )


def load_concepts_filtered(
    concepts: Union[str, List[str]],
    # 患者筛选条件
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
    first_icu_stay: Optional[bool] = None,
    los_min: Optional[float] = None,
    los_max: Optional[float] = None,
    gender: Optional[str] = None,
    survived: Optional[bool] = None,
    has_sepsis: Optional[bool] = None,
    # 其他load_concepts参数
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Optional[Union[str, pd.Timedelta]] = '1h',
    win_length: Optional[Union[str, pd.Timedelta]] = None,
    aggregate: Optional[Union[str, Dict]] = None,
    keep_components: bool = False,
    verbose: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    根据患者筛选条件加载概念数据 - 整合患者筛选和数据加载
    
    这是一个便捷函数，将患者队列筛选和概念加载整合为一步操作：
    1. 先根据人口统计学条件筛选患者
    2. 然后加载这些患者的概念数据
    
    Args:
        concepts: 要加载的概念名称或列表
        
        # === 患者筛选条件 ===
        age_min: 最小年龄
        age_max: 最大年龄
        first_icu_stay: 是否仅首次入ICU
        los_min: 最短住院时长（小时）
        los_max: 最长住院时长（小时）
        gender: 性别 ('M' 或 'F')
        survived: 是否存活出院
        has_sepsis: 是否有Sepsis诊断
        
        # === 数据加载参数 ===
        database: 数据库类型
        data_path: 数据路径
        interval: 时间对齐间隔
        win_length: 窗口长度
        aggregate: 聚合方式
        keep_components: 是否保留组件
        verbose: 显示详细信息
        **kwargs: 其他参数传递给load_concepts
    
    Returns:
        筛选后患者的概念数据DataFrame
    
    Examples:
        >>> # 加载成人首次入ICU患者的SOFA评分
        >>> sofa = load_concepts_filtered(
        ...     'sofa',
        ...     age_min=18, age_max=80,
        ...     first_icu_stay=True,
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     win_length='24h'
        ... )
        >>>
        >>> # 加载Sepsis患者的生命体征
        >>> sepsis_vitals = load_concepts_filtered(
        ...     ['hr', 'sbp', 'temp'],
        ...     has_sepsis=True,
        ...     database='miiv',
        ...     data_path='/path/to/data'
        ... )
    """
    # 自动检测数据库和路径
    if database is None:
        database = detect_database_type(data_path)
    if data_path is None:
        data_path = get_default_data_path(database)
    
    # 第1步：筛选患者
    has_filter = any([
        age_min is not None, age_max is not None,
        first_icu_stay is not None,
        los_min is not None, los_max is not None,
        gender is not None, survived is not None,
        has_sepsis is not None
    ])
    
    if has_filter:
        if verbose:
            print("🔍 第1步：筛选患者队列...")
        
        patient_ids = filter_patients(
            database=database,
            data_path=data_path,
            age_min=age_min, age_max=age_max,
            first_icu_stay=first_icu_stay,
            los_min=los_min, los_max=los_max,
            gender=gender, survived=survived,
            has_sepsis=has_sepsis,
            verbose=verbose
        )
        
        if verbose:
            print(f"   ✓ 筛选到 {len(patient_ids)} 名患者")
        
        if len(patient_ids) == 0:
            if verbose:
                print("   ❌ 没有符合条件的患者")
            return pd.DataFrame()
    else:
        patient_ids = None
    
    # 第2步：加载概念数据
    if verbose:
        print("📊 第2步：加载概念数据...")
    
    return load_concepts(
        concepts=concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        aggregate=aggregate,
        keep_components=keep_components,
        verbose=verbose,
        **kwargs
    )


def get_cohort_comparison(
    patient_ids: Optional[List[int]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    group_by: str = 'survived',
    custom_groups: Optional[Dict[str, List[int]]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    获取患者队列的分组对比统计
    
    可以按以下维度进行分组对比：
    - survived: 存活 vs 死亡
    - gender: 男性 vs 女性
    - first_icu_stay: 首次入ICU vs 再入ICU
    - 或提供自定义分组
    
    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型
        data_path: 数据路径
        group_by: 分组依据 ('survived', 'gender', 'first_icu_stay')
        custom_groups: 自定义分组 {组名: [患者ID列表]}
        verbose: 显示详细信息
    
    Returns:
        分组统计DataFrame
    
    Examples:
        >>> # 按存活状态对比
        >>> comparison = get_cohort_comparison(
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     group_by='survived'
        ... )
        >>> print(comparison)
        >>>
        >>> # 自定义分组对比（Sepsis vs 非Sepsis）
        >>> sepsis_ids = filter_patients(has_sepsis=True, ...)
        >>> non_sepsis_ids = filter_patients(has_sepsis=False, ...)
        >>> comparison = get_cohort_comparison(
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     custom_groups={'Sepsis': sepsis_ids, '非Sepsis': non_sepsis_ids}
        ... )
    """
    from .patient_filter import PatientFilter
    
    # 自动检测
    if database is None:
        database = detect_database_type(data_path)
    if data_path is None:
        data_path = get_default_data_path(database)
    
    pf = PatientFilter(database=database, data_path=data_path, verbose=verbose)
    
    # 如果提供了patient_ids，先筛选
    if patient_ids is not None:
        pf.filter(return_dataframe=True)  # 加载数据
        pf._last_result = pf._last_result[pf._last_result['patient_id'].isin(patient_ids)]
    else:
        pf.filter(return_dataframe=True)  # 加载所有患者
    
    return pf.get_cohort_comparison(group_by=group_by, custom_groups=custom_groups)


def get_cohort_stats(
    patient_ids: List[int],
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    获取患者队列的统计摘要
    
    Args:
        patient_ids: 患者ID列表
        database: 数据库类型
        data_path: 数据路径
    
    Returns:
        统计信息字典
    
    Examples:
        >>> ids = filter_patients(age_min=18, first_icu_stay=True, ...)
        >>> stats = get_cohort_stats(ids, database='miiv', data_path='/path/to/data')
        >>> print(f"患者数: {stats['患者数']}")
        >>> print(f"年龄: {stats['年龄']['均值']} ± {stats['年龄']['标准差']}")
    """
    from .patient_filter import get_cohort_stats as _get_cohort_stats
    
    if database is None:
        database = detect_database_type(data_path)
    if data_path is None:
        data_path = get_default_data_path(database)
    
    return _get_cohort_stats(patient_ids, database=database, data_path=data_path)


# =============================================================================
# 工具函数导出 - 供 webapp 和外部使用
# =============================================================================

def get_id_col_for_database(database: str) -> str:
    """获取指定数据库的患者ID列名
    
    Args:
        database: 数据库类型 ('miiv', 'eicu', 'aumc', 'hirid' 等)
    
    Returns:
        ID 列名，如 'stay_id', 'patientunitstayid' 等
    
    Examples:
        >>> get_id_col_for_database('miiv')
        'stay_id'
        >>> get_id_col_for_database('eicu')
        'patientunitstayid'
    """
    config = DATABASE_ID_CONFIG.get(database, DATABASE_ID_CONFIG['miiv'])
    return config['id_col']


def get_patient_table_for_database(database: str) -> str:
    """获取指定数据库的患者表名
    
    Args:
        database: 数据库类型
    
    Returns:
        表名，如 'icustays', 'patient', 'admissions' 等
    """
    config = DATABASE_ID_CONFIG.get(database, DATABASE_ID_CONFIG['miiv'])
    return config['table']


def get_all_patient_ids(
    data_path: Union[str, Path],
    database: Optional[str] = None,
    max_patients: Optional[int] = None,
) -> tuple:
    """获取数据库中所有（或部分）患者ID
    
    这是统一的患者ID获取接口，供 webapp 和其他模块使用。
    
    Args:
        data_path: 数据路径
        database: 数据库类型（可自动检测）
        max_patients: 限制返回的患者数量（None = 全部）
    
    Returns:
        (patient_ids_list, id_column_name)
    
    Examples:
        >>> ids, id_col = get_all_patient_ids('/path/to/miiv')
        >>> print(f"共 {len(ids)} 个患者, ID列: {id_col}")
    """
    if database is None:
        database = detect_database_type(data_path)
    
    id_col = get_id_col_for_database(database)
    table_name = get_patient_table_for_database(database)
    
    data_path = Path(data_path)

    # 患者表的备用文件名（部分数据库导出的 parquet 文件名与表名不同）
    _table_filename_aliases = {
        'general': ['general', 'general_table'],  # HiRID
    }
    _name_candidates = _table_filename_aliases.get(table_name, [table_name])

    # 尝试加载患者表
    try:
        all_ids = None
        # 首选：直接定位患者表 parquet（含 icu/ hosp/ 等子目录）。
        # 患者表都很小（icustays/patient/cases 仅几 MB），列裁剪后秒级读取；
        # 找不到就退回慢速 BaseICULoader 全库扫描——在慢速挂载上后者要十几分钟。
        parquet_file = None
        for _name in _name_candidates:
            flat = data_path / f'{_name}.parquet'
            if flat.exists():
                parquet_file = flat
                break
            # 一级子目录搜索（MIMIC-IV 的表位于 icu/ 或 hosp/ 下）
            for sub in sorted(data_path.glob(f'*/{_name}.parquet')):
                parquet_file = sub
                break
            if parquet_file is not None:
                break
        if parquet_file is not None and parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file, columns=[id_col])
                all_ids = df[id_col].dropna().unique().tolist()
            except Exception as e:
                logger.warning(f"读取患者表 parquet 失败，尝试 CSV 回退: {e}")

        if all_ids is None:
            for suffix in ('.csv', '.csv.gz'):
                csv_file = data_path / f'{table_name}{suffix}'
                if not csv_file.exists():
                    continue
                try:
                    df = pd.read_csv(csv_file, usecols=[id_col])
                    all_ids = df[id_col].dropna().unique().tolist()
                    break
                except Exception as e:
                    logger.warning(f"读取患者表 CSV 回退失败 ({csv_file.name}): {e}")

        if all_ids is None:
            # 备选：尝试分片目录
            shard_dir = data_path / table_name
            if shard_dir.exists() and shard_dir.is_dir():
                all_ids = []
                for sf in sorted(shard_dir.glob('*.parquet')):
                    shard_df = pd.read_parquet(sf, columns=[id_col])
                    all_ids.extend(shard_df[id_col].dropna().unique().tolist())
                all_ids = list(set(all_ids))
            else:
                # 最后尝试使用 BaseICULoader
                loader = BaseICULoader(database=database, data_path=data_path, verbose=False)
                sampled = _sample_patient_ids(loader, max_patients or 999999999, verbose=False)
                return (sampled or [], id_col)
        
        # 限制患者数量
        if max_patients and len(all_ids) > max_patients:
            all_ids = all_ids[:max_patients]
        
        return (all_ids, id_col)
    
    except Exception as e:
        logger.warning(f"获取患者ID失败: {e}")
        return ([], id_col)


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


# ============================================================================
# 全库提取 API — 按模块子进程隔离，16GB 安全
# ============================================================================

# Module definitions are derived from the shared web/export catalog so the
# public extract_database() API cannot drift from the 19-module full export.
EXTRACT_MODULES: Dict[str, List[str]] = {
    module: list(concepts)
    for module, concepts in CONCEPT_GROUPS_INTERNAL.items()
}

# Fast-to-slow preferred order. Unknown future modules are appended below.
_PREFERRED_EXTRACT_MODULE_ORDER: List[str] = [
    'vitals', 'demographics', 'outcome',
    'blood_gas', 'chemistry', 'hematology',
    'ventilator', 'respiratory', 'vasopressors',
    'medications', 'neurological', 'renal',
    'circulatory', 'other_scores', 'sepsis_shared',
    'sofa1_score', 'sofa2_score',
    'sepsis3_sofa1', 'sepsis3_sofa2',
]
EXTRACT_MODULE_ORDER: List[str] = [
    module for module in _PREFERRED_EXTRACT_MODULE_ORDER
    if module in EXTRACT_MODULES
] + [
    module for module in EXTRACT_MODULES
    if module not in _PREFERRED_EXTRACT_MODULE_ORDER
]

# 特殊概念 — 需要专用加载函数而非 load_concepts
_SPECIAL_CONCEPT_MODULES = {'sepsis3_sofa1', 'sepsis3_sofa2'}

# 已知数据库路径映射（可被 data_paths 参数或环境变量 EASYICU_DATA_PATH 覆盖）
# 默认使用环境变量中的数据根目录
_DEFAULT_DB_PATH_CACHE: Dict[str, str] = {}

def _get_default_db_path(database: str) -> Optional[str]:
    """惰性解析单个数据库的默认路径（按需，带缓存）。

    旧实现在 import api.py 时就为全部 6 个库递归扫描目录。
    在慢速 FUSE 挂载上，每个 os.listdir 要数秒，且每个提取子进程
    import 时都重复付出这笔开销。改为按需解析、只扫描真正用到的库。
    """
    if database in _DEFAULT_DB_PATH_CACHE:
        return _DEFAULT_DB_PATH_CACHE[database]
    _root = os.environ.get('EASYICU_DATA_PATH', '')
    if not _root:
        return None
    try:
        from easyicu.io.data_paths import find_database_path
        path = find_database_path(_root, database)
    except ImportError:
        path = os.path.join(_root, database)
    _DEFAULT_DB_PATH_CACHE[database] = path
    return path

def _build_default_db_paths() -> Dict[str, str]:
    """解析全部 6 个数据库的默认路径（仅 extract_all_databases 使用）。"""
    return {
        db: p
        for db in ['sic', 'aumc', 'hirid', 'mimic', 'miiv', 'eicu']
        if (p := _get_default_db_path(db)) is not None
    }


# 特殊模块（Sepsis-3）在分组临时目录下的输出子目录名
_SPECIAL_OUTPUT_DIRNAME = '_special'


def _extract_worker_env_setup(data_path: str) -> None:
    """提取子进程入口的共享环境准备。

    本 worker 已是隔离子进程：模块退出后 OS 完整回收内存，模块间无碎片累积。
    因此模块内部应一次性 in-process 加载，绝不要让 load_concepts 再启动“每批
    子进程 fork”——每次 fork 都会重读共享源表(chartevents/labevents…)，是数倍
    慢的根源。强制 in-process，让模块内单次扫表。
    """
    import os, sys
    os.environ.setdefault('EASYICU_DATA_PATH', data_path)
    os.environ.setdefault('EASYICU_FORCE_INPROCESS_BATCH', '1')
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Concept-bounds enforcement (physiological plausibility clamp)
# ─────────────────────────────────────────────────────────────────────────────
# R ricu applies `clamp_var` (out-of-range raw value → NA) BEFORE hourly
# aggregation, then `filter_bounds` after. EasyICU's DuckDB aggregation path
# (`load_bucketed_table_aggregated` / `_multi_aggregated` / `_wide_aggregated` in
# datasource.py) deliberately SKIPS the raw min/max WHERE-filter whenever a
# `value_transform` or an inline unit-convert is present (the raw column may be a
# different unit or VARCHAR — see datasource.py L3040-3049, 3108-3110), and the
# "post-agg filter_bounds handled in concept.py" step those comments defer to was
# never implemented (there is no concept.py and no filter_bounds anywhere in the
# package). `_filter_concept_data` (load_concepts.py:1142) enforces min/max but is
# only reached by the deprecated interactive loader, NOT the batch-export path.
# Net effect: declared concept `min`/`max` in concept-dict.json are NOT enforced
# for numeric concepts in `extract_database`, so gross source errors survive into
# the export (observed in mimiciv: hr 1e7, map 9e6, sbp 1e6, resp 7e6, spo2 9.9e6,
# peep 8.77e6, glu 1.28e6, wbc 1e6, lact 1.28e6). This is the single
# post-aggregation enforcement point for the LONG per-concept (`merge=False`)
# export: for each extracted concept it drops rows whose (post-conversion,
# target-unit) value lies outside the concept's declared [min, max]. NaN/missing
# and categorical (text-only) values are preserved. Idempotent — a no-op on data
# that is already within bounds.
_CONCEPT_BOUNDS_CACHE = None


_BOUNDS_METADATA_KEYS = (
    "rows_before",
    "bounds_dropped",
    "bounds_dropped_post_aggregation",
    "bounds_count_status",
    "bounds_raw_transformed_non_null",
    "bounds_bounded_transformed_non_null",
    "bounds_bounded_aggregate_non_null",
    "bounds_unit_suspect",
    "bounds_unbounded_retry",
    "bounds_skipped",
    "bounds_status",
)
def _load_concept_bounds_map():
    """Return ``{concept_name: (min, max)}`` from the active concept dictionary.

    Only concepts with at least one finite declared bound are included. Bounds are
    in the concept's declared (target) unit, matching the post-conversion value the
    aggregation path produces. Cached after first load.
    """
    global _CONCEPT_BOUNDS_CACHE
    if _CONCEPT_BOUNDS_CACHE is not None:
        return _CONCEPT_BOUNDS_CACHE
    import os as _os
    import json as _json
    bounds = {}
    data_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data")
    dict_paths = [
        _os.path.join(data_dir, "concept-dict.json"),
        _os.path.join(data_dir, "sofa2-dict.json"),
    ]
    try:
        for dict_path in dict_paths:
            with open(dict_path) as _f:
                _d = _json.load(_f)
            for _name, _entry in _d.items():
                if not isinstance(_entry, dict):
                    continue
                _mn = _entry.get("min")
                _mx = _entry.get("max")
                _mn = float(_mn) if _mn is not None else None
                _mx = float(_mx) if _mx is not None else None
                if _mn is not None or _mx is not None:
                    bounds[_name] = (_mn, _mx)
    except Exception as exc:
        import warnings as _warnings

        _warnings.warn(
            f"Could not load concept bounds from {dict_path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        bounds = {}
    _CONCEPT_BOUNDS_CACHE = bounds
    return bounds


def _bounds_metadata_from_manifest_info(info):
    """Subset persisted concept-bound audit fields from a worker manifest entry."""
    if not isinstance(info, dict):
        return {}
    return {k: info[k] for k in _BOUNDS_METADATA_KEYS if k in info}


def _attach_bounds_metadata(df, info):
    """Attach bounds audit metadata to an in-memory concept DataFrame."""
    meta = _bounds_metadata_from_manifest_info(info)
    if meta and hasattr(df, "attrs"):
        df.attrs["easyicu_bounds"] = meta
        for key, value in meta.items():
            df.attrs[f"easyicu_{key}"] = value
    return meta


def _concept_result_info(path, info):
    """Build the public output-dir concept entry, preserving bounds audit data."""
    out = {"path": path, "rows": info.get("rows", 0)}
    out.update(_bounds_metadata_from_manifest_info(info))
    return out
def _enforce_concept_bounds(df, concept_name):
    """Drop rows whose numeric value for ``concept_name`` is outside its declared
    [min, max]. The per-concept extraction DataFrame holds the value in a column
    named after the concept. NaN/missing and non-numeric (categorical) values are
    preserved. Returns ``(df, n_dropped)``.
    """
    import pandas as _pd
    if not isinstance(df, _pd.DataFrame) or concept_name not in df.columns:
        return df, 0
    bnd = _load_concept_bounds_map().get(concept_name)
    if bnd is None:
        return df, 0
    loader_diagnostics = df.attrs.get("easyicu_bounds_loader", {})
    if isinstance(loader_diagnostics, dict) and loader_diagnostics.get(
        "bounds_unit_suspect"
    ):
        # The SQL fast path saw at least 100 transformed non-null values but
        # none within the declared bounds. It already retried without bounds,
        # so retain those recovered values and surface the existing -1 signal.
        return df, -1
    mn, mx = bnd
    v = _pd.to_numeric(df[concept_name], errors='coerce')
    numeric = v.notna()
    # UNIT-SAFETY GUARD: if a concept has BOTH bounds and its central value (median)
    # falls outside [min,max], the values are almost certainly in the wrong unit for
    # this database (e.g. temperature still in Fahrenheit, median ~98 vs bounds
    # [32,42]). Bound-dropping would then delete valid-but-mis-united data, so SKIP
    # enforcement and leave the concept untouched (surfaced upstream as a WARN). A
    # correctly-united physiological concept always has its median well within bounds,
    # so this never suppresses legitimate outlier removal. Requires enough data to
    # make the median meaningful.
    if mn is not None and mx is not None and int(numeric.sum()) >= 100:
        med = float(v[numeric].median())
        if med < mn or med > mx:
            return df, -1  # sentinel: enforcement SKIPPED (unit-suspect), nothing dropped
    in_range = _pd.Series(True, index=df.index)
    if mn is not None:
        in_range &= (v >= mn)
    if mx is not None:
        in_range &= (v <= mx)
    # keep non-numeric/missing rows (NaN is "missing", not "out of range") and
    # numeric rows that are within [min, max]; drop only genuine out-of-range values.
    keep = (~numeric) | in_range
    n_drop = int((~keep).sum())
    if n_drop == 0:
        return df, 0
    return df.loc[keep].reset_index(drop=True), n_drop


def _run_module_extraction(
    module_name: str,
    concepts: List[str],
    database: str,
    data_path: str,
    patient_ids_filter: Optional[Dict],
    batch_size: Optional[int],
    output_dir: str,
    use_sofa2: bool = False,
    loader=None,
) -> None:
    """加载一个模块的所有概念并写入 parquet + _manifest.json。

    在 worker 子进程内运行；``loader`` 由分组 worker 传入，用于 OOM 降级
    重试前先清掉共享缓存释放内存。
    """
    import os, json, time, traceback
    import pandas as pd
    from easyicu import load_concepts as _lc

    t0 = time.time()
    saved = {}
    errors = []

    # 构造 load_concepts 参数
    # use_sofa2 显式传入：分组模式下保持全组 loader 配置一致，
    # 避免 sofa2 自动检测切换字典时重建 loader、丢掉组内共享缓存。
    warnings = []
    kwargs = dict(
        data_path=data_path, database=database,
        concepts=concepts, verbose=False, merge=True,
        concept_workers=1, use_sofa2=use_sofa2,
    )
    if patient_ids_filter:
        kwargs['patient_ids'] = patient_ids_filter

    # ── 一个模块一次 load、合并成一个宽表、写一个 {module}.parquet（不重复 io）──
    # load_concepts 一次拿到该模块**所有概念**（chartevents/labevents 等共享源表只扫
    # 一次；内部若按患者分批也由它自己 concat，对外仍是一次调用、一次扫描）。
    #
    # 分批策略：**除超大队列外一律一次性**。只有患者数 > ONESHOT_MAX_PATIENTS（15万，
    # 实际只有 eICU ~20万命中）才让 auto_batch_size 以 ≤ MAX_EXTRACT_CHUNKS（默认 3）份
    # 启用。实测最重非 eICU 模块 miiv medications（49 概念 × 9.4万患者）merge=True 一次性
    # 峰值仅 5.44GB，远低于预算；旧内存估算器约 3-5× 高估会把这类模块误判成要分批（见
    # web 端 dataio.py:1657 的同款观察），故对 ≤15万 的库直接跳过估算、强制一次性。
    ONESHOT_MAX_PATIENTS = 150_000
    _n_ids = 0
    if patient_ids_filter:
        try:
            _n_ids = len(next(iter(patient_ids_filter.values())))
        except Exception:
            _n_ids = 0
    if _n_ids > ONESHOT_MAX_PATIENTS and (not batch_size or batch_size >= _n_ids):
        try:
            from easyicu.runtime.memory_manager import auto_batch_size as _auto_bs
            # 稳定预算：用物理总内存判定（而非波动的当前可用），避免后台程序临时吃内存
            # 把本可一次性的模块误判成分批。EASYICU_ONESHOT_BUDGET_MB 可覆盖此上限(MB)。
            _stable_avail_mb = None
            _env_budget = os.environ.get('EASYICU_ONESHOT_BUDGET_MB')
            if _env_budget:
                _stable_avail_mb = float(_env_budget) / 0.6
            else:
                try:
                    import psutil as _ps
                    _stable_avail_mb = _ps.virtual_memory().total / (1024 * 1024)
                except Exception:
                    _stable_avail_mb = None
            _safe_bs = _auto_bs(list(concepts), database, _n_ids,
                                available_memory_mb=_stable_avail_mb)
            if _safe_bs and _safe_bs < _n_ids:
                batch_size = _safe_bs
        except Exception:
            pass

    if batch_size:
        kwargs['batch_size'] = batch_size

    try:
        result = _lc(**kwargs)
    except MemoryError:
        traceback.print_exc()
        if loader is not None:
            try:
                loader.concept_resolver.clear_table_cache()
            except Exception:
                pass
        _n = 0
        try:
            _n = len(next(iter(patient_ids_filter.values()))) if patient_ids_filter else 0
        except Exception:
            _n = 0
        from easyicu.runtime.memory_manager import (
            MAX_EXTRACT_CHUNKS as _MAX_CH,
            _ceil_div as _cdiv,
        )
        fallback_bs = max(10000, _cdiv(_n, _MAX_CH)) if _n else 10000
        errors.append(
            f"{module_name}: one-shot OOM, retrying batched (batch_size={fallback_bs})"
        )
        kwargs['batch_size'] = fallback_bs
        try:
            result = _lc(**kwargs)
        except Exception as e:
            traceback.print_exc()
            errors.append(f"load_concepts({module_name}) batched: {e}")
            result = {}
    except Exception as e:
        traceback.print_exc()
        errors.append(f"load_concepts({module_name}): {e}")
        result = {}

    # 写出：load_concepts(merge=True) 直接返回该模块宽表（id + time + 每概念一列），
    # 与 web 端(dataio.py)完全一致的成熟路径。**不再自造合并**——避免 endtime 列冲突、
    # 递归概念(oxygenation_index/adv_resp/ecmo…)一次性 load 爆内存、以及把含 numpy 的
    # 逐概念元数据塞进 manifest 导致 json.dump 崩溃等"手写合并"问题。生理边界在
    # load_concepts 内部按 filter_bounds 预聚合强制（与 web 端同一套）。
    if isinstance(result, pd.DataFrame) and len(result) > 0:
        try:
            _cols = [c for c in concepts if c in result.columns]
            # parquet 写盘前净化 object 混合列：指示类概念(如 mech_circ_support)可能返回
            # bool/float 混合 → object dtype，pyarrow 写 parquet 会报类型冲突。仅对"所有
            # 非空值都能无损转成数值"的 object 列转数值(bool→0/1)；纯字符串列(如 sex/
            # vent_mode)保持不动（str+None pyarrow 可正常写）。
            for _oc in result.columns:
                if result[_oc].dtype == object:
                    _num = pd.to_numeric(result[_oc], errors="coerce")
                    if bool((_num.notna() | result[_oc].isna()).all()):
                        result[_oc] = _num
            path = os.path.join(output_dir, f"{module_name}.parquet")
            result.to_parquet(path, index=False, engine="pyarrow")
            saved[module_name] = {
                "path": path,
                "rows": len(result),
                "concepts": _cols,
            }
        except Exception as e:
            traceback.print_exc()
            errors.append(f"write({module_name}): {e}")
    elif isinstance(result, dict) and result:
        # merge=True 应始终返回 DataFrame；若意外返回 dict，大声记错而不静默丢数据。
        errors.append(
            f"{module_name}: merge=True returned a dict ({len(result)} concepts) unexpectedly; not written"
        )

    elapsed = time.time() - t0
    manifest = {
        "module": module_name,
        "saved": saved,
        "errors": errors,
        "warnings": warnings,
        "elapsed_sec": round(elapsed, 1),
    }
    with open(os.path.join(output_dir, '_manifest.json'), 'w') as f:
        json.dump(manifest, f)


def _extract_module_worker(
    concepts: List[str],
    database: str,
    data_path: str,
    patient_ids_filter: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    output_dir: str = '',
    module_name: str = '',
):
    """（兼容包装）单模块子进程入口。

    新的默认入口是 _extract_module_group_worker（组内共享源表扫描）；
    保留此包装以兼容仍按单模块 spawn 的旧调用方。
    """
    _extract_worker_env_setup(data_path)
    _run_module_extraction(
        module_name, concepts, database, data_path,
        patient_ids_filter, batch_size, output_dir,
    )


def _run_special_extraction(
    special_modules: List[str],
    database: str,
    data_path: str,
    patient_ids_filter: Optional[Dict],
    batch_size: Optional[int],
    output_dir: str,
    use_sofa2: bool = False,
) -> None:
    """加载特殊概念（Sepsis-3 等）并写入 parquet + _manifest.json。

    sep3_sofa1/sep3_sofa2 不在 concept-dict 中，需要先加载 susp_inf + sofa/sofa2，
    然后通过 _load_sep3_diagnosis 逻辑计算 Sepsis-3 诊断。分组模式下与
    sofa1_score/sofa2_score 同进程运行，susp_inf/sofa/sofa2 直接命中组内缓存。
    """
    import os, json, time, traceback
    import pandas as pd
    from easyicu import load_concepts as _lc

    t0 = time.time()
    saved = {}
    errors = []

    # 构建公共加载参数（use_sofa2 显式传入以保持组内 loader 配置一致）
    load_kw = dict(data_path=data_path, database=database, verbose=False, merge=True,
                   use_sofa2=use_sofa2)
    if patient_ids_filter:
        load_kw['patient_ids'] = patient_ids_filter
    if batch_size:
        load_kw['batch_size'] = batch_size

    # 收集需要的概念: sep3_sofa1 需要 sofa, sep3_sofa2 需要 sofa2
    need_sofa1 = any('sep3_sofa1' in EXTRACT_MODULES.get(m, []) for m in special_modules)
    need_sofa2 = any('sep3_sofa2' in EXTRACT_MODULES.get(m, []) for m in special_modules)

    deps = ['susp_inf']
    if need_sofa1:
        deps.append('sofa')
    if need_sofa2:
        deps.append('sofa2')

    try:
        merged = _lc(concepts=deps, **load_kw)
    except Exception:
        # sofa2 可能不可用，回退到仅 sofa
        try:
            merged = _lc(concepts=['susp_inf', 'sofa'], **load_kw)
            need_sofa2 = False
        except Exception as e:
            traceback.print_exc()
            errors.append(f"Failed to load dependencies {deps}: {e}")
            merged = pd.DataFrame()

    if isinstance(merged, pd.DataFrame) and not merged.empty:
        # 检测 ID 和时间列
        id_col = next((c for c in ['stay_id', 'patientunitstayid', 'admissionid',
                                    'patientid', 'icustay_id', 'CaseID']
                       if c in merged.columns), None)
        time_col = next((c for c in ['charttime', 'time', 'starttime', 'datetime',
                                      'Offset', 'measuredat_minutes', 'measuredat']
                        if c in merged.columns), None)

        if id_col and time_col and 'susp_inf' in merged.columns:
            susp = merged['susp_inf'].fillna(0).astype(bool)

            # Sepsis-3 = a >=2-point SOFA increase WITHIN the suspected-infection
            # window (delta rule, R ricu sep3), NOT an absolute SOFA>=2. Use the
            # shared sep3()/sep3_sofa2() so both labels match load_sepsis3 and the
            # module export (unified to delta 2026-06-22).
            if need_sofa1 and 'sofa' in merged.columns:
                from .scores.sepsis import sep3 as _sep3
                result = _sep3(
                    merged[[id_col, time_col, 'sofa']],
                    merged[[id_col, time_col, 'susp_inf']],
                    id_cols=[id_col], index_col=time_col,
                ).rename(columns={'sep3': 'sep3_sofa1'})
                if 'sep3_sofa1' in result.columns:
                    result['sep3_sofa1'] = result['sep3_sofa1'].fillna(0).astype(int)
                if len(result) > 0:
                    path = os.path.join(output_dir, 'sep3_sofa1.parquet')
                    result.to_parquet(path, index=False, engine='pyarrow')
                    saved['sep3_sofa1'] = {'path': path, 'rows': len(result)}

            if need_sofa2 and 'sofa2' in merged.columns:
                from .scores.sepsis_sofa2 import sep3_sofa2 as _sep3_sofa2
                result = _sep3_sofa2(
                    merged[[id_col, time_col, 'sofa2']],
                    merged[[id_col, time_col, 'susp_inf']],
                    id_cols=[id_col], index_col=time_col,
                )
                if 'sep3_sofa2' in result.columns:
                    result['sep3_sofa2'] = result['sep3_sofa2'].fillna(0).astype(int)
                if len(result) > 0:
                    path = os.path.join(output_dir, 'sep3_sofa2.parquet')
                    result.to_parquet(path, index=False, engine='pyarrow')
                    saved['sep3_sofa2'] = {'path': path, 'rows': len(result)}
        else:
            missing = []
            if not id_col: missing.append('id_col')
            if not time_col: missing.append('time_col')
            if 'susp_inf' not in merged.columns: missing.append('susp_inf')
            # 🔧 FIX 2026-05-11: 对于 sic/hirid 等不支持 susp_inf 的数据库，
            # sep3_sofa1/sep3_sofa2 无法计算属正常情况，不应记为错误。
            # 只有当 id/time 列也缺失时才认为是真正的错误。
            if missing == ['susp_inf']:
                pass  # 静默跳过：数据库不支持 susp_inf，sep3 概念不适用
            else:
                errors.append(f"Missing columns: {missing}, available: {list(merged.columns)[:10]}")

    elapsed = time.time() - t0
    manifest = {
        'module': 'special_concepts',
        'saved': saved,
        'errors': errors,
        'elapsed_sec': round(elapsed, 1),
    }
    with open(os.path.join(output_dir, '_manifest.json'), 'w') as f:
        json.dump(manifest, f)


def _extract_special_worker(
    special_modules: List[str],
    database: str,
    data_path: str,
    patient_ids_filter: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    output_dir: str = '',
):
    """（兼容包装）特殊概念子进程入口 — 参见 _extract_module_group_worker。"""
    _extract_worker_env_setup(data_path)
    _run_special_extraction(
        special_modules, database, data_path,
        patient_ids_filter, batch_size, output_dir,
    )


def _extract_module_group_worker(
    module_specs: List[tuple],
    special_modules: List[str],
    database: str,
    data_path: str,
    patient_ids_filter: Optional[Dict],
    batch_size: Optional[int],
    output_root: str,
    use_sofa2: bool,
):
    """在一个子进程中顺序提取一组共享源表的模块。

    keep_cache 让组内模块共享 raw/table 缓存（受 EASYICU_CACHE_BUDGET_MB
    字节预算约束），chartevents/labevents 等重表每组只扫一次，而不是每
    模块重扫一遍；子进程退出后 OS 仍完整回收内存。分组因此是“缓存复用”
    与“内存隔离”之间的折中：组内复用，组间隔离。

    module_specs: [(module_name, [concepts...]), ...]，每个模块写
    ``output_root/<module_name>/``；特殊模块写 ``output_root/_special/``。
    """
    import os, traceback
    _extract_worker_env_setup(data_path)
    from easyicu.api import keep_cache as _keep_cache

    with _keep_cache(database=database, data_path=data_path, use_sofa2=use_sofa2) as _loader:
        for module_name, concepts in module_specs:
            out_dir = os.path.join(output_root, module_name)
            os.makedirs(out_dir, exist_ok=True)
            try:
                _run_module_extraction(
                    module_name, concepts, database, data_path,
                    patient_ids_filter, batch_size, out_dir,
                    use_sofa2=use_sofa2, loader=_loader,
                )
            except Exception:
                # _run_module_extraction 已内部捕获常规异常并写 manifest；
                # 这里兜底保证一个模块的意外崩溃不拖垮组内后续模块。
                traceback.print_exc()
        if special_modules:
            sp_dir = os.path.join(output_root, _SPECIAL_OUTPUT_DIRNAME)
            os.makedirs(sp_dir, exist_ok=True)
            try:
                _run_special_extraction(
                    special_modules, database, data_path,
                    patient_ids_filter, batch_size, sp_dir,
                    use_sofa2=use_sofa2,
                )
            except Exception:
                traceback.print_exc()


# 分组亲和表：同组模块共享同一批重源表（chartevents/labevents/inputevents
# 家族），或互为依赖（SOFA 闭包）。分组只影响“哪些模块共用一个子进程 +
# keep_cache”，不改变模块内容、输出布局或模块顺序语义。
_EXTRACT_MODULE_GROUP_AFFINITY: List[List[str]] = [
    # chartevents / nursecharting 家族
    ['vitals', 'neurological', 'respiratory', 'ventilator'],
    # 入科级小表（icustays/admissions/patients）
    ['demographics', 'outcome'],
    # labevents 家族
    ['blood_gas', 'chemistry', 'hematology', 'renal'],
    # inputevents / prescriptions 家族
    ['vasopressors', 'medications', 'circulatory'],
    # 评分闭包：SOFA 组件被 sofa1/sofa2 共享，sep3_* 复用 susp_inf+sofa/sofa2
    ['other_scores', 'sepsis_shared', 'sofa1_score', 'sofa2_score'],
]


def _group_modules_for_extraction(
    normal_modules: List[str],
    special_modules: List[str],
    group_modules: bool = True,
) -> List[Dict[str, List[str]]]:
    """把请求的模块划分为子进程组。

    返回 [{'modules': [...], 'special': [...]}, ...]。group_modules=False
    时退化为每模块一组（旧行为）。未出现在亲和表中的新模块各自成组。
    特殊模块（Sepsis-3）挂到评分组上（若本次请求包含评分组），使
    susp_inf/sofa/sofa2 命中组内缓存；否则单独成组。
    """
    if not group_modules:
        groups: List[Dict[str, List[str]]] = [
            {'modules': [m], 'special': []} for m in normal_modules
        ]
        if special_modules:
            groups.append({'modules': [], 'special': list(special_modules)})
        return groups

    groups = []
    assigned = set()
    for affinity in _EXTRACT_MODULE_GROUP_AFFINITY:
        members = [m for m in normal_modules if m in affinity]
        if members:
            groups.append({'modules': members, 'special': []})
            assigned.update(members)
    for m in normal_modules:
        if m not in assigned:
            groups.append({'modules': [m], 'special': []})

    if special_modules:
        target = next(
            (g for g in groups
             if any(m in ('sofa1_score', 'sofa2_score', 'sepsis_shared')
                    for m in g['modules'])),
            None,
        )
        if target is None:
            groups.append({'modules': [], 'special': list(special_modules)})
        else:
            target['special'] = list(special_modules)
    return groups


def extract_database(
    database: str,
    data_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    modules: Optional[List[str]] = None,
    patient_ids: Optional[Union[List, Dict]] = None,
    max_patients: Optional[int] = None,
    batch_size: Optional[int] = None,
    group_modules: bool = True,
    verbose: bool = True,
) -> Dict:
    """按 19 个模块分组、子进程隔离地提取整个数据库的全部特征。

    ★ 这是全量特征提取的推荐入口。 不要为了提取全量特征自己写
    `load_concepts` 循环——尤其不要按单概念或小批 patient_ids 循环，那会让
    共享源表(chartevents/labevents…)被反复重读，慢上数倍。

    工作原理与性能：
      * 概念按 19 个模块分组(EXTRACT_MODULE_ORDER)，每个模块一次性
        load_concepts(模块全部概念)，共享源表只扫一次。
      * 共享同族源表的模块进一步合并为分组(_EXTRACT_MODULE_GROUP_AFFINITY)，
        每组一个子进程、组内用 keep_cache 复用 raw/table 缓存：
        chartevents/labevents 等重表每组只扫一次，而不是每模块重扫一遍；
        SOFA 闭包只算一次并被 sofa1/sofa2/sep3_* 复用。缓存受
        EASYICU_CACHE_BUDGET_MB 字节预算约束（默认物理内存的 25%），
        8-16GB 机器安全。
      * 每组在独立子进程中运行，组退出后 OS 完整回收内存（含 pymalloc
        arena 碎片），主进程 RSS 几乎不增长。group_modules=False 或环境变量
        EASYICU_EXTRACT_GROUPING=0 退回每模块一个子进程的旧行为。
      * 模块内默认 **不分批、一次性 in-process** 加载：实测单模块峰值 RSS
        恒定 ~2-3GB(与队列规模无关)，故 16GB 机器也能对任意规模数据库一次性
        全量提取。仅当一次性确实 OOM(极小内存机器的最大队列)时，worker 自动
        降级为有界分批。
      * 参考实测：MIMIC-III 全量 61,532 stays 的 SOFA-2 六分量 ~6 分钟。

    Args:
        database: 数据库类型 ('miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic')
        data_path: 数据路径（None 则按数据库名自动解析）
        output_dir: 输出目录（None 则不写文件，仅返回 dict）
        modules: 要提取的模块列表（None = 全部 19 个模块）
        patient_ids: 患者 ID 列表或 dict（None = 全部患者）
        max_patients: 限制患者数量（与 patient_ids 互斥）
        batch_size: 模块内患者分批大小。None(默认) = 不分批，一次性 in-process
            加载(推荐，最快)。仅在极小内存机器上想强制限制峰值内存时才显式传值。
        group_modules: True(默认) = 共享源表的模块合并为分组子进程并复用
            keep_cache 缓存；False = 每模块一个子进程（旧行为）。
        verbose: 是否打印进度

    Returns:
        dict: {
            'database': str,
            'num_patients': int,
            'modules': {module_name: {'concepts': {name: DataFrame}, 'elapsed': float, 'errors': list}},
            'total_elapsed': float,
            'output_dir': str or None,
        }

    Examples:
        >>> # 提取 AUMC 全部特征到目录
        >>> result = extract_database('aumc', output_dir='/tmp/aumc_export')
        >>> print(f"共 {result['num_patients']} 患者, {result['total_elapsed']:.0f}s")

        >>> # 仅提取 vitals 和 demographics，返回 DataFrame
        >>> result = extract_database('miiv', modules=['vitals', 'demographics'])
        >>> hr_df = result['modules']['vitals']['concepts']['hr']
    """
    import multiprocessing as mp
    import tempfile
    import json
    import time
    import shutil

    from .runtime.memory_manager import get_rss_mb, get_available_memory_mb

    t_start = time.time()

    # 确定数据路径
    if data_path is None:
        data_path = _get_default_db_path(database)
        if data_path is None:
            data_path = get_default_data_path()
    data_path = str(data_path)

    # 磁盘溢写 / 批处理中间文件的默认落点：**输出目录旁的 .easyicu_spill/**，而不是
    # 系统临时目录（常在快满的系统盘上）。输出目录通常在用户为数据特意选的大盘上，
    # 这样零配置即安全，调用方无需每次手设 TMPDIR / EASYICU_DUCKDB_TEMP_DIR。放在
    # 最前，确保后续所有 DuckDB 连接与 fork 出的 worker 子进程都继承此设置。
    # opt-out：显式把 EASYICU_DUCKDB_TEMP_DIR 指向别处（非 .easyicu_spill）则完全尊重。
    # 多库循环：每库各自重指向本库输出旁，故用 basename 判定"是否用户自定义"。
    if output_dir is not None:
        _cur_spill = os.environ.get('EASYICU_DUCKDB_TEMP_DIR')
        _user_spill = (
            _cur_spill is not None
            and os.path.basename(os.path.normpath(_cur_spill)) != '.easyicu_spill'
        )
        if not _user_spill:
            _spill_root = os.path.join(os.path.abspath(str(output_dir)), '.easyicu_spill')
            try:
                os.makedirs(_spill_root, exist_ok=True)
                os.environ['EASYICU_DUCKDB_TEMP_DIR'] = _spill_root
                os.environ['TMPDIR'] = _spill_root
                tempfile.tempdir = _spill_root
            except Exception:
                pass

    # 获取患者 ID
    if patient_ids is None:
        all_ids, id_col = get_all_patient_ids(data_path, database, max_patients)
        if not all_ids:
            raise ValueError(f"无法获取 {database} 的患者ID，请检查 data_path: {data_path}")
        patient_ids_filter = {id_col: all_ids}
    else:
        patient_ids_filter = _normalize_patient_ids_for_db(database, patient_ids)
        id_col = list(patient_ids_filter.keys())[0]
        all_ids = list(patient_ids_filter.values())[0]

    num_patients = len(all_ids)

    # 默认 batch_size：不分批。
    # 每个模块已在独立子进程(_extract_module_worker)中运行，模块退出后 OS 完整
    # 回收内存，所以模块间不会累积碎片。实测单模块峰值 RSS 恒定 ~2-3GB(与队列
    # 规模无关，因为 load_concepts 按源表流式处理)，全量 6 个库都能一次装下。
    # 主动分批只会让 load_concepts 每批重读共享源表(chartevents/labevents…)，
    # 数倍变慢——这是用户“怎么这么慢”的根因。故默认用大于任意队列的哨兵值，
    # 让模块内单次扫表完成。仅在极端机器上由用户显式传 batch_size 覆盖。
    if batch_size is None:
        batch_size = max(num_patients + 1, 2_000_000)
        _auto_one_shot = True
    else:
        _auto_one_shot = False

    # 确定要提取的模块
    if modules is None:
        modules = list(EXTRACT_MODULE_ORDER)
    else:
        # 保持用户指定顺序，但验证模块名
        for m in modules:
            if m not in EXTRACT_MODULES:
                raise ValueError(f"未知模块 '{m}'，可选: {list(EXTRACT_MODULES.keys())}")

    # 创建输出目录
    if output_dir is not None:
        output_dir = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    if verbose:
        rss = get_rss_mb()
        print(f"{'='*60}")
        print(f"📊 extract_database: {database}")
        print(f"   患者数: {num_patients:,}, 模块数: {len(modules)}")
        print(f"   批策略: {'一次性 in-process (推荐)' if _auto_one_shot else f'batch_size={batch_size}'}")
        print(f"   RSS: {rss:.0f}MB, 输出: {output_dir or '仅内存'}")
        print(f"{'='*60}")

    result = {
        'database': database,
        'num_patients': num_patients,
        'modules': {},
        'total_elapsed': 0,
        'output_dir': output_dir,
    }

    # 分离普通模块和特殊模块
    normal_modules = [m for m in modules if m not in _SPECIAL_CONCEPT_MODULES]
    special_modules = [m for m in modules if m in _SPECIAL_CONCEPT_MODULES]

    mp_ctx = mp.get_context('fork' if os.name != 'nt' else 'spawn')

    # ---- 模块分组：组内共享源表扫描（keep_cache），组间子进程隔离 ----
    group_flag = group_modules
    _env_grouping = os.environ.get('EASYICU_EXTRACT_GROUPING', '').strip().lower()
    if _env_grouping in ('0', 'off', 'false', 'no'):
        group_flag = False

    groups = _group_modules_for_extraction(normal_modules, special_modules, group_flag)

    if verbose and group_flag:
        print(f"   分组: {len(groups)} 组（组内共享源表扫描；"
              f"EASYICU_EXTRACT_GROUPING=0 或 group_modules=False 关闭）")

    n_units_total = len(normal_modules) + len(special_modules)
    units_done = 0

    def _collect_module_result(tmp_mod_dir: str, mod_name: str) -> Dict:
        """读回单个模块 worker 的 manifest + parquet 输出。"""
        mod_result = {
            "concepts": {},
            "elapsed": 0.0,
            "errors": [],
            "warnings": [],
            "bounds": {},
        }
        manifest_path = os.path.join(tmp_mod_dir, '_manifest.json')
        if not os.path.exists(manifest_path):
            mod_result['errors'] = [
                f"{mod_name}: worker produced no manifest (process may have died)"
            ]
            return mod_result
        with open(manifest_path) as f:
            manifest = json.load(f)
        mod_result['errors'] = manifest.get('errors', [])
        mod_result["warnings"] = manifest.get("warnings", [])
        mod_result['elapsed'] = manifest.get('elapsed_sec', 0.0)
        output_manifest = {
            "module": mod_name,
            "saved": {},
            "errors": mod_result["errors"],
            "warnings": mod_result["warnings"],
            "bounds": mod_result["bounds"],
            "elapsed_sec": mod_result["elapsed"],
        }
        # 每个模块一个宽表 parquet：manifest["saved"] 只有一条（键=模块名），
        # info 里带 concepts（列名清单）+ concept_meta（逐概念 rows/bounds provenance）。
        for _saved_key, info in manifest.get("saved", {}).items():
            pq_path = info.get("path")
            if not pq_path or not os.path.exists(pq_path):
                continue
            module_rows = info.get("rows", 0)
            concept_meta = info.get("concept_meta", {}) or {}
            concept_names = info.get("concepts") or list(concept_meta.keys())
            # 逐概念 bounds 元数据（provenance）
            for cn, cmeta in concept_meta.items():
                bmeta = _bounds_metadata_from_manifest_info(cmeta)
                if bmeta:
                    mod_result["bounds"][cn] = bmeta
            if output_dir is not None:
                # flat：一个模块一个文件 output_dir/{module}.parquet（不重复 io）
                os.makedirs(output_dir, exist_ok=True)
                dst = os.path.join(output_dir, f"{mod_name}.parquet")
                shutil.move(pq_path, dst)
                module_info = {
                    "path": dst,
                    "rows": module_rows,
                    "concepts": concept_names,
                    "merge_keys": info.get("merge_keys", []),
                    "concept_meta": concept_meta,
                }
                output_manifest["saved"][mod_name] = module_info
                # 逐概念一条（path 都指向该模块宽表），供 summary CSV 保留每概念行数。
                for cn in concept_names:
                    cmeta = concept_meta.get(cn, {})
                    concept_info = {"path": dst, "rows": cmeta.get("rows", module_rows)}
                    for k, v in cmeta.items():
                        if k != "rows":
                            concept_info[k] = v
                    mod_result["concepts"][cn] = concept_info
            else:
                # 无输出目录：读回宽表 DataFrame 到内存（键=模块名）
                mod_result["concepts"][mod_name] = pd.read_parquet(pq_path)
        if output_dir is not None:
            with open(os.path.join(output_dir, f"{mod_name}.manifest.json"), "w") as f:
                json.dump(output_manifest, f)
        return mod_result

    def _count_rows(mod_result: Dict) -> int:
        n_rows = 0
        for v in mod_result['concepts'].values():
            if isinstance(v, dict):
                n_rows += v.get('rows', 0)
            elif isinstance(v, pd.DataFrame):
                n_rows += len(v)
        return n_rows

    def _collect_special_results(tmp_sp_dir: str, sp_modules: List[str]) -> None:
        """读回特殊模块（Sepsis-3）worker 输出到 result['modules']。"""
        nonlocal units_done
        manifest = None
        manifest_path = os.path.join(tmp_sp_dir, '_manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
        sp_elapsed = (manifest or {}).get('elapsed_sec', 0.0)
        for mod_name in sp_modules:
            concepts = EXTRACT_MODULES.get(mod_name, [])
            if manifest is None:
                mod_result = {
                    "concepts": {},
                    "elapsed": 0.0,
                    "errors": [
                        f"{mod_name}: worker produced no manifest (process may have died)"
                    ],
                    "warnings": [],
                    "bounds": {},
                }
            else:
                mod_result = {
                    "concepts": {},
                    "elapsed": sp_elapsed,
                    "errors": manifest.get("errors", []),
                    "warnings": manifest.get("warnings", []),
                    "bounds": {},
                }
                output_manifest = {
                    "module": mod_name,
                    "saved": {},
                    "errors": mod_result["errors"],
                    "warnings": mod_result["warnings"],
                    "bounds": mod_result["bounds"],
                    "elapsed_sec": sp_elapsed,
                }
                for c_name in concepts:
                    info = manifest.get("saved", {}).get(c_name)
                    if info and os.path.exists(info["path"]):
                        rows = info.get("rows", 0)
                        meta = _bounds_metadata_from_manifest_info(info)
                        if meta:
                            mod_result["bounds"][c_name] = meta
                        if output_dir is not None:
                            # flat：派生模块（sepsis3_*）每模块单概念，与普通模块
                            # 统一写 output_dir/{module}.parquet，不再嵌套
                            # {module}/{concept}.parquet（否则 17 扁平 + 2 嵌套的
                            # 混合布局违反"每模块一个宽表"契约）。
                            os.makedirs(output_dir, exist_ok=True)
                            dst = os.path.join(output_dir, f"{mod_name}.parquet")
                            shutil.move(info["path"], dst)
                            concept_info = _concept_result_info(dst, info)
                            concept_info["rows"] = rows
                            mod_result["concepts"][c_name] = concept_info
                            output_manifest["saved"][c_name] = concept_info
                        else:
                            df = pd.read_parquet(info["path"])
                            _attach_bounds_metadata(df, info)
                            mod_result["concepts"][c_name] = df
                if output_dir is not None:
                    with open(os.path.join(output_dir, f"{mod_name}.manifest.json"), "w") as f:
                        json.dump(output_manifest, f)
            result["modules"][mod_name] = mod_result
            units_done += 1
            if verbose:
                print(
                    f"   {'✅' if not mod_result['errors'] else '⚠️'} "
                    f"[{units_done}/{n_units_total}] {mod_name}: "
                    f"{len(mod_result['concepts'])} concepts, "
                    f"{_count_rows(mod_result):,} rows, {sp_elapsed:.1f}s"
                )

    # ---- 逐组在子进程中加载 ----
    from collections import deque
    pending_groups = deque(groups)
    while pending_groups:
        group = pending_groups.popleft()
        group_mods = [m for m in group["modules"] if EXTRACT_MODULES.get(m)]
        group_special = list(group["special"])
        if not group_mods and not group_special:
            continue

        module_specs = [(m, EXTRACT_MODULES[m]) for m in group_mods]
        group_use_sofa2 = any(_concepts_need_sofa2(c) for _, c in module_specs) or any(
            "sofa2" in m for m in group_special
        )

        tmp_root = tempfile.mkdtemp(prefix="easyicu_grp_")
        if verbose:
            rss = get_rss_mb()
            label = " + ".join(group_mods + group_special)
            print(f"\n⏳ {label} ... RSS={rss:.0f}MB")

        proc = mp_ctx.Process(
            target=_extract_module_group_worker,
            args=(
                module_specs,
                group_special,
                database,
                data_path,
                patient_ids_filter,
                batch_size,
                tmp_root,
                group_use_sofa2,
            ),
            daemon=True,
        )
        proc.start()
        proc.join()

        # 组 worker 硬崩溃（如 OOM kill）：已完成模块正常读回；未完成的
        # 模块拆成单模块组重试一次，避免一个组的失败拖垮整组输出。
        crashed = proc.exitcode not in (0, None)
        incomplete_mods = [
            m
            for m in group_mods
            if not os.path.exists(os.path.join(tmp_root, m, "_manifest.json"))
        ]
        special_incomplete = bool(group_special) and not os.path.exists(
            os.path.join(tmp_root, _SPECIAL_OUTPUT_DIRNAME, "_manifest.json")
        )
        can_split = len(group_mods) + (1 if group_special else 0) > 1
        if crashed and can_split and (incomplete_mods or special_incomplete):
            if verbose:
                retry_units = incomplete_mods + (
                    group_special if special_incomplete else []
                )
                print(
                    f"   ⚠️ group worker exit={proc.exitcode}; "
                    f"retrying individually: {retry_units}"
                )
            if special_incomplete:
                pending_groups.appendleft({"modules": [], "special": group_special})
                group_special = []
            for m in reversed(incomplete_mods):
                pending_groups.appendleft({"modules": [m], "special": []})
            group_mods = [m for m in group_mods if m not in incomplete_mods]

        for mod_name in group_mods:
            mod_result = _collect_module_result(
                os.path.join(tmp_root, mod_name), mod_name
            )
            result["modules"][mod_name] = mod_result
            units_done += 1
            if verbose:
                status = "✅" if not mod_result["errors"] else "⚠️"
                print(
                    f"   {status} [{units_done}/{n_units_total}] {mod_name}: "
                    f"{len(mod_result['concepts'])} concepts, "
                    f"{_count_rows(mod_result):,} rows, {mod_result['elapsed']:.1f}s"
                    + (
                        f" | errors: {mod_result['errors']}"
                        if mod_result["errors"]
                        else ""
                    )
                    + (
                        f" | warnings: {mod_result['warnings']}"
                        if mod_result.get("warnings")
                        else ""
                    )
                )

        if group_special:
            _collect_special_results(
                os.path.join(tmp_root, _SPECIAL_OUTPUT_DIRNAME), group_special
            )

        # 清理临时目录
        shutil.rmtree(tmp_root, ignore_errors=True)

    total_elapsed = time.time() - t_start
    result['total_elapsed'] = round(total_elapsed, 1)

    if verbose:
        rss = get_rss_mb()
        total_concepts = sum(len(m["concepts"]) for m in result["modules"].values())
        total_rows = 0
        for m in result["modules"].values():
            for v in m["concepts"].values():
                if isinstance(v, dict):
                    total_rows += v.get("rows", 0)
                elif isinstance(v, pd.DataFrame):
                    total_rows += len(v)
        all_errors = [e for m in result["modules"].values() for e in m["errors"]]
        all_warnings = [
            w for m in result["modules"].values() for w in m.get("warnings", [])
        ]
        print(f"\n{'='*60}")
        print(
            f"✅ {database} 完成: {total_concepts} concepts, "
            f"{total_rows:,} rows, {total_elapsed:.1f}s"
        )
        print(
            f"   RSS: {rss:.0f}MB" + (f"  |  输出: {output_dir}" if output_dir else "")
        )
        if all_errors:
            print(f"   ⚠️ {len(all_errors)} 错误: {all_errors[:5]}")
        if all_warnings:
            print(f"   ⚠️ {len(all_warnings)} 警告: {all_warnings[:5]}")
        print(f"{'='*60}")

    return result


def extract_all_databases(
    databases: Optional[List[str]] = None,
    data_paths: Optional[Dict[str, str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    modules: Optional[List[str]] = None,
    max_patients: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """逐库逐模块子进程隔离提取所有数据库的全部特征。

    每个模块运行在独立子进程中，主进程内存几乎不增长。
    适用于 16GB 内存环境。

    Args:
        databases: 要提取的数据库列表（None = 全部 6 个: sic, aumc, hirid, mimic, miiv, eicu）
        data_paths: {database: path} 覆盖默认路径
        output_dir: 输出根目录（每个库一个子目录）
        modules: 要提取的模块列表（None = 全部）
        max_patients: 每个库的患者数量限制
        batch_size: 子进程内患者分批大小
        verbose: 是否打印进度

    Returns:
        dict: {database_name: extract_database() 返回值}

    Examples:
        >>> results = extract_all_databases(output_dir='/tmp/all_export')
        >>> for db, r in results.items():
        ...     print(f"{db}: {r['num_patients']:,} patients, {r['total_elapsed']:.0f}s")
    """
    import time

    if databases is None:
        databases = ['sic', 'aumc', 'hirid', 'mimic', 'miiv', 'eicu']

    merged_paths = _build_default_db_paths()
    if data_paths:
        merged_paths.update(data_paths)

    t_start = time.time()
    results = {}

    if verbose:
        print(f"\n{'#'*60}")
        print(f"# extract_all_databases: {len(databases)} 个数据库")
        print(f"# 模块: {modules or '全部'}")
        print(f"# 输出: {output_dir or '仅内存'}")
        print(f"{'#'*60}")

    for db_idx, db in enumerate(databases):
        dp = merged_paths.get(db)
        if dp is None:
            if verbose:
                print(f"\n⚠️ 跳过 {db}: 无数据路径")
            continue

        if not os.path.isdir(dp):
            if verbose:
                print(f"\n⚠️ 跳过 {db}: 路径不存在 {dp}")
            continue

        db_output = None
        if output_dir is not None:
            db_output = os.path.join(str(output_dir), db)

        if verbose:
            print(f"\n{'━'*60}")
            print(f"  [{db_idx+1}/{len(databases)}] 🏥 {db.upper()}")
            print(f"{'━'*60}")

        try:
            r = extract_database(
                database=db,
                data_path=dp,
                output_dir=db_output,
                modules=modules,
                max_patients=max_patients,
                batch_size=batch_size,
                verbose=verbose,
            )
            results[db] = r
        except Exception as e:
            if verbose:
                print(f"  ❌ {db} 失败: {e}")
            results[db] = {'error': str(e)}

    total = time.time() - t_start

    if verbose:
        print(f"\n{'#'*60}")
        print(f"# 全部完成: {total:.1f}s")
        for db, r in results.items():
            if 'error' in r:
                print(f"#   {db}: ❌ {r['error']}")
            else:
                nc = sum(len(m['concepts']) for m in r['modules'].values())
                nr = 0
                for m in r['modules'].values():
                    for v in m['concepts'].values():
                        if isinstance(v, dict):
                            nr += v.get('rows', 0)
                        elif hasattr(v, '__len__'):
                            nr += len(v)
                print(f"#   {db}: {r['num_patients']:,} patients, "
                      f"{nc} concepts, {nr:,} rows, {r['total_elapsed']:.0f}s")
        print(f"{'#'*60}")

    return results
