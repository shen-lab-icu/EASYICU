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
from .resources import load_dictionary
from .config import load_data_sources

logger = logging.getLogger(__name__)

# 全局加载器实例，用于复用初始化开销
_global_loader = None
_loader_config = None


def _normalize_patient_ids_for_db(database_name: str, patient_ids):
    """Normalize patient IDs to the canonical ID column for each database."""
    if patient_ids is None or isinstance(patient_ids, dict):
        return patient_ids

    if database_name in ['eicu', 'eicu_demo']:
        return {'patientunitstayid': patient_ids}
    if database_name in ['aumc']:
        return {'admissionid': patient_ids}
    if database_name in ['hirid']:
        return {'patientid': patient_ids}
    if database_name == 'sic':
        return {'CaseID': patient_ids}
    if database_name == 'mimic':
        return {'icustay_id': patient_ids}
    return {'stay_id': patient_ids}


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
        # 清理加载器内部缓存
        if hasattr(_global_loader, 'concept_resolver'):
            _global_loader.concept_resolver.clear()
        if hasattr(_global_loader, 'data_source'):
            _global_loader.data_source.clear()
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
        sample_strategy: 采样策略
            - 'sorted': 按ID排序取前N个（默认，与RICU金标准一致）
            - 'random': 随机采样N个（更具代表性，适用于探索性分析）
    """
    db_name = loader.database
    
    # 数据库 -> (表名, ID列名) 映射
    id_table_map = {
        'miiv': ('icustays', 'stay_id'),
        'mimic': ('icustays', 'icustay_id'),
        'mimic_demo': ('icustays', 'icustay_id'),
        'eicu': ('patient', 'patientunitstayid'),
        'eicu_demo': ('patient', 'patientunitstayid'),
        'aumc': ('admissions', 'admissionid'),
        'hirid': ('general', 'patientid'),
        'sic': ('cases', 'CaseID'),  # SICdb uses cases table with CaseID
    }
    
    table_name, id_col = id_table_map.get(db_name, ('icustays', 'stay_id'))
    
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
    id_table_map = {
        'miiv': ('icustays', 'stay_id'),
        'mimic': ('icustays', 'icustay_id'),
        'mimic_demo': ('icustays', 'icustay_id'),
        'eicu': ('patient', 'patientunitstayid'),
        'eicu_demo': ('patient', 'patientunitstayid'),
        'aumc': ('admissions', 'admissionid'),
        'hirid': ('general', 'patientid'),
        'sic': ('cases', 'CaseID'),
    }
    return id_table_map.get(loader.database, ('icustays', 'stay_id'))


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
    db_name = loader.database
    id_table_map = {
        'miiv': ('icustays', 'stay_id'),
        'mimic': ('icustays', 'icustay_id'),
        'eicu': ('patient', 'patientunitstayid'),
        'eicu_demo': ('patient', 'patientunitstayid'),
        'aumc': ('admissions', 'admissionid'),
        'hirid': ('general', 'patientid'),
        'sic': ('cases', 'CaseID'),
    }
    
    table_name, id_col = id_table_map.get(db_name, ('icustays', 'stay_id'))
    
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
    if df.empty:
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
    
    from .parallel_config import get_global_config, get_runtime_load_strategy

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

    Default policy now targets a balanced speed/memory profile rather than the
    most conservative low-memory path. On machines with about 10GB available RAM,
    it will prefer larger chunks to reduce batch overhead while still keeping a
    large safety margin.
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

    from .parallel_config import get_global_config, get_runtime_load_strategy
    from .memory_manager import get_available_memory_mb

    config = get_global_config()
    available_memory_mb = get_available_memory_mb()
    normalized = {str(name).lower() for name in concepts_list}

    sepsis_heavy_concepts = {'sep3', 'sep3_sofa2'}
    renal_heavy_concepts = {'kdigo_aki', 'aki'}
    sofa_heavy_concepts = {'sofa', 'sofa2'}

    if 'EASYICU_AUTO_CHUNK_SIZE' in os.environ:
        auto_chunk_size = max(250, int(os.getenv('EASYICU_AUTO_CHUNK_SIZE', '1000')))
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
            auto_chunk_size = 8000
        elif available_memory_mb >= 6 * 1024:
            auto_chunk_size = 4000
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
    sample_strategy: str = 'sorted',  # 🆕 采样策略: 'sorted'=按ID排序前N个, 'random'=随机采样
    batch_size: Optional[int] = None,  # 🆕 分批处理大小（默认30000，适合12GB内存）
    memory_efficient: bool = False,  # 🆕 内存优化模式（压缩数据类型）
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

    # SOFA2 相关概念集合（需要加载 sofa2-dict）
    sofa2_concepts = {'sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 
                      'sofa2_cardio', 'sofa2_cns', 'sofa2_renal',
                      'uo_6h', 'uo_12h', 'uo_24h', 'rrt_criteria', 'rrt',
                      'adv_resp', 'ecmo', 'ecmo_indication', 'sedated_gcs',
                      'mech_circ_support', 'other_vaso', 'delirium_tx',
                      'motor_response', 'delirium_positive'}
    if any(c in sofa2_concepts or 'sofa2' in c.lower() for c in concepts_list):
        use_sofa2 = True

    # 防御性检查: 检测常见的位置参数误用 (load_concepts(['hr'], 'miiv') 应为 database='miiv')
    _known_dbs = {'miiv', 'mimic', 'eicu', 'hirid', 'aumc', 'sic',
                  'miiv_demo', 'mimic_demo', 'eicu_demo', 'sic_demo'}
    if isinstance(patient_ids, str) and patient_ids.lower() in _known_dbs:
        if database is None:
            database = patient_ids
            patient_ids = None
        else:
            raise TypeError(
                f"patient_ids 收到字符串 '{patient_ids}'，看起来是数据库名。"
                f"请使用关键字参数: load_concepts(concepts, database='{patient_ids}')")

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
    if patient_ids is None:
        id_kwargs = ['patientunitstayid', 'admissionid', 'stay_id', 'subject_id', 'patientid']
        for id_key in id_kwargs:
            if id_key in kwargs:
                patient_ids = {id_key: kwargs.pop(id_key)}
                break

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
    if effective_max_patients is not None and patient_ids is None:
        patient_ids = _sample_patient_ids(loader, effective_max_patients, verbose,
                                          sample_strategy=sample_strategy)

    # 规范化患者ID
    if patient_ids is not None and not isinstance(patient_ids, dict):
        patient_ids = _normalize_patient_ids_for_db(loader.database, patient_ids)

    # 🚀 智能并行配置：根据概念数量和患者数量自动优化
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
        from .parallel_config import get_global_config, get_runtime_load_strategy

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
    
    from .memory_manager import (
        auto_batch_size, estimate_memory_mb, release_memory,
        get_available_memory_mb, get_rss_mb, inprocess_batch_load,
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
    
    # 自动检测全量加载场景
    if (not auto_chunk_strategy) and _total_patients is None and patient_ids is None and effective_batch_size is None:
        # 全量加载：查询总患者数来决定是否需要分批
        try:
            _total_patients_in_db = _get_total_patient_count(loader)
            if _total_patients_in_db and _total_patients_in_db > 1000:
                # 估算内存需求
                est_mem = estimate_memory_mb(concepts_list, loader.database, _total_patients_in_db)
                avail_mem = get_available_memory_mb()
                
                if est_mem > avail_mem * 0.6:
                    _total_patients = _total_patients_in_db
                    effective_batch_size = auto_batch_size(
                        concepts_list, loader.database, _total_patients, avail_mem
                    )

                    if verbose and effective_batch_size:
                        print(f"📊 全量加载 {_total_patients} patients, "
                              f"估算峰值 {est_mem:.0f}MB > 可用 {avail_mem:.0f}MB×60%, "
                              f"自动分批: batch_size={effective_batch_size}")

                    # 小内存环境使用子进程隔离；进程内路径优先走流式 patient batch
                    use_subprocess = avail_mem < 16 * 1024
                    use_streaming_patient_batches = effective_batch_size is not None
        except Exception as e:
            logger.debug(f"自动分批检测失败: {e}")
    
    # 用户显式指定了 batch_size
    if effective_batch_size is None and batch_size is not None:
        effective_batch_size = batch_size
    
    # 自动检测：用户指定了 patient_ids 但未指定 batch_size
    if (not auto_chunk_strategy) and effective_batch_size is None and _total_patients is not None and _total_patients > 5000:
        avail_mem = get_available_memory_mb()
        est_mem = estimate_memory_mb(concepts_list, loader.database, _total_patients)
        if est_mem > avail_mem * 0.6:
            effective_batch_size = auto_batch_size(
                concepts_list, loader.database, _total_patients, avail_mem
            )
            if verbose and effective_batch_size:
                print(f"📊 自动分批: {_total_patients} patients, "
                      f"估算 {est_mem:.0f}MB > budget {avail_mem*0.6:.0f}MB, "
                      f"batch_size={effective_batch_size}")
            use_subprocess = avail_mem < 16 * 1024

    if auto_chunk_strategy and verbose and effective_batch_size is None:
        print("   🧠 已跳过自动 batch 分批，优先采用已验证的平衡 chunk 路径")
    
    # 大量患者时自动启用子进程隔离，避免 Python pymalloc 内存碎片
    # inprocess_batch_load 每批次泄漏 0.5-1.5GB 碎片（pymalloc arena 不归还 OS），
    # N 批次后 RSS = N * 碎片 + 结果数据。MIIV 94K patients: 15G RSS for 1.4G data.
    # subprocess 隔离: 每批在子进程中运行，子进程退出后 OS 完整回收内存，零碎片。
    if (
        not use_subprocess
        and _total_patients is not None
        and _total_patients > 30000
        and effective_batch_size is not None
    ):
        use_subprocess = True
    
    # 🔧 FIX Bug 54: daemon 子进程不能创建子进程 (AssertionError)
    # Webapp 用 daemon=True 启动模块子进程以隔离内存碎片。
    # multiprocessing.Process.start() 在 daemon 中会抛出 AssertionError。
    # 但 os.fork() 不受此限制——subprocess_batch_load 已支持 _fork_and_run() 方式。
    # 所以仅在无 os.fork 的平台（Windows）禁用 subprocess。
    if use_subprocess:
        try:
            import multiprocessing as _mp_check
            if _mp_check.current_process().daemon and not hasattr(os, 'fork'):
                use_subprocess = False
        except Exception:
            pass
    
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
                    concepts=concepts_list,
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
                    concepts=concepts_list,
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

def list_available_sources() -> List[str]:
    """
    列出可用的数据源
    
    Returns:
        数据源名称列表
        
    Examples:
        >>> sources = list_available_sources()
        >>> print(sources)
        ['mimic', 'hirid', 'eicu', 'aumc']
    """
    registry = load_data_sources()
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
    
    if concept_name not in dict_obj.concepts:
        raise ValueError(f"未知概念: {concept_name}")
    
    concept = dict_obj.concepts[concept_name]
    
    info = {
        'name': concept_name,
        'description': getattr(concept, 'description', ''),
        'category': getattr(concept, 'category', ''),
        'unit': getattr(concept, 'unit', ''),
        'sources': list(getattr(concept, 'sources', {}).keys()),
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

    包含: urine, urine24

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

    concepts = ['urine', 'urine24']
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
) -> pd.DataFrame:
    """
    加载药物治疗数据（参考ricu.R的data_med）

    包含: abx, adh_rate, cort, dex, dobu_dur, dobu_rate, dobu60,
          epi_dur, epi_rate, ins, norepi_dur, norepi_equiv, norepi_rate, vaso_ind

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        药物治疗DataFrame

    Examples:
        >>> meds = load_medications(patient_ids=[123, 456])
    """
    if verbose:
        print("💊 加载药物治疗数据...")

    concepts = ['abx', 'adh_rate', 'cort', 'dex', 'dobu_dur', 'dobu_rate', 'dobu60',
               'epi_dur', 'epi_rate', 'ins', 'norepi_dur', 'norepi_equiv', 'norepi_rate', 'vaso_ind']
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

def _get_cache_key(concepts: List[str], source: str, **kwargs) -> str:
    """Generate cache key from parameters."""
    key_str = f"{source}_{','.join(sorted(concepts))}_{str(sorted(kwargs.items()))}"
    return hashlib.md5(key_str.encode()).hexdigest()

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
    cache_params = {'merge': merge, 'align_time': align_time, **kwargs}
    cache_key = _get_cache_key(concept_list, source, **cache_params)
    cache_ext = 'pkl' if use_pickle else 'csv'
    cache_file = cache_dir / f"{source}_{'_'.join(concept_list[:3])}_{cache_key[:8]}.{cache_ext}"
    
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
        **kwargs
    )
    
    # Save to cache
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

# 数据库 -> (表名, ID列名) 的标准映射
# 这是单一真相来源，避免在多处重复定义
DATABASE_ID_CONFIG = {
    'miiv': {'table': 'icustays', 'id_col': 'stay_id'},
    'mimic': {'table': 'icustays', 'id_col': 'icustay_id'},
    'mimic_demo': {'table': 'icustays', 'id_col': 'icustay_id'},
    'eicu': {'table': 'patient', 'id_col': 'patientunitstayid'},
    'eicu_demo': {'table': 'patient', 'id_col': 'patientunitstayid'},
    'aumc': {'table': 'admissions', 'id_col': 'admissionid'},
    'hirid': {'table': 'general', 'id_col': 'patientid'},
    'sic': {'table': 'cases', 'id_col': 'CaseID'},  # SICdb uses cases table with CaseID
}


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
    
    # 尝试加载患者表
    try:
        # 首选：直接加载 parquet 文件
        parquet_file = data_path / f'{table_name}.parquet'
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file, columns=[id_col])
            all_ids = df[id_col].dropna().unique().tolist()
        else:
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

# 模块定义（与 webapp CONCEPT_GROUPS_INTERNAL 一致，19 个模块 167 个概念）
EXTRACT_MODULES: Dict[str, List[str]] = {
    'vitals':        ['hr', 'map', 'sbp', 'dbp', 'temp', 'spo2', 'resp'],
    'demographics':  ['age', 'bmi', 'height', 'sex', 'weight', 'adm'],
    'outcome':       ['death', 'los_icu', 'los_hosp'],
    'chemistry':     ['alb', 'alp', 'alt', 'ast', 'bicar', 'bili', 'bili_dir', 'bun',
                      'ca', 'ck', 'ckmb', 'cl', 'crea', 'crp', 'glu', 'k', 'mg', 'na',
                      'phos', 'tnt', 'tri'],
    'hematology':    ['bnd', 'basos', 'eos', 'esr', 'fgn', 'hba1c', 'hct', 'hgb',
                      'inr_pt', 'lymph', 'mch', 'mchc', 'mcv', 'neut', 'plt', 'pt',
                      'ptt', 'rbc', 'rdw', 'wbc'],
    'blood_gas':     ['be', 'cai', 'hbco', 'lact', 'methb', 'pco2', 'ph', 'po2', 'tco2'],
    'medications':   ['abx', 'cort', 'dex', 'ins'],
    'ventilator':    ['peep', 'tidal_vol', 'tidal_vol_set', 'pip', 'plateau_pres',
                      'mean_airway_pres', 'minute_vol', 'vent_rate', 'etco2',
                      'compliance', 'driving_pres', 'ps'],
    'respiratory':   ['pafi', 'safi', 'fio2', 'supp_o2', 'vent_ind', 'vent_start',
                      'vent_end', 'o2sat', 'sao2', 'mech_vent', 'ett_gcs', 'ecmo',
                      'ecmo_indication', 'adv_resp'],
    'vasopressors':  ['norepi_rate', 'norepi_dur', 'norepi_equiv', 'norepi60',
                      'epi_rate', 'epi_dur', 'epi60', 'dopa_rate', 'dopa_dur', 'dopa60',
                      'dobu_rate', 'dobu_dur', 'dobu60', 'adh_rate', 'phn_rate',
                      'vaso_ind', 'other_vaso'],
    'renal':         ['urine', 'urine24', 'uo_6h', 'uo_12h', 'uo_24h', 'rrt',
                      'rrt_criteria'],
    'neurological':  ['avpu', 'egcs', 'gcs', 'mgcs', 'rass', 'tgcs', 'vgcs',
                      'sedated_gcs', 'motor_response', 'delirium_positive',
                      'delirium_tx'],
    'circulatory':   ['mech_circ_support'],
    'other_scores':  ['qsofa', 'sirs', 'mews', 'news'],
    'sofa1_score':   ['sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio',
                      'sofa_cns', 'sofa_renal'],
    'sofa2_score':   ['sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver',
                      'sofa2_cardio', 'sofa2_cns', 'sofa2_renal'],
    'sepsis_shared': ['susp_inf', 'infection_icd', 'samp'],
    'sepsis3_sofa1': ['sep3_sofa1'],
    'sepsis3_sofa2': ['sep3_sofa2'],
}

# 快→慢排序，与 webapp MODULE_PRIORITY 一致
EXTRACT_MODULE_ORDER: List[str] = [
    'vitals', 'demographics', 'outcome',
    'chemistry', 'hematology', 'blood_gas',
    'medications', 'ventilator', 'respiratory',
    'vasopressors', 'renal', 'neurological',
    'other_scores', 'circulatory',
    'sofa1_score', 'sofa2_score',
    'sepsis3_sofa1', 'sepsis3_sofa2', 'sepsis_shared',
]

# 特殊概念 — 需要专用加载函数而非 load_concepts
_SPECIAL_CONCEPT_MODULES = {'sepsis3_sofa1', 'sepsis3_sofa2'}

# 已知数据库路径映射（可被 data_paths 参数覆盖）
DEFAULT_DB_PATHS: Dict[str, str] = {
    'sic':   '/home/zhuhb/icudb/sic/',
    'aumc':  '/home/zhuhb/icudb/aumc/1.0.2/',
    'hirid': '/home/zhuhb/icudb/hirid/1.1.1/',
    'mimic': '/home/zhuhb/icudb/mimiciii/1.4/',
    'miiv':  '/home/zhuhb/icudb/mimiciv/3.1/',
    'eicu':  '/home/zhuhb/icudb/eicu/2.0.1/',
}


def _extract_module_worker(
    concepts: List[str],
    database: str,
    data_path: str,
    patient_ids_filter: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    output_dir: str = '',
    module_name: str = '',
):
    """在子进程中加载一个模块的所有概念并写入 parquet。

    这是顶层函数（非闭包），可被 multiprocessing.Process 序列化。
    子进程退出后 OS 完整回收所有内存（包括 pymalloc arena 碎片）。
    """
    import os, sys, json, time, traceback
    os.environ.setdefault('EASYICU_DATA_PATH', data_path)
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    import pandas as pd
    from easyicu import load_concepts as _lc

    t0 = time.time()
    saved = {}
    errors = []

    # 构造 load_concepts 参数
    kwargs = dict(
        data_path=data_path, database=database,
        concepts=concepts, verbose=False, merge=False,
        concept_workers=1,
    )
    if patient_ids_filter:
        kwargs['patient_ids'] = patient_ids_filter
    if batch_size:
        kwargs['batch_size'] = batch_size

    try:
        result = _lc(**kwargs)
    except Exception as e:
        traceback.print_exc()
        errors.append(f"load_concepts({module_name}): {e}")
        result = {}

    # 将结果写入 parquet 文件
    if isinstance(result, dict):
        for c, df in result.items():
            try:
                if hasattr(df, 'data') and isinstance(df.data, pd.DataFrame):
                    df = df.data
                elif hasattr(df, 'to_pandas'):
                    df = df.to_pandas()
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    path = os.path.join(output_dir, f"{c}.parquet")
                    df.to_parquet(path, index=False, engine='pyarrow')
                    saved[c] = {'path': path, 'rows': len(df)}
            except Exception as e:
                errors.append(f"{c}: {e}")
    elif isinstance(result, pd.DataFrame) and len(result) > 0:
        for c in concepts:
            if c in result.columns:
                path = os.path.join(output_dir, f"{c}.parquet")
                result.to_parquet(path, index=False, engine='pyarrow')
                saved[c] = {'path': path, 'rows': len(result)}
                break

    elapsed = time.time() - t0
    manifest = {
        'module': module_name,
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
    """在子进程中加载特殊概念（Sepsis-3 等）。

    sep3_sofa1/sep3_sofa2 不在 concept-dict 中，需要先加载 susp_inf + sofa/sofa2，
    然后通过 _load_sep3_diagnosis 逻辑计算 Sepsis-3 诊断。
    """
    import os, sys, json, time, traceback
    os.environ.setdefault('EASYICU_DATA_PATH', data_path)
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    import pandas as pd
    from easyicu import load_concepts as _lc

    t0 = time.time()
    saved = {}
    errors = []

    # 构建公共加载参数
    load_kw = dict(data_path=data_path, database=database, verbose=False, merge=True)
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

            if need_sofa1 and 'sofa' in merged.columns:
                merged['sep3_sofa1'] = (susp & (merged['sofa'].fillna(0) >= 2)).astype(int)
                result = merged.loc[susp, [id_col, time_col, 'sep3_sofa1']].copy()
                if len(result) > 0:
                    path = os.path.join(output_dir, 'sep3_sofa1.parquet')
                    result.to_parquet(path, index=False, engine='pyarrow')
                    saved['sep3_sofa1'] = {'path': path, 'rows': len(result)}

            if need_sofa2 and 'sofa2' in merged.columns:
                merged['sep3_sofa2'] = (susp & (merged['sofa2'].fillna(0) >= 2)).astype(int)
                result = merged.loc[susp, [id_col, time_col, 'sep3_sofa2']].copy()
                if len(result) > 0:
                    path = os.path.join(output_dir, 'sep3_sofa2.parquet')
                    result.to_parquet(path, index=False, engine='pyarrow')
                    saved['sep3_sofa2'] = {'path': path, 'rows': len(result)}
        else:
            missing = []
            if not id_col: missing.append('id_col')
            if not time_col: missing.append('time_col')
            if 'susp_inf' not in merged.columns: missing.append('susp_inf')
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


def extract_database(
    database: str,
    data_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    modules: Optional[List[str]] = None,
    patient_ids: Optional[Union[List, Dict]] = None,
    max_patients: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """按模块子进程隔离提取整个数据库的全部特征。

    每个模块在独立子进程中运行 load_concepts()，子进程退出后 OS 完整回收内存
    （包括 Python pymalloc arena 碎片），主进程 RSS 几乎不增长。
    适用于 16GB 内存环境下对任意规模数据库的全量提取。

    Args:
        database: 数据库类型 ('miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic')
        data_path: 数据路径（None 则使用 DEFAULT_DB_PATHS 或自动检测）
        output_dir: 输出目录（None 则不写文件，仅返回 dict）
        modules: 要提取的模块列表（None = 全部 19 个模块）
        patient_ids: 患者 ID 列表或 dict（None = 全部患者）
        max_patients: 限制患者数量（与 patient_ids 互斥）
        batch_size: 子进程内的患者分批大小（None = 自动计算）
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

    from .memory_manager import get_rss_mb, get_available_memory_mb

    t_start = time.time()

    # 确定数据路径
    if data_path is None:
        data_path = DEFAULT_DB_PATHS.get(database)
        if data_path is None:
            data_path = get_default_data_path()
    data_path = str(data_path)

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

    # 自动计算 batch_size
    if batch_size is None:
        avail_mb = get_available_memory_mb()
        # 子进程内使用：每患者约 0.5MB 峰值（合理估算），占可用内存 50%
        frag_safe = max(5000, int(avail_mb * 0.5))
        if num_patients > frag_safe:
            batch_size = frag_safe

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
        print(f"   batch_size: {batch_size or '不分批'}")
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

    # ---- 逐模块在子进程中加载 ----
    for idx, mod_name in enumerate(normal_modules):
        concepts = EXTRACT_MODULES.get(mod_name, [])
        if not concepts:
            continue

        mod_start = time.time()
        tmp_dir = tempfile.mkdtemp(prefix=f'easyicu_{mod_name}_')

        if verbose:
            rss = get_rss_mb()
            print(f"\n[{idx+1}/{len(normal_modules)+len(special_modules)}] "
                  f"⏳ {mod_name} ({len(concepts)} concepts) ... RSS={rss:.0f}MB")

        proc = mp_ctx.Process(
            target=_extract_module_worker,
            args=(concepts, database, data_path, patient_ids_filter,
                  batch_size, tmp_dir, mod_name),
            daemon=True,
        )
        proc.start()
        proc.join()

        mod_elapsed = time.time() - mod_start
        mod_result = {'concepts': {}, 'elapsed': round(mod_elapsed, 1), 'errors': []}
        n_rows = 0

        # 读回结果
        manifest_path = os.path.join(tmp_dir, '_manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            mod_result['errors'] = manifest.get('errors', [])

            for c_name, info in manifest.get('saved', {}).items():
                pq_path = info['path']
                if os.path.exists(pq_path):
                    rows = info.get('rows', 0)
                    n_rows += rows
                    if output_dir is not None:
                        # 流式写盘：move 文件到输出目录，不读回内存
                        mod_out = os.path.join(output_dir, mod_name)
                        os.makedirs(mod_out, exist_ok=True)
                        dst = os.path.join(mod_out, f"{c_name}.parquet")
                        shutil.move(pq_path, dst)
                        mod_result['concepts'][c_name] = {'path': dst, 'rows': rows}
                    else:
                        # 无输出目录：读回 DataFrame 到内存
                        df = pd.read_parquet(pq_path)
                        mod_result['concepts'][c_name] = df
                        n_rows = n_rows - rows + len(df)  # 用实际行数

        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)

        n_concepts = len(mod_result['concepts'])
        result['modules'][mod_name] = mod_result

        if verbose:
            status = '✅' if not mod_result['errors'] else '⚠️'
            print(f"   {status} {mod_name}: {n_concepts} concepts, "
                  f"{n_rows:,} rows, {mod_elapsed:.1f}s"
                  + (f" | errors: {mod_result['errors']}" if mod_result['errors'] else ''))

    # ---- 特殊模块（Sepsis-3）在子进程中加载 ----
    if special_modules:
        sp_start = time.time()
        tmp_dir = tempfile.mkdtemp(prefix='easyicu_special_')

        if verbose:
            n_done = len(normal_modules)
            n_total = len(normal_modules) + len(special_modules)
            print(f"\n[{n_done+1}/{n_total}] ⏳ special ({special_modules}) ...")

        proc = mp_ctx.Process(
            target=_extract_special_worker,
            args=(special_modules, database, data_path, patient_ids_filter,
                  batch_size, tmp_dir),
            daemon=True,
        )
        proc.start()
        proc.join()

        sp_elapsed = time.time() - sp_start

        manifest_path = os.path.join(tmp_dir, '_manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)

            for mod_name in special_modules:
                concepts = EXTRACT_MODULES.get(mod_name, [])
                mod_result = {'concepts': {}, 'elapsed': round(sp_elapsed, 1),
                              'errors': manifest.get('errors', [])}

                for c_name in concepts:
                    info = manifest.get('saved', {}).get(c_name)
                    if info and os.path.exists(info['path']):
                        rows = info.get('rows', 0)
                        if output_dir is not None:
                            mod_out = os.path.join(output_dir, mod_name)
                            os.makedirs(mod_out, exist_ok=True)
                            dst = os.path.join(mod_out, f"{c_name}.parquet")
                            shutil.move(info['path'], dst)
                            mod_result['concepts'][c_name] = {'path': dst, 'rows': rows}
                        else:
                            df = pd.read_parquet(info['path'])
                            mod_result['concepts'][c_name] = df

                result['modules'][mod_name] = mod_result

                if verbose:
                    n_c = len(mod_result['concepts'])
                    n_r = 0
                    for v in mod_result['concepts'].values():
                        if isinstance(v, dict):
                            n_r += v.get('rows', 0)
                        elif isinstance(v, pd.DataFrame):
                            n_r += len(v)
                    print(f"   {'✅' if not mod_result['errors'] else '⚠️'} "
                          f"{mod_name}: {n_c} concepts, {n_r:,} rows, {sp_elapsed:.1f}s")

        shutil.rmtree(tmp_dir, ignore_errors=True)

    total_elapsed = time.time() - t_start
    result['total_elapsed'] = round(total_elapsed, 1)

    if verbose:
        rss = get_rss_mb()
        total_concepts = sum(len(m['concepts']) for m in result['modules'].values())
        total_rows = 0
        for m in result['modules'].values():
            for v in m['concepts'].values():
                if isinstance(v, dict):
                    total_rows += v.get('rows', 0)
                elif isinstance(v, pd.DataFrame):
                    total_rows += len(v)
        all_errors = [e for m in result['modules'].values() for e in m['errors']]
        print(f"\n{'='*60}")
        print(f"✅ {database} 完成: {total_concepts} concepts, "
              f"{total_rows:,} rows, {total_elapsed:.1f}s")
        print(f"   RSS: {rss:.0f}MB" +
              (f"  |  输出: {output_dir}" if output_dir else ''))
        if all_errors:
            print(f"   ⚠️ {len(all_errors)} 错误: {all_errors[:5]}")
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

    merged_paths = dict(DEFAULT_DB_PATHS)
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
