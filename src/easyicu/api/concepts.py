"""
easyicu 高层API - 提供简单易用的接口，同时支持高级自定义

重构后的统一API，整合了多个模块的功能:
- api/__init__.py: 稳定公共入口与兼容 facade
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
import re
import threading
from collections import OrderedDict
from contextlib import contextmanager
import numpy as np
import pandas as pd
import logging

from ..base import BaseICULoader
from ..databases.profiles import get_database_profile, public_database_keys
from ..table.duration import (
    get_dur_var_unit,
    resolve_dur_var_hours,
    set_dur_var_unit,
)
logger = logging.getLogger(__name__)

# 全局加载器实例，用于复用初始化开销。
# `_LOADER_LOCK` 保护 check-then-create-then-return 这一整段：没有它时，两个线程
# 分别请求 miiv 与 eicu 会互相覆盖 `_global_loader`，先进入的线程 return 到的可能
# 是另一个数据库的 loader（错误的 ID 列、错误的表、被污染的缓存）。
_global_loader = None
_loader_config = None
_LOADER_LOCK = threading.RLock()

# Loaders are cached PER CONFIG so switching database never has to tear down a
# loader another thread is mid-extraction on. Bounded so a long-lived process
# that cycles through many data paths cannot grow without limit.
_LOADER_CACHE_MAX = 4
_loader_cache: "OrderedDict[tuple, BaseICULoader]" = OrderedDict()


def _normalize_patient_ids_for_db(database_name: str, patient_ids):
    """Normalize patient IDs to the canonical ID column for each database."""
    if patient_ids is None or isinstance(patient_ids, dict):
        return patient_ids

    return {_database_profile_or_default(database_name).stay_id_col: patient_ids}


def _database_profile_or_default(database_name: str):
    """Resolve database metadata, defaulting to MIIV only when unspecified.

    A *named* database that does not resolve is an error, not a reason to fall
    back: silently handing back the MIIV profile for ``"mimic3"`` or
    ``"eicu_crd"`` applies MIMIC-IV's ``stay_id`` column to another database and
    yields an empty or wrong cohort with no diagnostic.
    """

    if database_name is None or not str(database_name).strip():
        return get_database_profile("miiv")

    try:
        return get_database_profile(database_name)
    except KeyError as exc:
        try:
            supported = ", ".join(sorted(public_database_keys()))
        except Exception:  # pragma: no cover - registry unavailable
            supported = "miiv, eicu, aumc, hirid, sicdb, miii"
        raise ValueError(
            f"Unsupported database {database_name!r}. Supported: {supported}"
        ) from exc


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


#: Chunk size used for every SOFA-family auto-chunked extraction, on every
#: host. Fixed rather than memory-tiered: chunk size is an execution parameter
#: that must not change a clinical score.
SOFA_FIXED_CHUNK_SIZE = 2000

#: Hard ceiling on the points one window may expand to. A correct hourly
#: window over a month-long stay stays far below this; blowing past it means
#: the duration was misread (wrong unit, nanoseconds, corrupt row) and we fail
#: loudly instead of allocating until the process dies.
MAX_WINDOW_EXPANSION_POINTS = 10_000


class WindowExpansionError(ValueError):
    """Raised when one win_tbl row expands to an implausible number of points."""


def _guard_window_expansion(
    emitted: int,
    *,
    concept_name: str,
    duration: float,
    unit: str,
    row: dict,
) -> None:
    """Fail closed when a single window expands past the plausible ceiling."""

    if emitted <= MAX_WINDOW_EXPANSION_POINTS:
        return
    # Describe the row, do not reproduce it. This message travels: it becomes a
    # web job error, a line in CI output, an agent finding, and part of an API
    # response. Pasting the offending record in put that patient's identifier
    # and event times into all four. The digest is enough to match the row
    # against the source table during debugging, and carries nothing on its own.
    import hashlib

    fingerprint = hashlib.sha256(
        repr(sorted((str(k), str(v)) for k, v in row.items())).encode("utf-8")
    ).hexdigest()
    raise WindowExpansionError(
        f"concept {concept_name!r}: one window expanded past "
        f"{MAX_WINDOW_EXPANSION_POINTS} points (duration={duration} {unit}). "
        "This almost always means dur_var was read in the wrong unit — declare "
        "it at the producer with set_dur_var_unit(). Offending row: field(s) "
        + ", ".join(sorted(str(key) for key in row))
        + f"; sha256 {fingerprint[:16]}"
    )


def _resolve_duration_hours(
    work: pd.DataFrame,
    result: pd.DataFrame,
    concept_name: str,
) -> pd.Series:
    """Read ``work['dur_var']`` as float hours using the declared unit.

    Shared by the datetime and numeric index branches so a unit declaration can
    never be honoured by one and ignored by the other. A ``timedelta64`` column
    is self-describing; a numeric one takes the declaration from whichever frame
    still carries the producer's ``attrs``.
    """

    dur_frame = work[["dur_var"]].copy()
    if not pd.api.types.is_timedelta64_dtype(dur_frame["dur_var"]):
        declared = get_dur_var_unit(work) or get_dur_var_unit(result)
        if declared:
            set_dur_var_unit(dur_frame, declared)
    return resolve_dur_var_hours(dur_frame, concept=concept_name)


def _expand_public_numeric_win_tbl_output(
    result: pd.DataFrame,
    concept_name: str,
    interval: Optional[Union[str, pd.Timedelta]],
) -> pd.DataFrame:
    """Expand single-concept numeric win_tbl output to ricu-compatible rows."""
    if not isinstance(result, pd.DataFrame) or result.empty:
        return result
    if concept_name not in result.columns or "dur_var" not in result.columns:
        return result

    numeric_values = pd.to_numeric(result[concept_name], errors="coerce")
    if numeric_values.notna().sum() == 0:
        return result

    index_candidates = [
        "charttime",
        "starttime",
        "start",
        "datetime",
        "measuredat",
        "measuredat_minutes",
        "givenat",
        "infusionoffset",
        "observationoffset",
        "labresultoffset",
    ]
    index_column = next(
        (col for col in index_candidates if col in result.columns), None
    )
    if index_column is None:
        return result

    id_priority = [
        "stay_id",
        "icustay_id",
        "patientunitstayid",
        "admissionid",
        "patientid",
        "CaseID",
        "subject_id",
    ]
    id_columns = [col for col in id_priority if col in result.columns]
    if not id_columns:
        id_columns = [
            col
            for col in result.columns
            if col.lower().endswith("id") and col not in {index_column, "dur_var"}
        ]
    if not id_columns:
        return result

    interval_td = pd.to_timedelta(interval or "1h")
    if pd.isna(interval_td) or interval_td <= pd.Timedelta(0):
        interval_td = pd.Timedelta(hours=1)

    work = result[id_columns + [index_column, "dur_var", concept_name]].copy()
    work[concept_name] = numeric_values
    work = work.dropna(subset=[index_column, concept_name])
    if work.empty:
        return result

    expanded_rows = []
    is_datetime_index = pd.api.types.is_datetime64_any_dtype(work[index_column])

    if is_datetime_index:
        work[index_column] = pd.to_datetime(work[index_column], errors="coerce")
        # Both branches read the SAME declared unit. This one used to hardcode
        # unit="m" for every numeric dur_var, so a frame that declared hours was
        # read as minutes — the same 60x error as the old distribution guess,
        # just on the datetime path.
        duration_values = pd.to_timedelta(
            _resolve_duration_hours(work, result, concept_name), unit="h"
        )
        epsilon = pd.Timedelta(microseconds=1)

        for row, duration in zip(work.itertuples(index=False), duration_values):
            row_dict = row._asdict()
            start = row_dict[index_column]
            if pd.isna(start):
                continue
            end = start + duration
            current = start
            emitted = 0
            while current <= end + epsilon:
                emitted += 1
                _guard_window_expansion(
                    emitted,
                    concept_name=concept_name,
                    duration=duration.total_seconds() / 3600.0,
                    unit="hours",
                    row=row_dict,
                )
                expanded_rows.append(
                    {
                        **{col: row_dict[col] for col in id_columns},
                        index_column: current,
                        concept_name: row_dict[concept_name],
                    }
                )
                current = current + interval_td
    else:
        work[index_column] = pd.to_numeric(work[index_column], errors="coerce")
        work = work.dropna(subset=[index_column])
        if work.empty:
            return result

        interval_hours = interval_td.total_seconds() / 3600.0
        if interval_hours <= 0:
            interval_hours = 1.0

        duration_hours = _resolve_duration_hours(work, result, concept_name)
        epsilon = 1e-9

        for row, duration_hour in zip(work.itertuples(index=False), duration_hours):
            row_dict = row._asdict()
            start = row_dict[index_column]
            if pd.isna(start):
                continue
            end = start + max(float(duration_hour), 0.0)
            current = float(start)
            emitted = 0
            while current <= end + epsilon:
                emitted += 1
                _guard_window_expansion(
                    emitted,
                    concept_name=concept_name,
                    duration=float(duration_hour),
                    unit="hours",
                    row=row_dict,
                )
                expanded_rows.append(
                    {
                        **{col: row_dict[col] for col in id_columns},
                        index_column: current,
                        concept_name: row_dict[concept_name],
                    }
                )
                current += interval_hours

    if not expanded_rows:
        return result

    expanded = pd.DataFrame(expanded_rows)
    expanded = (
        expanded.groupby(id_columns + [index_column], as_index=False)[concept_name]
        .median()
        .sort_values(id_columns + [index_column], kind="mergesort")
        .reset_index(drop=True)
    )
    return expanded


def _build_fast_scan_expr(loader: "BaseICULoader", table_name: str) -> Optional[str]:
    """Build a DuckDB scan expression for a table without materializing it in pandas."""
    data_source = getattr(loader, "datasource", None)
    if data_source is None or not hasattr(data_source, "_resolve_loader_from_disk"):
        return None

    source = data_source._resolve_loader_from_disk(table_name)
    if not isinstance(source, Path):
        return None

    def _escape(path: str) -> str:
        return path.replace("'", "''").replace("\\", "/")

    if source.is_dir():
        # 显式文件列表，过滤 AppleDouble (._*.parquet) — 见 datasource._enumerate_bucket_parquet_files
        try:
            from ..datasource import _enumerate_bucket_parquet_files as _enum
        except Exception:
            _enum = None
        if _enum is not None:
            files = _enum(source)
            if files:
                files_sql = "[" + ", ".join(f"'{_escape(f)}'" for f in files) + "]"
                return f"read_parquet({files_sql}, union_by_name=true)"
        # Fallback: 旧的 glob 路径（仅在 helper 不可用或空目录时）
        bucket_dirs = list(source.glob("bucket_id=*"))
        if bucket_dirs:
            pattern = str(source / "bucket_id=*" / "*.parquet").replace("\\", "/")
        else:
            parquet_files = list(source.glob("*.parquet")) + list(source.glob("*.pq"))
            if not parquet_files:
                return None
            pattern = (
                str(source / "*.parquet")
                if list(source.glob("*.parquet"))
                else str(source / "*.pq")
            ).replace("\\", "/")
        return f"read_parquet('{_escape(pattern)}', union_by_name=true)"

    suffixes = [s.lower() for s in source.suffixes]
    source_str = _escape(str(source))
    if source.suffix.lower() in {".parquet", ".pq"}:
        return f"read_parquet('{source_str}', union_by_name=true)"
    if ".csv" in suffixes or source.suffix.lower() == ".csv":
        return f"read_csv_auto('{source_str}')"
    return None


def _query_patient_ids_fast(
    loader: "BaseICULoader",
    table_name: str,
    id_col: str,
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    sample_strategy: str = "sorted",
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
    if sample_strategy == "random":
        order_expr = f'hash("{id_col}")'

    limit_clause = f" LIMIT {int(limit)}" if limit and limit > 0 else ""
    offset_clause = f" OFFSET {int(offset)}" if offset and offset > 0 else ""
    query = (
        f'SELECT DISTINCT "{id_col}" AS patient_id '
        f"FROM {scan_expr} "
        f'WHERE "{id_col}" IS NOT NULL '
        f"ORDER BY {order_expr}{limit_clause}{offset_clause}"
    )

    conn = duckdb.connect()
    try:
        conn.execute("SET timezone='UTC'")
        conn.execute("SET enable_progress_bar = false")
        conn.execute("SET enable_progress_bar_print = false")
        conn.execute("SET memory_limit = '2GB'")
        return conn.execute(query).fetchnumpy()["patient_id"].tolist()
    finally:
        conn.close()


def _count_patient_ids_fast(
    loader: "BaseICULoader", table_name: str, id_col: str
) -> Optional[int]:
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
        f"FROM {scan_expr} "
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


def _release_loader(loader) -> None:
    """Drop a loader's caches so its memory is reclaimed promptly."""

    if loader is None:
        return
    if hasattr(loader, "clear_cache"):
        loader.clear_cache()
        return
    # 清理加载器内部缓存
    if hasattr(loader, "concept_resolver"):
        loader.concept_resolver.clear()
    for attr in ("datasource", "data_source"):
        data_source = getattr(loader, attr, None)
        if data_source is not None and hasattr(data_source, "clear"):
            data_source.clear()


def clear_global_loader():
    """清除全局加载器，强制下一次调用重新创建。

    This is an explicit caller-driven teardown, so releasing caches here is
    intentional — unlike the implicit teardown that used to fire whenever
    another thread happened to request a different database.
    """
    global _global_loader, _loader_config
    with _LOADER_LOCK:
        cached = list(_loader_cache.values())
        _loader_cache.clear()
        for loader in cached:
            _release_loader(loader)
        if _global_loader is not None and _global_loader not in cached:
            _release_loader(_global_loader)
        _global_loader = None
        _loader_config = None


@contextmanager
def keep_cache(
    database=None, data_path=None, dict_path=None, use_sofa2=False, verbose=False
):
    """Context manager: keep raw/table cache between sequential load_concepts calls.

    Usage::

        with keep_cache(database='miiv'):
            df1 = load_concepts(['hr', 'sbp'], database='miiv', max_patients=1000)
            df2 = load_concepts(['sofa'], database='miiv', max_patients=1000)
            # sofa reuses cached hr/sbp/map/etc. from df1's sub-concept loads
    """
    loader = _get_global_loader(
        database=database,
        data_path=data_path,
        dict_path=dict_path,
        use_sofa2=use_sofa2,
        verbose=verbose,
    )
    resolver = loader.concept_resolver
    resolver._keep_cache_between_calls = True
    try:
        yield loader
    finally:
        resolver._keep_cache_between_calls = False
        if hasattr(resolver, "drop_source_caches"):
            resolver.drop_source_caches()
        else:  # pragma: no cover - legacy resolver without cache accounting
            with resolver._cache_lock:
                resolver._raw_concept_cache.clear()
                resolver._table_cache.clear()


def _sample_patient_ids(
    loader: "BaseICULoader",
    max_patients: int,
    verbose: bool = False,
    sample_strategy: str = "sorted",
) -> List:
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
            limit=max_patients if sample_strategy != "random" else max_patients,
            sample_strategy=sample_strategy,
        )
        if fast_ids is not None:
            if sample_strategy == "random":
                sampled_ids = sorted(fast_ids[:max_patients])
                strategy_label = "随机采样"
            else:
                sampled_ids = fast_ids[:max_patients]
                strategy_label = "已排序"

            if verbose:
                print(
                    f"🎯 max_patients={max_patients}: DuckDB 快速采样 {len(sampled_ids)} 个患者 ({strategy_label})"
                )
            return sampled_ids

        # 只加载ID列，限制行数
        id_table = loader.datasource.load_table(
            table_name, columns=[id_col], verbose=False
        )
        all_ids = id_table.data[id_col].dropna().unique()

        if sample_strategy == "random" and len(all_ids) > max_patients:
            import numpy as np

            rng = np.random.default_rng(seed=42)  # 固定种子保证可复现
            sampled_ids = sorted(
                rng.choice(all_ids, size=max_patients, replace=False).tolist()
            )
            strategy_label = "随机采样"
        else:
            # 🔧 按ID排序后再采样，确保与 RICU 金标准生成脚本一致
            all_ids = sorted(all_ids)
            sampled_ids = list(all_ids[:max_patients])
            strategy_label = "已排序"

        if verbose:
            print(
                f"🎯 max_patients={max_patients}: 从 {table_name}.{id_col} 采样 {len(sampled_ids)} 个患者 ({strategy_label})"
            )

        return sampled_ids
    except Exception as e:
        if verbose:
            print(f"⚠️ 采样患者ID失败: {e}，将加载所有患者")
        return None


def _get_patient_id_source(loader: "BaseICULoader") -> tuple[str, str]:
    """Return the canonical (table_name, id_col) pair for a database."""
    profile = _database_profile_or_default(loader.database)
    return profile.stay_table, profile.stay_id_col


def _iter_patient_id_batches(
    loader: "BaseICULoader",
    batch_size: int,
    *,
    total_patients: Optional[int] = None,
    sample_strategy: str = "sorted",
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
            all_ids = _sample_patient_ids(
                loader,
                total_patients or 999999999,
                verbose=False,
                sample_strategy=sample_strategy,
            )
            if not all_ids:
                return
            for start in range(0, len(all_ids), batch_size):
                yield {id_col: all_ids[start : start + batch_size]}
            return

        if not batch_ids:
            return

        yield {id_col: batch_ids}
        offset += len(batch_ids)
        if remaining is not None:
            remaining -= len(batch_ids)


def _get_total_patient_count(loader: "BaseICULoader") -> Optional[int]:
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
        id_table = loader.datasource.load_table(
            table_name, columns=[id_col], verbose=False
        )
        return id_table.data[id_col].nunique()
    except Exception:
        return None


#: Columns whose float64 precision is part of the result, not overhead.
#: float32 carries ~7 significant digits, which is ample for a heart rate and
#: not for a p value of 3.2e-9, a regression coefficient, a CI bound, or a
#: model probability compared against a threshold. Downcasting these silently
#: changes reported numbers and breaks R/Python parity, so they keep float64.
_PRECISION_COLUMN_RE = re.compile(
    r"(?:^|_)(?:p|pval|p_value|pvalue|q|qval|padj|prob|probability|proba|score"
    r"|coef|coefficient|beta|se|std_err|stderr|ci|ci_low|ci_lower|ci_high"
    r"|ci_upper|lower|upper|hr|or|rr|auroc|auc|brier|weight|weights|estimate"
    r"|statistic|pred|predicted|risk)$",
    re.IGNORECASE,
)


def _keeps_full_precision(name: str) -> bool:
    lowered = str(name).strip().lower()
    if lowered in {"time", "index", "duration", "dur_var", "offset"}:
        return True
    return bool(_PRECISION_COLUMN_RE.search(lowered))


def _compress_dtypes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    压缩 DataFrame 的数据类型以减少内存使用

    - int64 -> int32 (如果值范围允许)
    - float64 -> float32 (对于非精确值，统计量列除外)
    - 保持 datetime64 不变

    可以节省约 50-60% 的内存
    """
    if hasattr(df, "data") and isinstance(getattr(df, "data"), pd.DataFrame):
        df.data = _compress_dtypes(df.data, verbose=verbose)
        return df

    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    original_mem = df.memory_usage(deep=True).sum()

    for col in df.columns:
        col_type = df[col].dtype

        if _keeps_full_precision(col):
            continue

        # 整数类型压缩
        if col_type == np.int64:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif (
                col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max
            ):
                df[col] = df[col].astype(np.int16)

        # 浮点类型压缩 - SOFA 分数等小整数可以用 int8
        elif col_type == np.float64:
            # 检查是否都是整数值
            if df[col].dropna().apply(lambda x: x == int(x)).all():
                col_min, col_max = df[col].min(), df[col].max()
                if not np.isnan(col_min) and col_min >= -128 and col_max <= 127:
                    # 小整数用 Int8 (可空整数)
                    df[col] = df[col].astype("Int8")
                elif (
                    not np.isnan(col_min)
                    and col_min >= np.iinfo(np.int16).min
                    and col_max <= np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype("Int16")
            else:
                # 一般浮点数用 float32
                df[col] = df[col].astype(np.float32)

    if verbose:
        new_mem = df.memory_usage(deep=True).sum()
        saved = (original_mem - new_mem) / original_mem * 100
        print(
            f"💾 内存压缩: {original_mem/1024/1024:.1f}MB → {new_mem/1024/1024:.1f}MB (节省 {saved:.0f}%)"
        )

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
    config_kwargs = {k: v for k, v in kwargs.items() if k in ("use_sofa2",)}
    current_config = (
        database,
        str(data_path) if data_path else None,
        dict_key,
        frozenset(config_kwargs.items()),
    )

    # One loader PER CONFIG, not one global slot. Two rules matter here:
    #
    # 1. The lookup and the insert happen under the lock, so a caller can never
    #    be handed a loader built for a different database.
    # 2. Switching configs must NOT tear down the loader another thread is
    #    still using. Clearing a live loader's caches mid-extraction empties
    #    its ConceptResolver and DataSource under the caller's feet, which
    #    surfaces as intermittent KeyErrors, empty tables or drifting results.
    #    Evicted entries are simply dropped from the cache; Python frees them
    #    once their last user is done.
    with _LOADER_LOCK:
        loader = _loader_cache.get(current_config)
        if loader is None:
            loader = BaseICULoader(
                database=database,
                data_path=data_path,
                dict_path=dict_path,
                **kwargs,
            )
            _loader_cache[current_config] = loader
        _loader_cache.move_to_end(current_config)
        while len(_loader_cache) > _LOADER_CACHE_MAX:
            _loader_cache.popitem(last=False)

        # `_global_loader` / `_loader_config` stay in sync for the legacy
        # single-loader accessors (clear_global_loader, keep_cache).
        _global_loader = loader
        _loader_config = current_config
        return loader


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
    if os.getenv("EASYICU_NO_AUTO_PARALLEL"):
        return 1, None

    from ..runtime.parallel_config import get_global_config, get_runtime_load_strategy

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
    if os.getenv("EASYICU_DISABLE_AUTO_CHUNK"):
        return False
    if not merge:
        return False
    if chunk_size is not None or batch_size is not None:
        return False

    normalized = {str(name).lower() for name in concepts_list}
    heavy_concepts = {
        "sofa",
        "sofa2",
        "kdigo_aki",
        "aki",
        "sep3",
        "sep3_sofa2",
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

    The policy balances throughput and memory. SOFA-family concepts stay on the
    fixed ``SOFA_FIXED_CHUNK_SIZE`` profile regardless of available memory, so
    that the chunking of a given cohort is a property of the cohort and not of
    the host that happened to run it.

    Partition invariance was measured on real prepared data (2026-07-25):
    ``sofa`` and ``sofa2`` are byte-identical (``check_exact=True``) across
    ``chunk_size`` in {None, 250, 500, 1000, 2000, 4000} and
    ``parallel_workers`` in {1, 4}, for MIMIC-IV cohorts of 1,000 / 3,000 /
    10,000 stays and an eICU cohort of 3,000. An earlier note here claimed
    chunk size could change large-cohort window expansion; that is not
    reproducible at those scales and has been replaced by this measurement.
    Full-database scale (~94k stays) has NOT been measured, so the fixed
    profile stays as cheap insurance rather than being relaxed.
    See ``tests/test_sofa_partition_invariance.py``.
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

    from ..runtime.parallel_config import get_global_config, get_runtime_load_strategy
    from ..runtime.memory_manager import get_available_memory_mb

    config = get_global_config()
    available_memory_mb = get_available_memory_mb()
    normalized = {str(name).lower() for name in concepts_list}

    sepsis_heavy_concepts = {"sep3", "sep3_sofa2"}
    renal_heavy_concepts = {"kdigo_aki", "aki"}
    sofa_heavy_concepts = {"sofa", "sofa2"}

    if "EASYICU_AUTO_CHUNK_SIZE" in os.environ:
        auto_chunk_size = max(250, int(os.getenv("EASYICU_AUTO_CHUNK_SIZE", "1000")))
        if (
            normalized.intersection(sofa_heavy_concepts)
            and auto_chunk_size > SOFA_FIXED_CHUNK_SIZE
        ):
            logger.warning(
                "Capping SOFA auto chunk size at %d to stay on the profile whose "
                "partition invariance has been measured.",
                SOFA_FIXED_CHUNK_SIZE,
            )
            auto_chunk_size = SOFA_FIXED_CHUNK_SIZE
        elif normalized.intersection(sofa_heavy_concepts):
            logger.info(
                "EASYICU_AUTO_CHUNK_SIZE=%d overrides the fixed SOFA chunk "
                "profile (%d). SOFA/SOFA-2 measured byte-identical across chunk "
                "sizes up to 10k stays, so this is expected to be safe; it has "
                "not been measured at full-database scale.",
                auto_chunk_size,
                SOFA_FIXED_CHUNK_SIZE,
            )
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
        # Deliberately NOT memory-tiered: a fixed profile bounds resource use
        # and keeps the execution configuration stable, so a run is reproducible
        # from its parameters rather than from the host's free RAM. Partition
        # invariance itself is measured (see the docstring): SOFA/SOFA-2 agree
        # across chunk sizes up to 10k stays on MIMIC-IV and eICU. Full-database
        # scale is not yet measured. A low-memory host can opt into a smaller
        # chunk via EASYICU_AUTO_CHUNK_SIZE.
        auto_chunk_size = SOFA_FIXED_CHUNK_SIZE
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
    tuned_parallel_workers = min(int(runtime_strategy["parallel_workers"]), batches)
    tuned_concept_workers = int(runtime_strategy["concept_workers"])

    return {
        "chunk_size": auto_chunk_size,
        "parallel_workers": max(1, tuned_parallel_workers),
        "concept_workers": max(1, tuned_concept_workers),
    }


# SOFA2 相关概念集合（需要加载 sofa2-dict）。load_concepts 的自动检测和
# extract_database 的分组 worker 共用这一份定义，保证两边判定一致。
_SOFA2_TRIGGER_CONCEPTS = frozenset(
    {
        "sofa2",
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_liver",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_cns_proxy_sensitivity",
        "sofa2_cns_delirium_tx_ascertainment",
        "sofa2_cns_ascertainment",
        "sofa2_renal",
        "uo_6h",
        "uo_12h",
        "uo_24h",
        "rrt_criteria",
        "rrt",
        "adv_resp",
        "ecmo",
        "ecmo_indication",
        "sedated_gcs",
        "mech_circ_support",
        "other_vaso",
        "delirium_tx",
        "delirium_tx_proxy",
        "delirium_tx_evidence",
        "motor_response",
        "delirium_positive",
    }
)


def _concepts_need_sofa2(concepts) -> bool:
    """True when any concept requires the sofa2-dict overlay."""
    return any(
        c in _SOFA2_TRIGGER_CONCEPTS or "sofa2" in str(c).lower() for c in concepts
    )


def load_concepts(
    concepts: Union[str, List[str]],
    patient_ids: Optional[Union[List, Dict]] = None,
    # 数据源参数 - 智能默认值
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    # 时间参数 - 默认与ricu一致 (interval=hours(1L))
    interval: Optional[Union[str, pd.Timedelta]] = "1h",  # ricu默认: hours(1L)
    win_length: Optional[Union[str, pd.Timedelta]] = None,
    # 聚合参数
    aggregate: Optional[Union[str, Dict]] = None,
    # SOFA相关
    keep_components: bool = False,
    # 其他
    verbose: bool = False,
    use_sofa2: bool = False,  # 新增：是否使用SOFA2字典
    merge: bool = True,  # 新增：是否合并结果
    r_compatible: bool = True,  # 默认启用ricu.R兼容格式
    dict_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    chunk_size: Optional[int] = None,
    progress: bool = False,
    parallel_workers: Optional[int] = None,
    concept_workers: Optional[int] = None,  # 改为Optional，支持自动检测
    parallel_backend: str = "auto",
    max_patients: Optional[int] = None,  # 限制加载的患者数量（自动采样）
    limit: Optional[int] = None,  # max_patients 的别名（兼容 extract_sofa_data.py）
    sample_strategy: str = "random",  # 采样策略: 'random'=seeded 随机(默认,代表性);'sorted'=按ID排序前N个(ricu-parity 用)
    batch_size: Optional[int] = None,  # 🆕 分批处理大小（默认30000，适合12GB内存）
    memory_efficient: bool = False,  # 🆕 内存优化模式（压缩数据类型）
    require_bounded_sample: bool = False,
    allow_unbounded_fallback: bool = False,
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
        allow_unbounded_fallback: 默认 False —— 传了 max_patients 却采样失败时直接
            报错，不会把「取 100 例预览」悄悄变成全库提取。确实想在采样失败时
            读全库的调用方需显式传 True。
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
    _KDIGO_OUTPUTS = {
        "aki",
        "aki_stage",
        "aki_stage_creat",
        "aki_stage_uo",
        "aki_stage_rrt",
        "uo_rt_6hr",
        "uo_rt_12hr",
        "uo_rt_24hr",
        "creat_low_past_48hr",
        "creat_low_past_7day",
        # A zero stage is a definitive negative only when this receipt says
        # ``negative_complete``.  Keep these special KDIGO outputs routable
        # through ``load_concepts`` and the module exporter.
        "aki_assessable",
        "aki_ascertainment",
        "aki_assessment_reason",
        "observation_window_coverage",
        "creatinine_ascertainment",
        "urine_ascertainment",
        "rrt_ascertainment",
    }
    _CIRC_OUTPUTS = {"circ_failure", "circ_event"}
    # Comorbidity indices live in comorbidity.py (ICD code-set matching over
    # the diagnosis table), not concept-dict.json — route like kdigo/circ.
    _COMORB_OUTPUTS = {"charlson", "elixhauser"}
    # Composite outcome endpoints (outcomes.py) — fixed-horizon mortality etc.
    _OUTCOME_OUTPUTS = {
        "mort_28d",
        "mort_90d",
        "mort_365d",
        "icu_free_days_28",
        "icu_readmission",
        "vent_free_days_28",
    }
    # Microbiology culture-positivity (microbiology.py).
    _MICRO_OUTPUTS = {"culture_positive", "bld_culture_positive"}
    _requested = set(concepts_list)
    _need_kdigo = _requested & _KDIGO_OUTPUTS
    _need_circ = _requested & _CIRC_OUTPUTS
    _need_comorb = _requested & _COMORB_OUTPUTS
    _need_outcome = _requested & _OUTCOME_OUTPUTS
    _need_micro = _requested & _MICRO_OUTPUTS
    _special = _need_kdigo | _need_circ | _need_comorb | _need_outcome | _need_micro
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
                    f"请使用关键字参数: load_concepts(concepts, database='{patient_ids}')"
                )

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
        verbose=verbose,
    )

    # 🚀 从 kwargs 中提取患者 ID（支持通过 patientunitstayid=, admissionid=, stay_id= 等传入）

    # 🚀 处理患者数量别名（兼容旧测试/benchmark）
    n_patients_alias = kwargs.pop("n_patients", None)
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
            f"收到冲突的患者上限参数: n_patients={n_patients_alias}, " f"limit={limit}"
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
        patient_ids = _sample_patient_ids(
            loader, effective_max_patients, verbose, sample_strategy=sample_strategy
        )
        # An explicit max_patients is a bound the caller asked for. If sampling
        # fails we must not quietly turn a 100-patient preview into a
        # full-database extraction — that is a multi-hour job the caller never
        # requested. Callers that genuinely want the whole database omit
        # max_patients, or opt in via allow_unbounded_fallback.
        if patient_ids is None or len(patient_ids) == 0:
            if require_bounded_sample or not allow_unbounded_fallback:
                raise RuntimeError(
                    f"Unable to sample {effective_max_patients} patients from "
                    f"{loader.database!r}; refusing to fall back to an unbounded "
                    "database load. Pass allow_unbounded_fallback=True to load "
                    "every patient instead."
                )
            logger.warning(
                "Patient sampling failed for %r; allow_unbounded_fallback=True, "
                "so loading ALL patients instead of %d.",
                loader.database,
                effective_max_patients,
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
    if (
        inferred_total_patients is None
        and patient_ids is None
        and prefer_low_memory_chunk
    ):
        try:
            inferred_total_patients = _get_total_patient_count(loader)
        except Exception as e:
            logger.debug(f"低内存 chunk 总患者数检测失败: {e}")

    # 只有当用户没有指定时才使用智能配置
    effective_concept_workers = concept_workers
    effective_parallel_workers = parallel_workers

    if concept_workers is None or parallel_workers is None:
        from ..runtime.parallel_config import (
            get_global_config,
            get_runtime_load_strategy,
        )

        runtime_strategy = get_runtime_load_strategy(
            concepts_list,
            num_patients=inferred_total_patients,
            chunk_size=chunk_size,
            requested_concept_workers=concept_workers,
            requested_parallel_workers=parallel_workers,
            requested_backend=parallel_backend if parallel_backend != "auto" else None,
            config=get_global_config(),
        )
        if concept_workers is None:
            effective_concept_workers = int(runtime_strategy["concept_workers"])
        if parallel_workers is None:
            auto_parallel = int(runtime_strategy["parallel_workers"])
            effective_parallel_workers = auto_parallel if auto_parallel > 1 else None

        if verbose and (effective_concept_workers > 1 or effective_parallel_workers):
            print(
                f"   ⚡ 智能优化: concept_workers={effective_concept_workers}, "
                f"parallel_workers={effective_parallel_workers or '不分批'}"
            )

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
        effective_chunk_size = auto_chunk_strategy["chunk_size"]
        effective_parallel_workers = auto_chunk_strategy["parallel_workers"]
        effective_concept_workers = auto_chunk_strategy["concept_workers"]
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
                sample_strategy="sorted",
            )
            if patient_ids is not None and not isinstance(patient_ids, dict):
                patient_ids = _normalize_patient_ids_for_db(
                    loader.database, patient_ids
                )
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

    from ..runtime.memory_manager import (
        auto_batch_size,
        estimate_memory_mb,
        get_available_memory_mb,
        inprocess_batch_load,
        inprocess_batch_load_streaming,
        subprocess_batch_load,
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
    # 自动检测全量加载场景
    if (
        (not auto_chunk_strategy)
        and _total_patients is None
        and patient_ids is None
        and effective_batch_size is None
    ):
        # 全量加载：查询总患者数来决定是否需要分批
        try:
            _total_patients_in_db = _get_total_patient_count(loader)
            if _total_patients_in_db and _total_patients_in_db > 1000:
                # 估算内存需求
                est_mem = estimate_memory_mb(
                    concepts_list, loader.database, _total_patients_in_db
                )

                # 🚀 稳定预算分批判定（16GB 可用性 + 确定性）：用**物理总内存**这个稳定值
                # 判定是否分批，而不是波动的"当前可用"。否则 (a) 空闲机(avail 大)时宽模块
                # 一次性→放不下→静默返回空概念(截断)，(b) 繁忙机(avail 小)时连窄模块也被
                # 误判分批→过度分批变慢。只有估算峰值 > 0.6×总内存的模块才分批，并用同一
                # 稳定预算反算 batch_size(大批少次)。实测 16GB 上只有 49-概念宽模块
                # (medications/chemistry)在大队列分批，其余 17 个模块保持一次性(最快)。
                # EASYICU_ONESHOT_BUDGET_MB 可覆盖每模块一次性内存上限(MB)。
                try:
                    _env_b = os.environ.get("EASYICU_ONESHOT_BUDGET_MB")
                    if _env_b:
                        _oneshot_budget_mb = float(_env_b)
                    else:
                        import psutil as _psb

                        _oneshot_budget_mb = (
                            _psb.virtual_memory().total / (1024 * 1024)
                        ) * 0.6
                except Exception:
                    _oneshot_budget_mb = 9830.0  # 16GB*0.6 回退
                if est_mem > _oneshot_budget_mb:
                    _total_patients = _total_patients_in_db
                    effective_batch_size = auto_batch_size(
                        concepts_list,
                        loader.database,
                        _total_patients,
                        available_memory_mb=_oneshot_budget_mb / 0.6,
                    )

                    if verbose and effective_batch_size:
                        print(
                            f"⚠️  稳定预算分批 (估算 {est_mem:.0f}MB > 预算 {_oneshot_budget_mb:.0f}MB), "
                            f"全量加载 {_total_patients} patients 分批 (batch_size={effective_batch_size})"
                        )

                    # 分批时仍用子进程隔离；进程内路径优先走流式 patient batch
                    use_subprocess = True
                    use_streaming_patient_batches = effective_batch_size is not None
                elif verbose:
                    print(
                        f"🚀 全量加载 {_total_patients_in_db} patients, "
                        f"估算 {est_mem:.0f}MB ≤ 预算 {_oneshot_budget_mb:.0f}MB, 不分批（最优速度）"
                    )
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
                _fetched_ids = _sample_patient_ids(
                    loader,
                    _total_patients_in_db,
                    verbose=False,
                    sample_strategy="sorted",
                )
                if _fetched_ids:
                    _id_col = _database_profile_or_default(loader.database).stay_id_col
                    _all_ids = list(_fetched_ids)
                    _total_patients = len(_all_ids)
                    if verbose:
                        print(
                            f"📊 分批启用: 获取 {_total_patients} 患者ID, batch_size={effective_batch_size}"
                        )
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
        est_mem = estimate_memory_mb(concepts_list, loader.database, _total_patients)
        # 🚀 稳定预算判定（同上）：用物理总内存而非波动可用，确定性地只对真正放不下的
        # 宽模块分批（medications/chemistry），窄模块保持一次性(快)；空闲机也不会因
        # "avail 大"而漏判导致宽模块一次性→静默截断。EASYICU_ONESHOT_BUDGET_MB 可覆盖。
        try:
            _env_b = os.environ.get("EASYICU_ONESHOT_BUDGET_MB")
            if _env_b:
                _oneshot_budget_mb = float(_env_b)
            else:
                import psutil as _psb

                _oneshot_budget_mb = (_psb.virtual_memory().total / (1024 * 1024)) * 0.6
        except Exception:
            _oneshot_budget_mb = 9830.0
        if est_mem > _oneshot_budget_mb:
            effective_batch_size = auto_batch_size(
                concepts_list,
                loader.database,
                _total_patients,
                available_memory_mb=_oneshot_budget_mb / 0.6,
            )
            if verbose and effective_batch_size:
                print(
                    f"⚠️  稳定预算分批 (估算 {est_mem:.0f}MB > 预算 {_oneshot_budget_mb:.0f}MB): "
                    f"{_total_patients} patients, batch_size={effective_batch_size}"
                )
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
        if os.environ.get("EASYICU_FORCE_INPROCESS_BATCH"):
            pass  # 用户显式禁用 subprocess，保持 inprocess
        elif os.environ.get("EASYICU_FORCE_SUBPROCESS_BATCH"):
            use_subprocess = True
        else:
            try:
                import psutil

                _total_mb = psutil.virtual_memory().total / (1024 * 1024)
            except Exception:
                _total_mb = get_available_memory_mb()  # 降级
            if _total_mb < 12 * 1024:
                use_subprocess = True
            elif (
                _total_mb < 32 * 1024
                and _total_patients is not None
                and _total_patients > 60000
            ):
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
    if (
        effective_batch_size is not None
        and _id_col is not None
        and _all_ids is not None
    ):
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
                _explicit_keys = {"merge", "r_compatible", "verbose"}
                subprocess_kwargs = {
                    k: v for k, v in load_kwargs.items() if k not in _explicit_keys
                }
                final_result = subprocess_batch_load(
                    concepts=(_concepts_all if _special else concepts_list),
                    database=loader.database,
                    all_patient_ids={_id_col: _all_ids},
                    batch_size=effective_batch_size,
                    data_path=(
                        str(loader.data_path) if hasattr(loader, "data_path") else None
                    ),
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
            patient_ids = _sample_patient_ids(
                loader, _total_patients, verbose, sample_strategy="sorted"
            )
            if patient_ids is not None:
                patient_ids = _normalize_patient_ids_for_db(
                    loader.database, patient_ids
                )
                _id_col = list(patient_ids.keys())[0]
                _all_ids = list(patient_ids.values())[0]
                return subprocess_batch_load(
                    concepts=(_concepts_all if _special else concepts_list),
                    database=loader.database,
                    all_patient_ids={_id_col: _all_ids},
                    batch_size=effective_batch_size,
                    data_path=(
                        str(loader.data_path) if hasattr(loader, "data_path") else None
                    ),
                    verbose=verbose,
                    merge=merge,
                    r_compatible=r_compatible,
                    dict_path=dict_path,
                    use_sofa2=use_sofa2,
                    **load_kwargs,
                )
        # inprocess_batch_load_streaming calls the *base* loader with the STRIPPED
        # concepts_list (no special routing). If specials were requested but we reached
        # here (use_subprocess was set yet _sample_patient_ids returned None, so the
        # subprocess path above was skipped), returning here would silently drop the
        # whole special group — the exact class of bug this fix closes. Fall through to
        # the non-batched loader path instead, which re-attaches specials (~L1480).
        # Trade the memory of a non-batched full load for correctness over silent loss.
        if not _special:
            return inprocess_batch_load_streaming(
                loader=loader,
                concepts=concepts_list,
                patient_batches=_iter_patient_id_batches(
                    loader,
                    effective_batch_size,
                    total_patients=_total_patients,
                    sample_strategy="sorted",
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
            **kwargs,
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
                from ..scores.kdigo_aki import load_kdigo_aki

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
                from ..scores.circ_failure import load_circ_failure

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
            from ..scores.comorbidity import load_comorbidity

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
            from ..scores.outcomes import load_outcomes

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
            from ..scores.microbiology import load_microbiology

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
            result = {
                k: _compress_dtypes(v, verbose=verbose) for k, v in result.items()
            }

    if (
        r_compatible
        and merge
        and len(concepts_list) == 1
        and isinstance(result, pd.DataFrame)
    ):
        result = _expand_public_numeric_win_tbl_output(
            result, concepts_list[0], interval
        )

    return result


# 为了兼容旧代码，保留旧的函数名
def load_concept(*args, **kwargs):
    """load_concepts的别名（向后兼容）"""
    return load_concepts(*args, **kwargs)

__all__ = [
    "MAX_WINDOW_EXPANSION_POINTS",
    "SOFA_FIXED_CHUNK_SIZE",
    "WindowExpansionError",
    "_compress_dtypes",
    "_concepts_need_sofa2",
    "_count_patient_ids_fast",
    "_database_profile_or_default",
    "_expand_public_numeric_win_tbl_output",
    "_get_auto_chunk_strategy",
    "_get_global_loader",
    "_get_patient_id_source",
    "_get_smart_workers",
    "_get_total_patient_count",
    "_guard_window_expansion",
    "_iter_patient_id_batches",
    "_normalize_patient_ids_for_db",
    "_patient_filter_values",
    "_query_patient_ids_fast",
    "_release_loader",
    "_sample_patient_ids",
    "clear_global_loader",
    "keep_cache",
    "load_concept",
    "load_concepts",
]
