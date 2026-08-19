"""Data loading utilities for ICU datasets."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from threading import RLock

import numpy as np
import pandas as pd


def _duckdb_path(p) -> str:
    """Convert a Path or string to a DuckDB-safe path string.

    On Windows, Path.__str__() produces backslash separators (e.g. ``D:\\data\\file.parquet``).
    DuckDB's ``read_parquet()`` and glob functions expect forward slashes on all platforms.
    This helper normalises the path to forward slashes so DuckDB SQL works cross-platform.
    """
    return str(p).replace('\\', '/')


def _duckdb_sql_literal(value: Any) -> str:
    """Return one safely quoted scalar for an internal DuckDB predicate."""

    if value is None:
        return "NULL"
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("non-finite values cannot be used in DuckDB filters")
        return repr(numeric)
    return "'" + str(value).replace("'", "''") + "'"


def _aumc_admissions_relation_sql(data_source: Any) -> str:
    """Return a DuckDB relation for AUMC's ICU admission clock origin.

    AUMC event timestamps are measured on a database-wide millisecond clock.
    Hourly aggregation therefore needs ``admittedat`` before it can choose a
    bucket.  Prefer the configured/resolved table when available, while also
    supporting the CSV-only layout distributed by AmsterdamUMCdb.
    """
    base_path = Path(data_source.base_path)
    candidates: List[Path] = []

    try:
        resolved = data_source._resolve_loader_from_disk("admissions")
    except (AttributeError, KeyError, TypeError, ValueError):
        resolved = None
    if isinstance(resolved, (str, Path)):
        candidates.append(Path(resolved))

    candidates.extend(
        [
            base_path / "admissions.parquet",
            base_path / "admissions.csv",
            base_path / "admissions.csv.gz",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            continue
        sql_path = _duckdb_path(candidate).replace("'", "''")
        if candidate.suffix.lower() in {".parquet", ".pq"}:
            return f"read_parquet('{sql_path}')"
        if candidate.name.lower().endswith((".csv", ".csv.gz")):
            return f"read_csv_auto('{sql_path}', header=true)"

    raise FileNotFoundError(
        "AUMC relative-time aggregation requires admissions.parquet, "
        "admissions.csv, or admissions.csv.gz under the data source path"
    )


def _sic_cases_relation_sql(data_source: Any) -> str:
    """Return the SICdb cases relation containing the ICU clock origin.

    SICdb event ``Offset`` values are seconds from the primary PDMS/hospital
    admission, while ``cases.ICUOffset`` is the actual ICU-admission offset.
    Official SICdb documentation therefore requires ``Offset - ICUOffset``
    before conversion to ICU-relative hours.
    """

    base_path = Path(data_source.base_path)
    candidates: List[Path] = []
    try:
        resolved = data_source._resolve_loader_from_disk("cases")
    except (AttributeError, KeyError, TypeError, ValueError):
        resolved = None
    if isinstance(resolved, (str, Path)):
        candidates.append(Path(resolved))
    candidates.extend(
        [
            base_path / "cases.parquet",
            base_path / "cases.csv",
            base_path / "cases.csv.gz",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            continue
        sql_path = _duckdb_path(candidate).replace("'", "''")
        if candidate.suffix.lower() in {".parquet", ".pq"}:
            return f"read_parquet('{sql_path}')"
        if candidate.name.lower().endswith((".csv", ".csv.gz")):
            return f"read_csv_auto('{sql_path}', header=true)"

    raise FileNotFoundError(
        "SICdb ICU-relative aggregation requires cases.parquet, cases.csv, "
        "or cases.csv.gz under the data source path"
    )


def _is_valid_parquet_name(name: str) -> bool:
    """Whether a filename should be treated as a real parquet file.

    Filters macOS AppleDouble sidecars (._foo.parquet) and other hidden dotfiles
    that may be auto-created on NTFS/exFAT/FUSE mounts. DuckDB's read_parquet
    glob would otherwise pick them up and crash with 'No magic bytes found'.
    """
    return name.endswith('.parquet') and not name.startswith('.')


def _enumerate_bucket_parquet_files(directory) -> List[str]:
    """Return a sorted list of valid parquet file paths under a bucket-style dir.

    Supports both layouts:
    - directory/bucket_id=N/*.parquet  (canonical EasyICU bucket dir)
    - directory/*.parquet              (flat parquet dir)
    Always filters out AppleDouble / hidden sidecar files so DuckDB
    read_parquet([...]) never sees garbage. Returns DuckDB-safe forward-slash
    paths (suitable for embedding in SQL literals).

    ⚡ 性能：在 macFUSE/NTFS 等慢盘上直接 glob 82 个目录每次 ~12s。
    若目录里存在 ``_BUCKET_MANIFEST.json``（与 ``_COMPLETE`` mtime 绑定），
    直接读 manifest，跳过所有 listdir。
    """
    import json as _json
    from pathlib import Path as _Path
    base = _Path(directory)
    files: List[str] = []
    if not base.exists():
        return files

    # Fast path: 跨进程持久化 manifest（由 ICUDataSource._get_bucket_layout 写入）
    complete_marker = base / "_COMPLETE"
    manifest_path = base / "_BUCKET_MANIFEST.json"
    if complete_marker.exists() and manifest_path.exists():
        try:
            stat = complete_marker.stat()
            with manifest_path.open('r', encoding='utf-8') as fh:
                m = _json.load(fh)
            if (
                isinstance(m, dict)
                and m.get('complete_mtime_ns') == stat.st_mtime_ns
                and m.get('complete_size') == stat.st_size
                and isinstance(m.get('files_by_bucket'), dict)
            ):
                for rels in m['files_by_bucket'].values():
                    for rel in rels:
                        name = _Path(rel).name
                        if _is_valid_parquet_name(name):
                            files.append(_duckdb_path(base / rel))
                files.sort()
                return files
        except Exception:
            pass  # manifest unreadable → live scan below

    # bucket layout first (cheap glob)
    bucket_dirs = [d for d in base.glob('bucket_id=*') if d.is_dir()]
    if bucket_dirs:
        for sub in bucket_dirs:
            for p in sub.glob('*.parquet'):
                if _is_valid_parquet_name(p.name):
                    files.append(_duckdb_path(p))
    else:
        # flat or hive_partitioning=false case
        for p in base.rglob('*.parquet'):
            if _is_valid_parquet_name(p.name):
                files.append(_duckdb_path(p))
    files.sort()
    return files


def _duckdb_read_parquet_clause(directory, *, union_by_name: bool = True,
                                 hive_partitioning: Optional[bool] = None) -> str:
    """Build a DuckDB ``read_parquet([...])`` SQL fragment from a directory.

    Returns a SQL clause like ``read_parquet(['a','b'], union_by_name=true)``.
    Filters out AppleDouble / hidden sidecar files at enumeration time so
    no garbage path enters DuckDB. Falls back to a glob pattern only when the
    directory is empty (DuckDB will still error, but with a clearer message).
    """
    files = _enumerate_bucket_parquet_files(directory)
    opts: List[str] = []
    if union_by_name:
        opts.append('union_by_name=true')
    if hive_partitioning is not None:
        opts.append(f"hive_partitioning={'true' if hive_partitioning else 'false'}")
    opt_str = (', ' + ', '.join(opts)) if opts else ''
    if not files:
        # Fall back to glob (will most likely error downstream but preserves prior behaviour)
        return f"read_parquet('{_duckdb_path(directory)}/**/*.parquet'{opt_str})"
    files_sql = '[' + ', '.join(f"'{f}'" for f in files) + ']'
    return f"read_parquet({files_sql}{opt_str})"


def _coerce_string_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas StringDtype columns to object dtype for pandas 3.0 compatibility.

    In pandas 3.0, reading parquet/arrow data returns string columns as pd.StringDtype
    (backed by pyarrow) instead of numpy object. This causes TypeError when assigning
    float values using .loc[mask, col], which is common in EasyICU callbacks.

    This function converts StringDtype columns back to numpy object dtype so that
    existing code works correctly without modification.
    """
    if df.empty:
        return df
    str_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.StringDtype)]
    if str_cols:
        df = df.copy()
        for col in str_cols:
            df[col] = df[col].astype(object)
    return df


def _arrow_to_pandas_compat(arrow_table, **kwargs) -> pd.DataFrame:
    """Convert Arrow table to pandas DataFrame with StringDtype→object coercion.

    Wraps arrow_table.to_pandas() to ensure string columns use numpy object dtype
    rather than pd.StringDtype, preventing pandas 3.0 compatibility issues.
    Also downcasts float64 to float32 for memory savings (~50% reduction).
    """
    try:
        import pyarrow as pa
        # Map all string/large_string types to numpy object to avoid pd.StringDtype
        # Map float64 → float32 for memory savings (safe for medical values)
        def types_mapper(arrow_type):
            if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
                return np.dtype('O')
            #if pa.types.is_float64(arrow_type):
            #    return np.dtype('float32')
            return None
        df = arrow_table.to_pandas(types_mapper=types_mapper, **kwargs)
    except Exception:
        df = arrow_table.to_pandas(**kwargs)
        df = _coerce_string_dtypes(df)
    return df

from .config import DataSourceConfig, DataSourceRegistry, DatasetOptions
from .table import ICUTable

# 全局调试开关 - 设置为 False 可以减少输出
DEBUG_MODE = False
logger = logging.getLogger(__name__)

# 🚀 DuckDB 连接复用：每个线程维护一个 in-memory 连接，避免反复创建/销毁
import threading as _threading
_duckdb_local = _threading.local()


def _default_duckdb_threads() -> int:
    """Bound one DuckDB connection by the effective-memory worker policy."""

    import os

    raw = os.environ.get("EASYICU_DUCKDB_THREADS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("Ignoring invalid EASYICU_DUCKDB_THREADS=%r", raw)
    try:
        from easyicu.runtime.parallel_config import get_global_config

        return max(1, min(8, int(get_global_config().max_workers)))
    except Exception:
        return 1


def _default_duckdb_memory_limit_gb() -> float:
    """Size DuckDB buffers from process-effective, not host, free memory."""

    try:
        from easyicu.runtime.memory_manager import get_effective_memory_info

        available_gb = (
            get_effective_memory_info().effective_available_mb / 1024.0
        )
        return max(1.0, min(available_gb * 0.15, 4.0))
    except Exception:
        return 1.0

def _get_duckdb_connection():
    """获取当前线程的 DuckDB 连接（复用已有连接）"""
    import duckdb
    con = getattr(_duckdb_local, 'con', None)
    if con is None:
        con = duckdb.connect()
        con.execute("SET timezone='UTC'")
        con.execute("SET enable_progress_bar = false")
        con.execute("SET enable_progress_bar_print = false")
        # 🚀 性能优化: 支持通过环境变量限制 DuckDB 线程数
        # 并行模式下 6 个进程 × 默认线程数(half cores) → 严重过度订阅
        import os
        con.execute(f"SET threads = {_default_duckdb_threads()}")
        # 🚀 限制 DuckDB 内存使用，防止在大内存服务器上分配过多缓冲区
        # 默认值为系统内存的 75%（1.5TB 服务器上约 1.1TB），严重浪费
        # 使用保守自适应: min(effective available * 0.15, 4GB),
        # 下限 1GB。配合患者分批和 spill，避免 8GB 工作站上
        # DuckDB 缓冲区与 Pandas/Arrow 嵌套工作集争抢内存。
        _mem_limit = os.environ.get('EASYICU_DUCKDB_MEMORY_LIMIT')
        if _mem_limit:
            con.execute(f"SET memory_limit = '{_mem_limit}'")
        else:
            _duck_gb = _default_duckdb_memory_limit_gb()
            con.execute(f"SET memory_limit = '{_duck_gb:.1f}GB'")
        # 磁盘溢出目录：**纯 opt-in**，默认不动 DuckDB 原生行为（普通用户数据在
        # 本地盘，默认 spill 就够用，无需额外设置）。只有当用户显式设了
        # EASYICU_DUCKDB_TEMP_DIR（例如数据放在慢速外置/USB 盘、想把 spill 落到
        # 内部 SSD）时才覆盖。EASYICU_DUCKDB_MAX_TEMP_SIZE 可选限制溢出上限。
        _spill_dir = os.environ.get('EASYICU_DUCKDB_TEMP_DIR')
        if _spill_dir:
            try:
                os.makedirs(_spill_dir, exist_ok=True)
                con.execute(f"SET temp_directory = '{_spill_dir}'")
                _max_temp = os.environ.get('EASYICU_DUCKDB_MAX_TEMP_SIZE')
                if _max_temp:
                    con.execute(f"SET max_temp_directory_size = '{_max_temp}'")
            except Exception:
                pass
        _duckdb_local.con = con
    return con

def _close_duckdb_connections():
    """关闭所有缓存的 DuckDB 连接"""
    con = getattr(_duckdb_local, 'con', None)
    if con is not None:
        try:
            con.close()
        except Exception:
            pass
        _duckdb_local.con = None

# 🚀 大表预过滤：只加载概念字典声明过的 itemids/variableids。
# 原始表很大，过滤后性能提升明显。白名单必须跟随 concept-dict.json /
# sofa2-dict.json 自动变化，避免字典新增 id 后被底层大表预过滤丢掉。
AUMC_NUMERICITEMS_EXTRA_ITEMIDS: set[int] = set()
MIIV_CHARTEVENTS_EXTRA_ITEMIDS: set[int] = set()
MIIV_LABEVENTS_EXTRA_ITEMIDS: set[int] = set()
MIMIC_DEMO_CHARTEVENTS_EXTRA_ITEMIDS: set[int] = set()
MIMIC_DEMO_LABEVENTS_EXTRA_ITEMIDS: set[int] = set()
HIRID_OBSERVATIONS_EXTRA_VARIABLEIDS: set[int] = set()


def _source_ids(source_def: Mapping[str, Any]) -> set[int]:
    ids = source_def.get("ids")
    if isinstance(ids, int):
        return {ids}
    if isinstance(ids, list):
        return {item for item in ids if isinstance(item, int)}
    return set()


def _dictionary_source_ids(dataset: str, table: str, sub_var: str) -> set[int]:
    import json
    from importlib import resources

    source_ids: set[int] = set()
    data_package = "easyicu.data"
    for filename in ("concept-dict.json", "sofa2-dict.json"):
        resource = resources.files(data_package).joinpath(filename)
        payload = json.loads(resource.read_text(encoding="utf8"))
        if not isinstance(payload, Mapping):
            continue
        for concept_def in payload.values():
            if not isinstance(concept_def, Mapping):
                continue
            sources = concept_def.get("sources")
            if not isinstance(sources, Mapping):
                continue
            dataset_sources = sources.get(dataset, [])
            if not isinstance(dataset_sources, list):
                continue
            for source_def in dataset_sources:
                if not isinstance(source_def, Mapping):
                    continue
                if (
                    source_def.get("table") == table
                    and source_def.get("sub_var") == sub_var
                ):
                    source_ids.update(_source_ids(source_def))
    return source_ids


AUMC_NUMERICITEMS_ITEMIDS = (
    _dictionary_source_ids("aumc", "numericitems", "itemid")
    | AUMC_NUMERICITEMS_EXTRA_ITEMIDS
)
MIIV_CHARTEVENTS_ITEMIDS = (
    _dictionary_source_ids("miiv", "chartevents", "itemid")
    | MIIV_CHARTEVENTS_EXTRA_ITEMIDS
)
MIIV_LABEVENTS_ITEMIDS = (
    _dictionary_source_ids("miiv", "labevents", "itemid")
    | MIIV_LABEVENTS_EXTRA_ITEMIDS
)
MIMIC_DEMO_CHARTEVENTS_ITEMIDS = (
    _dictionary_source_ids("mimic_demo", "chartevents", "itemid")
    | MIMIC_DEMO_CHARTEVENTS_EXTRA_ITEMIDS
)
MIMIC_DEMO_LABEVENTS_ITEMIDS = (
    _dictionary_source_ids("mimic_demo", "labevents", "itemid")
    | MIMIC_DEMO_LABEVENTS_EXTRA_ITEMIDS
)

# eICU nursecharting has multiple semantic columns in the dictionary
# (nursingchartcelltypevalname, nursingchartcelltypevallabel, nursingchartvalue).
# Keep this only as a concept-specific exact-filter helper; do not apply it as a
# global table filter, because a single-column whitelist drops label/value regex
# sources such as CVP, ECMO, sedation, and delirium.
EICU_NURSECHARTING_IDS = {
    # GCS 相关
    'GCS Total', 'Eyes', 'Sedation Score', 'Motor', 'Verbal',
    # 谵妄相关 (SOFA-2 delirium_positive) - 使用 nursingchartcelltypevalname 的值
    'Value', 'Delirium Score', 'Delirium Scale',
    # ECMO 相关 (通过 O2 Admin Device 记录)
    'O2 Admin Device',
}

# 🚀 VALUE-TO-ITEMID 映射表：用于优化 sub_var: value 类型的概念加载
# 当概念定义使用 sub_var: value 时，需要扫描全表来匹配 value
# 但如果我们知道哪些 itemid 包含目标 value，就可以使用 bucket 优化
# 结构: {db_name: {table_name: {value_col: {value: [itemids]}}}}
# 例如: ett_gcs (miiv) 使用 value='No Response-ETT'，对应 itemid=223900
VALUE_TO_ITEMID_MAPPING = {
    'miiv': {
        'chartevents': {
            'value': {
                # ett_gcs: 用于识别气管插管状态
                # value='No Response-ETT' 只出现在 itemid=223900 (GCS - Verbal Response)
                'No Response-ETT': {223900},
                '1.0 ET/Trach': {223900},  # 同样是插管状态
            }
        }
    },
    'mimic': {
        'chartevents': {
            'value': {
                'No Response-ETT': {223900, 723},
                '1.0 ET/Trach': {223900, 723},
            }
        }
    },
    'mimic_demo': {
        'chartevents': {
            'value': {
                'No Response-ETT': {223900},
                '1.0 ET/Trach': {223900},
            }
        }
    },
}

HIRID_OBSERVATIONS_VARIABLEIDS = (
    _dictionary_source_ids("hirid", "observations", "variableid")
    | HIRID_OBSERVATIONS_EXTRA_VARIABLEIDS
)

# 🚀 性能优化：最小必要列集（自动应用）
MINIMAL_COLUMNS = {
    'chartevents': ['stay_id', 'charttime', 'itemid', 'valuenum', 'valueuom', 'value'],
    'labevents': ['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum', 'valueuom'],
    'outputevents': ['stay_id', 'charttime', 'itemid', 'value'],
    'procedureevents': ['stay_id', 'starttime', 'endtime', 'itemid', 'value'],  # 添加endtime列用于WinTbl
    'datetimeevents': ['stay_id', 'charttime', 'itemid', 'value'],
    'inputevents': ['stay_id', 'starttime', 'endtime', 'itemid', 'amount', 'amountuom'],
    'icustays': ['stay_id', 'subject_id', 'hadm_id', 'intime', 'outtime', 'los'],
    'd_items': ['itemid', 'label', 'category'],
}

class FilterOp(str, enum.Enum):
    """Supported filter operations for table loading."""

    EQ = "=="
    IN = "in"
    BETWEEN = "between"
    REGEX = "regex"

@dataclass
class FilterSpec:
    """Declarative filter specification for table loading."""

    column: str
    op: FilterOp
    value: Any
    metadata: Optional[Dict[str, Any]] = field(default=None)  # ✅ 存储额外信息，如原始 stay_id
    _value_set: Optional[set] = field(default=None, init=False, repr=False)  # ⚡ 缓存set版本的value

    def __post_init__(self):
        """⚡ 性能优化: 预计算value的set形式用于isin操作"""
        if self.op == FilterOp.IN:
            if isinstance(self.value, str):
                self._value_set = {self.value}
            elif hasattr(self.value, '__iter__'):
                self._value_set = set(self.value)
            else:
                self._value_set = {self.value}

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        # ⚡ 性能优化: 返回视图而非副本，由调用者决定是否需要copy
        if self.op == FilterOp.EQ:
            mask = frame[self.column] == self.value
            return frame.loc[mask]
        if self.op == FilterOp.IN:
            # ⚡ 使用预计算的set，避免每次都list转换
            mask = frame[self.column].isin(self._value_set)
            return frame.loc[mask]
        if self.op == FilterOp.BETWEEN:
            lower, upper = self.value
            mask = frame[self.column].between(lower, upper)
            return frame.loc[mask]
        if self.op == FilterOp.REGEX:
            # Regex filtering for rgx_itm concepts (e.g., drug names)
            mask = frame[self.column].str.contains(self.value, case=False, na=False, regex=True)
            return frame.loc[mask]
        raise ValueError(f"Unsupported filter operation: {self.op}")

class ICUDataSource:
    """Lightweight facade that loads tables for a concrete dataset instance."""

    # 全局格式优先级配置
    _global_format_priority: Optional[List[str]] = None

    @classmethod
    def set_format_priority(cls, priority: List[str]) -> None:
        """设置全局文件格式优先级
        
        Args:
            priority: 格式列表（当前只支持 ['parquet']）
        
        Examples:
            >>> # 只使用 Parquet 格式（纯 Python，无需 R）
            >>> ICUDataSource.set_format_priority(['parquet'])
        """
        cls._global_format_priority = priority

    @classmethod
    def get_format_priority(cls) -> List[str]:
        """获取当前的格式优先级配置
        
        Returns:
            格式列表，按优先级排序
        """
        # 1. 如果设置了全局优先级，使用全局配置
        if cls._global_format_priority is not None:
            return cls._global_format_priority
        
        # 2. 检查环境变量
        import os
        env_priority = os.environ.get('EASYICU_FORMAT_PRIORITY')
        if env_priority:
            return [fmt.strip() for fmt in env_priority.split(',')]
        
        # 只支持 Parquet 格式
        return ['parquet']

    def __init__(
        self,
        config: DataSourceConfig,
        *,
        base_path: str | Path | None = None,
        table_sources: Optional[Mapping[str, Any]] = None,
        registry: Optional[DataSourceRegistry] = None,
        default_format: str = "parquet",
        enable_cache: bool = True,
        format_priority: Optional[List[str]] = None,
    ) -> None:
        """初始化数据源
        
        Args:
            config: 数据源配置
            base_path: 数据文件基础路径
            table_sources: 表数据源映射（可选）
            registry: 数据源注册表（可选）
            default_format: 默认格式（已废弃，使用 format_priority）
            enable_cache: 是否启用缓存
            format_priority: 文件格式优先级（可选），例如 ['parquet', 'fst', 'csv']
                           如果未指定，使用全局配置或环境变量
        """
        self.config = config
        self.base_path = Path(base_path) if base_path else None
        self._table_sources: MutableMapping[str, Any] = dict(table_sources or {})
        self.default_format = default_format
        self.registry = registry
        self._dataset_sources: Dict[str, DatasetOptions] = {
            name: table.dataset
            for name, table in self.config.tables.items()
            if table.dataset is not None
        }
        self.enable_cache = enable_cache
        self._table_cache: dict = {}  # 缓存已加载的原始表数据
        self._preloaded_tables: dict = {}  # 🚀 预加载的完整表（用于多患者批处理）
        self._bucket_dir_logged: set = set()  # 🔧 已打印日志的分桶目录（避免重复日志）
        self._bucket_dir_cache: dict = {}  # 缓存已发现的分桶目录（只缓存正结果）
        self._flat_parquet_dir_cache: dict = {}  # 缓存已发现的扁平 parquet 目录（只缓存正结果）
        self._bucket_hash_cache: dict = {}  # 缓存 itemid 集合 -> bucket 集合
        self._bucket_hash_item_cache: dict = {}  # 缓存单个 itemid -> bucket
        self._bucket_layout_cache: dict = {}  # 缓存 bucket_id 目录与 parquet 文件列表
        self._parquet_columns_cache: dict = {}  # 缓存 parquet 文件集合的 schema 列名
        self._parquet_file_columns_cache: dict = {}  # 缓存单个 parquet 文件 schema 列名
        self.format_priority = format_priority or self.get_format_priority()
        self._lock = RLock()

    def register_table_source(self, table: str, source: Any) -> None:
        """Register a callable/file path used to load ``table``."""
        self._table_sources[table] = source
    
    def clear_cache(self) -> None:
        """清除表缓存,释放内存。"""
        with self._lock:
            self._table_cache.clear()
            self._preloaded_tables.clear()
            self._bucket_dir_cache.clear()
            self._flat_parquet_dir_cache.clear()
            self._bucket_hash_cache.clear()
            self._bucket_hash_item_cache.clear()
            self._bucket_layout_cache.clear()
            self._parquet_columns_cache.clear()
            self._parquet_file_columns_cache.clear()
    
    def clear(self) -> None:
        """Alias for clear_cache, used by CacheManager."""
        self.clear_cache()
    
    def preload_tables(self, table_names: List[str], patient_ids: Optional[List[int]] = None) -> None:
        """
        🚀 预加载大表到内存，避免重复I/O
        
        Args:
            table_names: 要预加载的表名列表
            patient_ids: 可选的患者ID列表，用于预过滤
        """
        base_patient_ids = list(patient_ids) if patient_ids is not None else None
        for table_name in table_names:
            with self._lock:
                if table_name in self._preloaded_tables:
                    continue
                
            # 加载完整表（使用最小列集）
            columns = MINIMAL_COLUMNS.get(table_name)
            
            # 不使用filters，直接加载完整表
            table = self.load_table(table_name, columns=columns, verbose=False)
            df = table.dataframe()  # 修正：这是个方法
            
            # 如果提供了patient_ids，预过滤
            if base_patient_ids is not None:
                id_col = None
                filter_ids = base_patient_ids
                if 'stay_id' in df.columns:
                    id_col = 'stay_id'
                elif 'subject_id' in df.columns:
                    # 需要从icustays获取subject_id映射
                    if table_name != 'icustays':
                        icustays = self.load_table('icustays', columns=['stay_id', 'subject_id'], verbose=False)
                        icustays_df = icustays.dataframe()
                        subject_ids = icustays_df[icustays_df['stay_id'].isin(base_patient_ids)]['subject_id'].dropna().astype(int).unique()
                        id_col = 'subject_id'
                        filter_ids = subject_ids.tolist()
                
                if id_col and filter_ids is not None:
                    df = df[df[id_col].isin(filter_ids)]
            
            with self._lock:
                self._preloaded_tables[table_name] = df
            logger.info(f"📦 预加载表 {table_name}: {len(df):,}行")
    
    def get_preloaded_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """获取预加载的表"""
        with self._lock:
            table = self._preloaded_tables.get(table_name)
        return table
    
    def get_cache_info(self) -> dict:
        """获取缓存信息。"""
        with self._lock:
            total_size = sum(df.memory_usage(deep=True).sum() for df in self._table_cache.values())
            cached_tables = len(self._table_cache)
        return {
            'cached_tables': cached_tables,
            'memory_mb': total_size / (1024 * 1024)
        }

    def load_table(
        self,
        table_name: str,
        *,
        columns: Optional[Iterable[str]] = None,
        filters: Optional[Iterable[FilterSpec]] = None,
        verbose: bool = False,
    ) -> ICUTable:
        """Load and wrap a table according to the stored configuration."""
        
        table_cfg = self.config.get_table(table_name)

        # ✅ 关键修复：提前保存原始 stay_id 过滤器值
        # 因为后续对于 hospital tables (labevents等) 会将 stay_id 转换成 subject_id/hadm_id
        # 但转换后无法恢复原始 stay_id，导致 join 后引入额外患者
        # 🔧 FIX 2026-02-07: 添加 services 表（用于 adm 概念），只有 hadm_id 无 stay_id
        # 🔧 FIX 2026-02-07: 添加 mimic (MIMIC-III) 支持，使用 icustay_id
        hospital_tables = ['prescriptions', 'labevents', 'microbiologyevents', 'emar', 'pharmacy', 'services']
        original_stay_ids = None
        if table_name in hospital_tables and self.config.name in ['miiv', 'mimic_demo', 'mimic']:
            if filters:
                for spec in filters:
                    # 🔧 FIX: MIMIC-III 使用 icustay_id
                    id_col_to_check = 'icustay_id' if self.config.name == 'mimic' else 'stay_id'
                    if spec.column == id_col_to_check and spec.op == FilterOp.IN:
                        original_stay_ids = set(spec.value)  # 保存原始目标 stay_ids
                        print(f"💾 [{table_name}] 保存原始 {id_col_to_check} 过滤器: {len(original_stay_ids)} 个患者")
                        break

        # 🚀 优化1：优先使用预加载的表
        preloaded_frame = None
        with self._lock:
            if table_name in self._preloaded_tables:
                preloaded_frame = self._preloaded_tables[table_name]

        if preloaded_frame is not None:
            frame_view = preloaded_frame

            # 应用列过滤（避免提前复制整张表）
            if columns is not None:
                available_cols = [c for c in columns if c in frame_view.columns]
                frame_view = frame_view.loc[ :, available_cols]

            # 应用行过滤
            if filters:
                frame_filtered = frame_view
                for spec in filters:
                    frame_filtered = spec.apply(frame_filtered)
            else:
                frame_filtered = frame_view

            frame = frame_filtered.copy()
        else:
            # 🚀 优化2：列选择策略
            # - 对于宽表（vitalperiodic等）：只加载 ID列 + 时间列 + 传入的 value_var
            # - 对于长表（chartevents等）：使用 MINIMAL_COLUMNS_MAP 预定义列集
            from .load_concepts import MINIMAL_COLUMNS_MAP, USE_MINIMAL_COLUMNS
            
            # 🎯 宽表列表：这些表的值直接存储在列名中（如 heartrate, temperature）
            # 不使用 itemid 过滤，应该只加载概念所需的值列
            WIDE_TABLES = {'vitalperiodic', 'vitalaperiodic'}
            
            # 🚀 宽表预加载优化：第一次加载时预加载所有常用value列
            # 这样后续概念可以直接从缓存取，避免重复读取parquet
            
            # 🚀 宽表优化：识别value列用于NULL过滤
            # 宽表的value列就是传入的columns中除了ID列和时间列以外的列
            wide_table_value_columns = None  # 用于DuckDB WHERE value IS NOT NULL优化
            
            if table_name in WIDE_TABLES and columns is not None:
                # 对于宽表，使用动态列选择：ID列 + 时间列 + 传入的值列
                table_cfg = self.config.get_table(table_name)
                base_cols = set()
                id_and_time_cols = set()  # 记录ID列和时间列
                
                # 添加 ID 列（优先使用表配置，否则使用 icustay 级别的 ID）
                if table_cfg.defaults.id_var:
                    base_cols.add(table_cfg.defaults.id_var)
                    id_and_time_cols.add(table_cfg.defaults.id_var)
                else:
                    # 从数据库 id_cfg 获取 icustay 级别 ID
                    # eICU: patientunitstayid, MIIV: stay_id, AUMC: admissionid
                    icustay_cfg = self.config.id_configs.get('icustay')
                    if icustay_cfg:
                        base_cols.add(icustay_cfg.id)
                        id_and_time_cols.add(icustay_cfg.id)
                    else:
                        # 回退到默认 ID
                        default_id = self.config.get_default_id()
                        if default_id:
                            base_cols.add(default_id)
                            id_and_time_cols.add(default_id)
                    
                # 添加时间列
                if table_cfg.defaults.index_var:
                    base_cols.add(table_cfg.defaults.index_var)
                    id_and_time_cols.add(table_cfg.defaults.index_var)
                    
                # 合并传入的值列（如 heartrate）
                for col in columns:
                    base_cols.add(col)
                
                # 🚀 提取value列（用于NULL过滤）= 传入的columns - ID列 - 时间列
                value_cols = [c for c in columns if c not in id_and_time_cols]
                if value_cols:
                    wide_table_value_columns = value_cols
                    
                columns = list(base_cols)
                if DEBUG_MODE:
                    logger.debug(f"🎯 宽表动态列选择: {table_name} -> {columns}")
                    logger.debug(f"🎯 宽表value列(用于NULL过滤): {wide_table_value_columns}")
                    
            elif USE_MINIMAL_COLUMNS and table_name in MINIMAL_COLUMNS_MAP:
                base_columns = list(MINIMAL_COLUMNS_MAP[table_name])
                
                # 🔧 FIX 2026-01-26: MIMIC-III 使用 icustay_id 而非 stay_id
                # 将 stay_id 替换为 icustay_id（对于 MIMIC-III）
                db_name = self.config.name if hasattr(self, 'config') and hasattr(self.config, 'name') else ''
                if db_name == 'mimic' and 'stay_id' in base_columns:
                    base_columns = [c if c != 'stay_id' else 'icustay_id' for c in base_columns]
                    if DEBUG_MODE:
                        logger.debug("🔄 MIMIC-III 列映射: stay_id -> icustay_id")
                
                if columns is not None:
                    # 合并最小列集和传入的额外列（去重）
                    extra_cols = [c for c in columns if c not in base_columns]
                    columns = base_columns + extra_cols
                    if DEBUG_MODE and extra_cols:
                        logger.debug(f"扩展最小列集: {table_name} + {extra_cols} -> {len(columns)}列")
                else:
                    columns = base_columns
                    if DEBUG_MODE:
                        logger.debug(f"应用最小列集优化: {table_name} -> {len(columns)}列")
            else:
                # 对于不在 MINIMAL_COLUMNS_MAP 中的表，加载所有列
                # 这确保 AUMC/HiRID 等数据库的表能正确加载必要的 ID 和值列
                columns = None

            # 提取 patient_ids 过滤器用于分区预过滤
            patient_ids_filter = None
            # 🚀 大表精确 selector 下推：提取除患者 ID 以外的
            # sub_var/ids IN 过滤器，在 DuckDB/PyArrow 读取层就缩小数据。
            # 这不能依赖列名白名单：eICU 的 nursingchartvalue、
            # respchartvaluelabel 等字符串 selector 同样来自已审计的概念字典。
            concept_itemid_filter = None  # (column_name, set_of_ids)
            
            # 🚀 优化：对于缺少 stay_id 的表（如 labevents），如果过滤条件是 stay_id，
            # 需要先查 icustays 转换成 hadm_id 或 subject_id，以便在读取 parquet 时就能过滤
            # 🔧 FIX 2026-02-07: 添加 services 表（用于 adm 概念）
            hospital_tables = ['prescriptions', 'labevents', 'microbiologyevents', 'emar', 'pharmacy', 'services']
            mapped_filter = None
            
            if filters:
                for spec in filters:
                    # 支持各数据库的ID列名
                    id_columns = ['subject_id', 'icustay_id', 'hadm_id', 'stay_id',  # MIMIC
                                 'admissionid', 'patientid',  # AUMC
                                 'patientunitstayid',  # eICU
                                 'patientid',  # HiRID
                                 'CaseID', 'caseid']  # 🔧 FIX 2026-01-26: 添加 SICdb CaseID
                    
                    # 🚀 VALUE-TO-ITEMID 映射优化：处理 sub_var: value 类型的概念
                    # 例如 ett_gcs 使用 value='No Response-ETT'，我们将其转换为 itemid=223900
                    # 这样可以使用 bucket 优化，而不是扫描全表
                    if spec.op == FilterOp.IN and spec.column == 'value':
                        db_name = self.config.name
                        value_mapping = VALUE_TO_ITEMID_MAPPING.get(db_name, {}).get(table_name, {}).get('value', {})
                        if value_mapping:
                            # 收集所有匹配的 itemid
                            mapped_itemids = set()
                            filter_values = spec.value
                            if isinstance(filter_values, str):
                                filter_values = [filter_values]
                            for val in filter_values:
                                if val in value_mapping:
                                    mapped_itemids.update(value_mapping[val])
                            
                            if mapped_itemids:
                                # 使用 itemid 进行 bucket 过滤（快 50x）
                                concept_itemid_filter = ('itemid', mapped_itemids)
                                logger.debug(f"🔄 VALUE-TO-ITEMID映射: value IN {filter_values} -> itemid IN {mapped_itemids}")
                                # 注意：仍需在内存中应用 value 过滤（因为 itemid 可能包含多种 value）
                                # 这个过滤会在后面的 filters 循环中应用
                                continue

                    # 概念 resolver 在这里只会产生 patient selector 与
                    # source.sub_var selector。患者列在下方单独处理；其他
                    # exact-IN selector 均可以无损前移，不应因新数据库
                    # 使用了新 sub_var 列名而退化为整批载入 Pandas。
                    # ``value`` 的已知 value→itemid 快路径必须先运行；仅当
                    # 没有映射时才直接下推原始 value selector。
                    if spec.op == FilterOp.IN and spec.column not in id_columns:
                        ids = spec.value
                        if isinstance(ids, (list, tuple, set, frozenset)):
                            ids = set(ids)
                        elif not isinstance(ids, set):
                            ids = {ids}
                        concept_itemid_filter = (spec.column, ids)
                        if DEBUG_MODE:
                            logger.info(
                                "🎯 概念精确 selector: %s IN %d 个值",
                                spec.column,
                                len(ids),
                            )
                        continue  # 继续处理，找 patient_id 过滤器
                    
                    if spec.op == FilterOp.IN and spec.column in id_columns:
                        patient_ids_filter = spec
                        
                        # 特殊处理：如果表是 hospital table 且过滤器是 stay_id 或 icustay_id
                        # 🔧 FIX 2026-01-26: 添加 mimic (MIMIC-III) 支持，使用 icustay_id
                        is_mimic_db = self.config.name in ['miiv', 'mimic_demo', 'mimic']
                        is_id_filter = spec.column in ['stay_id', 'icustay_id']
                        if table_name in hospital_tables and is_mimic_db and is_id_filter:
                            try:
                                if verbose:
                                    logger.info(f"🔄 [{table_name}] 将 {spec.column} 过滤器转换为 hadm_id 以优化读取...")
                                
                                # 加载 icustays 获取映射
                                # MIMIC-III 使用 icustay_id，MIMIC-IV 使用 stay_id
                                id_col = 'icustay_id' if self.config.name == 'mimic' else 'stay_id'
                                icustays_map = self.load_table(
                                    'icustays', 
                                    columns=[id_col, 'hadm_id'], 
                                    filters=[spec],
                                    verbose=False
                                )
                                icustays_df = icustays_map.dataframe()
                                
                                # 获取对应的 hadm_id 列表
                                valid_hadm_ids = icustays_df['hadm_id'].dropna().unique()
                                
                                if len(valid_hadm_ids) > 0:
                                    # 创建新的过滤器
                                    mapped_filter = FilterSpec(column='hadm_id', op=FilterOp.IN, value=valid_hadm_ids)
                                    patient_ids_filter = mapped_filter
                                    if verbose:
                                        logger.info(f"✅ [{table_name}] 映射成功: {len(spec.value)} stay_ids -> {len(valid_hadm_ids)} hadm_ids")
                            except Exception as e:
                                logger.warning(f"⚠️ [{table_name}] 过滤器映射失败: {e}")
                        
                        # 只在verbose模式下输出，且只输出一次
                        if verbose:
                            cache_key = f"_filter_logged_{table_name}"
                            if not hasattr(self, cache_key) or not getattr(self, cache_key, False):
                                if DEBUG_MODE:
                                    logger.debug(f"检测到患者ID过滤器: {len(spec.value)} 个患者, 列={spec.column}")
                                setattr(self, cache_key, True)
                        break

            frame = self._load_raw_frame(
                table_name, columns, 
                patient_ids_filter=patient_ids_filter,
                concept_itemid_filter=concept_itemid_filter,
                wide_table_value_columns=wide_table_value_columns  # 🚀 传递宽表value列用于NULL过滤
            )

            # 应用过滤器，但跳过已经被 patient_ids_filter 处理的过滤器
            # 关键修复：如果 patient_ids_filter 被转换过（例如 stay_id → hadm_id），
            # 不应该再应用原始的 stay_id 过滤器（因为表没有 stay_id 列）
            if filters:
                for spec in filters:
                    # 跳过已经作为 patient_ids_filter 处理的过滤器
                    if patient_ids_filter is not None:
                        # 如果原始过滤器的列名和 patient_ids_filter 的列名不同，
                        # 说明过滤器被转换过（例如 stay_id → hadm_id）
                        if spec.column != patient_ids_filter.column and spec.op == FilterOp.IN:
                            # 检查是否是同类型的 ID 过滤器（都是 patient ID 类型）
                            id_columns_set = {'subject_id', 'icustay_id', 'hadm_id', 'stay_id',
                                             'admissionid', 'patientid', 'patientunitstayid',
                                             'CaseID', 'caseid'}  # 🔧 FIX 2026-01-26: 添加 SICdb CaseID
                            if spec.column in id_columns_set and patient_ids_filter.column in id_columns_set:
                                # 这个过滤器已经被转换处理了，跳过
                                continue
                        elif spec.column == patient_ids_filter.column and spec.op == patient_ids_filter.op:
                            # 完全相同的过滤器，已经在 _load_raw_frame 中处理，跳过
                            continue
                    
                    # 安全检查：只应用列存在的过滤器
                    if spec.column in frame.columns:
                        frame = spec.apply(frame)
                    elif spec.op == FilterOp.IN and spec.column not in frame.columns:
                        # ID 列不存在，但可能通过后续的 join 补全，暂时跳过
                        pass
            else:
                frame = frame.copy()

        defaults = table_cfg.defaults
        id_columns = (
            [defaults.id_var]
            if defaults.id_var and defaults.id_var in frame.columns
            else []
        )
        index_column = defaults.index_var if defaults.index_var in frame.columns else None
        time_columns = [
            column for column in defaults.time_vars if column in frame.columns
        ]
        value_column = defaults.val_var if defaults.val_var in frame.columns else None
        unit_column = defaults.unit_var if defaults.unit_var in frame.columns else None

        time_like_cols = set(time_columns)
        if index_column:
            time_like_cols.add(index_column)
        
        # AUMC特殊处理：时间列是毫秒,需要转换为分钟 (参考R ricu的ms_as_mins)
        # R ricu: ms_as_mins <- function(x) min_as_mins(as.integer(x / 6e4))
        # 关键: as.integer() 会 floor 到整数分钟!
        # 这样处理后,AUMC的时间单位与其他数据库一致(都是分钟)
        if self.config.name == 'aumc':
            for column in time_like_cols:
                if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
                    # 将毫秒转换为整数分钟: floor(ms / 60000) - 匹配 R ricu 的 as.integer()
                    # 🔧 PERFORMANCE FIX: Use numpy floor instead of slow apply+lambda
                    # 🔧 FIX: Handle pd.NA values properly by converting to numpy float array first
                    import numpy as np
                    values = np.array(frame[column], dtype=float) / 60000.0
                    frame[column] = np.where(np.isnan(values), np.nan, np.floor(values))
        
        for column in time_like_cols:
            # 只有当列存在且不是numeric类型时才转换
            # 如果已经是numeric，可能是已经对齐过的小时数
            if column in frame.columns:
                frame[column] = _coerce_datetime(frame[column])

        # 🔧 FIX 2026-01-26: 支持 MIMIC-III 的 icustay_id
        # MIMIC-III 的 id 列是 icustay_id，需要补全
        target_id_col = 'icustay_id' if self.config.name == 'mimic' else 'stay_id'
        has_target_id = target_id_col in frame.columns and not frame[target_id_col].isna().all()
        
        if not has_target_id and 'hadm_id' in frame.columns:
            # ⚠️ 问题：对于 hospital tables (如 labevents), 原表没有 stay_id/icustay_id，需要通过 hadm_id join icustays 补全
            # 但 join 会引入该 hadm_id 的所有 stay_id (同一住院可能多次ICU入住)
            # 解决方案：在函数开始时已保存 original_stay_ids，join 后再过滤
            # 🔧 FIX 2026-02-07: 添加 services 表（用于 adm 概念）
            hospital_tables = ['prescriptions', 'labevents', 'microbiologyevents', 'emar', 'pharmacy', 'services']
            is_mimic_db = self.config.name in ['miiv', 'mimic_demo', 'mimic']
            if table_name in hospital_tables and is_mimic_db:
                try:
                    # 🔍 提取当前的患者ID过滤器（stay_id/icustay_id 或 subject_id）
                    # 这样 icustays 只加载我们需要的患者，避免 join 时产生额外的匹配
                    icustays_filters = []
                    if filters:
                        for spec in filters:
                            # stay_id/icustay_id 或 subject_id 过滤器都可以用于过滤 icustays
                            if spec.column in ['stay_id', 'icustay_id', 'subject_id'] and spec.op == FilterOp.IN:
                                icustays_filters.append(spec)
                                if verbose:
                                    logger.debug(f"[{table_name}] 提取患者ID过滤器: {spec.column} IN ({len(spec.value)} 个值)")
                                # 不要 break，可能有多个过滤器
                    
                    # 加载 icustays 映射（需要 hadm_id, stay_id/icustay_id, subject_id）
                    # 如果有患者ID过滤器，传递给 icustays 以避免加载全表
                    if verbose:
                        logger.debug(f"[{table_name}] 加载 icustays，filters={len(icustays_filters)}个")
                    icustays_map = self.load_table(
                        'icustays', 
                        columns=['hadm_id', target_id_col, 'subject_id', 'intime', 'outtime'],  # 需要 intime 和 outtime 用于 rolling join
                        filters=icustays_filters if icustays_filters else None,
                        verbose=False
                    )
                    icustays_df = icustays_map.data if hasattr(icustays_map, 'data') else icustays_map
                    if verbose:
                        logger.debug(f"[{table_name}] icustays 加载完成: {len(icustays_df)} 行")
                    
                    # 🔥 CRITICAL FIX: 为了正确实现 rolling join，需要加载同一 hadm_id 下的所有 stays
                    # 当请求单个 stay 时，可能同一 hadm_id 有多个 ICU stays
                    # ricu 的 rolling join 需要知道所有 stays 的 intime 来正确分配数据
                    requested_hadm_ids = icustays_df['hadm_id'].unique().tolist()
                    if requested_hadm_ids and len(icustays_df) > 0:
                        # 加载这些 hadm_ids 对应的所有 stays（可能比请求的更多）
                        all_stays_for_hadms = self.load_table(
                            'icustays',
                            columns=['hadm_id', target_id_col, 'subject_id', 'intime', 'outtime'],
                            filters=[FilterSpec(column='hadm_id', op=FilterOp.IN, value=requested_hadm_ids)],
                            verbose=False
                        )
                        all_stays_df = all_stays_for_hadms.data if hasattr(all_stays_for_hadms, 'data') else all_stays_for_hadms
                        
                        # 检查是否有新增的 stays（同一 hadm_id 下的其他 stays）
                        if len(all_stays_df) > len(icustays_df):
                            if verbose:
                                logger.debug(f"[{table_name}] 发现同一 hadm_id 下有额外的 stays: {len(icustays_df)} → {len(all_stays_df)}")
                            # 使用完整的 stays 列表进行 join
                            icustays_df = all_stays_df
                    
                    # 保存原始行数用于日志
                    before_rows = len(frame)
                    
                    # JOIN 补全 stay_id/icustay_id（包含 intime 和 outtime 用于 rolling join）
                    # 注意：同一 hadm_id 可能对应多个 stay_id（多次 ICU 入住）
                    frame = frame.merge(
                        icustays_df[['hadm_id', target_id_col, 'intime', 'outtime']],
                        on='hadm_id',
                        how='inner',  # 只保留有 ICU 住院的记录
                        suffixes=('', '_icu')
                    )
                    
                    # 清理可能的重复列
                    icu_col_name = f'{target_id_col}_icu'
                    if icu_col_name in frame.columns:
                        # 如果原来有 id 列但是全 NaN，用新的替换
                        if target_id_col not in frame.columns or frame[target_id_col].isna().all():
                            frame[target_id_col] = frame[icu_col_name]
                        frame = frame.drop(columns=[icu_col_name], errors='ignore')
                    
                    after_join_rows = len(frame)
                    
                    # 🔥 CRITICAL FIX: 实现 ricu 的 rolling join 逻辑
                    # 
                    # ricu 使用 roll = -Inf, rollends = TRUE：
                    # - 关键发现：ricu 使用 **ICU outtime** 作为 rolling join 的 key！
                    # - roll = -Inf：向未来滚动，找 outtime >= charttime 的最近 stay
                    # - rollends = TRUE：边界外的数据也会被分配给最近的边界 stay
                    #
                    # 这意味着：
                    # - 如果 charttime < 第一个 stay 的 outtime，分配给第一个 stay
                    # - 如果 charttime >= 第一个 stay 的 outtime 但 < 第二个 stay 的 outtime，
                    #   分配给第二个 stay
                    # - 以此类推
                    #
                    # 当同一 hadm_id 有多个 stay_id 时，需要使用真正的 rolling join
                    time_col = None
                    for cand in ['charttime', 'storetime', 'starttime', 'specimen_time']:
                        if cand in frame.columns:
                            time_col = cand
                            break
                    
                    if time_col and target_id_col in frame.columns and 'outtime' in frame.columns:
                        # 检查是否有同一 hadm_id 下的多个 stay_id/icustay_id
                        stays_per_hadm = frame.groupby('hadm_id')[target_id_col].nunique()
                        multi_stay_hadms = stays_per_hadm[stays_per_hadm > 1].index.tolist()
                        
                        if multi_stay_hadms:
                            if verbose:
                                logger.debug(f"[{table_name}] 检测到 {len(multi_stay_hadms)} 个 hadm_id 有多个 {target_id_col}，执行 rolling join (使用 outtime)")
                            
                            # 规范化时间列 - 统一为 datetime64[ns] 以兼容 merge_asof
                            def _normalize_datetime_ns(series: pd.Series) -> pd.Series:
                                """规范化datetime为ns精度，去时区"""
                                dt = pd.to_datetime(series, errors='coerce', utc=True)
                                if dt.dt.tz is not None:
                                    dt = dt.dt.tz_localize(None)
                                # 🔧 FIX: 统一转换为 datetime64[ns] 确保 merge_asof 兼容
                                return dt.astype('datetime64[ns]')
                            
                            frame[time_col] = _normalize_datetime_ns(frame[time_col])
                            if 'intime' in frame.columns:
                                frame['intime'] = _normalize_datetime_ns(frame['intime'])
                            frame['outtime'] = _normalize_datetime_ns(frame['outtime'])
                            
                            # 分离需要 rolling join 的数据和不需要的数据
                            single_stay_mask = ~frame['hadm_id'].isin(multi_stay_hadms)
                            single_stay_data = frame[single_stay_mask].copy()
                            multi_stay_data = frame[~single_stay_mask].copy()
                            
                            # 🔥 使用 pd.merge_asof 实现真正的 rolling join
                            # 首先，获取唯一的数据记录（去除 join 导致的重复）
                            data_cols = [c for c in multi_stay_data.columns 
                                        if c not in [target_id_col, 'intime', 'outtime']]
                            unique_data = multi_stay_data[data_cols].drop_duplicates()
                            
                            # 获取每个 hadm_id 的 stay 信息，按 outtime 排序
                            stay_cols = ['hadm_id', target_id_col, 'outtime']
                            if 'intime' in multi_stay_data.columns:
                                stay_cols.append('intime')
                            stay_info = multi_stay_data[stay_cols].drop_duplicates()
                            stay_info = stay_info.sort_values(['hadm_id', 'outtime'])
                            
                            # 🚀 OPT: 批量 merge_asof 替代 per-hadm_id 循环
                            # 使用 by='hadm_id' 一次完成所有多stay的rolling join
                            
                            # 过滤 NaN keys (merge_asof 要求无空值)
                            stay_info = stay_info.dropna(subset=['outtime'])
                            unique_data = unique_data.dropna(subset=[time_col])
                            
                            if unique_data.empty or stay_info.empty:
                                frame = single_stay_data
                            else:
                                # 排序: merge_asof 要求左右都按 key 排序
                                unique_data = unique_data.sort_values(time_col)
                                
                                merge_cols = ['hadm_id', target_id_col, 'outtime']
                                if 'intime' in stay_info.columns:
                                    merge_cols.append('intime')
                                stay_lookup = stay_info[merge_cols].drop_duplicates().sort_values('outtime')
                                
                                merged = pd.merge_asof(
                                    unique_data,
                                    stay_lookup,
                                    left_on=time_col,
                                    right_on='outtime',
                                    by='hadm_id',
                                    direction='forward',
                                    allow_exact_matches=True,
                                )
                                
                                # rollends = TRUE: 未匹配行分配给每个 hadm_id 的最后一个 stay
                                unmatched = merged[target_id_col].isna()
                                if unmatched.any():
                                    last_stays = stay_lookup.groupby('hadm_id').last()
                                    merged.loc[unmatched, target_id_col] = (
                                        merged.loc[unmatched, 'hadm_id'].map(last_stays[target_id_col])
                                    )
                                    merged.loc[unmatched, 'outtime'] = (
                                        merged.loc[unmatched, 'hadm_id'].map(last_stays['outtime'])
                                    )
                                
                                merged[target_id_col] = merged[target_id_col].astype(int)
                                frame = pd.concat([single_stay_data, merged], ignore_index=True)
                            
                            if verbose:
                                logger.debug(f"[{table_name}] rolling join 完成: {after_join_rows} → {len(frame)} 行")
                    
                    # 清理临时的 intime 和 outtime 列
                    for col in ['intime', 'outtime']:
                        if col in frame.columns:
                            frame = frame.drop(columns=[col], errors='ignore')
                    
                    after_rows = len(frame)
                    
                    # ✅ 关键修复：join 后必须再次应用原始 stay_id/icustay_id 过滤
                    # 因为 join 可能产生了额外的 stay_ids (同一 subject 或 hadm_id 的多个 ICU stays)
                    # 
                    # 三种情况：
                    # 1. 如果原始过滤器是 stay_id/icustay_id，使用保存的 original_stay_ids
                    # 2. 如果原始过滤器是 subject_id，从 FilterSpec.metadata 中提取原始 stay_id
                    # 3. 从 icustays_filters 中查找
                    target_stay_ids = original_stay_ids
                    
                    if not target_stay_ids and icustays_filters:
                        for spec in icustays_filters:
                            if spec.column in ['stay_id', 'icustay_id'] and spec.op == FilterOp.IN:
                                target_stay_ids = set(spec.value)
                                if verbose:
                                    logger.debug(f"[{table_name}] 从 {spec.column} 过滤器获取: {len(target_stay_ids)} stays")
                                break
                            elif spec.column == 'subject_id' and spec.op == FilterOp.IN:
                                # 从 metadata 中提取原始 stay_ids
                                if spec.metadata and 'original_stay_ids' in spec.metadata:
                                    target_stay_ids = set(spec.metadata['original_stay_ids'])
                                    if verbose:
                                        logger.debug(f"[{table_name}] 从 subject_id 过滤器的 metadata 获取原始 {target_id_col}: {len(target_stay_ids)} stays")
                                    break
                    
                    if target_stay_ids:
                        before_filter = len(frame)
                        if target_id_col in frame.columns:
                            frame = frame[frame[target_id_col].isin(target_stay_ids)]
                            if verbose:
                                logger.debug(
                                    f"[{table_name}] 应用 {target_id_col} 过滤: {before_filter}行 → {len(frame)}行 "
                                    f"(保留 {len(target_stay_ids)} 个目标 {target_id_col})"
                                )
                        else:
                            if verbose:
                                logger.warning(f"[{table_name}] join 后仍无 {target_id_col} 列，无法应用过滤")
                    
                    # 记录补全操作
                    if verbose and before_rows != after_rows:
                        logger.info(
                            "表 %s: 通过 hadm_id 补全 %s (%d → %d 行)",
                            table_name,
                            target_id_col,
                            before_rows,
                            after_rows
                        )
                    
                    # ✅ 关键修复：补全 stay_id 后，更新 id_columns
                    # 这样下游 concept.py 会保留 stay_id 列而不是只保留 subject_id
                    if 'stay_id' in frame.columns:
                        id_columns = ['stay_id']
                        if verbose:
                            logger.debug(f"[{table_name}] 补全 {target_id_col} 后更新 id_columns: subject_id → {target_id_col}")
                        
                except Exception as e:
                    # 如果补全失败，记录警告但不中断流程
                    logger.warning(
                        "⚠️  表 %s: 无法补全 %s: %s",
                        table_name,
                        target_id_col,
                        str(e)
                    )

        if verbose and logger.isEnabledFor(logging.INFO):
            id_label = id_columns[0] if id_columns else defaults.id_var or "N/A"
            (
                frame[id_label].nunique()
                if id_label in frame.columns
                else "N/A"
            )
            # 减少日志输出，只在 DEBUG 模式下显示
            if DEBUG_MODE:
                logger.debug(
                    "表 %s: %d 行, %d 个 %s",
                    table_name,
                    len(frame),
                    frame[id_label].nunique() if id_label in frame.columns else 0,
                    id_label,
                )

        return ICUTable(
            data=frame,
            id_columns=id_columns,
            index_column=index_column,
            value_column=value_column,
            unit_column=unit_column,
            time_columns=time_columns,
        )

    def _load_raw_frame(
        self,
        table_name: str,
        columns: Optional[Iterable[str]],
        patient_ids_filter: Optional[FilterSpec] = None,
        concept_itemid_filter: Optional[Tuple[str, set]] = None,  # 🚀 概念特定 itemid 过滤器
        wide_table_value_columns: Optional[List[str]] = None,  # 🚀 宽表value列用于NULL过滤
    ) -> pd.DataFrame:
        # 🔍 调试日志：显示请求的列（仅在DEBUG级别显示）
        if columns:
            logger.debug(f"_load_raw_frame: table={table_name}, columns={list(columns)}")

        if concept_itemid_filter is not None and not concept_itemid_filter[1]:
            return pd.DataFrame(columns=list(columns) if columns else [])
        
        # 🚀 OPTIMIZATION: 缓存键不包含patient_ids_filter以实现跨概念共享
        # 对于同一批患者的多个概念加载,只在第一次读取表,后续从缓存中过滤
        # 这将chartevents等大表的加载从N次(每概念一次)减少到1次
        # 🔧 FIX: inputevents 现在也可以缓存，因为 key 中包含了 filter 信息
        # 之前排除 inputevents 是因为担心 subject_id→stay_id 映射问题
        # 但实际上 inputevents 表有 stay_id 列，可以直接过滤
        # 🔧 HiRID observations: 由于概念特定的 itemid 过滤，不同概念有不同数据，禁用缓存
        # 🔧 FIX: 分桶目录(numericitems_bucket等)也需要禁用缓存或包含itemid在key中
        skip_cache_tables = ['microbiologyevents', 'admissions', 'observations']  # 添加 observations
        enable_caching = self.enable_cache and table_name not in skip_cache_tables
        
        # 🔧 FIX: 如果表是经过过滤加载的，必须将filter包含在cache key中
        # 否则不同批次的加载会混淆
        filter_key = None
        if patient_ids_filter:
            val = patient_ids_filter.value
            if isinstance(val, (list, tuple)):
                val = tuple(val)
            elif isinstance(val, set):
                val = tuple(sorted(val))
            elif hasattr(val, 'tolist'):  # numpy array
                val = tuple(val.tolist())
            # 包含列名和操作符，确保唯一性
            filter_key = (patient_ids_filter.column, patient_ids_filter.op, val)

        # 🔧 FIX: 分桶读取时，concept_itemid_filter 也需要加入缓存key
        # 否则不同概念（不同itemid）会错误共享缓存
        itemid_filter_key = None
        if concept_itemid_filter:
            col, ids = concept_itemid_filter
            itemid_filter_key = (col, tuple(sorted(ids)))

        cache_key = (table_name, tuple(sorted(columns)) if columns else None, filter_key, itemid_filter_key)
        
        # 检查缓存
        cached_frame = None
        if enable_caching:
            with self._lock:
                cached_frame = self._table_cache.get(cache_key)
        
        if cached_frame is not None:
            # 🚀 OPTIMIZATION: 从缓存中取数据后再应用patient过滤
            # 这样多个概念可以共享同一个缓存的表副本
            # ⚡ 性能优化: 避免copy(),直接返回过滤后的视图
            logger.debug(f"从缓存加载: table={table_name}, cached_columns={list(cached_frame.columns)}")
            if patient_ids_filter:
                # 如果缓存的key已经包含了filter，那么cached_frame已经是过滤过的了
                # 但为了安全起见（或者如果filter_key逻辑有变），再次检查
                # 如果filter_key存在，说明cached_frame已经是针对该filter的子集
                # 此时再次应用filter应该是安全的（no-op）
                return patient_ids_filter.apply(cached_frame)
            # 如果不需要过滤，返回切片视图而非副本
            return cached_frame[:]
        
        loader = self._table_sources.get(table_name)
        dataset_cfg = self._dataset_sources.get(table_name)
        if loader is None and dataset_cfg is not None:
            frame = self._read_dataset(table_name, dataset_cfg, columns, patient_ids_filter)
        elif loader is None:
            # 🚀 优先检查分桶目录（无论是单文件还是多文件配置）
            # 分桶目录性能远优于普通目录或单个parquet
            bucket_loader = self._resolve_bucket_directory(table_name)
            if bucket_loader is not None:
                loader = bucket_loader
            else:
                # 修复：检查是否为多文件配置，如果是，使用目录路径
                table_cfg = self.config.get_table(table_name)
                if len(table_cfg.files) > 1:
                    # 多文件配置：使用目录路径以启用多文件读取
                    base_path = self.base_path or Path.cwd()
                    if table_cfg.files:
                        # HiRID特殊处理：配置中的CSV路径与实际parquet目录不同
                        # observation_tables/csv/ -> observations/
                        # pharma_records/csv/ -> pharma/
                        if self.config.name == 'hirid':
                            hirid_table_dir_mapping = {
                                'observations': 'observations',
                                'pharma': 'pharma',
                            }
                            if table_name in hirid_table_dir_mapping:
                                mapped_dir = base_path / hirid_table_dir_mapping[table_name]
                                if mapped_dir.is_dir():
                                    parquet_files = list(mapped_dir.glob("*.parquet")) + list(mapped_dir.glob("*.pq"))
                                    if parquet_files:
                                        loader = mapped_dir
                        
                        # 如果HiRID映射未找到，使用默认逻辑
                        if loader is None:
                            # 获取目录路径（从第一个文件路径中提取）
                            first_file = table_cfg.files[0]
                            # 处理字符串或字典格式
                            if isinstance(first_file, dict):
                                first_path = Path(first_file.get('path', first_file.get('name', '')))
                            else:
                                first_path = Path(first_file)
                            
                            multi_file_dir = base_path / first_path.parent
                            if multi_file_dir.is_dir():
                                loader = multi_file_dir
                            else:
                                # 回退到单个文件解析
                                loader = self._resolve_loader_from_disk(table_name)
                    else:
                        # 回退到单个文件解析
                        loader = self._resolve_loader_from_disk(table_name)
                else:
                    loader = self._resolve_loader_from_disk(table_name)
            
            # 🚀 MIMIC-III chartevents CSV fallback
            # When bucket directory is not available (due to memory constraints during conversion),
            # fall back to reading directly from CSV.gz with proper VALUE type handling
            if loader is None and self.config.name == 'mimic' and table_name == 'chartevents':
                csv_path = self._resolve_mimic3_chartevents_csv()
                if csv_path is not None and concept_itemid_filter is not None:
                    # Use CSV fallback only when we have itemid filter (for performance)
                    logger.info("MIMIC-III chartevents: bucket directory not found; using CSV fallback mode")
                    frame = self._read_mimic3_csv_fallback(
                        csv_path=csv_path,
                        columns=columns,
                        itemid_filter=concept_itemid_filter,
                        patient_ids_filter=patient_ids_filter,
                    )
                    
                    if not frame.empty:
                        # Cache and return
                        if enable_caching:
                            with self._lock:
                                self._table_cache[cache_key] = frame
                        return frame
                    # If CSV fallback returns empty, continue to normal error handling
            
            # 如果解析失败，返回空 DataFrame（兼容性处理，避免阻断整个流程）
            if loader is None:
                # 对于miiv数据源，如果表在配置中定义了但文件不存在，返回空DataFrame
                # 这允许在demo数据中缺少某些表时继续运行
                if self.config.name == 'miiv' and table_name in self.config.tables:
                    logger.warning(f"Table {table_name} not found on disk, returning empty DataFrame")
                    # 返回空DataFrame，保持与配置中表结构一致的列
                    table_cfg = self.config.get_table(table_name)
                    defaults = table_cfg.defaults
                    # 尝试从配置中获取预期的列
                    expected_cols = []
                    if defaults.id_var:
                        expected_cols.append(defaults.id_var)
                    if defaults.index_var:
                        expected_cols.append(defaults.index_var)
                    if defaults.val_var:
                        expected_cols.append(defaults.val_var)
                    if defaults.unit_var:
                        expected_cols.append(defaults.unit_var)
                    if defaults.time_vars:
                        expected_cols.extend(defaults.time_vars)
                    
                    # 返回空DataFrame，避免抛出错误
                    return pd.DataFrame(columns=expected_cols if expected_cols else ['index'])
                
                raise KeyError(
                    f"No table source registered for '{table_name}' "
                    f"in data source '{self.config.name}'"
                )
        if callable(loader):
            frame = loader()
        else:
            frame = self._read_file(
                Path(loader), columns, 
                patient_ids_filter=patient_ids_filter, 
                table_name=table_name,
                concept_itemid_filter=concept_itemid_filter,  # 🚀 传递概念特定过滤器
                wide_table_value_columns=wide_table_value_columns  # 🚀 传递宽表value列用于NULL过滤
            )

        # 🔧 FIX 2026-02-07: MIMIC-III 专用列名小写化
        # MIMIC-III 的 CSV/parquet 文件使用大写列名（SUBJECT_ID, CHARTTIME 等）
        # 但 ricu 字典配置期望小写列名（subject_id, charttime 等）
        # 注意：SICdb 使用混合大小写列名（CaseID, Val），不能被小写化！
        db_name = getattr(self.config, 'name', '') if hasattr(self, 'config') else ''
        should_lowercase = db_name in ('mimic', 'mimic_demo')
        
        if should_lowercase and hasattr(frame, 'columns'):
            frame.columns = [c.lower() if isinstance(c, str) else c for c in frame.columns]

        if columns is not None:
            if should_lowercase:
                # MIMIC-III: 列名匹配时也转为小写
                columns_lower = [c.lower() if isinstance(c, str) else c for c in columns]
                missing = set(columns_lower) - set(frame.columns)
                if missing:
                    raise KeyError(
                        f"Columns {sorted(missing)} not found in table '{table_name}'"
                    )
                frame = frame[list(columns_lower)]
            else:
                # 其他数据库: 保持原始列名
                missing = set(columns) - set(frame.columns)
                if missing:
                    raise KeyError(
                        f"Columns {sorted(missing)} not found in table '{table_name}'"
                    )
                frame = frame[list(columns)]
        
        # 🚀 OPTIMIZATION: 缓存完整表(未经patient过滤)以实现跨概念共享
        # patient过滤在从缓存读取时应用(见上面cached_frame分支)
        # ⚡ 性能优化: 缓存原始frame，返回过滤后的结果
        # 不缓存需要特殊处理的表（labevents/admissions等）
        if enable_caching:
            with self._lock:
                # 缓存原始未过滤的表
                self._table_cache[cache_key] = frame
        
        # 应用patient过滤(如果有)
        if patient_ids_filter:
            return patient_ids_filter.apply(frame)
        
        # 未过滤且未缓存时返回切片
        return frame[:] if self.enable_cache else frame

    def _resolve_bucket_directory(self, table_name: str) -> Optional[Path]:
        """
        🚀 优先检查分桶目录（性能最优）
        
        分桶目录使用 bucket_id=* 子目录结构，通过 hash(itemid) % num_buckets 实现
        读取时只需扫描相关桶，跳过 99% 无关数据
        
        检查位置：
        - base_path / {table_name}_bucket
        - base_path / icu / {table_name}_bucket  (MIIV)
        - base_path / hosp / {table_name}_bucket  (MIIV)
        
        Returns:
            分桶目录路径（如果存在且有效），否则 None
        """
        if not self.base_path:
            return None

        cache_key = str(table_name).lower()
        with self._lock:
            cached_dir = self._bucket_dir_cache.get(cache_key)
        if cached_dir is not None:
            return cached_dir
        
        # 可能的表名变体
        name_variants = [table_name, table_name.lower()]
        
        # 可能的分桶目录位置
        for name in name_variants:
            possible_bucket_dirs = [
                self.base_path / f"{name}_bucket",  # 直接在 base_path 下
                self.base_path / "icu" / f"{name}_bucket",  # MIIV icu 子目录
                self.base_path / "hosp" / f"{name}_bucket",  # MIIV hosp 子目录
            ]
            for bucket_dir in possible_bucket_dirs:
                if bucket_dir.is_dir():
                    # 检查是否有 bucket_id=* 子目录（分桶格式标识）
                    bucket_subdirs = list(bucket_dir.glob("bucket_id=*"))
                    if bucket_subdirs:
                        # 🔧 避免重复日志：只在首次发现时打印info
                        bucket_key = str(bucket_dir)
                        if bucket_key not in self._bucket_dir_logged:
                            self._bucket_dir_logged.add(bucket_key)
                            logger.info("Using bucket directory: %s (%d buckets)", bucket_dir, len(bucket_subdirs))
                        with self._lock:
                            self._bucket_dir_cache[cache_key] = bucket_dir
                        return bucket_dir
        
        return None

    def _resolve_flat_parquet_directory(self, table_name: str) -> Optional[Path]:
        """检查扁平 parquet 目录（无 bucket_id 子目录结构）"""
        if not self.base_path:
            return None
        cache_key = str(table_name).lower()
        with self._lock:
            cached_dir = self._flat_parquet_dir_cache.get(cache_key)
        if cached_dir is not None:
            return cached_dir

        name_variants = [table_name, table_name.lower()]
        for name in name_variants:
            for dir_path in [
                self.base_path / name,
                self.base_path / "icu" / name,
                self.base_path / "hosp" / name,
            ]:
                if dir_path.is_dir() and list(dir_path.glob("*.parquet"))[:1]:
                    with self._lock:
                        self._flat_parquet_dir_cache[cache_key] = dir_path
                    return dir_path
        return None

    def _resolve_loader_from_disk(self, table_name: str) -> Optional[Callable[[], pd.DataFrame] | Path]:
        if not self.base_path:
            return None

        def _case_insensitive_existing_path(path: Path) -> Optional[Path]:
            """Resolve a file path even when the exported ICU table uses camelCase.

            Some public eICU exports keep table names such as ``vitalPeriodic``
            while EasyICU's concept dictionary normalizes table keys to
            lowercase (``vitalperiodic``). On case-sensitive volumes this would
            silently make valid real-data tables unavailable.
            """
            if path.exists():
                return path
            parent = path.parent
            if not parent.is_dir():
                return None
            target = path.name.lower()
            for child in parent.iterdir():
                if child.name.lower() == target:
                    return child
            return None

        if self.config.name == 'mimic' and table_name == 'services':
            # MIMIC-III services is a small hospital-level table used by the
            # demographics ``adm`` concept. Prefer the original CSV when it is
            # available because some user-side parquet conversions of this
            # file can abort inside Arrow before Python can catch the error.
            for candidate in [
                self.base_path / "SERVICES.csv.gz",
                self.base_path / "services.csv.gz",
                self.base_path / "SERVICES.csv",
                self.base_path / "services.csv",
            ]:
                resolved_csv = _case_insensitive_existing_path(candidate)
                if resolved_csv is not None:
                    return resolved_csv
        
        # 🚀 优先级最高：检查分桶目录（性能最优）
        # 分桶目录命名规则：{table_name}_bucket
        # 必须在检查配置文件之前，因为分桶目录是性能优化的关键
        bucket_dir = self._resolve_bucket_directory(table_name)
        if bucket_dir is not None:
            return bucket_dir
        
        table_cfg = self.config.get_table(table_name)
        explicit = table_cfg.first_file()
        
        if explicit:
            explicit_path = self.base_path / explicit
            if explicit_path.exists():
                # Accept directories (partitioned datasets) and Parquet files immediately
                if explicit_path.is_dir():
                    # 🔧 2026-05-11: 必须验证目录里真的有 parquet 文件再返回。
                    # 否则 eICU 等数据库会因为同名目录残留旧的 .fst 文件而被误认为
                    # parquet 分区数据集，导致 read_parquet('{dir}/*.parquet') 找不到文件
                    # 然后回退到 fst 路径，丢失大量数据（实测 vitalperiodic 5M → 357k 行）。
                    if list(explicit_path.glob("*.parquet"))[:1] or list(explicit_path.glob("*.pq"))[:1]:
                        return explicit_path
                    # 否则继续搜索同级 / 子目录的单文件 parquet（下方逻辑）
                elif explicit_path.suffix.lower() in {".parquet", ".pq"}:
                    return explicit_path
                # Otherwise continue searching for a Parquet counterpart below
        
        # MIMIC-IV 文件名映射: 配置中的表名 -> 文件系统中的实际文件名
        # 因为 MIMIC-IV 改了表名,但配置文件还是用的旧名
        # MIMIC-IV 将 MIMIC-III 的两个表合并了:
        # - procedureevents_mv -> procedureevents
        # - inputevents_cv 和 inputevents_mv -> inputevents
        # 注意：对于miiv数据源，概念字典中直接使用procedureevents和inputevents
        # 但如果数据源配置中没有定义这些表，需要从旧表名映射

        config_to_file_mappings = {
            'procedureevents_mv': 'procedureevents',  # 配置名 -> 文件名
            'inputevents_mv': 'inputevents',
            'inputevents_cv': 'inputevents',  # MIMIC-IV 合并了这两个表
        }
        
        # 对于miiv数据源，如果请求的表在配置中不存在，尝试从映射中查找
        # 例如：概念字典请求procedureevents，但配置中可能只有procedureevents_mv的定义
        if self.config.name == 'miiv':
            # 反向映射：如果请求的表名不在配置中，尝试查找对应的旧表名
            reverse_mapping = {
                'procedureevents': 'procedureevents_mv',
                'inputevents': 'inputevents_mv',  # 优先使用inputevents_mv
            }
            if table_name not in self.config.tables and table_name in reverse_mapping:
                # 使用映射后的表名查找文件
                mapped_table = reverse_mapping[table_name]
                file_base_name = config_to_file_mappings.get(mapped_table, table_name)
            else:
                file_base_name = config_to_file_mappings.get(table_name, table_name)
        else:
            # 获取实际要查找的文件名
            file_base_name = config_to_file_mappings.get(table_name, table_name)

            # 从表配置中获取实际的文件名（如果存在）
            if table_name in self.config.tables:
                table_config = self.config.tables[table_name]
                if hasattr(table_config, 'files') and table_config.files:
                    # 获取第一个文件的路径，去掉扩展名
                    file_info = table_config.files[0]
                    if isinstance(file_info, dict) and 'path' in file_info:
                        file_path = file_info['path']
                    elif hasattr(file_info, 'path'):
                        file_path = file_info.path
                    else:
                        file_path = str(file_info)
                    # 去掉扩展名，获取基础文件名（处理复合扩展名如.csv.gz）
                    parts = file_path.split('.')
                    if len(parts) >= 2:
                        # 处理复合扩展名如 .csv.gz
                        if parts[-1] == 'gz' and len(parts) >= 3 and parts[-2] == 'csv':
                            file_base_name = '.'.join(parts[:-2])
                        else:
                            file_base_name = '.'.join(parts[:-1])
                    else:
                        file_base_name = file_path
        
        # Only support Parquet format - try different name variations
        for name in [file_base_name, file_base_name.lower(), table_name, table_name.lower()]:
            # 🚀 优先检查 bucket 目录（分桶格式，性能最优）
            # 必须在检查 .parquet 文件之前，因为分桶目录可能与表同名
            # 检查多个可能的位置：base_path 和子目录（如 icu/, hosp/）
            bucket_dir = self._resolve_bucket_directory(name)
            if bucket_dir is not None:
                return bucket_dir
            
            # Try .parquet extension - 检查多个可能的位置
            # MIMIC-IV 的表可能在 icu/ 或 hosp/ 子目录下
            possible_parquet_paths = [
                self.base_path / f"{name}.parquet",  # 直接在 base_path 下
                self.base_path / "icu" / f"{name}.parquet",  # MIIV icu 子目录
                self.base_path / "hosp" / f"{name}.parquet",  # MIIV hosp 子目录
            ]
            for parquet_candidate in possible_parquet_paths:
                resolved = _case_insensitive_existing_path(parquet_candidate)
                if resolved is not None:
                    return resolved
            
            # Try .pq extension (short form) - 同样检查子目录
            possible_pq_paths = [
                self.base_path / f"{name}.pq",
                self.base_path / "icu" / f"{name}.pq",
                self.base_path / "hosp" / f"{name}.pq",
            ]
            for pq_candidate in possible_pq_paths:
                resolved = _case_insensitive_existing_path(pq_candidate)
                if resolved is not None:
                    return resolved
        
        # Check subdirectory for partitioned parquet data (common in hirid observations)
        if self.base_path is not None:
            for name in [table_name, table_name.lower()]:
                # 检查普通分区目录
                subdir = self.base_path / name
                if subdir.is_dir():
                    # Look for Parquet files
                    parquet_files = list(subdir.glob("*.parquet")) + list(subdir.glob("*.pq"))
                    if parquet_files:
                        return subdir
        
        # Fall back to explicit file if it exists (e.g., CSV) so that callers can handle it
        if explicit:
            explicit_path = self.base_path / explicit
            if explicit_path.exists():
                return explicit_path

        return None

    def _get_minimal_columns(self, table_name: str) -> Optional[List[str]]:
        """获取表的最小必要列集（性能优化）"""
        return MINIMAL_COLUMNS.get(table_name)
    
    def _read_file(
        self, path: Path, columns: Optional[Iterable[str]], 
        patient_ids_filter: Optional[FilterSpec] = None, 
        table_name: Optional[str] = None,
        concept_itemid_filter: Optional[Tuple[str, set]] = None,  # 🚀 概念特定 itemid 过滤器
        wide_table_value_columns: Optional[List[str]] = None  # 🚀 宽表value列用于NULL过滤
    ) -> pd.DataFrame:
        # 🚀 大表 itemid 预过滤配置
        # 检测是否为需要 itemid 过滤的大表
        # 🔧 2024-12-02: 重新启用白名单过滤，白名单已包含所有 sofa2-dict.json 中定义的 itemid
        itemid_filter_config = None
        db_name = self.config.name
        
        # 🚀 优先使用概念特定的 itemid 过滤器（精确过滤，性能最佳）
        # 如果传入了 concept_itemid_filter，直接使用，跳过全局白名单
        if concept_itemid_filter:
            itemid_filter_config = concept_itemid_filter
            if DEBUG_MODE:
                col, ids = concept_itemid_filter
                logger.info(f"🎯 使用概念特定过滤: {col} IN {len(ids)} 个 ID (精确模式)")
        elif db_name == 'aumc' and table_name == 'numericitems':
            # AUMC numericitems: 80GB → 约5GB
            itemid_filter_config = ('itemid', AUMC_NUMERICITEMS_ITEMIDS)
        elif db_name == 'miiv' and table_name == 'chartevents':
            # MIIV chartevents: 11GB
            itemid_filter_config = ('itemid', MIIV_CHARTEVENTS_ITEMIDS)
        elif db_name == 'mimic_demo' and table_name == 'chartevents':
            itemid_filter_config = ('itemid', MIMIC_DEMO_CHARTEVENTS_ITEMIDS)
        elif db_name == 'miiv' and table_name == 'labevents':
            # MIIV labevents: 8GB
            itemid_filter_config = ('itemid', MIIV_LABEVENTS_ITEMIDS)
        elif db_name == 'mimic_demo' and table_name == 'labevents':
            itemid_filter_config = ('itemid', MIMIC_DEMO_LABEVENTS_ITEMIDS)
        elif db_name == 'hirid' and table_name == 'observations':
            # 🚀 HiRID observations: 7.77亿行 (~72GB) → 大幅减少
            # 使用 variableid 过滤只加载概念字典中定义的变量
            itemid_filter_config = ('variableid', HIRID_OBSERVATIONS_VARIABLEIDS)
        
        # Handle directory (partitioned data)
        if path.is_dir():
            if DEBUG_MODE:
                logger.debug(f"读取分区目录: {path.name}, 请求列: {list(columns) if columns else '全部列'}")
            # 🚀 优先使用 DuckDB（单患者/小批量查询快 5-6 倍）
            # 对于大批量患者（>100），PyArrow 的并行读取更优
            use_duckdb = True
            if patient_ids_filter and patient_ids_filter.value:
                values = patient_ids_filter.value
                if isinstance(values, (list, tuple, set, pd.Series)):
                    n_patients = len(values)
                    if n_patients > 100:
                        # 🔧 FIX 2026-05-11: 对于 itemid 分桶的目录（如 HiRID observations_bucket），
                        # DuckDB 路径支持按 itemid→bucket 精准选择文件，只读 1-2 个桶；
                        # 而 PyArrow 的 `_read_partitioned_data_optimized` 会扫描所有桶。
                        # 对 HiRID 200-patient 呼吸模块，这导致 ~500s → ~10s 的差距。
                        # 因此仅对「非桶化且带 itemid 过滤不强」的情况才回退 PyArrow。
                        has_bucket_layout = False
                        if itemid_filter_config:
                            try:
                                _nb, _files_by_bucket = self._get_bucket_layout(path)
                                has_bucket_layout = bool(_files_by_bucket)
                            except Exception:
                                has_bucket_layout = False
                        use_duckdb = has_bucket_layout
            
            if use_duckdb:
                return self._read_partitioned_data_duckdb(
                    path, columns, patient_ids_filter, 
                    itemid_filter_config=itemid_filter_config, 
                    table_name=table_name,
                    wide_table_value_columns=wide_table_value_columns  # 🚀 传递宽表value列用于NULL过滤
                )
            else:
                return self._read_partitioned_data_optimized(path, columns, patient_ids_filter, itemid_filter_config=itemid_filter_config)
        
        suffix = path.suffix.lower()
        lower_name = path.name.lower()

        if suffix == ".csv" or lower_name.endswith(".csv.gz"):
            read_columns = list(columns) if columns else None
            filter_column = patient_ids_filter.column if patient_ids_filter else None

            try:
                header = pd.read_csv(path, nrows=0).columns.tolist()
                name_by_lower = {
                    str(name).lower(): name
                    for name in header
                    if isinstance(name, str)
                }
                if read_columns is not None:
                    mapped_columns = []
                    for col in read_columns:
                        mapped = name_by_lower.get(str(col).lower(), col)
                        if mapped not in mapped_columns:
                            mapped_columns.append(mapped)
                    read_columns = mapped_columns
                if filter_column is not None:
                    filter_column = name_by_lower.get(str(filter_column).lower(), filter_column)
                    if read_columns is not None and filter_column not in read_columns:
                        read_columns.append(filter_column)
            except Exception:
                pass

            df = pd.read_csv(path, usecols=read_columns, low_memory=False)
            if patient_ids_filter and patient_ids_filter.op == FilterOp.IN and filter_column in df.columns:
                if filter_column != patient_ids_filter.column:
                    df = df.rename(columns={filter_column: patient_ids_filter.column})
                df = patient_ids_filter.apply(df)
            return df
        
        # Preferred: Parquet format
        if suffix in {".parquet", ".pq"}:
            def _resolve_parquet_columns_and_filter():
                read_columns = list(columns) if columns else None
                filter_column = patient_ids_filter.column if patient_ids_filter else None
                if self.config.name not in ("mimic", "mimic_demo"):
                    return read_columns, filter_column
                try:
                    import pyarrow.parquet as pq
                    schema_names = list(pq.ParquetFile(path).schema_arrow.names)
                except Exception:
                    return read_columns, filter_column
                name_by_lower = {
                    str(name).lower(): name
                    for name in schema_names
                    if isinstance(name, str)
                }
                if read_columns is not None:
                    read_columns = [
                        name_by_lower.get(str(col).lower(), col)
                        for col in read_columns
                    ]
                if filter_column is not None:
                    filter_column = name_by_lower.get(str(filter_column).lower(), filter_column)
                return read_columns, filter_column

            read_columns, filter_column = _resolve_parquet_columns_and_filter()
            # 🚀 使用PyArrow过滤器优化大文件读取
            if patient_ids_filter and patient_ids_filter.op == FilterOp.IN:
                try:
                    import pyarrow.parquet as pq
                    import pyarrow as pa
                    # ⚡ 使用预计算的set
                    target_ids = list(patient_ids_filter._value_set)
                    
                    # 使用PyArrow读取并过滤 - 使用 DNF 格式
                    tbl = pq.read_table(
                        path,
                        columns=read_columns,
                        filters=[[(filter_column, 'in', target_ids)]]
                    )
                    df = _arrow_to_pandas_compat(tbl)
                    del tbl
                except (ImportError, Exception):
                    # 如果PyArrow过滤失败，回退到pandas后过滤
                    import pyarrow.parquet as _pq2
                    _tbl2 = _pq2.read_table(path, columns=read_columns)
                    df = _arrow_to_pandas_compat(_tbl2)
                    del _tbl2
                    if filter_column in df.columns:
                        if filter_column != patient_ids_filter.column:
                            df = df.rename(columns={filter_column: patient_ids_filter.column})
                        df = patient_ids_filter.apply(df)
            else:
                import pyarrow.parquet as _pq3
                _tbl3 = _pq3.read_table(path, columns=read_columns)
                df = _arrow_to_pandas_compat(_tbl3)
                del _tbl3
            
            # 处理重复列名（如果存在）
            if df.columns.duplicated().any():
                import pandas.io.common
                df.columns = pandas.io.common.dedup_names(df.columns, is_potential_multiindex=False)
            return df
        
        raise ValueError(
            f"Unsupported file format '{path.suffix}' for {path.name}. Only Parquet format is supported."
        )
    
    def _compute_target_buckets(self, itemids: set, num_buckets: int, duckdb_module) -> set:
        """使用 DuckDB hash 计算 itemid 对应的目标桶 ID
        
        必须使用 DuckDB 的 hash() 函数，因为分桶时使用的是 DuckDB hash。
        Python 的 hash() 函数与 DuckDB 不一致！
        
        Args:
            itemids: itemid 集合
            num_buckets: 总桶数
            duckdb_module: 已导入的 duckdb 模块
            
        Returns:
            目标桶 ID 集合
        """
        bucket_count = int(num_buckets)
        itemid_list = sorted(itemids, key=lambda item: (type(item).__name__, str(item)))
        normalized_items = tuple(self._bucket_item_cache_key(item) for item in itemid_list)
        cache_key = (bucket_count, normalized_items)
        with self._lock:
            cached = self._bucket_hash_cache.get(cache_key)
        if cached is not None:
            return set(cached)

        cached_buckets = set()
        missing_items = []
        with self._lock:
            for item in itemid_list:
                item_key = (bucket_count, self._bucket_item_cache_key(item))
                cached_bucket = self._bucket_hash_item_cache.get(item_key)
                if cached_bucket is None:
                    missing_items.append(item)
                else:
                    cached_buckets.add(cached_bucket)

        computed_buckets = set()
        conn = _get_duckdb_connection()
        try:
            if missing_items:
                conn.execute(
                    """
                    CREATE OR REPLACE TEMP TABLE items AS
                    SELECT row_number() OVER () - 1 AS _easyicu_ord, itemid
                    FROM (SELECT UNNEST(?) AS itemid)
                    """,
                    [missing_items],
                )
                result = conn.execute(
                    f"SELECT _easyicu_ord, hash(itemid) % {bucket_count} FROM items"
                ).fetchall()
                bucket_by_ord = {int(row[0]): row[1] for row in result}
                with self._lock:
                    for idx, item in enumerate(missing_items):
                        bucket = bucket_by_ord[idx]
                        computed_buckets.add(bucket)
                        self._bucket_hash_item_cache[
                            (bucket_count, self._bucket_item_cache_key(item))
                        ] = bucket

            buckets = frozenset(cached_buckets | computed_buckets)
            with self._lock:
                self._bucket_hash_cache[cache_key] = buckets
            return set(buckets)
        except Exception:
            raise

    @staticmethod
    def _bucket_item_cache_key(item: object) -> Tuple[str, str, str, str]:
        item_type = type(item)
        return (item_type.__module__, item_type.__qualname__, str(item), repr(item))

    def _get_bucket_layout(self, bucket_dir: Path) -> Tuple[int, Dict[int, Tuple[Path, ...]]]:
        """Return bucket count and parquet files per bucket.

        Bucketed datasets are immutable after conversion in normal EasyICU use.
        When the converter writes a ``_COMPLETE`` marker we use it as a cheap
        cache signature; directories without the marker are read live to avoid
        stale partial-conversion metadata.

        ⚡ 性能：在 macFUSE/NTFS 等慢盘上，``glob('bucket_id=*')`` + 每个子目录
        ``glob('*.parquet')`` 一次就要 10+s。我们在磁盘上缓存一份
        ``_BUCKET_MANIFEST.json``：第一次扫描后写入，后续进程（包括子进程）
        直接读 manifest，跳过 82 次 listdir。manifest 与 ``_COMPLETE`` mtime
        绑定校验，转换器重写时自动失效。
        """
        import json as _json

        bucket_dir = Path(bucket_dir)
        complete_marker = bucket_dir / "_COMPLETE"
        manifest_path = bucket_dir / "_BUCKET_MANIFEST.json"
        cache_key = str(bucket_dir.resolve())
        cache_signature = None
        if complete_marker.exists():
            stat = complete_marker.stat()
            cache_signature = (stat.st_mtime_ns, stat.st_size)
            with self._lock:
                cached = self._bucket_layout_cache.get(cache_key)
            if cached is not None and cached[0] == cache_signature:
                return cached[1], dict(cached[2])

            # 进程级 cache miss → 尝试磁盘 manifest（跨进程加速）
            if manifest_path.exists():
                try:
                    with manifest_path.open('r', encoding='utf-8') as fh:
                        m = _json.load(fh)
                    if (
                        isinstance(m, dict)
                        and m.get('complete_mtime_ns') == stat.st_mtime_ns
                        and m.get('complete_size') == stat.st_size
                        and isinstance(m.get('files_by_bucket'), dict)
                    ):
                        files_by_bucket = {
                            int(bid): tuple(bucket_dir / rel for rel in rels)
                            for bid, rels in m['files_by_bucket'].items()
                        }
                        num_buckets = int(m.get('num_buckets', 0))
                        with self._lock:
                            self._bucket_layout_cache[cache_key] = (
                                cache_signature, num_buckets, files_by_bucket,
                            )
                        return num_buckets, files_by_bucket
                except Exception:
                    pass  # manifest corrupt → fall through to live scan

        bucket_ids = set()
        files_by_bucket: Dict[int, Tuple[Path, ...]] = {}
        for bucket_subdir in bucket_dir.glob("bucket_id=*"):
            if not bucket_subdir.is_dir():
                continue
            try:
                bucket_id = int(bucket_subdir.name.split("=", 1)[1])
            except (IndexError, ValueError):
                continue
            bucket_ids.add(bucket_id)
            # 过滤 macOS AppleDouble 元数据（NTFS/exFAT 等非 Apple 文件系统上自动产生 ._*.parquet）
            # DuckDB read_parquet 会把它们当真 parquet 读，触发 'No magic bytes found' 错误
            files = tuple(
                sorted(
                    p for p in bucket_subdir.glob("*.parquet")
                    if not p.name.startswith(".")
                )
            )
            if files:
                files_by_bucket[bucket_id] = files

        num_buckets = max(bucket_ids) + 1 if bucket_ids else 0

        if cache_signature is not None:
            with self._lock:
                self._bucket_layout_cache[cache_key] = (
                    cache_signature,
                    num_buckets,
                    files_by_bucket,
                )
            # 写入磁盘 manifest 让后续进程跳过 listdir（best-effort，写失败忽略）
            try:
                rels = {
                    str(bid): [str(p.relative_to(bucket_dir)) for p in files]
                    for bid, files in files_by_bucket.items()
                }
                tmp = manifest_path.with_suffix('.json.tmp')
                with tmp.open('w', encoding='utf-8') as fh:
                    _json.dump({
                        'complete_mtime_ns': cache_signature[0],
                        'complete_size': cache_signature[1],
                        'num_buckets': num_buckets,
                        'files_by_bucket': rels,
                    }, fh)
                tmp.replace(manifest_path)
            except Exception:
                pass

        return num_buckets, files_by_bucket

    def _get_bucket_files_for_ids(
        self,
        bucket_dir: Path,
        itemids: Iterable[object],
        duckdb_module,
    ) -> Tuple[set, int, Tuple[Path, ...]]:
        """Resolve the parquet files that can contain the requested bucket keys."""
        num_buckets, files_by_bucket = self._get_bucket_layout(bucket_dir)
        if num_buckets <= 0:
            return set(), 0, tuple()

        target_buckets = self._compute_target_buckets(set(itemids), num_buckets, duckdb_module)
        target_files = tuple(
            file_path
            for bucket_id in sorted(target_buckets)
            for file_path in files_by_bucket.get(bucket_id, tuple())
        )
        return target_buckets, num_buckets, target_files

    def _get_parquet_columns_for_files(self, files: Iterable[Path]) -> set:
        """Return the union of Parquet schema columns for an explicit file list."""
        file_list = tuple(sorted((Path(file_path) for file_path in files), key=lambda path: str(path)))
        if not file_list:
            return set()

        signature = []
        for file_path in file_list:
            try:
                stat = file_path.stat()
            except OSError:
                return set()
            signature.append((str(file_path.resolve()), stat.st_mtime_ns, stat.st_size))
        cache_key = tuple(signature)

        with self._lock:
            cached = self._parquet_columns_cache.get(cache_key)
        if cached is not None:
            return set(cached)

        try:
            import pyarrow.parquet as pq
        except Exception:
            return set()

        columns = set()
        missing_files = []
        with self._lock:
            for file_path, file_signature in zip(file_list, signature):
                cached_file_columns = self._parquet_file_columns_cache.get(file_signature)
                if cached_file_columns is None:
                    missing_files.append((file_path, file_signature))
                else:
                    columns.update(cached_file_columns)

        for file_path, file_signature in missing_files:
            try:
                file_columns = frozenset(str(name) for name in pq.ParquetFile(file_path).schema_arrow.names)
            except Exception:
                return set()
            columns.update(file_columns)
            with self._lock:
                self._parquet_file_columns_cache[file_signature] = file_columns

        frozen = frozenset(columns)
        with self._lock:
            self._parquet_columns_cache[cache_key] = frozen
        return set(frozen)
    
    def _read_mimic3_csv_fallback(
        self,
        csv_path: Path,
        columns: Optional[Iterable[str]],
        itemid_filter: Optional[Tuple[str, set]],
        patient_ids_filter: Optional[FilterSpec] = None,
    ) -> pd.DataFrame:
        """🚀 MIMIC-III chartevents CSV fallback: read directly from CSV.gz with correct VALUE type
        
        When chartevents_bucket directory doesn't exist (due to memory constraints during conversion),
        this method reads directly from the original CSV.gz file using DuckDB with proper type hints.
        
        The key issue is that DuckDB's read_csv_auto incorrectly detects the VALUE column as DOUBLE
        when early rows have numeric values, causing text values like "4 Spontaneously" (GCS scores)
        to become NaN. We use types={'VALUE': 'VARCHAR'} to preserve these text values.
        
        Args:
            csv_path: Path to CHARTEVENTS.csv.gz
            columns: Columns to select (will be uppercased for MIMIC-III)
            itemid_filter: (column_name, set_of_itemids) for filtering
            patient_ids_filter: Optional patient ID filter
            
        Returns:
            DataFrame with the requested data
        """
        try:
            import duckdb
        except ImportError:
            logger.warning("DuckDB not installed, cannot use MIMIC-III CSV fallback")
            return pd.DataFrame()
        
        logger.info("MIMIC-III CSV fallback mode: reading from %s (keeping VALUE as string)", csv_path.name)
        
        # Build column selection - MIMIC-III uses UPPERCASE column names in CSV
        if columns:
            # Map common column names to MIMIC-III uppercase
            col_mapping = {
                'icustay_id': 'ICUSTAY_ID',
                'subject_id': 'SUBJECT_ID', 
                'hadm_id': 'HADM_ID',
                'charttime': 'CHARTTIME',
                'itemid': 'ITEMID',
                'value': 'VALUE',
                'valuenum': 'VALUENUM',
                'valueuom': 'VALUEUOM',
            }
            upper_cols = []
            for c in columns:
                upper_cols.append(col_mapping.get(c.lower(), c.upper()))
            columns_sql = ", ".join(upper_cols)
        else:
            columns_sql = "*"
        
        # Build WHERE conditions
        where_conditions = []
        
        # ITEMID filter (critical for performance - filters 330M rows to small subset)
        if itemid_filter:
            filter_col, filter_ids = itemid_filter
            # MIMIC-III uses uppercase ITEMID
            filter_col_upper = filter_col.upper()
            itemids_list = ", ".join(str(int(x)) for x in filter_ids)
            where_conditions.append(f"{filter_col_upper} IN ({itemids_list})")
            logger.debug(f"🎯 CSV 过滤: {filter_col_upper} IN ({len(filter_ids)} 个 ID)")
        
        # Patient ID filter
        if patient_ids_filter and patient_ids_filter.value:
            id_col = patient_ids_filter.column.upper()  # MIMIC-III uses uppercase
            values = patient_ids_filter.value
            if isinstance(values, (list, tuple, set)):
                value_list = list(values)
            elif isinstance(values, pd.Series):
                value_list = values.tolist()
            else:
                value_list = [values]
            
            if value_list:
                if len(value_list) == 1:
                    where_conditions.append(f"{id_col} = {value_list[0]}")
                else:
                    values_str = ", ".join(map(str, value_list))
                    where_conditions.append(f"{id_col} IN ({values_str})")
        
        # Build WHERE clause
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # 🔑 CRITICAL: Use types={'VALUE': 'VARCHAR'} to preserve GCS text values
        # Without this, values like "4 Spontaneously" become NaN because DuckDB
        # incorrectly detects VALUE as DOUBLE from early numeric rows
        # NOTE: Do NOT use sample_size=-1 as it causes full file scan for type detection
        chartevents_type_hint = ",\n                types={'VALUE': 'VARCHAR'}" if csv_path.name.upper().startswith("CHARTEVENTS") else ""
        query = f"""
            SELECT {columns_sql}
            FROM read_csv_auto(
                '{_duckdb_path(csv_path)}',
                ignore_errors=true,
                null_padding=true{chartevents_type_hint}
            )
            {where_clause}
        """
        
        try:
            con = _get_duckdb_connection()
            try:
                arrow_table = con.execute(query).fetch_arrow_table()
                df = _arrow_to_pandas_compat(arrow_table, split_blocks=True)
                del arrow_table
            except Exception:
                df = con.execute(query).fetchdf()
            
            # Normalize column names to lowercase (MIMIC-III CSV uses uppercase)
            df.columns = [c.lower() for c in df.columns]
            
            logger.info("CSV fallback succeeded: loaded %d rows", len(df))
            return df
            
        except Exception as e:
            logger.error("MIMIC-III CSV fallback failed: %s", e)
            return pd.DataFrame()
    
    def _resolve_mimic3_csv(self, table_name: str) -> Optional[Path]:
        """Find an original MIMIC-III CSV/CSV.GZ file for a table.
        
        Returns:
            Path to CSV file if found, None otherwise
        """
        if not self.base_path:
            return None
        
        table_upper = table_name.upper()
        table_lower = table_name.lower()
        possible_names = [
            f'{table_upper}.csv.gz',
            f'{table_lower}.csv.gz',
            f'{table_upper}.csv',
            f'{table_lower}.csv',
        ]
        
        for name in possible_names:
            csv_path = self.base_path / name
            if csv_path.exists():
                return csv_path
        
        return None

    def _resolve_mimic3_chartevents_csv(self) -> Optional[Path]:
        """Find MIMIC-III CHARTEVENTS.csv.gz file."""
        return self._resolve_mimic3_csv('chartevents')

    def _read_partitioned_data_duckdb(self, directory: Path, columns: Optional[Iterable[str]], patient_ids_filter: Optional[FilterSpec] = None, itemid_filter_config: Optional[tuple] = None, table_name: Optional[str] = None, wide_table_value_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """使用 DuckDB 读取分区数据（高性能版本）
        
        DuckDB 对单患者/小批量患者查询特别高效，比 PyArrow 快 5-6 倍。
        支持两种目录结构：
        - 普通分区: directory/*.parquet
        - 分桶格式: directory/bucket_id=*/*.parquet (AUMC numericitems_bucket)
        
        Args:
            itemid_filter_config: 可选的 (列名, itemid集合) 元组，用于大表预过滤
            table_name: 表名，用于确定预排序键
            wide_table_value_columns: 🚀 宽表value列列表，用于NULL过滤优化
                                      对于vitalperiodic等宽表，传入如['heartrate']
                                      会生成WHERE heartrate IS NOT NULL条件
        """
        try:
            import duckdb
        except ImportError:
            # DuckDB 未安装，回退到 PyArrow
            return self._read_partitioned_data_optimized(directory, columns, patient_ids_filter, itemid_filter_config=itemid_filter_config)
        
        # 🚀 检测目录结构：分桶格式 vs 普通分区
        num_buckets, files_by_bucket = self._get_bucket_layout(directory)
        if files_by_bucket:
            # 🚀 关键优化：如果有 itemid 过滤条件，计算目标桶，只扫描这些桶
            if itemid_filter_config:
                filter_col, filter_ids = itemid_filter_config
                # 只有数值型 itemid 才能使用 hash 分桶
                numeric_ids = {int(x) for x in filter_ids if isinstance(x, (int, float)) and not isinstance(x, bool)}
                if numeric_ids:
                    # 使用 DuckDB hash 计算目标桶（与分桶转换时一致）
                    target_buckets, _, target_files = self._get_bucket_files_for_ids(directory, numeric_ids, duckdb)
                    if target_files:
                        # 使用精确的文件列表而非全扫描
                        file_list_str = ", ".join(f"'{_duckdb_path(f)}'" for f in target_files)
                        glob_pattern = f"[{file_list_str}]"
                        logger.debug(
                            "Bucket targeted read: %d/%d buckets, %d files",
                            len(target_buckets),
                            num_buckets,
                            len(target_files),
                        )
                    else:
                        # 目标桶不存在，可能是空数据 — 退化为全扫描但过滤 AppleDouble
                        logger.warning("Target bucket does not exist: bucket_id in %s", target_buckets)
                        _full = _enumerate_bucket_parquet_files(directory)
                        glob_pattern = ("[" + ", ".join(f"'{f}'" for f in _full) + "]") if _full else _duckdb_path(directory / "**/*.parquet")
                else:
                    # 字符串型 ID，无法使用 hash 分桶优化 — 列出所有有效 parquet。
                    _full = _enumerate_bucket_parquet_files(directory)
                    glob_pattern = ("[" + ", ".join(f"'{f}'" for f in _full) + "]") if _full else _duckdb_path(directory / "**/*.parquet")
                    logger.debug("Using bucket read mode (full scan, %d files): %s", len(_full), directory.name)
            else:
                # 没有 itemid 过滤，全扫描 — 同样走显式文件列表
                _full = _enumerate_bucket_parquet_files(directory)
                glob_pattern = ("[" + ", ".join(f"'{f}'" for f in _full) + "]") if _full else _duckdb_path(directory / "**/*.parquet")
                logger.debug("Using bucket read mode (no filter, %d files): %s", len(_full), directory.name)
        else:
            # 普通分区: directory/*.parquet — 同样走显式文件列表过滤 AppleDouble
            _flat = _enumerate_bucket_parquet_files(directory)
            glob_pattern = ("[" + ", ".join(f"'{f}'" for f in _flat) + "]") if _flat else _duckdb_path(directory / "*.parquet")
        
        # 列选择
        if columns:
            select_cols = ", ".join(list(columns))
        else:
            select_cols = "*"
        
        # WHERE 子句（支持多个条件）
        where_conditions = []
        
        # 患者 ID 过滤
        if patient_ids_filter and patient_ids_filter.value:
            id_col = patient_ids_filter.column
            values = patient_ids_filter.value
            
            if isinstance(values, (list, tuple, set)):
                value_list = list(values)
            elif isinstance(values, pd.Series):
                value_list = values.tolist()
            else:
                try:
                    value_list = list(values)
                except TypeError:
                    value_list = [values]
            
            if value_list:
                if len(value_list) == 1:
                    where_conditions.append(f"{id_col} = {value_list[0]}")
                else:
                    values_str = ", ".join(map(str, value_list))
                    where_conditions.append(f"{id_col} IN ({values_str})")
        
        # 🚀 大表 itemid 预过滤优化
        if itemid_filter_config:
            filter_col, filter_ids = itemid_filter_config
            if not filter_ids:
                return pd.DataFrame(columns=list(columns) if columns else [])
            # 检查是否为字符串类型的 ID（如 eICU nursecharting）
            ordered_filter_ids = sorted(
                filter_ids,
                key=lambda item: (type(item).__name__, str(item)),
            )
            id_str = ", ".join(
                _duckdb_sql_literal(value) for value in ordered_filter_ids
            )
            escaped_filter_col = str(filter_col).replace('"', '""')
            where_conditions.append(f'"{escaped_filter_col}" IN ({id_str})')
            if DEBUG_MODE:
                logger.info("Large-table optimization: filtering %s to %d IDs", filter_col, len(filter_ids))
        
        # 🚀 宽表NULL过滤优化：跳过value列为NULL的行
        # 这对于eICU vitalperiodic等宽表非常重要
        # 例如：加载hr概念时，heartrate为NULL的行没有意义，可以跳过
        # 这将145M行→12M行，大幅减少数据传输和pandas处理开销
        if wide_table_value_columns:
            for val_col in wide_table_value_columns:
                where_conditions.append(f"{val_col} IS NOT NULL")
            if DEBUG_MODE:
                logger.info(f"🚀 宽表NULL过滤: {wide_table_value_columns} IS NOT NULL")
        
        # 构建 WHERE 子句
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # 🔧 DuckDB 预排序优化：针对超大表在查询时直接排序
        # pandas sort_values 在 1.46 亿行上需要 25 秒，而 DuckDB ORDER BY 只需 1.9 秒
        # 宽表 (vitalperiodic, vitalaperiodic) 必须预排序，否则后续 sort_values 非常慢
        order_by_clause = ""
        PRESORT_TABLES = {'vitalperiodic', 'vitalaperiodic'}  # 需要预排序的宽表
        if table_name and table_name.lower() in PRESORT_TABLES:
            # 获取表配置以确定排序键
            try:
                table_cfg = self.config.get_table(table_name)
                sort_keys = []
                
                # 获取 ID 列
                if table_cfg.defaults.id_var:
                    sort_keys.append(table_cfg.defaults.id_var)
                else:
                    icustay_cfg = self.config.id_configs.get('icustay')
                    if icustay_cfg:
                        sort_keys.append(icustay_cfg.id)
                
                # 获取时间列
                if table_cfg.defaults.index_var:
                    sort_keys.append(table_cfg.defaults.index_var)
                
                if sort_keys:
                    order_by_clause = f" ORDER BY {', '.join(sort_keys)}"
                    logger.debug(f"🚀 宽表预排序: {table_name} ORDER BY {sort_keys}")
            except Exception as e:
                logger.debug(f"无法获取表配置进行预排序: {e}")
        
        # 🔧 CRITICAL FIX: 使用 union_by_name=true 处理不同分区的 schema 差异
        # HiRID observations 的不同分区有不同的列类型（如 stringvalue）
        # 注意：glob_pattern 可能是单引号包裹的路径，也可能是列表语法 [...]
        if glob_pattern.startswith("["):
            # 列表语法：多个文件
            query = f"SELECT {select_cols} FROM read_parquet({glob_pattern}, union_by_name=true) {where_clause}{order_by_clause}"
        else:
            # 路径/glob 语法
            query = f"SELECT {select_cols} FROM read_parquet('{glob_pattern}', union_by_name=true) {where_clause}{order_by_clause}"
        
        try:
            con = _get_duckdb_connection()
            try:
                arrow_table = con.execute(query).fetch_arrow_table()
                df = _arrow_to_pandas_compat(arrow_table, split_blocks=True)
                del arrow_table
            except Exception:
                df = con.execute(query).fetchdf()
            return df
        except Exception as e:
            logger.warning(f"DuckDB 读取失败，回退到 PyArrow: {e}")
            return self._read_partitioned_data_optimized(directory, columns, patient_ids_filter, itemid_filter_config=itemid_filter_config)
    
    def _read_partitioned_data_optimized(self, directory: Path, columns: Optional[Iterable[str]], patient_ids_filter: Optional[FilterSpec] = None, itemid_filter_config: Optional[tuple] = None) -> pd.DataFrame:
        """读取分区数据（优化版本）
        
        Args:
            itemid_filter_config: 可选的 (列名, itemid集合) 元组，用于大表预过滤
        """
        try:
            import pyarrow.dataset as ds
            
            # 🚀 使用PyArrow Dataset - 最快的方式
            dataset = ds.dataset(
                directory,
                format='parquet',
                partitioning=None,
                exclude_invalid_files=True
            )
            schema_names = list(dataset.schema.names)
            schema_by_lower = {
                str(name).lower(): str(name) for name in schema_names
            }
            requested_columns = list(columns) if columns else None
            actual_columns = None
            actual_to_requested = {}
            if requested_columns is not None:
                actual_columns = []
                for requested in requested_columns:
                    actual = schema_by_lower.get(str(requested).lower())
                    if actual is None:
                        continue
                    if actual not in actual_columns:
                        actual_columns.append(actual)
                    actual_to_requested[actual] = str(requested)
            
            # 构建过滤表达式（支持多个条件）
            filter_exprs = []
            
            # 患者 ID 过滤
            if patient_ids_filter:
                id_col = schema_by_lower.get(
                    str(patient_ids_filter.column).lower()
                )
                values = patient_ids_filter.value
                if isinstance(values, (list, tuple, set)):
                    value_list = list(values)
                elif isinstance(values, pd.Series):
                    value_list = values.tolist()
                else:
                    try:
                        value_list = list(values)
                    except TypeError:
                        value_list = [values]

                if not value_list:
                    wanted_cols = requested_columns or schema_names
                    return pd.DataFrame(columns=wanted_cols)
                if id_col is None:
                    logger.warning(
                        "Partitioned read cannot apply patient filter: "
                        "column %s is absent from %s",
                        patient_ids_filter.column,
                        directory,
                    )
                    return pd.DataFrame(columns=requested_columns or schema_names)

                try:
                    filter_exprs.append(ds.field(id_col).isin(value_list))
                except Exception:
                    pass
            
            # 🚀 大表优化：itemid 预过滤 (AUMC numericitems, MIIV chartevents/labevents)
            if itemid_filter_config:
                try:
                    filter_col, filter_ids = itemid_filter_config
                    actual_filter_col = schema_by_lower.get(
                        str(filter_col).lower()
                    )
                    if actual_filter_col is None:
                        return pd.DataFrame(
                            columns=requested_columns or schema_names
                        )
                    filter_exprs.append(
                        ds.field(actual_filter_col).isin(list(filter_ids))
                    )
                    if DEBUG_MODE:
                        logger.info("Large-table optimization (PyArrow): filtering %s to %d IDs", actual_filter_col, len(filter_ids))
                except Exception:
                    pass
            
            # 合并过滤条件
            filter_expr = None
            if filter_exprs:
                filter_expr = filter_exprs[0]
                for expr in filter_exprs[1:]:
                    filter_expr = filter_expr & expr

            # 批量读取，启用受控的 PyArrow 全局 CPU 线程池。
            #
            # Dataset.to_table(use_threads=...) 接受的是 bool，不是线程数。旧代码
            # 把整数 ``thread_count`` 直接传进去；任何正整数都会变成 True，因此
            # 在 384 核服务器上实际放开了整个 Arrow CPU 池，而不是预期的 8 个
            # 线程。这会在大分区扫描时造成明显的短时内存尖峰。先显式限制全局
            # Arrow CPU 池，再用 bool 控制本次扫描，才能让
            # EASYICU_ARROW_THREADS 在 Windows/macOS/Linux 上真正生效。
            import os as _os
            import pyarrow as _pa
            from easyicu.runtime.parallel_config import get_global_config

            _env_at = _os.environ.get('EASYICU_ARROW_THREADS')
            if _env_at:
                try:
                    thread_count = max(1, int(_env_at))
                except ValueError:
                    logger.warning(
                        "Ignoring invalid EASYICU_ARROW_THREADS=%r",
                        _env_at,
                    )
                    thread_count = min(
                        (_os.cpu_count() or 4),
                        8,
                        get_global_config().max_workers,
                    )
            else:
                thread_count = min(
                    (_os.cpu_count() or 4),
                    8,
                    get_global_config().max_workers,
                )

            if _pa.cpu_count() != thread_count:
                _pa.set_cpu_count(thread_count)
            use_arrow_threads = thread_count > 1

            if requested_columns is not None:
                table = dataset.to_table(
                    columns=actual_columns,
                    filter=filter_expr,
                    use_threads=use_arrow_threads,
                )
            else:
                table = dataset.to_table(
                    filter=filter_expr,
                    use_threads=use_arrow_threads,
                )

            # 转换为 pandas，使用 zero-copy 优化
            frame = _arrow_to_pandas_compat(table, split_blocks=True)
            if requested_columns is not None:
                frame = frame.rename(columns=actual_to_requested)
                frame = frame.reindex(columns=requested_columns)
            return frame
            
        except Exception:
            # 回退到简单方式
            try:
                parquet_files = sorted(directory.glob("*.parquet"))
                if not parquet_files:
                    parquet_files = sorted(directory.glob("*.pq"))
                
                if not parquet_files:
                    return pd.DataFrame(columns=list(columns) if columns else [])
                
                # 准备过滤条件
                filter_ids = None
                id_column = None
                if patient_ids_filter:
                    id_column = patient_ids_filter.column
                    if isinstance(patient_ids_filter.value, (list, tuple, set)):
                        filter_ids = set(patient_ids_filter.value)
                    else:
                        filter_ids = {patient_ids_filter.value}
                
                # 快速读取+过滤
                chunks = []
                for file_path in parquet_files:
                    if columns:
                        df_chunk = pd.read_parquet(file_path, columns=list(columns))
                    else:
                        df_chunk = pd.read_parquet(file_path)
                    
                    # 立即应用过滤（减少内存占用）
                    if filter_ids and id_column and id_column in df_chunk.columns:
                        df_chunk = df_chunk[df_chunk[id_column].isin(filter_ids)]
                    
                    # 只保留有数据的chunk
                    if len(df_chunk) > 0:
                        chunks.append(df_chunk)
                
                # 合并所有chunks
                if chunks:
                    return pd.concat(chunks, ignore_index=True)
                else:
                    # 返回空DataFrame，保持列结构
                    if columns:
                        return pd.DataFrame(columns=list(columns))
                    else:
                        return pd.DataFrame()
            
            except Exception as e:
                # 最终回退到原始实现
                logger.warning(f"优化读取失败: {e}，回退到fallback方法")
                return self._read_partitioned_data_fallback(directory, columns, patient_ids_filter)
    
    def _read_partitioned_data_fallback(self, directory: Path, columns: Optional[Iterable[str]], patient_ids_filter: Optional[FilterSpec] = None) -> pd.DataFrame:
        """Read partitioned data from a directory, respecting format priority."""
        
        # 🔍 调试日志：显示分区加载请求的列
        if DEBUG_MODE and columns:
            logger.debug(f"分区表 {directory.name} 请求的列: {list(columns)}")
        
        # 只支持 Parquet 格式
        files = sorted(directory.glob("*.parquet")) + sorted(directory.glob("*.pq"))
        if not files:
            # 没有找到 parquet 文件
            return pd.DataFrame()
        
        num_files = len(files)
        
        # 准备患者ID过滤器 (支持多种数据库的ID列)
        filter_tuple = None
        if patient_ids_filter and patient_ids_filter.column in ['subject_id', 'hadm_id', 'icustay_id', 'stay_id', 'admissionid', 'patientid']:
            target_ids = set(patient_ids_filter.value) if not isinstance(patient_ids_filter.value, str) else {patient_ids_filter.value}
            filter_tuple = (patient_ids_filter.column, target_ids)
            if DEBUG_MODE: print(f"   📁 加载 {directory.name} ({num_files} 个 parquet 分区) - 过滤 {len(target_ids)} 个患者...")
        else:
            if DEBUG_MODE: print(f"   📁 加载 {directory.name} ({num_files} 个 parquet 分区)...")
        
        # 修复：传递具体的parquet文件列表，而不是目录，避免混合格式问题
        dataset_df = self._read_parquet_dataset(
            directory,
            files,  # 传递具体的parquet文件列表
            columns=list(columns) if columns else None,
            filter_spec=patient_ids_filter,
        )
        if dataset_df is not None:
            return dataset_df
        # Fallback: iterate individual parquet files
        dfs = []
        arrow_filters = None
        if patient_ids_filter:
            arrow_filters = self._build_dataset_filter(patient_ids_filter)
        for f in files:
            if arrow_filters is not None or columns is not None:
                try:
                    import pyarrow.parquet as pq  # type: ignore
                    table = pq.read_table(
                        f,
                        columns=list(columns) if columns else None,
                    )
                    if arrow_filters is not None:
                        table = table.filter(arrow_filters)
                    df = _arrow_to_pandas_compat(table)
                    dfs.append(df)
                    continue
                except Exception:
                    pass  # Fallback to pandas.read_parquet below
            df = _coerce_string_dtypes(pd.read_parquet(f, columns=list(columns) if columns else None))
            # NOTE: This fallback path is rare; primary path uses _arrow_to_pandas_compat above
            if filter_tuple:
                col_name, target_ids = filter_tuple
                if col_name in df.columns:
                    df = df[df[col_name].isin(target_ids)]
            dfs.append(df)
        
        # 合并所有分区
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        
        # 没有找到任何parquet文件
        return pd.DataFrame()

    def _read_parquet_dataset(
        self,
        directory: Path,
        files: Optional[List[Path]] = None,
        columns: Optional[Sequence[str]] = None,
        filter_spec: Optional[FilterSpec] = None,
    ) -> Optional[pd.DataFrame]:
        """Attempt to read a parquet directory via PyArrow Dataset for fast filtering."""
        try:
            import pyarrow.dataset as ds  # type: ignore
        except ImportError:
            return None

        filter_expr = None
        if filter_spec is not None:
            filter_expr = self._build_dataset_filter(filter_spec)
        try:
            # 修复：使用传入的文件列表创建dataset，避免混合格式问题
            if files is not None:
                # 使用具体的parquet文件列表
                dataset = ds.dataset(files, format="parquet")
            else:
                # 回退到原始逻辑（仅包含parquet文件的目录）
                try:
                    dataset = ds.dataset(directory, format="parquet", partitioning="hive")
                except (ValueError, TypeError):
                    dataset = ds.dataset(directory, format="parquet")

            table = dataset.to_table(columns=columns, filter=filter_expr)
            return _arrow_to_pandas_compat(table)
        except (OSError, ValueError, TypeError) as exc:
            if DEBUG_MODE:
                logger.debug("PyArrow dataset read failed for %s: %s", directory, exc)
            return None

    @staticmethod
    def _build_dataset_filter(filter_spec: FilterSpec):
        """Convert FilterSpec to a PyArrow Dataset expression."""
        try:
            import pyarrow.dataset as ds  # type: ignore
        except ImportError:
            return None

        field = ds.field(filter_spec.column)
        if filter_spec.op == FilterOp.EQ:
            return field == filter_spec.value
        if filter_spec.op == FilterOp.IN:
            values = _ensure_sequence(filter_spec.value)
            return field.isin(values)
        if filter_spec.op == FilterOp.BETWEEN:
            lower, upper = filter_spec.value
            return (field >= lower) & (field <= upper)
        return None

    def _read_dataset(
        self,
        table_name: str,
        dataset_cfg: DatasetOptions,
        columns: Optional[Iterable[str]],
        patient_ids_filter: Optional[FilterSpec],
    ) -> pd.DataFrame:
        """Read a table via explicit PyArrow Dataset configuration."""
        try:
            import pyarrow.dataset as ds  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PyArrow is required for dataset-backed tables. "
                "Install pyarrow or remove the dataset configuration."
            ) from exc

        root = dataset_cfg.path or table_name
        root_path = Path(root)
        if not root_path.is_absolute():
            if self.base_path is None:
                raise ValueError(
                    f"Dataset path '{root}' for table '{table_name}' is relative, "
                    "but data source has no base_path."
                )
            root_path = self.base_path / root_path

        partitioning = dataset_cfg.partitioning or "hive"
        format_name = dataset_cfg.format or "parquet"
        options = dataset_cfg.options or {}

        try:
            dataset = ds.dataset(
                root_path,
                format=format_name,
                partitioning=partitioning,
                **options,
            )
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Failed to initialise dataset for table '{table_name}' at {root_path}: {exc}"
            ) from exc

        filter_expr = self._build_dataset_filter(patient_ids_filter) if patient_ids_filter else None
        logger.info("📁 Using PyArrow dataset for %s (%s)", table_name, root_path)

        requested_columns = list(columns) if columns is not None else None
        effective_columns: Optional[List[str]] = None
        missing_columns: List[str] = []
        if requested_columns:
            available = set(dataset.schema.names)
            missing_columns = [col for col in requested_columns if col not in available]
            effective_columns = [col for col in requested_columns if col in available]
            if not effective_columns:
                effective_columns = None

        table = dataset.to_table(columns=effective_columns, filter=filter_expr)
        frame = _arrow_to_pandas_compat(table)

        if requested_columns:
            frame = frame.reindex(columns=requested_columns)
        if missing_columns:
            logger.warning(
                "Dataset %s missing columns %s; filled with NA values", table_name, ", ".join(missing_columns)
            )
        return frame
    

def load_table(
    data_source: ICUDataSource,
    table_name: str,
    *,
    columns: Optional[Iterable[str]] = None,
    filters: Optional[Iterable[FilterSpec]] = None,
) -> ICUTable:
    """Functional façade delegating to :meth:`ICUDataSource.load_table`."""

    return data_source.load_table(table_name, columns=columns, filters=filters)

def _ensure_sequence(value: Any) -> List[Any]:
    """Normalise scalars/iterables for filter construction."""
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    try:
        # 🔧 FIX: 字符串不应该被转换为字符列表
        if isinstance(value, (str, bytes)):
            return [value]
        return list(value)
    except TypeError:
        return [value]

def _coerce_datetime(series: pd.Series) -> pd.Series:
    """Coerce a Series to datetime type, handling various edge cases.
    
    ⚡ 性能优化: 减少重复的类型检查和转换
    """
    # ⚡ 快速路径1: 已经是datetime且无时区
    if pd.api.types.is_datetime64_any_dtype(series):
        if hasattr(series.dt, 'tz') and series.dt.tz is not None:
            return series.dt.tz_localize(None)
        return series
    
    # ⚡ 快速路径2: 数值型不转换
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    # ⚡ 优化: 一次性检查和reset index
    has_dup_idx = series.index.duplicated().any()
    if has_dup_idx:
        series = series.reset_index(drop=True)
    
    # ⚡ 优化: 统一使用coerce模式，避免try-except开销
    try:
        converted = pd.to_datetime(series, errors="coerce", utc=True)
        # 只在转换成功时移除时区
        if converted is not None and hasattr(converted, 'dt'):
            return converted.dt.tz_localize(None)
        return series
    except Exception:
        # 极端情况：返回原值
        return series


def load_bucketed_table_aggregated(
    data_source: "ICUDataSource",
    table_name: str,
    value_column: str,
    itemids: List[int],
    interval_minutes: float = 60.0,
    patient_ids: Optional[List] = None,
    agg_func: str = 'median',  # 'median', 'mean', 'max', 'min', 'first', 'sum'
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    value_min: Optional[float] = None,
    value_max: Optional[float] = None,
    include_unit: bool = False,  # 🚀 包含unit列（供convert_unit callback使用）
    convert_unit_op: Optional[str] = None,  # 🚀 DuckDB内联convert_unit运算符
    convert_unit_factor: Optional[float] = None,  # 🚀 DuckDB内联convert_unit因子
    convert_unit_filter: Optional[str] = None,  # 🚀 DuckDB内联convert_unit单位过滤模式
    value_transform: Optional[str] = None,  # 🚀 通用SQL值转换表达式（如 percent_as_numeric, set_val_na）
    itemid_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    🚀 高性能分桶表加载：在DuckDB中完成聚合降采样

    针对AUMC numericitems、HiRID observations等分桶表优化。
    直接在DuckDB中完成小时聚合，避免加载3700万行到Python再降采样。

    关键优化：
    - 只扫描目标桶（而非全部100个桶）
    - 在DuckDB中完成时间聚合（37M → 2.5M行）
    - 显著降低内存使用（<500MB vs 5GB+）

    Args:
        data_source: ICU数据源
        table_name: 原始表名（如'numericitems'），会自动查找对应的分桶目录
        value_column: 值列名（如'value'）
        itemids: 要提取的itemid列表
        interval_minutes: 时间聚合间隔（分钟），默认60分钟
        patient_ids: 可选的患者ID过滤
        agg_func: 聚合函数，默认'median'（与R ricu一致）
        id_col: ID列名（可选，默认根据数据库推断）
        time_col: 时间列名（可选，默认根据数据库推断）

    Returns:
        聚合后的DataFrame
    """
    import duckdb
    
    # 确定数据库类型
    if patient_ids is not None and len(patient_ids) == 0:
        return pd.DataFrame()
    db_name = data_source.config.name if hasattr(data_source.config, 'name') else 'unknown'
    
    # 🔧 直接查找分桶目录（不依赖_resolve_loader_from_disk）
    base_path = data_source.base_path
    bucket_dir = data_source._resolve_bucket_directory(table_name)
    
    # 🚀 FIX 2026-03-13: 如果没有 bucket 目录，回退到扁平 parquet 目录
    flat_parquet_dir = None
    if bucket_dir is None:
        flat_parquet_dir = data_source._resolve_flat_parquet_directory(table_name)
    
    if bucket_dir is None and flat_parquet_dir is None:
        raise ValueError(f"Cannot find bucketed or flat parquet directory for {table_name}")
    
    # 确定ID列和时间列
    if id_col is None:
        if db_name == 'aumc':
            id_col = 'admissionid'
        elif db_name == 'hirid':
            id_col = 'patientid'
        elif db_name in ('mimic', 'mimic_demo'):
            id_col = 'icustay_id'
        elif db_name in ('sic', 'sic_demo'):
            id_col = 'CaseID'
        elif db_name in ('eicu', 'eicu_demo'):
            id_col = 'patientunitstayid'
        else:
            id_col = 'stay_id'
    
    table_cfg = None
    try:
        table_cfg = data_source.config.get_table(table_name)
    except Exception:
        table_cfg = None

    if time_col is None:
        if db_name == 'aumc':
            # AUMC: most tables use 'measuredat', but drugitems uses 'start'
            if table_name == 'drugitems':
                time_col = 'start'
            else:
                time_col = 'measuredat'  # AUMC使用measuredat（毫秒时间戳）
        elif db_name == 'hirid':
            time_col = 'datetime'
        elif db_name in ('sic', 'sic_demo'):
            time_col = 'Offset'
        elif db_name in ('eicu', 'eicu_demo'):
            configured_time_col = getattr(getattr(table_cfg, 'defaults', None), 'index_var', None)
            time_col = configured_time_col or 'labresultoffset'
        else:
            # MIIV/MIMIC: most tables use 'charttime', but inputevents/procedureevents use 'starttime'
            _starttime_tables = {'inputevents', 'inputevents_mv', 'inputevents_cv', 'procedureevents'}
            if table_name in _starttime_tables:
                time_col = 'starttime'
            else:
                time_col = 'charttime'
    
    # 确定itemid列名。概念源显式传入的 sub_var 优先于表默认值：
    # eICU intakeoutput 的常规液体概念使用 celllabel，官方尿量表型使用
    # 更严格的 cellpath。
    if itemid_col is not None:
        itemid_col = str(itemid_col)
    elif db_name == 'aumc':
        itemid_col = 'itemid'
    elif db_name == 'hirid':
        itemid_col = 'variableid'
    elif db_name in ('sic', 'sic_demo'):
        # SIC tables: data_float_h uses DataID, laboratory uses LaboratoryID
        # Caller should pass the correct column via source.sub_var
        itemid_col = 'DataID' if table_name in ('data_float_h',) else 'LaboratoryID'
    elif db_name in ('eicu', 'eicu_demo'):
        configured_sub_var = getattr(getattr(table_cfg, 'defaults', None), 'sub_var', None)
        if configured_sub_var:
            itemid_col = configured_sub_var
        elif table_name == 'nursecharting':
            itemid_col = 'nursingchartcelltypevalname'
        elif table_name == 'intakeoutput':
            # 2026-05-19 fix: intakeoutput's defaults block in
            # data-sources.json has no `sub_var`, but the concept-dict
            # consistently passes `sub_var: celllabel`. Without this
            # explicit fall-back the DuckDB query referenced `labname`
            # and broke all eicu fluid-balance / total_input_ml extractions.
            itemid_col = 'celllabel'
        elif table_name in ('infusiondrug', 'medication'):
            itemid_col = 'drugname'
        elif table_name == 'treatment':
            itemid_col = 'treatmentstring'
        elif table_name == 'respiratorycare':
            itemid_col = 'respchartvaluelabel'
        else:
            itemid_col = 'labname'
    else:
        itemid_col = 'itemid'
    
    # 计算目标桶
    conn = _get_duckdb_connection()
    
    # 根据目录类型构建文件列表
    if bucket_dir is not None:
        # 分桶目录：通过 hash(itemid) 选择目标桶
        itemid_list = list(itemids)
        _, _, target_files = data_source._get_bucket_files_for_ids(
            bucket_dir,
            itemid_list,
            duckdb,
        )
        
        if not target_files:
            return pd.DataFrame()

        file_list_str = ", ".join(f"'{_duckdb_path(f)}'" for f in target_files)
        glob_pattern = f"[{file_list_str}]"
    else:
        # 扁平 parquet 目录：扫描所有文件，依赖 WHERE 过滤
        target_files = tuple(sorted(flat_parquet_dir.glob("*.parquet")))
        glob_pattern = f"'{_duckdb_path(flat_parquet_dir)}/*.parquet'"

    raw_columns = data_source._get_parquet_columns_for_files(target_files)
    if not raw_columns:
        try:
            raw_columns = {
                col[0]
                for col in conn.execute(
                    f"SELECT * FROM read_parquet({glob_pattern}, union_by_name=true) LIMIT 0"
                ).description
            }
        except Exception:
            raw_columns = set()

    # 🔧 FIX 2026-05-11: case-insensitive 列存在性检查（同 multi-concept 路径，应对 MIMIC-III 1.4 大写列名）
    raw_columns_by_lower = {str(c).lower(): str(c) for c in raw_columns}
    raw_columns_lower = set(raw_columns_by_lower)

    # 🔧 FIX 2026-05-16: detect VARCHAR-typed numeric value columns (e.g. eicu
    # respiratorycharting.respchartvalue) and rewrite SQL references to use
    # TRY_CAST(... AS DOUBLE). Without this, WHERE bounds and MEDIAN() fail with
    # "Cannot compare values of type VARCHAR and type DECIMAL".
    _value_col_is_string = False
    try:
        _types = conn.execute(
            f"DESCRIBE SELECT {value_column} FROM read_parquet({glob_pattern}, union_by_name=true) LIMIT 0"
        ).fetchall()
        if _types:
            _t = str(_types[0][1]).upper()
            _value_col_is_string = _t.startswith("VARCHAR") or _t == "STRING" or _t.startswith("TEXT")
    except Exception:
        _value_col_is_string = False
    if _value_col_is_string:
        _value_col_numeric = f"TRY_CAST({value_column} AS DOUBLE)"
    else:
        _value_col_numeric = value_column

    patient_filter_col = id_col
    patient_filter_values = patient_ids
    join_left_col = id_col
    output_id_col = id_col

    hospital_tables = {'prescriptions', 'labevents', 'microbiologyevents', 'emar', 'pharmacy', 'services'}
    if (
        db_name in ('miiv', 'miiv_demo', 'mimic', 'mimic_demo')
        and table_name in hospital_tables
        and id_col not in raw_columns_lower
        and 'hadm_id' in raw_columns_lower
    ):
        stay_col = 'icustay_id' if db_name in ('mimic', 'mimic_demo') else 'stay_id'
        source_id_col = 'icustay_id' if db_name in ('mimic', 'mimic_demo') else 'stay_id'
        join_left_col = 'hadm_id'
        output_id_col = stay_col
        if patient_ids:
            try:
                icustays_tbl = data_source.load_table(
                    'icustays',
                    columns=[source_id_col, 'hadm_id', 'intime'],
                    filters=[FilterSpec(column=source_id_col, op=FilterOp.IN, value=patient_ids)],
                    verbose=False,
                )
                icustays_df = icustays_tbl.dataframe()
                mapped_hadm_ids = icustays_df['hadm_id'].dropna().unique().tolist() if 'hadm_id' in icustays_df.columns else []
                if mapped_hadm_ids:
                    patient_filter_col = 'hadm_id'
                    patient_filter_values = mapped_hadm_ids
            except Exception:
                pass

    # 构建WHERE条件
    _has_inline_convert = (
        convert_unit_op is not None and convert_unit_factor is not None
    )
    where_conditions = []
    
    # itemid过滤 (handle both numeric and string itemids)
    _is_string_ids = any(isinstance(x, str) for x in itemids)
    if _is_string_ids:
        ids_str = ", ".join(f"'{x}'" for x in itemids)
    else:
        ids_str = ", ".join(str(x) for x in itemids)
    where_conditions.append(f"{itemid_col} IN ({ids_str})")
    where_conditions.append(f"{value_column} IS NOT NULL")
    
    # 患者过滤
    if patient_filter_values:
        patient_str = ", ".join(str(x) for x in patient_filter_values)
        where_conditions.append(f"{patient_filter_col} IN ({patient_str})")
    
    # 🔧 FIX 2026-02: 在DuckDB层过滤原始值范围（匹配R ricu的行为）
    # R ricu: clamp_var() 先将超范围值设为NA → 再按小时聚合（NA不参与median）
    # EasyICU: 必须在聚合前过滤，否则per-itemid-per-hour的median可能超范围
    # 导致某些小时全部被丢弃（即使该小时有其他itemid的合法值）
    # 🚀 当 value_transform 或 inline convert_unit 存在时，min/max 不在 WHERE 过滤 raw column
    # - value_transform (fahr_to_cels/percent_as_numeric): raw column 可能是 VARCHAR 或不同单位
    # - _has_inline_convert (binary_op(*, factor)): raw 值单位与 min/max 不同（如 µmol/L vs mg/dL）
    # 对这些路径不能在 WHERE 中按 raw column 过滤；下方会先转换每个原始值，
    # 再用 CASE 应用 bounds 后聚合，并保留 raw/bounded counts 供 unit-suspect 审计。
    
    where_clause = "WHERE " + " AND ".join(where_conditions)
    
    # 聚合函数映射
    agg_map = {
        'median': 'MEDIAN',
        'mean': 'AVG',
        'max': 'MAX',
        'min': 'MIN',
        'first': 'FIRST',
        'sum': 'SUM',
    }
    duckdb_agg = agg_map.get(agg_func, 'MEDIAN')
    
    # 构建时间聚合表达式
    # AUMC: measuredat是毫秒时间戳，需要转换后再聚合
    # 🚀 DuckDB内联convert_unit：在聚合前转换单位值
    # 当 convert_unit_op/factor 有效时，用 CASE WHEN 在聚合前转换值
    # 并且不按 itemid 分组（R ricu 行为：跨 itemid 取 median）
    _raw_count_col = "__easyicu_raw_transformed_non_null"
    _bounded_count_col = "__easyicu_bounded_transformed_non_null"
    _needs_bounds_diagnostics = bool(
        value_min is not None or value_max is not None
    )

    def _bounded_value(expr: str) -> str:
        """Return the transformed value with declared bounds applied per row."""
        conditions = []
        if value_min is not None:
            conditions.append(f"({expr}) >= {value_min}")
        if value_max is not None:
            conditions.append(f"({expr}) <= {value_max}")
        if conditions:
            return f"CASE WHEN {' AND '.join(conditions)} THEN ({expr}) END"
        return expr

    def _aggregate_select(expr: str):
        """Build bounded aggregation plus raw-value diagnostics."""
        bounded_expr = _bounded_value(expr)
        bounded_agg = f"{duckdb_agg}({bounded_expr})"
        unbounded_agg = f"{duckdb_agg}({expr})"
        select = f"{bounded_agg} as {value_column}"
        if _needs_bounds_diagnostics:
            select += (
                f", COUNT({expr}) as {_raw_count_col}"
                f", COUNT({bounded_expr}) as {_bounded_count_col}"
            )
        return select, bounded_agg, unbounded_agg
    
    # 🔧 FIX: 如果 convert_unit_filter 需要 unit 列但表没有该列，
    # 仍然应用转换（匹配 R ricu 行为：无 unit 列时按 item ID 的隐含单位转换）。
    # 例如 HiRID hgb 三个 variableid 已知为 g/L，filter='g/l' 在表上无 unit 列时
    # 应当无条件应用 *0.1 转换为 g/dL，否则后续 filter_bounds 会丢掉所有行。
    if _has_inline_convert and convert_unit_filter:
        _unit_col_name = None
        if 'unit' in raw_columns_by_lower:
            _unit_col_name = raw_columns_by_lower['unit']
        else:
            try:
                _table_cfg = data_source.config.get_table(table_name)
                _configured_unit = getattr(_table_cfg.defaults, 'unit_var', None)
                if _configured_unit:
                    _unit_col_name = raw_columns_by_lower.get(
                        str(_configured_unit).lower()
                    )
            except Exception:
                pass
        if _unit_col_name is None:
            # 表无 unit 列：按 item ID 隐含单位，filter 视为"应用此转换"提示
            import logging
            logging.getLogger(__name__).info(
                f"ℹ️ 表无 unit 列，按隐含单位应用 convert_unit（filter={convert_unit_filter}）"
            )
            convert_unit_filter = None  # 清空 filter，使后续直接走无条件转换分支
    
    if _has_inline_convert:
        # 构建 CASE WHEN 表达式做单位转换
        _op_sql = {"*": "*", "/": "/", "+": "+", "-": "-"}.get(convert_unit_op, "*")
        if convert_unit_filter:
            # 有单位过滤：只对匹配的行做转换
            # 使用 DuckDB regexp_matches 进行正则匹配（case-insensitive）
            _value_expr = f"CASE WHEN regexp_matches(CAST({_unit_col_name} AS VARCHAR), '(?i){convert_unit_filter}') THEN {value_column} {_op_sql} {convert_unit_factor} ELSE {value_column} END"
        else:
            # 无单位过滤：转换所有行
            _value_expr = f"{value_column} {_op_sql} {convert_unit_factor}"
        _agg_value_expr, _bounded_agg, _unbounded_agg = _aggregate_select(_value_expr)
    elif value_transform:
        # 🚀 通用值转换表达式（percent_as_numeric, set_val_na, fahr_to_cels）
        # 在聚合表达式内部对转换后的原始值应用 bounds。
        _agg_value_expr, _bounded_agg, _unbounded_agg = _aggregate_select(
            value_transform
        )
    else:
        _agg_value_expr, _bounded_agg, _unbounded_agg = _aggregate_select(
            _value_col_numeric
        )
    
    _query_bounded_agg = _bounded_agg
    _query_unbounded_agg = _unbounded_agg
    unit_select = ",\n            ANY_VALUE(unit) as unit" if include_unit else ""
    
    # R ricu 的 change_interval 聚合不按 itemid 分组:
    # 它将同一 (patient, hour) 的所有 itemid 数据池化后取 median/max 等。
    # 2026-03-11 FIX: 始终池化所有 itemid。R ricu 从不按 itemid 分组聚合。
    # 对有 convert_unit 回调的概念，_has_inline_convert 已在 SQL 中统一单位。
    # 对无回调的概念（如 resp: 8874+12266），所有 itemid 共享同一单位，池化安全。
    # 混合单位 itemid（如 AUMC pco2 的 mmHg+kPa）通过池化中位数的鲁棒性 +
    # 后续 filter_bounds (min/max) 处理。
    _group_itemid = ""
    _select_itemid = ""
    _order_itemid = ""
    
    if db_name == "aumc":
        # AUMC measuredat/admittedat share a database-wide millisecond clock.
        # The bucket boundary must be chosen *after* subtracting admittedat.
        # Flooring the absolute clock first shifts every non-hour admission and
        # can mix observations from adjacent ICU-relative hours.
        admissions_relation = _aumc_admissions_relation_sql(data_source)
        relative_bucket_minutes = (
            f"FLOOR(((o.{time_col} - a.admittedat) / 60000.0) / "
            f"{interval_minutes}) * {interval_minutes}"
        )
        # Keep the existing downstream contract: measuredat_minutes remains on
        # AUMC's absolute minute clock so _align_time_to_admission subtracts the
        # origin exactly once. Adding a floored relative bucket to admittedat
        # also preserves FLOOR semantics for negative pre-admission times.
        time_round_expr = (
            f"(a.admittedat / 60000.0 + {relative_bucket_minutes})"
        )
        output_time_expr = f"{time_round_expr} as measuredat_minutes"
        aumc_where_conditions = [
            f"o.{itemid_col} IN ({ids_str})",
            f"o.{value_column} IS NOT NULL",
            "a.admittedat IS NOT NULL",
        ]
        if patient_filter_values:
            patient_str = ", ".join(str(x) for x in patient_filter_values)
            aumc_where_conditions.append(
                f"o.{patient_filter_col} IN ({patient_str})"
            )
        aumc_where_clause = "WHERE " + " AND ".join(aumc_where_conditions)
        query = f"""
        SELECT 
            o.{id_col} AS {id_col},
            {output_time_expr}{_select_itemid},
            {_agg_value_expr}{unit_select}
        FROM read_parquet({glob_pattern}, union_by_name=true) o
        JOIN {admissions_relation} a ON o.{id_col} = a.admissionid
        {aumc_where_clause}
        GROUP BY o.{id_col}, {time_round_expr}{_group_itemid}
        ORDER BY o.{id_col}, 2{_order_itemid}
        """
    elif db_name == "hirid":
        # 🚀 HiRID 优化: 在 DuckDB 中直接完成时间转换（datetime → 相对入院小时数）
        # 这样避免了 Python 中的 merge + 时间计算开销（从 20s 优化到 0.6s）
        # 🔧 FIX: HiRID 的 general_table 可能有多种文件名和格式
        general_path = data_source.base_path / "general_table.parquet"
        general_read_func = "read_parquet"
        if not general_path.exists():
            # R ricu 惯例使用 general.parquet
            general_alt = data_source.base_path / "general.parquet"
            if general_alt.exists():
                general_path = general_alt
            else:
                # 最后回退到 CSV
                general_csv = data_source.base_path / "general_table.csv"
                if general_csv.exists():
                    general_path = general_csv
                    general_read_func = "read_csv"

        # HiRID: 使用 general 表的 admissiontime 计算相对小时数
        time_round_expr = f"FLOOR(EPOCH(o.{time_col} - CAST(a.admissiontime AS TIMESTAMP)) / 3600.0 / {interval_minutes / 60}) * {interval_minutes / 60}"
        output_time_expr = f"{time_round_expr} as charttime"

        # 🔧 修复: 为 HiRID 的 JOIN 查询添加表别名前缀
        # 因为使用了 JOIN，列名需要明确来自哪个表
        hirid_where_clause = where_clause.replace(f"{itemid_col}", f"o.{itemid_col}")
        hirid_where_clause = hirid_where_clause.replace(
            f"{id_col} IN", f"o.{id_col} IN"
        )

        # 🚀 HiRID 内联 convert_unit
        if _has_inline_convert:
            # HiRID 内联 convert_unit: 为列名添加表别名 'o.'
            _op_sql = {"*": "*", "/": "/", "+": "+", "-": "-"}.get(convert_unit_op, "*")
            if convert_unit_filter:
                _hirid_value_expr = f"CASE WHEN regexp_matches(CAST(o.{_unit_col_name} AS VARCHAR), '(?i){convert_unit_filter}') THEN o.{value_column} {_op_sql} {convert_unit_factor} ELSE o.{value_column} END"
            else:
                _hirid_value_expr = f"o.{value_column} {_op_sql} {convert_unit_factor}"
            _hirid_agg_expr, _query_bounded_agg, _query_unbounded_agg = (
                _aggregate_select(_hirid_value_expr)
            )
        elif value_transform:
            # 🚀 通用值转换表达式 — bounds 在聚合表达式内部应用
            _vt_aliased = value_transform.replace(
                f'"{value_column}"', f'o."{value_column}"'
            )
            _hirid_agg_expr, _query_bounded_agg, _query_unbounded_agg = (
                _aggregate_select(_vt_aliased)
            )
        else:
            _hirid_value_expr = (
                f"TRY_CAST(o.{value_column} AS DOUBLE)"
                if _value_col_is_string
                else f"o.{value_column}"
            )
            _hirid_agg_expr, _query_bounded_agg, _query_unbounded_agg = (
                _aggregate_select(_hirid_value_expr)
            )
        # R ricu 不按 variableid 分组（跨 variableid 池化取聚合）
        _hirid_group_itemid = ""
        _hirid_select_itemid = ""
        _hirid_order_itemid = ""

        hirid_unit_select = (
            ",\n            ANY_VALUE(o.unit) as unit" if include_unit else ""
        )

        query = f"""
        WITH adm AS (
            SELECT patientid, CAST(admissiontime AS TIMESTAMP) as admissiontime 
            FROM {general_read_func}('{_duckdb_path(general_path)}')
        )
        SELECT 
            o.{id_col},
            {output_time_expr}{_hirid_select_itemid},
            {_hirid_agg_expr}{hirid_unit_select}
        FROM read_parquet({glob_pattern}, union_by_name=true) o
        JOIN adm a ON o.{id_col} = a.patientid
        {hirid_where_clause}
        GROUP BY o.{id_col}, {time_round_expr}{_hirid_group_itemid}
        ORDER BY o.{id_col}, 2{_hirid_order_itemid}
        """
    elif db_name in ("miiv", "miiv_demo", "mimic", "mimic_demo"):
        # MIIV/MIMIC: charttime is absolute datetime, need JOIN with icustays to compute relative hours
        # Find icustays parquet
        icustays_path = base_path / "icustays.parquet"
        if not icustays_path.exists():
            icustays_path = base_path / "icu" / "icustays.parquet"
        if not icustays_path.exists():
            # Fallback: try CSV
            icustays_csv = base_path / "icustays.csv.gz"
            if icustays_csv.exists():
                icustays_path = icustays_csv

        stay_col = "icustay_id" if db_name in ("mimic", "mimic_demo") else "stay_id"

        # Compute hours from admission: FLOOR(EPOCH(charttime - intime) / 3600)
        _interval_hours = interval_minutes / 60.0
        time_round_expr = f"FLOOR(EPOCH(o.{time_col} - CAST(a.intime AS TIMESTAMP)) / 3600.0 / {_interval_hours}) * {_interval_hours}"
        output_time_expr = f"{time_round_expr} as charttime"

        # Alias value references for JOIN query
        if _has_inline_convert:
            _op_sql = {"*": "*", "/": "/", "+": "+", "-": "-"}.get(convert_unit_op, "*")
            if convert_unit_filter:
                _mimic_value_expr = f"CASE WHEN regexp_matches(CAST(o.{_unit_col_name} AS VARCHAR), '(?i){convert_unit_filter}') THEN o.{value_column} {_op_sql} {convert_unit_factor} ELSE o.{value_column} END"
            else:
                _mimic_value_expr = f"o.{value_column} {_op_sql} {convert_unit_factor}"
            _mimic_agg_expr, _query_bounded_agg, _query_unbounded_agg = (
                _aggregate_select(_mimic_value_expr)
            )
        elif value_transform:
            # 🚀 通用值转换表达式 — bounds 在聚合表达式内部应用
            _vt_aliased = value_transform.replace(
                f'"{value_column}"', f'o."{value_column}"'
            )
            _mimic_agg_expr, _query_bounded_agg, _query_unbounded_agg = (
                _aggregate_select(_vt_aliased)
            )
        else:
            _mimic_value_expr = (
                f"TRY_CAST(o.{value_column} AS DOUBLE)"
                if _value_col_is_string
                else f"o.{value_column}"
            )
            _mimic_agg_expr, _query_bounded_agg, _query_unbounded_agg = (
                _aggregate_select(_mimic_value_expr)
            )
        _mimic_unit_select = (
            ",\n            ANY_VALUE(o.unit) as unit" if include_unit else ""
        )

        # Alias WHERE clause for JOIN
        _mimic_where = where_clause.replace(f"{itemid_col}", f"o.{itemid_col}")
        _mimic_where = _mimic_where.replace(
            f"{patient_filter_col} IN", f"o.{patient_filter_col} IN"
        )
        # Also handle value_column filters
        _mimic_where = _mimic_where.replace(
            f"{value_column} >=", f"o.{value_column} >="
        )
        _mimic_where = _mimic_where.replace(
            f"{value_column} <=", f"o.{value_column} <="
        )
        _mimic_where = _mimic_where.replace(
            f"{value_column} IS NOT NULL", f"o.{value_column} IS NOT NULL"
        )

        _read_func = (
            "read_csv" if str(icustays_path).endswith(".csv.gz") else "read_parquet"
        )

        if join_left_col == "hadm_id":
            # Hospital tables (labevents, prescriptions, etc.): need rolling join
            # R ricu uses data.table rolling join with roll=-Inf on outtime:
            # - For each event, find the ICU stay with smallest outtime >= charttime (NOCB)
            # - Events after all outtimes go to the last stay (rollends=TRUE)
            # Must join with ALL stays for matching hadm_ids (not just target stays),
            # then filter to target stays after assignment.
            _target_stay_ids_str = (
                ", ".join(str(x) for x in patient_ids) if patient_ids else ""
            )

            # Build event WHERE: reuse _mimic_where but remove hadm_id patient filter
            # (we filter by hadm_id IN target_hadms instead, and by stay_id after assignment)
            _rolling_event_where = _mimic_where
            if patient_filter_col == "hadm_id" and patient_filter_values:
                _hadm_ids_str = ", ".join(str(x) for x in patient_filter_values)
                _rolling_event_where = _rolling_event_where.replace(
                    f"AND o.hadm_id IN ({_hadm_ids_str})", ""
                )

            # Time expression without table alias (for outer query on events_joined CTE)
            _outer_time_expr = f"FLOOR(EPOCH({time_col} - CAST(intime AS TIMESTAMP)) / 3600.0 / {_interval_hours}) * {_interval_hours}"
            # Value agg expression without table alias
            _outer_agg_expr = _mimic_agg_expr.replace("o.", "")
            _query_bounded_agg = _query_bounded_agg.replace("o.", "")
            _query_unbounded_agg = _query_unbounded_agg.replace("o.", "")
            # Extra columns needed in CTE for outer agg expression (e.g. unit column for convert_unit)
            _cte_extra_cols = ""
            if _has_inline_convert and convert_unit_filter and _unit_col_name:
                _cte_extra_cols = f", o.{_unit_col_name}"
            elif value_transform:
                # set_val(NA)-style value transforms can reference the configured
                # unit column even though they are not represented by
                # convert_unit_op/factor. Hospital-table rolling assignment moves
                # aggregation to an outer CTE, so that referenced unit must travel
                # with the event row (D-dimer FEU exclusion is the canonical case).
                _configured_unit_col = None
                try:
                    _configured_unit = getattr(
                        data_source.config.get_table(table_name).defaults,
                        "unit_var",
                        None,
                    )
                    if _configured_unit:
                        _configured_unit_col = raw_columns_by_lower.get(
                            str(_configured_unit).lower()
                        )
                except Exception:
                    _configured_unit_col = None
                if (
                    _configured_unit_col
                    and str(_configured_unit_col).lower()
                    in value_transform.lower()
                ):
                    _cte_extra_cols = f", o.{_configured_unit_col}"

            if patient_ids:
                # Targeted patient loading: filter hadm_ids by target stay_ids
                _hadm_filter = f"WHERE {stay_col} IN ({_target_stay_ids_str})"
                _hadm_scope_filter = (
                    "AND o.hadm_id IN (SELECT hadm_id FROM target_hadms)"
                )
                _final_stay_filter = f"AND {stay_col} IN ({_target_stay_ids_str})"

                query = f"""
                WITH target_hadms AS (
                    SELECT DISTINCT hadm_id
                    FROM {_read_func}('{_duckdb_path(icustays_path)}')
                    {_hadm_filter}
                ),
                all_stays AS (
                    SELECT {stay_col}, hadm_id, CAST(intime AS TIMESTAMP) as intime, CAST(outtime AS TIMESTAMP) as outtime
                    FROM {_read_func}('{_duckdb_path(icustays_path)}')
                    WHERE hadm_id IN (SELECT hadm_id FROM target_hadms)
                ),
                events_joined AS (
                    SELECT o.{value_column}, o.{time_col}, o.hadm_id{_cte_extra_cols},
                           a.{stay_col}, a.intime,
                           ROW_NUMBER() OVER (
                               PARTITION BY o.hadm_id, o.{time_col}, o.{value_column}
                               ORDER BY
                                   CASE WHEN a.outtime >= o.{time_col} THEN a.outtime END ASC NULLS LAST,
                                   a.outtime DESC
                           ) as _rn
                    FROM read_parquet({glob_pattern}, union_by_name=true) o
                    JOIN all_stays a ON o.hadm_id = a.hadm_id
                    {_rolling_event_where}
                    AND o.{value_column} IS NOT NULL
                    {_hadm_scope_filter}
                )
                SELECT {stay_col} as {output_id_col},
                       {_outer_time_expr} as charttime,
                       {_outer_agg_expr}
                FROM events_joined
                WHERE _rn = 1 {_final_stay_filter}
                GROUP BY {stay_col}, {_outer_time_expr}
                ORDER BY {stay_col}, 2
                """
            else:
                # ALL patients: join with all icustays, no stay_id filter
                query = f"""
                WITH all_stays AS (
                    SELECT {stay_col}, hadm_id, CAST(intime AS TIMESTAMP) as intime, CAST(outtime AS TIMESTAMP) as outtime
                    FROM {_read_func}('{_duckdb_path(icustays_path)}')
                ),
                events_joined AS (
                    SELECT o.{value_column}, o.{time_col}, o.hadm_id{_cte_extra_cols},
                           a.{stay_col}, a.intime,
                           ROW_NUMBER() OVER (
                               PARTITION BY o.hadm_id, o.{time_col}, o.{value_column}
                               ORDER BY
                                   CASE WHEN a.outtime >= o.{time_col} THEN a.outtime END ASC NULLS LAST,
                                   a.outtime DESC
                           ) as _rn
                    FROM read_parquet({glob_pattern}, union_by_name=true) o
                    JOIN all_stays a ON o.hadm_id = a.hadm_id
                    {_rolling_event_where}
                    AND o.{value_column} IS NOT NULL
                )
                SELECT {stay_col} as {output_id_col},
                       {_outer_time_expr} as charttime,
                       {_outer_agg_expr}
                FROM events_joined
                WHERE _rn = 1
                GROUP BY {stay_col}, {_outer_time_expr}
                ORDER BY {stay_col}, 2
                """
        else:
            # ICU tables (chartevents, etc.): direct JOIN on stay_id
            join_predicate = f"o.{join_left_col} = a.{join_left_col}"

            query = f"""
            WITH adm AS (
                SELECT {stay_col}, hadm_id, CAST(intime AS TIMESTAMP) as intime, CAST(outtime AS TIMESTAMP) as outtime
                FROM {_read_func}('{_duckdb_path(icustays_path)}')
            )
            SELECT
                a.{output_id_col} as {output_id_col},
                {output_time_expr},
                {_mimic_agg_expr}{_mimic_unit_select}
            FROM read_parquet({glob_pattern}, union_by_name=true) o
            JOIN adm a ON {join_predicate}
            {_mimic_where}
            GROUP BY a.{output_id_col}, {time_round_expr}
            ORDER BY a.{output_id_col}, 2
            """
    elif db_name in ("sic", "sic_demo"):
        # SICdb Offset is relative to the primary PDMS admission, not ICU
        # admission. Anchor every bucket to cases.ICUOffset. See the official
        # CITI-USZ/SICdb schema documentation for cases.ICUOffset.
        _interval_seconds = interval_minutes * 60.0
        cases_relation = _sic_cases_relation_sql(data_source)
        time_round_expr = (
            f'FLOOR((o."Offset" - c.ICUOffset) / {_interval_seconds})'
        )
        output_time_expr = f"{time_round_expr} as charttime"

        if _has_inline_convert:
            _op_sql = {"*": "*", "/": "/", "+": "+", "-": "-"}.get(convert_unit_op, "*")
            if convert_unit_filter:
                _sic_value_expr = f"CASE WHEN regexp_matches(CAST(o.{_unit_col_name} AS VARCHAR), '(?i){convert_unit_filter}') THEN o.{value_column} {_op_sql} {convert_unit_factor} ELSE o.{value_column} END"
            else:
                _sic_value_expr = f"o.{value_column} {_op_sql} {convert_unit_factor}"
            _sic_agg_expr, _query_bounded_agg, _query_unbounded_agg = _aggregate_select(
                _sic_value_expr
            )
        elif value_transform:
            # 🚀 通用值转换表达式 — bounds 在聚合表达式内部应用
            _vt_aliased = value_transform.replace(
                f'"{value_column}"', f'o."{value_column}"'
            )
            _sic_agg_expr, _query_bounded_agg, _query_unbounded_agg = _aggregate_select(
                _vt_aliased
            )
        else:
            _sic_value_expr = (
                f"TRY_CAST(o.{value_column} AS DOUBLE)"
                if _value_col_is_string
                else f"o.{value_column}"
            )
            _sic_agg_expr, _query_bounded_agg, _query_unbounded_agg = _aggregate_select(
                _sic_value_expr
            )

        sic_where_clause = where_clause.replace(
            f"{itemid_col}", f"o.{itemid_col}"
        )
        sic_where_clause = sic_where_clause.replace(
            f"{id_col} IN", f"o.{id_col} IN"
        )
        sic_where_clause = sic_where_clause.replace(
            f"{value_column} IS NOT NULL", f"o.{value_column} IS NOT NULL"
        )
        query = f"""
        SELECT
            o.{id_col} AS {id_col},
            {output_time_expr},
            {_sic_agg_expr}
        FROM read_parquet({glob_pattern}, union_by_name=true) o
        JOIN {cases_relation} c ON o.{id_col} = c.CaseID
        {sic_where_clause}
          AND c.ICUOffset IS NOT NULL
        GROUP BY o.{id_col}, {time_round_expr}
        ORDER BY o.{id_col}, 2
        """
    elif db_name in ("eicu", "eicu_demo"):
        # eICU: time columns are already relative offsets in minutes
        # 🔧 FIX: 输出保持分钟单位（与原始数据一致），让 _align_time_to_admission 统一转换为小时
        # 之前输出小时导致 _align_time_to_admission 再除以60（双重转换）
        time_round_expr = f"FLOOR({time_col} / {interval_minutes}) * {interval_minutes}"
        output_time_expr = f"{time_round_expr} as charttime"
        default_unit_select = (
            ",\n            ANY_VALUE(unit) as unit" if include_unit else ""
        )
        query = f"""
        SELECT 
            {id_col},
            {output_time_expr}{_select_itemid},
            {_agg_value_expr}{default_unit_select}
        FROM read_parquet({glob_pattern}, union_by_name=true)
        {where_clause}
        GROUP BY {id_col}, {time_round_expr}{_group_itemid}
        ORDER BY {id_col}, 2{_order_itemid}
        """
    else:
        # Generic fallback: assume time_col is numeric in minutes
        time_round_expr = f"FLOOR({time_col} / {interval_minutes}) * {interval_minutes}"
        output_time_expr = f"{time_round_expr} as charttime"
        default_unit_select = (
            ",\n            ANY_VALUE(unit) as unit" if include_unit else ""
        )
        query = f"""
        SELECT 
            {id_col},
            {output_time_expr}{_select_itemid},
            {_agg_value_expr}{default_unit_select}
        FROM read_parquet({glob_pattern}, union_by_name=true)
        {where_clause}
        GROUP BY {id_col}, {time_round_expr}{_group_itemid}
        ORDER BY {id_col}, 2{_order_itemid}
        """
    
    def _execute_aggregation(sql: str) -> pd.DataFrame:
        try:
            arrow_table = conn.execute(sql).fetch_arrow_table()
            result = _arrow_to_pandas_compat(arrow_table, split_blocks=True)
            del arrow_table
            return result
        except Exception:
            return conn.execute(sql).fetchdf()

    try:
        df = _execute_aggregation(query)
        bounds_diagnostics = None
        if _needs_bounds_diagnostics and _raw_count_col in df.columns:
            raw_non_null = int(
                pd.to_numeric(df[_raw_count_col], errors="coerce").fillna(0).sum()
            )
            bounded_non_null = int(
                pd.to_numeric(df[_bounded_count_col], errors="coerce").fillna(0).sum()
            )
            bounded_aggregate_non_null = int(
                pd.to_numeric(df[value_column], errors="coerce").notna().sum()
            )
            unit_suspect = bool(
                raw_non_null >= 100
                and bounded_non_null == 0
                and bounded_aggregate_non_null == 0
            )
            unbounded_retry = False
            if unit_suspect:
                unbounded_query = query.replace(
                    _query_bounded_agg,
                    _query_unbounded_agg,
                    1,
                )
                if unbounded_query == query:
                    raise RuntimeError(
                        "Could not construct unbounded retry for unit-suspect values"
                    )
                df = _execute_aggregation(unbounded_query)
                unbounded_retry = True
            df = df.drop(columns=[_raw_count_col, _bounded_count_col], errors="ignore")
            bounds_diagnostics = {
                "bounds_raw_transformed_non_null": raw_non_null,
                "bounds_bounded_transformed_non_null": bounded_non_null,
                "bounds_bounded_aggregate_non_null": bounded_aggregate_non_null,
                "bounds_unit_suspect": unit_suspect,
                "bounds_unbounded_retry": unbounded_retry,
            }
            df.attrs["easyicu_bounds_loader"] = bounds_diagnostics
        if (
            bounds_diagnostics is not None
            and not value_transform
            and not _has_inline_convert
            and not bounds_diagnostics["bounds_unit_suspect"]
        ):
            # The former raw-value WHERE bounds omitted groups containing only
            # invalid values. Preserve that output shape while retaining the
            # new pre-aggregation diagnostic counts.
            df = df.loc[df[value_column].notna()].reset_index(drop=True)
        # Downcast float64 value columns to float32 (skip ID/time columns)
        _skip_cols = {id_col, "charttime", "measuredat_minutes", "unit"}
        for c in df.columns:
            if c not in _skip_cols and df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        if bounds_diagnostics is not None:
            df.attrs["easyicu_bounds_loader"] = bounds_diagnostics
        logger.info(
            "Bucketed DuckDB aggregation complete: %s itemids=%d -> %d rows",
            table_name,
            len(itemids),
            len(df),
        )
        return df
    except Exception as e:
        logger.warning("DuckDB aggregation failed: %s", e)
        raise


def load_bucketed_table_multi_aggregated(
    data_source: "ICUDataSource",
    table_name: str,
    concept_itemids: Dict[str, List],
    *,
    value_column: str = "valuenum",
    interval_minutes: float = 60.0,
    patient_ids: Optional[List] = None,
    agg_func: str = "median",
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    itemid_col: Optional[str] = None,
    concept_bounds: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
) -> pd.DataFrame:
    """
    Batch-load multiple concepts from the same long table in one DuckDB scan.

    Supports all 6 ICU databases (MIIV, MIMIC, AUMC, HiRID, SIC, eICU).
    Each concept is pivoted via CASE WHEN + AGG, producing one column per concept.
    """
    import duckdb

    if not concept_itemids:
        return pd.DataFrame()

    db_name = data_source.config.name if hasattr(data_source.config, 'name') else 'unknown'
    _base_db = db_name.replace('_demo', '')
    base_path = data_source.base_path

    # ── DB-specific defaults ───────────────────────────────────
    _db_defaults = {
        'miiv':  {'id_col': 'stay_id',           'time_col': 'charttime',        'itemid_col': 'itemid'},
        'mimic': {'id_col': 'icustay_id',        'time_col': 'charttime',        'itemid_col': 'itemid'},
        'aumc':  {'id_col': 'admissionid',       'time_col': 'measuredat',       'itemid_col': 'itemid'},
        'hirid': {'id_col': 'patientid',         'time_col': 'datetime',         'itemid_col': 'variableid'},
        'sic':   {'id_col': 'CaseID',            'time_col': 'Offset',           'itemid_col': 'DataID'},
        'eicu':  {'id_col': 'patientunitstayid', 'time_col': 'labresultoffset',  'itemid_col': 'labname'},
    }
    defaults = _db_defaults.get(_base_db, {})
    if id_col is None:
        id_col = defaults.get('id_col', 'stay_id')
    if time_col is None:
        time_col = defaults.get('time_col', 'charttime')
    if itemid_col is None:
        itemid_col = defaults.get('itemid_col', 'itemid')

    # ── Detect if itemids are strings (e.g. eICU labname) ─────
    _sample_ids = next(iter(concept_itemids.values()))
    ids_are_strings = isinstance(_sample_ids[0], str) if _sample_ids else False

    # ── Collect all itemids ────────────────────────────────────
    if ids_are_strings:
        all_itemids = sorted({str(x) for ids in concept_itemids.values() for x in ids})
    else:
        all_itemids = sorted({int(x) for ids in concept_itemids.values() for x in ids})

    # ── Find parquet files ─────────────────────────────────────
    bucket_dir = data_source._resolve_bucket_directory(table_name)
    flat_parquet_dir = None
    if bucket_dir is None:
        flat_parquet_dir = data_source._resolve_flat_parquet_directory(table_name)
    if bucket_dir is None and flat_parquet_dir is None:
        raise ValueError(f"Cannot find bucketed directory for {table_name}")

    conn = _get_duckdb_connection()

    # ── Build glob pattern from bucket or flat dir ─────────
    if bucket_dir is not None:
        _, _, target_files = data_source._get_bucket_files_for_ids(
            bucket_dir,
            all_itemids,
            duckdb,
        )
        if not target_files:
            return pd.DataFrame()
        file_list_str = ", ".join(f"'{_duckdb_path(f)}'" for f in target_files)
        glob_pattern = f"[{file_list_str}]"
    else:
        target_files = tuple(sorted(flat_parquet_dir.glob("*.parquet")))
        glob_pattern = f"'{_duckdb_path(flat_parquet_dir)}/*.parquet'"

    # ── Aggregation ────────────────────────────────────────
    agg_map = {'median': 'MEDIAN', 'mean': 'AVG', 'max': 'MAX', 'min': 'MIN', 'first': 'FIRST', 'sum': 'SUM'}
    duckdb_agg = agg_map.get(agg_func, 'MEDIAN')

    _interval_hours = interval_minutes / 60.0
    _interval_seconds = interval_minutes * 60.0

    # ── Build CASE WHEN select expressions ─────────────────
    select_exprs = []
    for concept_name, ids in concept_itemids.items():
        if ids_are_strings:
            ids_sql = ", ".join(f"'{x}'" for x in ids)
        else:
            ids_sql = ", ".join(str(int(x)) for x in ids)
        bounds = (concept_bounds or {}).get(concept_name, {})
        value_conditions = [f"o.{itemid_col} IN ({ids_sql})"]
        if bounds.get("min") is not None:
            value_conditions.append(f"o.{value_column} >= {bounds['min']}")
        if bounds.get("max") is not None:
            value_conditions.append(f"o.{value_column} <= {bounds['max']}")
        case_when = " AND ".join(value_conditions)
        select_exprs.append(
            f"{duckdb_agg}(CASE WHEN {case_when} THEN o.{value_column} END) AS {concept_name}"
        )
    select_sql = ",\n            ".join(select_exprs)

    if ids_are_strings:
        all_ids_sql = ", ".join(f"'{x}'" for x in all_itemids)
    else:
        all_ids_sql = ", ".join(str(int(x)) for x in all_itemids)

    # ── DB-specific patient filter ─────────────────────────
    def _patient_clause(filter_col, values):
        if not values:
            return ""
        return f" AND o.{filter_col} IN ({', '.join(str(x) for x in values)})"

    non_null_clause = f" AND o.{value_column} IS NOT NULL"

    # ── DB-specific query construction ─────────────────────
    if _base_db in ('miiv', 'mimic'):
        stay_col = 'icustay_id' if _base_db == 'mimic' else 'stay_id'
        icustays_path = base_path / 'icustays.parquet'
        if not icustays_path.exists():
            icustays_path = base_path / 'icu' / 'icustays.parquet'
        if not icustays_path.exists() and (base_path / 'icustays.csv.gz').exists():
            icustays_path = base_path / 'icustays.csv.gz'
        _read_func = 'read_csv' if str(icustays_path).endswith('.csv.gz') else 'read_parquet'

        # hadm_id mapping for hospital tables (labevents etc.)
        raw_columns = data_source._get_parquet_columns_for_files(target_files)
        if not raw_columns:
            try:
                raw_columns = {col[0] for col in conn.execute(
                    f"SELECT * FROM read_parquet({glob_pattern}, union_by_name=true) LIMIT 0"
                ).description}
            except Exception:
                pass
        # 🔧 FIX 2026-05-11: MIMIC-III 1.4 早期转换的 parquet 列名为大写 (HADM_ID/ICUSTAY_ID)，
        # 而 Python 端 `in` 判断区分大小写，会错判 hospital_tables 是否需要 hadm_id rolling join。
        # DuckDB 本身 case-insensitive，所以 SQL 端无需改动，只需要这里小写化比较。
        raw_columns_lower = {str(c).lower() for c in raw_columns}
        patient_filter_col = id_col
        patient_filter_values = patient_ids
        join_left_col = id_col
        output_id_col = stay_col
        hospital_tables = {'prescriptions', 'labevents', 'microbiologyevents', 'emar', 'pharmacy', 'services'}
        _needs_hadm_join = (table_name in hospital_tables
                and id_col not in raw_columns_lower and 'hadm_id' in raw_columns_lower)
        if _needs_hadm_join:
            join_left_col = 'hadm_id'
            patient_filter_col = 'hadm_id'
            if patient_ids:
                try:
                    icustays_tbl = data_source.load_table(
                        'icustays', columns=[stay_col, 'hadm_id', 'intime', 'outtime'],
                        filters=[FilterSpec(column=stay_col, op=FilterOp.IN, value=patient_ids)],
                        verbose=False)
                    icustays_df = icustays_tbl.dataframe()
                    mapped = icustays_df['hadm_id'].dropna().unique().tolist() if 'hadm_id' in icustays_df.columns else []
                    if mapped:
                        patient_filter_values = mapped
                except Exception:
                    pass

        p_clause = _patient_clause(patient_filter_col, patient_filter_values)
        time_round_expr = f"FLOOR(EPOCH(o.{time_col} - CAST(a.intime AS TIMESTAMP)) / 3600.0 / {_interval_hours}) * {_interval_hours}"

        if join_left_col == 'hadm_id':
            # Hospital tables (labevents): need rolling join to assign events to correct ICU stay
            # Matches the single-concept load_bucketed_table_aggregated rolling JOIN approach

            # Outer time expression without table alias (for outer query on events_joined CTE)
            _outer_time_expr = f"FLOOR(EPOCH({time_col} - CAST(intime AS TIMESTAMP)) / 3600.0 / {_interval_hours}) * {_interval_hours}"
            # Outer select expressions without table alias
            _outer_select_exprs = []
            for concept_name, ids in concept_itemids.items():
                if ids_are_strings:
                    ids_sql = ", ".join(f"'{x}'" for x in ids)
                else:
                    ids_sql = ", ".join(str(int(x)) for x in ids)
                bounds = (concept_bounds or {}).get(concept_name, {})
                value_conditions = [f"{itemid_col} IN ({ids_sql})"]
                if bounds.get("min") is not None:
                    value_conditions.append(f"{value_column} >= {bounds['min']}")
                if bounds.get("max") is not None:
                    value_conditions.append(f"{value_column} <= {bounds['max']}")
                case_when = " AND ".join(value_conditions)
                _outer_select_exprs.append(
                    f"{duckdb_agg}(CASE WHEN {case_when} THEN {value_column} END) AS {concept_name}"
                )
            _outer_select_sql = ",\n                ".join(_outer_select_exprs)

            if patient_ids:
                _target_stay_ids_str = ", ".join(str(x) for x in patient_ids)
                query = f"""
                WITH target_hadms AS (
                    SELECT DISTINCT hadm_id
                    FROM {_read_func}('{_duckdb_path(icustays_path)}')
                    WHERE {stay_col} IN ({_target_stay_ids_str})
                ),
                all_stays AS (
                    SELECT {stay_col}, hadm_id, CAST(intime AS TIMESTAMP) as intime, CAST(outtime AS TIMESTAMP) as outtime
                    FROM {_read_func}('{_duckdb_path(icustays_path)}')
                    WHERE hadm_id IN (SELECT hadm_id FROM target_hadms)
                ),
                events_joined AS (
                    SELECT o.{value_column}, o.{time_col}, o.hadm_id, o.{itemid_col},
                           a.{stay_col}, a.intime,
                           ROW_NUMBER() OVER (
                               PARTITION BY o.hadm_id, o.{time_col}, o.{itemid_col}, o.{value_column}
                               ORDER BY
                                   CASE WHEN a.outtime >= o.{time_col} THEN a.outtime END ASC NULLS LAST,
                                   a.outtime DESC
                           ) as _rn
                    FROM read_parquet({glob_pattern}, union_by_name=true) o
                    JOIN all_stays a ON o.hadm_id = a.hadm_id
                    WHERE o.{itemid_col} IN ({all_ids_sql})
                    AND o.{value_column} IS NOT NULL
                    AND o.hadm_id IN (SELECT hadm_id FROM target_hadms)
                )
                SELECT {stay_col} AS {output_id_col},
                       {_outer_time_expr} AS charttime,
                       {_outer_select_sql}
                FROM events_joined
                WHERE _rn = 1 AND {stay_col} IN ({_target_stay_ids_str})
                GROUP BY {stay_col}, {_outer_time_expr}
                ORDER BY {stay_col}, 2
                """
            else:
                # ALL patients: no stay_id filter
                query = f"""
                WITH all_stays AS (
                    SELECT {stay_col}, hadm_id, CAST(intime AS TIMESTAMP) as intime, CAST(outtime AS TIMESTAMP) as outtime
                    FROM {_read_func}('{_duckdb_path(icustays_path)}')
                ),
                events_joined AS (
                    SELECT o.{value_column}, o.{time_col}, o.hadm_id, o.{itemid_col},
                           a.{stay_col}, a.intime,
                           ROW_NUMBER() OVER (
                               PARTITION BY o.hadm_id, o.{time_col}, o.{itemid_col}, o.{value_column}
                               ORDER BY
                                   CASE WHEN a.outtime >= o.{time_col} THEN a.outtime END ASC NULLS LAST,
                                   a.outtime DESC
                           ) as _rn
                    FROM read_parquet({glob_pattern}, union_by_name=true) o
                    JOIN all_stays a ON o.hadm_id = a.hadm_id
                    WHERE o.{itemid_col} IN ({all_ids_sql})
                    AND o.{value_column} IS NOT NULL
                )
                SELECT {stay_col} AS {output_id_col},
                       {_outer_time_expr} AS charttime,
                       {_outer_select_sql}
                FROM events_joined
                WHERE _rn = 1
                GROUP BY {stay_col}, {_outer_time_expr}
                ORDER BY {stay_col}, 2
                """
        else:
            # Direct join (stay_id in parquet, e.g. chartevents)
            join_pred = f"o.{join_left_col} = a.{join_left_col}"
            query = f"""
            WITH adm AS (
                SELECT {stay_col}, hadm_id, CAST(intime AS TIMESTAMP) as intime, CAST(outtime AS TIMESTAMP) as outtime
                FROM {_read_func}('{_duckdb_path(icustays_path)}')
            )
            SELECT a.{output_id_col} AS {output_id_col}, {time_round_expr} AS charttime,
                {select_sql}
            FROM read_parquet({glob_pattern}, union_by_name=true) o
            JOIN adm a ON {join_pred}
            WHERE o.{itemid_col} IN ({all_ids_sql}){p_clause}{non_null_clause}
            GROUP BY a.{output_id_col}, {time_round_expr}
            ORDER BY a.{output_id_col}, 2
            """

    elif _base_db == 'hirid':
        general_path = base_path / 'general_table.parquet'
        gen_read = 'read_parquet'
        if not general_path.exists():
            general_alt = base_path / 'general.parquet'
            if general_alt.exists():
                general_path = general_alt
            else:
                general_csv = base_path / 'general_table.csv'
                if general_csv.exists():
                    general_path = general_csv
                    gen_read = 'read_csv'
        p_clause = _patient_clause(id_col, patient_ids)
        time_round_expr = f"FLOOR(EPOCH(o.{time_col} - CAST(a.admissiontime AS TIMESTAMP)) / 3600.0 / {_interval_hours}) * {_interval_hours}"
        query = f"""
        WITH adm AS (
            SELECT patientid, CAST(admissiontime AS TIMESTAMP) as admissiontime
            FROM {gen_read}('{_duckdb_path(general_path)}')
        )
        SELECT o.{id_col} AS {id_col}, {time_round_expr} AS charttime,
            {select_sql}
        FROM read_parquet({glob_pattern}, union_by_name=true) o
        JOIN adm a ON o.{id_col} = a.patientid
        WHERE o.{itemid_col} IN ({all_ids_sql}){p_clause}{non_null_clause}
        GROUP BY o.{id_col}, {time_round_expr}
        ORDER BY o.{id_col}, 2
        """

    elif _base_db == 'aumc':
        p_clause = _patient_clause(id_col, patient_ids)
        admissions_relation = _aumc_admissions_relation_sql(data_source)
        relative_bucket_minutes = (
            f"FLOOR(((o.{time_col} - a.admittedat) / 60000.0) / "
            f"{interval_minutes}) * {interval_minutes}"
        )
        # Preserve the measuredat_minutes contract consumed by
        # ConceptResolver._align_time_to_admission while anchoring the bucket
        # itself to ICU admission rather than the database-wide clock.
        time_round_expr = (
            f"(a.admittedat / 60000.0 + {relative_bucket_minutes})"
        )
        query = f"""
        SELECT o.{id_col} AS {id_col}, {time_round_expr} AS measuredat_minutes,
            {select_sql}
        FROM read_parquet({glob_pattern}, union_by_name=true) o
        JOIN {admissions_relation} a ON o.{id_col} = a.admissionid
        WHERE o.{itemid_col} IN ({all_ids_sql}){p_clause}{non_null_clause}
            AND a.admittedat IS NOT NULL
        GROUP BY o.{id_col}, {time_round_expr}
        ORDER BY o.{id_col}, 2
        """

    elif _base_db == 'sic':
        p_clause = _patient_clause(id_col, patient_ids)
        _time_ref = f'o."{time_col}"' if time_col == 'Offset' else f'o.{time_col}'
        cases_relation = _sic_cases_relation_sql(data_source)
        time_round_expr = (
            f"FLOOR(({_time_ref} - c.ICUOffset) / {_interval_seconds})"
        )
        query = f"""
        SELECT o.{id_col} AS {id_col}, {time_round_expr} AS charttime,
            {select_sql}
        FROM read_parquet({glob_pattern}, union_by_name=true) o
        JOIN {cases_relation} c ON o.{id_col} = c.CaseID
        WHERE o.{itemid_col} IN ({all_ids_sql}){p_clause}{non_null_clause}
            AND c.ICUOffset IS NOT NULL
        GROUP BY o.{id_col}, {time_round_expr}
        ORDER BY o.{id_col}, 2
        """

    elif _base_db == 'eicu':
        p_clause = _patient_clause(id_col, patient_ids)
        # Output hours (not minutes) — consistent with load_wide_table_aggregated
        # Batch results bypass _align_time_to_admission, so must output hours directly
        time_round_expr = f"FLOOR(o.{time_col} / {interval_minutes})"
        query = f"""
        SELECT o.{id_col} AS {id_col}, {time_round_expr} AS charttime,
            {select_sql}
        FROM read_parquet({glob_pattern}, union_by_name=true) o
        WHERE o.{itemid_col} IN ({all_ids_sql}){p_clause}{non_null_clause}
        GROUP BY o.{id_col}, {time_round_expr}
        ORDER BY o.{id_col}, 2
        """

    else:
        raise ValueError(f"Unsupported database for multi-concept batch: {db_name}")

    try:
        arrow_table = conn.execute(query).fetch_arrow_table()
        df = _arrow_to_pandas_compat(arrow_table, split_blocks=True)
        del arrow_table
    except Exception:
        df = conn.execute(query).fetchdf()
    logger.info(
        "Bucketed multi-concept DuckDB aggregation complete: %s concepts=%d itemids=%d -> %d rows",
        table_name,
        len(concept_itemids),
        len(all_itemids),
        len(df),
    )
    # Downcast float64 value columns to float32 (skip ID/time columns)
    _skip_cols = {id_col, 'charttime', 'measuredat_minutes'}
    for c in df.columns:
        if c not in _skip_cols and df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def _resolve_wide_column_batch_size(value_count: int) -> int:
    """Choose how many wide-table median states to aggregate per query."""
    import os

    if value_count < 1:
        return 1
    raw_limit = os.environ.get("EASYICU_WIDE_COLUMN_BATCH_SIZE")
    if raw_limit:
        try:
            return min(value_count, max(1, int(raw_limit)))
        except ValueError:
            logger.warning(
                "Ignoring invalid EASYICU_WIDE_COLUMN_BATCH_SIZE=%r",
                raw_limit,
            )
    try:
        from .runtime.parallel_config import get_global_config

        if get_global_config().total_memory_gb <= 16:
            return min(value_count, 2)
    except Exception:
        pass
    return value_count


def _release_wide_chunk_memory() -> None:
    """Release temporary query buffers between constrained wide-column chunks."""
    import gc
    import sys

    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim(0)
        except Exception:
            pass


def load_wide_table_aggregated(
    data_source: "ICUDataSource",
    table_name: str,
    value_columns: List[str],
    interval_hours: float = 1.0,
    patient_ids: Optional[List] = None,
    agg_func: str = 'median',  # 'median' (default, matches R ricu), 'first', 'mean', 'max', 'min'
    column_bounds: Optional[dict] = None,  # {col_name: (min_val, max_val)} for pre-aggregation filtering
) -> pd.DataFrame:
    """
    🚀 高性能宽表批量加载：在DuckDB中完成聚合和去重

    针对eICU vitalperiodic等宽表优化，一次加载多个概念列，
    直接在DuckDB中完成小时聚合，避免pandas后处理开销。

    Args:
        data_source: ICU数据源
        table_name: 表名（如'vitalperiodic'）
        value_columns: 需要加载的值列列表（如['heartrate', 'respiration']）
        interval_hours: 时间聚合间隔（小时）
        patient_ids: 可选的患者ID过滤
        agg_func: 聚合函数（'first', 'mean', 'max', 'min'）

    Returns:
        聚合后的DataFrame，包含id列、时间列和所有值列

    Example:
        >>> df = load_wide_table_aggregated(
        ...     data_source, 'vitalperiodic',
        ...     ['heartrate', 'respiration', 'sao2'],
        ...     interval_hours=1.0
        ... )
        >>> # 返回: patientunitstayid | charttime | heartrate | respiration | sao2
    """
    import duckdb
    
    # 获取表配置
    if patient_ids is not None and len(patient_ids) == 0:
        return pd.DataFrame()
    table_cfg = data_source.config.get_table(table_name)
    
    # 确定ID列和时间列
    id_col = table_cfg.defaults.id_var
    if not id_col:
        icustay_cfg = data_source.config.id_configs.get('icustay')
        id_col = icustay_cfg.id if icustay_cfg else 'patientunitstayid'
    
    time_col = table_cfg.defaults.index_var or 'observationoffset'

    # DuckDB's MEDIAN keeps one aggregate state per output group and value
    # column.  On a full eICU batch, six simultaneous vital columns exceeded
    # 16 GiB even with a 2 GB DuckDB memory_limit because complex aggregate
    # state and result buffers are not all governed by that setting.  Keep the
    # same patient batch, but scan a bounded number of value columns per query
    # and reassemble their identical key grids afterwards.
    column_batch_size = _resolve_wide_column_batch_size(len(value_columns))
    if len(value_columns) > column_batch_size:
        combined = None
        key_columns = [id_col, "charttime"]
        for start in range(0, len(value_columns), column_batch_size):
            column_chunk = value_columns[start : start + column_batch_size]
            chunk = load_wide_table_aggregated(
                data_source,
                table_name,
                column_chunk,
                interval_hours=interval_hours,
                patient_ids=patient_ids,
                agg_func=agg_func,
                column_bounds=column_bounds,
            )
            chunk = chunk.sort_values(
                key_columns,
                kind="mergesort",
                ignore_index=True,
            )
            if combined is None:
                combined = chunk
                continue
            if not combined[key_columns].equals(chunk[key_columns]):
                raise ValueError(
                    "wide-table column chunks produced different key grids"
                )
            for column in column_chunk:
                if column in chunk.columns:
                    combined[column] = chunk[column].to_numpy(copy=False)
            del chunk
            _release_wide_chunk_memory()
        return combined if combined is not None else pd.DataFrame()
    
    # 确定数据目录
    table_path = data_source._resolve_loader_from_disk(table_name)
    if table_path is None:
        raise ValueError(f"Cannot find data for table {table_name}")
    
    # table_path 返回 Path 对象
    directory = table_path if isinstance(table_path, Path) else Path(table_path)
    
    # 构建glob pattern
    if directory.is_dir():
        glob_pattern = _duckdb_path(directory / "*.parquet")
    else:
        glob_pattern = _duckdb_path(directory)
    
    # 构建DuckDB聚合函数映射 (median为R ricu默认)
    agg_map = {
        'median': 'MEDIAN',  # R ricu default
        'first': 'FIRST',
        'mean': 'AVG',
        'max': 'MAX', 
        'min': 'MIN',
    }
    duckdb_agg = agg_map.get(agg_func, 'MEDIAN')
    
    # 🚀 perf: register patient_ids as a DuckDB view rather than embedding
    # tens of thousands of literals into a SQL IN list (which inflates plan
    # time and breaks the parquet predicate pushdown path).
    conn = _get_duckdb_connection()
    where_clause = ""
    _wide_pid_view = None
    if patient_ids:
        _wide_pid_view = "_easyicu_wide_pids"
        conn.register(_wide_pid_view, pd.DataFrame({id_col: list(patient_ids)}))
        where_clause = f"WHERE {id_col} IN (SELECT {id_col} FROM {_wide_pid_view})"

    # 🚀 优化：单个 GROUP BY 替代 per-column CTE + FULL OUTER JOIN
    # MEDIAN() 自动忽略 NULL，不需要 WHERE col IS NOT NULL 分支
    # 单次扫描，单次 GROUP BY，无 FULL OUTER JOIN 中间结果膨胀
    # 🔧 FIX: 使用 CASE WHEN 实现 per-column filter_bounds（R ricu pre-aggregation filtering）
    # 将超出 [min, max] 范围的值替换为 NULL，这样 MEDIAN 会忽略它们
    if column_bounds:
        agg_parts = []
        for col in value_columns:
            bounds = column_bounds.get(col)
            if bounds:
                vmin, vmax = bounds
                conditions = []
                if vmin is not None:
                    conditions.append(f"{col} >= {vmin}")
                if vmax is not None:
                    conditions.append(f"{col} <= {vmax}")
                filter_expr = " AND ".join(conditions)
                agg_parts.append(f"{duckdb_agg}(CASE WHEN {filter_expr} THEN {col} END) as {col}")
            else:
                agg_parts.append(f"{duckdb_agg}({col}) as {col}")
        agg_exprs = ", ".join(agg_parts)
    else:
        agg_exprs = ", ".join(f"{duckdb_agg}({col}) as {col}" for col in value_columns)

    # 🚀 perf B1: same-directory bucket parquets share schema (the bucket
    # converter writes them in one pass). Drop union_by_name=true so DuckDB
    # does not pre-sweep every file's metadata for schema reconciliation —
    # on slow filesystems this dominated wall-clock. Fall back to
    # union_by_name=true if the fast path raises a schema-mismatch error.
    # 🚀 perf B5: drop the trailing ORDER BY — downstream groupby/merge
    # does not depend on it, and sorting tens of millions of rows after
    # aggregation was pure overhead.
    _wide_query_tpl = (
        "SELECT\n"
        f"    {id_col},\n"
        f"    FLOOR({time_col} / {interval_hours * 60.0}) as charttime,\n"
        f"    {agg_exprs}\n"
        "FROM {read_expr}\n"
        f"{where_clause}\n"
        f"GROUP BY {id_col}, FLOOR({time_col} / {interval_hours * 60.0})\n"
    )
    _wide_read_fast = f"read_parquet('{glob_pattern}')"
    _wide_read_safe = f"read_parquet('{glob_pattern}', union_by_name=true)"

    def _exec_wide(read_expr: str) -> pd.DataFrame:
        q = _wide_query_tpl.format(read_expr=read_expr)
        try:
            arrow_table = conn.execute(q).fetch_arrow_table()
            df_ = _arrow_to_pandas_compat(arrow_table, split_blocks=True)
            del arrow_table
            return df_
        except Exception:
            return conn.execute(q).fetchdf()

    try:
        df = _exec_wide(_wide_read_fast)
    except Exception as _wide_exc:
        # Schema actually differs across files (rare) → safe variant.
        logger.warning(
            "wide_table fast path failed (%s); falling back to union_by_name=true",
            _wide_exc,
        )
        df = _exec_wide(_wide_read_safe)
    finally:
        if _wide_pid_view is not None:
            try:
                conn.unregister(_wide_pid_view)
            except Exception:
                pass
    # Downcast float64 value columns to float32 (skip ID/time columns)
    _skip_cols = {id_col, 'charttime'}
    for c in df.columns:
        if c not in _skip_cols and df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    logger.info("Wide-table batch load complete: %s %s -> %d rows", table_name, value_columns, len(df))
    return df
