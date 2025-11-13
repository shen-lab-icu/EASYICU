"""Data loading utilities for ICU datasets."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Optional

import pandas as pd

from .config import DataSourceConfig, DataSourceRegistry
from .table import ICUTable

# 全局调试开关 - 设置为 False 可以减少输出
DEBUG_MODE = False


class FilterOp(str, enum.Enum):
    """Supported filter operations for table loading."""

    EQ = "=="
    IN = "in"
    BETWEEN = "between"


@dataclass
class FilterSpec:
    """Declarative filter specification for table loading."""

    column: str
    op: FilterOp
    value: Any

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.op == FilterOp.EQ:
            mask = frame[self.column] == self.value
            return frame.loc[mask].copy()
        if self.op == FilterOp.IN:
            if isinstance(self.value, str):
                candidate = [self.value]
            else:
                candidate = list(self.value)
            mask = frame[self.column].isin(candidate)
            return frame.loc[mask].copy()
        if self.op == FilterOp.BETWEEN:
            lower, upper = self.value
            mask = frame[self.column].between(lower, upper)
            return frame.loc[mask].copy()
        raise ValueError(f"Unsupported filter operation: {self.op}")


class ICUDataSource:
    """Lightweight facade that loads tables for a concrete dataset instance."""

    # 全局格式优先级配置
    _global_format_priority: Optional[List[str]] = None

    @classmethod
    def set_format_priority(cls, priority: List[str]) -> None:
        """设置全局文件格式优先级
        
        Args:
            priority: 格式列表，按优先级排序，例如 ['parquet', 'fst', 'csv']
        
        Examples:
            >>> # 优先使用 Parquet（纯 Python，无需 R）
            >>> ICUDataSource.set_format_priority(['parquet', 'fst', 'csv'])
            >>> 
            >>> # 只使用 Parquet（跳过 FST）
            >>> ICUDataSource.set_format_priority(['parquet', 'csv'])
            >>> 
            >>> # 优先 FST（旧行为，需要 R 环境）
            >>> ICUDataSource.set_format_priority(['fst', 'parquet', 'csv'])
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
        env_priority = os.environ.get('PYRICU_FORMAT_PRIORITY')
        if env_priority:
            return [fmt.strip() for fmt in env_priority.split(',')]
        
        # 3. 默认优先级：Parquet > FST > CSV
        # Parquet 优先因为：纯 Python，无需 R，列式存储，压缩好
        return ['parquet', 'fst', 'csv']

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
        self.enable_cache = enable_cache
        self._table_cache: dict = {}  # 缓存已加载的原始表数据
        self.format_priority = format_priority or self.get_format_priority()

    def register_table_source(self, table: str, source: Any) -> None:
        """Register a callable/file path used to load ``table``."""
        self._table_sources[table] = source
    
    def clear_cache(self) -> None:
        """清除表缓存,释放内存。"""
        self._table_cache.clear()
    
    def get_cache_info(self) -> dict:
        """获取缓存信息。"""
        total_size = sum(df.memory_usage(deep=True).sum() for df in self._table_cache.values())
        return {
            'cached_tables': len(self._table_cache),
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
        
        # 提取 patient_ids 过滤器用于分区预过滤
        patient_ids_filter = None
        if filters:
            for spec in filters:
                # 支持各数据库的ID列名
                id_columns = ['subject_id', 'icustay_id', 'hadm_id', 'stay_id',  # MIMIC
                             'admissionid', 'patientid',  # AUMC
                             'patientunitstayid',  # eICU
                             'patientid']  # HiRID
                if spec.op == FilterOp.IN and spec.column in id_columns:
                    patient_ids_filter = spec
                    # 只在verbose模式下输出，且只输出一次
                    if verbose:
                        cache_key = f"_filter_logged_{table_name}"
                        if not hasattr(self, cache_key) or not getattr(self, cache_key, False):
                            if DEBUG_MODE: print(f"   🎯 检测到患者ID过滤器: {len(spec.value)} 个患者, 列={spec.column}")
                            setattr(self, cache_key, True)
                    break
        
        frame = self._load_raw_frame(table_name, columns, patient_ids_filter=patient_ids_filter)
        
        if filters:
            for spec in filters:
                frame = spec.apply(frame)
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
        
        # 🔧 AUMC特殊处理：时间列是毫秒,需要转换为分钟 (参考R ricu的ms_as_mins)
        # R ricu: ms_as_mins <- function(x) min_as_mins(as.integer(x / 6e4))
        # 这样处理后,AUMC的时间单位与其他数据库一致(都是分钟)
        if self.config.name == 'aumc':
            for column in time_like_cols:
                if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
                    # 将毫秒转换为分钟: ms / 60000
                    frame[column] = (frame[column] / 60000.0).astype('float64')
        
        for column in time_like_cols:
            # 只有当列存在且不是numeric类型时才转换
            # 如果已经是numeric，可能是已经对齐过的小时数
            if column in frame.columns:
                frame[column] = _coerce_datetime(frame[column])

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
    ) -> pd.DataFrame:
        # 缓存键：表名 + 列集合 + 患者过滤器
        # 对于有患者过滤器的情况,也使用缓存(因为同一批患者会被多个概念使用)
        if patient_ids_filter:
            # 使用frozenset来确保患者ID列表的哈希一致性
            patient_ids_set = frozenset(patient_ids_filter.value) if not isinstance(patient_ids_filter.value, str) else frozenset([patient_ids_filter.value])
            cache_key = (table_name, tuple(sorted(columns)) if columns else None, patient_ids_filter.column, patient_ids_set)
        else:
            cache_key = (table_name, tuple(sorted(columns)) if columns else None, None, None)
        
        # 检查缓存
        if self.enable_cache and cache_key in self._table_cache:
            return self._table_cache[cache_key].copy()
        
        loader = self._table_sources.get(table_name)
        if loader is None:
            loader = self._resolve_loader_from_disk(table_name)
        if loader is None:
            # 对于miiv数据源，如果表在配置中定义了但文件不存在，返回空DataFrame
            # 这允许在demo数据中缺少某些表时继续运行
            if self.config.name == 'miiv' and table_name in self.config.tables:
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
            frame = self._read_file(Path(loader), columns, patient_ids_filter=patient_ids_filter)

        if columns is not None:
            missing = set(columns) - set(frame.columns)
            if missing:
                raise KeyError(
                    f"Columns {sorted(missing)} not found in table '{table_name}'"
                )
            frame = frame[list(columns)]
        
        # 缓存加载的数据（使用之前构建的cache_key）
        if self.enable_cache:
            self._table_cache[cache_key] = frame.copy()
        
        return frame

    def _resolve_loader_from_disk(self, table_name: str) -> Optional[Callable[[], pd.DataFrame] | Path]:
        if not self.base_path:
            return None
        
        table_cfg = self.config.get_table(table_name)
        explicit = table_cfg.first_file()
        if explicit:
            explicit_path = self.base_path / explicit
            # Only use explicit path if it actually exists
            if explicit_path.exists():
                return explicit_path
        
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
        
        # Try different formats in order of preference
        # 1. FST (R ricu format) - highest priority for existing ricu data
        # Try both original case and lowercase
        for name in [file_base_name, file_base_name.lower()]:
            fst_candidate = self.base_path / f"{name}.fst"
            if fst_candidate.exists():
                return fst_candidate
        
        # 2. Parquet (Python default)
        for name in [file_base_name, file_base_name.lower(), table_name, table_name.lower()]:
            candidate = self.base_path / f"{name}.{self.default_format}"
            if candidate.exists():
                return candidate
        
        # 3. CSV (fallback)
        if self.base_path is not None:
            for name in [table_name, table_name.lower()]:
                csv_candidate = self.base_path / f"{name}.csv"
                if csv_candidate.exists():
                    return csv_candidate
                # Also try .csv.gz
                csv_gz_candidate = self.base_path / f"{name}.csv.gz"
                if csv_gz_candidate.exists():
                    return csv_gz_candidate
        
        # 4. Check subdirectory for partitioned data (common in hirid observations)
        if self.base_path is not None:
            for name in [table_name, table_name.lower()]:
                subdir = self.base_path / name
            if subdir.is_dir():
                # Look for FST files first
                fst_files = list(subdir.glob("*.fst"))
                if fst_files:
                    return subdir  # Return directory, will handle in _read_file
                # Then Parquet
                parquet_files = list(subdir.glob("*.parquet")) + list(subdir.glob("*.pq"))
                if parquet_files:
                    return subdir
        
        return None

    def _read_file(self, path: Path, columns: Optional[Iterable[str]], patient_ids_filter: Optional[FilterSpec] = None) -> pd.DataFrame:
        # Handle directory (partitioned data)
        if path.is_dir():
            return self._read_partitioned_data(path, columns, patient_ids_filter=patient_ids_filter)
        
        suffix = path.suffix.lower()
        
        # Handle .csv.gz files (compressed CSV)
        if str(path).endswith('.csv.gz') or str(path).endswith('.CSV.GZ'):
            return pd.read_csv(path, compression='gzip', usecols=list(columns) if columns else None)
        
        # Handle regular formats
        if suffix == ".csv":
            return pd.read_csv(path, usecols=list(columns) if columns else None)
        if suffix == ".gz":
            # Try to read as compressed CSV
            return pd.read_csv(path, compression='gzip', usecols=list(columns) if columns else None)
        if suffix in {".parquet", ".pq"}:
            # 🚀 使用PyArrow过滤器优化大文件读取
            if patient_ids_filter:
                try:
                    import pyarrow.parquet as pq
                    import pyarrow as pa
                    # 使用 DNF (Disjunctive Normal Form) 格式，兼容性更好
                    target_ids = patient_ids_filter.value if isinstance(patient_ids_filter.value, list) else [patient_ids_filter.value]
                    
                    # 使用PyArrow读取并过滤 - 使用 DNF 格式
                    df = pq.read_table(
                        path,
                        columns=list(columns) if columns else None,
                        filters=[[( patient_ids_filter.column, 'in', target_ids)]]
                    ).to_pandas()
                except (ImportError, Exception) as e:
                    # 如果PyArrow过滤失败，回退到pandas后过滤
                    df = pd.read_parquet(path, columns=list(columns) if columns else None)
                    if patient_ids_filter.column in df.columns:
                        target_ids = set(patient_ids_filter.value) if isinstance(patient_ids_filter.value, list) else {patient_ids_filter.value}
                        df = df[df[patient_ids_filter.column].isin(target_ids)]
            else:
                df = pd.read_parquet(path, columns=list(columns) if columns else None)
            
            # 处理重复列名（如果存在）
            if df.columns.duplicated().any():
                import pandas.io.common
                df.columns = pandas.io.common.dedup_names(df.columns, is_potential_multiindex=False)
            return df
        if suffix == ".feather":
            return pd.read_feather(path, columns=list(columns) if columns else None)
        if suffix == ".fst":
            return self._read_fst_file(path, columns)
        
        raise ValueError(f"Unsupported file format for table loading: {path.suffix}")
    
    def _read_partitioned_data(self, directory: Path, columns: Optional[Iterable[str]], patient_ids_filter: Optional[FilterSpec] = None) -> pd.DataFrame:
        """Read partitioned data from a directory, respecting format priority."""
        
        # 按优先级顺序查找文件
        format_map = {
            'parquet': lambda: sorted(directory.glob("*.parquet")) + sorted(directory.glob("*.pq")),
            'fst': lambda: sorted(directory.glob("*.fst")),
            'csv': lambda: sorted(directory.glob("*.csv")) + sorted(directory.glob("*.csv.gz")),
        }
        
        # 尝试按优先级读取
        for fmt in self.format_priority:
            if fmt not in format_map:
                continue
                
            files = format_map[fmt]()
            if not files:
                continue
            
            # 找到文件，根据格式读取
            num_files = len(files)
            
            # 准备患者ID过滤器 (支持多种数据库的ID列)
            filter_tuple = None
            if patient_ids_filter and patient_ids_filter.column in ['subject_id', 'hadm_id', 'icustay_id', 'stay_id', 'admissionid', 'patientid']:
                target_ids = set(patient_ids_filter.value) if not isinstance(patient_ids_filter.value, str) else {patient_ids_filter.value}
                filter_tuple = (patient_ids_filter.column, target_ids)
                if DEBUG_MODE: print(f"   📁 加载 {directory.name} ({num_files} 个 {fmt} 分区) - 过滤 {len(target_ids)} 个患者...")
            else:
                if DEBUG_MODE: print(f"   📁 加载 {directory.name} ({num_files} 个 {fmt} 分区)...")
            
            if fmt == 'fst':
                # FST 特殊处理：支持并行读取
                if num_files > 3:
                    try:
                        from .fst_reader_fast import read_fst_parallel
                        return read_fst_parallel(
                            files, 
                            columns=list(columns) if columns else None, 
                            verbose=True,
                            patient_ids_filter=filter_tuple
                        )
                    except Exception:
                        pass  # Fallback to sequential
                # Sequential FST reading
                dfs = [self._read_fst_file(f, columns) for f in files]
                
            elif fmt == 'parquet':
                # Parquet 读取（支持列选择）
                dfs = []
                for f in files:
                    df = pd.read_parquet(f, columns=list(columns) if columns else None)
                    # 如果有患者过滤器，应用过滤
                    if filter_tuple:
                        col_name, target_ids = filter_tuple
                        if col_name in df.columns:
                            df = df[df[col_name].isin(target_ids)]
                    dfs.append(df)
                    
            elif fmt == 'csv':
                # CSV 读取
                dfs = []
                for f in files:
                    compression = 'gzip' if str(f).endswith('.gz') else None
                    df = pd.read_csv(f, usecols=list(columns) if columns else None, compression=compression)
                    # 如果有患者过滤器，应用过滤
                    if filter_tuple:
                        col_name, target_ids = filter_tuple
                        if col_name in df.columns:
                            df = df[df[col_name].isin(target_ids)]
                    dfs.append(df)
            
            # 合并所有分区
            if dfs:
                return pd.concat(dfs, ignore_index=True)
        
        # 没有找到任何支持的文件
        tried_formats = ', '.join(self.format_priority)
        raise ValueError(f"No supported data files found in directory: {directory} (tried: {tried_formats})")
    
    def _read_fst_file(self, path: Path, columns: Optional[Iterable[str]]) -> pd.DataFrame:
        """Read an FST file using the fst_reader module."""
        try:
            # First try the fast reader (uses R fst package directly)
            try:
                from .fst_reader_fast import read_fst_fast
                df = read_fst_fast(path, columns=list(columns) if columns else None)
                return df
            except Exception as e:
                # Fallback to regular fst_reader if fast reader fails
                from .fst_reader import read_fst
                df = read_fst(path)
                if columns is not None:
                    missing = set(columns) - set(df.columns)
                    if missing:
                        raise KeyError(f"Columns {sorted(missing)} not found in FST file '{path}'")
                    df = df[list(columns)]
                return df
        except ImportError:
            raise ImportError(
                "FST file support requires either:\n"
                "  1. Python fst package: pip install fst\n"
                "  2. R with fst package installed (recommended for fst_reader_fast)\n"
                f"Cannot read: {path}"
            )


def load_table(
    data_source: ICUDataSource,
    table_name: str,
    *,
    columns: Optional[Iterable[str]] = None,
    filters: Optional[Iterable[FilterSpec]] = None,
) -> ICUTable:
    """Functional façade delegating to :meth:`ICUDataSource.load_table`."""

    return data_source.load_table(table_name, columns=columns, filters=filters)


def _coerce_datetime(series: pd.Series) -> pd.Series:
    """Coerce a Series to datetime type, handling various edge cases."""
    if pd.api.types.is_datetime64_any_dtype(series):
        # 如果已经是datetime，移除时区信息以避免后续时区不一致错误
        if hasattr(series.dt, 'tz') and series.dt.tz is not None:
            return series.dt.tz_localize(None)
        return series
    
    # 如果已经是numeric类型，不要转换！
    # 这可能是已经对齐到入院时间的小时数
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    # Reset index if it has duplicates (which can cause "cannot assemble with duplicate keys")
    if series.index.duplicated().any():
        series = series.reset_index(drop=True)
    
    try:
        # Try direct conversion first, with UTC then remove timezone
        converted = pd.to_datetime(series, errors="raise", utc=True).dt.tz_localize(None)
        return converted
    except (TypeError, ValueError) as e:
        # If raise fails, try with coerce
        try:
            converted = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)
            return converted
        except (TypeError, ValueError):
            # If all else fails, return original series
            # This handles cases where conversion is not possible
            return series
