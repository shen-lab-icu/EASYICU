"""Data loading utilities for ICU datasets."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence
from threading import RLock

import pandas as pd

from .config import DataSourceConfig, DataSourceRegistry, DatasetOptions, TableConfig
from .table import ICUTable

# 全局调试开关 - 设置为 False 可以减少输出
DEBUG_MODE = False
logger = logging.getLogger(__name__)

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
        env_priority = os.environ.get('PYRICU_FORMAT_PRIORITY')
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
        hospital_tables = ['prescriptions', 'labevents', 'microbiologyevents', 'emar', 'pharmacy']
        original_stay_ids = None
        if table_name in hospital_tables and self.config.name in ['miiv', 'mimic_demo']:
            if filters:
                for spec in filters:
                    if spec.column == 'stay_id' and spec.op == FilterOp.IN:
                        original_stay_ids = set(spec.value)  # 保存原始目标 stay_ids
                        print(f"💾 [{table_name}] 保存原始 stay_id 过滤器: {len(original_stay_ids)} 个患者")
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
            # 🚀 优化2：使用最小列集 + 传入的额外列（如 value_var）
            # 只对在 MINIMAL_COLUMNS_MAP 中定义的表应用列优化
            # 对于其他表（如 AUMC numericitems），加载所有列以确保包含必要的 ID/时间/值列
            from .load_concepts import MINIMAL_COLUMNS_MAP, USE_MINIMAL_COLUMNS
            if USE_MINIMAL_COLUMNS and table_name in MINIMAL_COLUMNS_MAP:
                base_columns = list(MINIMAL_COLUMNS_MAP[table_name])
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
            
            # 🚀 优化：对于缺少 stay_id 的表（如 labevents），如果过滤条件是 stay_id，
            # 需要先查 icustays 转换成 hadm_id 或 subject_id，以便在读取 parquet 时就能过滤
            hospital_tables = ['prescriptions', 'labevents', 'microbiologyevents', 'emar', 'pharmacy']
            mapped_filter = None
            
            if filters:
                for spec in filters:
                    # 支持各数据库的ID列名
                    id_columns = ['subject_id', 'icustay_id', 'hadm_id', 'stay_id',  # MIMIC
                                 'admissionid', 'patientid',  # AUMC
                                 'patientunitstayid',  # eICU
                                 'patientid']  # HiRID
                    
                    if spec.op == FilterOp.IN and spec.column in id_columns:
                        patient_ids_filter = spec
                        
                        # 特殊处理：如果表是 hospital table 且过滤器是 stay_id
                        if table_name in hospital_tables and self.config.name in ['miiv', 'mimic_demo'] and spec.column == 'stay_id':
                            try:
                                if verbose:
                                    logger.info(f"🔄 [{table_name}] 将 stay_id 过滤器转换为 hadm_id 以优化读取...")
                                
                                # 加载 icustays 获取映射
                                icustays_map = self.load_table(
                                    'icustays', 
                                    columns=['stay_id', 'hadm_id'], 
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
        
        # AUMC特殊处理：时间列是毫秒,需要转换为分钟 (参考R ricu的ms_as_mins)
        # R ricu: ms_as_mins <- function(x) min_as_mins(as.integer(x / 6e4))
        # 关键: as.integer() 会 floor 到整数分钟!
        # 这样处理后,AUMC的时间单位与其他数据库一致(都是分钟)
        if self.config.name == 'aumc':
            for column in time_like_cols:
                if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
                    # 将毫秒转换为整数分钟: floor(ms / 60000) - 匹配 R ricu 的 as.integer()
                    frame[column] = (frame[column] / 60000.0).apply(lambda x: int(x) if pd.notna(x) else x).astype('float64')
        
        for column in time_like_cols:
            # 只有当列存在且不是numeric类型时才转换
            # 如果已经是numeric，可能是已经对齐过的小时数
            if column in frame.columns:
                frame[column] = _coerce_datetime(frame[column])

        # 自动补全 stay_id：某些表（如 prescriptions, labevents）只有 hadm_id，需要 JOIN icustays
        # 这对于使用这些表的概念（如 delirium_tx）至关重要
        if ('stay_id' not in frame.columns or frame['stay_id'].isna().all()) and 'hadm_id' in frame.columns:
            # ⚠️ 问题：对于 hospital tables (如 labevents), 原表没有 stay_id，需要通过 hadm_id join icustays 补全
            # 但 join 会引入该 hadm_id 的所有 stay_id (同一住院可能多次ICU入住)
            # 解决方案：在函数开始时已保存 original_stay_ids，join 后再过滤
            hospital_tables = ['prescriptions', 'labevents', 'microbiologyevents', 'emar', 'pharmacy']
            if table_name in hospital_tables and self.config.name in ['miiv', 'mimic_demo']:
                try:
                    # 🔍 提取当前的患者ID过滤器（stay_id 或 subject_id）
                    # 这样 icustays 只加载我们需要的患者，避免 join 时产生额外的匹配
                    icustays_filters = []
                    if filters:
                        for spec in filters:
                            # stay_id 或 subject_id 过滤器都可以用于过滤 icustays
                            if spec.column in ['stay_id', 'subject_id'] and spec.op == FilterOp.IN:
                                icustays_filters.append(spec)
                                if verbose:
                                    logger.debug(f"[{table_name}] 提取患者ID过滤器: {spec.column} IN ({len(spec.value)} 个值)")
                                # 不要 break，可能有多个过滤器
                    
                    # 加载 icustays 映射（需要 hadm_id, stay_id, subject_id）
                    # 如果有患者ID过滤器，传递给 icustays 以避免加载全表
                    if verbose:
                        logger.debug(f"[{table_name}] 加载 icustays，filters={len(icustays_filters)}个")
                    icustays_map = self.load_table(
                        'icustays', 
                        columns=['hadm_id', 'stay_id', 'subject_id', 'intime', 'outtime'],  # 需要 intime 和 outtime 用于 rolling join
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
                            columns=['hadm_id', 'stay_id', 'subject_id', 'intime', 'outtime'],
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
                    
                    # JOIN 补全 stay_id（包含 intime 和 outtime 用于 rolling join）
                    # 注意：同一 hadm_id 可能对应多个 stay_id（多次 ICU 入住）
                    frame = frame.merge(
                        icustays_df[['hadm_id', 'stay_id', 'intime', 'outtime']],
                        on='hadm_id',
                        how='inner',  # 只保留有 ICU 住院的记录
                        suffixes=('', '_icu')
                    )
                    
                    # 清理可能的重复列
                    if 'stay_id_icu' in frame.columns:
                        # 如果原来有 stay_id 列但是全 NaN，用新的替换
                        if 'stay_id' not in frame.columns or frame['stay_id'].isna().all():
                            frame['stay_id'] = frame['stay_id_icu']
                        frame = frame.drop(columns=['stay_id_icu'], errors='ignore')
                    
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
                    
                    if time_col and 'stay_id' in frame.columns and 'outtime' in frame.columns:
                        # 检查是否有同一 hadm_id 下的多个 stay_id
                        stays_per_hadm = frame.groupby('hadm_id')['stay_id'].nunique()
                        multi_stay_hadms = stays_per_hadm[stays_per_hadm > 1].index.tolist()
                        
                        if multi_stay_hadms:
                            if verbose:
                                logger.debug(f"[{table_name}] 检测到 {len(multi_stay_hadms)} 个 hadm_id 有多个 stay_id，执行 rolling join (使用 outtime)")
                            
                            # 规范化时间列
                            frame[time_col] = pd.to_datetime(frame[time_col], errors='coerce', utc=True)
                            if frame[time_col].dt.tz is not None:
                                frame[time_col] = frame[time_col].dt.tz_localize(None)
                            if 'intime' in frame.columns:
                                frame['intime'] = pd.to_datetime(frame['intime'], errors='coerce', utc=True)
                                if frame['intime'].dt.tz is not None:
                                    frame['intime'] = frame['intime'].dt.tz_localize(None)
                            frame['outtime'] = pd.to_datetime(frame['outtime'], errors='coerce', utc=True)
                            if frame['outtime'].dt.tz is not None:
                                frame['outtime'] = frame['outtime'].dt.tz_localize(None)
                            
                            # 分离需要 rolling join 的数据和不需要的数据
                            single_stay_mask = ~frame['hadm_id'].isin(multi_stay_hadms)
                            single_stay_data = frame[single_stay_mask].copy()
                            multi_stay_data = frame[~single_stay_mask].copy()
                            
                            # 🔥 使用 pd.merge_asof 实现真正的 rolling join
                            # 首先，获取唯一的数据记录（去除 join 导致的重复）
                            data_cols = [c for c in multi_stay_data.columns 
                                        if c not in ['stay_id', 'intime', 'outtime']]
                            unique_data = multi_stay_data[data_cols].drop_duplicates()
                            
                            # 获取每个 hadm_id 的 stay 信息，按 outtime 排序
                            stay_cols = ['hadm_id', 'stay_id', 'outtime']
                            if 'intime' in multi_stay_data.columns:
                                stay_cols.append('intime')
                            stay_info = multi_stay_data[stay_cols].drop_duplicates()
                            stay_info = stay_info.sort_values(['hadm_id', 'outtime'])
                            
                            # 对每个 hadm_id 分别做 merge_asof
                            result_frames = [single_stay_data]
                            
                            for hadm_id in multi_stay_hadms:
                                # 获取该 hadm_id 的数据
                                hadm_unique = unique_data[unique_data['hadm_id'] == hadm_id].copy()
                                if hadm_unique.empty:
                                    continue
                                    
                                # 获取该 hadm_id 的 stay 信息，按 outtime 排序
                                hadm_stays = stay_info[stay_info['hadm_id'] == hadm_id].copy()
                                hadm_stays = hadm_stays.sort_values('outtime')
                                stays_list = hadm_stays['stay_id'].tolist()
                                outtimes_list = hadm_stays['outtime'].tolist()
                                
                                # 确保数据按时间排序
                                hadm_unique = hadm_unique.sort_values(time_col)
                                
                                # 🔥 关键修正：使用 outtime 而不是 intime 做 rolling join
                                # direction='forward' 等价于 roll = -Inf（向未来滚动）
                                # 找 outtime >= charttime 的最近 stay
                                merge_cols = ['stay_id', 'outtime']
                                if 'intime' in hadm_stays.columns:
                                    merge_cols.append('intime')
                                merged = pd.merge_asof(
                                    hadm_unique,
                                    hadm_stays[merge_cols],
                                    left_on=time_col,
                                    right_on='outtime',
                                    direction='forward',  # 向未来滚动：找 outtime >= charttime
                                    allow_exact_matches=True
                                )
                                
                                # 处理 rollends = TRUE: 
                                # 如果 charttime > 最后一个 outtime，分配给最后一个 stay
                                last_stay = stays_list[-1]
                                last_outtime = outtimes_list[-1]
                                merged.loc[merged['stay_id'].isna(), 'stay_id'] = last_stay
                                merged.loc[merged['outtime'].isna(), 'outtime'] = last_outtime
                                
                                # 确保 stay_id 是整数
                                merged['stay_id'] = merged['stay_id'].astype(int)
                                
                                result_frames.append(merged)
                            
                            frame = pd.concat(result_frames, ignore_index=True)
                            
                            if verbose:
                                logger.debug(f"[{table_name}] rolling join 完成: {after_join_rows} → {len(frame)} 行")
                    
                    # 清理临时的 intime 和 outtime 列
                    for col in ['intime', 'outtime']:
                        if col in frame.columns:
                            frame = frame.drop(columns=[col], errors='ignore')
                    
                    after_rows = len(frame)
                    
                    # ✅ 关键修复：join 后必须再次应用原始 stay_id 过滤
                    # 因为 join 可能产生了额外的 stay_ids (同一 subject 或 hadm_id 的多个 ICU stays)
                    # 
                    # 三种情况：
                    # 1. 如果原始过滤器是 stay_id，使用保存的 original_stay_ids
                    # 2. 如果原始过滤器是 subject_id，从 FilterSpec.metadata 中提取原始 stay_id
                    # 3. 从 icustays_filters 中查找
                    target_stay_ids = original_stay_ids
                    
                    if not target_stay_ids and icustays_filters:
                        for spec in icustays_filters:
                            if spec.column == 'stay_id' and spec.op == FilterOp.IN:
                                target_stay_ids = set(spec.value)
                                if verbose:
                                    logger.debug(f"[{table_name}] 从 stay_id 过滤器获取: {len(target_stay_ids)} stays")
                                break
                            elif spec.column == 'subject_id' and spec.op == FilterOp.IN:
                                # 从 metadata 中提取原始 stay_ids
                                if spec.metadata and 'original_stay_ids' in spec.metadata:
                                    target_stay_ids = set(spec.metadata['original_stay_ids'])
                                    if verbose:
                                        logger.debug(f"[{table_name}] 从 subject_id 过滤器的 metadata 获取原始 stay_id: {len(target_stay_ids)} stays")
                                    break
                    
                    if target_stay_ids:
                        before_filter = len(frame)
                        if 'stay_id' in frame.columns:
                            frame = frame[frame['stay_id'].isin(target_stay_ids)]
                            if verbose:
                                logger.debug(
                                    f"[{table_name}] 应用 stay_id 过滤: {before_filter}行 → {len(frame)}行 "
                                    f"(保留 {len(target_stay_ids)} 个目标 stay_id)"
                                )
                        else:
                            if verbose:
                                logger.warning(f"[{table_name}] join 后仍无 stay_id 列，无法应用过滤")
                    
                    # 记录补全操作
                    if verbose and before_rows != after_rows:
                        logger.info(
                            "表 %s: 通过 hadm_id 补全 stay_id (%d → %d 行)",
                            table_name,
                            before_rows,
                            after_rows
                        )
                    
                    # ✅ 关键修复：补全 stay_id 后，更新 id_columns
                    # 这样下游 concept.py 会保留 stay_id 列而不是只保留 subject_id
                    if 'stay_id' in frame.columns:
                        id_columns = ['stay_id']
                        if verbose:
                            logger.debug(f"[{table_name}] 补全 stay_id 后更新 id_columns: subject_id → stay_id")
                        
                except Exception as e:
                    # 如果补全失败，记录警告但不中断流程
                    logger.warning(
                        "⚠️  表 %s: 无法补全 stay_id: %s",
                        table_name,
                        str(e)
                    )

        if verbose and logger.isEnabledFor(logging.INFO):
            id_label = id_columns[0] if id_columns else defaults.id_var or "N/A"
            unique_count = (
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
    ) -> pd.DataFrame:
        # 🔍 调试日志：显示请求的列（仅在DEBUG级别显示）
        if columns:
            logger.debug(f"_load_raw_frame: table={table_name}, columns={list(columns)}")
        
        # 🚀 OPTIMIZATION: 缓存键不包含patient_ids_filter以实现跨概念共享
        # 对于同一批患者的多个概念加载,只在第一次读取表,后续从缓存中过滤
        # 这将chartevents等大表的加载从N次(每概念一次)减少到1次
        # 跳过需要subject_id→stay_id映射的表，这些表缓存会导致patient过滤失效
        # 🔧 FIX: labevents 也可以缓存，只要我们在key中包含filter信息
        skip_cache_tables = ['microbiologyevents', 'inputevents', 'admissions']
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
            # 包含列名和操作符，确保唯一性
            filter_key = (patient_ids_filter.column, patient_ids_filter.op, val)

        cache_key = (table_name, tuple(sorted(columns)) if columns else None, filter_key)
        
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
            # 修复：检查是否为多文件配置，如果是，使用目录路径
            table_cfg = self.config.get_table(table_name)
            if len(table_cfg.files) > 1:
                # 多文件配置：使用目录路径以启用多文件读取
                base_path = self.base_path or Path.cwd()
                if table_cfg.files:
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
            frame = self._read_file(Path(loader), columns, patient_ids_filter=patient_ids_filter)

        if columns is not None:
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

    def _resolve_loader_from_disk(self, table_name: str) -> Optional[Callable[[], pd.DataFrame] | Path]:
        if not self.base_path:
            return None
        
        table_cfg = self.config.get_table(table_name)
        explicit = table_cfg.first_file()
        
        if explicit:
            explicit_path = self.base_path / explicit
            if explicit_path.exists():
                # Accept directories (partitioned datasets) and Parquet files immediately
                if explicit_path.is_dir():
                    return explicit_path
                if explicit_path.suffix.lower() in {".parquet", ".pq"}:
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
            # Try .parquet extension
            parquet_candidate = self.base_path / f"{name}.parquet"
            if parquet_candidate.exists():
                return parquet_candidate
            # Try .pq extension (short form)
            pq_candidate = self.base_path / f"{name}.pq"
            if pq_candidate.exists():
                return pq_candidate
        
        # Check subdirectory for partitioned parquet data (common in hirid observations)
        if self.base_path is not None:
            for name in [table_name, table_name.lower()]:
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
    
    def _read_file(self, path: Path, columns: Optional[Iterable[str]], patient_ids_filter: Optional[FilterSpec] = None) -> pd.DataFrame:
        # Handle directory (partitioned data)
        if path.is_dir():
            if DEBUG_MODE:
                logger.debug(f"读取分区目录: {path.name}, 请求列: {list(columns) if columns else '全部列'}")
            # 🚀 优先使用 DuckDB（单患者/小批量查询快 5-6 倍）
            # 对于大批量患者（>100），PyArrow 的并行读取更优
            use_duckdb = True
            if patient_ids_filter and patient_ids_filter.value:
                values = patient_ids_filter.value
                if isinstance(values, (list, tuple, set)):
                    use_duckdb = len(values) <= 100
                elif isinstance(values, pd.Series):
                    use_duckdb = len(values) <= 100
            
            if use_duckdb:
                return self._read_partitioned_data_duckdb(path, columns, patient_ids_filter)
            else:
                return self._read_partitioned_data_optimized(path, columns, patient_ids_filter)
        
        suffix = path.suffix.lower()
        suffixes = [s.lower() for s in path.suffixes]
        
        # Preferred: Parquet format
        if suffix in {".parquet", ".pq"}:
            # 🚀 使用PyArrow过滤器优化大文件读取
            if patient_ids_filter and patient_ids_filter.op == FilterOp.IN:
                try:
                    import pyarrow.parquet as pq
                    import pyarrow as pa
                    # ⚡ 使用预计算的set
                    target_ids = list(patient_ids_filter._value_set)
                    
                    # 使用PyArrow读取并过滤 - 使用 DNF 格式
                    df = pq.read_table(
                        path,
                        columns=list(columns) if columns else None,
                        filters=[[(patient_ids_filter.column, 'in', target_ids)]]
                    ).to_pandas()
                except (ImportError, Exception) as e:
                    # 如果PyArrow过滤失败，回退到pandas后过滤
                    df = pd.read_parquet(path, columns=list(columns) if columns else None, engine='pyarrow')
                    if patient_ids_filter.column in df.columns:
                        df = patient_ids_filter.apply(df)
            else:
                df = pd.read_parquet(path, columns=list(columns) if columns else None, engine='pyarrow')
            
            # 处理重复列名（如果存在）
            if df.columns.duplicated().any():
                import pandas.io.common
                df.columns = pandas.io.common.dedup_names(df.columns, is_potential_multiindex=False)
            return df
        
        raise ValueError(
            f"Unsupported file format '{path.suffix}' for {path.name}. Only Parquet format is supported."
        )
    
    def _read_partitioned_data_duckdb(self, directory: Path, columns: Optional[Iterable[str]], patient_ids_filter: Optional[FilterSpec] = None) -> pd.DataFrame:
        """使用 DuckDB 读取分区数据（高性能版本）
        
        DuckDB 对单患者/小批量患者查询特别高效，比 PyArrow 快 5-6 倍。
        """
        try:
            import duckdb
        except ImportError:
            # DuckDB 未安装，回退到 PyArrow
            return self._read_partitioned_data_optimized(directory, columns, patient_ids_filter)
        
        # 构建 SQL 查询
        glob_pattern = str(directory / "*.parquet")
        
        # 列选择
        if columns:
            select_cols = ", ".join(list(columns))
        else:
            select_cols = "*"
        
        # WHERE 子句
        where_clause = ""
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
                    where_clause = f"WHERE {id_col} = {value_list[0]}"
                else:
                    values_str = ", ".join(map(str, value_list))
                    where_clause = f"WHERE {id_col} IN ({values_str})"
        
        query = f"SELECT {select_cols} FROM read_parquet('{glob_pattern}') {where_clause}"
        
        try:
            con = duckdb.connect()
            # 🔧 CRITICAL FIX: 设置 DuckDB 时区为 UTC
            # DuckDB 默认将 UTC 时间转换为本地时区，这会导致时间偏移
            # 例如：UTC 15:37 会被转换成 Asia/Shanghai 23:37 (+8 小时)
            # 设置时区为 UTC 可以保持原始 UTC 时间不变
            con.execute("SET timezone='UTC'")
            df = con.execute(query).fetchdf()
            con.close()
            return df
        except Exception as e:
            logger.warning(f"DuckDB 读取失败，回退到 PyArrow: {e}")
            return self._read_partitioned_data_optimized(directory, columns, patient_ids_filter)
    
    def _read_partitioned_data_optimized(self, directory: Path, columns: Optional[Iterable[str]], patient_ids_filter: Optional[FilterSpec] = None) -> pd.DataFrame:
        """读取分区数据（优化版本）"""
        try:
            import pyarrow.dataset as ds
            
            # 🚀 使用PyArrow Dataset - 最快的方式
            dataset = ds.dataset(
                directory,
                format='parquet',
                partitioning=None,
                exclude_invalid_files=True
            )
            
            filter_expr = None
            if patient_ids_filter:
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

                if not value_list:
                    wanted_cols = list(columns) if columns else dataset.schema.names
                    return pd.DataFrame(columns=wanted_cols)

                try:
                    filter_expr = ds.field(id_col).isin(value_list)
                except Exception:
                    filter_expr = None

            # 批量读取，启用多线程（优化大规模提取）
            # 🚀 优化：为90000+患者提取增加线程池
            import os
            thread_count = 32  # 最优配置：32线程
            
            if columns:
                table = dataset.to_table(
                    columns=list(columns), 
                    filter=filter_expr,
                    use_threads=thread_count  # 明确线程数
                )
            else:
                table = dataset.to_table(
                    filter=filter_expr,
                    use_threads=thread_count
                )

            # 转换为 pandas，使用 zero-copy 优化
            return table.to_pandas(split_blocks=True, self_destruct=True)
            
        except Exception:
            # 回退到简单方式
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
                        import pyarrow.compute as pc  # type: ignore
                        table = table.filter(arrow_filters)
                    df = table.to_pandas()
                    dfs.append(df)
                    continue
                except Exception:
                    pass  # Fallback to pandas.read_parquet below
            df = pd.read_parquet(f, columns=list(columns) if columns else None)
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
            return table.to_pandas()
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
        frame = table.to_pandas()

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
