"""Data loading utilities for ICU datasets."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from threading import RLock

import pandas as pd

from .config import DataSourceConfig, DataSourceRegistry, DatasetOptions
from .table import ICUTable

# 全局调试开关 - 设置为 False 可以减少输出
DEBUG_MODE = False
logger = logging.getLogger(__name__)

# 🚀 AUMC numericitems 优化：只加载 SOFA 相关的 itemids
# 原始表 80GB，过滤后约 5GB，性能提升约 15 倍
# 这些 itemids 来自 concept-dict.json 和 sofa2-dict.json 中 AUMC numericitems 源
AUMC_NUMERICITEMS_ITEMIDS = {
    6640, 6641, 6642, 6643, 6684, 6707, 6709, 6773, 6774, 6776, 6777, 6778, 6779, 
    6786, 6789, 6796, 6797, 6800, 6801, 6803, 6806, 6807, 6808, 6810, 6812, 6813, 
    6815, 6817, 6822, 6824, 6825, 6828, 6833, 6835, 6836, 6837, 6839, 6840, 6846, 
    6848, 6850, 7433, 8658, 8794, 8874, 8884, 8885, 8903, 8915, 9553, 9555, 9556, 
    9557, 9560, 9561, 9580, 9658, 9924, 9927, 9930, 9933, 9935, 9937, 9941, 9943, 
    9945, 9947, 9952, 9960, 9962, 9964, 9965, 9967, 9968, 9989, 9990, 9992, 9994, 
    9996, 10053, 10079, 10175, 10267, 10284, 10285, 10286, 10407, 10409, 11423, 
    11545, 11586, 11679, 11690, 11692, 11710, 11812, 11846, 11856, 11893, 11902, 
    11978, 11984, 11990, 11998, 12266, 12279, 12310, 12311, 12356, 12805, 13076, 
    13952, 14216, 14252, 14254, 14256, 14258, 16110, 16166, 17982, 18666, 18952, 
    19703, 20656, 21213, 21214,
    # 🆕 SOFA-2 adv_resp (高级呼吸支持) itemids - 用于 PEEP 检测
    # 6694=Eind exp. druk/PEEP (15.6M rows), 12284=PEEP Set (15.6M rows), 8862=PEEP/CPAP (85K rows)
    6694, 12284, 8862,
    # 🆕 SOFA-2 rrt (肾脏替代治疗) itemids
    # 7666, 7667, 7668=透析相关, 8805=CRRT, 10736=血液透析, 12444=腹膜透析
    7666, 7667, 7668, 8805, 10736, 12444,
}

# 🚀 MIIV chartevents 优化：只加载 SOFA 相关的 93 个 itemids
# 原始表 11GB，过滤后大幅减少
MIIV_CHARTEVENTS_ITEMIDS = {
    467, 469, 220045, 220050, 220051, 220052, 220128, 220179, 220180, 220181, 
    220210, 220227, 220277, 220339, 220739, 223761, 223762, 223835, 223848, 223849, 
    223900, 223901, 224027, 224309, 224310, 224311, 224322, 224419, 224652, 224654, 
    224660, 224684, 224685, 224686, 224687, 224688, 224689, 224690, 224695, 224696, 
    224697, 224700, 224701, 224702, 224703, 224704, 224705, 224706, 224707, 224709, 
    224738, 224746, 224747, 224750, 225312, 225436, 225949, 225979, 226253, 226512, 
    226707, 226732, 226873, 227187, 227290, 227577, 227578, 227579, 227580, 227583, 
    227980, 228096, 228151, 228154, 228156, 228158, 228193, 228198, 228300, 228332, 
    228337, 228640, 228866, 229254, 229266, 229268, 229270, 229274, 229277, 229278, 
    229280, 229314, 229326,
}

# 🚀 MIIV labevents 优化：只加载 SOFA 相关的 53 个 itemids
# 原始表 8GB，过滤后大幅减少
MIIV_LABEVENTS_ITEMIDS = {
    50802, 50804, 50808, 50809, 50813, 50814, 50816, 50817, 50818, 50820, 50821, 
    50822, 50852, 50861, 50862, 50863, 50878, 50882, 50883, 50885, 50889, 50893, 
    50902, 50910, 50911, 50912, 50931, 50960, 50970, 50971, 50983, 51002, 51003, 
    51006, 51144, 51146, 51200, 51214, 51221, 51222, 51237, 51244, 51248, 51249, 
    51250, 51256, 51265, 51274, 51275, 51277, 51279, 51288, 51301,
}

# 🚀 eICU nursecharting 优化：只加载 SOFA 相关的字符串 IDs
# 原始表 4.3GB，过滤后大幅减少
# 注意：eICU nursecharting 使用 nursingchartcelltypevalname 列进行过滤
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
                'No Response-ETT': {223900},
                '1.0 ET/Trach': {223900},
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

# 🚀 HiRID observations 优化：只加载概念字典中定义的 198 个 variableids
# 原始表 7.77 亿行（~72GB内存），过滤后大幅减少
# 这些 variableids 来自 concept-dict.json 和 sofa2-dict.json 中 HiRID observations 源
HIRID_OBSERVATIONS_VARIABLEIDS = {
    15, 71, 100, 110, 112, 113, 120, 146, 151, 163, 176, 181, 186, 189, 200, 239, 
    300, 310, 326, 331, 351, 400, 405, 410, 426, 610, 2010, 2200, 3845, 4000, 7100, 
    8280, 8290, 1000022, 1000060, 1000234, 1000272, 1000273, 1000274, 1000284, 
    1000299, 1000300, 1000302, 1000304, 1000305, 1000306, 1000315, 1000317, 1000318, 
    1000320, 1000321, 1000322, 1000325, 1000335, 1000348, 1000352, 1000363, 1000365, 
    1000383, 1000390, 1000407, 1000408, 1000424, 1000425, 1000426, 1000431, 1000432, 
    1000433, 1000434, 1000435, 1000437, 1000462, 1000483, 1000486, 1000487, 1000488, 
    1000507, 1000508, 1000518, 1000519, 1000544, 1000545, 1000549, 1000567, 1000601, 
    1000648, 1000649, 1000650, 1000655, 1000656, 1000657, 1000658, 1000666, 1000670, 
    1000671, 1000689, 1000690, 1000724, 1000746, 1000750, 1000760, 1000769, 1000770, 
    1000781, 1000791, 1000797, 1000812, 1000825, 1000829, 1000830, 1000835, 1000837, 
    1000838, 1000854, 1000855, 1000893, 1000894, 1000929, 1001005, 1001068, 1001075, 
    1001079, 1001084, 1001086, 1001095, 1001096, 1001097, 1001098, 1001168, 1001169, 
    1001170, 1001171, 1001173, 1001193, 1001198, 10000100, 10000200, 10000300, 
    10000400, 10000450, 15001552, 15001565, 20000110, 20000200, 20000300, 20000400, 
    20000500, 20000600, 20000700, 20000800, 20000900, 20001200, 20001300, 20002200, 
    20002500, 20002600, 20002700, 20004100, 20004200, 20004300, 20004410, 20005100, 
    20005110, 24000150, 24000160, 24000170, 24000210, 24000220, 24000230, 24000330, 
    24000439, 24000480, 24000519, 24000520, 24000521, 24000522, 24000523, 24000524, 
    24000526, 24000536, 24000548, 24000549, 24000550, 24000557, 24000560, 24000567, 
    24000585, 24000605, 24000658, 24000668, 24000806, 24000833, 24000835, 24000836, 
    24000866, 24000867, 30005110, 30010009,
}

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
        self._bucket_dir_logged: set = set()  # 🔧 已打印日志的分桶目录（避免重复日志）
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
            # 🚀 HiRID 大表优化：提取 sub_var/ids 过滤器用于 DuckDB 精确过滤
            # 这确保加载 hr 时只查询 variableid=200，而不是全局白名单的 198 个 ID
            concept_itemid_filter = None  # (column_name, set_of_ids)
            
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
                                 'patientid',  # HiRID
                                 'CaseID', 'caseid']  # 🔧 FIX 2026-01-26: 添加 SICdb CaseID
                    
                    # 🚀 检测 sub_var/ids 过滤器（用于 HiRID observations 等大表）
                    # 这些过滤器应该在 DuckDB 层应用，而不是内存中应用
                    # 🔧 FIX 2026-01-26: 添加 SICdb DataID, LaboratoryID, DrugID
                    sub_var_columns = ['variableid', 'itemid', 'nursingchartcelltypevalname',
                                       'DataID', 'LaboratoryID', 'DrugID']
                    if spec.op == FilterOp.IN and spec.column in sub_var_columns:
                        # 提取概念特定的 itemid 过滤器
                        ids = spec.value
                        if isinstance(ids, (list, tuple)):
                            ids = set(ids)
                        elif not isinstance(ids, set):
                            ids = {ids}
                        concept_itemid_filter = (spec.column, ids)
                        if DEBUG_MODE:
                            logger.info(f"🎯 概念特定过滤器: {spec.column} IN {len(ids)} 个 ID")
                        continue  # 继续处理，找 patient_id 过滤器
                    
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
            hospital_tables = ['prescriptions', 'labevents', 'microbiologyevents', 'emar', 'pharmacy']
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
                            
                            # 对每个 hadm_id 分别做 merge_asof
                            result_frames = [single_stay_data]
                            
                            for hadm_id in multi_stay_hadms:
                                # 获取该 hadm_id 的数据
                                hadm_unique = unique_data[unique_data['hadm_id'] == hadm_id].copy()
                                if hadm_unique.empty:
                                    continue
                                    
                                # 获取该 hadm_id 的 stay 信息，按 outtime 排序
                                hadm_stays = stay_info[stay_info['hadm_id'] == hadm_id].copy()
                                # 🔧 FIX: 过滤掉 outtime 为空的行，避免 merge_asof 报错
                                # "Merge keys contain null values on right side"
                                hadm_stays = hadm_stays.dropna(subset=['outtime'])
                                if hadm_stays.empty:
                                    continue
                                hadm_stays = hadm_stays.sort_values('outtime')
                                stays_list = hadm_stays[target_id_col].tolist()
                                outtimes_list = hadm_stays['outtime'].tolist()
                                
                                # 🔧 FIX: 过滤掉时间列为空的行，避免 merge_asof 报错
                                # "Merge keys contain null values on left side"
                                hadm_unique = hadm_unique.dropna(subset=[time_col])
                                if hadm_unique.empty:
                                    continue
                                
                                # 确保数据按时间排序
                                hadm_unique = hadm_unique.sort_values(time_col)
                                
                                # 🔥 关键修正：使用 outtime 而不是 intime 做 rolling join
                                # direction='forward' 等价于 roll = -Inf（向未来滚动）
                                # 找 outtime >= charttime 的最近 stay
                                merge_cols = [target_id_col, 'outtime']
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
                                merged.loc[merged[target_id_col].isna(), target_id_col] = last_stay
                                merged.loc[merged['outtime'].isna(), 'outtime'] = last_outtime
                                
                                # 确保 id 是整数
                                merged[target_id_col] = merged[target_id_col].astype(int)
                                
                                result_frames.append(merged)
                            
                            frame = pd.concat(result_frames, ignore_index=True)
                            
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
                    logger.info(f"🔄 MIMIC-III chartevents: 分桶目录不存在，使用 CSV 回退模式")
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
                            logger.info(f"🪣 使用分桶目录: {bucket_dir} ({len(bucket_subdirs)} 个桶)")
                        return bucket_dir
        
        return None

    def _resolve_loader_from_disk(self, table_name: str) -> Optional[Callable[[], pd.DataFrame] | Path]:
        if not self.base_path:
            return None
        
        # 🚀 优先级最高：检查分桶目录（性能最优）
        # 分桶目录命名规则：{table_name}_bucket
        # 必须在检查配置文件之前，因为分桶目录是性能优化的关键
        possible_bucket_dirs = [
            self.base_path / f"{table_name}_bucket",  # 直接在 base_path 下
            self.base_path / "icu" / f"{table_name}_bucket",  # MIIV icu 子目录
            self.base_path / "hosp" / f"{table_name}_bucket",  # MIIV hosp 子目录
        ]
        for bucket_dir in possible_bucket_dirs:
            if bucket_dir.is_dir():
                bucket_subdirs = list(bucket_dir.glob("bucket_id=*"))
                if bucket_subdirs:
                    # 🔧 避免重复日志：只在首次发现时打印info
                    bucket_key = str(bucket_dir)
                    if bucket_key not in self._bucket_dir_logged:
                        self._bucket_dir_logged.add(bucket_key)
                        logger.info(f"🪣 使用分桶目录: {bucket_dir} ({len(bucket_subdirs)} 个桶)")
                    return bucket_dir
        
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
            # 🚀 优先检查 bucket 目录（分桶格式，性能最优）
            # 必须在检查 .parquet 文件之前，因为分桶目录可能与表同名
            # 检查多个可能的位置：base_path 和子目录（如 icu/, hosp/）
            possible_bucket_dirs = [
                self.base_path / f"{name}_bucket",  # 直接在 base_path 下
                self.base_path / "icu" / f"{name}_bucket",  # MIIV icu 子目录
                self.base_path / "hosp" / f"{name}_bucket",  # MIIV hosp 子目录
            ]
            for bucket_dir in possible_bucket_dirs:
                if bucket_dir.is_dir():
                    # 检查是否有 bucket_id=* 子目录
                    bucket_subdirs = list(bucket_dir.glob("bucket_id=*"))
                    if bucket_subdirs:
                        # 🔧 避免重复日志：只在首次发现时打印info
                        bucket_key = str(bucket_dir)
                        if bucket_key not in self._bucket_dir_logged:
                            self._bucket_dir_logged.add(bucket_key)
                            logger.info(f"🪣 使用分桶目录: {bucket_dir} ({len(bucket_subdirs)} 个桶)")
                        return bucket_dir
            
            # Try .parquet extension - 检查多个可能的位置
            # MIMIC-IV 的表可能在 icu/ 或 hosp/ 子目录下
            possible_parquet_paths = [
                self.base_path / f"{name}.parquet",  # 直接在 base_path 下
                self.base_path / "icu" / f"{name}.parquet",  # MIIV icu 子目录
                self.base_path / "hosp" / f"{name}.parquet",  # MIIV hosp 子目录
            ]
            for parquet_candidate in possible_parquet_paths:
                if parquet_candidate.exists():
                    return parquet_candidate
            
            # Try .pq extension (short form) - 同样检查子目录
            possible_pq_paths = [
                self.base_path / f"{name}.pq",
                self.base_path / "icu" / f"{name}.pq",
                self.base_path / "hosp" / f"{name}.pq",
            ]
            for pq_candidate in possible_pq_paths:
                if pq_candidate.exists():
                    return pq_candidate
        
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
        elif db_name in ('miiv', 'mimic_demo') and table_name == 'chartevents':
            # MIIV chartevents: 11GB
            itemid_filter_config = ('itemid', MIIV_CHARTEVENTS_ITEMIDS)
        elif db_name in ('miiv', 'mimic_demo') and table_name == 'labevents':
            # MIIV labevents: 8GB
            itemid_filter_config = ('itemid', MIIV_LABEVENTS_ITEMIDS)
        elif db_name == 'eicu' and table_name == 'nursecharting':
            # eICU nursecharting: 4.3GB - 使用字符串列
            itemid_filter_config = ('nursingchartcelltypevalname', EICU_NURSECHARTING_IDS)
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
                if isinstance(values, (list, tuple, set)):
                    use_duckdb = len(values) <= 100
                elif isinstance(values, pd.Series):
                    use_duckdb = len(values) <= 100
            
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
        [s.lower() for s in path.suffixes]
        
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
                except (ImportError, Exception):
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
        conn = duckdb_module.connect()
        itemid_list = list(itemids)
        conn.execute("CREATE TEMP TABLE items AS SELECT UNNEST(?) as itemid", [itemid_list])
        result = conn.execute(f"SELECT DISTINCT hash(itemid) % {num_buckets} FROM items").fetchall()
        conn.close()
        return {row[0] for row in result}
    
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
        
        logger.info(f"📄 MIMIC-III CSV 回退模式: 从 {csv_path.name} 读取 (VALUE 列保持为字符串)")
        
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
        query = f"""
            SELECT {columns_sql}
            FROM read_csv_auto(
                '{csv_path}',
                ignore_errors=true,
                null_padding=true,
                types={{'VALUE': 'VARCHAR'}}
            )
            {where_clause}
        """
        
        try:
            con = duckdb.connect()
            con.execute("SET timezone='UTC'")
            con.execute("SET enable_progress_bar = false")
            con.execute("SET enable_progress_bar_print = false")
            df = con.execute(query).fetchdf()
            con.close()
            
            # Normalize column names to lowercase (MIMIC-III CSV uses uppercase)
            df.columns = [c.lower() for c in df.columns]
            
            logger.info(f"✅ CSV 回退成功: 加载 {len(df)} 行")
            return df
            
        except Exception as e:
            logger.error(f"❌ MIMIC-III CSV 回退失败: {e}")
            return pd.DataFrame()
    
    def _resolve_mimic3_chartevents_csv(self) -> Optional[Path]:
        """Find MIMIC-III CHARTEVENTS.csv.gz file
        
        Returns:
            Path to CSV file if found, None otherwise
        """
        if not self.base_path:
            return None
        
        # Try different possible file names (MIMIC-III uses uppercase)
        possible_names = [
            'CHARTEVENTS.csv.gz',
            'chartevents.csv.gz', 
            'CHARTEVENTS.csv',
            'chartevents.csv',
        ]
        
        for name in possible_names:
            csv_path = self.base_path / name
            if csv_path.exists():
                return csv_path
        
        return None

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
        bucket_subdirs = list(directory.glob("bucket_id=*"))
        if bucket_subdirs:
            # 🔧 CRITICAL: 使用最大 bucket_id + 1 作为桶数
            # 不能用 len(bucket_subdirs)，因为某些桶可能是空的（没有目录）
            # 例如 HiRID 有 100 个桶但只有 81 个非空桶
            max_bucket_id = max(int(d.name.split("=")[1]) for d in bucket_subdirs)
            num_buckets = max_bucket_id + 1
            
            # 🚀 关键优化：如果有 itemid 过滤条件，计算目标桶，只扫描这些桶
            if itemid_filter_config:
                filter_col, filter_ids = itemid_filter_config
                # 只有数值型 itemid 才能使用 hash 分桶
                numeric_ids = {int(x) for x in filter_ids if isinstance(x, (int, float)) and not isinstance(x, bool)}
                if numeric_ids:
                    # 使用 DuckDB hash 计算目标桶（与分桶转换时一致）
                    target_buckets = self._compute_target_buckets(numeric_ids, num_buckets, duckdb)
                    # 构建只包含目标桶的文件列表
                    target_files = []
                    for bucket_id in target_buckets:
                        bucket_dir = directory / f"bucket_id={bucket_id}"
                        if bucket_dir.exists():
                            target_files.extend(bucket_dir.glob("*.parquet"))
                    if target_files:
                        # 使用精确的文件列表而非全扫描
                        file_list_str = ", ".join(f"'{f}'" for f in target_files)
                        glob_pattern = f"[{file_list_str}]"
                        logger.debug(f"🪣 分桶精准读取: {len(target_buckets)}/{num_buckets} 个桶, {len(target_files)} 个文件")
                    else:
                        # 目标桶不存在，可能是空数据
                        logger.warning(f"⚠️ 目标桶不存在: bucket_id in {target_buckets}")
                        glob_pattern = str(directory / "**/*.parquet")
                else:
                    # 字符串型 ID，无法使用 hash 分桶优化
                    glob_pattern = str(directory / "**/*.parquet")
                    logger.debug(f"🪣 使用分桶模式读取(全扫描): {directory.name}")
            else:
                # 没有 itemid 过滤，全扫描
                glob_pattern = str(directory / "**/*.parquet")
                logger.debug(f"🪣 使用分桶模式读取(无过滤): {directory.name}")
        else:
            # 普通分区: directory/*.parquet
            glob_pattern = str(directory / "*.parquet")
        
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
            # 检查是否为字符串类型的 ID（如 eICU nursecharting）
            is_string_ids = any(isinstance(x, str) for x in filter_ids)
            if is_string_ids:
                # 字符串 ID 需要加引号
                id_str = ", ".join(f"'{x}'" for x in sorted(filter_ids))
            else:
                # 数字 ID
                id_str = ", ".join(map(str, sorted(filter_ids)))
            where_conditions.append(f"{filter_col} IN ({id_str})")
            if DEBUG_MODE:
                logger.info(f"🚀 大表优化: {filter_col} 过滤 {len(filter_ids)} 个 ID")
        
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
            con = duckdb.connect()
            # 🔧 CRITICAL FIX: 设置 DuckDB 时区为 UTC
            # DuckDB 默认将 UTC 时间转换为本地时区，这会导致时间偏移
            # 例如：UTC 15:37 会被转换成 Asia/Shanghai 23:37 (+8 小时)
            # 设置时区为 UTC 可以保持原始 UTC 时间不变
            con.execute("SET timezone='UTC'")
            # 🔧 禁用DuckDB进度条，避免终端输出开销
            con.execute("SET enable_progress_bar = false")
            con.execute("SET enable_progress_bar_print = false")
            df = con.execute(query).fetchdf()
            con.close()
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
            
            # 构建过滤表达式（支持多个条件）
            filter_exprs = []
            
            # 患者 ID 过滤
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
                    filter_exprs.append(ds.field(id_col).isin(value_list))
                except Exception:
                    pass
            
            # 🚀 大表优化：itemid 预过滤 (AUMC numericitems, MIIV chartevents/labevents)
            if itemid_filter_config:
                try:
                    filter_col, filter_ids = itemid_filter_config
                    filter_exprs.append(ds.field(filter_col).isin(list(filter_ids)))
                    if DEBUG_MODE:
                        logger.info(f"🚀 大表优化 (PyArrow): {filter_col} 过滤 {len(filter_ids)} 个 ID")
                except Exception:
                    pass
            
            # 合并过滤条件
            filter_expr = None
            if filter_exprs:
                filter_expr = filter_exprs[0]
                for expr in filter_exprs[1:]:
                    filter_expr = filter_expr & expr

            # 批量读取，启用多线程（优化大规模提取）
            # 🚀 优化：为90000+患者提取增加线程池
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
    db_name = data_source.config.name if hasattr(data_source.config, 'name') else 'unknown'
    
    # 🔧 直接查找分桶目录（不依赖_resolve_loader_from_disk）
    base_path = data_source.base_path
    bucket_table_name = f"{table_name}_bucket"
    
    possible_bucket_dirs = [
        base_path / bucket_table_name,
        base_path / "icu" / bucket_table_name,
        base_path / "hosp" / bucket_table_name,
    ]
    
    bucket_dir = None
    for dir_path in possible_bucket_dirs:
        if dir_path.is_dir():
            bucket_subdirs = list(dir_path.glob("bucket_id=*"))
            if bucket_subdirs:
                bucket_dir = dir_path
                break
    
    if bucket_dir is None:
        raise ValueError(f"Cannot find bucketed directory for {table_name} (tried: {[str(p) for p in possible_bucket_dirs]})")
    
    # 确定ID列和时间列
    if id_col is None:
        if db_name == 'aumc':
            id_col = 'admissionid'
        elif db_name == 'hirid':
            id_col = 'patientid'
        else:
            id_col = 'stay_id'
    
    if time_col is None:
        if db_name == 'aumc':
            time_col = 'measuredat'  # AUMC使用measuredat（毫秒时间戳）
        elif db_name == 'hirid':
            time_col = 'datetime'
        else:
            time_col = 'charttime'
    
    # 确定itemid列名
    if db_name == 'aumc':
        itemid_col = 'itemid'
    elif db_name == 'hirid':
        itemid_col = 'variableid'
    else:
        itemid_col = 'itemid'
    
    # 计算目标桶
    conn = duckdb.connect()
    conn.execute("SET timezone='UTC'")
    # 🔧 禁用DuckDB进度条，避免16秒的终端输出开销
    conn.execute("SET enable_progress_bar = false")
    conn.execute("SET enable_progress_bar_print = false")
    
    # 获取桶数
    bucket_subdirs = list(bucket_dir.glob("bucket_id=*"))
    if not bucket_subdirs:
        conn.close()
        return pd.DataFrame()
    
    num_buckets = max(int(d.name.split('=')[1]) for d in bucket_subdirs) + 1
    
    # 计算目标桶ID
    itemid_list = list(itemids)
    conn.execute("CREATE TEMP TABLE items AS SELECT UNNEST(?) as itemid", [itemid_list])
    result = conn.execute(f"SELECT DISTINCT hash(itemid) % {num_buckets} FROM items").fetchall()
    target_buckets = [row[0] for row in result]
    
    # 构建文件列表
    target_files = []
    for bucket_id in target_buckets:
        bucket_subdir = bucket_dir / f"bucket_id={bucket_id}"
        if bucket_subdir.exists():
            target_files.extend(bucket_subdir.glob("*.parquet"))
    
    if not target_files:
        conn.close()
        return pd.DataFrame()
    
    file_list_str = ", ".join(f"'{f}'" for f in target_files)
    glob_pattern = f"[{file_list_str}]"
    
    # 构建WHERE条件
    where_conditions = []
    
    # itemid过滤
    ids_str = ", ".join(str(x) for x in itemids)
    where_conditions.append(f"{itemid_col} IN ({ids_str})")
    
    # 患者过滤
    if patient_ids:
        patient_str = ", ".join(str(x) for x in patient_ids)
        where_conditions.append(f"{id_col} IN ({patient_str})")
    
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
    if db_name == 'aumc':
        # AUMC measuredat是Unix毫秒时间戳，转换为分钟后再取整
        time_round_expr = f"FLOOR(({time_col} / 60000.0) / {interval_minutes}) * {interval_minutes}"
        # 输出时间列为分钟偏移量（相对于admittedat）
        output_time_expr = f"{time_round_expr} as measuredat_minutes"
        # 标准查询
        query = f"""
        SELECT 
            {id_col},
            {output_time_expr},
            {itemid_col},
            {duckdb_agg}({value_column}) as {value_column}
        FROM read_parquet({glob_pattern}, union_by_name=true)
        {where_clause}
        GROUP BY {id_col}, {time_round_expr}, {itemid_col}
        ORDER BY {id_col}, 2, {itemid_col}
        """
    elif db_name == 'hirid':
        # 🚀 HiRID 优化: 在 DuckDB 中直接完成时间转换（datetime → 相对入院小时数）
        # 这样避免了 Python 中的 merge + 时间计算开销（从 20s 优化到 0.6s）
        # 🔧 FIX: HiRID 的 general_table 可能是 CSV 或 Parquet 格式
        general_path = data_source.base_path / 'general_table.parquet'
        general_read_func = 'read_parquet'
        if not general_path.exists():
            general_csv = data_source.base_path / 'general_table.csv'
            if general_csv.exists():
                general_path = general_csv
                general_read_func = 'read_csv'
        
        # HiRID: 使用 general 表的 admissiontime 计算相对小时数
        time_round_expr = f"FLOOR(EPOCH(o.{time_col} - CAST(a.admissiontime AS TIMESTAMP)) / 3600.0 / {interval_minutes / 60}) * {interval_minutes / 60}"
        output_time_expr = f"{time_round_expr} as charttime"
        
        # 🔧 修复: 为 HiRID 的 JOIN 查询添加表别名前缀
        # 因为使用了 JOIN，列名需要明确来自哪个表
        hirid_where_clause = where_clause.replace(f'{itemid_col}', f'o.{itemid_col}')
        hirid_where_clause = hirid_where_clause.replace(f'{id_col} IN', f'o.{id_col} IN')
        
        query = f"""
        WITH adm AS (
            SELECT patientid, CAST(admissiontime AS TIMESTAMP) as admissiontime 
            FROM {general_read_func}('{general_path}')
        )
        SELECT 
            o.{id_col},
            {output_time_expr},
            o.{itemid_col},
            {duckdb_agg}(o.{value_column}) as {value_column}
        FROM read_parquet({glob_pattern}, union_by_name=true) o
        JOIN adm a ON o.{id_col} = a.patientid
        {hirid_where_clause}
        GROUP BY o.{id_col}, {time_round_expr}, o.{itemid_col}
        ORDER BY o.{id_col}, 2, o.{itemid_col}
        """
    else:
        time_round_expr = f"FLOOR({time_col} / {interval_minutes}) * {interval_minutes}"
        output_time_expr = f"{time_round_expr} as charttime"
        # 标准查询
        query = f"""
        SELECT 
            {id_col},
            {output_time_expr},
            {itemid_col},
            {duckdb_agg}({value_column}) as {value_column}
        FROM read_parquet({glob_pattern}, union_by_name=true)
        {where_clause}
        GROUP BY {id_col}, {time_round_expr}, {itemid_col}
        ORDER BY {id_col}, 2, {itemid_col}
        """
    
    try:
        df = conn.execute(query).fetchdf()
        logger.info(f"🚀 分桶表DuckDB聚合完成: {table_name} itemids={len(itemids)} -> {len(df):,} 行")
        return df
    except Exception as e:
        logger.warning(f"DuckDB聚合失败: {e}")
        raise
    finally:
        conn.close()


def load_wide_table_aggregated(
    data_source: "ICUDataSource",
    table_name: str,
    value_columns: List[str],
    interval_hours: float = 1.0,
    patient_ids: Optional[List] = None,
    agg_func: str = 'median',  # 'median' (default, matches R ricu), 'first', 'mean', 'max', 'min'
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
    table_cfg = data_source.config.get_table(table_name)
    
    # 确定ID列和时间列
    id_col = table_cfg.defaults.id_var
    if not id_col:
        icustay_cfg = data_source.config.id_configs.get('icustay')
        id_col = icustay_cfg.id if icustay_cfg else 'patientunitstayid'
    
    time_col = table_cfg.defaults.index_var or 'observationoffset'
    
    # 确定数据目录
    table_path = data_source._resolve_loader_from_disk(table_name)
    if table_path is None:
        raise ValueError(f"Cannot find data for table {table_name}")
    
    # table_path 返回 Path 对象
    directory = table_path if isinstance(table_path, Path) else Path(table_path)
    
    # 构建glob pattern
    if directory.is_dir():
        glob_pattern = str(directory / "*.parquet")
    else:
        glob_pattern = str(directory)
    
    # 构建DuckDB聚合函数映射 (median为R ricu默认)
    agg_map = {
        'median': 'MEDIAN',  # R ricu default
        'first': 'FIRST',
        'mean': 'AVG',
        'max': 'MAX', 
        'min': 'MIN',
    }
    duckdb_agg = agg_map.get(agg_func, 'MEDIAN')
    
    # 构建CTE：每个值列单独聚合（处理NULL）
    cte_parts = []
    for i, val_col in enumerate(value_columns):
        cte_name = f"agg_{i}"
        cte_sql = f"""
        {cte_name} AS (
            SELECT 
                {id_col},
                FLOOR({time_col} / {interval_hours * 60.0}) as charttime,
                {duckdb_agg}({val_col}) as {val_col}
            FROM raw_data
            WHERE {val_col} IS NOT NULL
            GROUP BY {id_col}, FLOOR({time_col} / {interval_hours * 60.0})
        )"""
        cte_parts.append(cte_sql)
    
    # 构建WHERE条件
    where_conditions = []
    if patient_ids:
        ids_str = ", ".join(str(x) for x in patient_ids)
        where_conditions.append(f"{id_col} IN ({ids_str})")
    
    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)
    
    # 构建最终合并查询（FULL OUTER JOIN所有CTE）
    if len(value_columns) == 1:
        # 单列简单处理
        query = f"""
        WITH raw_data AS (
            SELECT {id_col}, {time_col}, {value_columns[0]}
            FROM read_parquet('{glob_pattern}', union_by_name=true)
            {where_clause}
        ),
        {cte_parts[0]}
        SELECT {id_col}, charttime, {value_columns[0]}
        FROM agg_0
        ORDER BY {id_col}, charttime
        """
    else:
        # 多列合并
        # 使用COALESCE逐步合并所有CTE
        coalesce_id = f"COALESCE(agg_0.{id_col}"
        coalesce_time = "COALESCE(agg_0.charttime"
        
        for i in range(1, len(value_columns)):
            coalesce_id += f", agg_{i}.{id_col}"
            coalesce_time += f", agg_{i}.charttime"
        
        coalesce_id += f") as {id_col}"
        coalesce_time += ") as charttime"
        
        # 构建JOIN链
        join_sql = "agg_0"
        for i in range(1, len(value_columns)):
            prev_id = ", ".join(f"agg_{j}.{id_col}" for j in range(i))
            prev_time = ", ".join(f"agg_{j}.charttime" for j in range(i))
            join_sql += f"""
            FULL OUTER JOIN agg_{i} 
                ON COALESCE({prev_id}) = agg_{i}.{id_col} 
                AND COALESCE({prev_time}) = agg_{i}.charttime"""
        
        select_cols = [coalesce_id, coalesce_time]
        select_cols += [f"agg_{i}.{col}" for i, col in enumerate(value_columns)]
        
        query = f"""
        WITH raw_data AS (
            SELECT {id_col}, {time_col}, {', '.join(value_columns)}
            FROM read_parquet('{glob_pattern}', union_by_name=true)
            {where_clause}
        ),
        {','.join(cte_parts)}
        SELECT {', '.join(select_cols)}
        FROM {join_sql}
        ORDER BY 1, 2
        """
    
    # 执行查询
    conn = duckdb.connect()
    conn.execute("SET timezone='UTC'")
    # 🔧 禁用DuckDB进度条，避免终端输出开销
    conn.execute("SET enable_progress_bar = false")
    conn.execute("SET enable_progress_bar_print = false")
    
    try:
        df = conn.execute(query).fetchdf()
        logger.info(f"🚀 宽表批量加载完成: {table_name} {value_columns} -> {len(df):,} 行")
        return df
    finally:
        conn.close()
