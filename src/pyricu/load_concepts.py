"""
完整的概念加载系统
实现 R ricu 的 load_concepts 功能
"""
from typing import List, Optional, Union, Dict, Any, Callable, Iterable, Sequence, Mapping
import logging
from datetime import timedelta

import pandas as pd

from .concept import Concept, load_dictionary
from .config import DataSourceConfig, TableConfig, load_src_cfg
from .datasource import ICUDataSource
from .table import load_table
from .ts_utils import change_interval, aggregate_data
from .callback_utils import combine_callbacks

# DataSource 别名用于向后兼容
DataSource = ICUDataSource

# 常见列名集合，用于推测可能需要的列
COMMON_ID_COLUMNS = [
    'stay_id', 'icustay_id', 'subject_id', 'hadm_id',
    'patientunitstayid', 'patientid', 'patient_id', 'admissionid',
    'admission_id', 'patienthealthsystemstayid', 'uniquepid',
    'encounter', 'encounter_id', 'visit_id', 'visitid', 'episode_id',
]

ID_TYPE_HINTS = {
    'patient': ['subject_id', 'patientid', 'patient_id', 'uniquepid'],
    'hadm': ['hadm_id', 'admissionid', 'admission_id', 'visit_id', 'encounter_id'],
    'icustay': ['stay_id', 'icustay_id', 'patientunitstayid'],
}

COMMON_TIME_COLUMNS = [
    'charttime', 'time', 'datetime', 'timestamp', 'starttime', 'endtime',
    'intime', 'outtime', 'admittime', 'dischtime', 'createtime',
    'observationoffset', 'chartoffset', 'eventtime', 'realtime'
]

COMMON_VALUE_COLUMNS = [
    'valuenum', 'value', 'valuetext', 'valueasnumber', 'value_as_number',
    'amount', 'totalamount', 'rate', 'dose', 'doseamount', 'dose_val_rx',
    'volume', 'chartvalue', 'resultvalue', 'value1', 'value2', 'value3',
    'drugname', 'amountuom'
]

# 🚀 表特定的最小列集 - 只加载必要的列以提升性能
MINIMAL_COLUMNS_MAP = {
    # MIMIC-IV chartevents: 只需要6列而非全部11列
    # 包含value列以支持字符串型数据（如药物名称等）
    'chartevents': ['stay_id', 'charttime', 'itemid', 'value', 'valuenum', 'valueuom'],
    
    # MIMIC-IV labevents: 只需要5列而非全部16列  
    # 注意: labevents没有stay_id，需要subject_id+hadm_id后续关联
    'labevents': ['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum'],
    
    # MIMIC-IV inputevents: 输入事件的核心列
    # 包含hadm_id用于某些需要住院级别聚合的概念（如abx）
    'inputevents': ['stay_id', 'hadm_id', 'starttime', 'endtime', 'itemid', 'amount', 'amountuom', 'rate', 'linkorderid'],
    
    # MIMIC-IV outputevents: 输出事件的核心列
    'outputevents': ['stay_id', 'charttime', 'itemid', 'value'],
    
    # MIMIC-IV procedureevents: 操作事件的核心列
    'procedureevents': ['stay_id', 'starttime', 'endtime', 'itemid', 'value'],
    
    # eICU vitalperiodic: 生命体征周期表
    'vitalperiodic': ['patientunitstayid', 'observationoffset', 'temperature', 'heartrate', 
                      'respiration', 'systemicsystolic', 'systemicdiastolic', 'systemicmean'],
    
    # eICU lab: 实验室检查
    'lab': ['patientunitstayid', 'labresultoffset', 'labname', 'labresult'],
}

# 性能优化开关 - 如果遇到问题可以禁用
USE_MINIMAL_COLUMNS = True


logger = logging.getLogger(__name__)


class ConceptLoader:
    """概念加载器 - 复刻 R ricu 的 load_concepts"""
    
    def __init__(self, src: Union[str, DataSource, DataSourceConfig]):
        """
        初始化概念加载器
        
        Args:
            src: 数据源名称或 DataSource 对象
        """
        self._data_source: Optional[ICUDataSource] = None
        if isinstance(src, ICUDataSource):
            self._data_source = src
            self.src = src.config
        elif isinstance(src, DataSourceConfig):
            self.src = src
        elif isinstance(src, str):
            self.src = load_src_cfg(src)
        else:
            raise TypeError(f"不支持的数据源类型: {type(src)}")
        self._src_name = self.src.name
        self._id_lookup_cache: Optional[pd.DataFrame] = None
    
    def _get_table_config(self, table_name: Optional[str]) -> Optional[TableConfig]:
        """根据表名获取配置。"""
        if not table_name or not hasattr(self.src, 'tables'):
            return None
        return self.src.tables.get(table_name)
    
    def _infer_required_columns(
        self,
        table_name: Optional[str],
        id_type: str,
        extra_candidates: Optional[Sequence[str]] = None,
    ) -> Optional[List[str]]:
        """根据表配置和概念需求推断需要加载的列 - 优化版，只加载必要列"""
        
        # 🚀 性能优化：优先使用最小列集（减少50-70%的I/O）
        if USE_MINIMAL_COLUMNS and table_name in MINIMAL_COLUMNS_MAP:
            base_cols = list(MINIMAL_COLUMNS_MAP[table_name])
            
            # 添加额外需要的列（如sub_var, val_var等）
            if extra_candidates:
                for col in extra_candidates:
                    if col and col not in base_cols:
                        base_cols.append(col)
            
            # 确保有ID列
            has_id = any(id_col in base_cols for id_col in 
                        ['stay_id', 'icustay_id', 'subject_id', 'patientunitstayid', 'hadm_id'])
            if not has_id:
                # 添加ID类型对应的列
                id_candidates = ID_TYPE_HINTS.get(id_type, ['stay_id'])
                base_cols.insert(0, id_candidates[0])
            
            return base_cols
        
        # 回退到原有逻辑（用于不在最小列集映射中的表，如icustays等）
        table_cfg = self._get_table_config(table_name)
        defaults = table_cfg.defaults if table_cfg else None
        available = (
            set(table_cfg.columns.keys())
            if table_cfg and table_cfg.columns
            else None
        )
        
        candidates: List[str] = []
        if defaults:
            if defaults.id_var:
                candidates.append(defaults.id_var)
            if defaults.index_var:
                candidates.append(defaults.index_var)
            if defaults.val_var:
                candidates.append(defaults.val_var)
            if defaults.unit_var:
                candidates.append(defaults.unit_var)
            candidates.extend(defaults.time_vars or [])
        
        if extra_candidates:
            candidates.extend(extra_candidates)
        
        # ID 列和通用列候选
        candidates.extend(ID_TYPE_HINTS.get(id_type, []))
        candidates.extend(COMMON_ID_COLUMNS)
        candidates.extend(COMMON_TIME_COLUMNS)
        candidates.extend(COMMON_VALUE_COLUMNS)
        
        filtered: List[str] = []
        seen: set[str] = set()
        for col in candidates:
            if not col or col in seen:
                continue
            if available is not None and col not in available:
                continue
            filtered.append(col)
            seen.add(col)
        
        return filtered or None
    
    def _safe_load_table(
        self,
        table_name: str,
        columns: Optional[Iterable[str]],
    ) -> pd.DataFrame:
        """在列过滤失败时回退到全表加载。"""
        if columns:
            try:
                return load_table(self._src_name, table_name, columns=list(columns))
            except Exception:
                # 回退到加载全部列，确保兼容缺少列描述的表
                return load_table(self._src_name, table_name)
        return load_table(self._src_name, table_name)
    
    def _columns_for_source(self, source, id_type: str) -> Optional[List[str]]:
        """提取 ConceptSource 所需的列。"""
        extra: List[str] = []
        if getattr(source, 'sub_var', None):
            extra.append(source.sub_var)
        if getattr(source, 'value_var', None):
            extra.append(source.value_var)
        if getattr(source, 'index_var', None):
            extra.append(source.index_var)
        if getattr(source, 'unit_var', None):
            extra.append(source.unit_var)
        
        result = self._infer_required_columns(source.table, id_type, extra)
        return result
    
    def _columns_for_item(self, item: Mapping[str, Any], id_type: str) -> Optional[List[str]]:
        """提取旧式 item 配置所需列。"""
        extra: List[str] = []
        for key in ['sub_var', 'val_var', 'value_var', 'time_var', 'index_var']:
            value = item.get(key)
            if isinstance(value, str):
                extra.append(value)
        return self._infer_required_columns(item.get('table'), id_type, extra)

    def _canonical_id_column(self, id_type: str) -> str:
        """根据数据源配置返回指定ID类型的标准列名。"""
        cfg = self.src.id_configs.get(id_type) if hasattr(self.src, 'id_configs') else None
        if cfg and getattr(cfg, 'id', None):
            return cfg.id
        fallback = {
            'icustay': 'stay_id',
            'hadm': 'hadm_id',
            'patient': 'subject_id',
        }
        return fallback.get(id_type, id_type)

    def _coerce_patient_list(self, patient_ids: Union[List, Sequence, set, pd.Series, pd.DataFrame, None]) -> List[Any]:
        """将 patient_ids 归一化为简单列表。"""
        if patient_ids is None:
            return []
        values: List[Any] = []
        if isinstance(patient_ids, pd.DataFrame):
            for column in patient_ids.columns:
                col_vals = patient_ids[column].tolist()
                for value in col_vals:
                    if pd.isna(value):
                        continue
                    values.append(value)
            return values
        if isinstance(patient_ids, pd.Series):
            return [value for value in patient_ids.tolist() if not pd.isna(value)]
        if isinstance(patient_ids, (list, tuple, set)):
            return [value for value in patient_ids if not pd.isna(value)]
        return [patient_ids] if not pd.isna(patient_ids) else []

    def _load_id_lookup(self) -> pd.DataFrame:
        """加载包含 stay/hadm/subject 映射的参考表，用于ID转换。"""
        if self._id_lookup_cache is not None:
            return self._id_lookup_cache

        cfg = getattr(self.src, 'id_configs', {}).get('icustay') if hasattr(self.src, 'id_configs') else None
        table_name = cfg.table if cfg and getattr(cfg, 'table', None) else None
        if not table_name:
            self._id_lookup_cache = pd.DataFrame()
            return self._id_lookup_cache

        desired_cols = {
            'stay_id', 'icustay_id', 'patientunitstayid', 'hadm_id', 'subject_id',
            'admissionid', 'patientid', 'patient_id', 'admission_id'
        }
        for id_cfg in getattr(self.src, 'id_configs', {}).values():
            if getattr(id_cfg, 'id', None):
                desired_cols.add(id_cfg.id)

        table_cfg = self._get_table_config(table_name)
        available = set(table_cfg.columns.keys()) if table_cfg and table_cfg.columns else None
        columns = [col for col in desired_cols if (available is None or col in available)]
        columns = columns or None

        try:
            lookup = self._safe_load_table(table_name, columns)
        except Exception as exc:
            logger.warning("无法加载ID映射表 %s: %s", table_name, exc)
            lookup = pd.DataFrame()

        self._id_lookup_cache = lookup
        return lookup

    def _map_patient_ids_to_column(
        self,
        patient_ids: List[Any],
        id_type: str,
        target_column: Optional[str],
    ) -> Optional[List[Any]]:
        """将基于 id_type 的 patient_ids 映射到目标列的取值集合。"""
        if target_column is None:
            return patient_ids
        canonical_col = self._canonical_id_column(id_type)
        if canonical_col.lower() == target_column.lower():
            return patient_ids
        lookup = self._load_id_lookup()
        if lookup.empty or canonical_col not in lookup.columns or target_column not in lookup.columns:
            return None
        if not patient_ids:
            return []
        subset = lookup[lookup[canonical_col].isin(patient_ids)]
        if subset.empty:
            return []
        mapped = subset[target_column].dropna().unique().tolist()
        return mapped
            
    def load_concepts(
        self,
        concepts: Union[str, List[str], Concept, List[Concept]],
        patient_ids: Optional[Union[List, pd.DataFrame]] = None,
        id_type: str = 'icustay',
        interval: Optional[timedelta] = None,
        aggregate: Optional[Union[str, Dict[str, str]]] = None,
        merge_data: bool = True,
        verbose: bool = True,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        加载概念数据
        
        Args:
            concepts: 概念名称、ID或Concept对象
            patient_ids: 患者ID列表或包含ID的DataFrame
            id_type: ID类型 (patient, hadm, icustay等)
            interval: 时间间隔 (如 timedelta(hours=1))
            aggregate: 聚合函数 ('mean', 'sum', 'min', 'max' 或字典)
            merge_data: 是否合并为宽格式表
            verbose: 是否显示进度信息
            
        Returns:
            DataFrame 或字典 (取决于 merge_data)
        """
        # 1. 解析概念
        if isinstance(concepts, str):
            concepts = [concepts]
        
        if isinstance(concepts, list) and all(isinstance(c, str) for c in concepts):
            # 从字典加载概念
            # 如果请求的概念中包含 SOFA-2 相关概念，自动加载 sofa2-dict
            sofa2_concepts = {'sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 
                              'sofa2_cardio', 'sofa2_cns', 'sofa2_renal',
                              'uo_6h', 'uo_12h', 'uo_24h', 'rrt_criteria',
                              'adv_resp', 'ecmo', 'ecmo_indication', 'sedated_gcs',
                              'mech_circ_support', 'other_vaso', 'delirium_tx'}
            include_sofa2 = any(c in sofa2_concepts for c in concepts)
            
            concept_dict = load_dictionary(self._src_name, include_sofa2=include_sofa2)
            concept_objs = [concept_dict[name] for name in concepts]
        elif isinstance(concepts, Concept):
            concept_objs = [concepts]
        elif isinstance(concepts, list) and all(isinstance(c, Concept) for c in concepts):
            concept_objs = concepts
        else:
            raise ValueError(f"不支持的概念类型: {type(concepts)}")
        
        # 2. 设置默认值
        if interval is None:
            interval = timedelta(hours=1)
        
        # 3. 加载每个概念
        results = {}
        for concept in concept_objs:
            if verbose:
                print(f"加载概念: {concept.name}")
            
            # 加载单个概念
            data = self._load_one_concept(
                concept=concept,
                patient_ids=patient_ids,
                id_type=id_type,
                interval=interval,
                aggregate=aggregate if not isinstance(aggregate, dict) else aggregate.get(concept.name),
                **kwargs
            )
            
            if data is not None and len(data) > 0:
                results[concept.name] = data
        
        # 4. 合并或返回
        if not merge_data:
            return results
        
        if len(results) == 0:
            return pd.DataFrame()
        
        if len(results) == 1:
            return list(results.values())[0]
        
        # 合并多个概念为宽格式
        return self._merge_concepts(results, id_type)
    
    def _load_one_concept(
        self,
        concept: Concept,
        patient_ids: Optional[Union[List, pd.DataFrame]],
        id_type: str,
        interval: timedelta,
        aggregate: Optional[str],
        **kwargs
    ) -> pd.DataFrame:
        """
        加载单个概念
        
        Args:
            concept: Concept 对象
            patient_ids: 患者ID
            id_type: ID类型
            interval: 时间间隔
            aggregate: 聚合函数
            
        Returns:
            DataFrame
        """
        # 检查是否为递归概念（有子概念）
        if concept.sub_concepts and len(concept.sub_concepts) > 0:
            # 递归概念 - 使用回调
            return self._load_recursive_concept(
                concept, patient_ids, id_type, interval, aggregate, **kwargs
            )
        
        # 2. 普通概念 - 从表中加载
        # 获取当前数据源的 ConceptSource 配置
        sources = concept.for_data_source(self.src)
        if not sources:
            return pd.DataFrame()
        
        all_data = []
        
        for source in sources:
            # 加载source数据
            df = self._load_concept_source(
                source=source,
                concept_name=concept.name,
                patient_ids=patient_ids,
                id_type=id_type,
                interval=interval
            )
            
            if df is not None and len(df) > 0:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        # 3. 合并所有source数据
        data = pd.concat(all_data, ignore_index=True)
        
        # 4. 过滤和转换
        data = self._filter_concept_data(data, concept)
        
        # 5. 重命名列
        if 'value' in data.columns:
            data = data.rename(columns={'value': concept.name})
        
        # 6. 聚合
        if aggregate and len(data) > 0:
            data = self._aggregate_concept(data, concept, aggregate, id_type, interval)
        
        return data
    
    def _load_item(
        self,
        item: Dict[str, Any],
        patient_ids: Optional[Union[List, pd.DataFrame]],
        id_type: str,
        interval: timedelta
    ) -> pd.DataFrame:
        """
        加载单个item
        
        Args:
            item: item字典
            patient_ids: 患者ID
            id_type: ID类型
            interval: 时间间隔
            
        Returns:
            DataFrame
        """
        # 1. 加载表
        table_name = item.get('table')
        if not table_name:
            return pd.DataFrame()
        
        required_columns = self._columns_for_item(item, id_type)
        
        # 🔍 调试：显示推断的列
        if required_columns:
            import logging
            logger = logging.getLogger('pyricu.load_concepts')
            if logger.isEnabledFor(logging.DEBUG):
        
        try:
            df = self._safe_load_table(table_name, required_columns)
        except Exception as e:
            print(f"警告: 无法加载表 {table_name}: {e}")
            return pd.DataFrame()
        
        # 2. 过滤患者
        if patient_ids is not None:
            id_col = self._get_id_column(df, id_type)
            if id_col:
                filter_values: Optional[List[Any]] = None
                if isinstance(patient_ids, pd.DataFrame):
                    if id_col in patient_ids.columns:
                        filter_values = [val for val in patient_ids[id_col].tolist() if not pd.isna(val)]
                    else:
                        canonical_col = self._canonical_id_column(id_type)
                        if canonical_col in patient_ids.columns:
                            base_values = [val for val in patient_ids[canonical_col].tolist() if not pd.isna(val)]
                            filter_values = self._map_patient_ids_to_column(base_values, id_type, id_col)
                else:
                    base_values = self._coerce_patient_list(patient_ids)
                    filter_values = self._map_patient_ids_to_column(base_values, id_type, id_col)

                if filter_values is not None:
                    if not filter_values:
                        return pd.DataFrame()
                    df = df[df[id_col].isin(filter_values)]
        
        # 3. 过滤item值
        val_col = item.get('val_var', 'value')
        sub_col = item.get('sub_var')
        
        if sub_col and sub_col in df.columns:
            # 过滤特定值
            target_vals = item.get('target', [])
            if target_vals:
                df = df[df[sub_col].isin(target_vals)]
        
        # 4. 选择需要的列
        required_cols = [self._get_id_column(df, id_type)]
        
        # 时间列
        time_col = self._get_time_column(df)
        if time_col:
            required_cols.append(time_col)
        
        # 值列
        if val_col in df.columns:
            required_cols.append(val_col)
        
        # 过滤列
        required_cols = [c for c in required_cols if c and c in df.columns]
        df = df[required_cols].copy()
        
        # 5. 重命名为标准列名
        rename_map = {}
        if time_col and time_col != 'time':
            rename_map[time_col] = 'time'
        if val_col and val_col != 'value':
            rename_map[val_col] = 'value'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # 6. 对齐时间间隔
        if 'time' in df.columns and interval:
            df = change_interval(df, interval=interval, time_col='time')
        
        return df
    
    def _load_concept_source(
        self,
        source,  # ConceptSource object
        concept_name: str,
        patient_ids: Optional[Union[List, pd.DataFrame]],
        id_type: str,
        interval: timedelta
    ) -> pd.DataFrame:
        """
        从 ConceptSource 加载数据
        
        Args:
            source: ConceptSource 对象
            concept_name: 概念名称
            patient_ids: 患者ID
            id_type: ID类型
            interval: 时间间隔
            
        Returns:
            DataFrame
        """
        # 1. 加载表
        table_name = source.table
        if not table_name:
            return pd.DataFrame()
        
        required_columns = self._columns_for_source(source, id_type)
        try:
            df = self._safe_load_table(table_name, required_columns)
        except Exception as e:
            print(f"警告: 无法加载表 {table_name}: {e}")
            return pd.DataFrame()
        
        # 2. 过滤 sub_var (如 itemid)
        if source.sub_var and source.ids:
            if source.sub_var not in df.columns:
                print(f"警告: 表 {table_name} 中找不到列 {source.sub_var}")
                return pd.DataFrame()
            df = df[df[source.sub_var].isin(source.ids)]
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # 3. 过滤患者
        if patient_ids is not None:
            id_col = self._get_id_column(df, id_type)
            if id_col:
                filter_values: Optional[List[Any]] = None
                if isinstance(patient_ids, pd.DataFrame):
                    if id_col in patient_ids.columns:
                        filter_values = [val for val in patient_ids[id_col].tolist() if not pd.isna(val)]
                    else:
                        canonical_col = self._canonical_id_column(id_type)
                        if canonical_col in patient_ids.columns:
                            base_values = [val for val in patient_ids[canonical_col].tolist() if not pd.isna(val)]
                            filter_values = self._map_patient_ids_to_column(base_values, id_type, id_col)
                else:
                    base_values = self._coerce_patient_list(patient_ids)
                    filter_values = self._map_patient_ids_to_column(base_values, id_type, id_col)

                if filter_values is not None:
                    if not filter_values:
                        return pd.DataFrame()
                    df = df[df[id_col].isin(filter_values)]
        
        # 4. 确定值列
        val_col = source.value_var or 'valuenum'  # 默认使用 valuenum
        if val_col not in df.columns:
            # 尝试其他可能的值列
            for candidate in ['valuenum', 'value', 'amount']:
                if candidate in df.columns:
                    val_col = candidate
                    break
        
        # 5. 选择需要的列
        id_col = self._get_id_column(df, id_type)
        required_cols = [id_col] if id_col else []
        
        # 时间列
        time_col = source.index_var or self._get_time_column(df)
        if time_col and time_col in df.columns:
            required_cols.append(time_col)
        
        # 值列
        if val_col and val_col in df.columns:
            required_cols.append(val_col)
        
        # 过滤列
        required_cols = [c for c in required_cols if c and c in df.columns]
        if not required_cols:
            return pd.DataFrame()
        
        df = df[required_cols].copy()
        
        # 6. 重命名为标准列名
        rename_map = {}
        if time_col and time_col != 'time' and time_col in df.columns:
            rename_map[time_col] = 'time'
        if val_col and val_col != 'value' and val_col in df.columns:
            rename_map[val_col] = 'value'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # 7. 对齐时间间隔
        if 'time' in df.columns and interval:
            df = change_interval(df, interval=interval, time_col='time')
        
        return df
    
    def _load_recursive_concept(
        self,
        concept: Concept,
        patient_ids: Optional[Union[List, pd.DataFrame]],
        id_type: str,
        interval: timedelta,
        aggregate: Optional[str],
        **kwargs
    ) -> pd.DataFrame:
        """
        加载递归概念（使用回调）- 修复循环依赖检测
        
        完全复刻 R ricu 的递归概念加载逻辑，包括：
        1. 循环依赖检测
        2. 依赖解析缓存
        3. 正确的子概念加载顺序
        
        Args:
            concept: Concept对象
            patient_ids: 患者ID
            id_type: ID类型
            interval: 时间间隔
            aggregate: 聚合函数
            
        Returns:
            DataFrame
            
        Raises:
            ValueError: 如果检测到循环依赖
        """
        # 初始化加载栈（用于检测循环依赖）
        if not hasattr(self, '_loading_stack'):
            self._loading_stack = set()
        
        # 初始化缓存（避免重复加载相同概念）
        if not hasattr(self, '_concept_cache'):
            self._concept_cache = {}
        
        # 检查循环依赖
        if concept.name in self._loading_stack:
            chain = ' -> '.join(self._loading_stack) + f' -> {concept.name}'
            raise ValueError(f"检测到循环依赖: {chain}")
        
        # 检查缓存
        cache_key = (
            concept.name, 
            str(patient_ids) if patient_ids is not None else None,
            id_type,
            str(interval),
            aggregate
        )
        if cache_key in self._concept_cache:
            return self._concept_cache[cache_key].copy()
        
        # 将当前概念加入加载栈
        self._loading_stack.add(concept.name)
        
        try:
            # 1. 加载子概念
            sub_concepts = concept.items if hasattr(concept, 'items') else {}
            sub_data = {}
            
            # 按照依赖顺序加载子概念
            for sub_name in sub_concepts:
                try:
                    # 获取子概念定义
                    if isinstance(sub_concepts[sub_name], Concept):
                        sub_concept = sub_concepts[sub_name]
                    else:
                        # 从字典中加载
                        concept_dict = load_dictionary(self._src_name)
                        if sub_name not in concept_dict:
                            print(f"警告: 找不到子概念 {sub_name}")
                            continue
                        sub_concept = concept_dict[sub_name]
                    
                    # 递归加载子概念
                    data = self._load_one_concept(
                        sub_concept, patient_ids, id_type, interval, aggregate, **kwargs
                    )
                    
                    if data is not None and len(data) > 0:
                        sub_data[sub_name] = data
                        
                except Exception as e:
                    print(f"警告: 加载子概念 {sub_name} 失败: {e}")
                    continue
            
            if not sub_data:
                result = pd.DataFrame()
            else:
                # 2. 应用回调函数
                callback = concept.callback if hasattr(concept, 'callback') else None
                
                if callback:
                    # 构建回调函数并应用
                    if callable(callback):
                        result = callback(sub_data, interval=interval, src=self.src, **kwargs)
                    else:
                        # 如果是字符串或其他类型，尝试从callback_utils构建
                        from .callback_utils import build_callback
                        cb_func = build_callback(callback)
                        result = cb_func(sub_data, interval=interval, src=self.src, **kwargs)
                else:
                    # 如果没有回调，尝试简单合并
                    if len(sub_data) == 1:
                        result = list(sub_data.values())[0]
                    else:
                        # 多个子概念，需要合并
                        result = self._merge_sub_concepts(sub_data, id_type, interval)
            
            # 缓存结果
            self._concept_cache[cache_key] = result.copy() if len(result) > 0 else result
            
            return result
            
        finally:
            # 从加载栈中移除当前概念
            self._loading_stack.discard(concept.name)
    
    def _filter_concept_data(self, data: pd.DataFrame, concept: Concept) -> pd.DataFrame:
        """
        根据概念定义过滤数据
        
        Args:
            data: 原始数据
            concept: 概念对象
            
        Returns:
            过滤后的数据
        """
        if 'value' not in data.columns:
            return data
        
        # 1. 过滤NA
        data = data.dropna(subset=['value'])
        
        # 2. 数值范围过滤
        if hasattr(concept, 'min') and concept.min is not None:
            data = data[data['value'] >= concept.min]
        
        if hasattr(concept, 'max') and concept.max is not None:
            data = data[data['value'] <= concept.max]
        
        # 3. 分类值过滤
        if hasattr(concept, 'levels') and concept.levels:
            data = data[data['value'].isin(concept.levels)]
        
        # 4. 单位转换（如果需要）
        if hasattr(concept, 'unit') and concept.unit and 'unit' in data.columns:
            data = self._convert_units(data, concept.unit)
        
        return data
    
    def _convert_units(self, data: pd.DataFrame, target_unit: str) -> pd.DataFrame:
        """
        单位转换
        
        Args:
            data: 数据
            target_unit: 目标单位
            
        Returns:
            转换后的数据
        """
        # TODO: 实现完整的单位转换系统
        # 这里先做简单处理
        return data
    
    def _aggregate_concept(
        self,
        data: pd.DataFrame,
        concept: Concept,
        aggregate: str,
        id_type: str,
        interval: timedelta
    ) -> pd.DataFrame:
        """
        聚合概念数据
        
        Args:
            data: 数据
            concept: 概念
            aggregate: 聚合函数名
            id_type: ID类型
            interval: 时间间隔
            
        Returns:
            聚合后的数据
        """
        id_col = self._get_id_column(data, id_type)
        
        group_cols = [id_col]
        if 'time' in data.columns:
            group_cols.append('time')
        
        value_col = concept.name
        
        # 执行聚合
        agg_dict = {value_col: aggregate}
        result = data.groupby(group_cols, as_index=False).agg(agg_dict)
        
        return result
    
    def _merge_concepts(
        self,
        results: Dict[str, pd.DataFrame],
        id_type: str
    ) -> pd.DataFrame:
        """
        合并多个概念为宽格式
        
        Args:
            results: 概念名 -> DataFrame 字典
            id_type: ID类型
            
        Returns:
            合并后的宽格式DataFrame
        """
        if not results:
            return pd.DataFrame()
        
        # 找出公共列
        first_df = list(results.values())[0]
        id_col = self._get_id_column(first_df, id_type)
        
        merge_cols = [id_col]
        if 'time' in first_df.columns:
            merge_cols.append('time')
        
        # 逐步合并
        merged = None
        for name, df in results.items():
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on=merge_cols, how='outer')
        
        return merged
    
    def _merge_sub_concepts(
        self,
        sub_data: Dict[str, pd.DataFrame],
        id_type: str,
        interval: timedelta
    ) -> pd.DataFrame:
        """
        合并多个子概念数据
        
        Args:
            sub_data: 子概念数据字典
            id_type: ID类型
            interval: 时间间隔
            
        Returns:
            合并后的DataFrame
        """
        if not sub_data:
            return pd.DataFrame()
        
        if len(sub_data) == 1:
            return list(sub_data.values())[0]
        
        # 确定ID列和时间列
        id_col = self._determine_id_column(id_type)
        merge_cols = [id_col]
        
        # 检查是否有时间列
        first_df = list(sub_data.values())[0]
        if 'time' in first_df.columns:
            merge_cols.append('time')
        
        # 逐步合并
        result = None
        for name, df in sub_data.items():
            if result is None:
                result = df.copy()
            else:
                result = result.merge(df, on=merge_cols, how='outer', suffixes=('', f'_{name}'))
        
        return result
    
    def _determine_id_column(self, id_type: str) -> str:
        """
        根据ID类型确定ID列名
        
        Args:
            id_type: ID类型
            
        Returns:
            ID列名
        """
        # 数据源特定的ID列名映射
        id_mappings = {
            'mimic_demo': {
                'icustay': 'stay_id',
                'hadm': 'hadm_id',
                'subject': 'subject_id',
            },
            'mimic': {
                'icustay': 'stay_id',
                'hadm': 'hadm_id',
                'subject': 'subject_id',
            },
            'eicu_demo': {
                'icustay': 'patientunitstayid',
                'hadm': 'patienthealthsystemstayid',
                'subject': 'uniquepid',
            },
            'eicu': {
                'icustay': 'patientunitstayid',
                'hadm': 'patienthealthsystemstayid',
                'subject': 'uniquepid',
            },
        }
        
        src_name = self._src_name
        
        if src_name in id_mappings and id_type in id_mappings[src_name]:
            return id_mappings[src_name][id_type]
        
        # 默认返回 stay_id
        return 'stay_id'
    
    def clear_cache(self):
        """清除概念加载缓存"""
        if hasattr(self, '_concept_cache'):
            self._concept_cache.clear()
        if hasattr(self, '_loading_stack'):
            self._loading_stack.clear()
    
    def _get_id_column(self, df: pd.DataFrame, id_type: str) -> Optional[str]:
        """
        获取ID列名
        
        Args:
            df: DataFrame
            id_type: ID类型
            
        Returns:
            列名或None
        """
        # 常见的ID列名映射
        id_mappings = {
            'patient': ['subject_id', 'patientid', 'patient_id'],
            'hadm': ['hadm_id', 'admissionid', 'admission_id'],
            'icustay': ['icustay_id', 'stay_id', 'patientunitstayid'],
        }
        
        possible_names = id_mappings.get(id_type, [id_type])
        
        for col in df.columns:
            if col.lower() in [n.lower() for n in possible_names]:
                return col
        
        # 返回第一个包含'id'的列
        for col in df.columns:
            if 'id' in col.lower():
                return col
        
        return None
    
    def _get_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        获取时间列名
        
        Args:
            df: DataFrame
            
        Returns:
            列名或None
        """
        time_cols = ['charttime', 'time', 'datetime', 'timestamp', 
                     'starttime', 'observationoffset']
        
        for col in df.columns:
            if col.lower() in [t.lower() for t in time_cols]:
                return col
        
        return None


def load_concepts(
    concepts: Union[str, List[str]],
    src: Union[str, DataSource],
    **kwargs
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    便捷函数：加载概念
    
    Args:
        concepts: 概念名称或列表
        src: 数据源
        **kwargs: 传递给 ConceptLoader.load_concepts
        
    Returns:
        DataFrame 或字典
    
    Examples:
        >>> # 加载单个概念
        >>> hr = load_concepts('hr', 'mimic')
        >>> 
        >>> # 加载多个概念并合并
        >>> vitals = load_concepts(['hr', 'sbp', 'dbp'], 'mimic', 
        ...                        interval=timedelta(hours=1))
    """
    loader = ConceptLoader(src)
    return loader.load_concepts(concepts, **kwargs)
