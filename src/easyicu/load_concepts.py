"""
完整的概念加载系统
实现 R ricu 的 load_concepts 功能
"""
from typing import List, Optional, Union, Dict, Any, Iterable, Sequence, Mapping
import logging
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .concept import Concept, load_dictionary
from .config import DataSourceConfig, TableConfig, load_src_cfg
from .datasource import ICUDataSource
from .table import load_table
from .io.ts_utils import change_interval

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
    
    # 🔧 FIX 2026-01-26: MIMIC-III 使用 icustay_id 而非 stay_id
    # 系统会自动根据数据库类型选择正确的列（见 datasource.py 中的列检测逻辑）
    'chartevents_mimic': ['icustay_id', 'charttime', 'itemid', 'value', 'valuenum', 'valueuom'],
    
    # MIMIC-IV labevents: 只需要6列而非全部16列  
    # 注意: labevents没有stay_id，需要subject_id+hadm_id后续关联
    # 包含valueuom用于单位转换回调（如CRP的mg/dL转mg/L）
    'labevents': ['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum', 'valueuom'],
    
    # 🔧 FIX 2026-01-26: MIMIC-III labevents 使用 icustay_id（可能为空）
    'labevents_mimic': ['subject_id', 'hadm_id', 'icustay_id', 'charttime', 'itemid', 'valuenum', 'valueuom'],
    
    # MIMIC-IV inputevents: 输入事件的核心列
    # 包含hadm_id用于某些需要住院级别聚合的概念（如abx）
    'inputevents': ['stay_id', 'hadm_id', 'starttime', 'endtime', 'itemid', 'amount', 'amountuom', 'rate', 'linkorderid'],
    
    # MIMIC-IV outputevents: 输出事件的核心列
    'outputevents': ['stay_id', 'charttime', 'itemid', 'value'],
    
    # MIMIC-IV procedureevents: 操作事件的核心列
    'procedureevents': ['stay_id', 'starttime', 'endtime', 'itemid', 'value'],
    
    # eICU vitalperiodic: 生命体征周期表
    # 🔧 FIX: 添加 sao2 列用于 o2sat 和 spo2 概念
    'vitalperiodic': ['patientunitstayid', 'observationoffset', 'temperature', 'heartrate', 
                      'respiration', 'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'sao2'],
    
    # eICU lab: 实验室检查
    # 包含labmeasurenameinterface用于单位转换回调（如calcium的mmol/l转mg/dL）
    'lab': ['patientunitstayid', 'labresultoffset', 'labname', 'labresult', 'labmeasurenameinterface'],
    
    # AUMC numericitems: 数值项目表 - 包含 measuredat 时间列
    # AUMC 时间单位: measuredat 是毫秒，需要减去 admittedat 并转换为小时
    # 🔧 FIX: 添加 tag 列用于 aumc_bxs 回调（be 概念需要根据 tag='-' 取反值）
    'numericitems': ['admissionid', 'itemid', 'value', 'unit', 'measuredat', 'tag'],
    
    # 注意：admissions 表不同数据库列名不同，不纳入优化
    # AUMC: admissionid, patientid, admittedat, dischargedat, destination
    # MIIV: hadm_id, subject_id, admittime, dischtime, deathtime, hospital_expire_flag
    # 因此不在此处配置，让系统加载所有列
}

# 性能优化开关 - 如果遇到问题可以禁用
USE_MINIMAL_COLUMNS = True

logger = logging.getLogger(__name__)


def _is_missing_column_projection_error(exc: Exception) -> bool:
    """Return whether *exc* explicitly reports a projected column is absent."""
    message = str(exc).lower()
    if isinstance(exc, KeyError):
        return "columns " in message and " not found in table " in message
    if isinstance(exc, ValueError):
        return "usecols do not match columns" in message
    return (
        type(exc).__name__ == "ArrowInvalid"
        and (
            "no match for fieldref" in message
            or ("field" in message and "not found" in message)
        )
    )


class ConceptLoader:
    """概念加载器 - 复刻 R ricu 的 load_concepts"""
    
    def __init__(self, src: Union[str, DataSource, DataSourceConfig], data_path: Optional[str] = None, low_memory: Optional[bool] = None):
        """
        初始化概念加载器
        
        Args:
            src: 数据源名称或 DataSource 对象
            data_path: 数据路径
            low_memory: 低内存模式（None=自动检测，True=强制启用，False=禁用）
                        低内存模式使用 DuckDB filter pushdown，避免加载全表
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
        self.data_path = data_path
        
        # 🚀 低内存模式：自动检测或手动指定
        # HiRID 强制使用低内存模式（observations 表有 7.77 亿行）
        if self._src_name == 'hirid':
            self._low_memory = True
        elif low_memory is not None:
            self._low_memory = low_memory
        else:
            # 自动检测：可用内存 < 24GB 时启用低内存模式
            try:
                import psutil
                available_gb = psutil.virtual_memory().available / (1024 ** 3)
                self._low_memory = available_gb < 24
            except ImportError:
                self._low_memory = False
        
        # 🚀 低内存模式下，自动创建 ICUDataSource 以启用 filter pushdown
        if self._low_memory and self._data_source is None and data_path:
            try:
                self._data_source = ICUDataSource(self.src, base_path=data_path)
                logger.info(f"🧠 低内存模式启用 for {self._src_name}")
            except Exception as e:
                logger.warning(f"无法创建 ICUDataSource: {e}")
        
        self._id_lookup_cache: Optional[pd.DataFrame] = None
        self._table_cache: Dict[str, pd.DataFrame] = {}
    
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
            
            # 确保有ID列 - 包括 AUMC 的 admissionid
            has_id = any(id_col in base_cols for id_col in 
                        ['stay_id', 'icustay_id', 'subject_id', 'patientunitstayid', 'hadm_id', 'admissionid'])
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
        """Load a projection, with a tightly bounded compatibility fallback.

        Only an explicit missing-column projection error may retry without a
        projection. I/O, corruption, permission and memory failures propagate
        unchanged. Low-memory mode never retries with a full-table read.
        """
        # A cached narrow projection is reusable only when it contains every
        # column requested by the next concept.  Reusing by table name alone
        # made results depend on concept order when callbacks needed extra
        # source columns.
        cached = self._table_cache.get(table_name)
        requested_columns = list(dict.fromkeys(columns or []))
        if cached is not None and (
            not requested_columns
            or set(requested_columns).issubset(cached.columns)
        ):
            return cached

        # 🚀 加载表并存入缓存
        df = None
        if requested_columns:
            try:
                df = load_table(
                    self._src_name,
                    table_name,
                    columns=requested_columns,
                    path=self.data_path,
                )
            except Exception as exc:
                if self._low_memory or not _is_missing_column_projection_error(exc):
                    raise
                logger.warning(
                    "Table %s has an incomplete column descriptor; retrying "
                    "without projection outside low-memory mode",
                    table_name,
                )
                df = load_table(self._src_name, table_name, path=self.data_path)
        else:
            df = load_table(self._src_name, table_name, path=self.data_path)
        
        # 🚀 存入缓存以供后续复用
        if df is not None:
            self._table_cache[table_name] = df
        
        return df
    
    def _columns_for_source(self, source, id_type: str) -> Optional[List[str]]:
        """提取 ConceptSource 所需的列。"""
        extra: List[str] = []
        if getattr(source, 'sub_var', None):
            extra.append(source.sub_var)
        if getattr(source, 'value_var', None):
            extra.append(source.value_var)
        if getattr(source, 'index_var', None):
            extra.append(source.index_var)
        if getattr(source, 'dur_var', None):
            extra.append(source.dur_var)
        if getattr(source, 'unit_var', None):
            extra.append(source.unit_var)

        # Source callbacks may declare semantic input columns that are not a
        # table default (for example MetaVision ``statusdescription`` and
        # CareVue ``stopped``).  The main ConceptResolver already honours this
        # dictionary-owned contract; the compatibility ConceptLoader must not
        # silently project those columns away.
        params = getattr(source, 'params', None) or {}
        source_extra_vars = params.get('extra_vars', [])
        if isinstance(source_extra_vars, str):
            source_extra_vars = [source_extra_vars]
        if not isinstance(source_extra_vars, (list, tuple)):
            raise TypeError(
                "Concept source 'extra_vars' must be a string or a list of strings"
            )
        for extra_var in source_extra_vars:
            if not isinstance(extra_var, str) or not extra_var.strip():
                raise TypeError(
                    "Concept source 'extra_vars' entries must be non-empty strings"
                )
            extra.append(extra_var)

        # Duration callbacks also need their explicit end and grouping
        # columns.  These are params rather than ConceptSource attributes and
        # are not guaranteed to be part of a table's default projection.
        for param_name in ('stop_var', 'grp_var'):
            column = params.get(param_name)
            if isinstance(column, str) and column.strip():
                extra.append(column)

        extra = list(dict.fromkeys(extra))
        
        # DEBUG: 输出提取的列信息
        result = self._infer_required_columns(source.table, id_type, extra)
        logger.debug(f"_columns_for_source: table={source.table}, sub_var={getattr(source, 'sub_var', None)}, extra={extra}, result={result}")
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
        # numpy array support
        import numpy as np
        if isinstance(patient_ids, np.ndarray):
            return [v for v in patient_ids.tolist() if not pd.isna(v)]
        if isinstance(patient_ids, (list, tuple, set)):
            return [value for value in patient_ids if not pd.isna(value)]
        try:
            return [patient_ids] if not pd.isna(patient_ids) else []
        except (ValueError, TypeError):
            # pd.isna on array-like returns array, not scalar
            return list(patient_ids)

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
                              'sofa2_cardio', 'sofa2_cns',
                              'sofa2_cns_proxy_sensitivity',
                              'sofa2_cns_delirium_tx_ascertainment',
                              'sofa2_cns_ascertainment',
                              'sofa2_renal',
                              'uo_6h', 'uo_12h', 'uo_24h', 'rrt_criteria', 'rrt',
                              'adv_resp', 'ecmo', 'ecmo_indication', 'sedated_gcs',
                              'mech_circ_support', 'other_vaso', 'delirium_tx',
                              'delirium_tx_proxy', 'delirium_tx_evidence',
                              'motor_response', 'delirium_positive'}
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
        
        # 🚀 智能并行配置：根据概念数量自动优化
        user_concept_workers = kwargs.get('concept_workers')
        if user_concept_workers is None:
            # 自动计算最佳并行数
            import os
            num_concepts = len(concept_objs)
            if num_concepts >= 3:
                cpu_count = os.cpu_count() or 4
                parallel_workers = min(num_concepts, max(2, cpu_count // 2))
            elif num_concepts == 2:
                parallel_workers = 2
            else:
                parallel_workers = 1
        else:
            parallel_workers = user_concept_workers
            
        enable_parallel = len(concept_objs) > 1 and parallel_workers > 1
        
        # 🚀 Preload tables（优化：并行模式下更激进的预加载）
        self._preload_tables(concept_objs, patient_ids, id_type, verbose=verbose, 
                             parallel_mode=enable_parallel)
        
        # 3. 加载每个概念 - 支持并行加载
        results = {}
        
        if enable_parallel:
            # 🚀 并行加载概念
            max_workers = min(parallel_workers, len(concept_objs))
            if verbose:
                print(f"🚀 并行加载 {len(concept_objs)} 个概念 (工作线程: {max_workers})...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_concept = {
                    executor.submit(
                        self._load_one_concept,
                        concept=concept,
                        patient_ids=patient_ids,
                        id_type=id_type,
                        interval=interval,
                        aggregate=aggregate if not isinstance(aggregate, dict) else aggregate.get(concept.name),
                        **kwargs
                    ): concept
                    for concept in concept_objs
                }
                
                for future in as_completed(future_to_concept):
                    concept = future_to_concept[future]
                    try:
                        data = future.result()
                        if verbose:
                            print(f"  ✅ {concept.name}")
                        if data is not None and len(data) > 0:
                            results[concept.name] = data
                    except Exception as e:
                        if verbose:
                            print(f"  ❌ {concept.name}: {e}")
                        logger.error(f"加载概念 {concept.name} 失败", exc_info=True)
        else:
            # 串行加载（默认行为）
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
            # 转换时间列为相对小时数
            for name in results:
                results[name] = self._convert_time_to_hours(results[name], id_type)
            return results
        
        if len(results) == 0:
            return pd.DataFrame()
        
        if len(results) == 1:
            single_result = list(results.values())[0]
            return self._convert_time_to_hours(single_result, id_type)
        
        # 合并多个概念为宽格式
        merged = self._merge_concepts(results, id_type)
        return self._convert_time_to_hours(merged, id_type)
    
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
        # 🚀 优化：对于 rec_cncpt 类型的概念（如 vent_ind），
        # 直接使用 ConceptResolver，因为它们需要concept_callbacks中的回调函数
        if hasattr(concept, 'class_name') and concept.class_name == 'rec_cncpt':
            from .concept import ConceptResolver
            from .io.ts_utils import ICUTable
            
            # 🚀 重要：必须重用 ConceptLoader 的数据源，以便共享表缓存和预加载的数据
            # 如果创建新的数据源，预加载的表会丢失
            if self._data_source is None:
                raise RuntimeError("rec_cncpt concepts require a data source, but none is available")
            
            data_source = self._data_source
            
            # 创建 ConceptResolver（它会从数据源加载表）
            resolver = ConceptResolver(load_dictionary(self._src_name))
            
            # 使用 ConceptResolver 加载
            # 过滤掉 ConceptLoader 特有的参数和已经显式传递的参数
            excluded_kwargs = {'verbose', 'merge_data', 'id_type', 'merge', 'patient_ids', 'interval', 'aggregate', 'r_compatible'}
            resolver_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_kwargs}
            
            result = resolver.load_concepts(
                [concept.name],
                data_source=data_source,
                merge=False,
                patient_ids=patient_ids,
                interval=interval,
                aggregate=aggregate,
                verbose=kwargs.get('verbose', False),
                r_compatible=False,  # 🔧 FIX: 强制返回 dict[str, ICUTable]，以便正确提取数据
                **resolver_kwargs
            )
            
            # 提取DataFrame
            if isinstance(result, dict) and concept.name in result:
                result_table = result[concept.name]
                if isinstance(result_table, ICUTable):
                    return result_table.data
                return result_table
            # 🔧 FIX: 如果返回的是 DataFrame（r_compatible=True 的情况），直接返回
            elif isinstance(result, pd.DataFrame):
                return result
            return pd.DataFrame()
        
        # 检查是否为递归概念（有子概念）- 这个分支现在主要用于非 rec_cncpt 类型
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
            logger.debug(f"   🔹 表 {table_name} 推断的列: {required_columns}")
        
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
        
        # 🔧 FIX: 对于有callback的概念，需要加载callback所需的所有列
        # 例如 hirid_rate_kg 需要 givendose, doseunit, infusionid 等
        has_callback = getattr(source, 'callback', None) is not None
        
        if has_callback:
            # 对于有callback的概念，加载表中所有相关列而不是只加载标准列
            # 因为callback函数需要访问更多的列（如 givendose, doseunit, infusionid）
            required_columns = self._columns_for_source(source, id_type)
            # 添加callback可能需要的额外列
            callback_extra_cols = []
            if source.callback in ('hirid_rate_kg', 'hirid_rate', 'hirid_duration'):
                callback_extra_cols = ['givendose', 'doseunit', 'infusionid', 'givenat']
            elif source.callback in ('aumc_rate_kg', 'aumc_rate'):
                callback_extra_cols = ['dose', 'doseunit', 'doseunitid', 'rate', 'rateunit', 'infusionid', 'start', 'stop']
            elif source.callback in ('mimic_rate_cv', 'mimic_rate_mv') or (
                source.callback and 'mimv_rate' in source.callback
            ):
                callback_extra_cols = ['amount', 'amountuom', 'rate', 'rateuom', 'ordercategorydescription']
            
            if required_columns is None:
                required_columns = callback_extra_cols
            else:
                for col in callback_extra_cols:
                    if col not in required_columns:
                        required_columns.append(col)
        else:
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
        id_col = self._get_id_column(df, id_type)
        if patient_ids is not None:
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
        
        # 🔧 FIX: 对于有callback的概念，调用callback处理数据
        # callback会处理列选择、值转换、时间扩展等逻辑
        if has_callback:
            from .concept import _apply_callback
            
            # 获取patient weight（如果callback需要）
            if source.callback in ('hirid_rate_kg', 'aumc_rate_kg', 'sic_rate_kg') and 'weight' not in df.columns:
                weight_df = self._get_patient_weights(df, id_col, id_type)
                if weight_df is not None and not weight_df.empty:
                    df = df.merge(weight_df, on=id_col, how='left')
            
            # 转换时间列为数值（如果需要）
            time_col = source.index_var or self._get_time_column(df)
            mimic_duration_callback = source.callback in {
                'mimic_dur_inmv',
                'mimic_dur_incv',
            }
            if (
                time_col
                and time_col in df.columns
                and not mimic_duration_callback
            ):
                df = self._convert_time_column_to_hours(df, time_col, id_col)
            
            # 调用callback
            df = _apply_callback(
                frame=df,
                source=source,
                concept_name=concept_name,
                unit_column=source.unit_var,
                resolver=None,  # ConceptLoader不使用resolver
                patient_ids=patient_ids,
                data_source=self._data_source,
                interval=interval,
            )

            # MIMIC duration arithmetic must run while both the start and end
            # clocks are still absolute datetimes.  Only after the callback
            # has clipped and measured the episode do we place its start on
            # the public relative-hour axis.  Normalise both CareVue and
            # MetaVision to one compatibility-loader time column so their
            # source frames concatenate and aggregate on the same key.
            if mimic_duration_callback and time_col and time_col in df.columns:
                df = self._convert_mimic_duration_time_to_hours(
                    df, time_col, id_col
                )
                if time_col != 'time':
                    df = df.rename(columns={time_col: 'time'})
            
            return df
        
        # 4. 确定值列（无callback时的原有逻辑）
        val_col = source.value_var or 'valuenum'  # 默认使用 valuenum
        if val_col not in df.columns:
            # 尝试其他可能的值列
            for candidate in ['valuenum', 'value', 'amount']:
                if candidate in df.columns:
                    val_col = candidate
                    break
        
        # 5. 选择需要的列
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
    
    def _get_patient_weights(
        self,
        df: pd.DataFrame,
        id_col: str,
        id_type: str
    ) -> Optional[pd.DataFrame]:
        """获取患者体重数据"""
        try:
            from .concept import load_dictionary
            concept_dict = load_dictionary(self._src_name)
            if 'weight' not in concept_dict:
                return None
            
            unique_ids = df[id_col].unique().tolist()
            weight_data = self._load_one_concept(
                concept=concept_dict['weight'],
                patient_ids=unique_ids,
                id_type=id_type,
                interval=timedelta(hours=1),
                aggregate='median'
            )
            
            if weight_data is not None and not weight_data.empty:
                # 确保只返回id和weight列
                if 'value' in weight_data.columns and 'weight' not in weight_data.columns:
                    weight_data = weight_data.rename(columns={'value': 'weight'})
                
                # 取每个患者的中位数体重
                if 'weight' in weight_data.columns:
                    weight_data['weight'] = pd.to_numeric(weight_data['weight'], errors='coerce')
                    weight_data = weight_data.groupby(id_col)['weight'].median().reset_index()
                    return weight_data
            return None
        except Exception as e:
            logger.debug(f"获取体重数据失败: {e}")
            return None
    
    def _convert_time_column_to_hours(
        self,
        df: pd.DataFrame,
        time_col: str,
        id_col: str
    ) -> pd.DataFrame:
        """将时间列从datetime转换为相对ICU入院的小时数"""
        if time_col not in df.columns:
            return df
        
        time_series = df[time_col]
        if pd.api.types.is_numeric_dtype(time_series):
            # 已经是数值，不需要转换
            return df
        
        # 尝试转换为datetime
        try:
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # 获取ICU入院时间
            from .table import load_id_tbl
            icu_times = load_id_tbl(self._src_name, 'icustay', path=self.data_path)
            
            # 🔧 FIX: 支持不同数据库的入院时间列名
            # MIMIC-IV: intime, HiRID: admissiontime, eICU: hospitaladmittime, AUMC: admittedat
            intime_candidates = ['intime', 'admissiontime', 'hospitaladmittime', 'admittedat']
            intime_col = None
            for cand in intime_candidates:
                if cand in icu_times.columns:
                    intime_col = cand
                    break
            
            if not icu_times.empty and intime_col:
                if id_col and id_col in df.columns and id_col in icu_times.columns:
                    df = df.merge(icu_times[[id_col, intime_col]], on=id_col, how='left')
                    df[intime_col] = pd.to_datetime(df[intime_col], errors='coerce')
                    
                    time_diff = (df[time_col] - df[intime_col]).dt.total_seconds() / 3600
                    df[time_col] = time_diff
                    df = df.drop(columns=[intime_col])
        except Exception as e:
            logger.debug(f"时间转换失败: {e}")
        
        return df

    def _convert_mimic_duration_time_to_hours(
        self,
        df: pd.DataFrame,
        time_col: str,
        id_col: str,
    ) -> pd.DataFrame:
        """Place a completed MIMIC duration episode on the ICU-hour axis."""

        if time_col not in df.columns or pd.api.types.is_numeric_dtype(
            df[time_col]
        ):
            return df
        if self._data_source is None or not id_col or id_col not in df.columns:
            raise ValueError(
                "MIMIC duration time alignment requires a data source and ICU stay id"
            )
        try:
            table = self._data_source.load_table(
                'icustays',
                columns=[id_col, 'intime'],
                verbose=False,
            )
            bounds = table.data if hasattr(table, 'data') else table
        except Exception as exc:
            raise ValueError(
                "MIMIC duration time alignment could not load ICU intime"
            ) from exc
        if not isinstance(bounds, pd.DataFrame):
            bounds = pd.DataFrame(bounds)
        if id_col not in bounds.columns or 'intime' not in bounds.columns:
            raise ValueError(
                "MIMIC duration time alignment requires ICU stay id and intime"
            )

        bounds = bounds[[id_col, 'intime']].dropna(subset=[id_col]).copy()
        conflicting = bounds.dropna(subset=['intime']).groupby(id_col)[
            'intime'
        ].nunique(dropna=True)
        if conflicting.gt(1).any():
            raise ValueError("MIMIC ICU stay table contains conflicting intimes")
        bounds = bounds.drop_duplicates(subset=[id_col], keep='last')
        intime = df[id_col].map(bounds.set_index(id_col)['intime'])
        intime = pd.to_datetime(intime, errors='coerce', utc=True).dt.tz_localize(
            None
        )
        event_time = pd.to_datetime(
            df[time_col], errors='coerce', utc=True
        ).dt.tz_localize(None)
        unresolved = intime.isna() | event_time.isna()
        if unresolved.any():
            raise ValueError(
                "MIMIC duration time alignment found episodes without ICU intime"
            )

        result = df.copy()
        relative_seconds = (event_time - intime).dt.total_seconds()
        result[time_col] = (relative_seconds // 60.0) / 60.0
        return result
    
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
            # 使用 sub_concepts 属性（而非 items），这是 ConceptDefinition 的标准字段
            sub_concept_names = concept.sub_concepts if hasattr(concept, 'sub_concepts') else []
            sub_data = {}
            
            # 按照依赖顺序加载子概念
            for sub_name in sub_concept_names:
                try:
                    # 从字典中加载子概念定义
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
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not sub_data:
                result = pd.DataFrame()
            else:
                # 2. 应用回调函数
                # rec_cncpt 类型的概念需要通过回调函数处理子概念
                # 但是 ConceptLoader 架构与 ConceptResolver 不同
                # 我们需要委托给 ConceptResolver 来处理回调
                
                # 如果没有回调函数，简单合并子概念
                callback_name = concept.callback if hasattr(concept, 'callback') else None
                
                if not callback_name:
                    # 没有回调，尝试简单合并
                    if len(sub_data) == 1:
                        result = list(sub_data.values())[0]
                    else:
                        result = self._merge_sub_concepts(sub_data, id_type, interval)
                else:
                    # 有回调，需要通过 ConceptResolver 处理
                    # 这里我们需要使用不同的路径：直接委托给 ConceptResolver
                    from .concept import ConceptResolver
                    from .datasource import ICUDataSource
                    from .config import load_src_cfg
                    from pathlib import Path
                    
                    # 创建或获取数据源
                    if self._data_source is not None:
                        data_source = self._data_source
                    else:
                        # 创建新的数据源
                        config = load_src_cfg(self._src_name)
                        data_source = ICUDataSource(config, base_path=Path(self.data_path) if self.data_path else None)
                    
                    # 创建 ConceptResolver
                    resolver = ConceptResolver(load_dictionary(self._src_name))
                    
                    # 使用 ConceptResolver 加载这个概念
                    result_dict = resolver.load_concepts(
                        [concept.name],
                        data_source=data_source,
                        merge=False,
                        patient_ids=patient_ids,
                        interval=interval,
                        aggregate=aggregate,
                        verbose=kwargs.get('verbose', False),
                        **kwargs
                    )
                    
                    # 提取结果
                    if isinstance(result_dict, dict) and concept.name in result_dict:
                        from .io.ts_utils import ICUTable
                        result_table = result_dict[concept.name]
                        if isinstance(result_table, ICUTable):
                            result = result_table.data
                        else:
                            result = result_table
                    else:
                        result = pd.DataFrame()
            
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
    
    def _convert_units(
        self,
        data: pd.DataFrame,
        target_unit: str,
        substance: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convert each row's ``value`` from its ``unit`` column to ``target_unit``.

        Uses :class:`easyicu.unit_conversion.UnitConverter`. Rows whose unit
        cannot be converted are left unchanged (with a debug log) so we never
        silently drop data; previously this was a no-op pass-through that
        ignored ``target_unit`` entirely.
        """
        if not target_unit or 'value' not in data.columns or 'unit' not in data.columns:
            return data
        try:
            from .utils.unit_conversion import UnitConverter
        except ImportError:
            logger.debug("unit_conversion module unavailable; skipping conversion")
            return data

        target_norm = UnitConverter.normalize_unit(target_unit)
        out = data.copy()
        values = pd.to_numeric(out['value'], errors='coerce')
        units = out['unit'].astype(str)

        # Group by source unit so we call convert once per unique unit.
        unique_units = units.dropna().unique()
        unconvertible: list[str] = []
        for unit_value in unique_units:
            try:
                src_norm = UnitConverter.normalize_unit(unit_value)
            except Exception:
                unconvertible.append(unit_value)
                continue
            if src_norm == target_norm:
                continue
            if not UnitConverter.can_convert(unit_value, target_unit, substance):
                unconvertible.append(unit_value)
                continue
            mask = units == unit_value
            try:
                converted = UnitConverter.convert(
                    values[mask].to_numpy(), unit_value, target_unit, substance
                )
            except Exception as exc:
                logger.debug(
                    "unit conversion %r -> %r failed: %s", unit_value, target_unit, exc
                )
                unconvertible.append(unit_value)
                continue
            values.loc[mask] = converted
            out.loc[mask, 'unit'] = target_unit

        if unconvertible:
            logger.debug(
                "skipped unit conversion to %s for unsupported units: %s",
                target_unit, sorted(set(unconvertible)),
            )
        out['value'] = values
        return out
    
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
    
    def _convert_time_to_hours(self, df: pd.DataFrame, id_type: str) -> pd.DataFrame:
        """
        将time列从datetime转换为相对ICU入院的小时数,并清理列格式以匹配ricu输出
        
        Args:
            df: 包含time列的DataFrame
            id_type: ID类型
            
        Returns:
            time列转换为数值、列格式与ricu一致的DataFrame
        """
        logger.debug(f"[_convert_time_to_hours] Input: shape={df.shape}, columns={df.columns.tolist()}, id_type={id_type}")
        
        if df.empty:
            return df
        
        df = df.copy()
        
        # 检测时间列名(可能是time或charttime)
        time_col_name = None
        if 'time' in df.columns:
            time_col_name = 'time'
        elif 'charttime' in df.columns:
            time_col_name = 'charttime'
        
        # 1. 转换时间列为相对小时数
        if time_col_name and not pd.api.types.is_numeric_dtype(df[time_col_name]):
            # 加载ICU入院时间
            from easyicu.table import load_id_tbl
            icu_times = load_id_tbl(self._src_name, id_type, path=self.data_path)
            
            if not icu_times.empty:
                # 检测 intime 列名（不同数据库使用不同列名）
                intime_candidates = ['intime', 'admittedat', 'admissiontime', 
                                     'hospitaladmittime', 'ICUOffset']
                intime_col = None
                for cand in intime_candidates:
                    if cand in icu_times.columns:
                        intime_col = cand
                        break
                
                if intime_col is not None:
                    # 确定ID列名
                    id_col = self._get_id_column(df, id_type)
                    if id_col and id_col in df.columns:
                        # 合并入院时间
                        df = df.merge(icu_times[[id_col, intime_col]], on=id_col, how='left')
                        
                        # 转换时间列为相对小时数
                        df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')
                        df[intime_col] = pd.to_datetime(df[intime_col], errors='coerce')
                        
                        # 计算时间差(小时)
                        time_diff = (df[time_col_name] - df[intime_col]).dt.total_seconds() / 3600
                        df[time_col_name] = time_diff.round(2)
                    
                        # 删除辅助列
                        df = df.drop(columns=[intime_col])
        
        # 2. 清理列格式以匹配ricu输出
        # 确定主ID列(根据id_type)
        id_mappings = {
            'patient': ['subject_id', 'patientid', 'patient_id'],
            'hadm': ['hadm_id', 'admissionid', 'admission_id'],
            'icustay': ['stay_id', 'icustay_id', 'patientunitstayid', 'admissionid'],
        }
        
        # 找到主ID列
        id_col = None
        possible_names = id_mappings.get(id_type, [id_type])
        for name in possible_names:
            if name in df.columns:
                id_col = name
                break
        
        if not id_col:
            # 回退到_get_id_column
            id_col = self._get_id_column(df, id_type)
        
        logger.debug(f"[_convert_time_to_hours] id_col={id_col}, all columns={df.columns.tolist()}")
        
        # 移除多余的ID列(保留主ID列)
        all_id_cols = set()
        for names in id_mappings.values():
            all_id_cols.update(names)
        
        extra_id_cols = [col for col in df.columns if col in all_id_cols and col != id_col]
        
        logger.debug(f"[_convert_time_to_hours] all_id_cols={all_id_cols}, extra_id_cols={extra_id_cols}")
        
        if extra_id_cols:
            df = df.drop(columns=extra_id_cols)
            logger.debug(f"[_convert_time_to_hours] After drop: columns={df.columns.tolist()}")
        
        # 3. 统一时间列名为charttime(ricu使用charttime)
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'charttime'})
            time_col_name = 'charttime'
        
        # 4. 调整列顺序: [id_col, charttime, concept1, concept2, ...]
        cols = [id_col]
        if time_col_name and time_col_name in df.columns:
            cols.append(time_col_name)
        
        # 添加其他列(概念值、辅助列等)
        other_cols = [col for col in df.columns if col not in cols]
        cols.extend(other_cols)
        
        df = df[cols]
        
        return df
    
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
            'miiv': {
                'icustay': 'stay_id',
                'hadm': 'hadm_id',
                'subject': 'subject_id',
            },
            'aumc': {
                'icustay': 'admissionid',
                'hadm': 'admissionid',
                'subject': 'patientid',
            },
            'hirid': {
                'icustay': 'patientid',
                'hadm': 'patientid',
                'subject': 'patientid',
            },
            'sic': {
                'icustay': 'CaseID',
                'hadm': 'CaseID',
                'subject': 'PatientID',
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
        # 🔧 FIX: Added more time column candidates for different databases
        # - givenat: HiRID pharma table
        # - infusionoffset: eICU inputOutput table
        # - start, stop: AUMC tables
        time_cols = ['charttime', 'time', 'datetime', 'timestamp', 
                     'starttime', 'observationoffset', 'givenat',
                     'infusionoffset', 'start', 'stop', 'entertime']
        
        for col in df.columns:
            if col.lower() in [t.lower() for t in time_cols]:
                return col
        
        return None

    def _ensure_id_column(self, df: pd.DataFrame, id_type: str) -> pd.DataFrame:
        """Ensure the dataframe has the target ID column, augmenting if necessary."""
        target_col = self._canonical_id_column(id_type)
        
        # Check if target column already exists
        existing_col = self._get_id_column(df, id_type)
        if existing_col:
            return df
            
        # If not exists, try to map from other ID columns
        available_ids = []
        for col in df.columns:
            if col in ['hadm_id', 'subject_id', 'stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']:
                available_ids.append(col)
        
        if not available_ids:
            return df
            
        # Load ID lookup table
        lookup = self._load_id_lookup()
        if lookup.empty:
            return df
            
        if target_col not in lookup.columns:
            return df
            
        for avail_id in available_ids:
            if avail_id in lookup.columns:
                # Merge
                subset = lookup[[avail_id, target_col]].dropna().drop_duplicates()
                # Use left merge to preserve data rows
                df = df.merge(subset, on=avail_id, how='left')
                return df
                
        return df

    def _filter_by_patient(
        self, 
        df: pd.DataFrame, 
        patient_ids: Union[List, pd.DataFrame], 
        id_type: str
    ) -> pd.DataFrame:
        """Filter dataframe by patient IDs."""
        if patient_ids is None:
            return df
            
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
                    return df.iloc[0:0] # Empty dataframe with same columns
                df = df[df[id_col].isin(filter_values)]
        
        return df

    def _preload_tables(
        self,
        concept_objs: List[Concept],
        patient_ids: Optional[Union[List, pd.DataFrame]],
        id_type: str,
        verbose: bool = False,
        parallel_mode: bool = False
    ):
        """Preload and filter tables for all concepts.
        
        Args:
            parallel_mode: If True, use more aggressive caching strategy for parallel execution
        """
        if verbose:
            mode_str = "并行" if parallel_mode else "串行"
            print(f"⚡ 预加载表 ({mode_str}模式)...")
        
        # 🚀 初始化 ICUDataSource（如果还没有）
        # 这对 rec_cncpt 概念至关重要，因为 ConceptResolver 需要数据源对象
        if self._data_source is None:
            from .datasource import ICUDataSource
            from pathlib import Path
            self._data_source = ICUDataSource(
                self.src,
                base_path=Path(self.data_path) if self.data_path else None
            )
            if verbose:
                print(f"  初始化数据源: {self._src_name}")
            
        # 1. Identify required tables and columns - 递归收集所有依赖
        table_columns = {} # table_name -> set of columns
        
        # 🚀 优化：递归收集所有依赖概念的表（特别是SOFA组件）
        def collect_dependencies(concept_name: str, visited: set = None):
            """递归收集概念的所有依赖表"""
            if visited is None:
                visited = set()
            if concept_name in visited:
                return
            visited.add(concept_name)
            
            try:
                from .concept import load_dictionary
                dict_obj = load_dictionary(self._src_name, include_sofa2='sofa2' in concept_name)
                if concept_name not in dict_obj._concepts:
                    return
                    
                concept = dict_obj._concepts[concept_name]
                
                # 处理当前概念
                sources = concept.for_data_source(self.src)
                for source in sources:
                    if not source.table:
                        continue
                    cols = self._columns_for_source(source, id_type)
                    if cols:
                        if source.table not in table_columns:
                            table_columns[source.table] = set()
                        table_columns[source.table].update(cols)
                
                # 递归处理依赖
                if hasattr(concept, 'items') and concept.items:
                    for dep_name in concept.items.keys():
                        collect_dependencies(dep_name, visited)
                        
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  收集依赖 {concept_name} 失败: {e}")
        
        # Helper to process a concept
        def process_concept(c):
            sources = c.for_data_source(self.src)
            for source in sources:
                if not source.table:
                    continue
                
                cols = self._columns_for_source(source, id_type)
                if cols:
                    if source.table not in table_columns:
                        table_columns[source.table] = set()
                    table_columns[source.table].update(cols)
            
            # 🚀 并行模式：递归收集依赖以避免后续重复加载
            if parallel_mode and hasattr(c, 'name'):
                collect_dependencies(c.name)
            elif hasattr(c, 'items') and c.items:
                # 串行模式：只处理直接子概念
                for sub in c.items.values():
                    if isinstance(sub, Concept):
                        process_concept(sub)

        for concept in concept_objs:
            process_concept(concept)
        
        if verbose and table_columns:
            print(f"  需要加载 {len(table_columns)} 张表")
        
        # 🚀 HiRID observations 等超大表不应该在预加载阶段加载
        # 这些表需要概念特定的 variableid 过滤，预加载时无法提供
        # 跳过这些表，让每个概念单独加载时使用精确过滤
        skip_preload_tables = set()
        if self._src_name == 'hirid':
            # HiRID observations: 7.77亿行，必须按概念精确过滤
            skip_preload_tables.add('observations')
        
        # 2. Load and filter
        for table_name, columns in table_columns.items():
            cached = self._table_cache.get(table_name)
            if cached is not None and set(columns).issubset(cached.columns):
                continue
            
            # 🚀 跳过需要概念特定过滤的超大表
            if table_name in skip_preload_tables:
                if verbose:
                    print(f"  ⏭️  跳过预加载 {table_name} (将按概念精确过滤)")
                continue
                
            if verbose:
                print(f"  Loading {table_name} with {len(columns)} columns...")
            
            try:
                # 🚀 优化：如果有 data_source，使用它并传递患者过滤器以在读取时就过滤
                if self._data_source is not None and patient_ids is not None:
                    # 构造患者过滤器
                    from .datasource import FilterSpec, FilterOp
                    # 确定ID列名
                    id_col = self._canonical_id_column(id_type)
                    # 转换患者ID列表
                    if isinstance(patient_ids, pd.DataFrame):
                        if id_col in patient_ids.columns:
                            patient_list = patient_ids[id_col].dropna().unique().tolist()
                        else:
                            patient_list = None
                    else:
                        patient_list = self._coerce_patient_list(patient_ids)
                    
                    if patient_list:
                        patient_filter = FilterSpec(column=id_col, op=FilterOp.IN, value=patient_list)
                        icu_table = self._data_source.load_table(
                            table_name,
                            columns=list(columns),
                            filters=[patient_filter],
                            verbose=verbose
                        )
                        df = icu_table.data
                    else:
                        # 没有有效的患者ID，回退到普通加载
                        try:
                            df = load_table(self._src_name, table_name, columns=list(columns), path=self.data_path)
                        except Exception:
                            df = load_table(self._src_name, table_name, path=self.data_path)
                        df = self._ensure_id_column(df, id_type)
                        if patient_ids is not None:
                            df = self._filter_by_patient(df, patient_ids, id_type)
                else:
                    # 没有 data_source 或 patient_ids，使用原有逻辑
                    try:
                        df = load_table(self._src_name, table_name, columns=list(columns), path=self.data_path)
                    except Exception:
                        df = load_table(self._src_name, table_name, path=self.data_path)
                    
                    df = self._ensure_id_column(df, id_type)
                    if patient_ids is not None:
                        df = self._filter_by_patient(df, patient_ids, id_type)
                
                # Store in cache
                self._table_cache[table_name] = df
                
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Failed to preload {table_name}: {e}")

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
    
    .. deprecated::
        此函数已被废弃，请使用 `easyicu.api.load_concepts` 代替。
        该函数仅保留用于向后兼容。
    """
    import warnings
    warnings.warn(
        "load_concepts from easyicu.load_concepts is deprecated. "
        "Use easyicu.load_concepts (from api module) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    data_path = kwargs.pop('data_path', None)
    loader = ConceptLoader(src, data_path=data_path)
    return loader.load_concepts(concepts, **kwargs)


# 向后兼容别名 - 已废弃，请使用 easyicu.api.load_concept
load_concept = load_concepts
