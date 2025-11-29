"""Concept dictionary utilities inspired by ricu."""

from __future__ import annotations

import copy
import json
import logging
import re
import operator
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
from threading import RLock, local as thread_local
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Union

import numpy as np
import pandas as pd

from .config import DataSourceConfig
from .datasource import FilterOp, FilterSpec, ICUDataSource
from .table import ICUTable, WinTbl
from .concept_callbacks import ConceptCallbackContext, execute_concept_callback
from . import ricu_compat

logger = logging.getLogger(__name__)

# 全局调试开关 - 设置为 False 可以减少输出
DEBUG_MODE = False

# Concepts that require hourly maxima (vasoactive infusion rates)
VASO_RATE_CONCEPTS = {"dopa_rate", "dobu_rate", "epi_rate", "norepi_rate", "adh_rate"}

def _debug(msg: str) -> None:
    if DEBUG_MODE:
        logger.debug(msg)

def _safe_serialize(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_safe_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    return str(value)

def _default_id_columns_for_db(db_name: Optional[str]) -> List[str]:
    """Return canonical identifier columns for a given database."""

    db = (db_name or "").lower()
    mapping = {
        "eicu": ["patientunitstayid"],
        "eicu_demo": ["patientunitstayid"],
        "aumc": ["admissionid"],
        "hirid": ["patientid"],
        "sic": ["caseid"],
        "miiv": ["stay_id"],
        "mimic_demo": ["stay_id"],
    }

    if db.startswith("mimic"):
        return ["stay_id"]
    return mapping.get(db, ["stay_id"])

@dataclass
class ConceptSource:
    """Describe how to load a concept for a specific data source."""

    table: Optional[str] = None
    sub_var: Optional[str] = None
    ids: Optional[List[object]] = None
    value_var: Optional[str] = None
    unit_var: Optional[str] = None
    index_var: Optional[str] = None
    dur_var: Optional[str] = None  # 持续时间列，可能是duration或endtime
    regex: Optional[str] = None
    class_name: Optional[str] = None
    callback: Optional[str] = None
    interval: Optional[pd.Timedelta] = None
    target: Optional[str] = None
    params: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "ConceptSource":
        payload = dict(mapping)

        table = payload.pop("table", None)
        sub_var = payload.pop("sub_var", None)
        if isinstance(sub_var, bool):
            sub_var = None
        ids = payload.pop("ids", None)

        if ids is not None:
            if isinstance(ids, bool):
                ids_list = None
            elif isinstance(ids, (str, int, float)):
                ids_list = [ids]
            elif isinstance(ids, Iterable):
                ids_list = list(ids)
            else:
                raise TypeError("Concept source 'ids' must be scalar or iterable")
        else:
            ids_list = None

        value_var = payload.pop("value_var", payload.pop("val_var", None))
        if isinstance(value_var, bool):
            value_var = None
        unit_var = payload.pop("unit_var", payload.pop("unit", None))
        if isinstance(unit_var, bool):
            unit_var = None
        index_var = payload.pop("index_var", payload.pop("time_var", None))
        if isinstance(index_var, bool):
            index_var = None
        dur_var = payload.pop("dur_var", None)
        if isinstance(dur_var, bool):
            dur_var = None

        regex = payload.pop("regex", None)
        class_name = payload.pop("class", payload.pop("class_name", None))
        callback = payload.pop("callback", None)
        interval = payload.pop("interval", None)
        target = payload.pop("target", None)

        return cls(
            table=str(table) if table is not None else None,
            sub_var=str(sub_var) if sub_var is not None else None,
            ids=ids_list,
            value_var=str(value_var) if value_var is not None else None,
            unit_var=str(unit_var) if unit_var is not None else None,
            index_var=str(index_var) if index_var is not None else None,
            dur_var=str(dur_var) if dur_var is not None else None,
            regex=str(regex) if regex is not None else None,
            class_name=str(class_name) if class_name is not None else None,
            callback=str(callback) if callback is not None else None,
            interval=_maybe_timedelta(interval),
            target=str(target) if target is not None else None,
            params=payload,
        )

@dataclass
class ConceptDefinition:
    """Full description of a concept across multiple data sources."""

    name: str
    sources: Dict[str, List[ConceptSource]]
    units: Optional[List[str]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    description: Optional[str] = None
    category: Optional[str] = None
    target: Optional[str] = None
    interval: Optional[pd.Timedelta] = None
    aggregate: Optional[object] = None
    class_name: Optional[str] = None
    callback: Optional[str] = None
    sub_concepts: List[str] = field(default_factory=list)
    family: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    levels: Optional[List[object]] = None
    keep_components: Optional[bool] = None
    omop_id: Optional[int] = None

    @classmethod
    def from_name_and_payload(
        cls,
        name: str,
        payload: Mapping[str, object],
    ) -> "ConceptDefinition":
        raw_sources = payload.get("sources", {})
        sources: Dict[str, List[ConceptSource]] = {}
        for src_name, entries in raw_sources.items():
            sources[src_name] = [
                ConceptSource.from_mapping(entry) for entry in entries
            ]

        unit_value = payload.get("unit")
        if isinstance(unit_value, str):
            units: Optional[List[str]] = [unit_value]
        elif isinstance(unit_value, Iterable):
            units = [str(item) for item in unit_value]
        else:
            units = None

        raw_concepts = payload.get("concepts")
        if raw_concepts is None:
            sub_concepts: List[str] = []
        elif isinstance(raw_concepts, (list, tuple)):
            sub_concepts = [str(item) for item in raw_concepts]
        else:
            sub_concepts = [str(raw_concepts)]

        depends_raw = payload.get("depends_on", [])
        if isinstance(depends_raw, str):
            depends_list = [depends_raw]
        elif isinstance(depends_raw, Iterable):
            depends_list = [str(item) for item in depends_raw]
        else:
            depends_list = []

        return cls(
            name=name,
            sources=sources,
            units=units,
            minimum=_maybe_float(payload.get("min")),
            maximum=_maybe_float(payload.get("max")),
            description=payload.get("description"),
            category=payload.get("category"),
            target=payload.get("target"),
            interval=_maybe_timedelta(payload.get("interval")),
            aggregate=payload.get("aggregate"),
            class_name=payload.get("class") or payload.get("class_name"),
            callback=payload.get("callback"),
            sub_concepts=sub_concepts,
            levels=payload.get("levels"),
            keep_components=payload.get("keep_components"),
            omop_id=_maybe_int(payload.get("omopid")),
            family=payload.get("family"),
            depends_on=depends_list,
        )

    def for_data_source(self, config: DataSourceConfig) -> List[ConceptSource]:
        candidates: List[ConceptSource] = []
        keys = [config.name, *config.class_prefix]
        for key in keys:
            if key in self.sources:
                candidates.extend(self.sources[key])
        return candidates

class ConceptDictionary:
    """Container for all concept definitions."""

    def __init__(self, concepts: Mapping[str, ConceptDefinition]):
        self._concepts = dict(concepts)

    def __contains__(self, name: object) -> bool:
        return name in self._concepts

    def __getitem__(self, name: str) -> ConceptDefinition:
        return self._concepts[name]

    def get(self, name: str, default=None) -> Optional[ConceptDefinition]:
        """Get a concept by name, returning default if not found."""
        return self._concepts.get(name, default)

    def items(self):
        return self._concepts.items()

    def keys(self):
        return self._concepts.keys()

    def values(self):
        return self._concepts.values()

    def copy(self) -> "ConceptDictionary":
        """Create a shallow copy of this dictionary."""
        return ConceptDictionary(self._concepts.copy())

    def update(self, other: "ConceptDictionary") -> None:
        """Merge another dictionary into this one with per-concept granularity."""
        if not isinstance(other, ConceptDictionary):
            raise TypeError("Can only update from another ConceptDictionary")

        for name, incoming in other._concepts.items():
            if name not in self._concepts:
                self._concepts[name] = incoming
                continue

            current = self._concepts[name]

            merged_sources: Dict[str, List[ConceptSource]] = copy.deepcopy(current.sources)
            for source_name, entries in incoming.sources.items():
                merged_sources[source_name] = copy.deepcopy(entries)

            def _pick(new_value, old_value, *, allow_empty: bool = False):
                if allow_empty:
                    return copy.deepcopy(new_value) if new_value is not None else copy.deepcopy(old_value)
                if isinstance(new_value, list):
                    return copy.deepcopy(new_value) if new_value else copy.deepcopy(old_value)
                return new_value if new_value not in (None,) else old_value

            merged_definition = ConceptDefinition(
                name=name,
                sources=merged_sources,
                units=_pick(incoming.units, current.units, allow_empty=True),
                minimum=incoming.minimum if incoming.minimum is not None else current.minimum,
                maximum=incoming.maximum if incoming.maximum is not None else current.maximum,
                description=incoming.description if incoming.description is not None else current.description,
                category=incoming.category if incoming.category is not None else current.category,
                target=incoming.target if incoming.target is not None else current.target,
                interval=incoming.interval if incoming.interval is not None else current.interval,
                aggregate=incoming.aggregate if incoming.aggregate is not None else current.aggregate,
                class_name=incoming.class_name if incoming.class_name is not None else current.class_name,
                callback=incoming.callback if incoming.callback is not None else current.callback,
                sub_concepts=_pick(incoming.sub_concepts, current.sub_concepts),
                levels=_pick(incoming.levels, current.levels),
                keep_components=(
                    incoming.keep_components
                    if incoming.keep_components is not None
                    else current.keep_components
                ),
                omop_id=incoming.omop_id if incoming.omop_id is not None else current.omop_id,
                family=incoming.family if incoming.family is not None else current.family,
                depends_on=_pick(incoming.depends_on, current.depends_on, allow_empty=True),
            )

            self._concepts[name] = merged_definition

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ConceptDictionary":
        concepts = {
            name: ConceptDefinition.from_name_and_payload(name, definition)
            for name, definition in payload.items()
        }
        return cls(concepts)

    @classmethod
    def from_json(cls, file_path: str | Path) -> "ConceptDictionary":
        path = Path(file_path)
        with path.open("r", encoding="utf8") as handle:
            raw_dict = json.load(handle)
        return cls.from_payload(raw_dict)
    
    @classmethod
    def from_multiple_json(cls, file_paths: List[str | Path]) -> "ConceptDictionary":
        """从多个 JSON 文件加载概念字典并合并
        
        Args:
            file_paths: JSON 文件路径列表，后面的文件会覆盖前面的同名概念
            
        Returns:
            合并后的概念字典
            
        Examples:
            >>> dict1 = ConceptDictionary.from_multiple_json([
            ...     'data/concept-dict.json',
            ...     'data/sofa2-dict.json'
            ... ])
        """
        merged_payload = {}
        for file_path in file_paths:
            path = Path(file_path)
            with path.open("r", encoding="utf8") as handle:
                raw_dict = json.load(handle)
            # 合并，后面的覆盖前面的
            merged_payload.update(raw_dict)
        return cls.from_payload(merged_payload)

class ConceptResolver:
    """Resolve concept definitions into concrete tabular data."""

    def __init__(self, dictionary: ConceptDictionary, cache_dir: Optional[Path] = None) -> None:
        self.dictionary = dictionary
        # Cache for icustays table to avoid repeated loading
        self._icustays_cache: Optional[pd.DataFrame] = None
        # Cache for ID mappings (stay_id <-> subject_id)
        self._id_mapping_cache: Optional[pd.DataFrame] = None
        # Cache for loaded tables to avoid repeated loading
        # Key: (table_name, frozenset(patient_ids), frozenset(filters))
        self._table_cache: Dict[tuple, pd.DataFrame] = {}
        self._cache_lock = RLock()
        self._concept_cache: Dict[str, ICUTable] = {}
        # 🚀 新增：概念数据缓存（避免重复加载相同概念，如urine）
        # Key: (concept_name, patient_ids_hash, interval, aggregate)
        self._concept_data_cache: Dict[tuple, pd.DataFrame] = {}
        # 多线程支持：使用线程局部存储避免循环依赖误报
        self._thread_local = thread_local()
        # 🔧 嵌套调用深度跟踪：防止递归概念的内部调用清除缓存
        self._load_depth = 0
        self.cache_dir = cache_dir if cache_dir else None
        self.cache_schema_version = "1"
        self.dictionary_signature = self._compute_dictionary_signature()

    def available_concepts(self) -> List[str]:
        return sorted(self.dictionary.keys())

    def _compute_dictionary_signature(self) -> str:
        payload: Dict[str, object] = {}
        for name, definition in self.dictionary.items():
            payload[name] = {
                "callback": definition.callback,
                "aggregate": definition.aggregate,
                "sub_concepts": definition.sub_concepts,
                "sources": {
                    key: [asdict(source) for source in sources]
                    for key, sources in definition.sources.items()
                },
            }
        encoded = json.dumps(payload, sort_keys=True, default=_safe_serialize).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def clear_table_cache(self) -> None:
        """Clear cached source tables."""
        with self._cache_lock:
            self._table_cache.clear()
            self._concept_cache.clear()
            self._concept_data_cache.clear()  # 🚀 清除概念数据缓存
            # 清除当前线程的inflight集合
            if hasattr(self._thread_local, 'inflight'):
                self._thread_local.inflight.clear()

    def _get_inflight(self) -> set:
        """获取当前线程的inflight集合（线程安全）"""
        if not hasattr(self._thread_local, 'inflight'):
            self._thread_local.inflight = set()
        return self._thread_local.inflight

    def _should_fill_gaps(self, concept_name: str, definition: ConceptDefinition) -> bool:
        category = (definition.category or "").lower() if definition.category else ""
        concept = concept_name.lower()

        raw_class = getattr(definition, "class_name", None)
        class_names: List[str] = []
        if isinstance(raw_class, str):
            class_names = [raw_class.lower()]
        elif isinstance(raw_class, Iterable):
            class_names = [str(item).lower() for item in raw_class if item]
        else:
            class_names = []

        # Never fill gaps for logical/boolean concepts (abx, samp, etc.)
        # These are event indicators that should remain sparse
        if "lgl_cncpt" in class_names:
            return False
        
        # 🔧 CRITICAL FIX 2024-12: Do NOT fill gaps for medication rate concepts
        # These concepts (norepi_rate, dobu_rate, etc.) have interval data (start/end times)
        # and are already correctly expanded by expand() in _apply_aggregation.
        # Global fill_gaps with ffill would incorrectly fill across DISCONTINUOUS time segments.
        # Example: Patient with norepi from hour 8-150 and hour 980-982 would have
        # hours 151-979 incorrectly filled with the value from hour 150.
        # This caused pyricu coverage (90%) >> ricu coverage (36%) for norepi_rate.
        # Solution: disable global fill_gaps; ricu handles this per-segment in expand().
        if concept.endswith('_rate') or concept.endswith('_equiv'):
            return False  # Changed from True to False
        
        # 🔧 CRITICAL FIX: Do NOT fill gaps for vent_ind
        # R ricu's vent_ind callback only returns time points where ventilation is active.
        # It does NOT fill gaps between ventilation windows.
        # The expand() function in sofa_resp handles vent_ind expansion, not fill_gaps.
        # Filling gaps would create NaN rows for non-ventilated time points,
        # which causes row inflation (67 → 157 rows for patient 30009597).
        if concept == 'vent_ind':
            return False
        
        # 🔧 CRITICAL FIX 2024-12: Do NOT fill gaps for urine
        # R ricu's fill_gaps for urine only fills the FIRST continuous segment (~50 hours),
        # then only keeps original data points for later segments.
        # Simple fill_gaps fills the entire range (min_time to max_time), which is wrong.
        # The urine24 callback handles the proper ricu-style segmented fill logic.
        # ONLY fill for urine24 if needed (but the callback does its own fill)
        if concept == 'urine':
            return False
        
        # urine24 doesn't need fill_gaps either - callback handles it
        if concept == 'urine24':
            return False
        
        # All other concepts: no fill_gaps by default
        return False
    
    def _get_fill_method(self, concept_name: str, definition: ConceptDefinition) -> str:
        """Determine fill method for fill_gaps.
        
        Returns:
            - 'ffill': Forward fill for medication rate concepts (locf)
            - 'none': Only fill time points, do NOT fill values (keep NaN)
        """
        concept = concept_name.lower()
        
        # Medication rate concepts need locf (last observation carried forward)
        if concept.endswith('_rate') or concept.endswith('_equiv'):
            return 'ffill'
        
        # ⚠️ CRITICAL FIX: For urine/vent_ind, use 'none' to match ricu
        # ricu does NOT fill missing urine values with 0 - it keeps them as NaN
        # Only the time grid is filled, not the data values
        # This prevents false coverage (pyricu 100% vs ricu 2.74% for urine)
        
        # Default to none (only fill time grid, keep NaN for missing values)
        return 'none'
    
    def _expand_patient_ids(
        self, 
        patient_ids: Optional[Union[Dict[str, List], List]], 
        target_id_var: str,
        data_source: ICUDataSource,
        verbose: bool = False
    ) -> Optional[Dict[str, List]]:
        """自动扩展 patient_ids 以支持不同表的 ID 列
        
        如果用户只提供了 stay_id，但表需要 subject_id（或反之），
        自动查询 icustays 表获取映射关系。
        
        Args:
            patient_ids: 用户提供的患者ID（dict或list）
            target_id_var: 目标表需要的ID列名（如 'subject_id' 或 'stay_id'）
            data_source: 数据源
            verbose: 是否显示调试信息
            
        Returns:
            扩展后的 patient_ids 字典，包含所有必要的ID映射
            
        Examples:
            >>> # 用户只提供 stay_id
            >>> patient_ids = {'stay_id': [30018045]}
            >>> # 表需要 subject_id
            >>> expanded = _expand_patient_ids(patient_ids, 'subject_id', ds)
            >>> # 结果: {'stay_id': [30018045], 'subject_id': [18369403]}
        """
        if verbose and DEBUG_MODE:
            _debug(f'  🔍 _expand_patient_ids 被调用')
            _debug(f'     patient_ids: {patient_ids}')
            _debug(f'     target_id_var: {target_id_var}')
        
        if not patient_ids:
            return patient_ids
        
        # 转换为字典格式
        if not isinstance(patient_ids, dict):
            # 如果是列表，根据数据库类型选择合适的ID列名
            db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
            
            _debug(f'  patient_ids类型: {type(patient_ids)}')
            _debug(f'  db_name: {db_name}')
            
            if db_name in ['eicu', 'eicu_demo']:
                # eICU使用patientunitstayid
                patient_ids = {'patientunitstayid': list(patient_ids)}
                _debug(f'  转换为: {patient_ids}')
            elif db_name in ['aumc']:
                # AUMC使用admissionid
                patient_ids = {'admissionid': list(patient_ids)}
                _debug(f'  转换为: {patient_ids}')
            elif db_name in ['hirid']:
                # HiRID使用patientid
                patient_ids = {'patientid': list(patient_ids)}
                _debug(f'  转换为: {patient_ids}')
            else:
                # MIMIC-IV等使用stay_id
                patient_ids = {'stay_id': list(patient_ids)}
                _debug(f'  转换为: {patient_ids}')
        else:
            patient_ids = dict(patient_ids)  # 复制，避免修改原始数据
        
        # 如果已经包含目标ID，直接返回
        if target_id_var in patient_ids and patient_ids[target_id_var]:
            return patient_ids
        
        # 需要进行 ID 转换
        # 支持的转换：stay_id <-> subject_id
        if target_id_var == 'subject_id' and 'stay_id' in patient_ids:
            # 需要从 stay_id 获取 subject_id
            source_var = 'stay_id'
            source_values = patient_ids['stay_id']
        elif target_id_var == 'stay_id' and 'subject_id' in patient_ids:
            # 需要从 subject_id 获取 stay_id
            source_var = 'subject_id'
            source_values = patient_ids['subject_id']
        else:
            # 无法转换，返回原始值
            return patient_ids
        
        if not source_values:
            return patient_ids
        
        # 加载或使用缓存的 ID 映射表
        if self._id_mapping_cache is None:
            try:
                # eICU doesn't use icustays table
                db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                if db_name in ['eicu', 'eicu_demo']:
                    # eICU uses patientunitstayid as the primary ID, no mapping needed
                    return patient_ids
                
                from .datasource import FilterSpec, FilterOp
                # 加载 icustays 表（只需要 stay_id 和 subject_id）
                filters = [
                    FilterSpec(
                        column=source_var,
                        op=FilterOp.IN,
                        value=source_values,
                    )
                ]
                icustays_table = data_source.load_table(
                    'icustays', 
                    columns=['stay_id', 'subject_id'],
                    filters=filters,
                    verbose=False
                )
                if hasattr(icustays_table, 'data'):
                    self._id_mapping_cache = icustays_table.data[['stay_id', 'subject_id']].drop_duplicates()
                else:
                    self._id_mapping_cache = icustays_table[['stay_id', 'subject_id']].drop_duplicates()
                    
                if verbose:
                    if DEBUG_MODE: print(f"   🔗 加载 ID 映射表: {len(self._id_mapping_cache)} 条记录")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  无法加载 icustays 进行 ID 转换: {e}")
                return patient_ids
        
        # 从映射表中获取目标ID
        mapping_df = self._id_mapping_cache
        mask = mapping_df[source_var].isin(source_values)
        target_values = mapping_df.loc[mask, target_id_var].unique().tolist()
        
        if target_values:
            patient_ids[target_id_var] = target_values
            if verbose:
                if DEBUG_MODE: print(f"   🔗 ID 转换: {source_var}={len(source_values)}个 → {target_id_var}={len(target_values)}个")
        
        return patient_ids

    def load_concepts(
        self,
        concept_names: Iterable[str],
        data_source: ICUDataSource,
        *,
        merge: bool = True,
        aggregate: Optional[Union[str, bool, Mapping[str, object]]] = None,
        patient_ids: Optional[Iterable[object]] = None,
        verbose: bool = True,
        interval: Optional[pd.Timedelta] = None,  # Default 1 hour interval
        align_to_admission: bool = True,  # Align time to ICU admission as anchor
        ricu_compatible: bool = True,  # 默认启用ricu.R兼容格式
        concept_workers: int = 1,
        _batch_loading: bool = False,  # 🔧 批量加载模式标志，减少诊断输出
        _skip_concept_cache: bool = False,  # 🔧 跳过概念缓存，用于回调内部加载
        **kwargs,  # Additional parameters for callbacks (e.g., win_length, worst_val_fun)
    ):
        names = [name for name in concept_names]
        required_names = self._expand_dependencies(names)  # Ensure dependencies are expanded
        tables: Dict[str, ICUTable] = {}
        aggregators = self._normalise_aggregators(aggregate, required_names)
        
        # 🔧 嵌套调用深度跟踪：递归概念会嵌套调用 load_concepts
        # 只有顶层调用才应该清除缓存
        is_top_level = self._load_depth == 0
        self._load_depth += 1
        
        # 🚀 性能优化: 不要清空 _concept_cache，保留用于递归调用的缓存
        # 只在顶层调用时初始化（检查是否已存在）
        if not hasattr(self, '_concept_cache') or self._concept_cache is None:
            self._concept_cache = {}
        # 初始化当前线程的inflight集合
        self._get_inflight().clear()

        # 存储患者ID用于ricu格式转换
        self._last_patient_ids = list(patient_ids) if patient_ids else None
        
        # 🔧 关键修复：在merge模式下，设置标志以保留NaN行，匹配ricu的完整时间网格风格
        if merge and len(names) > 1:
            kwargs = dict(kwargs)  # 复制kwargs避免修改原始字典
            kwargs['_keep_na_rows'] = True
            # 设置批量加载标志以减少诊断输出
            if len(names) > 3:  # 只在加载多个概念时启用
                _batch_loading = True
                kwargs['_batch_loading'] = True

        if merge and len(names) > 1 and any(
            aggregators[name] is False for name in names
        ):
            raise ValueError(
                "Aggregation must be enabled for all concepts when merge=True."
            )

        # 🔧 CRITICAL FIX: Match R ricu's default interval behavior
        # R ricu uses interval=hours(1L) by default when aggregation is enabled
        # If user specifies aggregate but not interval, default to 1 hour
        if interval is None and aggregate is not None and aggregate is not False:
            # Check if any aggregator is not False
            has_aggregation = any(agg is not False for agg in aggregators.values())
            if has_aggregation:
                interval = pd.Timedelta(hours=1)
        
        total = len(names)

        for name in names:
            if name not in self.dictionary:
                raise KeyError(f"Concept '{name}' not present in dictionary")

        def _resolve(name: str, position: int) -> tuple[str, ICUTable]:
            if verbose and logger.isEnabledFor(logging.INFO):
                logger.info("➡️  [%d/%d] 加载概念 '%s'", position, total, name)

            concept_table = self._ensure_concept_loaded(
                name,
                data_source,
                aggregators,
                patient_ids,
                verbose,
                interval,
                align_to_admission,
                kwargs,
                _skip_concept_cache=_skip_concept_cache,
            )
            if verbose and logger.isEnabledFor(logging.INFO):
                if isinstance(concept_table, ICUTable):
                    row_count = len(concept_table.data)
                elif isinstance(concept_table, pd.DataFrame):
                    row_count = len(concept_table)
                else:
                    row_count = "N/A"
                logger.info("✅  概念 '%s' 已加载 (行数: %s)", name, row_count)
            return name, concept_table

        try:
            results: Dict[str, ICUTable] = {}
            if concept_workers > 1 and total > 1:
                with ThreadPoolExecutor(max_workers=concept_workers) as executor:
                    future_map = {
                        executor.submit(_resolve, name, idx): name
                        for idx, name in enumerate(names, start=1)
                    }
                    for future in as_completed(future_map):
                        name, concept_table = future.result()
                        results[name] = concept_table
            else:
                for idx, name in enumerate(names, start=1):
                    name, concept_table = _resolve(name, idx)
                    results[name] = concept_table

            tables = {
                name: results[name]
                for name in names
            }

            if not merge:
                # 如果是ricu_compatible模式且只有一个概念，返回ricu.R格式的DataFrame
                if ricu_compatible and len(tables) == 1:
                    concept_name = list(tables.keys())[0]
                    logger.debug("调试：调用_to_ricu_format处理概念 %s", concept_name)
                    # 计算interval_hours
                    interval_hours = 1.0
                    if interval is not None:
                        if hasattr(interval, 'total_seconds'):
                            interval_hours = interval.total_seconds() / 3600.0
                        elif isinstance(interval, (int, float)):
                            interval_hours = float(interval)
                    return self._to_ricu_format(tables[concept_name], concept_name, interval_hours=interval_hours)
                return tables

            # 如果是ricu_compatible模式，使用增强的ricu风格合并
            if ricu_compatible:
                return self._to_ricu_format_merged_enhanced(tables, names, interval)

            merged = self._merge_tables(tables)
            return merged
        finally:
            # 🔧 嵌套调用深度跟踪：减少深度计数器
            self._load_depth -= 1
            # 🔧 只有顶层调用才清除缓存，避免递归概念内部调用清除外层所需的缓存
            if is_top_level:
                with self._cache_lock:
                    self._concept_cache.clear()
                    self._concept_data_cache.clear()
                    # 清除当前线程的inflight集合
                    self._get_inflight().clear()

    def _load_single_concept(
        self,
        concept_name: str,
        data_source: ICUDataSource,
        *,
        aggregator: object,
        patient_ids: Optional[Iterable[object]],
        verbose: bool = True,
        interval: Optional[pd.Timedelta] = None,
        align_to_admission: bool = True,
        **kwargs,  # Additional parameters for callbacks
    ) -> ICUTable:
        # 🔧 批量加载模式：减少诊断输出
        batch_loading = kwargs.get('_batch_loading', False)
        if batch_loading:
            verbose = False  # 批量加载时抑制verbose输出
        definition = self.dictionary[concept_name]
        
        # 🔧 FIX: 对于 rec_cncpt 概念（如 tgcs），检查是否有数据库特定的直接 source 定义
        # 如果有，应该使用直接定义而不是递归加载子概念
        # 例如：eICU 的 tgcs 应该从 nursecharting 表的 'GCS Total' 直接读取
        # 而不是通过 egcs + mgcs + vgcs 计算（因为 eICU 没有单独的 GCS 组件数据）
        use_recursive = False
        has_direct_source = False  # 标记是否有直接的表 source
        if definition.sub_concepts:
            # 检查当前数据源是否有直接的表定义
            db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
            
            if db_name and hasattr(definition, 'sources') and db_name in definition.sources:
                db_sources = definition.sources[db_name]
                if isinstance(db_sources, list):
                    for db_source in db_sources:
                        db_source_dict = db_source.__dict__ if hasattr(db_source, '__dict__') else db_source
                        # 如果 source 有 table 定义（而不仅仅是 concepts），则使用直接加载
                        if 'table' in db_source_dict and db_source_dict.get('table'):
                            has_direct_source = True
                            break
            
            # 只有当没有直接 source 定义时，才使用递归加载
            use_recursive = not has_direct_source
        
        if use_recursive:
            return self._load_recursive_concept(
                concept_name,
                definition,
                data_source,
                aggregator=aggregator,
                patient_ids=patient_ids,
                verbose=verbose,  # 传递verbose参数
                interval=interval,  # Pass interval
                align_to_admission=align_to_admission,  # Pass align flag
                **kwargs,  # Pass kwargs to recursive concept
            )
        
        # Check if this concept has a concept-level callback
        # Skip callback if _bypass_callback flag is set (to avoid infinite recursion)
        # 🔧 FIX: 也跳过 callback 如果这是一个有直接 source 的 rec_cncpt
        # 例如 eICU 的 tgcs - 有 sum_components callback 但应该直接从表加载
        skip_callback = kwargs.get('_bypass_callback', False) or has_direct_source
        if definition.callback and not skip_callback:
            # Try to execute the callback if it's registered
            try:
                # Create empty tables dict - callback will load dependencies if needed
                tables = {}
                
                callback_context = ConceptCallbackContext(
                    concept_name=concept_name,
                    target=definition.target,
                    interval=interval,
                    resolver=self,
                    data_source=data_source,
                    patient_ids=patient_ids,
                    kwargs=kwargs,
                )
                result = execute_concept_callback(definition.callback, tables, callback_context)
                if result is not None:
                    return result
            except NotImplementedError:
                pass
            
            # If callback not found or failed, raise error
            raise NotImplementedError(
                f"Concept '{concept_name}' relies on a concept-level callback "
                f"'{definition.callback}' that is not yet supported."
            )
        
        config = data_source.config
        sources = definition.for_data_source(config)
        if not sources:
            # For optional sub-concepts (e.g., mech_vent in eICU), return empty table
            # instead of raising error - let callback handle missing concepts
            if kwargs.get('_allow_missing_concept', False):
                # Return empty ICUTable with database-appropriate default ID columns
                db_name = config.name if hasattr(config, 'name') else 'unknown'
                default_id_cols = _default_id_columns_for_db(db_name)
                
                empty_df = pd.DataFrame(columns=default_id_cols)
                return ICUTable(
                    data=empty_df,
                    id_columns=default_id_cols,
                    index_column=None,
                    value_column=None,
                )
            
            raise KeyError(
                f"No source configuration for concept '{concept_name}' "
                f"in data source '{config.name}'"
            )

        frames: List[pd.DataFrame] = []
        id_columns: List[str] = []
        index_column: Optional[str] = None
        unit_column: Optional[str] = None
        time_columns: List[str] = []

        for source in sources:
            if source.class_name == "fun_itm":
                return self._load_fun_item(
                    concept_name,
                    definition,
                    source,
                    data_source,
                    aggregator=aggregator,
                    patient_ids=patient_ids,
                    **kwargs,  # Pass kwargs to fun_item
                )

            if source.table is None:
                raise NotImplementedError(
                    f"Concept '{concept_name}' relies on a functional item "
                    "that is not yet supported."
                )

            if source.ids is not None and not source.sub_var:
                raise ValueError(
                    f"Concept '{concept_name}' requires 'sub_var' when specifying ids."
                )
            
            if hasattr(source, 'regex') and source.regex and not source.sub_var:
                raise ValueError(
                    f"Concept '{concept_name}' requires 'sub_var' when specifying regex."
                )
            
            table_cfg = data_source.config.get_table(source.table)
            defaults = table_cfg.defaults
            
            # Build filters for sub_var (only for ids, NOT regex)
            # Regex filtering is handled later after table loading (see line ~1428)
            filters = []
            if source.ids is not None:
                filters.append(FilterSpec(
                    column=source.sub_var,
                    op=FilterOp.IN,
                    value=source.ids,
                ))
            
            # 修复：添加患者过滤器
            # 即使 defaults.id_var 为 None，仍尝试添加患者过滤器
            # 对于 MIMIC-IV hosp 表（如 microbiologyevents），使用 subject_id
            # 对于 eICU 表，使用 patientunitstayid
            effective_id_var = defaults.id_var
            if patient_ids:
                if not effective_id_var:
                    # 如果没有配置 id_var，尝试检测常见的ID列
                    # 先检查数据库类型
                    db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                    
                    if db_name in ['eicu', 'eicu_demo']:
                        # eICU使用patientunitstayid
                        effective_id_var = 'patientunitstayid'
                    elif db_name in ['aumc']:
                        # AUMC使用admissionid
                        effective_id_var = 'admissionid'
                    elif db_name in ['hirid']:
                        # HiRID使用patientid
                        effective_id_var = 'patientid'
                    elif source.table in ['patients', 'admissions']:
                        # MIMIC-IV patients/admissions 表使用 subject_id
                        effective_id_var = 'subject_id'
                    elif source.table in ['microbiologyevents', 'd_labitems', 'prescriptions']:
                        # MIMIC-IV hosp表使用subject_id（labevents除外，它同时支持stay_id和subject_id）
                        effective_id_var = 'subject_id'
                    elif source.table in ['inputevents', 'chartevents', 'outputevents', 'procedureevents']:
                        # MIMIC-IV icu表使用stay_id
                        effective_id_var = 'stay_id'
                
                if effective_id_var:
                    # 🔗 自动扩展 patient_ids：如果用户只提供了 stay_id 但表需要 subject_id（或反之），
                    # 自动查询 icustays 获取映射关系
                    expanded_patient_ids = self._expand_patient_ids(
                        patient_ids, 
                        effective_id_var, 
                        data_source,
                        verbose=verbose
                    )
                    
                    # DEBUG
                    # patient_ids可能是dict(包含stay_id和subject_id)或列表
                    if isinstance(expanded_patient_ids, dict):
                        # 使用对应列的ID
                        id_values = expanded_patient_ids.get(effective_id_var)
                        
                        # DEBUG
                        if id_values:
                            # ✅ 关键修复：对于 hospital tables（如 labevents），如果使用 subject_id 过滤
                            # 需要在 metadata 中保存原始的 stay_id，供 datasource 在 join 后精确过滤
                            metadata = None
                            db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                            hospital_tables = ['labevents', 'prescriptions', 'microbiologyevents', 'emar', 'pharmacy']
                            
                            if (db_name in ['miiv', 'mimic_demo'] and 
                                source.table in hospital_tables and 
                                effective_id_var == 'subject_id' and 
                                'stay_id' in expanded_patient_ids):
                                # 保存原始 stay_id 到 metadata
                                original_stay_ids = expanded_patient_ids.get('stay_id')
                                if original_stay_ids:
                                    metadata = {'original_stay_ids': original_stay_ids}
                                    if DEBUG_MODE:
                                        print(f"   💾 在 subject_id 过滤器中附加原始 stay_id: {len(original_stay_ids)} 个")
                            
                            filters.append(
                                FilterSpec(
                                    column=effective_id_var,
                                    op=FilterOp.IN,
                                    value=id_values,
                                    metadata=metadata,
                                )
                            )
                    else:
                        # 原有逻辑：expanded_patient_ids 是列表（理论上不会到这里，因为已经转换为dict了）
                        filters.append(
                            FilterSpec(
                                column=effective_id_var,
                                op=FilterOp.IN,
                                value=expanded_patient_ids,
                            )
                        )
            
            # 🔄 表级缓存策略：
            # - 缓存键：(表名, 患者ID过滤器)
            # - 不包括 sub_var/ids 过滤器，因为不同概念可能有不同的 sub_var 过滤
            # - 缓存 ICUTable 对象（包含元数据）
            # - 从缓存加载后，仍需应用 sub_var/ids 过滤器
            
            # 分离患者ID过滤器和其他过滤器
            patient_filter_in_filters = None
            other_filters_list = []
            for f in filters:
                # 判断是否为患者ID过滤器（使用 effective_id_var 或常见的 ID 列）
                is_patient_filter = (
                    (effective_id_var and f.column == effective_id_var and f.op == FilterOp.IN) or
                    (f.column in ['subject_id', 'stay_id', 'hadm_id'] and f.op == FilterOp.IN)
                )
                if is_patient_filter:
                    patient_filter_in_filters = f
                else:
                    other_filters_list.append(f)
            
            # 创建缓存键
            patient_filter_key = None
            if patient_filter_in_filters:
                # 使用sorted tuple作为key
                patient_filter_key = (
                    patient_filter_in_filters.column,
                    tuple(sorted(patient_filter_in_filters.value))
                )
            
            cache_key = (source.table, patient_filter_key)
            
            # 跳过需要特殊处理表的缓存
            # labevents/admissions等需要subject_id→stay_id映射，缓存会保存映射前的数据导致patient过滤失效
            skip_cache_for_special_tables = source.table in ['labevents', 'microbiologyevents', 'inputevents', 'admissions']
            
            # 尝试从缓存获取
            cached_table = None
            if not skip_cache_for_special_tables:
                with self._cache_lock:
                    cached_table = self._table_cache.get(cache_key)
            if cached_table is not None:
                if verbose or DEBUG_MODE:
                    if DEBUG_MODE: print(f"   ♻️  使用缓存的表: {source.table} (跳过 {len(patient_filter_in_filters.value) if patient_filter_in_filters else 0} 个患者的加载)")
                # 从缓存获取ICUTable对象
                frame = cached_table.data.copy()
                
                if DEBUG_MODE:
                    print(f"   🔍 缓存数据: {len(frame)} 行, 列={list(frame.columns)[:5]}")
                
                # 应用其他过滤器（如 sub_var/ids）
                for f in other_filters_list:
                    before_count = len(frame)
                    frame = f.apply(frame)
                    if DEBUG_MODE:
                        print(f"   缓存分支过滤 {f.column}: {before_count:,} → {len(frame):,} 行")
                
                # 重新构建 table 对象（使用过滤后的 frame）
                table = ICUTable(
                    data=frame,
                    id_columns=cached_table.id_columns,
                    index_column=cached_table.index_column,
                    value_column=cached_table.value_column,
                    unit_column=cached_table.unit_column,
                )
            else:
                # 从数据源加载
                try:
                    # 🔧 构建需要的列列表：基于 source 的 value_var, sub_var, unit_var, index_var
                    # 这确保了像 eICU vitalperiodic 的 sao2 等特定值列会被加载
                    extra_columns: List[str] = []
                    if getattr(source, 'sub_var', None):
                        extra_columns.append(source.sub_var)
                    if getattr(source, 'value_var', None):
                        extra_columns.append(source.value_var)
                    if getattr(source, 'index_var', None):
                        extra_columns.append(source.index_var)
                    if getattr(source, 'unit_var', None):
                        extra_columns.append(source.unit_var)
                    
                    # Load table with filters and required columns
                    table = data_source.load_table(
                        source.table, 
                        columns=extra_columns if extra_columns else None,
                        filters=filters, 
                        verbose=verbose
                    )
                    
                    # 🔍 DEBUG: 检查table.data
                    if DEBUG_MODE:
                        print(f"   🔎 table.data类型: {type(table.data)}, 长度: {len(table.data) if hasattr(table.data, '__len__') else 'N/A'}")
                        if hasattr(table.data, 'columns'):
                            print(f"       列: {list(table.data.columns)}")
                        if hasattr(table.data, 'head'):
                            print(f"       前3行:\\n{table.data.head(3)}")
                    
                    frame = table.data.copy()
                    
                    # 🔍 DEBUG: 检查 datasource 返回的数据（只在调试模式下显示）
                    if DEBUG_MODE and source.table in ['labevents', 'microbiologyevents', 'inputevents']:
                        has_stay_id = 'stay_id' in frame.columns
                        has_subject_id = 'subject_id' in frame.columns
                        print(f"   📊 [{source.table}] datasource返回: {len(frame)}行, stay_id={has_stay_id}, subject_id={has_subject_id}")
                        if has_stay_id:
                            print(f"       stay_id 唯一值: {frame['stay_id'].nunique()} 个")
                    # 全局调试：在加载任何表后打印 AUMC numericitems 的负时间计数（便于排查）
                    if DEBUG_MODE and source.table == 'numericitems':
                        if 'measuredat' in frame.columns:
                            try:
                                negc = int((frame['measuredat'] < 0).sum())
                                print(f"   🐞 [LOAD] {source.table}: rows={len(frame)}, neg_measuredat={negc}")
                            except Exception:
                                pass
                    
                    # 调试：检查过滤是否成功（只在调试模式下显示）
                    if DEBUG_MODE and patient_ids and table.id_columns:
                        id_col = table.id_columns[0] if table.id_columns else None
                        if id_col and id_col in frame.columns:
                            unique_ids = frame[id_col].unique()
                            print(f"   🔍 表 {source.table} 加载后: {len(frame)} 行, 唯一{id_col}: {len(unique_ids)}个")
                            if len(unique_ids) <= 10:
                                print(f"       ID列表: {sorted(unique_ids)}")
                    
                    # 性能优化：对于AUMC/HiRID等高频数据，在表加载后立即降采样
                    # 检测数据库类型和数据频率
                    db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                    is_high_freq_db = db_name in ['aumc', 'hirid']
                    
                    if is_high_freq_db and table.index_column and len(frame) > 1000:
                        time_col = table.index_column
                        is_numeric_time = pd.api.types.is_numeric_dtype(frame[time_col])
                        
                        # 使用interval参数（如果提供）
                        target_interval = kwargs.get('interval', pd.Timedelta(hours=1))
                        if isinstance(target_interval, str):
                            target_interval = pd.Timedelta(target_interval)
                        
                        # 检测当前数据频率
                        need_resample = False
                        if len(frame) > 100:
                            frame_sorted = frame.sort_values(time_col)
                            time_diffs = frame_sorted[time_col].diff().dropna()
                            if len(time_diffs) > 10:
                                median_diff = time_diffs.median()
                                if is_numeric_time:
                                    # 数值时间（小时）
                                    target_hours = target_interval.total_seconds() / 3600.0
                                    # AUMC数据频率很高，中位差通常<0.1小时
                                    if median_diff < target_hours * 0.5:  # 如果中位间隔小于目标间隔的一半
                                        need_resample = True
                                else:
                                    # datetime时间
                                    if median_diff < target_interval * 0.5:
                                        need_resample = True
                        
                        # 执行降采样
                        if need_resample:
                            if verbose:
                                print(f"   ⚡ 检测到高频数据（{source.table}），降采样到 {target_interval}")
                            
                            id_cols = table.id_columns if table.id_columns else []
                            value_col = table.value_column
                            
                            # 如果id_cols为空但frame中有明显的ID列，尝试推断
                            if not id_cols:
                                potential_id_cols = ['admissionid', 'patientunitstayid', 'stay_id', 'patientid']
                                for col in potential_id_cols:
                                    if col in frame.columns:
                                        id_cols = [col]
                                        if verbose:
                                            print(f"   ℹ️  推断ID列: {col}")
                                        break
                            
                            if value_col and value_col in frame.columns:
                                if is_numeric_time:
                                    # 数值时间：四舍五入到interval
                                    interval_hours = target_interval.total_seconds() / 3600.0
                                    # 对于某些高频数据库（AUMC/HiRID），数值时间列单位为分钟（而不是小时）
                                    # 因此需要在原始单位上进行取整，以保留负时间点并避免单位错位。
                                    if db_name in ['aumc', 'hirid']:
                                        # 原始单位为分钟：将 interval 从小时转换为分钟
                                        native_interval = interval_hours * 60.0
                                    else:
                                        native_interval = interval_hours
                                    # 使用向下取整保留入ICU前的负时间点（避免 .round() 将小于0的值四舍五入到0）
                                    frame[time_col + '_rounded'] = np.floor(frame[time_col] / native_interval) * native_interval
                                    
                                    # 聚合：根据数据类型选择聚合函数
                                    # 对于输出类数据（尿量等）使用sum，其他使用mean
                                    agg_func = 'sum' if 'urine' in value_col.lower() or 'output' in value_col.lower() else 'mean'
                                    group_cols = id_cols + [time_col + '_rounded']
                                    
                                    # 保留所有列，不只是value_col
                                    # 对于数值列使用agg_func，其他列使用first
                                    agg_dict = {}
                                    for col in frame.columns:
                                        # 跳过分组列（ID列和时间列已经在group_cols中）
                                        # 🔧 FIX: 也跳过原始时间列，避免重命名后产生重复列
                                        if col in group_cols or col == time_col + '_rounded' or col == time_col:
                                            continue
                                        # value列：先检查类型，只有数值型才能聚合
                                        elif col == value_col:
                                            if pd.api.types.is_numeric_dtype(frame[col]) and not pd.api.types.is_bool_dtype(frame[col]):
                                                agg_dict[col] = agg_func
                                            else:
                                                agg_dict[col] = 'first'  # object类型用first
                                        # 其他数值列使用聚合函数（排除布尔类型）
                                        elif pd.api.types.is_numeric_dtype(frame[col]) and not pd.api.types.is_bool_dtype(frame[col]):
                                            agg_dict[col] = agg_func
                                        # 其他列（包括object、string等）使用first
                                        else:
                                            agg_dict[col] = 'first'
                                    
                                    if agg_dict:  # 只有当有列需要聚合时才执行
                                        try:
                                            frame = frame.groupby(group_cols, as_index=False).agg(agg_dict)
                                            frame = frame.rename(columns={time_col + '_rounded': time_col})
                                        except Exception as e:
                                            if verbose:
                                                print(f"   ⚠️  聚合失败: {e}")
                                                print(f"       group_cols={group_cols}")
                                                print(f"       agg_dict={agg_dict}")
                                                print(f"       frame列类型:")
                                                for col, dtype in frame.dtypes.items():
                                                    print(f"         {col}: {dtype}")
                                            raise
                                    else:
                                        # 没有需要聚合的列，只保留唯一的时间点
                                        frame = frame.drop_duplicates(subset=group_cols)
                                        frame = frame.rename(columns={time_col + '_rounded': time_col})
                                else:
                                    # datetime时间：使用resample
                                    if id_cols:
                                        resampled_groups = []
                                        agg_func = 'sum' if 'urine' in value_col.lower() or 'output' in value_col.lower() else 'mean'
                                        
                                        for group_id, group_df in frame.groupby(id_cols):
                                            group_df = group_df.set_index(time_col)
                                            
                                            # 聚合所有数值列
                                            numeric_cols = group_df.select_dtypes(include=[np.number]).columns.tolist()
                                            if value_col in numeric_cols:
                                                # value_col使用特定的聚合函数
                                                agg_dict = {value_col: agg_func}
                                                # 其他数值列使用mean
                                                for col in numeric_cols:
                                                    if col != value_col:
                                                        agg_dict[col] = 'mean'
                                            else:
                                                agg_dict = {col: 'mean' for col in numeric_cols}
                                            
                                            resampled = group_df[numeric_cols].resample(target_interval).agg(agg_dict)
                                            resampled = resampled.reset_index()
                                            
                                            # 添加ID列
                                            if isinstance(group_id, tuple):
                                                for i, col in enumerate(id_cols):
                                                    resampled[col] = group_id[i]
                                            else:
                                                resampled[id_cols[0]] = group_id
                                            
                                            resampled_groups.append(resampled)
                                        
                                        if resampled_groups:
                                            frame = pd.concat(resampled_groups, ignore_index=True)
                                    else:
                                        frame = frame.set_index(time_col)
                                        agg_func = 'sum' if 'urine' in value_col.lower() or 'output' in value_col.lower() else 'mean'
                                        numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
                                        agg_dict = {col: agg_func if col == value_col else 'mean' for col in numeric_cols}
                                        frame = frame[numeric_cols].resample(target_interval).agg(agg_dict).reset_index()
                                
                                if verbose:
                                    print(f"   ✓ 降采样完成：{len(table.data)} 行 → {len(frame)} 行")
                                
                                # 更新table对象以反映降采样后的数据
                                table = ICUTable(
                                    data=frame,
                                    id_columns=table.id_columns,
                                    index_column=table.index_column,
                                    value_column=table.value_column,
                                    unit_column=table.unit_column,
                                )
                    
                    
                    # 仅当有患者过滤器且不是特殊处理表时才缓存
                    # labevents/admissions等需要subject_id→stay_id映射，不应缓存原始subject_id级别数据
                    if patient_filter_in_filters and not skip_cache_for_special_tables:
                        # 缓存只应用了患者过滤器的表
                        patient_only_table = data_source.load_table(
                            source.table,
                            filters=[patient_filter_in_filters],
                            verbose=False
                        )
                        with self._cache_lock:
                            self._table_cache[cache_key] = patient_only_table
                        if verbose:
                            if DEBUG_MODE: print(f"   💾 缓存表 {source.table}: {len(patient_filter_in_filters.value)} 个患者")
                except (KeyError, FileNotFoundError, ValueError) as e:
                    # 如果表不存在，跳过这个源
                    if DEBUG_MODE or logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Table '{source.table}' not available: {type(e).__name__}: {str(e)[:100]}")
                    continue
            
            # MIMIC-IV 特殊处理：patients 表只有 subject_id，需要与 icustays 关联获取 stay_id
            if source.table == 'patients' and 'subject_id' in frame.columns and 'stay_id' not in frame.columns:
                try:
                    # 加载 icustays 表以获取 subject_id -> stay_id 的映射
                    icustay_filters = []
                    if patient_ids:
                        # patient_ids 可能是 dict 或 list
                        if isinstance(patient_ids, dict):
                            stay_ids = patient_ids.get('stay_id', [])
                        else:
                            stay_ids = patient_ids
                        if stay_ids:
                            icustay_filters.append(
                                FilterSpec(column='stay_id', op=FilterOp.IN, value=stay_ids)
                            )
                    
                    icustays = data_source.load_table('icustays', filters=icustay_filters if icustay_filters else None, verbose=False)
                    if hasattr(icustays, 'data'):
                        icu_df = icustays.data[['subject_id', 'stay_id']].drop_duplicates()
                    else:
                        icu_df = icustays[['subject_id', 'stay_id']].drop_duplicates()
                    
                    # 与 patients 表做内连接
                    frame = frame.merge(icu_df, on='subject_id', how='inner')
                    
                    # 删除 subject_id 列，只保留 stay_id
                    if 'stay_id' in frame.columns:
                        frame = frame.drop(columns=['subject_id'], errors='ignore')
                    
                    if verbose or DEBUG_MODE:
                        print(f"   🔗 patients 表与 icustays 关联: {len(frame)} 行")
                        
                    # 更新 table 对象
                    table = ICUTable(
                        data=frame,
                        id_columns=['stay_id'],
                        index_column=table.index_column if table.index_column in frame.columns else None,
                        value_column=table.value_column,
                        unit_column=table.unit_column,
                    )
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"   ⚠️  patients 表关联失败: {e}")
            
            # MIMIC-IV特殊处理：若表为labevents/microbiologyevents/inputevents，仅有subject_id，按时间窗口映射到对应ICU stay
            if DEBUG_MODE:
                print(f"   📊 加载后数据: {source.table}, 行数={len(frame)}, itemid过滤={source.ids}")
                if source.ids and source.sub_var and source.sub_var in frame.columns:
                    print(f"       - {source.sub_var} 唯一值: {sorted(frame[source.sub_var].unique())[:10]}")
                print(f"       - frame列: {list(frame.columns)}")
                print(f"       - frame前3行:\\n{frame.head(3)}")
            if DEBUG_MODE:
                if DEBUG_MODE: print(f"   🔍 调试 {source.table}: 'subject_id' in frame={('subject_id' in frame.columns)}, 'stay_id' in frame={('stay_id' in frame.columns)}, defaults.id_var={defaults.id_var}")
            if source.table in ['labevents', 'microbiologyevents', 'inputevents'] and 'subject_id' in frame.columns and 'stay_id' not in frame.columns:
                if DEBUG_MODE: print(f"   ➡️  进入 MIMIC-IV 特殊处理: {source.table}")
                try:
                    # 仅加载相关stay的icustays，并携带intime/outtime用于窗口过滤
                    icustay_filters = []
                    # 保存expanded_patient_ids到当前作用域,避免后续locals()检查失效
                    current_expanded_patient_ids = None
                    
                    # 🔥 关键修复: 使用原始 stay_id 而不是 subject_id
                    # 这样避免加载同一患者的所有ICU入住记录
                    if patient_ids:
                        # patient_ids 本身就是 stay_id 列表
                        icustay_filters.append(
                            FilterSpec(column='stay_id', op=FilterOp.IN, value=patient_ids)
                        )
                        if DEBUG_MODE: print(f"   🎯 [icustays] 使用原始 stay_id 过滤: {len(patient_ids)} 个, IDs={patient_ids}")
                    
                    icustays = data_source.load_table('icustays', filters=icustay_filters if icustay_filters else None, verbose=verbose)
                    if hasattr(icustays, 'data'):
                        # 包含hadm_id以便匹配同一住院的数据
                        cols = ['subject_id', 'stay_id', 'hadm_id', 'intime', 'outtime']
                        icu_df = icustays.data[[c for c in cols if c in icustays.data.columns]].drop_duplicates()
                    else:
                        cols = ['subject_id', 'stay_id', 'hadm_id', 'intime', 'outtime']
                        icu_df = icustays[[c for c in cols if c in icustays.columns]].drop_duplicates()
                    
                    if DEBUG_MODE: print(f"   ✅ [icustays] 加载后: {len(icu_df)} stays, stay_id={sorted(icu_df['stay_id'].unique())[:10]}")
                    
                    # 🔥 CRITICAL FIX: 为了实现 rolling join，需要加载同一 hadm_id 下的所有 stays
                    # 这样才能正确判断数据点属于哪个 stay
                    if 'hadm_id' in icu_df.columns and 'hadm_id' in frame.columns and len(icu_df) > 0:
                        target_hadm_ids = icu_df['hadm_id'].unique().tolist()
                        # 加载同一 hadm_id 下的所有 stays（用于 rolling join 时间边界判断）
                        all_stays_in_hadm = data_source.load_table(
                            'icustays',
                            filters=[FilterSpec(column='hadm_id', op=FilterOp.IN, value=target_hadm_ids)],
                            verbose=False
                        )
                        if hasattr(all_stays_in_hadm, 'data'):
                            all_stays_df = all_stays_in_hadm.data[[c for c in cols if c in all_stays_in_hadm.data.columns]].drop_duplicates()
                        else:
                            all_stays_df = all_stays_in_hadm[[c for c in cols if c in all_stays_in_hadm.columns]].drop_duplicates()
                        
                        if len(all_stays_df) > len(icu_df):
                            if DEBUG_MODE: print(f"   🔄 [Rolling Join准备] 同一 hadm_id 下有更多 stays: {len(icu_df)} → {len(all_stays_df)}")
                            # 用完整的 stays 列表替换 icu_df，用于后续 rolling join
                            icu_df = all_stays_df

                    # 选择用于时间匹配的列
                    time_col = None
                    if index_column and index_column in frame.columns:
                        time_col = index_column
                    else:
                        # 对于 inputevents，优先使用 starttime
                        if source.table == 'inputevents':
                            for cand in ['starttime', 'charttime', 'storetime']:
                                if cand in frame.columns:
                                    time_col = cand
                                    break
                        else:
                            for cand in ['charttime', 'storetime', 'specimen_time']:
                                if cand in frame.columns:
                                    time_col = cand
                                    break

                    if time_col is not None:
                        # 规范时间类型
                        frame[time_col] = pd.to_datetime(frame[time_col], errors='coerce', utc=True).dt.tz_localize(None)
                        icu_df['intime'] = pd.to_datetime(icu_df['intime'], errors='coerce', utc=True).dt.tz_localize(None)
                        icu_df['outtime'] = pd.to_datetime(icu_df['outtime'], errors='coerce', utc=True).dt.tz_localize(None)

                        # 先按subject_id合并，如果有hadm_id则同时匹配
                        # 修复：只保留同一住院（hadm_id）的数据，避免混入患者其他住院的历史数据
                        if 'hadm_id' in frame.columns and 'hadm_id' in icu_df.columns:
                            # 同时匹配subject_id和hadm_id，确保只取同一次住院的数据
                            tmp = frame.merge(icu_df, on=['subject_id', 'hadm_id'], how='inner')
                        else:
                            # 如果没有hadm_id，只能按subject_id匹配（可能混入其他住院数据）
                            tmp = frame.merge(icu_df, on='subject_id', how='inner')
                        
                        # CRITICAL FIX: 实现 ricu 的 rolling join 逻辑
                        # 当同一个 hadm_id/subject_id 有多个 stay_id 时，数据会被复制到所有匹配的 stay_id
                        # 需要根据时间将数据只保留在正确的 stay_id 下
                        # ricu 使用 roll = -Inf (向前滚动)：数据分配给时间之后最近的 stay_id
                        target_stay_ids = set(patient_ids) if patient_ids else None
                        
                        if time_col is not None and 'stay_id' in tmp.columns and 'intime' in tmp.columns and len(tmp) > 0:
                            # 获取所有唯一的 stay_id 及其 intime，按 intime 排序
                            stay_info = tmp[['stay_id', 'intime']].drop_duplicates().sort_values('intime')
                            
                            if len(stay_info) > 1:
                                # 有多个 stay_id，需要实现 rolling join
                                stays_list = stay_info['stay_id'].tolist()
                                intimes_list = stay_info['intime'].tolist()
                                
                                if DEBUG_MODE:
                                    print(f"      🔄 [Rolling Join] 检测到多个 stay_id: {stays_list}")
                                    print(f"      🔄 [Rolling Join] 对应 intime: {intimes_list}")
                                    print(f"      🔄 [Rolling Join] 目标 stay_id: {target_stay_ids}")
                                
                                # 为每个 stay_id 计算其有效时间范围
                                # stay_i 的有效范围是: [prev_stay_outtime, next_stay_intime)
                                # 但使用 roll = -Inf 意味着：data_time < next_stay_intime
                                
                                result_frames = []
                                for i, (stay_id, intime) in enumerate(zip(stays_list, intimes_list)):
                                    # 只处理用户请求的 stay_id
                                    if target_stay_ids and stay_id not in target_stay_ids:
                                        continue
                                    
                                    # 过滤属于当前 stay_id 的行
                                    stay_mask = tmp['stay_id'] == stay_id
                                    
                                    if i < len(stays_list) - 1:
                                        # 不是最后一个 stay，数据时间必须小于下一个 stay 的 intime
                                        next_intime = intimes_list[i + 1]
                                        time_mask = tmp[time_col] < next_intime
                                        stay_data = tmp[stay_mask & time_mask].copy()
                                        if DEBUG_MODE:
                                            print(f"      🔄 [Rolling Join] stay_id={stay_id}: time < {next_intime}, 保留 {len(stay_data)} 行")
                                    else:
                                        # 最后一个 stay，没有时间上限
                                        stay_data = tmp[stay_mask].copy()
                                        if DEBUG_MODE:
                                            print(f"      🔄 [Rolling Join] stay_id={stay_id}: 最后一个stay, 保留 {len(stay_data)} 行")
                                    
                                    result_frames.append(stay_data)
                                
                                if result_frames:
                                    tmp = pd.concat(result_frames, ignore_index=True)
                                    if DEBUG_MODE:
                                        print(f"      🔄 [Rolling Join] 多 stay_id 时间过滤完成: {len(tmp)} 行")
                        
                        # 确保只保留用户请求的 stay_id（防止遗漏过滤）
                        if target_stay_ids and 'stay_id' in tmp.columns:
                            before_filter = len(tmp)
                            tmp = tmp[tmp['stay_id'].isin(target_stay_ids)]
                            if DEBUG_MODE and len(tmp) != before_filter:
                                print(f"      🎯 [最终过滤] 只保留目标 stay_id: {before_filter} → {len(tmp)} 行")

                        # CRITICAL FIX: Use ICU outtime as upper bound
                        # ricu.R uses ICU discharge (outtime) as the time window, NOT hospital discharge
                        # See ricu/R/data-utils.R: id_win_helper.miiv_env uses icustay's intime/outtime
                        before_filter = len(tmp)

                        # Debug output for ICU window filter
                        if DEBUG_MODE:
                            print(f"      🏥 [ICU窗口] 开始处理: 表={source.table}, 行数={len(tmp)}")
                            if 'outtime' in tmp.columns:
                                print(f"      🏥 [ICU窗口] tmp包含outtime: {tmp['outtime'].notna().sum()}个有效值")
                            else:
                                print(f"      🏥 [ICU窗口] ❌ tmp不包含outtime列!")

                        # Use ICU outtime for filtering (ricu.R behavior)
                        # Data points after ICU discharge should be excluded
                        if 'outtime' in tmp.columns:
                            mask_time = tmp['outtime'].isna() | (tmp[time_col] <= tmp['outtime'])
                            tmp = tmp[mask_time].copy()
                            filter_type = "ICU出院窗口"
                        else:
                            filter_type = "无时间过滤(缺少outtime)"

                        after_filter = len(tmp)
                        # 只在调试模式下打印时间过滤信息
                        if DEBUG_MODE and before_filter > after_filter:
                            print(f"      ⏱️  [{concept_name}] ricu.R-style时间过滤 ({filter_type}): {source.table} 从 {before_filter} 行 → {after_filter} 行")
                        
                        # CRITICAL FIX: 无论tmp是否为空，都要更新frame
                        # 如果tmp为空（没有匹配的数据或被时间过滤），frame也应该为空
                        if not tmp.empty:
                            # 将过滤后的数据作为新frame，仅保留必要列
                            frame = tmp.drop(columns=['intime', 'outtime'])
                            if DEBUG_MODE: print(f"   ✅ [{concept_name}] MIMIC-IV {source.table}: 合并+过滤后 {len(frame)} 行")
                        else:
                            # tmp为空的原因可能是：1) 没有匹配的住院数据，2) 时间过滤后为空
                            # 这是正常的数据过滤行为（例如实验室结果在ICU出院后采集，或在miiv中是ICU入院前的数据）
                            if DEBUG_MODE:
                                reason = "ricu.R-style时间过滤" if before_filter > 0 else "ICU住院匹配"
                                print(f"   ⚠️  [{concept_name}] MIMIC-IV {source.table}: {reason}后为空 (原始{len(frame)}行 → 匹配{before_filter}行 → 过滤后0行)")
                            frame = pd.DataFrame(columns=frame.columns)
                            
                        # 🔗 关键修复：如果用户提供了特定的 stay_id，在映射后再次过滤
                        # 确保只返回用户指定的 stay_id 的数据
                        if 'stay_id' in frame.columns and patient_ids:
                            # 使用之前保存的current_expanded_patient_ids
                            if current_expanded_patient_ids and isinstance(current_expanded_patient_ids, dict) and 'stay_id' in current_expanded_patient_ids:
                                specified_stay_ids = current_expanded_patient_ids['stay_id']
                                if specified_stay_ids:
                                    before_stay_filter = len(frame)
                                    frame = frame[frame['stay_id'].isin(specified_stay_ids)].copy()
                                    if DEBUG_MODE and before_stay_filter > len(frame):
                                        print(f"      🔍 [{concept_name}] stay_id过滤: {before_stay_filter}行 → {len(frame)}行 (保留{len(specified_stay_ids)}个stay_id)")
                        
                        if defaults.id_var == 'subject_id' and 'stay_id' in frame.columns:
                                id_columns = ['stay_id']
                                if DEBUG_MODE: print(f"   🔄 MIMIC-IV特殊处理: {source.table} ID列从 subject_id → stay_id (行数: {len(frame)})")
                    else:
                        # 没有明确时间列，退化为subject级合并（可能产生冗余），但仍补充stay_id
                        frame = frame.merge(icu_df[['subject_id', 'stay_id']], on='subject_id', how='inner')
                        if defaults.id_var == 'subject_id' and 'stay_id' in frame.columns:
                            id_columns = ['stay_id']
                            if DEBUG_MODE: print(f"   🔄 MIMIC-IV特殊处理(无时间列): {source.table} ID列从 subject_id → stay_id (行数: {len(frame)})")
                except Exception as ex:
                    print(f"⚠️  Warning: Failed to time-map labevents to icu stays: {ex}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    # 失败时不做强制映射，保持原逻辑
            
            # MIMIC-IV特殊处理：admissions表只有subject_id和hadm_id，需要映射到stay_id
            if source.table == 'admissions' and 'subject_id' in frame.columns and 'stay_id' not in frame.columns:
                if DEBUG_MODE: print(f"   ➡️  进入 MIMIC-IV admissions特殊处理")
                try:
                    # 加载icustays获取subject_id→hadm_id→stay_id映射
                    icustay_filters = []
                    current_expanded_patient_ids = None
                    if patient_ids:
                        current_expanded_patient_ids = self._expand_patient_ids(
                            patient_ids, 
                            'subject_id',
                            data_source,
                            verbose=False
                        )
                        subj_vals = current_expanded_patient_ids.get('subject_id') if isinstance(current_expanded_patient_ids, dict) else current_expanded_patient_ids
                        if subj_vals:
                            icustay_filters.append(
                                FilterSpec(column='subject_id', op=FilterOp.IN, value=subj_vals)
                            )
                    
                    icustays = data_source.load_table('icustays', filters=icustay_filters if icustay_filters else None, verbose=verbose)
                    if hasattr(icustays, 'data'):
                        icu_df = icustays.data[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()
                    else:
                        icu_df = icustays[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()
                    
                    # 通过hadm_id映射到stay_id（admissions是hospital级别，icustays是ICU级别）
                    if 'hadm_id' in frame.columns and 'hadm_id' in icu_df.columns:
                        before_merge = len(frame)
                        frame = frame.merge(icu_df[['hadm_id', 'stay_id']], on='hadm_id', how='inner')
                        if DEBUG_MODE:
                            print(f"      🏥 [{concept_name}] admissions→icustays映射: {before_merge}行 → {len(frame)}行")
                        
                        # 最终stay_id过滤
                        if patient_ids and current_expanded_patient_ids and isinstance(current_expanded_patient_ids, dict) and 'stay_id' in current_expanded_patient_ids:
                            specified_stay_ids = current_expanded_patient_ids['stay_id']
                            if specified_stay_ids:
                                before_stay_filter = len(frame)
                                frame = frame[frame['stay_id'].isin(specified_stay_ids)].copy()
                                if DEBUG_MODE and before_stay_filter > len(frame):
                                    print(f"      🔍 [{concept_name}] stay_id过滤: {before_stay_filter}行 → {len(frame)}行")
                        
                        if defaults.id_var == 'subject_id' and 'stay_id' in frame.columns:
                            id_columns = ['stay_id']
                            if DEBUG_MODE: print(f"   🔄 MIMIC-IV特殊处理: admissions ID列从 subject_id → stay_id")
                except Exception as ex:
                    print(f"⚠️  Warning: Failed to map admissions to icu stays: {ex}")
                    if verbose:
                        import traceback
                        traceback.print_exc()

            # 如果配置中没有ID列，尝试从数据中自动检测
            if not table.id_columns:
                # 检查数据中是否有常见的ID列
                # 🔧 修复: 根据数据库类型优先选择合适的ID列
                db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                
                # 数据库特定的ID列优先顺序
                if db_name in ['aumc']:
                    common_id_cols = ['admissionid', 'patientid']
                elif db_name in ['eicu', 'eicu_demo']:
                    common_id_cols = ['patientunitstayid', 'patientid']
                elif db_name in ['hirid']:
                    common_id_cols = ['patientid']
                else:
                    # MIMIC-IV 等
                    common_id_cols = ['stay_id', 'icustay_id', 'hadm_id', 'subject_id']
                
                found_id_cols = [col for col in common_id_cols if col in frame.columns]
                if found_id_cols:
                    # 使用第一个找到的ID列（已按数据库优先顺序排列）
                    preferred_id = found_id_cols[0]
                    id_columns = [preferred_id]
                    if DEBUG_MODE: print(f"   🔍 自动检测到ID列: {preferred_id} (db={db_name})")
            else:
                id_columns = id_columns or list(table.id_columns)
            
            # 每个源使用自己的 index_column 和 unit_column
            # 不要复用循环外的变量，避免多源概念时第一个源的配置覆盖后续源
            source_index_column = source.index_var or table.index_column
            source_unit_column = source.unit_var or table.unit_column
            
            # 更新全局 index_column 和 unit_column（用于后续的时间对齐等操作）
            # 但确保每个源处理时使用自己的配置
            if not index_column:
                index_column = source_index_column
            if not unit_column:
                unit_column = source_unit_column

            time_columns = list(
                {
                    *time_columns,
                    *(table.time_columns or []),
                    *( [source_index_column] if source_index_column else []),
                }
            )

            # 处理 dur_var：如果指定了 dur_var="endtime"，计算 duration = endtime - starttime
            # 参考 R ricu load_win.R 中的 dur_is_end 逻辑:
            # if (dur_is_end) {
            #   res <- res[, c(dur_var) := get(dur_var) - get(index_var)]
            # }
            if source.dur_var and source.dur_var in frame.columns:
                if source_index_column and source_index_column in frame.columns:
                    duration_col = concept_name + '_dur'
                    dur_is_end = False  # 是否需要计算 duration = endtime - starttime
                    
                    # Case 1: datetime 类型的 endtime
                    if pd.api.types.is_datetime64_any_dtype(frame[source.dur_var]):
                        dur_is_end = True
                        # 确保 starttime 也是 datetime
                        if not pd.api.types.is_datetime64_any_dtype(frame[source_index_column]):
                            frame[source_index_column] = pd.to_datetime(frame[source_index_column], errors='coerce')
                        
                        # 计算 duration (timedelta)
                        frame[duration_col] = frame[source.dur_var] - frame[source_index_column]
                    
                    # Case 2: 数值类型的 endtime (如 AUMC 的毫秒时间)
                    # 检测：如果 dur_var 是数值且通常大于 index_var，说明是 endtime
                    elif pd.api.types.is_numeric_dtype(frame[source.dur_var]) and \
                         pd.api.types.is_numeric_dtype(frame[source_index_column]):
                        # 检查 dur_var 是否大于 index_var（表示它是 endtime）
                        # 使用抽样检查以提高性能
                        sample_size = min(100, len(frame))
                        if sample_size > 0:
                            sample = frame.head(sample_size)
                            dur_vals = pd.to_numeric(sample[source.dur_var], errors='coerce')
                            idx_vals = pd.to_numeric(sample[source_index_column], errors='coerce')
                            valid_mask = dur_vals.notna() & idx_vals.notna()
                            if valid_mask.sum() > 0:
                                # 如果大部分 dur_var > index_var，则认为是 endtime
                                ratio = (dur_vals[valid_mask] > idx_vals[valid_mask]).mean()
                                if ratio > 0.8:  # 80% 以上的值满足 dur_var > index_var
                                    dur_is_end = True
                                    # 计算 duration = endtime - starttime (数值)
                                    # 结果单位与 start/stop 相同：
                                    # - AUMC: 分钟（datasource.py 已将 ms 转为分钟）
                                    # - eICU: 分钟（offset 列本身就是分钟）
                                    frame[duration_col] = frame[source.dur_var] - frame[source_index_column]
                                    
                                    # 🔧 FIX: 将 duration 从分钟转换为小时
                                    # 这与 _align_time_to_admission 对 start/stop 的转换保持一致
                                    db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                                    if db_name in ['eicu', 'eicu_demo', 'aumc']:
                                        frame[duration_col] = frame[duration_col] / 60.0
                                        if DEBUG_MODE:
                                            print(f"   🔧 {db_name}: 将 {duration_col} 从分钟转换为小时")
                                    
                                    if DEBUG_MODE:
                                        print(f"   🔧 AUMC dur_is_end=True: {source.dur_var}={dur_vals.head(3).tolist()}, "
                                              f"{source_index_column}={idx_vals.head(3).tolist()}")
                    
                    if dur_is_end and DEBUG_MODE:
                        print(f"   dur_var '{source.dur_var}' → duration '{duration_col}' (示例: {frame[duration_col].head(1).tolist()})")

            value_column = source.value_var or table.value_column
            if value_column is None:
                raise ValueError(
                    f"Concept '{concept_name}' has no value column in table "
                    f"'{source.table}'. Provide 'value_var' in the dictionary."
                )

            # 检查是否有 apply_map(var='sub_var') 回调
            # 这种情况下，应该使用映射后的 sub_var 作为最终的值列
            uses_sub_var_mapping = False
            if source.callback and 'apply_map' in source.callback and 'var' in source.callback:
                # 匹配 apply_map(..., var='sub_var') 或 apply_map(..., var="sub_var")
                match = re.search(r"var\s*=\s*['\"]sub_var['\"]", source.callback)
                if match and source.sub_var:
                    uses_sub_var_mapping = True

            # 如果value_column不在frame中，可能需要先创建（例如从callback创建）
            # 先检查value_column是否存在，如果不存在，可能需要通过callback创建
            if DEBUG_MODE:
                print(f"   🔎 重命名前: value_column={value_column}, 在frame中={value_column in frame.columns}, frame行数={len(frame)}")
            
            # 标记回调是否已被应用，避免重复调用
            callback_applied = False
            
            if value_column not in frame.columns:
                # 对于某些概念（如lgl_cncpt），value_column可能通过callback创建
                # 先尝试应用callback，然后再检查
                frame = _apply_callback(
                    frame,
                    source,
                    concept_name,
                    unit_column,
                    resolver=self,
                    patient_ids=patient_ids,
                    data_source=data_source,
                )
                callback_applied = True  # 标记回调已应用
                # 如果callback创建了concept_name，更新value_column
                if concept_name in frame.columns:
                    value_column = concept_name
                elif value_column not in frame.columns:
                    # 如果仍然不存在，跳过这个源
                    if DEBUG_MODE:
                        print(f"   ⚠️  value_column '{value_column}' 不存在，跳过此源")
                    frame = pd.DataFrame()
                    continue

            rename_map = {value_column: concept_name}
            frame = frame.rename(columns=rename_map)
            
            if DEBUG_MODE:
                print(f"   🔄 重命名后: concept_name={concept_name}, 在frame中={concept_name in frame.columns}, frame行数={len(frame)}")

            # If unit_column is specified but not in frame, set to None
            # This can happen if callbacks don't preserve unit columns
            if unit_column and unit_column not in frame.columns:
                unit_column = None

            if source.regex:
                # 确定 regex 应该应用在哪一列：
                # - 如果同时存在 ids 和 regex，ids 用于过滤 sub_var，regex 用于过滤 value_var
                # - 如果只有 regex（没有 ids），regex 用于过滤 sub_var（ricu 的 rgx_itm 行为）
                # 注意：此时 value_var 可能已被重命名为 concept_name，需要检查两者
                if source.ids is not None and source.value_var:
                    # 混合模式：ids 过滤 sub_var，regex 过滤 value_var
                    # 但 value_var 可能已被重命名为 concept_name
                    if source.value_var in frame.columns:
                        regex_column = source.value_var
                    elif concept_name in frame.columns:
                        # value_var 已被重命名为 concept_name
                        regex_column = concept_name
                    else:
                        regex_column = source.value_var  # 会在下面触发跳过
                else:
                    # 标准 rgx_itm 模式：regex 过滤 sub_var
                    # 🔧 FIX: 如果 sub_var == value_var，则 sub_var 已被重命名为 concept_name
                    # 需要使用 concept_name 而不是 sub_var
                    if source.sub_var == source.value_var:
                        regex_column = concept_name
                    elif source.sub_var in frame.columns:
                        regex_column = source.sub_var
                    else:
                        # sub_var 可能被重命名了，尝试 concept_name
                        regex_column = concept_name if concept_name in frame.columns else source.sub_var
                
                if not regex_column:
                    raise ValueError(
                        f"Concept '{concept_name}' specifies a regex but no column to match against."
                    )
                if regex_column not in frame.columns:
                    # 如果目标列不存在，跳过这个源
                    if DEBUG_MODE:
                        print(f"   ⚠️ regex 列 '{regex_column}' 不存在，跳过此源")
                    frame = pd.DataFrame()
                    continue
                # 使用 regex=True 并抑制 UserWarning
                # str.contains 会警告如果正则表达式有捕获组但没有使用 str.extract
                # 这里我们只需要匹配，不需要提取，所以抑制这个警告
                pattern = source.regex
                series = frame[regex_column].astype(str)
                before_regex = len(frame)
                # 使用 regex=True, na=False, 并抑制捕获组警告
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 
                                          message='This pattern is interpreted as a regular expression',
                                          category=UserWarning)
                    frame = frame[series.str.contains(pattern, case=False, na=False, regex=True)]
                if DEBUG_MODE:
                    print(f"   ✓ regex 过滤 (列={regex_column}, pattern='{pattern}'): {before_regex} → {len(frame)} 行")

            # ⚠️ CRITICAL FIX: 必须先应用回调函数（如 convert_unit）再应用值范围过滤和单位过滤
            # 原因：temp 等概念可能需要先将华氏度转换为摄氏度，然后再过滤 32-42°C 的范围和 C/°C 单位
            # 如果先过滤，华氏度值（97-100°F）和单位（°F）会因为不符合要求而被误删
            
            # 应用回调（在值范围过滤和单位过滤之前）
            # 只有当回调尚未被应用时才调用（避免重复调用导致duration变为0的问题）
            if not callback_applied:
                frame = _apply_callback(
                    frame,
                    source,
                    concept_name,
                    source_unit_column,
                    resolver=self,
                    patient_ids=patient_ids,
                    data_source=data_source,
                )
            
            # 单位过滤（在回调之后）
            if definition.units and source_unit_column and source_unit_column in frame.columns:
                allowed_units = {unit.lower() for unit in definition.units}
                
                # 单位归一化：处理等价单位
                # 例如 '10^9/l' 等价于 'G/l' (Giga = 10^9)
                # 🔧 mcL 和 uL 是等价单位：micro-Liter
                unit_equivalents = {
                    '10^9/l': 'g/l',
                    '10^9/L': 'g/l',
                    '10e9/l': 'g/l',
                    'K/ul': 'k/ul',  # 大小写归一化
                    'K/mcL': 'k/ul',  # eICU uses mcL instead of uL (microliter)
                    'k/mcl': 'k/ul',  # eICU uses mcL instead of uL (microliter)
                    '10^3/mcL': '10(3)/mcl',  # Alternative notation
                    '10^3/uL': '10(3)/mcl',   # Alternative notation
                    # eICU 单位归一化
                    'Units/L': 'iu/l',  # eICU uses 'Units/L' for enzyme activities (ALP, ALT, AST, etc.)
                    'units/l': 'iu/l',  # eICU uses 'Units/L' for enzyme activities
                    'U/L': 'iu/l',      # Common alternative
                    'u/l': 'iu/l',      # Common alternative (lowercase)
                    # AUMC 荷兰语单位
                    'ie': 'units',  # Internationale Eenheden (国际单位)
                    'ie/uur': 'units/hr',  # 单位/小时
                    'iu': 'units',  # International Units
                    'iu/hr': 'units/hr',
                }
                
                # 🔧 CRITICAL: 对于 AUMC 数据库，放宽单位匹配
                # AUMC 使用荷兰语单位（如 IE 代表国际单位）
                # 并且某些概念（如 ins）使用 dose 列但概念定义期望 units/hr
                # 为保持与 R ricu 一致，对 AUMC 禁用严格单位过滤
                db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                skip_unit_filter = db_name == 'aumc'
                
                if skip_unit_filter:
                    # AUMC: 跳过严格单位过滤，但仍记录调试信息
                    if DEBUG_MODE:
                        series = frame[source_unit_column].astype(str).str.strip()
                        print(f"   ⚠️ AUMC: 跳过单位过滤 (原单位: {series.unique()[:5]}, 期望: {definition.units})")
                else:
                    # 非 AUMC 数据库：应用严格单位过滤
                    # 归一化数据中的单位
                    series = frame[source_unit_column].astype(str).str.strip()
                    normalized_series = series.replace(unit_equivalents).str.lower()
                
                    # 🔧 进一步归一化：去除非字母数字字符后比较
                    # 这处理了 mmHg 的各种变体：mm Hg, mm/Hg, mm(hg), mm[Hg] 等
                    # NOTE: re 模块已在文件顶部导入，不要在此处重新导入，否则会导致 UnboundLocalError
                    def normalize_unit_for_comparison(unit_str):
                        """归一化单位字符串，仅保留字母数字字符"""
                        if not unit_str or pd.isna(unit_str) or unit_str in ['', 'none', 'None', 'nan']:
                            return ''
                        return re.sub(r'[^a-z0-9]', '', str(unit_str).lower())
                    
                    normalized_allowed = {normalize_unit_for_comparison(u) for u in definition.units}
                    normalized_data = normalized_series.apply(normalize_unit_for_comparison)

                    # 处理None/空字符串单位的情况
                    # 对于FiO2等数据，valueuom=None时应该保留数据，而不是过滤掉
                    # 将'none'和空字符串视为匹配任何单位
                    # 🔧 FIX: 添加 'geen' (荷兰语 "无") 和其他无单位标记的支持
                    # AUMC 数据使用 'Geen' 表示无单位（如 sao2 的 0.xx 格式值）
                    no_unit_markers = {'', 'none', 'geen', 'null', 'na', 'n/a', '-'}
                    mask = (
                        normalized_series.isin(allowed_units) |  # 原始比较
                        normalized_data.isin(normalized_allowed) |  # 归一化比较
                        (normalized_series.isin(no_unit_markers))  # 无单位标记
                    )

                    before_unit = len(frame)
                    frame = frame[mask]
                    if before_unit != len(frame) and DEBUG_MODE:
                        print(f"   ✓ 单位过滤 (允许{definition.units}): {before_unit} → {len(frame)} 行")
            
            # 只有在concept_name列存在时才dropna
            # 但不要过早删除，因为某些回调函数可能会处理NaN值
            # 只在明确需要时才删除NaN（例如，在应用min/max过滤之前）
            if concept_name in frame.columns:
                # 先不删除NaN，因为某些概念（如urine24）可能需要保留NaN
                # 只在值范围过滤之前删除明显无效的NaN
                # 但如果值范围已定义，可以在过滤后删除NaN
                pass  # 暂时不删除NaN，让后续处理决定

            # 值范围过滤（在回调之后）
            # 现在值已经经过转换（如华氏度→摄氏度），可以安全过滤
            if definition.minimum is not None:
                # 确保列是数值类型，避免字符串比较错误
                if concept_name in frame.columns:
                    frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                frame = frame[frame[concept_name] >= definition.minimum]
            if definition.maximum is not None:
                # 确保列是数值类型
                if concept_name in frame.columns:
                    frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                frame = frame[frame[concept_name] <= definition.maximum]
            
            # 在值范围过滤后，删除无效的NaN（但保留有效范围内的NaN用于后续处理）
            # 🔧 关键修复：在merge模式下保留NaN行，以匹配ricu的完整时间网格风格
            if concept_name in frame.columns:
                # 检查是否在merge模式（通过kwargs传递）
                keep_na_rows = kwargs.get('_keep_na_rows', False)
                if not keep_na_rows:
                    # 只在非merge模式下删除NaN（单独加载概念时）
                    frame = frame.dropna(subset=[concept_name])
                # 在merge模式下，保留NaN行以便后续合并时创建完整时间网格

            # 如果使用了 apply_map(var='sub_var')，将映射后的 sub_var 复制到 concept_name
            if uses_sub_var_mapping and source.sub_var in frame.columns:
                # sub_var 列已经被 apply_map 映射为类别值，将其复制到 concept_name 列
                # 但是要先保存原始的数值列（如果需要的话，用于后续持续时间计算）
                # 对于 mech_vent 这种概念，原始 value 列包含持续时间，需要保留
                if concept_name in frame.columns and source.value_var:
                    # 保存原始数值列为 _duration_val
                    frame['_duration_val'] = frame[concept_name]
                # 将映射后的类别值复制到 concept_name
                frame[concept_name] = frame[source.sub_var]

            # DEBUG: 在keep_cols过滤前打印
            keep_cols = {
                *(id_columns or []),
                *( [source_index_column] if source_index_column else []),
                concept_name,
            }
            # 添加实际存在的time_columns（不强制要求所有time_columns都存在）
            for tc in (time_columns or []):
                if tc in frame.columns:
                    keep_cols.add(tc)
            
            if source_unit_column and source_unit_column in frame.columns:
                keep_cols.add(source_unit_column)
            
            # 保留 _duration_val 列（如果存在），用于后续持续时间计算
            if '_duration_val' in frame.columns:
                keep_cols.add('_duration_val')
            
            # 保留 duration 列（如果存在），用于 WinTbl
            # duration列通常命名为 concept_name + '_dur'
            duration_col_name = concept_name + '_dur'
            if duration_col_name in frame.columns:
                keep_cols.add(duration_col_name)
            
            # 只检查必需的列：id_columns, index_column, concept_name
            # 注意：对于多源概念，不同源可能使用不同的时间列名（如starttime vs charttime）
            # 所以对于索引列，我们只检查是否在数据中有任何时间列
            required_cols = {
                *(id_columns or []),
                concept_name,
            }
            missing = required_cols - set(frame.columns)
            
            # 对于索引列，检查是否在数据中有任何时间列
            if source_index_column:
                # 检查是否有source_index_column，或者有类似的时间列
                time_aliases = {"starttime", "endtime", "charttime", "storetime"}
                time_cols = []
                for col in frame.columns:
                    if not isinstance(col, str):
                        continue
                    lowered = col.lower()
                    if "time" in lowered or lowered in time_aliases:
                        time_cols.append(col)
                if source_index_column not in frame.columns and not time_cols:
                    missing.add(source_index_column)
            
            if missing:
                # 对于labevents等表，如果缺少stay_id但映射过程已处理，应该已经有stay_id了
                # 如果还是没有，说明映射失败，跳过这个源并继续（不报错）
                if 'stay_id' in missing and source.table in ['labevents', 'microbiologyevents']:
                    frame = pd.DataFrame()
                    continue
                # 对于eICU的infusiondrug表，patientunitstayid在某些情况下可能被过滤掉
                # 这是由于eICU数据处理管道的特殊性造成的，我们应该放宽要求
                if (hasattr(data_source, 'config') and
                    hasattr(data_source.config, 'name') and
                    data_source.config.name in ['eicu', 'eicu_demo'] and
                    source.table == 'infusiondrug' and
                    missing.issubset({'patientunitstayid', 'infusiondrugid', 'volumeoffluid'})):
                    logging.debug(f"eICU infusiondrug missing ID columns {missing}, but continuing with available data")
                    missing.discard('patientunitstayid')
                    missing.discard('infusiondrugid')
                    missing.discard('volumeoffluid')

                # 对于多源概念，如果某个源缺少index_column但其他源有，这是可以接受的
                if source_index_column in missing and len(sources) > 1:
                    missing.discard(source_index_column)

                if missing:
                    raise KeyError(
                        f"Missing expected columns {sorted(missing)} in concept "
                        f"data for '{concept_name}' (table '{source.table}')"
                    )
            # 确保ID列在数据中
            available_id_cols = [col for col in id_columns if col in frame.columns]
            if not available_id_cols and id_columns:
                logging.debug(f"配置的ID列 {id_columns} 不在数据中，可用列: {list(frame.columns)[:10]}")
            
            ordered_cols: List[str] = []
            # 保留所有可用的ID列（不只是第一个）
            ordered_cols.extend(available_id_cols)
            if source_index_column and source_index_column not in ordered_cols:
                ordered_cols.append(source_index_column)
            extra_time = [
                col for col in time_columns if col and col not in ordered_cols
            ]
            ordered_cols.extend(extra_time)
            ordered_cols.append(concept_name)
            if source_unit_column and source_unit_column not in ordered_cols:
                ordered_cols.append(source_unit_column)
            
            # 添加 duration 列（如果存在）
            duration_col_name = concept_name + '_dur'
            if duration_col_name in frame.columns and duration_col_name not in ordered_cols:
                ordered_cols.append(duration_col_name)
            
            # 🔧 FIX: 保留 endtime 列用于窗口概念展开
            # mech_vent 等窗口概念需要 endtime 来进行时间展开
            # 如果有 dur_var="endtime" 的定义，endtime 列必须保留
            for endtime_candidate in ['endtime', 'end_time', 'stop']:
                if endtime_candidate in frame.columns and endtime_candidate not in ordered_cols:
                    ordered_cols.append(endtime_candidate)
                    break
            
            ordered_cols = [col for col in ordered_cols if col in frame.columns]
            
            # Check and remove duplicate columns before appending
            frame_subset = frame.loc[:, ordered_cols]
            if frame_subset.columns.duplicated().any():
                frame_subset = frame_subset.loc[:, ~frame_subset.columns.duplicated()]
            
            frames.append(frame_subset)

        if not frames:
            # 返回空 DataFrame 而不是报错（某些概念可能在测试数据中没有数据）
            # 检查是否是因为缺少必要的表文件
            missing_tables = []
            db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else 'unknown'

            for source in sources:
                if hasattr(source, 'table'):
                    import os
                    if hasattr(data_source, 'base_path') and data_source.base_path is not None:
                        table_file = data_source.base_path / f"{source.table}.parquet"
                        csv_file = data_source.base_path / f"{source.table}.csv"
                        csv_gz_file = data_source.base_path / f"{source.table}.csv.gz"

                        if not (table_file.exists() or csv_file.exists() or csv_gz_file.exists()):
                            missing_tables.append(source.table)

            if missing_tables and db_name in ['eicu', 'eicu_demo']:
                logging.debug(f"eICU测试数据缺少表 {missing_tables}，概念 '{concept_name}' 暂时不可用")
            else:
                # 只对某些高级治疗概念显示INFO级别信息
                advanced_concepts = ['ecmo', 'ecmo_indication', 'mech_circ_support', 'rrt']
                if concept_name in advanced_concepts:
                    logging.info(f"概念 '{concept_name}' 在测试数据中不可用（高级治疗）")
                else:
                    logging.debug(f"概念 '{concept_name}' 的所有 {len(sources)} 个数据源都返回空数据")
            # 创建一个空的 DataFrame，包含必要的列
            # 确保有 ID 列：使用配置的 id_columns，如果没有则使用数据库的默认ID列
            if not id_columns:
                # 从数据源名称推断默认ID列
                db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else 'unknown'
                id_columns = _default_id_columns_for_db(db_name)
            empty_cols = list(id_columns) + ([index_column] if index_column else []) + [concept_name]
            combined = pd.DataFrame(columns=empty_cols)
        else:
            # Check for duplicate column names before concat
            for i, frame in enumerate(frames):
                if frame.columns.duplicated().any():
                    # Keep only first occurrence of duplicate columns
                    frames[i] = frame.loc[:, ~frame.columns.duplicated()]
            
            # 🔍 DEBUG: 检查每个 frame 的患者数（只在调试模式下显示）
            if DEBUG_MODE and concept_name == 'plt':
                print(f"\\n🔍 [plt合并] 准备合并 {len(frames)} 个 sources:")
                for i, frame in enumerate(frames):
                    if 'stay_id' in frame.columns:
                        print(f"  Source {i+1}: {len(frame)}行, {frame['stay_id'].nunique()}个患者, IDs={sorted(frame['stay_id'].unique())[:5]}")
                    else:
                        print(f"  Source {i+1}: {len(frame)}行, 无stay_id列")
            
            combined = pd.concat(frames, ignore_index=True)
            
        # DEBUG
        # Standardize time column name for eICU BEFORE any processing
        # eICU uses different time column names (labresultoffset, observationoffset, etc.)
        # For multi-source concepts (like abx), different sources may use different offset columns
        # Rename all offset columns to 'charttime' to enable unified processing
        db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
        if db_name in ['eicu', 'eicu_demo'] and index_column:
            # All possible eICU time offset columns
            eicu_time_cols = [
                'labresultoffset', 'observationoffset', 'nursecharting_offset', 
                'respiratorycharting_offset', 'intakeoutput_offset', 'respchartoffset',
                'infusionoffset', 'drugstartoffset', 'drugstopoffset', 'drugorderoffset',
                'culturetakenoffset', 'cultureoffset',
                # 🔥 添加 respiratorycare 表的时间列
                'respcarestatusoffset', 'ventstartoffset', 'ventendoffset',
                'priorventstartoffset', 'priorventendoffset',
            ]
            
            offset_cols_in_data = [col for col in combined.columns if col in eicu_time_cols]
            
            if offset_cols_in_data:
                # 重命名第一个offset列为charttime
                first_offset = offset_cols_in_data[0]
                combined = combined.rename(columns={first_offset: 'charttime'})
                
                # 合并其他offset列到charttime（使用第一个非NaN值）
                for offset_col in offset_cols_in_data[1:]:
                    if offset_col in combined.columns:
                        combined['charttime'] = combined['charttime'].fillna(combined[offset_col])
                        combined = combined.drop(columns=[offset_col])
                
                index_column = 'charttime'  # Update index_column for subsequent processing
        
        sort_keys = [col for col in id_columns if col]
        if index_column:
            sort_keys.append(index_column)
        if sort_keys:
            # 修复：确保sort_keys中的列都存在于combined中
            sort_keys = [k for k in sort_keys if k in combined.columns]
            
            if not sort_keys:
                # 如果没有有效的排序键，跳过排序
                pass
            else:
                # 修复：如果列名重复，先去重
                if combined.columns.duplicated().any():
                    # 保留第一个出现的列，删除重复的
                    combined = combined.loc[:, ~combined.columns.duplicated()]

                # 修复：确保排序键中的列具有一致的类型，避免混合类型排序问题
                try:
                    combined = combined.sort_values(by=sort_keys)
                except TypeError as e:
                    if 'ordered' in str(e) or 'not supported between instances' in str(e):
                        # 处理混合类型排序问题
                        if DEBUG_MODE:
                            print(f"      [排序修复] 检测到混合类型排序问题: {e}")

                        # 尝试逐个检查和修复排序键的类型
                        cleaned_combined = combined.copy()
                        for key in sort_keys:
                            if key in cleaned_combined.columns:
                                # 如果是时间列，确保都是datetime类型
                                if 'time' in key.lower() or key == 'charttime':
                                    try:
                                        cleaned_combined[key] = pd.to_datetime(cleaned_combined[key], errors='coerce')
                                    except:
                                        pass
                                # 如果有混合类型，转换为字符串进行排序
                                else:
                                    try:
                                        # 尝试排序以检测问题
                                        cleaned_combined.sort_values(by=[key])
                                    except TypeError:
                                        if DEBUG_MODE:
                                            print(f"      [排序修复] 列{key}存在混合类型，转换为字符串")
                                        cleaned_combined[key] = cleaned_combined[key].astype(str)

                        # 重新排序
                        combined = cleaned_combined.sort_values(by=sort_keys)
                    else:
                        # 其他类型的错误，重新抛出
                        raise
        combined = combined.reset_index(drop=True)
        agg_value = self._coerce_final_aggregator(aggregator)
        if agg_value in (None, "auto"):
            fallback_agg = definition.aggregate
            if fallback_agg is not None:
                agg_value = self._coerce_final_aggregator(fallback_agg)

        # CRITICAL FIX: Avoid double aggregation issue
        # Strategy: Only use change_interval's aggregation (on relative time after floor)
        # Do NOT use _apply_aggregation before time alignment
        should_aggregate_in_change_interval = agg_value is not False
        
        # 🔧 FIX: 确保 index_column 实际存在于 combined 中
        # 对于 id_tbl 类型的概念（如 los_icu），可能从表配置继承了 index_column，但数据中不包含该列
        if index_column and index_column not in combined.columns:
            index_column = None
        
        # 如果数据为空，返回空 ICUTable
        if combined.empty:
            return ICUTable(
                data=combined,
                id_columns=id_columns,
                index_column=index_column,  # 此时已验证存在或为 None
                value_column=concept_name,
                unit_column=None,
                time_columns=[col for col in time_columns if col and col in combined.columns],
            )
        
        # Only set unit_column if it actually exists in the combined data
        final_unit_column = unit_column if unit_column and unit_column in combined.columns else None
        
        # Apply interval alignment and aggregation if interval is specified
        if interval is not None and index_column and index_column in combined.columns:
            # DEBUG
            from .ts_utils import change_interval
            
            # Align time to ICU admission if requested (BEFORE any aggregation)
            if align_to_admission:
                # DEBUG
                combined = self._align_time_to_admission(
                    combined,
                    data_source,
                    id_columns,
                    index_column
                )
                
                # 🔧 FIX: _align_time_to_admission 可能会删除 intime/outtime 列
                # 需要重新检查 index_column 是否仍在 combined 中
                if index_column and index_column not in combined.columns:
                    # 尝试查找可能的时间列
                    time_cols = [c for c in combined.columns if c in ['start', 'charttime', 'measuredat']]
                    if time_cols:
                        index_column = time_cols[0]
                    else:
                        # 没有有效的时间列，跳过 interval 处理
                        index_column = None
                
                # 如果 index_column 变成 None，跳过后续的 interval 处理
                if index_column is None:
                    # 返回不带 interval 处理的数据
                    return ICUTable(
                        data=combined.reset_index(drop=True),
                        id_columns=id_columns,
                        index_column=None,
                        value_column=concept_name,
                        unit_column=final_unit_column,
                        time_columns=[col for col in time_columns if col and col in combined.columns],
                    )
                
                # DEBUG
            # Determine aggregation method for change_interval
            # This is the ONLY aggregation we should do (on relative time)
            agg_method = agg_value if agg_value not in (None, False, "auto") else None
            if agg_method in (None, "auto"):
                agg_method = None
            # Default aggregation based on value type (matches R ricu)
            if agg_method is None:
                # Check value column type
                if concept_name in combined.columns:
                    col_dtype = combined[concept_name].dtype
                    if pd.api.types.is_bool_dtype(col_dtype):
                        agg_method = 'any'  # R ricu: logical -> "any"
                    elif pd.api.types.is_numeric_dtype(col_dtype):
                        agg_method = 'median'  # R ricu: numeric -> "median"
                    else:
                        agg_method = 'first'  # R ricu: character/other -> "first"
            
            # Create ICUTable temporarily to use change_interval
            temp_table = ICUTable(
                data=combined,
                id_columns=id_columns,
                index_column=index_column,
                value_column=concept_name,
                unit_column=final_unit_column,
                time_columns=[col for col in time_columns if col],
            )

            fill_missing = self._should_fill_gaps(concept_name, definition)
            fill_method = self._get_fill_method(concept_name, definition)
            
            # Apply interval change with aggregation (SINGLE aggregation on relative time)
            combined_result = change_interval(
                temp_table,
                interval=interval,
                aggregation=agg_method,
                fill_gaps=fill_missing,
                fill_method=fill_method,
                copy=False
            )
            
            # Extract data if ICUTable is returned
            if hasattr(combined_result, 'data'):
                combined = combined_result.data
                # 更新index_column：change_interval可能改变了时间列名(如变为'start')
                if hasattr(combined_result, 'index_column') and combined_result.index_column:
                    index_column = combined_result.index_column
            else:
                combined = combined_result
        elif align_to_admission:
            # Just alignment, no interval/aggregation
            combined = self._align_time_to_admission(
                combined,
                data_source,
                id_columns,
                index_column
            )
        
        # 🔧 NOTE: 不过滤负时间（入ICU前的数据），ricu 保留这些数据
        # 例如：AUMC esr measuredat=-2 表示入院前2小时的数据，ricu 也保留
        
        # 最终验证：确保index_column存在于combined中
        if index_column and index_column not in combined.columns:
            # 尝试查找可能的时间列
            time_cols = [c for c in combined.columns if c in ['start', 'charttime', 'measuredat', index_column]]
            if time_cols:
                index_column = time_cols[0]
            else:
                # 没有有效的时间列，设为None
                index_column = None
        
        # CRITICAL: Check if target is 'win_tbl', and convert to WinTbl if needed
        # For concepts like mech_vent that have target='win_tbl' but no concept-level callback
        # DISABLED for now - WinTbl conversion has issues with endtime handling
        # Return raw ICUTable and let expansion happen in _ensure_concept_loaded
        if False and definition.target == 'win_tbl' and interval is not None:
            from .table import WinTbl
            # WinTbl needs: index_var (time), dur_var (duration), id_vars (IDs)
            # Check if we have endtime or duration columns
            has_endtime = any(col in combined.columns for col in ['endtime', 'end_time', 'stop'])
            has_duration = any(col in combined.columns for col in ['duration', 'dur', concept_name + '_dur'])
            
            if has_endtime or has_duration:
                # Find the appropriate columns
                endtime_col = next((col for col in ['endtime', 'end_time', 'stop'] if col in combined.columns), None)
                duration_col = next((col for col in ['duration', 'dur', concept_name + '_dur'] if col in combined.columns), None)
                
                # If we have endtime, calculate duration
                if endtime_col and index_column:
                    # Ensure both are numeric (hours)
                    # endtime might still be datetime if it wasn't properly aligned
                    if pd.api.types.is_datetime64_any_dtype(combined[endtime_col]):
                        # Skip endtime conversion for now - has issues
                        # TODO: Fix endtime handling for procedureevents
                        pass
                    
                    # Now both should be numeric
                    if pd.api.types.is_numeric_dtype(combined[endtime_col]) and pd.api.types.is_numeric_dtype(combined[index_column]):
                        # Calculate duration as endtime - starttime
                        combined[concept_name + '_dur'] = combined[endtime_col] - combined[index_column]
                        duration_col = concept_name + '_dur'
                        # Remove endtime column (WinTbl uses duration, not endtime)
                        combined = combined.drop(columns=[endtime_col], errors='ignore')
                    else:
                        # Can't calculate duration, skip WinTbl conversion
                        duration_col = None
                
                if duration_col and index_column:
                    # Create WinTbl
                    return WinTbl(
                        data=combined,
                        id_vars=id_columns,
                        index_var=index_column,
                        dur_var=duration_col,
                    )
        
        if concept_name == "infusionoffset" and index_column and index_column in combined.columns:
            combined[concept_name] = combined[index_column]
            combined = combined.drop(columns=["drugrate"], errors="ignore")
        try:
            return ICUTable(
                data=combined,
                id_columns=id_columns,
                index_column=index_column,  # Already updated for eICU if needed
                value_column=concept_name,
                unit_column=final_unit_column,
                time_columns=[col for col in time_columns if col],
            )
        except KeyError as exc:
            if concept_name == "infusionoffset" and index_column and index_column in combined.columns:
                combined[concept_name] = combined[index_column]
                combined = combined.drop(columns=["drugrate"], errors="ignore")
                return ICUTable(
                    data=combined,
                    id_columns=id_columns,
                    index_column=index_column,
                    value_column=concept_name,
                    unit_column=final_unit_column,
                    time_columns=[col for col in time_columns if col],
                )
            raise exc
    
    def _align_time_to_admission(
        self,
        data: pd.DataFrame,
        data_source: ICUDataSource,
        id_columns: List[str],
        index_column: str,
        time_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Align time column to ICU admission time as anchor (R ricu as_dt_min).
        
        Converts absolute time to relative time (hours or minutes since ICU admission).
        This replicates R ricu's behavior where time is relative to admission.
        
        Args:
            data: Input DataFrame with time column
            data_source: Data source instance
            id_columns: ID columns (e.g., ['stay_id'])
            index_column: Time column name (e.g., 'charttime')
            time_columns: Additional time columns to convert (e.g., ['stop', 'mech_vent_dur'])
            
        Returns:
            DataFrame with time converted to hours since ICU admission
        """
        # eICU和AUMC时间列需要特殊处理
        # eICU uses offset columns (labresultoffset, observationoffset, etc.) which are
        # already in MINUTES from ICU admission. Convert to HOURS for consistency.
        # AUMC times are ABSOLUTE timestamps in MINUTES (converted from ms in datasource.py).
        # For AUMC, we need to subtract admittedat to get relative time since ICU admission.
        db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
        if db_name in ['eicu', 'eicu_demo']:
            # eICU时间列是相对于入院时间的offset,单位是分钟
            # 转换为小时以与其他数据库保持一致
            
            # 收集所有需要转换的时间列
            cols_to_convert = set()
            if index_column and index_column in data.columns:
                cols_to_convert.add(index_column)
            
            # 添加额外的时间列 (如 stop 等)
            if time_columns:
                for col in time_columns:
                    if col and col in data.columns:
                        if not col.endswith('_dur'):
                            cols_to_convert.add(col)
            
            # 自动检测其他可能的时间列 (start, stop)
            for col in data.columns:
                if col in ['start', 'stop']:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        cols_to_convert.add(col)
            
            # 转换所有时间列（从分钟到小时）
            for col in cols_to_convert:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col] / 60.0
            return data
        
        if db_name == 'aumc':
            # AUMC时间列是绝对时间戳（毫秒，已在datasource.py中转换为分钟）
            # 需要减去 admittedat 得到相对于 ICU 入住的时间
            # 这对于多次入住的患者（如patient 14，admittedat=208661820000ms）很重要
            
            # 收集所有需要转换的时间列
            cols_to_convert = set()
            if index_column and index_column in data.columns:
                cols_to_convert.add(index_column)
            
            if time_columns:
                for col in time_columns:
                    if col and col in data.columns:
                        if not col.endswith('_dur'):
                            cols_to_convert.add(col)
            
            for col in data.columns:
                if col in ['start', 'stop']:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        cols_to_convert.add(col)
            
            if not cols_to_convert:
                return data
            
            # 获取 admittedat 以计算相对时间
            # 对于 AUMC，ID 列是 admissionid
            id_col = 'admissionid' if 'admissionid' in data.columns else (id_columns[0] if id_columns else None)
            
            if id_col and id_col in data.columns:
                try:
                    # 加载 admissions 表获取 admittedat
                    admissions = data_source.load_table('admissions', 
                                                         columns=['admissionid', 'admittedat'], 
                                                         verbose=False)
                    if hasattr(admissions, 'data'):
                        admissions_df = admissions.data
                    else:
                        admissions_df = admissions
                    
                    # admittedat 也是毫秒，需要转换为分钟
                    if 'admittedat' in admissions_df.columns:
                        admissions_df['admittedat_min'] = (admissions_df['admittedat'] / 60000.0).apply(
                            lambda x: int(x) if pd.notna(x) else x).astype('float64')
                        
                        # 合并 admittedat 到数据中
                        data = data.merge(admissions_df[['admissionid', 'admittedat_min']], 
                                         on='admissionid', how='left')
                        
                        # 从时间列中减去 admittedat_min 得到相对时间
                        for col in cols_to_convert:
                            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                                if DEBUG_MODE:
                                    try:
                                        print(f"   🐞 [AUMC _align_time] {col} before subtract: min/max = {data[col].min()} / {data[col].max()}")
                                    except Exception:
                                        pass
                                # 减去 admittedat_min 得到相对分钟
                                data[col] = data[col] - data['admittedat_min']
                                # 转换为小时
                                data[col] = data[col] / 60.0
                                if DEBUG_MODE:
                                    try:
                                        print(f"   🐞 [AUMC _align_time] {col} after subtract & hours: min/max = {data[col].min()} / {data[col].max()}")
                                    except Exception:
                                        pass
                        
                        # 删除辅助列
                        if 'admittedat_min' in data.columns:
                            data = data.drop(columns=['admittedat_min'])
                        
                        return data
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"   ⚠️ [AUMC _align_time] Failed to load admittedat: {e}")
            
            # 回退：如果无法获取 admittedat，只做单位转换
            for col in cols_to_convert:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col] / 60.0
            return data
        
        # Early return checks (no verbose output for performance)
        if data.empty or not index_column or index_column not in data.columns:
            return data
        
        # Get the primary ID column (usually stay_id for MIMIC-IV)
        if not id_columns:
            return data
        
        primary_id = id_columns[0]
        if primary_id not in data.columns:
            return data
        
        # 特殊处理：如果primary_id不是stay_id，需要先join icustays获取stay_id
        # 这对于labevents（使用subject_id）很重要
        if primary_id != 'stay_id' and 'stay_id' not in data.columns:
            try:
                # Use cached icustays table if available
                cache_key = f"{primary_id}_stay_id_intime"
                if self._icustays_cache is None or cache_key not in self._icustays_cache.columns:
                    icustays_temp = data_source.load_table('icustays', columns=[primary_id, 'stay_id', 'intime'], verbose=False)
                if hasattr(icustays_temp, 'data'):
                    icustays_temp_df = icustays_temp.data
                else:
                    icustays_temp_df = icustays_temp
                
                # 确保intime是tz-naive datetime
                if pd.api.types.is_datetime64_any_dtype(icustays_temp_df['intime']):
                    if hasattr(icustays_temp_df['intime'].dt, 'tz') and icustays_temp_df['intime'].dt.tz is not None:
                        icustays_temp_df['intime'] = icustays_temp_df['intime'].dt.tz_localize(None)
                    
                    # Cache the table
                    self._icustays_cache = icustays_temp_df
                else:
                    icustays_temp_df = self._icustays_cache
                
                # Join获取stay_id和intime
                data = data.merge(icustays_temp_df[[primary_id, 'stay_id', 'intime']], 
                                 on=primary_id, how='left')
                
                # 更新primary_id为stay_id
                primary_id = 'stay_id'
                # 已经有intime了，后面不需要再加载
            except Exception as e:
                return data
        
        # 若时间列已是numeric（相对小时），仍尝试按ICU窗口裁剪范围
        if pd.api.types.is_numeric_dtype(data[index_column]):
            try:
                # 确保存在intime/outtime以计算窗口长度（小时）
                if 'intime' not in data.columns or 'outtime' not in data.columns:
                    # Use cached icustays if available, otherwise load
                    if self._icustays_cache is not None and all(c in self._icustays_cache.columns for c in [primary_id, 'intime', 'outtime', 'los']):
                        icu_df = self._icustays_cache.copy()
                    else:
                        icu_cols = [primary_id, 'intime', 'outtime', 'los']
                        icustays_table = data_source.load_table('icustays', columns=icu_cols, verbose=False)
                        icu_df = icustays_table.data if hasattr(icustays_table, 'data') else icustays_table
                        # Cache it
                        self._icustays_cache = icu_df.copy()
                    icu_df['intime'] = pd.to_datetime(icu_df['intime'], errors='coerce', utc=True).dt.tz_localize(None)
                    if 'outtime' in icu_df.columns:
                        icu_df['outtime'] = pd.to_datetime(icu_df['outtime'], errors='coerce', utc=True).dt.tz_localize(None)
                    # 若outtime缺失，尝试用los推断
                    if 'los' in icu_df.columns:
                        los_hours = pd.to_numeric(icu_df['los'], errors='coerce') * 24.0
                        icu_df['outtime_fallback'] = icu_df['intime'] + pd.to_timedelta(los_hours, unit='h')
                        if 'outtime' in icu_df.columns:
                            icu_df['outtime'] = icu_df['outtime'].fillna(icu_df['outtime_fallback'])
                        else:
                            icu_df['outtime'] = icu_df['outtime_fallback']
                    data = data.merge(icu_df[[primary_id] + [c for c in ['intime', 'outtime'] if c in icu_df.columns]], on=primary_id, how='left')

                # 计算ICU窗口长度（小时）
                icu_len_hours = None
                if 'outtime' in data.columns and data['outtime'].notna().any():
                    icu_len = (pd.to_datetime(data['outtime']) - pd.to_datetime(data['intime']))
                    icu_len_hours = icu_len.dt.total_seconds() / 3600.0

                # 修复：R ricu保留所有数据，包括：
                # 1. 入ICU前的数据（负时间）
                # 2. ICU住院期间的数据（0到icu_len_hours）
                # 3. 出ICU后的数据（超过icu_len_hours）
                # 不过滤任何时间数据，完全匹配R ricu的行为
                # 注释掉时间过滤，保留所有原始数据点
                # if icu_len_hours is not None:
                #     mask = data[index_column] <= icu_len_hours
                #     data = data[mask].copy()
                # 清理临时列
                drop_cols = [c for c in ['intime', 'outtime'] if c in data.columns]
                if drop_cols:
                    data = data.drop(columns=drop_cols)
            except Exception as _:
                # 过滤失败则原样返回
                pass
            return data
        
        # 检查时间列是否是有效的datetime类型
        if not pd.api.types.is_datetime64_any_dtype(data[index_column]):
            # 如果不是datetime也不是numeric，尝试转换为datetime
            try:
                data[index_column] = pd.to_datetime(data[index_column], errors='coerce', utc=True).dt.tz_localize(None)
            except Exception as e:
                print(f"  ⚠️  警告: 无法将时间列 {index_column} 转换为datetime: {e}")
                return data
        
        try:
            # 如果已经有intime列（从前面的join得到），直接使用，不需要再次加载
            if 'intime' not in data.columns:
                # Use cached icustays if available
                if self._icustays_cache is not None and all(c in self._icustays_cache.columns for c in [primary_id, 'intime', 'outtime', 'los']):
                    icustays_df = self._icustays_cache.copy()
                else:
                # Load icustays table to get admission times
                    icustays_table = data_source.load_table('icustays', columns=[primary_id, 'intime', 'outtime', 'los'], verbose=False)
                if hasattr(icustays_table, 'data'):
                    icustays_df = icustays_table.data
                else:
                    icustays_df = icustays_table
                    # Cache it
                    self._icustays_cache = icustays_df.copy()
                
                if 'intime' not in icustays_df.columns:
                    # No admission time available, return as-is
                    return data
                
                # Merge with admission times
                admission_times = icustays_df[[primary_id, 'intime', 'outtime', 'los'] if 'los' in icustays_df.columns else [primary_id, 'intime', 'outtime']].copy()
                # 移除时区信息以避免时区不一致错误
                admission_times['intime'] = pd.to_datetime(admission_times['intime'], errors='coerce', utc=True).dt.tz_localize(None)
                if 'outtime' in admission_times.columns:
                    admission_times['outtime'] = pd.to_datetime(admission_times['outtime'], errors='coerce', utc=True).dt.tz_localize(None)
                # 如果outtime缺失，使用los推断
                if 'los' in admission_times.columns:
                    los_hours = pd.to_numeric(admission_times['los'], errors='coerce') * 24.0
                    admission_times['outtime_fallback'] = admission_times['intime'] + pd.to_timedelta(los_hours, unit='h')
                    if 'outtime' in admission_times.columns:
                        admission_times['outtime'] = admission_times['outtime'].fillna(admission_times['outtime_fallback'])
                    else:
                        admission_times['outtime'] = admission_times['outtime_fallback']
                    admission_times = admission_times.drop(columns=[c for c in ['los','outtime_fallback'] if c in admission_times.columns])
                
                # Merge with data
                data = data.merge(admission_times, on=primary_id, how='left')
            else:
                # 确保intime是tz-naive datetime
                if pd.api.types.is_datetime64_any_dtype(data['intime']):
                    if hasattr(data['intime'].dt, 'tz') and data['intime'].dt.tz is not None:
                        data['intime'] = data['intime'].dt.tz_localize(None)
            # 若存在outtime，亦规范化
            if 'outtime' in data.columns and pd.api.types.is_datetime64_any_dtype(data['outtime']):
                if hasattr(data['outtime'].dt, 'tz') and data['outtime'].dt.tz is not None:
                    data['outtime'] = data['outtime'].dt.tz_localize(None)
            
            # 确保时间列是datetime类型（如果不是，移除时区信息）
            if pd.api.types.is_datetime64_any_dtype(data[index_column]):
                if hasattr(data[index_column].dt, 'tz') and data[index_column].dt.tz is not None:
                    data[index_column] = data[index_column].dt.tz_localize(None)
            else:
                # 如果仍然不是datetime，尝试转换
                data[index_column] = pd.to_datetime(data[index_column], errors='coerce', utc=True).dt.tz_localize(None)
            
            # 关键修复：R ricu 不过滤超出 ICU 时间窗口的数据
            # R ricu 保留所有数据点，包括：
            # 1. 入 ICU 前的数据（负时间）
            # 2. 出 ICU 后的数据（超过 outtime）
            # 这是因为临床数据可能在 ICU 入住前后测量，但仍然与 ICU 住院相关
            # 例如：实验室检验、生命体征等可能在入ICU前或转出后记录
            
            # Calculate hours since admission (不进行任何时间窗口过滤)
            time_diff = data[index_column] - data['intime']
            # Convert to hours (as float, matching ricu's behavior)
            hours = time_diff.dt.total_seconds() / 3600.0
            
            data[index_column] = hours
            
            # 🔧 CRITICAL FIX: Also convert ALL other datetime columns to relative hours
            # This fixes the norepi_rate issue where starttime was float but endtime was datetime,
            # causing expand() to generate 30 million invalid rows
            # Common time columns that need conversion: endtime, stop_var, stoptime, etc.
            time_related_cols = [col for col in data.columns 
                                if col not in [index_column, 'intime', 'outtime', primary_id] 
                                and pd.api.types.is_datetime64_any_dtype(data[col])]
            
            for time_col in time_related_cols:
                # Remove timezone if present
                if hasattr(data[time_col].dt, 'tz') and data[time_col].dt.tz is not None:
                    data[time_col] = data[time_col].dt.tz_localize(None)
                # Convert to hours since admission
                time_diff_col = data[time_col] - data['intime']
                data[time_col] = time_diff_col.dt.total_seconds() / 3600.0
            
            # 注意：不过滤负时间（入ICU前）或超过outtime的数据，匹配 R ricu 行为
            
            # Drop the temporary alignment columns
            drop_cols = ['intime']
            if 'outtime' in data.columns:
                drop_cols.append('outtime')
            data = data.drop(columns=drop_cols)
            
        except Exception as e:
            # If alignment fails, return original data silently
            pass
        
        return data

    def _load_recursive_concept(
        self,
        concept_name: str,
        definition: ConceptDefinition,
        data_source: ICUDataSource,
        *,
        aggregator: object,
        patient_ids: Optional[Iterable[object]],
        verbose: bool = True,
        interval: Optional[pd.Timedelta] = None,
        align_to_admission: bool = True,
        **kwargs,  # Additional parameters for callbacks
    ) -> ICUTable:
        if not definition.callback:
            raise NotImplementedError(
                f"Recursive concept '{concept_name}' requires a callback."
            )

        # Check for database-specific sub_concepts override
        db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
        sub_names = list(definition.sub_concepts)

        # DEBUG: Print database detection info
        if verbose and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🔍 Database-specific config check for concept '{concept_name}':")
            logger.debug(f"   db_name: '{db_name}'")
            logger.debug(f"   original sub_concepts: {sub_names}")
            logger.debug(f"   definition has sources: {hasattr(definition, 'sources')}")
            if hasattr(definition, 'sources'):
                logger.debug(f"   definition.sources: {definition.sources}")

        # Check if there's a database-specific configuration that overrides sub_concepts
        if db_name and hasattr(definition, 'sources') and db_name in definition.sources:
            db_sources = definition.sources[db_name]
            # db_sources is a list of ConceptSource objects, but concept-dict.
            if isinstance(db_sources, list):
                for db_source in db_sources:
                    if hasattr(db_source, '__dict__'):
                        # This is a ConceptSource object
                        db_source_dict = db_source.__dict__
                    else:
                        # This is already a dict
                        db_source_dict = db_source

                    if 'concepts' in db_source_dict:
                        # Use database-specific sub_concepts
                        sub_names = list(db_source_dict['concepts'])
                        if verbose and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"🔄 Using {db_name}-specific sub_concepts for '{concept_name}': {sub_names}")
                        break
                    elif 'params' in db_source_dict and isinstance(db_source_dict['params'], dict) and 'concepts' in db_source_dict['params']:
                        # Use database-specific sub_concepts from params
                        sub_names = list(db_source_dict['params']['concepts'])
                        if verbose and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"🔄 Using {db_name}-specific sub_concepts from params for '{concept_name}': {sub_names}")
                        break
            else:
                # db_sources is a dict (loaded from JSON)
                if 'concepts' in db_sources:
                    sub_names = list(db_sources['concepts'])
                    if verbose and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"🔄 Using {db_name}-specific sub_concepts for '{concept_name}': {sub_names}")

        if not sub_names:
            raise ValueError(
                f"Recursive concept '{concept_name}' specifies no sub concepts."
            )

        agg_value = self._coerce_final_aggregator(aggregator)
        if agg_value in (None, "auto"):
            fallback_agg = definition.aggregate
            if fallback_agg is not None:
                agg_value = self._coerce_final_aggregator(fallback_agg)

        aggregate_mapping = self._build_sub_aggregate(definition.aggregate, sub_names)

        # Prepare kwargs for sub-concepts, allowing them to be optional
        sub_kwargs = {**kwargs, '_allow_missing_concept': True}
        
        # 🔥 CRITICAL: 内部递归调用必须使用 ricu_compatible=False
        # 否则会返回 DataFrame 而不是 Dict[str, ICUTable]，导致后续处理失败
        sub_tables = self.load_concepts(
            sub_names,
            data_source,
            merge=False,
            aggregate=aggregate_mapping,
            patient_ids=patient_ids,
            verbose=verbose,
            interval=interval,  # Pass interval to recursive calls
            align_to_admission=align_to_admission,  # Pass align flag
            ricu_compatible=False,  # 🔥 内部调用必须返回 Dict[str, ICUTable]
            concept_workers=1,  # 🔧 子概念顺序加载，避免过度并行导致线程竞争
            **sub_kwargs,  # Pass kwargs with allow_missing flag
        )

        if isinstance(sub_tables, ICUTable):
            sub_tables = {sub_names[0]: sub_tables}

        # Standardize time column names for eICU BEFORE passing to callbacks
        # eICU uses different time column names (labresultoffset, observationoffset, etc.)
        # Rename them to a standard name 'charttime' to enable merging across concepts
        db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
        if db_name in ['eicu', 'eicu_demo']:
            # All possible eICU time offset columns
            eicu_time_cols = [
                'labresultoffset', 'observationoffset', 'nursecharting_offset', 
                'respiratorycharting_offset', 'intakeoutput_offset', 'respchartoffset',
                'infusionoffset', 'drugstartoffset', 'drugstopoffset', 'drugorderoffset',
                'culturetakenoffset', 'cultureoffset'  # 添加微生物培养时间列
            ]
            
            standardized_sub_tables = {}
            for name, table in sub_tables.items():
                if isinstance(table, ICUTable) and table.index_column:
                    # Check if this table uses an eICU-specific time column
                    if table.index_column in eicu_time_cols and table.index_column != 'charttime':
                        # Rename the column in the DataFrame
                        if table.index_column in table.data.columns:
                            renamed_data = table.data.rename(columns={table.index_column: 'charttime'})
                            # Create new ICUTable with updated index_column
                            table = ICUTable(
                                data=renamed_data,
                                id_columns=table.id_columns,
                                index_column='charttime',  # Update metadata
                                value_column=table.value_column,
                                unit_column=table.unit_column,
                                time_columns=table.time_columns,
                            )
                standardized_sub_tables[name] = table
            sub_tables = standardized_sub_tables

        # Align WinTbl time columns BEFORE passing to callbacks
        # This ensures _merge_tables can properly merge WinTbl concepts with numeric time columns
        if align_to_admission:
            from .table import WinTbl
            aligned_sub_tables = {}
            for name, table in sub_tables.items():
                if isinstance(table, WinTbl):
                    # WinTbl needs both index_var and dur_var aligned
                    idx_col = table.index_var
                    dur_col = table.dur_var
                    id_cols = table.id_vars
                    
                    if verbose and logger.isEnabledFor(logging.DEBUG):
                        logger.debug("   对齐 WinTbl '%s': index_var=%s, dur_var=%s", name, idx_col, dur_col)
                        if idx_col in table.data.columns:
                            logger.debug("      index_var 类型: %s", table.data[idx_col].dtype)
                        if dur_col and dur_col in table.data.columns:
                            logger.debug("      dur_var 类型: %s", table.data[dur_col].dtype)
                    
                    # Align index_var (start time) if it's datetime
                    if idx_col and idx_col in table.data.columns and pd.api.types.is_datetime64_any_dtype(table.data[idx_col]):
                        if verbose and logger.isEnabledFor(logging.DEBUG):
                            logger.debug("      ✅ 转换 index_var 从 datetime 到小时")
                        table.data = self._align_time_to_admission(
                            table.data,
                            data_source,
                            id_cols,
                            idx_col
                        )
                    
                    # Convert dur_var (duration) from timedelta to hours
                    if dur_col and dur_col in table.data.columns:
                        if pd.api.types.is_timedelta64_dtype(table.data[dur_col]):
                            if verbose and logger.isEnabledFor(logging.DEBUG):
                                logger.debug("      ✅ 转换 dur_var 从 timedelta 到小时")
                            table.data[dur_col] = table.data[dur_col].dt.total_seconds() / 3600.0
                        elif pd.api.types.is_datetime64_any_dtype(table.data[dur_col]):
                            # If dur_var is datetime (shouldn't happen), warn
                            logger.warning("⚠️  WinTbl '%s' 的 dur_var '%s' 是 datetime 类型，预期是 timedelta", name, dur_col)
                
                aligned_sub_tables[name] = table
            sub_tables = aligned_sub_tables

        ctx = ConceptCallbackContext(
            concept_name=concept_name,
            target=definition.target,
            interval=definition.interval,
            resolver=self,
            data_source=data_source,
            patient_ids=patient_ids,
            kwargs=kwargs,  # Pass kwargs to callback context
        )

        # Check for database-specific callback override
        callback_name = definition.callback

        # DEBUG: Print callback detection info
        if verbose and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🔍 Callback detection for concept '{concept_name}':")
            logger.debug(f"   original callback: '{callback_name}'")
            logger.debug(f"   db_name: '{db_name}'")
            logger.debug(f"   has sources: {hasattr(definition, 'sources')}")

        if db_name and hasattr(definition, 'sources') and db_name in definition.sources:
            db_sources = definition.sources[db_name]
            # db_sources is a list of ConceptSource objects, but concept-dict.
            if isinstance(db_sources, list):
                for db_source in db_sources:
                    if hasattr(db_source, '__dict__'):
                        # This is a ConceptSource object
                        db_source_dict = db_source.__dict__
                    else:
                        # This is already a dict
                        db_source_dict = db_source

                    if 'callback' in db_source_dict and db_source_dict['callback'] is not None:
                        # Use database-specific callback only if explicitly specified
                        callback_name = db_source_dict['callback']
                        if verbose and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"🔄 Using {db_name}-specific callback '{callback_name}' for '{concept_name}'")
                        break
            else:
                # db_sources is a dict (loaded from JSON)
                if 'callback' in db_sources and db_sources['callback'] is not None:
                    callback_name = db_sources['callback']
                    if verbose and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"🔄 Using {db_name}-specific callback '{callback_name}' for '{concept_name}'")

        # Validate callback_name before execution
        if callback_name is None:
            raise ValueError(f"Concept '{concept_name}' has no callback specified. Both original and database-specific callbacks are None.")

        if verbose and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🎯 Executing callback '{callback_name}' for concept '{concept_name}' with {len(sub_tables)} sub-tables")

        result = execute_concept_callback(callback_name, sub_tables, ctx)

        # CRITICAL: Align WinTbl result time columns immediately after callback
        # This ensures that when this concept is used as a sub-concept in parent recursion,
        # it already has numeric time columns (not datetime)
        from .table import WinTbl
        if isinstance(result, WinTbl) and align_to_admission and not result.data.empty:
            idx_col = result.index_var
            dur_col = result.dur_var
            id_cols = result.id_vars
            
            # Align index_var if it's still datetime
            if idx_col and idx_col in result.data.columns and pd.api.types.is_datetime64_any_dtype(result.data[idx_col]):
                if verbose and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "   对齐 WinTbl 结果 '%s': index_var=%s (datetime → 小时)",
                        concept_name,
                        idx_col,
                    )
                result.data = self._align_time_to_admission(
                    result.data,
                    data_source,
                    id_cols,
                    idx_col
                )
            
            # Convert dur_var from timedelta to hours
            if dur_col and dur_col in result.data.columns and pd.api.types.is_timedelta64_dtype(result.data[dur_col]):
                if verbose and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "   转换 WinTbl 结果 '%s': dur_var=%s (timedelta → 小时)",
                        concept_name,
                        dur_col,
                    )
                result.data[dur_col] = result.data[dur_col].dt.total_seconds() / 3600.0

        # R代码中，递归概念的回调返回结果就是最终结果，不需要再次聚合
        # aggregate参数已经在加载子概念时应用了
        # 我们只需要应用时间对齐和interval处理（如果需要）
        
        # Apply interval alignment and aggregation for recursive concepts
        # Handle both ICUTable and WinTbl
        if isinstance(result, WinTbl):
            idx_col = result.index_var
            dur_col = result.dur_var  # WinTbl 还有 duration 列
        else:
            idx_col = result.index_column
            dur_col = None
        
        if interval is not None and idx_col and idx_col in result.data.columns:
            from .ts_utils import change_interval
            
            # 关键修复：如果时间列是datetime类型但应该是numeric（align_to_admission=True），
            # 强制转换为相对小时数
            # 对于 WinTbl，需要同时转换 index_var 和 dur_var
            if align_to_admission and not result.data.empty and idx_col in result.data.columns:
                if pd.api.types.is_datetime64_any_dtype(result.data[idx_col]):
                    # 时间列是datetime，但应该是numeric（相对ICU入院时间的小时数）
                    # 这可能是因为callback复制了数据但没有保持类型转换
                    # 强制重新对齐
                    if isinstance(result, WinTbl):
                        id_cols = result.id_vars
                    else:
                        id_cols = result.id_columns
                    
                    # 对齐 index_var（开始时间）
                    result.data = self._align_time_to_admission(
                        result.data,
                        data_source,
                        id_cols,
                        idx_col
                    )
                    
                    # WinTbl 特殊处理：dur_var（持续时间）也需要转换
                    # 注意：dur_var 是时间间隔（如 timedelta），需要转换为小时数
                    if dur_col and dur_col in result.data.columns:
                        if pd.api.types.is_timedelta64_dtype(result.data[dur_col]):
                            # timedelta 转换为小时数
                            result.data[dur_col] = result.data[dur_col].dt.total_seconds() / 3600.0
                        elif pd.api.types.is_datetime64_any_dtype(result.data[dur_col]):
                            # 如果是 datetime（不应该，但保险起见），记录警告
                            print(f"   ⚠️  警告: WinTbl 的 dur_var '{dur_col}' 是 datetime 类型，预期是 timedelta")
            
            # Align time to ICU admission if requested
            if align_to_admission and not result.data.empty:
                # Get id_columns based on result type
                if isinstance(result, WinTbl):
                    id_cols = result.id_vars
                else:
                    id_cols = result.id_columns
                
                # 只有在时间列不是numeric类型时才对齐（避免重复对齐）
                if not pd.api.types.is_numeric_dtype(result.data[idx_col]):
                    result.data = self._align_time_to_admission(
                        result.data,
                        data_source,
                        id_cols,
                        idx_col
                    )
                    
                    # WinTbl: 同时转换 dur_var
                    if dur_col and dur_col in result.data.columns:
                        if pd.api.types.is_timedelta64_dtype(result.data[dur_col]):
                            result.data[dur_col] = result.data[dur_col].dt.total_seconds() / 3600.0
            
            # CRITICAL: Expand WinTbl to time series before applying interval aggregation
            # WinTbl represents time windows (start_time, duration) and must be expanded
            # to individual time points when interval is specified
            if isinstance(result, WinTbl) and not result.data.empty:
                idx_col = result.index_var
                dur_col = result.dur_var
                id_cols = result.id_vars
                
                if idx_col and dur_col and idx_col in result.data.columns and dur_col in result.data.columns:
                    if verbose:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("   扩展 WinTbl '%s' 到时间序列 (interval=%s)", concept_name, interval)
                    
                    # 扩展窗口到时间序列
                    interval_hours = interval.total_seconds() / 3600.0
                    expanded_rows = []
                    for _, row in result.data.iterrows():
                        start_time = row[idx_col]
                        duration = row[dur_col]
                        
                        # FIX: 对于 duration=0 的行，只添加一个时间点（对齐到 interval）
                        if duration <= 0:
                            aligned_time = np.floor(start_time / interval_hours) * interval_hours
                            new_row = {idx_col: aligned_time}
                            # 复制 ID 列
                            for col in id_cols:
                                if col in row.index:
                                    new_row[col] = row[col]
                            # 复制值列（除了 dur_col）
                            for col in result.data.columns:
                                if col not in [idx_col, dur_col] and col not in id_cols:
                                    new_row[col] = row[col]
                            expanded_rows.append(new_row)
                            continue
                        
                        # 计算结束时间（小时）
                        # R ricu 使用 seq(min, max, step) 包含终点，所以这里用 <=
                        end_time = start_time + duration
                        
                        # 生成时间序列（每个 interval）
                        current_time = np.floor(start_time / interval_hours) * interval_hours
                        
                        while current_time <= end_time:
                            new_row = {idx_col: current_time}
                            # 复制 ID 列
                            for col in id_cols:
                                if col in row.index:
                                    new_row[col] = row[col]
                            # 复制值列（除了 dur_col）
                            for col in result.data.columns:
                                if col not in [idx_col, dur_col] and col not in id_cols:
                                    new_row[col] = row[col]
                            expanded_rows.append(new_row)
                            current_time += interval_hours
                    
                    # 转换为 DataFrame
                    if expanded_rows:
                        expanded_df = pd.DataFrame(expanded_rows)
                        # 转换为 ICUTable
                        value_col = [c for c in expanded_df.columns if c not in id_cols and c != idx_col]
                        value_col = value_col[0] if value_col else None
                        result = ICUTable(
                            data=expanded_df,
                            id_columns=id_cols,
                            index_column=idx_col,
                            value_column=value_col,
                            unit_column=None,
                            time_columns=[],
                        )
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("   ✅ 扩展完成: %d 行", len(expanded_df))
                        elif verbose:
                            print(f"   ✅ 扩展完成: {len(expanded_df)} 行")
                    else:
                        # 没有数据，返回空的 ICUTable
                        result = ICUTable(
                            data=pd.DataFrame(columns=[*id_cols, idx_col]),
                            id_columns=id_cols,
                            index_column=idx_col,
                            value_column=None,
                            unit_column=None,
                            time_columns=[],
                        )
            
            # Apply change_interval: round to interval and aggregate same-hour records
            # CRITICAL: For sofa_single type callbacks (sofa_coag, sofa_liver, sofa_cns),
            # the sub-concept already has the correct time points after interval alignment.
            # The callback just calculates a new column and removes the input column,
            # so time points should remain unchanged. However, ricu_code still applies
            # change_interval after callback, so we do the same for consistency.
            # But we should NOT re-aggregate if the result already has the correct interval.
            
            # 确定聚合方法：使用传入的aggregator或definition.aggregate
            agg_method = agg_value if agg_value not in (None, False, "auto") else None
            if agg_method in (None, "auto"):
                agg_method = None
            # GCS total score should use 'min' aggregation (for recursive concepts)
            # But GCS sub-components should use default aggregation (median)
            if concept_name == 'gcs':
                if agg_method is None or (isinstance(agg_method, str) and agg_method != 'min'):
                    agg_method = 'min'
            if agg_method in (None, "auto") and concept_name in VASO_RATE_CONCEPTS:
                agg_method = "max"
            # SOFA cardiovascular components must retain the highest severity within the window.
            # Using the default 'median' aggregation diluted vasopressor-driven spikes (e.g. 2 and 4
            # becoming 3, or 1 and 2 becoming 1.5). ricu keeps the window maximum, so align here.
            sofa_max_concepts = {'sofa_cardio', 'sofa2_cardio'}
            if agg_method is None and concept_name in sofa_max_concepts:
                agg_method = 'max'
            # 如果仍然没有指定，根据值列类型自动选择
            if agg_method is None:
                # Get value column based on result type
                if isinstance(result, WinTbl):
                    value_col = None  # WinTbl doesn't have a single value column
                else:
                    value_col = getattr(result, 'value_column', None)
                
                if value_col and value_col in result.data.columns:
                    if pd.api.types.is_numeric_dtype(result.data[value_col]):
                        agg_method = 'median'  # Changed from 'mean' to 'median' to match R ricu default
                    else:
                        agg_method = 'first'
                else:
                    # Default to 'first' if no value column found
                    agg_method = 'first'
            
            # 只有指定了聚合方法时才应用change_interval
            # For sofa_single type, the time points should already be correct,
            # but we still apply change_interval to match ricu_code's behavior
            # Skip if result is still WinTbl (not expanded)
            has_time_column = getattr(result, 'index_column', None)
            if agg_method and has_time_column and has_time_column in result.data.columns and not result.data.empty and not isinstance(result, WinTbl):
                try:
                    fill_missing = self._should_fill_gaps(concept_name, definition)
                    fill_method = self._get_fill_method(concept_name, definition)
                    combined_result = change_interval(
                        result,
                        interval=interval,
                        aggregation=agg_method,
                        fill_gaps=fill_missing,
                        fill_method=fill_method,
                        copy=False
                    )
                    
                    # Extract data if ICUTable is returned
                    if hasattr(combined_result, 'data'):
                        result.data = combined_result.data
                    else:
                        result.data = combined_result
                except Exception as e:
                    # If change_interval fails, log but continue
                    if verbose:
                        print(f"  ⚠️ 警告: {concept_name} 的interval处理失败: {e}")

        # 🔧 NOTE: 不过滤负时间（入ICU前的数据），ricu 保留这些数据

        return result

    @staticmethod
    def _build_sub_aggregate(
        aggregate_spec: object,
        sub_names: List[str],
    ) -> Optional[Mapping[str, object]]:
        def normalise(value: object) -> object:
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return normalise(value[0])
                return [normalise(item) for item in value]
            return value

        if aggregate_spec is None:
            return None

        if isinstance(aggregate_spec, Mapping):
            return {name: normalise(aggregate_spec.get(name)) for name in sub_names}

        if isinstance(aggregate_spec, (list, tuple)):
            return {
                name: normalise(aggregate_spec[i])
                for i, name in enumerate(sub_names)
                if i < len(aggregate_spec)
            }

        return {name: normalise(aggregate_spec) for name in sub_names}

    @staticmethod
    def _coerce_final_aggregator(aggregator: object) -> object:
        if isinstance(aggregator, (list, tuple, dict)):
            return "auto"
        return aggregator

    def _load_fun_item(
        self,
        concept_name: str,
        definition: ConceptDefinition,
        source: ConceptSource,
        data_source: ICUDataSource,
        *,
        aggregator: object,
        patient_ids: Optional[Iterable[object]],
        **kwargs,  # Additional parameters (not used in fun_item but accepted for consistency)
    ) -> ICUTable:
        callback = (source.callback or "").strip()

        if callback == "los_callback":
            raw = self._load_fun_item_los(concept_name, source, data_source, patient_ids)
        elif "fwd_concept" in callback:
            raw = self._load_fun_item_forward(
                concept_name,
                source,
                data_source,
                patient_ids,
            )
        else:
            raise NotImplementedError(
                f"Function item callback '{callback}' is not yet supported."
            )

        agg_value = aggregator
        if agg_value in (None, "auto") and definition.aggregate is not None:
            agg_value = definition.aggregate

        agg_value = self._coerce_final_aggregator(agg_value)

        # WinTbl 不需要聚合，直接返回
        from .table import WinTbl
        if isinstance(raw, WinTbl):
            return raw

        if agg_value is not False:
            frame = self._apply_aggregation(
                raw.data,
                raw.value_column or concept_name,
                list(raw.id_columns),
                raw.index_column,
                raw.unit_column,
                agg_value,
            )
            raw = ICUTable(
                data=frame,
                id_columns=list(raw.id_columns),
                index_column=raw.index_column,
                value_column=raw.value_column or concept_name,
                unit_column=raw.unit_column,
                time_columns=list(raw.time_columns),
            )

        return raw

    def _load_fun_item_los(
        self,
        concept_name: str,
        source: ConceptSource,
        data_source: ICUDataSource,
        patient_ids: Optional[Iterable[object]],
    ) -> ICUTable:
        win_type = source.params.get("win_type")
        if not win_type:
            raise ValueError("los_callback requires 'win_type' parameter.")

        id_cfg = data_source.config.id_configs.get(win_type)
        if id_cfg is None or not id_cfg.table or not id_cfg.start or not id_cfg.end:
            raise ValueError(f"Identifier configuration for '{win_type}' is incomplete.")

        required_cols = [id_cfg.id, id_cfg.start, id_cfg.end]
        table = data_source.load_table(id_cfg.table, columns=required_cols)

        base_frame = table.data.copy()
        missing_required = [col for col in required_cols if col not in base_frame.columns]
        if missing_required:
            for column in missing_required:
                fallback = self._synthesise_los_column(column, data_source, base_frame)
                if fallback is None:
                    raise KeyError(
                        f"Required column '{column}' missing for LOS calculation in table '{id_cfg.table}'"
                    )
                base_frame[column] = fallback

        frame = base_frame[required_cols].copy()
        frame = frame.dropna(subset=[id_cfg.start, id_cfg.end])

        # Detect time format and database type
        start_col = frame[id_cfg.start]
        end_col = frame[id_cfg.end]
        is_numeric_time = pd.api.types.is_numeric_dtype(start_col)
        ds_name = (data_source.config.name or "").lower()
        
        # Determine time unit: eICU uses minutes, AUMC uses milliseconds
        is_eicu = ds_name.startswith("eicu")
        
        if is_numeric_time:
            start_val = pd.to_numeric(start_col, errors="coerce")
            end_val = pd.to_numeric(end_col, errors="coerce")
            valid_mask = start_val.notna() & end_val.notna() & (end_val >= start_val)
            frame = frame.loc[valid_mask].copy()
            if frame.empty:
                return ICUTable(
                    data=pd.DataFrame(columns=[id_cfg.id, concept_name]),
                    id_columns=[id_cfg.id],
                    index_column=None,
                    value_column=concept_name,
                )
            
            if is_eicu:
                # eICU: times are relative MINUTES from ICU admission
                los_days = (end_val.loc[valid_mask] - start_val.loc[valid_mask]) / (60 * 24)
                duration_hours = (end_val.loc[valid_mask] - start_val.loc[valid_mask]) / 60
                start_hours = start_val.loc[valid_mask] / 60
            else:
                # AUMC/HiRID: times are relative MILLISECONDS from admission
                los_days = (end_val.loc[valid_mask] - start_val.loc[valid_mask]) / (1000 * 60 * 60 * 24)
                duration_hours = (end_val.loc[valid_mask] - start_val.loc[valid_mask]) / (1000 * 60 * 60)
                start_hours = start_val.loc[valid_mask] / (1000 * 60 * 60)
            
            frame[concept_name] = los_days
        else:
            # MIIV/eICU: times are datetime objects
            start_time = pd.to_datetime(start_col, errors="coerce")
            end_time = pd.to_datetime(end_col, errors="coerce")
            valid_mask = start_time.notna() & end_time.notna() & (end_time >= start_time)
            frame = frame.loc[valid_mask].copy()
            if frame.empty:
                return ICUTable(
                    data=pd.DataFrame(columns=[id_cfg.id, concept_name]),
                    id_columns=[id_cfg.id],
                    index_column=None,
                    value_column=concept_name,
                )
            frame[concept_name] = (end_time.loc[valid_mask] - start_time.loc[valid_mask]).dt.total_seconds() / 86400.0
            duration_hours = (end_time.loc[valid_mask] - start_time.loc[valid_mask]).dt.total_seconds() / 3600.0
            start_hours = None  # Will use datetime-based approach

        frame = frame[frame[concept_name] >= 0]
        if frame.empty:
            return ICUTable(
                data=pd.DataFrame(columns=[id_cfg.id, concept_name]),
                id_columns=[id_cfg.id],
                index_column=None,
                value_column=concept_name,
            )

        if patient_ids is not None:
            if isinstance(patient_ids, dict):
                candidates = patient_ids.get(id_cfg.id) or patient_ids.get(str(id_cfg.id)) or []
            else:
                candidates = patient_ids
            if candidates:
                mask = frame[id_cfg.id].isin(set(candidates))
                frame = frame[mask]
                if is_numeric_time:
                    duration_hours = duration_hours.loc[frame.index]
                    start_hours = start_hours.loc[frame.index]
                else:
                    duration_hours = duration_hours.loc[frame.index]

        if frame.empty:
            return ICUTable(
                data=pd.DataFrame(columns=[id_cfg.id, concept_name]),
                id_columns=[id_cfg.id],
                index_column=None,
                value_column=concept_name,
            )

        # Generate hourly time grid
        rows: List[dict] = []
        for idx, row in frame.iterrows():
            stay_id = row[id_cfg.id]
            los_val = row[concept_name]
            dur_h = duration_hours.loc[idx] if hasattr(duration_hours, 'loc') else duration_hours[idx]
            
            if pd.isna(dur_h) or dur_h < 0:
                continue
            
            if is_numeric_time:
                # AUMC: use relative hours directly
                st_h = start_hours.loc[idx] if hasattr(start_hours, 'loc') else start_hours[idx]
                # Generate hourly grid from (start - 1) to end
                start_hour = int(st_h) - 1
                end_hour = int(st_h + dur_h) + 1
                for hour in range(start_hour, end_hour):
                    rows.append({
                        id_cfg.id: stay_id,
                        "index_var": float(hour),
                        concept_name: los_val,
                    })
            else:
                # MIIV/eICU: use datetime and convert later
                start_dt = start_time.loc[idx]
                end_dt = end_time.loc[idx]
                current_time = start_dt - pd.Timedelta(hours=1)
                while current_time < end_dt:
                    rows.append({
                        id_cfg.id: stay_id,
                        "index_var": current_time,
                        concept_name: los_val,
                    })
                    current_time += pd.Timedelta(hours=1)

        if not rows:
            return ICUTable(
                data=pd.DataFrame(columns=[id_cfg.id, concept_name]),
                id_columns=[id_cfg.id],
                index_column=None,
                value_column=concept_name,
            )

        ts_df = pd.DataFrame(rows)
        # Note: For los_icu, index_var is already in hours relative to ICU admission,
        # so we skip _align_time_to_admission which would incorrectly divide by 60 again for eICU
        ts_df["index_var"] = pd.to_numeric(ts_df["index_var"], errors="coerce")
        ts_df = ts_df.dropna(subset=["index_var"]).reset_index(drop=True)
        return ICUTable(
            data=ts_df,
            id_columns=[id_cfg.id],
            index_column="index_var",
            value_column=concept_name,
        )

    def _synthesise_los_column(
        self,
        column_name: str,
        data_source: ICUDataSource,
        frame: pd.DataFrame,
    ) -> Optional[pd.Series]:
        ds_name = (data_source.config.name or "").lower()
        if column_name == "unitadmitoffset" and ds_name.startswith("eicu"):
            logger.warning(
                "Column '%s' missing for %s; assuming zero-minute ICU admission offsets.",
                column_name,
                data_source.config.name,
            )
            return pd.Series(0, index=frame.index, dtype="float64")
        return None

    def _load_fun_item_forward(
        self,
        concept_name: str,
        source: ConceptSource,
        data_source: ICUDataSource,
        patient_ids: Optional[Iterable[object]],
    ) -> ICUTable:
        callback = source.callback or ""
        match = re.search(r"fwd_concept\('(.+?)'\)", callback)
        if not match:
            raise ValueError("fwd_concept callback is missing concept name.")

        base_name = match.group(1)
        # 🔧 FIX: 禁用 ricu_compatible 模式，确保返回 dict[str, ICUTable]
        base_tables = self.load_concepts(
            [base_name],
            data_source,
            merge=False,
            aggregate=None,
            patient_ids=patient_ids,
            ricu_compatible=False,  # 确保返回原始 ICUTable 格式
        )
        if isinstance(base_tables, ICUTable):
            base_table = base_tables
        elif isinstance(base_tables, dict):
            base_table = base_tables[base_name]
        else:
            # 如果返回的是 DataFrame（不应该发生，但做防御性处理）
            raise TypeError(
                f"Expected ICUTable or dict, got {type(base_tables).__name__} "
                f"when loading '{base_name}' for fwd_concept in '{concept_name}'"
            )

        data = base_table.data.copy()
        value_col = base_table.value_column or base_name

        comp_match = re.search(r"comp_na\(`(.+?)`,\s*(.+?)\)", callback, flags=re.DOTALL)
        if comp_match:
            op_symbol = comp_match.group(1)
            literal = _parse_literal(comp_match.group(2))
            series = data[value_col]
            if op_symbol == "<=":
                # comp_na: NA -> False, 否则根据比较结果
                numeric_series = pd.to_numeric(series, errors="coerce")
                mask = (~numeric_series.isna()) & (numeric_series <= literal)
            elif op_symbol == "==":
                mask = (~series.isna()) & (series.astype(str) == str(literal))
            elif op_symbol == "!=":
                mask = (~series.isna()) & (series.astype(str) != str(literal))
            else:
                raise NotImplementedError(f"Unsupported comparison operator '{op_symbol}'")
        else:
            mask = pd.Series(True, index=data.index)

        if "ts_to_win_tbl" in callback:
            # 如果 base_table 为空或没有 index_column，返回空的 WinTbl
            if base_table.index_column is None or base_table.data.empty:
                # 使用 base_table 的 ID 列（优先），否则使用数据库特定的默认值
                # WinTbl 已在模块顶部导入，不需要重复导入
                
                # 确定数据库特定的默认 ID 列
                db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else 'unknown'
                default_id_cols = _default_id_columns_for_db(db_name)
                
                if isinstance(base_table, WinTbl):
                    id_cols = list(base_table.id_vars) if base_table.id_vars else default_id_cols
                else:
                    id_cols = list(base_table.id_columns) if base_table.id_columns else default_id_cols
                idx_col = base_table.index_column if base_table.index_column else 'charttime'  # 默认时间列
                # 创建空 DataFrame 并设置正确的 dtype
                empty_win_df = pd.DataFrame(columns=id_cols + [idx_col, concept_name + "_dur", concept_name])
                # 设置 index 列为 datetime 类型（即使为空）
                empty_win_df[idx_col] = pd.to_datetime(empty_win_df[idx_col])
                # dur_var 应该是 float（小时），而不是 timedelta
                empty_win_df[concept_name + "_dur"] = empty_win_df[concept_name + "_dur"].astype(float)
                empty_win_df[concept_name] = empty_win_df[concept_name].astype(bool)
                return WinTbl(
                    data=empty_win_df,
                    id_vars=id_cols,
                    index_var=idx_col,
                    dur_var=concept_name + "_dur",
                )
            # 使用非贪婪匹配，并支持嵌套括号（如 mins(360L)）
            dur_match = re.search(r"ts_to_win_tbl\(([^)]+\))\)", callback, flags=re.DOTALL)
            if not dur_match:
                # 备用：简单匹配
                dur_match = re.search(r"ts_to_win_tbl\((.+?)\)", callback, flags=re.DOTALL)
            duration = self._parse_interval_expression(dur_match.group(1).strip() if dur_match else "mins(60)")
            # 将 timedelta 转换为小时（float）
            if isinstance(duration, pd.Timedelta):
                duration_hours = duration.total_seconds() / 3600.0
            else:
                duration_hours = float(duration)
            
            # FIX: 为所有行创建 WinTbl，True 行有窗口持续时间，False 行持续时间为 0
            # 这样在 downsampling 时，True 的窗口会扩展，False 的只保留原始时间点
            win_df = data[list(base_table.id_columns) + [base_table.index_column]].copy()
            # True 行使用完整窗口持续时间，False 行使用 0（只表示该时间点存在）
            win_df["duration"] = np.where(mask.values, duration_hours, 0.0)
            win_df[concept_name] = mask.values
            return WinTbl(
                data=win_df.rename(columns={"duration": concept_name + "_dur"}),
                id_vars=list(base_table.id_columns),
                index_var=base_table.index_column,
                dur_var=concept_name + "_dur",
            )

        cols = list(base_table.id_columns)
        if base_table.index_column:
            cols.append(base_table.index_column)
        cols.append(value_col)
        result = data[cols].rename(columns={value_col: concept_name})

        return ICUTable(
            data=result.reset_index(drop=True),
            id_columns=list(base_table.id_columns),
            index_column=base_table.index_column,
            value_column=concept_name,
        )

    @staticmethod
    def _parse_interval_expression(expression: str) -> pd.Timedelta:
        expr = expression.strip()
        match = re.fullmatch(r"([a-zA-Z]+)\((.+)\)", expr)
        if not match:
            raise ValueError(f"Unsupported interval expression '{expression}'")

        unit = match.group(1).lower()
        raw_value = match.group(2).strip()
        
        # 移除 R 语言的整数后缀 'L'（例如 360L -> 360）
        # 注意：R 的 'L' 只是表示整数，不是时间单位
        if raw_value.endswith('L'):
            raw_value = raw_value[:-1]
        
        value = _parse_literal(raw_value)
        if isinstance(value, pd.Timedelta):
            return value
        # 如果value是字符串，尝试解析为数值
        if isinstance(value, str):
            # 移除可能的尾随字符（如括号）
            value = value.strip().rstrip(')')
            # 再次检查 L 后缀
            if value.endswith('L'):
                value = value[:-1]
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"Cannot parse interval value '{value}' in expression '{expression}'")
        # value是数值，需要加上unit
        if unit in {"min", "mins", "minute", "minutes"}:
            return pd.to_timedelta(value, unit="m")
        if unit in {"hour", "hours"}:
            return pd.to_timedelta(value, unit="h")
        if unit in {"sec", "secs", "second", "seconds"}:
            return pd.to_timedelta(value, unit="s")
        if unit in {"day", "days"}:
            return pd.to_timedelta(value, unit="d")
        raise ValueError(f"Unsupported interval unit '{unit}' in expression '{expression}'")

    def _merge_tables(self, tables: Mapping[str, ICUTable]) -> pd.DataFrame:
        from .table import WinTbl
        merged: Optional[pd.DataFrame] = None
        index_column: Optional[str] = None
        id_columns: Optional[List[str]] = None

        for name, table in tables.items():
            frame = table.data.copy()

            # Handle both ICUTable and WinTbl
            if isinstance(table, WinTbl):
                id_columns = id_columns or list(table.id_vars)
                index_column = index_column or table.index_var
                expected_id = id_columns or []
                if list(table.id_vars) != expected_id:
                    raise ValueError(
                        "All concepts must share identical identifier columns to merge"
                    )
                if table.index_var != index_column:
                    raise ValueError(
                        "All concepts must share identical index column to merge"
                    )
            else:
                id_columns = id_columns or list(table.id_columns)
                index_column = index_column or table.index_column
                expected_id = id_columns or []
                if list(table.id_columns) != expected_id:
                    raise ValueError(
                        "All concepts must share identical identifier columns to merge"
                    )
                if table.index_column != index_column:
                    raise ValueError(
                        "All concepts must share identical index column to merge"
                    )

            key_cols = expected_id + ([index_column] if index_column else [])
            
            # 确保所有必需的列都存在
            missing_key_cols = [col for col in key_cols if col not in frame.columns]
            if missing_key_cols:
                # 如果缺少关键列，跳过这个表
                print(f"⚠️  警告: 表 '{name}' 缺少关键列 {missing_key_cols}，跳过合并")
                continue
            
            if name not in frame.columns:
                # 如果概念值列不存在，检查是否有其他值列
                # 这种情况可能发生在keep_components=True时，回调返回了组件列而不是概念名称列
                value_cols = [col for col in frame.columns if col not in key_cols]
                if not value_cols:
                    print(f"⚠️  警告: 表 '{name}' 没有值列，跳过合并")
                    continue
            
            # 选择要保留的列：ID列 + 时间列 + 所有非关键列（包括概念值列和组件列）
            # 保留所有值列，不仅仅是概念名称列
            # 这对于 keep_components=True 的情况很重要（如 SOFA 组件）
            # 但是要排除单位列（valueuom），因为它会导致合并时的列冲突
            # 单位列通常不需要保留，因为值已经标准化了
            excluded_cols = ['valueuom', 'unit']  # 排除这些列以避免合并冲突
            
            # 处理 MultiIndex 列（当 aggregate=['min', 'max'] 时产生）
            if isinstance(frame.columns, pd.MultiIndex):
                # MultiIndex 列：保留所有非关键列
                # key_cols 是简单字符串，需要匹配 MultiIndex 的第一层
                key_cols_set = set(key_cols)
                excluded_set = set(excluded_cols)
                
                # 选择列：保留第一层不在 key_cols 和 excluded_cols 中的列
                cols_to_keep = [col for col in frame.columns 
                               if col[0] not in key_cols_set and col[0] not in excluded_set]
                # 添加 key_cols（它们是简单列，不是 MultiIndex）
                # 先展平 MultiIndex，然后选择
                frame = frame.copy()
                # 重置列：将 MultiIndex 展平为单层（如 ('pafi', 'min') -> 'pafi_min'）
                if frame.columns.nlevels == 2:
                    frame.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                    for col in frame.columns.values]
                # 现在 key_cols 和 value_cols 都是简单字符串
                excluded_cols_flat = excluded_cols
                value_cols = [col for col in frame.columns 
                             if col not in key_cols and col not in excluded_cols_flat]
                cols_to_keep = key_cols + value_cols
                frame = frame[cols_to_keep].copy()
            else:
                # 普通列：原有逻辑
                value_cols = [col for col in frame.columns 
                             if col not in key_cols and col not in excluded_cols]
                cols_to_keep = key_cols + value_cols
                frame = frame[cols_to_keep].copy()
            
            # 在设置索引前排序，避免"left keys must be sorted"错误
            frame = frame.sort_values(key_cols)
            
            # 设置索引用于合并
            frame = frame.set_index(key_cols)

            if merged is None:
                merged = frame
            else:
                # 确保索引层级一致
                if merged.index.nlevels != frame.index.nlevels:
                    print(f"⚠️  警告: 索引层级不一致 ({merged.index.nlevels} vs {frame.index.nlevels})，重置索引后重新合并")
                    # 重置为共同的索引列
                    common_keys = [col for col in merged.index.names if col in frame.index.names]
                    merged = merged.reset_index()
                    frame = frame.reset_index()
                    # 检测列重叠，使用suffixes避免冲突
                    overlapping_cols = set(merged.columns) & set(frame.columns) - set(common_keys)
                    if overlapping_cols:
                        merged = merged.merge(frame, on=common_keys, how='outer', suffixes=('', '_dup'))
                        # 删除重复列
                        merged = merged[[c for c in merged.columns if not c.endswith('_dup')]]
                    else:
                        merged = merged.merge(frame, on=common_keys, how='outer')
                    merged = merged.sort_values(common_keys)
                    merged = merged.set_index(common_keys)
                else:
                    merged = merged.join(frame, how="outer", rsuffix='_dup')
                    # 删除join产生的重复列
                    merged = merged[[c for c in merged.columns if not c.endswith('_dup')]]

        if merged is None:
            return pd.DataFrame()

        merged = merged.reset_index()
        return merged

    def _build_cache_key(
        self,
        concept_name: str,
        data_source: ICUDataSource,
        patient_ids: Optional[Iterable[object]],
        interval: Optional[pd.Timedelta],
        align_to_admission: bool,
        aggregator: object,
        kwargs: Dict[str, object],
    ) -> str:
        """Build a cache key for a concept based on all relevant parameters."""
        import hashlib
        import json
        
        # Create a dictionary of all parameters that affect the result
        cache_params = {
            "concept_name": concept_name,
            "database": data_source.config.name if hasattr(data_source.config, 'name') else str(data_source.config),
            "patient_ids": sorted(list(patient_ids)) if patient_ids else None,
            "interval": str(interval) if interval else None,
            "align_to_admission": align_to_admission,
            "aggregator": str(aggregator),
            "kwargs": {k: str(v) for k, v in kwargs.items()},
            "dictionary_signature": self.dictionary_signature,
            "schema_version": self.cache_schema_version,
        }
        
        # Serialize and hash the parameters
        serialized = json.dumps(cache_params, sort_keys=True, default=str)
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def _load_from_disk_cache(
        self,
        concept_name: str,
        data_source: ICUDataSource,
        cache_key: str,
    ) -> Optional[ICUTable]:
        """Load a concept from disk cache if available."""
        if self.cache_dir is None:
            return None
            
        try:
            import pickle
            from pathlib import Path
            
            # Create cache file path
            cache_file = Path(self.cache_dir) / f"{cache_key}.pkl"
            if not cache_file.exists():
                return None
                
            # Load from cache
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                
            # Verify the cached data is an ICUTable
            if isinstance(cached_data, ICUTable):
                return cached_data
                
        except Exception:
            # If anything goes wrong, silently return None to force recomputation
            pass
            
        return None

    def _store_in_disk_cache(
        self,
        concept_name: str,
        data_source: ICUDataSource,
        cache_key: str,
        result: ICUTable,
    ) -> None:
        """Store a concept result in disk cache."""
        if self.cache_dir is None:
            return
            
        try:
            import pickle
            from pathlib import Path
            
            # Ensure cache directory exists
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Create cache file path
            cache_file = Path(self.cache_dir) / f"{cache_key}.pkl"
            
            # Store in cache
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
                
        except Exception:
            # If anything goes wrong, silently continue without caching
            pass

    def _expand_dependencies(self, requested: List[str]) -> List[str]:
        """Return dependency-closed list of concept names."""
        ordered: List[str] = []
        seen: set[str] = set()

        def visit(name: str) -> None:
            if name in seen:
                return
            if name not in self.dictionary:
                raise KeyError(f"Concept '{name}' not present in dictionary")
            seen.add(name)
            definition = self.dictionary[name]
            for dep in definition.depends_on:
                visit(dep)
            ordered.append(name)

        for concept in requested:
            visit(concept)
        return ordered

    def _ensure_concept_loaded(
        self,
        concept_name: str,
        data_source: ICUDataSource,
        aggregators: Dict[str, object],
        patient_ids: Optional[Iterable[object]],
        verbose: bool,
        interval: pd.Timedelta,
        align_to_admission: bool,
        kwargs: Dict[str, object],
        _skip_concept_cache: bool = False,  # 🔧 跳过概念缓存
    ) -> ICUTable:
        # 🚀 优化：增强概念数据缓存（避免重复加载相同概念，如urine、vaso_ind、pafi）
        patient_ids_hash = hash(frozenset(patient_ids)) if patient_ids else None
        agg_value = aggregators.get(concept_name, "auto")
        if agg_value in (None, "auto"):
            definition = self.dictionary.get(concept_name)
            if definition and definition.aggregate is not None:
                agg_value = definition.aggregate
        
        # 🔥 关键优化: 扩展缓存键包含kwargs中的关键参数，确保不同配置不会混淆
        # 但对于子概念（如vaso_ind），kwargs通常相同，所以可以安全缓存
        # 🔧 修复: 对不可哈希的值（如list）转换为字符串
        def _hashable_kwargs_items(kw):
            for k, v in sorted(kw.items()):
                try:
                    hash(v)
                    yield (k, v)
                except TypeError:
                    yield (k, str(v))
        kwargs_hash = hash(frozenset(_hashable_kwargs_items(kwargs))) if kwargs else 0
        concept_cache_key = (concept_name, patient_ids_hash, str(interval), str(agg_value), kwargs_hash)
        
        # 🔧 如果 _skip_concept_cache=True，跳过所有缓存检查和缓存写入
        # 这用于回调内部加载概念，避免污染主缓存
        if not _skip_concept_cache:
            with self._cache_lock:
                # 检查增强的概念数据缓存
                if concept_cache_key in self._concept_data_cache:
                    if verbose and logger.isEnabledFor(logging.DEBUG):
                        logger.debug("✨ 从内存缓存加载概念 '%s' (命中增强缓存)", concept_name)
                    return self._concept_data_cache[concept_cache_key]
                
                # 回退检查旧的简单缓存（用于向后兼容）
                simple_key = (concept_name, patient_ids_hash, str(interval), str(agg_value))
                if simple_key in self._concept_data_cache:
                    if verbose and logger.isEnabledFor(logging.DEBUG):
                        logger.debug("✨ 从内存缓存加载概念 '%s' (命中简单缓存)", concept_name)
                    result = self._concept_data_cache[simple_key]
                    # 同步到新的缓存键
                    self._concept_data_cache[concept_cache_key] = result
                    return result
                
                # 检查旧的概念缓存
                cached = self._concept_cache.get(concept_name)
                if cached is not None:
                    # 同时更新到新缓存
                    self._concept_data_cache[concept_cache_key] = cached
                    return cached
                # 线程安全的循环依赖检测
                inflight = self._get_inflight()
                if concept_name in inflight:
                    raise RuntimeError(f"Circular dependency detected for concept '{concept_name}'")
                inflight.add(concept_name)
        else:
            # 跳过缓存模式也需要设置 inflight 以检测循环依赖
            with self._cache_lock:
                inflight = self._get_inflight()
                if concept_name in inflight:
                    raise RuntimeError(f"Circular dependency detected for concept '{concept_name}'")
                inflight.add(concept_name)

        definition = self.dictionary[concept_name]
        for dependency in definition.depends_on:
            self._ensure_concept_loaded(
                dependency,
                data_source,
                aggregators,
                patient_ids,
                verbose,
                interval,
                align_to_admission,
                kwargs,
                _skip_concept_cache=_skip_concept_cache,  # 传递跳过缓存标志
            )

        cache_key = self._build_cache_key(
            concept_name,
            data_source,
            patient_ids,
            interval,
            align_to_admission,
            agg_value,
            kwargs,
        )

        # 🔧 如果 _skip_concept_cache=True，跳过磁盘缓存
        if not _skip_concept_cache:
            disk_hit = self._load_from_disk_cache(concept_name, data_source, cache_key)
            if disk_hit is not None:
                with self._cache_lock:
                    self._concept_cache[concept_name] = disk_hit
                    self._concept_data_cache[concept_cache_key] = disk_hit  # 🚀 也存入新缓存
                    self._get_inflight().discard(concept_name)
                return disk_hit

        try:
            result = self._load_single_concept(
                concept_name,
                data_source,
                aggregator=agg_value,
                patient_ids=patient_ids,
                verbose=verbose,
                interval=interval,
                align_to_admission=align_to_admission,
                **kwargs,
            )
            
            # CRITICAL: Expand WinTbl to time series if interval is specified
            # This must happen after loading but before caching, so all concepts
            # (including those without sub_concepts) get expanded
            from .table import WinTbl
            if isinstance(result, WinTbl) and interval is not None and not result.data.empty:
                idx_col = result.index_var
                dur_col = result.dur_var
                id_cols = result.id_vars
                
                if idx_col and dur_col and idx_col in result.data.columns and dur_col in result.data.columns:
                    if verbose:
                        logger.info("   扩展 WinTbl '%s' 到时间序列 (interval=%s)", concept_name, interval)
                    
                    # 扩展窗口到时间序列
                    expanded_rows = []
                    for _, row in result.data.iterrows():
                        start_time = row[idx_col]
                        duration = row[dur_col]
                        
                        # 计算结束时间（小时）
                        # R ricu 使用 seq(min, max, step) 包含终点，所以这里用 <=
                        end_time = start_time + duration
                        
                        # 生成时间序列（每个 interval）
                        interval_hours = interval.total_seconds() / 3600.0
                        current_time = np.floor(start_time / interval_hours) * interval_hours
                        
                        while current_time <= end_time:
                            new_row = {idx_col: current_time}
                            # 复制 ID 列
                            for col in id_cols:
                                if col in row.index:
                                    new_row[col] = row[col]
                            # 复制值列（除了 dur_col）
                            for col in result.data.columns:
                                if col not in [idx_col, dur_col] and col not in id_cols:
                                    new_row[col] = row[col]
                            expanded_rows.append(new_row)
                            current_time += interval_hours
                    
                    # 转换为 DataFrame
                    if expanded_rows:
                        expanded_df = pd.DataFrame(expanded_rows)
                        # 转换为 ICUTable
                        value_col = [c for c in expanded_df.columns if c not in id_cols and c != idx_col]
                        value_col = value_col[0] if value_col else None
                        from .table import ICUTable
                        result = ICUTable(
                            data=expanded_df,
                            id_columns=id_cols,
                            index_column=idx_col,
                            value_column=value_col,
                            unit_column=None,
                            time_columns=[],
                        )
                        if verbose:
                            logger.info("   ✅ 扩展完成: %d 行", len(expanded_df))
                        
        except Exception:
            with self._cache_lock:
                self._get_inflight().discard(concept_name)
            raise

        # 🔧 如果 _skip_concept_cache=True，跳过缓存写入
        if not _skip_concept_cache:
            self._store_in_disk_cache(concept_name, data_source, cache_key, result)

            with self._cache_lock:
                self._concept_cache[concept_name] = result
                self._concept_data_cache[concept_cache_key] = result  # 🚀 存入新缓存
                self._get_inflight().discard(concept_name)
        else:
            # 仅清除 inflight 标记
            with self._cache_lock:
                self._get_inflight().discard(concept_name)
        return result

    def _apply_aggregation(
        self,
        frame: pd.DataFrame,
        concept_name: str,
        id_columns: List[str],
        index_column: Optional[str],
        unit_column: Optional[str],
        aggregator: object,
    ) -> pd.DataFrame:
        # 🚀 CRITICAL FIX for norepi_rate: WinTbl expand before aggregation
        # R ricu does: aggregate(expand(...)), but pyricu was skipping expand
        # 
        # Check if this is WinTbl-style data (has endtime/duration)
        # and needs to be expanded before aggregation
        has_endtime = 'endtime' in frame.columns
        has_duration = 'duration' in frame.columns
        
        if (has_endtime or has_duration) and index_column and aggregator not in (None, False):
            # This is WinTbl data that needs expansion to time series
            from .ts_utils import expand
            
            # Determine end column
            end_col = 'duration' if has_duration else 'endtime'
            
            # Determine step size (default 1 hour for ICU data)
            step_size = pd.Timedelta(hours=1)
            
            # Determine columns to keep (value columns + unit if present)
            keep_vars = [concept_name] if concept_name in frame.columns else []
            if unit_column and unit_column in frame.columns:
                keep_vars.append(unit_column)
            
            # Additional value columns (not ID, not time, not end, not unit)
            excluded = set(id_columns + [index_column, end_col])
            if unit_column:
                excluded.add(unit_column)
            value_cols = [col for col in frame.columns 
                         if col not in excluded and col != concept_name]
            keep_vars.extend(value_cols)
            
            # Expand windows to hourly time series
            try:
                frame = expand(
                    frame,
                    start_var=index_column,
                    end_var=end_col,
                    step_size=step_size,
                    id_cols=id_columns,
                    keep_vars=keep_vars,
                )
                # After expand, index_column becomes the time column (no more endtime/duration)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to expand WinTbl data for {concept_name}: {e}")
                # Continue without expansion
        
        key_cols = [col for col in id_columns if col]
        if index_column:
            key_cols.append(index_column)

        if not key_cols:
            return frame

        # Check if concept_name column exists, if not try to find it
        if concept_name not in frame.columns:
            # Try to find the value column - could be from callback result
            value_cols = [col for col in frame.columns if col not in key_cols and col != unit_column]
            if value_cols:
                concept_name = value_cols[0]  # Use first non-key column as concept value
        
        if concept_name not in frame.columns:
            # Still not found, return frame as-is
            return frame
        
        agg_value = self._resolve_aggregator(frame[concept_name], aggregator)
        agg_spec: MutableMapping[str, object] = {concept_name: agg_value}

        if unit_column and unit_column in frame.columns:
            agg_spec[unit_column] = "first"

        grouped = frame.groupby(key_cols, dropna=False, as_index=False)
        aggregated = grouped.agg(agg_spec)
        
        # Flatten MultiIndex columns if any (from multiple aggregation functions)
        if isinstance(aggregated.columns, pd.MultiIndex):
            # Flatten: keep last level if it's meaningful, otherwise join
            new_columns = []
            for col in aggregated.columns:
                if isinstance(col, tuple):
                    # Join tuple elements, skipping empty strings
                    parts = [str(c) for c in col if c and str(c).strip()]
                    new_col = '_'.join(parts) if parts else concept_name
                    new_columns.append(new_col)
                else:
                    new_columns.append(str(col))
            aggregated.columns = new_columns
            # If concept_name is not in columns, try to find it
            if concept_name not in aggregated.columns:
                # Look for column that contains concept_name
                for col in aggregated.columns:
                    if concept_name.lower() in col.lower():
                        aggregated = aggregated.rename(columns={col: concept_name})
                        break

        ordered_cols = key_cols + [concept_name]
        if unit_column and unit_column in aggregated.columns:
            ordered_cols.append(unit_column)
        
        # Ensure all ordered_cols exist
        ordered_cols = [col for col in ordered_cols if col in aggregated.columns]

        return aggregated.loc[:, ordered_cols]

    @staticmethod
    def _resolve_aggregator(series: pd.Series, aggregator: object) -> object:
        if aggregator in (None, "auto"):
            return _default_aggregator_for_dtype(series)
        return aggregator

    @staticmethod
    def _normalise_aggregators(
        aggregate: Optional[Union[str, bool, Mapping[str, object]]],
        names: List[str],
    ) -> Dict[str, object]:
        if aggregate is None:
            return {name: "auto" for name in names}

        if not isinstance(aggregate, Mapping):
            return {name: aggregate for name in names}

        result: Dict[str, object] = {}
        for name in names:
            result[name] = aggregate.get(name, aggregate.get("*", "auto"))
        return result

    def _to_ricu_format(self, icu_table: ICUTable, concept_name: str, interval_hours: float = 1.0) -> pd.DataFrame:
        """
        将ICUTable转换为ricu.R兼容的格式

        Args:
            icu_table: ICUTable对象
            concept_name: 概念名称
            interval_hours: 时间间隔（小时），用于窗口展开

        Returns:
            ricu.R格式的DataFrame（只包含ID列、charttime和概念值列，静态数据只包含ID列和概念值列）
        """
        frame = icu_table.data.copy()

        # 识别静态数据（无时间列的概念）
        is_static_data = (
            icu_table.index_column is None or
            icu_table.index_column not in frame.columns or
            concept_name in ['age', 'sex', 'height', 'weight', 'bmi']  # 强制将这些识别为静态数据
        )

        if is_static_data:
            # 静态数据（如age, sex）: 返回ID列和概念值列
            if len(frame) == 0:
                return pd.DataFrame(columns=[concept_name])

            # 构建结果列：ID列 + 概念值列
            result_cols = []
            
            # 添加ID列
            for id_col in icu_table.id_columns:
                if id_col in frame.columns:
                    result_cols.append(id_col)
            
            # 添加概念值列
            if concept_name in frame.columns:
                result_cols.append(concept_name)
            elif icu_table.value_column and icu_table.value_column in frame.columns:
                # 重命名值列为概念名
                frame = frame.rename(columns={icu_table.value_column: concept_name})
                result_cols.append(concept_name)
            
            if not result_cols:
                return pd.DataFrame(columns=[concept_name])
            
            # 返回所有行（ricu格式保留所有匹配的行）
            return frame[result_cols].copy()
        else:
            # 时间序列数据: 只返回charttime和概念值列
            time_col = icu_table.index_column

            # 如果没有index_column，尝试识别时间列
            if time_col is None:
                possible_time_cols = [col for col in frame.columns if any(time_key in col.lower() for time_key in ['charttime', 'time', 'timestamp', 'measuredat', 'observationoffset'])]
                if possible_time_cols:
                    time_col = possible_time_cols[0]

            # 如果仍然没有时间列，但有数据，创建一个默认时间列
            if time_col is None and len(frame) > 0:
                frame = frame.copy()
                frame['charttime'] = range(len(frame))
                time_col = 'charttime'

            if time_col is None:
                # 如果真的没有时间列，返回只有概念值的数据框
                value_cols = [col for col in frame.columns if col not in icu_table.id_columns]
                if concept_name in value_cols:
                    return frame[[concept_name]]
                elif value_cols:
                    return frame[value_cols[0]].to_frame()
                else:
                    return pd.DataFrame(columns=[concept_name])

            value_cols = [col for col in frame.columns if col not in icu_table.id_columns + [time_col]]

            # 构建ricu.R格式 - 🔧 FIX: 也需要包含ID列，否则无法合并
            # ricu.R 的时间序列格式: ID列 + 时间列 + 值列
            result_cols = []
            
            # 添加ID列
            id_col_name = None
            for id_col in icu_table.id_columns:
                if id_col in frame.columns:
                    result_cols.append(id_col)
                    id_col_name = id_col
            
            # 添加时间列
            result_cols.append(time_col)

            # 添加概念值列（优先使用concept_name，否则使用第一个值列）
            value_col_name = None
            if concept_name in value_cols:
                result_cols.append(concept_name)
                value_col_name = concept_name
            elif value_cols:
                result_cols.append(value_cols[0])
                value_col_name = value_cols[0]

            # 确保只返回需要的列
            available_cols = [col for col in result_cols if col in frame.columns]
            result = frame[available_cols].copy()

            # 🔧 FIX: 窗口概念展开
            # 检查是否是窗口概念（如 mech_vent, vent_ind, supp_o2 等）
            if concept_name in ricu_compat.WINDOW_CONCEPTS or concept_name.endswith('_rate'):
                # 检查是否有结束时间或持续时间列
                endtime_col = None
                duration_col = None
                
                # 查找结束时间列
                for candidate in ['endtime', 'stop', 'end_time', 'end']:
                    if candidate in frame.columns:
                        endtime_col = candidate
                        break
                
                # 查找持续时间列
                for candidate in ['duration', 'dur', 'durationhours']:
                    if candidate in frame.columns:
                        duration_col = candidate
                        break
                
                # 如果有结束时间或持续时间，进行展开
                if endtime_col is not None or duration_col is not None:
                    # 准备展开数据
                    expand_df = result.copy()
                    
                    # 添加结束时间列（如果存在）
                    if endtime_col is not None and endtime_col not in expand_df.columns:
                        expand_df[endtime_col] = frame[endtime_col]
                    
                    # 添加持续时间列（如果存在）
                    if duration_col is not None and duration_col not in expand_df.columns:
                        expand_df[duration_col] = frame[duration_col]
                    
                    # 标准化列名用于展开函数
                    rename_map = {}
                    if id_col_name and id_col_name != 'id':
                        rename_map[id_col_name] = 'id'
                    if time_col != 'time':
                        rename_map[time_col] = 'time'
                    if value_col_name and value_col_name != concept_name:
                        rename_map[value_col_name] = concept_name
                    
                    if rename_map:
                        expand_df = expand_df.rename(columns=rename_map)
                    
                    # 调用窗口展开函数
                    expanded = ricu_compat.expand_interval_rows(
                        expand_df,
                        concept_name,
                        id_col='id',
                        time_col='time',
                        value_col=concept_name,
                        endtime_col=endtime_col if endtime_col else 'endtime',
                        duration_col=duration_col if duration_col else 'duration',
                        interval_hours=interval_hours,
                    )
                    
                    # 恢复原始列名
                    reverse_map = {v: k for k, v in rename_map.items()}
                    if reverse_map:
                        expanded = expanded.rename(columns=reverse_map)
                    
                    return expanded

            # 对于AUMC等数据库，保持原始时间列名称以支持ricu.R兼容性
            # 不强制重命名为charttime，让验证工具识别原始列名

            return result

    def _to_ricu_format_merged(self, merged_df: pd.DataFrame, concept_names: List[str]) -> pd.DataFrame:
        """
        将合并后的DataFrame转换为ricu.R兼容的格式

        Args:
            merged_df: 合并后的DataFrame
            concept_names: 概念名称列表

        Returns:
            ricu.R格式的DataFrame
        """
        frame = merged_df.reset_index()

        # 识别时间列和ID列 - 包含所有可能的时间列名称
        time_cols = [col for col in frame.columns if any(time_key in col.lower() for time_key in ['charttime', 'time', 'timestamp', 'measuredat', 'observationoffset', 'labresultoffset'])]
        id_cols = [col for col in frame.columns if any(id_key in col.lower() for id_key in ['id', 'stay_id', 'subject_id', 'patient'])]

        # 选择ricu.R需要的列
        result_cols = []

        # 添加时间列
        if time_cols:
            result_cols.append(time_cols[0])  # 使用第一个时间列

        # 添加概念值列
        for concept_name in concept_names:
            if concept_name in frame.columns:
                result_cols.append(concept_name)

        # 过滤并重命名
        if result_cols:
            result = frame[result_cols].copy()
            # 重命名时间列为charttime
            if time_cols:
                result = result.rename(columns={time_cols[0]: 'charttime'})
            return result
        else:
            return frame

    def _to_ricu_format_merged_enhanced(
        self, 
        tables: Mapping[str, ICUTable], 
        concept_names: List[str],
        interval: Optional[pd.Timedelta] = None,
    ) -> pd.DataFrame:
        """
        将多个概念表以ricu风格合并，实现完整的时间网格对齐和窗口展开
        
        这是增强版本，直接在原始tables上操作，实现：
        1. 窗口型概念的时间展开（mech_vent, *_rate等）
        2. 统一时间网格构建
        3. 所有概念对齐到网格
        4. 静态概念填充
        
        Args:
            tables: 概念名称到ICUTable的映射
            concept_names: 概念名称列表（保持顺序）
            interval: 时间间隔，默认1小时
            
        Returns:
            ricu风格的宽格式DataFrame
        """
        interval_hours = 1.0
        if interval is not None:
            if hasattr(interval, 'total_seconds'):
                interval_hours = interval.total_seconds() / 3600.0
            elif isinstance(interval, (int, float)):
                interval_hours = float(interval)
            else:
                interval_hours = 1.0
        
        # 将ICUTable转换为DataFrame字典
        concept_data: Dict[str, pd.DataFrame] = {}
        for name, table in tables.items():
            if isinstance(table, ICUTable):
                df = table.data.copy()
                # 重命名值列为概念名
                if name not in df.columns:
                    # 查找可能的值列
                    value_candidates = ['value', 'valuenum', table.index_column] if hasattr(table, 'index_column') else ['value', 'valuenum']
                    for cand in value_candidates:
                        if cand in df.columns and cand != name:
                            df = df.rename(columns={cand: name})
                            break
                concept_data[name] = df
            elif isinstance(table, pd.DataFrame):
                df = table.copy()
                if name not in df.columns:
                    for cand in ['value', 'valuenum']:
                        if cand in df.columns and cand != name:
                            df = df.rename(columns={cand: name})
                            break
                concept_data[name] = df
        
        if not concept_data:
            return pd.DataFrame()
        
        # 检测ID列和时间列
        id_col = None
        time_col = None
        for df in concept_data.values():
            if df is None or df.empty:
                continue
            # 检测ID列
            for cand in ['stay_id', 'subject_id', 'patientunitstayid', 'admissionid', 'patientid']:
                if cand in df.columns:
                    id_col = cand
                    break
            # 检测时间列 - FIX: 添加 eICU 的时间列和区间格式的 start 列
            for cand in ['charttime', 'time', 'starttime', 'start', 'index_var', 'measuredat',
                         'nursingchartoffset', 'labresultoffset', 'observationoffset',
                         'respchartoffset', 'intakeoutputoffset', 'infusionoffset']:
                if cand in df.columns:
                    time_col = cand
                    break
            if id_col and time_col:
                break
        
        if not id_col:
            id_col = 'stay_id'  # 默认值
        if not time_col:
            time_col = 'charttime'  # 默认值
        
        # 使用ricu_compat模块进行合并
        result = ricu_compat.merge_concepts_ricu_style(
            concept_data,
            id_col=id_col,
            time_col=time_col,
            interval_hours=interval_hours,
        )
        
        # 确保概念列按请求的顺序排列
        final_cols = [id_col, time_col]
        for name in concept_names:
            if name in result.columns:
                final_cols.append(name)
        
        # 添加任何其他值列（可能是子组件）
        for col in result.columns:
            if col not in final_cols:
                final_cols.append(col)
        
        final_cols = [c for c in final_cols if c in result.columns]
        result = result[final_cols]
        
        return result

def _apply_callback(
    frame: pd.DataFrame,
    source: ConceptSource,
    concept_name: str,
    unit_column: Optional[str] = None,
    resolver: Optional['ConceptResolver'] = None,
    patient_ids: Optional[List] = None,
    data_source: Optional['ICUDataSource'] = None,
) -> pd.DataFrame:
    callback = source.callback
    if not callback:
        return frame

    expr = callback.strip()

    if expr == "identity_callback":
        return frame

    if expr == "aumc_death":
        # R ricu logic: is_true(index_var - val_var < hours(72L))
        def _pick(col: Optional[str], fallbacks: List[str]) -> Optional[str]:
            ordered = [col] if col else []
            ordered.extend(fallbacks)
            for candidate in ordered:
                if candidate and candidate in frame.columns:
                    return candidate
            return None

        index_col = _pick(source.index_var, ["dateofdeath", "deathdate", "dod", "death_time"])
        value_col = _pick(source.value_var, [concept_name, "dischargedat", "dischargetime", "dischargeat"])

        if index_col is None or value_col is None:
            return frame

        df = frame.copy()
        death_ts = pd.to_datetime(df[index_col], errors="coerce")
        discharge_ts = pd.to_datetime(df[value_col], errors="coerce")
        delta = death_ts - discharge_ts
        within_window = delta < pd.Timedelta(hours=72)
        within_window = within_window & death_ts.notna() & discharge_ts.notna()
        df[value_col] = within_window.astype(int)
        return df

    # Handle eicu_age - process eICU age data (convert '> 89' to 90)
    if re.fullmatch(r"transform_fun\(eicu_age\)", expr):
        from .callback_utils import eicu_age
        return eicu_age(frame, val_col=concept_name)

    # Handle percent_as_numeric - remove '%' and convert to numeric
    if re.fullmatch(r"transform_fun\(percent_as_numeric\)", expr):
        series = frame[concept_name].copy()

        def parse_percent(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, str):
                val_clean = val.strip().rstrip('%')
                try:
                    return float(val_clean)
                except (ValueError, AttributeError):
                    return np.nan
            try:
                return float(val)
            except (TypeError, ValueError):
                return np.nan

        def missing_mask(values: pd.Series) -> pd.Series:
            mask = values.isna()
            as_str = values.astype(str).str.strip().str.lower()
            mask |= as_str.eq("") | as_str.eq("nan") | as_str.eq("none")
            return mask

        mask = missing_mask(series)
        for fallback_col in ("value", "valuetext"):
            if fallback_col in frame.columns and fallback_col != concept_name:
                fallback_series = frame[fallback_col]
                series = series.where(~mask, fallback_series)
                mask = missing_mask(series)

        frame.loc[:, concept_name] = series.apply(parse_percent)
        return frame

    match = re.fullmatch(r"transform_fun\(set_val\((.+)\)\)", expr, flags=re.DOTALL)
    if match:
        value = _parse_literal(match.group(1))
        frame = frame.copy()
        if concept_name in frame.columns:
            frame.drop(columns=[concept_name], inplace=True)
        dtype = "boolean" if isinstance(value, bool) else None
        result_series = pd.Series([value] * len(frame), index=frame.index, dtype=dtype)
        frame[concept_name] = result_series
        return frame

    # Handle comp_na() without arguments - check if value is not NA
    if re.fullmatch(r"transform_fun\(comp_na\(\)\)", expr):
        series = frame[concept_name]
        # Convert to boolean: True if not NA, False if NA
        frame.loc[:, concept_name] = series.notna().astype(float)
        return frame

    match = re.fullmatch(r"transform_fun\(comp_na\(`(.+?)`,\s*(.+)\)\)", expr, flags=re.DOTALL)
    if match:
        op_token = match.group(1)
        value = _parse_literal(match.group(2))
        op_map = {
            "==": operator.eq,
            "!=": operator.ne,
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
        }
        if op_token not in op_map:
            raise NotImplementedError(
                f"Unsupported comparison operator '{op_token}' in callback '{expr}'."
            )
        series = frame[concept_name]
        if isinstance(value, (int, float)) and not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
        comparator = op_map[op_token]
        comparison = series.apply(
            lambda item: False if pd.isna(item) else comparator(item, value)
        ).astype("boolean")
        frame = frame.copy()
        frame.drop(columns=[concept_name], inplace=True)
        frame[concept_name] = comparison
        return frame

    match = re.fullmatch(r"transform_fun\(binary_op\(`(.+?)`,\s*(.+)\)\)", expr, flags=re.DOTALL)
    if match:
        symbol = match.group(1)
        value = _parse_literal(match.group(2))
        frame = frame.copy()
        series = pd.to_numeric(frame[concept_name], errors="coerce")
        result = _apply_binary_op(symbol, series, value)
        frame.loc[:, concept_name] = result
        return frame

    # 匹配 mimic_sampling (R ricu callback-itm.R)
    # mimic_sampling(x, val_var, aux_time, ...)
    # 功能：1) combine_date_time(x, aux_time, hours(12L))
    #      2) set(x, j = val_var, value = !is.na(x[[val_var]]))
    if expr == "mimic_sampling":
        frame = frame.copy()
        val_var = source.value_var or concept_name
        aux_time = source.params.get("aux_time") if source.params else None
        
        # 1. combine_date_time: 如果aux_time是NA，使用index_column + 12小时
        if aux_time and aux_time in frame.columns:
            # 找到实际的index列（通常是charttime, starttime等）
            # 检查是否有明确的index_var
            index_col = source.index_var
            if not index_col:
                # 尝试从表配置中获取
                time_cols = [col for col in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[col])]
                if time_cols:
                    # 优先使用非aux_time的datetime列
                    index_col = next((col for col in time_cols if col != aux_time), time_cols[0])
            
            if index_col and index_col in frame.columns:
                # 如果aux_time是NA，使用index_col + 12小时
                mask = frame[aux_time].isna()
                if mask.any():
                    frame.loc[mask, aux_time] = pd.to_datetime(frame.loc[mask, index_col], errors='coerce') + pd.Timedelta(hours=12)
                # 更新index_column为aux_time（使用aux_time作为时间索引）
                if index_col != aux_time:
                    # 将aux_time的值复制到index_col，然后删除aux_time
                    frame[index_col] = pd.to_datetime(frame[aux_time], errors='coerce')
                    frame = frame.drop(columns=[aux_time])
        
        # 2. 将val_var转换为布尔值（非NA为True）
        if val_var in frame.columns:
            frame[concept_name] = frame[val_var].notna().astype(bool)
            if val_var != concept_name:
                frame = frame.drop(columns=[val_var])
        else:
            # 如果val_var不存在，创建concept_name列（全False）
            frame[concept_name] = False
        
        return frame
    
    # 匹配 apply_map(c(...), var = 'sub_var') 或 apply_map(c(...))
    match = re.fullmatch(r"apply_map\(\s*c\((.+?)\)\s*(?:,\s*var\s*=\s*['\"](.+?)['\"])?\s*\)", expr, flags=re.DOTALL)
    if match:
        mapping = _parse_mapping(match.group(1))
        var_param = match.group(2) if match.group(2) else None
        
        frame = frame.copy()
        
        # 解析 var_param，如果是 'sub_var'，使用 source.sub_var 的实际值
        target_col = None
        if var_param:
            if var_param == 'sub_var' and source.sub_var:
                # var='sub_var' 表示映射 sub_var 列（如 itemid）
                target_col = source.sub_var
            elif var_param == 'val_col' and concept_name in frame.columns:
                # var='val_col' 表示映射值列（concept_name）
                target_col = concept_name
            elif var_param in frame.columns:
                # 直接使用 var_param 作为列名
                target_col = var_param
        
        # 如果指定了目标列且存在，映射该列；否则映射concept_name列
        if target_col and target_col in frame.columns:
            # 映射指定的列
            series = frame[target_col]
            def mapper(val):
                if pd.isna(val):
                    return val
                # 尝试直接匹配，然后尝试字符串匹配
                result = mapping.get(val, mapping.get(str(val), val))
                return result
            
            # 显式转换为 object 类型以避免 FutureWarning
            # 当映射值的类型与原列类型不兼容时（如字符串映射到 int32），需要先转换类型
            mapped_series = series.map(mapper)
            if frame[target_col].dtype != mapped_series.dtype:
                frame[target_col] = frame[target_col].astype(object)
            frame.loc[:, target_col] = mapped_series
        elif concept_name in frame.columns:
            # 默认映射concept_name列
            series = frame[concept_name]
            def mapper(val):
                if pd.isna(val):
                    return val
                return mapping.get(val, mapping.get(str(val), val))
            
            # 同样处理类型不兼容问题
            mapped_series = series.map(mapper)
            if frame[concept_name].dtype != mapped_series.dtype:
                frame[concept_name] = frame[concept_name].astype(object)
            frame.loc[:, concept_name] = mapped_series
        
        return frame

    match = re.fullmatch(r"convert_unit\((.+)\)", expr, flags=re.DOTALL)
    if match:
        arguments = _split_arguments(match.group(1))
        if not arguments:
            raise NotImplementedError(f"Callback '{callback}' is empty.")

        symbol, value = _parse_binary_op(arguments[0])
        new_unit = _strip_quotes(arguments[1]) if len(arguments) > 1 else None
        old_unit = _strip_quotes(arguments[2]) if len(arguments) > 2 else None

        frame = frame.copy()
        
        # 如果 source.unit_var 未指定，尝试自动检测单位列
        actual_unit_var = source.unit_var or unit_column
        
        # 如果仍然没有，尝试常见的单位列名
        if not actual_unit_var and 'valueuom' in frame.columns:
            actual_unit_var = 'valueuom'
        elif not actual_unit_var and 'unit' in frame.columns:
            actual_unit_var = 'unit'
        
        if actual_unit_var and actual_unit_var in frame.columns:
            unit_series = frame[actual_unit_var].fillna('').astype(str)
            if old_unit:
                case_flag = False
                try:
                    mask = unit_series.str.contains(old_unit, case=case_flag, na=False, regex=True)
                except re.error:
                    mask = unit_series.str.contains(re.escape(old_unit), case=case_flag, na=False, regex=True)
                # ⚠️ 不匹配空单位行: MIMIC-IV中单位为空时值已经正确
            else:
                # 如果old_unit为None，转换所有行（R ricu行为）
                mask = pd.Series(True, index=frame.index)
        else:
            mask = pd.Series(True, index=frame.index)

        numeric = pd.to_numeric(frame.loc[mask, concept_name], errors="coerce")
        transformed = _apply_binary_op(symbol, numeric, value)
        
        # 明确转换类型以避免 dtype 不兼容警告
        frame.loc[mask, concept_name] = transformed.astype('float64')

        # 更新单位列
        if new_unit and actual_unit_var and actual_unit_var in frame.columns:
            frame.loc[mask, actual_unit_var] = new_unit

        return frame

    match = re.fullmatch(r"combine_callbacks\((.+)\)", expr, flags=re.DOTALL)
    if match:
        frame_result = frame
        for arg in _split_arguments(match.group(1)):
            nested = arg.strip()
            if not nested:
                continue
            nested_source = replace(source, callback=nested)
            frame_result = _apply_callback(frame_result, nested_source, concept_name, unit_column)
        return frame_result
    
    # Handle dex_to_10 callback (convert different dextrose concentrations to D10 equivalent)
    # Format: dex_to_10(ids, factors) or dex_to_10(c(...), c(...))
    match = re.fullmatch(r"dex_to_10\((.+)\)", expr, flags=re.DOTALL)
    if match:
        args = _split_arguments(match.group(1))
        if len(args) >= 2:
            # Parse itemids and factors
            id_arg = args[0].strip()
            factor_arg = args[1].strip()
            
            # Parse list/vector syntax: c(228140L, 220952L) or list(...)
            def parse_vector(s):
                # Handle c(...) or list(...)
                vec_match = re.search(r'(?:c|list)\(([^)]+)\)', s)
                if vec_match:
                    items_str = vec_match.group(1)
                    items = [int(re.sub(r'L$', '', x.strip())) for x in items_str.split(',')]
                    return items
                # Handle single value
                else:
                    return [int(re.sub(r'L$', '', s.strip()))]
            
            try:
                itemids = parse_vector(id_arg)
                factors = parse_vector(factor_arg)
                
                # Apply conversion factors
                sub_var = source.sub_var if hasattr(source, 'sub_var') else 'itemid'
                # Try to find the value column: concept_name, or unit_column (which is the value column before renaming)
                val_col = None
                if concept_name in frame.columns:
                    val_col = concept_name
                elif unit_column and unit_column in frame.columns:
                    val_col = unit_column
                # Fallback: try common value column names
                elif 'rate' in frame.columns:
                    val_col = 'rate'
                elif 'amount' in frame.columns:
                    val_col = 'amount'
                elif 'valuenum' in frame.columns:
                    val_col = 'valuenum'
                
                if sub_var in frame.columns and val_col:
                    frame = frame.copy()
                    for itemid, factor in zip(itemids, factors):
                        mask = frame[sub_var] == itemid
                        if mask.any():
                            frame.loc[mask, val_col] = frame.loc[mask, val_col] * factor
            except Exception:
                # Silently skip if parsing fails
                pass
        return frame
    
    # Handle ts_to_win_tbl callback
    match = re.fullmatch(r"ts_to_win_tbl\((.+)\)", expr, flags=re.DOTALL)
    if match:
        # Parse the duration expression (e.g., "mins(1L)")
        dur_expr = match.group(1).strip()
        # Simple parsing for common duration patterns
        if 'mins(' in dur_expr:
            mins_match = re.search(r'mins\((\d+)', dur_expr)
            if mins_match:
                duration = pd.Timedelta(minutes=int(mins_match.group(1)))
            else:
                duration = pd.Timedelta(minutes=1)  # default
        elif 'hours(' in dur_expr:
            hours_match = re.search(r'hours\((\d+)', dur_expr)
            if hours_match:
                duration = pd.Timedelta(hours=int(hours_match.group(1)))
            else:
                duration = pd.Timedelta(hours=1)  # default
        else:
            duration = pd.Timedelta(minutes=1)  # default fallback
        
        # Add duration column
        frame = frame.copy()
        frame['dur_var'] = duration
        return frame
    
    # Handle mimic_rate_mv callback (for infusion rates)
    if expr.strip() == "mimic_rate_mv":
        from .callback_utils import mimic_rate_mv
        # Call the callback with appropriate parameters
        id_cols = [col for col in frame.columns if 'id' in col.lower()]
        # stop_var is stored in params dict
        stop_var = source.params.get('stop_var', None) if source.params else None
        unit_col = source.unit_var if hasattr(source, 'unit_var') else None
        val_col = concept_name
        
        return mimic_rate_mv(
            frame,
            val_col=val_col,
            unit_col=unit_col,
            stop_var=stop_var,
            id_cols=id_cols
        )
    
    # Handle mimic_dur_inmv callback (for infusion durations)
    if expr.strip() == "mimic_dur_inmv":
        from .callback_utils import mimic_dur_inmv
        # Call the callback with appropriate parameters
        id_cols = [col for col in frame.columns if 'id' in col.lower()]
        # stop_var and grp_var are stored in params dict
        stop_var = source.params.get('stop_var', None) if source.params else None
        grp_var = source.params.get('grp_var', None) if source.params else None
        # Use unit_column from parent context or source.unit_var
        unit_col = unit_column or (source.unit_var if hasattr(source, 'unit_var') else None)
        val_col = concept_name
        
        return mimic_dur_inmv(
            frame,
            val_col=val_col,
            grp_var=grp_var,
            stop_var=stop_var,
            id_cols=id_cols,
            unit_col=unit_col
        )
    
    # Handle mimic_dur_incv callback (for CareVue durations)
    if expr.strip() == "mimic_dur_incv":
        from .callback_utils import mimic_dur_incv
        # Call the callback with appropriate parameters
        id_cols = [col for col in frame.columns if 'id' in col.lower()]
        # grp_var is stored in params dict
        grp_var = source.params.get('grp_var', None) if source.params else None
        # Use unit_column from parent context or source.unit_var
        unit_col = unit_column or (source.unit_var if hasattr(source, 'unit_var') else None)
        val_col = concept_name
        
        return mimic_dur_incv(
            frame,
            val_col=val_col,
            grp_var=grp_var,
            id_cols=id_cols,
            unit_col=unit_col
        )
    
    # Handle mimic_rate_cv callback (for CareVue infusion rates)
    if expr.strip() == "mimic_rate_cv":
        from .callback_utils import mimic_rate_cv
        # Call the callback with appropriate parameters
        id_cols = [col for col in frame.columns if 'id' in col.lower()]
        # grp_var is stored in params dict
        grp_var = source.params.get('grp_var', None) if source.params else None
        unit_col = source.unit_var if hasattr(source, 'unit_var') else None
        val_col = concept_name
        
        return mimic_rate_cv(
            frame,
            val_col=val_col,
            grp_var=grp_var,
            unit_col=unit_col,
            id_cols=id_cols
        )

    if expr.strip() == "vent_flag":
        from .callback_utils import vent_flag

        id_cols = [col for col in frame.columns if 'id' in col.lower()]
        index_var = source.index_var
        
        # 🔥 FIX: 如果 source.index_var 是 None，尝试从表配置获取默认 index_var
        # eICU vent_start 源有 index_var: None，但表 respiratorycare 的默认是 respcarestatusoffset
        if index_var is None and data_source is not None:
            try:
                table_cfg = data_source.config.get_table(source.table)
                if table_cfg and table_cfg.defaults:
                    index_var = table_cfg.defaults.index_var
                    if DEBUG_MODE:
                        print(f"   🔧 vent_flag: source.index_var=None，使用表默认 index_var='{index_var}'")
            except Exception:
                pass
        
        # 🔥 R ricu vent_flag: val_var 是原始列名（如 ventstartoffset），不是概念名
        # vent_flag 会将 val_var 的值作为新的时间索引，然后将 val_var 设为 TRUE
        # 🔧 FIX: 如果 value_var 已被重命名为 concept_name，使用 concept_name
        val_col = source.value_var if hasattr(source, 'value_var') and source.value_var else concept_name
        if val_col not in frame.columns and concept_name in frame.columns:
            val_col = concept_name
        return vent_flag(
            frame,
            val_col=val_col,
            index_var=index_var,
            id_cols=id_cols,
        )

    match = re.fullmatch(r"eicu_duration\(\s*gap_length\s*=\s*(.+)\)", expr, flags=re.DOTALL)
    if match:
        from .callback_utils import eicu_duration_callback

        gap_arg = match.group(1)
        # Parse interval expression directly
        gap_expr = gap_arg.strip()
        interval_match = re.fullmatch(r"([a-zA-Z]+)\((.+)\)", gap_expr)
        if interval_match:
            unit = interval_match.group(1).lower()
            value = _parse_literal(interval_match.group(2))
            if unit in {"min", "mins", "minute", "minutes"}:
                gap = pd.to_timedelta(value, unit="m")
            elif unit in {"hour", "hours"}:
                gap = pd.to_timedelta(value, unit="h")
            elif unit in {"sec", "secs", "second", "seconds"}:
                gap = pd.to_timedelta(value, unit="s")
            elif unit in {"day", "days"}:
                gap = pd.to_timedelta(value, unit="d")
            else:
                raise ValueError(f"Unsupported interval unit '{unit}' in expression '{gap_expr}'")
        else:
            raise ValueError(f"Unsupported interval expression '{gap_arg}'")
        
        callback_fn = eicu_duration_callback(gap)
        # 只使用患者级别的ID列进行分组，不要使用行级别的唯一ID（如infusiondrugid）
        # 否则每组只有一行，duration计算会变成0
        patient_id_cols = ['patientunitstayid', 'stay_id', 'icustay_id', 'hadm_id', 'admissionid', 'patientid']
        id_cols = [col for col in patient_id_cols if col in frame.columns]
        if not id_cols:
            # 回退到通用检测，但排除明显的行级别ID
            excluded_patterns = ['infusion', 'drug', 'event', 'row', 'fluid']
            id_cols = [col for col in frame.columns 
                      if 'id' in col.lower() 
                      and not any(pat in col.lower() for pat in excluded_patterns)]
        index_var = source.index_var
        return callback_fn(
            frame,
            val_col=concept_name,
            index_var=index_var,
            id_cols=id_cols,
        )

    # Handle eicu_rate_kg(ml_to_mcg = VALUE) - eICU dose rate conversion with weight
    match = re.fullmatch(r"eicu_rate_kg\(\s*ml_to_mcg\s*=\s*(.+)\)", expr, flags=re.DOTALL)
    if match:
        from .callback_utils import eicu_rate_kg_callback
        
        ml_to_mcg = float(match.group(1))
        callback_fn = eicu_rate_kg_callback(ml_to_mcg)
        
        # Get necessary variables
        val_var = source.value_var or concept_name
        sub_var = source.sub_var
        weight_var = source.params.get('weight_var', 'admissionweight') if source.params else 'admissionweight'
        
        return callback_fn(
            frame,
            val_var=val_var,
            sub_var=sub_var,
            weight_var=weight_var,
            concept_name=concept_name,
            data_source=data_source,
            patient_ids=patient_ids,
        )
        
    match = re.fullmatch(r"eicu_rate_units\((.+)\)", expr, flags=re.DOTALL)
    if match:
        from .callback_utils import eicu_rate_units_callback

        args = _split_arguments(match.group(1))
        if len(args) < 2:
            raise ValueError(f"eicu_rate_units requires two arguments, got '{expr}'")

        def _arg_to_float(text: str) -> float:
            part = text.split("=", 1)[1] if "=" in text else text
            return float(_parse_literal(part.strip()))

        ml_to_mcg = _arg_to_float(args[0])
        mcg_to_units = _arg_to_float(args[1])
        callback_fn = eicu_rate_units_callback(ml_to_mcg, mcg_to_units)

        val_var = source.value_var or concept_name
        sub_var = source.sub_var

        return callback_fn(
            frame,
            val_var=val_var,
            sub_var=sub_var,
            concept_name=concept_name,
        )

    if expr == "aumc_rate_kg":
        from .callback_utils import aumc_rate_kg

        val_var = source.value_var or concept_name
        unit_var = source.unit_var or unit_column
        rel_weight = source.params.get("rel_weight") if source.params else None
        rate_uom = source.params.get("rate_uom") if source.params else None
        if rate_uom is None and "rateunit" in frame.columns:
            rate_uom = "rateunit"
        stop_var = source.params.get("stop_var") if source.params else None
        index_var = source.index_var
        
        # source.index_var may be None, use table default as fallback
        # For AUMC drugitems, the index_var should be 'start'
        if not index_var and source.table == 'drugitems':
            index_var = 'start'

        # 🔧 FIX: 获取体重概念并合并到 frame 中
        # R ricu 在回调中使用 add_weight(res, env, "weight") 获取体重
        # pyricu 需要在调用回调前加载 weight 概念
        if 'weight' not in frame.columns and resolver is not None and data_source is not None:
            try:
                # 获取患者ID列
                id_cols = [c for c in frame.columns if c.lower().endswith('id') and c != 'itemid']
                if id_cols:
                    unique_ids = frame[id_cols[0]].unique().tolist()
                    # 加载 weight 概念
                    weight_table = resolver._load_single_concept(
                        'weight',
                        data_source,
                        aggregator=False,  # 不聚合，保留原始值
                        patient_ids={id_cols[0]: unique_ids},
                        verbose=False,
                        _bypass_callback=True,  # 避免回调循环
                    )
                    if weight_table is not None and not weight_table.data.empty:
                        weight_df = weight_table.data
                        # 确保weight列是数值型
                        if 'weight' in weight_df.columns:
                            weight_df['weight'] = pd.to_numeric(weight_df['weight'], errors='coerce')
                            # 合并到frame
                            merge_cols = [c for c in id_cols if c in weight_df.columns]
                            if merge_cols:
                                frame = frame.merge(
                                    weight_df[merge_cols + ['weight']].drop_duplicates(),
                                    on=merge_cols,
                                    how='left'
                                )
            except Exception as e:
                # 如果获取体重失败，使用默认值
                if DEBUG_MODE:
                    print(f"   ⚠️  获取体重失败: {e}")
                pass

        return aumc_rate_kg(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            unit_col=unit_var,
            rel_weight_col=rel_weight,
            rate_unit_col=rate_uom,
            index_col=index_var,
            stop_col=stop_var,
        )

    # Handle aumc_rate callback - combine unit_var and rate_var into unit/rate format
    # R: x <- x[, c(unit_var) := do_call(.SD, paste, sep = "/"), .SDcols = c(unit_var, rate_var)]
    if expr == "aumc_rate":
        rate_var = getattr(source, 'rate_var', None)
        if not rate_var and source.params:
            rate_var = source.params.get("rate_var")
        unit_var = source.unit_var or unit_column
        
        if rate_var and unit_var and rate_var in frame.columns and unit_var in frame.columns:
            frame = frame.copy()
            # Combine unit and rate into "unit/rate" format
            frame[unit_var] = frame[unit_var].astype(str) + "/" + frame[rate_var].astype(str)
        return frame

    match = re.fullmatch(r"aumc_rate_units\(\s*([0-9eE+\-\.]+)\s*\)", expr)
    if match:
        from .callback_utils import aumc_rate_units_callback

        factor = float(match.group(1))
        callback_fn = aumc_rate_units_callback(factor)

        val_var = source.value_var or concept_name
        unit_var = source.unit_var or unit_column
        rate_uom = source.params.get("rate_uom") if source.params else None
        if rate_uom is None and "rateunit" in frame.columns:
            rate_uom = "rateunit"
        stop_var = source.params.get("stop_var") if source.params else None

        return callback_fn(
            frame,
            val_col=val_var,
            unit_col=unit_var,
            rate_unit_col=rate_uom,
            stop_col=stop_var,
            concept_name=concept_name,
        )

    if expr == "aumc_dur":
        from .callback_utils import aumc_dur

        val_var = source.value_var or concept_name
        # stop_var and grp_var can be direct attributes on source or in source.params
        stop_var = getattr(source, 'stop_var', None)
        if not stop_var and source.params:
            stop_var = source.params.get("stop_var")
        grp_var = getattr(source, 'grp_var', None)
        if not grp_var and source.params:
            grp_var = source.params.get("grp_var")
        index_var = source.index_var

        return aumc_dur(
            frame,
            val_col=val_var,
            stop_var=stop_var,
            grp_var=grp_var,
            index_var=index_var,
            concept_name=concept_name,
        )

    # Handle aumc_bxs callback - negate values where direction is '-'
    # R implementation: x[get(dir_var) == "-", val_var := -1L * get(val_var)]
    if expr == "aumc_bxs":
        dir_var = getattr(source, 'dir_var', None)
        if not dir_var and source.params:
            dir_var = source.params.get("dir_var")
        if not dir_var:
            dir_var = "tag"  # default for AUMC
        
        val_var = concept_name  # Value column has already been renamed to concept_name
        
        if dir_var in frame.columns and val_var in frame.columns:
            # Negate values where direction is '-'
            mask = frame[dir_var] == '-'
            if mask.any():
                frame = frame.copy()
                frame.loc[mask, val_var] = -1 * frame.loc[mask, val_var]
        return frame

    # Handle eicu_age callback
    if expr == "transform_fun(eicu_age)":
        from .callback_utils import eicu_age
        return eicu_age(frame, val_col=concept_name)

    # Handle aumc_rass callback
    if expr == "transform_fun(aumc_rass)":
        # Apply aumc_rass transformation: extract first 2 characters as integer
        # Similar to ricu's: as.integer(substr(x, 1L, 2L))
        series = frame[concept_name].copy()
        series = series.astype(str).str[:2]
        series = pd.to_numeric(series, errors='coerce')
        frame[concept_name] = series
        return frame

    if expr.strip() == "distribute_amount":
        from .callback_utils import distribute_amount
        end_col = source.params.get("end_var") if source.params else None
        if not end_col:
            end_col = source.params.get("dur_var") if source.params else None
        if not end_col and "endtime" in frame.columns:
            end_col = "endtime"
        index_col = source.index_var
        # 🔧 FIX: 添加 starttime 作为 fallback，用于 inputevents 表的数据 (如 ins)
        if not index_col:
            for candidate in ["charttime", "starttime", "time"]:
                if candidate in frame.columns:
                    index_col = candidate
                    break
        unit_col = unit_column or source.unit_var
        if not unit_col:
            if "rateuom" in frame.columns:
                unit_col = "rateuom"
            elif "valueuom" in frame.columns:
                unit_col = "valueuom"
        if not end_col or end_col not in frame.columns:
            return frame
        if not index_col or index_col not in frame.columns:
            return frame
        return distribute_amount(
            frame,
            val_col=concept_name,
            unit_col=unit_col,
            end_col=end_col,
            index_col=index_col,
        )

    if expr.strip() == "mimv_rate":
        from .callback_utils import mimv_rate
        duration_col = None
        start_col = source.index_var
        if not start_col:
            if "starttime" in frame.columns:
                start_col = "starttime"
        end_col = None
        if source.params:
            end_col = source.params.get("dur_var") or source.params.get("end_var")
        if not end_col and "endtime" in frame.columns:
            end_col = "endtime"
        
        # 首先检查是否已经有计算好的duration列 (概念名_dur格式)
        possible_dur_cols = [concept_name + '_dur', 'duration', '__duration__']
        for col in possible_dur_cols:
            if col in frame.columns:
                duration_col = col
                break
        
        # 如果没有现成的duration列，尝试从start和end计算
        if not duration_col:
            if end_col and end_col in frame.columns and start_col and start_col in frame.columns:
                start = pd.to_datetime(frame[start_col], errors="coerce")
                stop = pd.to_datetime(frame[end_col], errors="coerce")
                frame = frame.copy()
                frame["__duration__"] = stop - start
                duration_col = "__duration__"
            elif end_col and end_col in frame.columns:
                duration_col = end_col
        
        if not duration_col or duration_col not in frame.columns:
            return frame
        amount_col = concept_name
        if source.params:
            alt_amount = source.params.get("amount_var")
            if alt_amount and alt_amount in frame.columns:
                amount_col = alt_amount
        unit_col = unit_column or source.unit_var
        if not unit_col:
            if "rateuom" in frame.columns:
                unit_col = "rateuom"
            elif "valueuom" in frame.columns:
                unit_col = "valueuom"
        auom_col = None
        if source.params:
            auom_col = source.params.get("auom_var")
        if not auom_col or auom_col not in frame.columns:
            if "amountuom" in frame.columns:
                auom_col = "amountuom"
            else:
                auom_col = unit_col
        return mimv_rate(
            frame,
            val_col=concept_name,
            unit_col=unit_col,
            dur_var=duration_col,
            amount_var=amount_col,
            auom_var=auom_col,
        )

    match = re.fullmatch(r"dex_to_10\((.+)\)", expr, flags=re.DOTALL)
    if match:
        from .callback_utils import dex_to_10

        args = _split_arguments(match.group(1))
        if len(args) < 2:
            return frame

        ids = _parse_r_value(args[0])
        factors = _parse_r_value(args[1])
        if not isinstance(ids, list):
            ids = [ids]
        if not isinstance(factors, list):
            factors = [factors]

        callback_fn = dex_to_10(ids, factors)
        sub_var = source.sub_var
        if not sub_var or sub_var not in frame.columns:
            return frame
        return callback_fn(
            frame,
            sub_var=sub_var,
            val_col=concept_name,
        )

    if expr.strip() == "eicu_dex_med":
        from .callback_utils import eicu_dex_med as eicu_dex_med_cb

        val_var = source.value_var or concept_name
        dur_var = None
        if source.params:
            dur_var = source.params.get("dur_var") or source.params.get("stop_var")
        if not dur_var or dur_var not in frame.columns:
            if "duration" in frame.columns:
                dur_var = "duration"
            elif "drugstopoffset" in frame.columns:
                dur_var = "drugstopoffset"
        if not dur_var or dur_var not in frame.columns:
            return frame

        return eicu_dex_med_cb(
            frame,
            val_var=val_var,
            dur_var=dur_var,
            concept_name=concept_name,
        )

    if expr.strip() == "eicu_dex_inf":
        from .callback_utils import eicu_dex_inf as eicu_dex_inf_cb

        val_var = source.value_var or concept_name
        index_var = source.index_var

        return eicu_dex_inf_cb(
            frame,
            val_var=val_var,
            index_var=index_var,
        )

    # blood_cell_ratio callback - convert absolute cell counts to percentage
    # R ricu logic: 100 * value / wbc
    # Used for lymphocytes, neutrophils, etc.
    if expr.strip() == "blood_cell_ratio":
        DEBUG_CALLBACK = False  # Toggle for debugging
        if DEBUG_CALLBACK:
            print(f"  [CALLBACK DEBUG] {concept_name} blood_cell_ratio 开始")
            print(f"    frame.shape = {frame.shape}, columns = {list(frame.columns)}")
            if concept_name in frame.columns:
                print(f"    输入值: {frame[concept_name].values}")
        
        if resolver is None:
            if DEBUG_CALLBACK:
                print(f"    [SKIP] resolver is None")
            # Cannot convert without resolver to load WBC, return as-is
            if concept_name in frame.columns:
                frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            return frame
        
        # Determine ID column based on database
        # AUMC uses 'admissionid', MIMIC uses 'stay_id', eICU uses 'patientunitstayid'
        id_col = None
        for possible_id in ['admissionid', 'stay_id', 'patientunitstayid', 'subject_id', 'icustay_id']:
            if possible_id in frame.columns:
                id_col = possible_id
                break
        
        if id_col is None:
            if DEBUG_CALLBACK:
                print(f"    [SKIP] id_col is None")
            if concept_name in frame.columns:
                frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            return frame
        
        if DEBUG_CALLBACK:
            print(f"    id_col = {id_col}")
        
        frame_patient_ids = frame[id_col].unique().tolist()
        if len(frame_patient_ids) == 0:
            if DEBUG_CALLBACK:
                print(f"    [SKIP] no patients")
            if concept_name in frame.columns:
                frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            return frame
        
        if DEBUG_CALLBACK:
            print(f"    patients = {frame_patient_ids}")
        
        try:
            # Load WBC concept for the same patients
            # IMPORTANT: Use merge=False to get Dict[str, ICUTable] instead of merged DataFrame
            # IMPORTANT: Must pass data_source for resolver.load_concepts to work
            # IMPORTANT: Use _skip_concept_cache=True to avoid polluting the main cache
            # This way, the internal wbc load won't affect subsequent wbc loads
            if data_source is None:
                if DEBUG_CALLBACK:
                    print(f"    [SKIP] data_source is None")
                if concept_name in frame.columns:
                    frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                return frame
            
            if DEBUG_CALLBACK:
                print(f"    加载 WBC (跳过缓存)...")
            
            wbc_result = resolver.load_concepts(
                ['wbc'],
                data_source,
                patient_ids=frame_patient_ids,  # Only load for needed patients
                ricu_compatible=False,
                merge=False,
                _skip_concept_cache=True,  # Don't cache this internal call
            )
            
            if 'wbc' not in wbc_result or wbc_result['wbc'].data.empty:
                if DEBUG_CALLBACK:
                    print(f"    [SKIP] WBC 为空或不存在")
                if concept_name in frame.columns:
                    frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                return frame
            
            wbc_df = wbc_result['wbc'].data.copy()
            if DEBUG_CALLBACK:
                print(f"    WBC loaded: {len(wbc_df)} rows, columns = {list(wbc_df.columns)}")
                print(f"    WBC样本:\n{wbc_df.head(10)}")
            
            # Find index column for merging (time column)
            index_col = source.index_var
            if not index_col:
                for possible_idx in ['measuredat', 'charttime', 'starttime', 'labresultoffset']:
                    if possible_idx in frame.columns:
                        index_col = possible_idx
                        break
            
            if DEBUG_CALLBACK:
                print(f"    index_col = {index_col}")
            
            # Prepare WBC for merge - rename value column
            wbc_val_col = wbc_result['wbc'].value_column or 'wbc'
            if DEBUG_CALLBACK:
                print(f"    wbc_val_col = {wbc_val_col}")
            if wbc_val_col != 'wbc' and wbc_val_col in wbc_df.columns:
                wbc_df = wbc_df.rename(columns={wbc_val_col: 'wbc'})
            
            # Ensure ID column exists in WBC data
            if id_col not in wbc_df.columns:
                if DEBUG_CALLBACK:
                    print(f"    [SKIP] id_col {id_col} not in wbc_df")
                if concept_name in frame.columns:
                    frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                return frame
            
            # Ensure numeric types
            frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            wbc_df['wbc'] = pd.to_numeric(wbc_df['wbc'], errors='coerce')
            
            # Ensure matching dtypes for merge columns (fix int32 vs int64 issue)
            if id_col in frame.columns and id_col in wbc_df.columns:
                wbc_df[id_col] = wbc_df[id_col].astype(frame[id_col].dtype)
            
            # For each row in frame, find the closest WBC measurement
            # This is a time-based merge (asof merge)
            if index_col and index_col in frame.columns and index_col in wbc_df.columns:
                # CRITICAL FIX: For AUMC, frame's measuredat is in MINUTES (raw from datasource),
                # but wbc_df's measuredat is in HOURS (after load_concepts processing).
                # We need to convert frame's time to HOURS before merge.
                # Detect AUMC by checking for large time values (>1000 typically means minutes)
                frame_time_max = frame[index_col].abs().max()
                wbc_time_max = wbc_df[index_col].abs().max() if not wbc_df.empty else 0
                
                # Create copies to avoid modifying original
                frame_work = frame.copy()
                wbc_work = wbc_df.copy()
                
                # If frame has much larger time values than wbc, convert frame from minutes to hours
                if frame_time_max > 1000 and wbc_time_max < 1000 and wbc_time_max > 0:
                    if DEBUG_CALLBACK:
                        print(f"    [TIME FIX] 检测到时间单位不匹配:")
                        print(f"      frame max time: {frame_time_max} (可能是分钟)")
                        print(f"      wbc max time: {wbc_time_max} (可能是小时)")
                        print(f"      -> 将 frame 时间从分钟转换为小时")
                    frame_work[index_col] = frame_work[index_col] / 60.0
                elif frame_time_max < 1000 and wbc_time_max > 1000:
                    # Opposite case: wbc is in minutes, frame is in hours
                    if DEBUG_CALLBACK:
                        print(f"    [TIME FIX] 检测到时间单位不匹配（反向）:")
                        print(f"      frame max time: {frame_time_max}")
                        print(f"      wbc max time: {wbc_time_max}")
                        print(f"      -> 将 wbc 时间从分钟转换为小时")
                    wbc_work[index_col] = wbc_work[index_col] / 60.0
                
                # Ensure matching dtypes for index column
                wbc_work[index_col] = wbc_work[index_col].astype(frame_work[index_col].dtype)
                
                # CRITICAL: merge_asof requires the 'on' column to be sorted globally.
                # With multiple patients, their time ranges may overlap. 
                # Solution: Process each patient separately and concat results.
                merged_parts = []
                for patient_id in frame_work[id_col].unique():
                    frame_patient = frame_work[frame_work[id_col] == patient_id].sort_values(index_col)
                    wbc_patient = wbc_work[wbc_work[id_col] == patient_id].sort_values(index_col)
                    
                    if wbc_patient.empty:
                        # No WBC data for this patient, keep original frame
                        merged_parts.append(frame_patient)
                        continue
                    
                    try:
                        merged_patient = pd.merge_asof(
                            frame_patient,
                            wbc_patient[[id_col, index_col, 'wbc']],
                            on=index_col,
                            by=id_col,
                            direction='nearest',
                        )
                        merged_parts.append(merged_patient)
                    except Exception as e:
                        if DEBUG_CALLBACK:
                            print(f"    [WARN] merge_asof failed for patient {patient_id}: {e}")
                        merged_parts.append(frame_patient)
                
                if merged_parts:
                    frame_merged = pd.concat(merged_parts, ignore_index=True)
                else:
                    frame_merged = frame_work.copy()
                
                if DEBUG_CALLBACK:
                    print(f"    Frame before merge:\n{frame_work[[id_col, index_col, concept_name]]}")
                    print(f"    After merge_asof:\n{frame_merged[[id_col, index_col, concept_name] + (['wbc'] if 'wbc' in frame_merged.columns else [])]}")
                
                # Calculate ratio: 100 * value / wbc
                if 'wbc' in frame_merged.columns:
                    valid_mask = (frame_merged['wbc'].notna()) & (frame_merged['wbc'] != 0)
                    if DEBUG_CALLBACK:
                        print(f"    valid_mask: {valid_mask.values}, sum={valid_mask.sum()}")
                    frame_merged.loc[valid_mask, concept_name] = (
                        100 * frame_merged.loc[valid_mask, concept_name] / 
                        frame_merged.loc[valid_mask, 'wbc']
                    )
                    if DEBUG_CALLBACK:
                        print(f"    计算后值: {frame_merged[concept_name].values}")
                    # Set unit to %
                    if unit_column and unit_column in frame_merged.columns:
                        frame_merged.loc[valid_mask, unit_column] = '%'
                    # Drop WBC column
                    frame_merged = frame_merged.drop(columns=['wbc'])
                else:
                    if DEBUG_CALLBACK:
                        print(f"    [WARNING] 'wbc' not in frame_merged.columns!")
                
                # CRITICAL: Convert time back to original format (minutes) for AUMC
                # The subsequent processing will apply the minutes->hours conversion again
                if frame_time_max > 1000 and wbc_time_max < 1000 and wbc_time_max > 0:
                    # We converted frame from minutes to hours, now convert back
                    frame_merged[index_col] = frame_merged[index_col] * 60.0
                    if DEBUG_CALLBACK:
                        print(f"    [TIME RESTORE] 将时间从小时转换回分钟")
                
                if DEBUG_CALLBACK:
                    print(f"    返回 frame_merged, shape={frame_merged.shape}")
                return frame_merged
            else:
                if DEBUG_CALLBACK:
                    print(f"    [FALLBACK] index_col 不在两个 frame 中, 使用平均 WBC")
                # No index column, use simple merge on ID (average WBC per patient)
                wbc_grouped = wbc_df.groupby(id_col)['wbc'].mean().reset_index()
                frame = frame.merge(wbc_grouped, on=id_col, how='left')
                
                valid_mask = (frame['wbc'].notna()) & (frame['wbc'] != 0)
                frame.loc[valid_mask, concept_name] = (
                    100 * frame.loc[valid_mask, concept_name] / 
                    frame.loc[valid_mask, 'wbc']
                )
                if unit_column and unit_column in frame.columns:
                    frame.loc[valid_mask, unit_column] = '%'
                frame = frame.drop(columns=['wbc'])
                
                return frame
                
        except Exception as e:
            if DEBUG_CALLBACK:
                print(f"    [EXCEPTION] {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            # On error, return frame as-is with numeric conversion
            if concept_name in frame.columns:
                frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            return frame

    raise NotImplementedError(
        f"Callback '{callback}' is not yet supported."
    )

def _apply_binary_op(symbol: str, series: pd.Series, value: object) -> pd.Series:
    """Apply binary operation or conversion function."""
    # Import conversion functions
    from .callback_utils import fahr_to_cels
    from .unit_conversion import celsius_to_fahrenheit, fahrenheit_to_celsius
    
    # Special case: set_val_na - set all values to NA
    if symbol == "set_val_na":
        return pd.Series([np.nan] * len(series), index=series.index)
    
    # Function map for unit conversions
    func_map = {
        "fahr_to_cels": fahr_to_cels,
        "fahrenheit_to_celsius": fahrenheit_to_celsius,
        "celsius_to_fahrenheit": celsius_to_fahrenheit,
    }
    
    # If it's a known function name, apply it
    if symbol in func_map:
        return func_map[symbol](series)
    
    # Otherwise treat as binary operator
    op_map = {
        "*": operator.mul,
        "/": operator.truediv,
        "+": operator.add,
        "-": operator.sub,
        "^": operator.pow,
    }

    if symbol not in op_map:
        raise NotImplementedError(f"Unsupported binary operator '{symbol}'")

    # Safe handling for division operations
    if symbol == "/":
        from .callback_utils import binary_op
        # Convert series to apply safe binary operation element-wise
        safe_op = binary_op(op_map[symbol], value)
        return series.apply(safe_op)
    else:
        try:
            return op_map[symbol](series, value)
        except (TypeError, ZeroDivisionError):
            return series  # Return original series on error

def _parse_binary_op(expr: str) -> tuple[str, object]:
    """Parse binary_op expression.
    
    Handles:
    - binary_op(`+`, 10)
    - fahr_to_cels (function name only)
    - set_val(NA) (special: set all values to NA)
    """
    # Check for set_val(NA) - special case for convert_unit
    if re.fullmatch(r'set_val\(NA\)', expr.strip(), re.IGNORECASE):
        return 'set_val_na', None
    
    # Check if it's just a function name (like fahr_to_cels)
    if re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', expr.strip()):
        # It's a function name - return it as a special operator
        return expr.strip(), None
    
    # Otherwise parse as binary_op(symbol, value)
    match = re.fullmatch(r"binary_op\(`(.+?)`,\s*(.+)\)", expr.strip(), flags=re.DOTALL)
    if not match:
        raise NotImplementedError(f"Unsupported binary_op expression '{expr}'")
    symbol = match.group(1)
    value = _parse_literal(match.group(2))
    return symbol, value

def _parse_mapping(body: str) -> Dict[object, object]:
    mapping: Dict[object, object] = {}
    for pair in _split_arguments(body):
        if "=" not in pair:
            continue
        key_text, value_text = pair.split("=", 1)
        key = _parse_literal(key_text.strip())
        value = _parse_literal(value_text.strip())
        mapping[key] = value
    return mapping

def _parse_r_arguments(expr: str) -> list:
    return [_parse_r_value(arg) for arg in _split_arguments(expr)]

def _parse_r_value(token: str):
    text = token.strip()
    if text.startswith("list(") and text.endswith(")"):
        inner = text[5:-1]
        return [_parse_r_value(arg) for arg in _split_arguments(inner)]
    if text.startswith("c(") and text.endswith(")"):
        inner = text[2:-1]
        return [_parse_r_value(arg) for arg in _split_arguments(inner)]
    return _parse_literal(text)

def _split_arguments(argument_str: str) -> List[str]:
    args: List[str] = []
    level = 0
    current: List[str] = []

    for char in argument_str:
        if char == "(":
            level += 1
        elif char == ")":
            level = max(level - 1, 0)
        elif char == "," and level == 0:
            arg = "".join(current).strip()
            if arg:
                args.append(arg)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        args.append(tail)

    return args

def _strip_quotes(token: str | None) -> Optional[str]:
    if token is None:
        return None
    text = token.strip()
    if text in {"NA", "NULL", ""}:
        return None
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        text = text[1:-1]
    return text.encode("utf8").decode("unicode_escape")

def _maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _default_aggregator_for_dtype(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "sum"
    if pd.api.types.is_numeric_dtype(dtype):
        return "median"
    return "first"

def _maybe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _maybe_timedelta(value: object) -> Optional[pd.Timedelta]:
    if value in (None, False, ""):
        return None
    if isinstance(value, pd.Timedelta):
        return value
    try:
        return pd.to_timedelta(value)
    except (TypeError, ValueError):
        return None

def _parse_literal(token: str):
    raw = token.strip()
    if raw in {"TRUE", "True"}:
        return True
    if raw in {"FALSE", "False"}:
        return False
    if raw in {"NA", "NA_real_", "NA_integer_", "NA_character_"}:
        return pd.NA
    if raw in {"NULL", "null"}:
        return None
    # 支持反引号（R语言中用于标识符）
    if raw.startswith("`") and raw.endswith("`"):
        # 去掉反引号，然后尝试解析为数字或返回字符串
        raw = raw[1:-1]
        try:
            # 优先尝试整数，如果失败再尝试浮点数
            if "." not in raw:
                return int(raw)
            return float(raw)
        except ValueError:
            return raw
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return _strip_quotes(raw)
    if raw.endswith("L"):
        raw = raw[:-1]
    try:
        # 优先尝试整数，如果失败再尝试浮点数
        if "." not in raw:
            return int(raw)
        return float(raw)
    except ValueError:
        return raw

# 别名 - 为了兼容性
Concept = ConceptDefinition  # Concept 类别名，指向 ConceptDefinition

def load_dictionary(src_name: Optional[str] = None, include_sofa2: bool = False) -> ConceptDictionary:
    """
    加载概念字典 - 兼容函数
    
    Args:
        src_name: 数据源名称（可选）
        include_sofa2: 是否包含 SOFA-2 概念字典
        
    Returns:
        ConceptDictionary 实例
    """
    from .resources import load_dictionary as _load_dictionary

    # 当前实现不根据数据源过滤概念，但保留参数以兼容既有调用
    return _load_dictionary(include_sofa2=include_sofa2)
