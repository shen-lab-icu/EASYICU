"""Concept dictionary utilities inspired by ricu."""

from __future__ import annotations

import copy
import functools
import json
import logging
import re
import operator
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
from threading import RLock, local as thread_local
from typing import Dict, FrozenSet, Iterable, List, Mapping, MutableMapping, Optional, Union

import numpy as np
import pandas as pd

from .config import DataSourceConfig
from .datasource import FilterOp, FilterSpec, ICUDataSource, _duckdb_path
from .table import ICUTable, WinTbl
from .concept_callbacks import ConceptCallbackContext, execute_concept_callback
from . import compat

logger = logging.getLogger(__name__)

# 全局调试开关 - 设置为 False 可以减少输出
DEBUG_MODE = False

# 避免在分批/分块加载时重复打印同一条“数据库未配置数据源”提示
_MISSING_SOURCE_WARNED: set[tuple[str, str]] = set()

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
        "sic": ["CaseID"],  # 🔧 FIX 2025-01-31: Use CaseID (uppercase) to match actual SICdb data
        "miiv": ["stay_id"],
        "mimic_demo": ["stay_id"],
        "mimic": ["icustay_id"],  # 🔧 FIX 2026-02-06: MIMIC-III uses icustay_id
    }

    # Check explicit mapping first (mimic → icustay_id, miiv → stay_id)
    if db in mapping:
        return mapping[db]
    # Fallback for mimic variants (e.g. mimic_demo if not in mapping)
    if db.startswith("mimic"):
        return ["stay_id"]
    return mapping.get(db, ["stay_id"])


def _normalize_patient_ids_for_cache(patient_ids: Optional[Iterable[object]]) -> Optional[object]:
    """Normalize patient identifiers for stable cache keys."""
    if patient_ids is None:
        return None

    if isinstance(patient_ids, Mapping):
        normalized: Dict[str, object] = {}
        for key in sorted(patient_ids):
            value = patient_ids[key]
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                normalized[str(key)] = sorted(list(value))
            else:
                normalized[str(key)] = value
        return normalized

    if isinstance(patient_ids, (set, frozenset)):
        return sorted(list(patient_ids))

    if isinstance(patient_ids, Iterable) and not isinstance(patient_ids, (str, bytes)):
        return sorted(list(patient_ids))

    return patient_ids


def _is_patient_id_filter_column(column: object, effective_id_var: Optional[str] = None) -> bool:
    """Return whether a filter column represents patient/stay identifiers."""
    if column is None:
        return False

    column_str = str(column)
    if effective_id_var and column_str == effective_id_var:
        return True

    return column_str.lower() in {
        "subject_id",
        "stay_id",
        "hadm_id",
        "icustay_id",
        "patientunitstayid",
        "admissionid",
        "patientid",
        "caseid",
        "case_id",
    }

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


# 从 utils 导入统一的 hash 函数，避免循环导入
from .utils import compute_patient_ids_hash as _compute_patient_ids_hash


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
        # 🆕 原始数据缓存：回调函数和重复聚合重建都可复用
        # Key: (concept_name, patient_ids_hash, agg_token)
        # Legacy key: (concept_name, patient_ids_hash)
        self._raw_concept_cache: Dict[tuple, ICUTable] = {}
        # 🚀 预聚合缓存：存储 change_interval 之前的原始数据
        # Key: (concept_name, patient_ids_hash)
        # Value: dict with temp_table, interval, fill_missing, fill_method, skip_ci
        self._pre_agg_cache: Dict[tuple, dict] = {}
        # 多线程支持：使用线程局部存储避免循环依赖误报
        self._thread_local = thread_local()
        # 🔧 嵌套调用深度跟踪：防止递归概念的内部调用清除缓存
        self._load_depth = 0
        # ⚡ PERF: 跨模块缓存复用模式 — 保留 raw/table 缓存在 top-level 调用间
        self._keep_cache_between_calls = False
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

    def clear_table_cache(self, keep_concept_cache: bool = False) -> None:
        """Clear cached source tables.
        
        Args:
            keep_concept_cache: If True, only clear table cache but preserve concept data cache.
                              This is useful for batch loading multiple concepts where shared
                              sub-concepts can benefit from caching.
        """
        with self._cache_lock:
            self._table_cache.clear()
            self._concept_cache.clear()
            if not keep_concept_cache:
                self._concept_data_cache.clear()  # 🚀 清除概念数据缓存
                self._raw_concept_cache.clear()   # 🆕 清除原始数据缓存
                self._pre_agg_cache.clear()        # 🆕 清除预聚合缓存
            # 清除 rgx_itm DISTINCT 缓存
            if hasattr(self, '_rgx_distinct_cache'):
                self._rgx_distinct_cache.clear()
            # 清除当前线程的inflight集合
            if hasattr(self._thread_local, 'inflight'):
                self._thread_local.inflight.clear()

    def clear(self) -> None:
        """Alias for clear_table_cache, used by CacheManager."""
        self.clear_table_cache(keep_concept_cache=False)

    @staticmethod
    def _downcast_float64_to_float32(df: pd.DataFrame) -> pd.DataFrame:
        """Downcast float64 value columns to float32 to save ~50% memory.
        
        Skips ID columns and time columns which need full precision.
        Idempotent: float32 columns are unchanged.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        _skip = {'charttime', 'measuredat_minutes',
                 'stay_id', 'patientunitstayid', 'admissionid',
                 'patientid', 'icustay_id', 'CaseID'}
        for col in df.columns:
            if col not in _skip and df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)
        return df

    def get_raw_concept(
        self,
        concept_name: str,
        data_source: 'ICUDataSource',
        patient_ids: Optional[Union[FrozenSet, Dict, List]] = None,
    ) -> Optional[ICUTable]:
        """获取概念的原始数据（不带时间对齐），使用缓存避免重复加载。
        
        这个方法用于回调函数中需要获取原始时间的场景。
        """
        # 🔧 使用统一的 hash 函数
        patient_ids_hash = _compute_patient_ids_hash(patient_ids)

        with self._cache_lock:
            cached = self._get_raw_concept_from_cache(
                concept_name,
                patient_ids_hash,
                aggregator=None,
                allow_aggregated=False,
            )
            if cached is not None:
                return cached.copy() if hasattr(cached, 'copy') else cached
        
        # 加载原始数据（interval=None）
        try:
            loaded = self.load_concepts(
                [concept_name],
                data_source,
                merge=False,
                aggregate=None,
                interval=None,
                patient_ids=patient_ids,
                align_to_admission=True,
            )
            result = loaded.get(concept_name) if isinstance(loaded, dict) else loaded
            
            # 缓存原始数据
            # 🔧 FIX 2026-03-10: Also cache WinTbl/IdTbl/TsTbl (they have .data but aren't ICUTable)
            if isinstance(result, ICUTable) or hasattr(result, 'data'):
                with self._cache_lock:
                    self._store_raw_concept_cache(
                        concept_name,
                        patient_ids_hash,
                        result,
                        aggregator=None,
                        store_legacy=True,
                    )
                return result.copy() if hasattr(result, 'copy') else result
        except Exception:
            pass
        return None

    @staticmethod
    def _normalize_raw_cache_aggregator(aggregator: object) -> str:
        """Normalize aggregator value for raw cache keys."""
        if aggregator in (None, False):
            return "__raw__"
        return str(aggregator)

    def _raw_cache_key(
        self,
        concept_name: str,
        patient_ids_hash: object,
        aggregator: object,
    ) -> tuple:
        return (
            concept_name,
            patient_ids_hash,
            self._normalize_raw_cache_aggregator(aggregator),
        )

    def _get_raw_concept_from_cache(
        self,
        concept_name: str,
        patient_ids_hash: object,
        aggregator: object,
        *,
        allow_aggregated: bool,
    ):
        """Lookup raw cache entries with backward-compatible fallback order."""
        keys = [self._raw_cache_key(concept_name, patient_ids_hash, aggregator)]
        if allow_aggregated:
            keys.extend(
                [
                    self._raw_cache_key(concept_name, patient_ids_hash, "auto"),
                    self._raw_cache_key(concept_name, patient_ids_hash, None),
                ]
            )
            keys.append((concept_name, patient_ids_hash))
        # 🔧 FIX: allow_aggregated=False 时不退回到 None key 和 legacy 2-tuple
        # 否则特定聚合器请求会命中默认聚合的缓存

        seen = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            cached = self._raw_concept_cache.get(key)
            if cached is not None:
                return cached
        return None

    def _store_raw_concept_cache(
        self,
        concept_name: str,
        patient_ids_hash: object,
        table: object,
        *,
        aggregator: object,
        store_legacy: bool = False,
    ) -> None:
        """Store raw cache entries without duplicating table objects."""
        self._raw_concept_cache[self._raw_cache_key(concept_name, patient_ids_hash, aggregator)] = table
        if store_legacy:
            self._raw_concept_cache[(concept_name, patient_ids_hash)] = table

    def _get_inflight(self) -> set:
        """获取当前线程的inflight集合（线程安全）"""
        if not hasattr(self._thread_local, 'inflight'):
            self._thread_local.inflight = set()
        return self._thread_local.inflight

    def _should_fill_gaps(self, concept_name: str, definition: ConceptDefinition) -> bool:
        """Determine if fill_gaps should be applied.
        
        🔧 CRITICAL FIX 2025-02-14: R ricu's load_concepts does NOT call fill_gaps!
        fill_gaps is only called in specific callbacks (e.g., combine_callbacks, sofa).
        This function should almost always return False for normal concept loading.
        """
        # 🔧 R ricu behavior: fill_gaps is NOT part of load_concepts.
        # EasyICU was incorrectly applying fill_gaps during load, causing row inflation.
        # Example: crea had 157 rows in R ricu but 2150 rows in EasyICU (filled to hourly grid).
        # Solution: Always return False for normal load_concepts.
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
        # This prevents false coverage (easyicu 100% vs ricu 2.74% for urine)
        
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
            _debug('  🔍 _expand_patient_ids 被调用')
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
            elif db_name in ['mimic']:
                # 🔧 FIX 2026-02-08: MIMIC-III 使用 icustay_id（不是 stay_id）
                patient_ids = {'icustay_id': list(patient_ids)}
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
        # 支持的转换：stay_id <-> subject_id 或 icustay_id <-> subject_id
        db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
        
        # 🔧 FIX 2026-02-08: MIMIC-III 使用 icustay_id
        stay_id_col = 'icustay_id' if db_name == 'mimic' else 'stay_id'
        
        if target_id_var == 'subject_id' and stay_id_col in patient_ids:
            # 需要从 stay_id/icustay_id 获取 subject_id
            source_var = stay_id_col
            source_values = patient_ids[stay_id_col]
        elif target_id_var == stay_id_col and 'subject_id' in patient_ids:
            # 需要从 subject_id 获取 stay_id/icustay_id
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
                # 🔧 FIX 2026-02-08: MIMIC-III 使用 icustay_id
                stay_id_col = 'icustay_id' if db_name == 'mimic' else 'stay_id'
                
                # 加载 icustays 表（只需要 stay_id/icustay_id 和 subject_id）
                filters = [
                    FilterSpec(
                        column=source_var,
                        op=FilterOp.IN,
                        value=source_values,
                    )
                ]
                icustays_table = data_source.load_table(
                    'icustays', 
                    columns=[stay_id_col, 'subject_id'],
                    filters=filters,
                    verbose=False
                )
                if hasattr(icustays_table, 'data'):
                    self._id_mapping_cache = icustays_table.data[[stay_id_col, 'subject_id']].drop_duplicates()
                else:
                    self._id_mapping_cache = icustays_table[[stay_id_col, 'subject_id']].drop_duplicates()
                    
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
        r_compatible: bool = True,  # 默认启用ricu.R兼容格式
        concept_workers: int = -1,  # 🔧 -1 表示自动检测
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

        # 🚀 智能并行策略：根据系统资源自动配置并行度
        # -1 表示自动检测，0 表示禁用并行，>0 表示指定的并行数
        if concept_workers == -1:
            # 自动检测系统资源
            try:
                from .parallel_config import get_global_config
                parallel_config = get_global_config()
                # 对于多概念加载，使用自动检测的并行数
                if total > 1:
                    concept_workers = parallel_config.max_workers
                    if verbose and not _batch_loading:
                        logger.info(
                            f"🔧 自动并行配置: {parallel_config.performance_tier} "
                            f"(内存: {parallel_config.total_memory_gb:.1f}GB, "
                            f"workers: {concept_workers})"
                        )
                else:
                    concept_workers = 1
            except Exception:
                concept_workers = 1  # 回退到单线程
        
        effective_workers = concept_workers
        
        # 分析每个概念使用的主表和value_var
        table_to_concepts = {}  # {table_name: [(concept_name, value_var), ...]}
        # 🔧 FIX: 跟踪有多个数据源的概念，这些概念不应使用宽表批量加载
        multi_source_concepts = set()
        src_name = data_source.config.name if hasattr(data_source, 'config') else 'miiv'
        for name in names:
            concept = self.dictionary.get(name)
            if concept and hasattr(concept, 'sources') and concept.sources:
                src_list = concept.sources.get(src_name, [])
                if src_list:
                    # 🔧 FIX: 检查是否有多个数据源（如eICU的map有vitalperiodic和vitalaperiodic）
                    if isinstance(src_list, list) and len(src_list) > 1:
                        multi_source_concepts.add(name)
                    # 获取第一个source的表名和value_var
                    first_src = src_list[0] if isinstance(src_list, list) else src_list
                    table_name = getattr(first_src, 'table', None)
                    value_var = getattr(first_src, 'value_var', None)
                    if table_name:
                        if table_name not in table_to_concepts:
                            table_to_concepts[table_name] = []
                        table_to_concepts[table_name].append((name, value_var))
        
        # 🚀 批量加载优化：检测是否可以使用DuckDB批量加载
        # 宽表如vitalperiodic有多个值列，可以一次性加载并聚合
        WIDE_TABLES = {'vitalperiodic', 'vitalaperiodic'}
        wide_table_batch_results = {}  # {concept_name: DataFrame}
        wide_table_merged_df = None  # 🚀 保存批量加载的合并结果，避免重复合并
        wide_table_covered_names = set()
        wide_table_frames: List[pd.DataFrame] = []
        
        # 🔧 FIX: 只有当没有多数据源概念时才使用批量加载
        # 因为批量加载只处理一个表，不支持多表合并（如eICU的map需要合并vitalperiodic和vitalaperiodic）
        # 🚀 长表多概念批量加载支持的表（有 _bucket 目录和 sub_var 过滤）
        _MULTI_BATCH_TABLES = {
            'chartevents',       # MIIV/MIMIC
            'data_float_h',      # SIC vitals
            'laboratory',        # SIC labs
            'labevents',         # MIIV/MIMIC labs
            'lab',               # eICU labs
        }
        for shared_table, concepts_info in table_to_concepts.items():
            table_batch_results = {}
            table_batch_df = None
            table_covered_names = set()

            def _detect_batch_id_col(frame: pd.DataFrame, fallback: str) -> str:
                for cand in [fallback, 'stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID', 'subject_id']:
                    if cand in frame.columns:
                        return cand
                return frame.columns[0]

            # 检查是否为宽表且所有概念都有value_var
            # 🚀 优化：对于单个概念也使用批量加载，因为DuckDB聚合比pandas快10倍
            # 🔧 FIX: 排除使用非默认聚合的概念（如sofa_cardio的MAP需要min而非median）
            # 这些概念需要精确控制聚合函数，必须走正常加载路径
            concepts_info_filtered = [
                (name, val_var) for name, val_var in concepts_info
                if aggregators.get(name) in (None, False, 'auto', 'median')
                and name not in multi_source_concepts
            ]
            # ⚡ PERF: 跳过已在 _raw_concept_cache 中的概念，避免重复读取宽表
            if self._keep_cache_between_calls and concepts_info_filtered:
                _pid_hash = _compute_patient_ids_hash(patient_ids)
                concepts_info_filtered = [
                    (name, val_var) for name, val_var in concepts_info_filtered
                    if self._get_raw_concept_from_cache(
                        name, _pid_hash,
                        aggregator=aggregators.get(name, "auto"),
                        allow_aggregated=True,
                    ) is None
                ]
            if shared_table in WIDE_TABLES and len(concepts_info_filtered) >= 1:
                all_have_value_var = all(val_var is not None for _, val_var in concepts_info_filtered)
                
                if all_have_value_var:
                    try:
                        from .datasource import load_wide_table_aggregated
                        
                        # 获取所有需要的value_var列（去重，避免DuckDB重复加载同一列）
                        value_columns = list(dict.fromkeys(val_var for _, val_var in concepts_info_filtered))
                        
                        # 计算interval_hours
                        interval_hours = 1.0
                        if interval is not None:
                            if hasattr(interval, 'total_seconds'):
                                interval_hours = interval.total_seconds() / 3600.0
                            elif isinstance(interval, (int, float)):
                                interval_hours = float(interval)
                        
                        # 🔧 获取patient_ids列表 (关键: 处理dict和list两种格式)
                        # patient_ids 可能是 dict 格式 {'patientunitstayid': [1,2,3]} 或 list 格式 [1,2,3]
                        if patient_ids is None:
                            patient_ids_list = None
                        elif isinstance(patient_ids, dict):
                            # dict格式：取第一个values
                            patient_ids_list = list(next(iter(patient_ids.values())))
                        else:
                            patient_ids_list = list(patient_ids)
                        
                        # 🔧 构建 per-column filter bounds（与 R ricu 的 pre-aggregation filter_bounds 一致）
                        column_bounds = {}
                        for concept_name, val_var in concepts_info_filtered:
                            concept = self.dictionary.get(concept_name)
                            if concept:
                                vmin = concept.minimum
                                vmax = concept.maximum
                                if vmin is not None or vmax is not None:
                                    column_bounds[val_var] = (vmin, vmax)
                        
                        if verbose:
                            logger.info(f"🚀 宽表批量加载: {shared_table} ({len(value_columns)} 列)")
                        
                        # 一次性加载所有列 (使用median与R ricu保持一致)
                        batch_df = load_wide_table_aggregated(
                            data_source,
                            shared_table,
                            value_columns,
                            interval_hours=interval_hours,
                            patient_ids=patient_ids_list,
                            agg_func='median',
                            column_bounds=column_bounds,
                        )
                        
                        # 确定ID列和时间列
                        table_cfg = data_source.config.get_table(shared_table)
                        id_col = table_cfg.defaults.id_var
                        if not id_col:
                            icustay_cfg = data_source.config.id_configs.get('icustay')
                            id_col = icustay_cfg.id if icustay_cfg else 'patientunitstayid'
                        id_col = _detect_batch_id_col(batch_df, id_col)
                        
                        # 🚀 处理多概念共享同一 val_var 的情况（如 spo2 和 o2sat 都用 sao2）
                        # 不能用简单 rename — dict 键冲突会导致 last-write-wins
                        # 策略：对共享 val_var 的概念，复制列
                        _val_to_concepts = {}
                        for concept_name, val_var in concepts_info_filtered:
                            _val_to_concepts.setdefault(val_var, []).append(concept_name)
                        
                        for val_var, concept_names in _val_to_concepts.items():
                            if val_var in batch_df.columns:
                                for cname in concept_names:
                                    if cname != val_var:
                                        batch_df[cname] = batch_df[val_var]
                        # 删除不再需要的原始 val_var 列（避免列名混乱）
                        _orig_val_vars = set(_val_to_concepts.keys())
                        _concept_names = {cn for cns in _val_to_concepts.values() for cn in cns}
                        for vv in _orig_val_vars:
                            if vv not in _concept_names and vv in batch_df.columns:
                                batch_df = batch_df.drop(columns=[vv])
                        
                        # 🚀 保存合并后的DataFrame
                        table_batch_df = batch_df
                        table_covered_names = _concept_names
                        
                        # 🚀 将每个概念放入 batch_results，使 _resolve 能跳过 _ensure_concept_loaded
                        for concept_name, val_var in concepts_info_filtered:
                            concept_df = batch_df[[id_col, 'charttime', concept_name]].copy()
                            concept_df = concept_df.dropna(subset=[concept_name])
                            _tbl = ICUTable(
                                data=concept_df, 
                                id_columns=[id_col],
                                index_column='charttime',
                                value_column=concept_name
                            )
                            _tbl._pre_aggregated = True
                            table_batch_results[concept_name] = _tbl
                        
                        if verbose:
                            logger.info(f"✅ 宽表批量加载完成，加载了 {len(concepts_info_filtered)} 个概念")
                        
                    except Exception as e:
                        logger.warning(f"宽表批量加载失败，回退到普通加载: {e}")
                        table_batch_results = {}
                        table_batch_df = None
                        table_covered_names = set()

            # 🚀 长表多概念批量加载：同一 itemid/sub_var 表的一次扫描多列聚合
            elif shared_table in _MULTI_BATCH_TABLES and len(concepts_info) >= 2:
                try:
                    from .datasource import load_bucketed_table_multi_aggregated

                    # ⚡ PERF: 跳过已在缓存中的概念，避免重复读取长表
                    _concepts_for_batch = list(concepts_info)
                    if self._keep_cache_between_calls:
                        _pid_hash = _compute_patient_ids_hash(patient_ids)
                        _concepts_for_batch = [
                            (name, val_var) for name, val_var in _concepts_for_batch
                            if self._get_raw_concept_from_cache(
                                name, _pid_hash,
                                aggregator=aggregators.get(name, "auto"),
                                allow_aggregated=True,
                            ) is None
                        ]

                    batch_itemids = {}
                    batch_bounds = {}
                    batch_sub_var = None
                    for concept_name, _ in _concepts_for_batch:
                        if concept_name in multi_source_concepts:
                            continue
                        concept = self.dictionary.get(concept_name)
                        if not concept or not getattr(concept, 'sources', None):
                            continue
                        src_list = concept.sources.get(src_name, [])
                        if not src_list or len(src_list) != 1:
                            continue
                        src = src_list[0]
                        if getattr(src, 'callback', None):
                            continue
                        _sub_var = getattr(src, 'sub_var', None)
                        if not _sub_var:
                            continue
                        ids = getattr(src, 'ids', None)
                        if ids is None:
                            continue
                        # Track sub_var (must be consistent within a table)
                        if batch_sub_var is None:
                            batch_sub_var = _sub_var
                        elif _sub_var != batch_sub_var:
                            continue
                        ids_list = ids if isinstance(ids, list) else [ids]
                        # Handle string IDs (eICU labname) vs int IDs
                        if isinstance(ids_list[0], str):
                            batch_itemids[concept_name] = [str(x) for x in ids_list]
                        else:
                            batch_itemids[concept_name] = [int(x) for x in ids_list]
                        batch_bounds[concept_name] = {
                            "min": _get_concept_bounds(concept_name, "min"),
                            "max": _get_concept_bounds(concept_name, "max"),
                        }

                    if len(batch_itemids) >= 2:
                        interval_hours = 1.0
                        if interval is not None:
                            if hasattr(interval, 'total_seconds'):
                                interval_hours = interval.total_seconds() / 3600.0
                            elif isinstance(interval, (int, float)):
                                interval_hours = float(interval)

                        if patient_ids is None:
                            patient_ids_list = None
                        elif isinstance(patient_ids, dict):
                            patient_ids_list = list(next(iter(patient_ids.values())))
                        else:
                            patient_ids_list = list(patient_ids)

                        # Determine value_column from table config
                        table_cfg = data_source.config.get_table(shared_table)
                        _val_col = getattr(table_cfg.defaults, 'val_var', None) or 'valuenum'
                        _time_col = getattr(table_cfg.defaults, 'index_var', None)

                        batch_df = load_bucketed_table_multi_aggregated(
                            data_source,
                            shared_table,
                            batch_itemids,
                            value_column=_val_col,
                            interval_minutes=interval_hours * 60.0,
                            patient_ids=patient_ids_list,
                            agg_func='median',
                            itemid_col=batch_sub_var,
                            concept_bounds=batch_bounds,
                            time_col=_time_col,
                        )

                        id_col = _detect_batch_id_col(batch_df, 'stay_id')

                        # Detect the time column in batch output
                        # AUMC outputs measuredat_minutes, others output charttime
                        _time_col_out = 'charttime'
                        if 'measuredat_minutes' in batch_df.columns:
                            _time_col_out = 'measuredat_minutes'

                        covered_names = set()
                        for concept_name in batch_itemids:
                            if concept_name not in batch_df.columns:
                                continue
                            concept_df = batch_df[[id_col, _time_col_out, concept_name]].copy()
                            concept_df = concept_df.dropna(subset=[concept_name])
                            _tbl = ICUTable(
                                data=concept_df,
                                id_columns=[id_col],
                                index_column=_time_col_out,
                                value_column=concept_name,
                            )
                            _tbl._pre_aggregated = True
                            table_batch_results[concept_name] = _tbl
                            covered_names.add(concept_name)

                        if covered_names:
                            keep_cols = [id_col, _time_col_out] + [name for name in names if name in covered_names and name in batch_df.columns]
                            table_batch_df = batch_df[keep_cols].copy()
                            table_covered_names = set(covered_names)

                        if verbose and covered_names:
                            logger.info(
                                "✅ 长表批量加载完成: %s (%d/%d 概念)",
                                shared_table,
                                len(covered_names),
                                len(names),
                            )
                except Exception as e:
                    logger.warning(f"长表批量加载失败，回退到普通加载: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())

            if table_batch_results:
                wide_table_batch_results.update(table_batch_results)
            if table_batch_df is not None and not table_batch_df.empty:
                wide_table_frames.append(table_batch_df)
                wide_table_covered_names.update(table_covered_names)

        if wide_table_frames:
            wide_table_merged_df = wide_table_frames[0]
            for frame in wide_table_frames[1:]:
                # Find common time column (charttime or measuredat_minutes)
                _time_merge = 'charttime'
                for candidate in ('charttime', 'measuredat_minutes'):
                    if candidate in wide_table_merged_df.columns and candidate in frame.columns:
                        _time_merge = candidate
                        break
                merge_keys = [wide_table_merged_df.columns[0], _time_merge]
                wide_table_merged_df = wide_table_merged_df.merge(frame, on=merge_keys, how='outer', sort=False)

        # 如果没有使用批量加载，使用串行加载
        if not wide_table_batch_results and len(table_to_concepts) == 1 and concept_workers > 1 and total > 1:
            shared_table = list(table_to_concepts.keys())[0]
            if verbose:
                logger.info(f"🔄 所有 {total} 个概念共享表 '{shared_table}'，使用串行加载以共享缓存")
            effective_workers = 1
        
        # 🚀 新优化：检测共享子概念，强制串行加载以利用缓存
        # 例如 uo_6h, uo_12h, uo_24h 都有子概念 urine, weight
        # 如果并行加载，每个线程都会各自加载 urine/weight，浪费时间
        if effective_workers > 1 and total > 1:
            # 收集所有概念的子概念（递归展开）
            all_sub_concepts = {}  # {sub_concept: [parent_concepts]}
            
            def collect_sub_concepts(concept_name: str, visited: set) -> set:
                """递归收集概念的所有子概念"""
                if concept_name in visited:
                    return set()
                visited.add(concept_name)
                result = set()
                concept = self.dictionary.get(concept_name)
                if concept and concept.sub_concepts:
                    for sub in concept.sub_concepts:
                        result.add(sub)
                        # 递归收集子概念的子概念
                        result.update(collect_sub_concepts(sub, visited))
                return result
            
            for name in names:
                visited = set()  # 🔧 FIX: 空集，让函数自己添加避免循环
                subs = collect_sub_concepts(name, visited)
                for sub in subs:
                    if sub not in all_sub_concepts:
                        all_sub_concepts[sub] = []
                    all_sub_concepts[sub].append(name)
            
            # 检查是否有共享的子概念（被多个概念引用）
            shared_sub_concepts = [sub for sub, parents in all_sub_concepts.items() if len(parents) > 1]
            if shared_sub_concepts:
                if verbose:
                    logger.info(f"🔄 检测到共享子概念 {shared_sub_concepts}，使用串行加载以利用缓存")
                effective_workers = 1

        def _resolve(name: str, position: int) -> tuple[str, ICUTable]:
            # 🚀 如果已经通过批量加载获取了数据，直接返回
            if name in wide_table_batch_results:
                concept_table = wide_table_batch_results[name]
                if verbose and logger.isEnabledFor(logging.INFO):
                    row_count = len(concept_table.data) if (isinstance(concept_table, ICUTable) or hasattr(concept_table, 'data')) else len(concept_table)
                    logger.info("✅  概念 '%s' (批量) 已加载 (行数: %s)", name, row_count)
                return name, concept_table
            
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
                if isinstance(concept_table, ICUTable) or hasattr(concept_table, 'data'):
                    row_count = len(concept_table.data)
                elif isinstance(concept_table, pd.DataFrame):
                    row_count = len(concept_table)
                else:
                    row_count = "N/A"
                logger.info("✅  概念 '%s' 已加载 (行数: %s)", name, row_count)
            return name, concept_table

        try:
            results: Dict[str, ICUTable] = {}
            if effective_workers > 1 and total > 1:
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
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
                # 如果是r_compatible模式且只有一个概念，返回ricu.R格式的DataFrame
                if r_compatible and len(tables) == 1:
                    concept_name = list(tables.keys())[0]
                    logger.debug("调试：调用_to_r_format处理概念 %s", concept_name)
                    # 计算interval_hours
                    interval_hours = 1.0
                    if interval is not None:
                        if hasattr(interval, 'total_seconds'):
                            interval_hours = interval.total_seconds() / 3600.0
                        elif isinstance(interval, (int, float)):
                            interval_hours = float(interval)
                    return self._to_r_format(tables[concept_name], concept_name, interval_hours=interval_hours)
                return tables

            # 🚀 如果有宽表批量加载的合并结果，直接使用（避免重复合并）
            if wide_table_merged_df is not None and not r_compatible:
                if verbose:
                    logger.info("🚀 使用宽表批量加载的合并结果，跳过合并步骤")
                return self._downcast_float64_to_float32(wide_table_merged_df)

            # 如果是r_compatible模式，使用增强的ricu风格合并
            if r_compatible:
                # 🔧 FIX 2025-02-13: For single win_tbl target concepts, return original format
                # R ricu returns win_tbl format directly without merging/aggregation
                if len(names) == 1:
                    concept_name = names[0]
                    definition = self.dictionary.get(concept_name)
                    if definition and getattr(definition, 'target', 'ts_tbl') == 'win_tbl':
                        if concept_name in tables:
                            table = tables[concept_name]
                            # Handle different table types: ICUTable, WinTbl, DataFrame
                            df = None
                            if hasattr(table, 'data'):
                                df = table.data.copy()
                            elif isinstance(table, pd.DataFrame):
                                df = table.copy()
                            
                            if df is not None and not df.empty:
                                # Ensure value column is named after concept
                                if concept_name not in df.columns:
                                    for cand in ['value', 'valuenum', 'rate']:
                                        if cand in df.columns:
                                            df = df.rename(columns={cand: concept_name})
                                            break
                                # Determine ID column for this database
                                db_name = data_source.config.name if data_source and hasattr(data_source, 'config') else 'miiv'
                                id_col_map = {
                                    'miiv': 'stay_id', 'eicu': 'patientunitstayid',
                                    'aumc': 'admissionid', 'hirid': 'patientid',
                                    'mimic': 'icustay_id', 'sic': 'CaseID'
                                }
                                target_id_col = id_col_map.get(db_name, 'stay_id')
                                result_cols = []
                                for cand in [target_id_col, 'stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']:
                                    if cand in df.columns:
                                        result_cols.append(cand)
                                        break
                                time_col = None
                                for cand in ['nursingchartoffset', 'nursingchartentryoffset', 'observationoffset',
                                             'charttime', 'measuredat', 'datetime', 'givenat', 'starttime', 'start']:
                                    if cand in df.columns:
                                        time_col = cand
                                        break
                                if time_col:
                                    result_cols.append(time_col)
                                if 'dur_var' in df.columns:
                                    result_cols.append('dur_var')
                                elif 'dex_dur' in df.columns:
                                    df = df.rename(columns={'dex_dur': 'dur_var'})
                                    result_cols.append('dur_var')
                                if concept_name in df.columns:
                                    result_cols.append(concept_name)
                                df = df[[c for c in result_cols if c in df.columns]]
                                if verbose:
                                    logger.info(f"🔧 win_tbl 概念 '{concept_name}' 直接返回原始格式: {len(df)} 行, 列={df.columns.tolist()}")
                                return df
                
                # 🚀 宽表优化：如果有批量加载的合并结果，直接用它
                if wide_table_merged_df is not None:
                    _all_covered = wide_table_covered_names >= set(names)
                    if not _all_covered and wide_table_covered_names:
                        merged_partial = self._merge_partial_wide_result(
                            wide_table_merged_df,
                            tables,
                            names,
                            wide_table_covered_names,
                            data_source=data_source,
                        )
                        if merged_partial is not None:
                            if verbose:
                                logger.info("🚀 使用部分宽表批量结果并仅合并剩余概念")
                            return self._downcast_float64_to_float32(merged_partial)
                        # Partial merge failed (e.g. non-charttime time column) — fall through to full merge
                    elif _all_covered:
                        if verbose:
                            logger.info("🚀 使用宽表批量加载的合并结果 (ricu格式)")
                        _sort_time = 'charttime'
                        for _tc in ('charttime', 'measuredat_minutes'):
                            if _tc in wide_table_merged_df.columns:
                                _sort_time = _tc
                                break
                        return self._downcast_float64_to_float32(wide_table_merged_df.reset_index(drop=True))
                return self._to_r_format_merged_enhanced(tables, names, interval, data_source=data_source)

            merged = self._merge_tables(tables)
            return merged
        finally:
            # 🔧 嵌套调用深度跟踪：减少深度计数器
            self._load_depth -= 1
            # 🔧 只有顶层调用才清除概念缓存和inflight，避免递归概念内部调用清除外层所需的缓存
            if is_top_level:
                with self._cache_lock:
                    self._concept_cache.clear()
                    # 🔧 _concept_data_cache 始终清除：在全患者规模(200K+)下
                    # 保留会导致 ~30GB 内存占用，引发严重的 swap 性能退化
                    self._concept_data_cache.clear()
                    if not self._keep_cache_between_calls:
                        # 🔧 FIX: 顶层调用结束后清除内存缓存，避免碎片内存泄漏
                        # 磁盘缓存（_store_in_disk_cache）已负责跨调用复用
                        self._raw_concept_cache.clear()
                        self._table_cache.clear()
                    # 清除当前线程的inflight集合
                    self._get_inflight().clear()
                
                # 🔧 释放碎片内存
                from .memory_manager import release_memory
                release_memory()

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
        
        # 🔧 FIX 2025-02-14: vent_ind 是特殊的 rec_cncpt - 它有 callback 且需要正确的 win_tbl 格式

        # 🔧 FIX 2025-02-14: vent_ind 是特殊的 rec_cncpt - 它有 callback 且需要正确的 win_tbl 格式
        # 其他有 callback 的概念（如 gcs, sofa_*）可能有直接的数据源，应该优先使用直接加载
        use_recursive = False
        has_direct_source = False  # 标记是否有直接的表 source
        # 只有 vent_ind 必须使用 callback（R ricu 的行为）
        must_use_callback = definition.callback is not None and concept_name == 'vent_ind'
        if definition.sub_concepts and not must_use_callback:
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
                # 🔧 FIX 2025-02-14: Load sub-concepts for callbacks that need them
                tables = {}
                if definition.sub_concepts:
                    for sub_name in definition.sub_concepts:
                        try:
                            sub_result = self._load_single_concept(
                                sub_name, 
                                data_source, 
                                aggregator=aggregator,
                                patient_ids=patient_ids,
                                verbose=False,
                                interval=interval,
                                align_to_admission=align_to_admission,
                                _bypass_callback=True,  # Prevent infinite recursion
                            )
                            if sub_result is not None:
                                tables[sub_name] = sub_result
                        except Exception as e:
                            if DEBUG_MODE:
                                print(f"   DEBUG: Failed to load sub-concept {sub_name}: {e}")
                            continue
                
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
            # 🔧 FIX: 当数据源未配置时，返回空表而不是报错
            # 这样用户可以继续提取其他概念，并在结果中看到哪些概念没有数据
            db_name = config.name if hasattr(config, 'name') else 'unknown'
            default_id_cols = _default_id_columns_for_db(db_name)
            
            # 记录友好的警告信息
            warn_key = (str(db_name), str(concept_name))
            if warn_key not in _MISSING_SOURCE_WARNED:
                logger.info(
                    f"⚠️  概念 '{concept_name}' 在数据库 '{db_name}' 中未配置数据源，返回空结果。"
                    f"（这是正常的，该特征在此数据库中可能不可用）"
                )
                _MISSING_SOURCE_WARNED.add(warn_key)
            
            empty_df = pd.DataFrame(columns=default_id_cols + ['charttime', concept_name])
            return ICUTable(
                data=empty_df,
                id_columns=default_id_cols,
                index_column='charttime',
                value_column=concept_name,
            )

        frames: List[pd.DataFrame] = []
        id_columns: List[str] = []
        index_column: Optional[str] = None
        unit_column: Optional[str] = None
        time_columns: List[str] = []
        _duckdb_source_count = 0  # 🚀 跟踪 DuckDB 预聚合源数量
        
        # 🔧 提取数据库名称，用于后续的数据库特定处理
        db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''

        # 🚀 预检查：多源概念的 value_transform 回调计数
        # 当 2+ 个源使用 value_transform（percent_as_numeric, set_val_na, fahr_to_cels）时，
        # DuckDB 每源单独 MEDIAN 聚合会产生 median-of-medians（≠ median-of-all-raw-data）。
        # 禁止 DuckDB 聚合以保持与 R ricu 一致的跨源池化 MEDIAN。
        _n_value_transform_sources = 0
        for _src in sources:
            _cb = getattr(_src, 'callback', None)
            if isinstance(_cb, str):
                _cb_stripped = _cb.strip()
                if (_cb_stripped == 'transform_fun(percent_as_numeric)' or
                    (_cb_stripped.startswith('convert_unit(') and
                     ('set_val(' in _cb_stripped or 'fahr_to_cels' in _cb_stripped))):
                    _n_value_transform_sources += 1
        _block_duckdb_value_transform = _n_value_transform_sources > 1

        for source in sources:
            _convert_unit_callback_for_duckdb = False  # 每个 source 重置
            use_duckdb_aggregation = False  # 每个 source 重置
            _idtbl_done = False  # 每个 source 重置
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
            
            # 🚀 rgx_itm 预匹配优化：对于有 regex 但没有 ids 的概念（rgx_itm 类型），
            # 在加载全表之前，先用 DuckDB DISTINCT 查询 sub_var 唯一值做正则预匹配。
            # 如果无匹配则直接跳过（节省数秒），有匹配则转化为精确 ids 过滤器利用谓词下推。
            _rgx_pre_matched_ids = None  # 保存预匹配结果，供后续 regex 过滤使用
            if (getattr(source, 'regex', None) and source.sub_var and 
                source.ids is None and hasattr(data_source, '_resolve_loader_from_disk')):
                try:
                    import re as _re_module
                    # 使用实例级缓存：同一表的 DISTINCT 值只查询一次
                    _distinct_cache_key = (source.table, source.sub_var)
                    if not hasattr(self, '_rgx_distinct_cache'):
                        self._rgx_distinct_cache = {}
                    
                    if _distinct_cache_key in self._rgx_distinct_cache:
                        _all_vals = self._rgx_distinct_cache[_distinct_cache_key]
                    else:
                        _table_path = data_source._resolve_loader_from_disk(source.table)
                        if _table_path is not None and isinstance(_table_path, Path):
                            import duckdb as _ddb
                            _con = _ddb.connect()
                            if _table_path.is_dir():
                                _glob = _duckdb_path(_table_path / '**' / '*.parquet')
                            else:
                                _glob = _duckdb_path(_table_path)
                            _sub_var_col = source.sub_var
                            _distinct_vals = _con.execute(
                                f"SELECT DISTINCT \"{_sub_var_col}\" FROM read_parquet('{_glob}', hive_partitioning=true) WHERE \"{_sub_var_col}\" IS NOT NULL"
                            ).fetchdf()
                            _con.close()
                            _all_vals = _distinct_vals[_sub_var_col].astype(str).tolist() if len(_distinct_vals) > 0 else []
                        else:
                            _all_vals = None  # 无法解析路径，跳过优化
                        self._rgx_distinct_cache[_distinct_cache_key] = _all_vals
                    
                    if _all_vals is not None:
                        if len(_all_vals) == 0:
                            frame = pd.DataFrame()
                            continue
                        _pattern = _re_module.compile(source.regex, _re_module.IGNORECASE)
                        _matched_vals = [v for v in _all_vals if _pattern.search(v)]
                        
                        if _matched_vals:
                            _rgx_pre_matched_ids = _matched_vals
                            if verbose or DEBUG_MODE:
                                print(f"   🔍 rgx_itm 预匹配: {len(_all_vals)} 个唯一值中匹配到 {len(_matched_vals)} 个")
                        else:
                            if verbose or DEBUG_MODE:
                                print(f"   ⏭️  rgx_itm 预匹配: {len(_all_vals)} 个唯一值中无匹配 (regex='{source.regex}')，跳过")
                            frame = pd.DataFrame()
                            continue
                except Exception as _rgx_err:
                    if DEBUG_MODE:
                        print(f"   ⚠️  rgx_itm 预匹配失败 ({_rgx_err})，回退到全表加载")
            
            # Build filters for sub_var (only for ids, NOT regex)
            # Regex filtering is handled later after table loading (see line ~1428)
            filters = []
            # 如果 rgx_itm 预匹配成功，使用匹配到的值作为精确 IN 过滤器
            if _rgx_pre_matched_ids is not None:
                filters.append(FilterSpec(
                    column=source.sub_var,
                    op=FilterOp.IN,
                    value=_rgx_pre_matched_ids,
                ))
            elif source.ids is not None:
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
            db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
            
            if patient_ids:
                # 🔧 FIX 2026-02-08: 始终检查特殊表，覆盖 defaults.id_var
                # MIMIC-III/MIMIC-IV 的 hospital tables (labevents等) 需要使用 subject_id 过滤
                # 因为这些表没有 stay_id/icustay_id 列，需要通过 hadm_id join icustays
                
                if source.table == 'labevents' and db_name in ['mimic', 'miiv', 'mimic_demo']:
                    # 🔧 MIMIC-III/IV labevents 需要使用 subject_id 过滤
                    # datasource 会通过 hadm_id join icustays 补全 icustay_id/stay_id
                    effective_id_var = 'subject_id'
                elif source.table == 'services' and db_name in ['mimic', 'miiv', 'mimic_demo']:
                    # services 表使用 stay_id/icustay_id，datasource 会自动转换为 hadm_id
                    effective_id_var = 'icustay_id' if db_name == 'mimic' else 'stay_id'
                elif source.table in ['microbiologyevents', 'd_labitems', 'prescriptions'] and db_name in ['mimic', 'miiv', 'mimic_demo']:
                    # 其他 hosp 表使用 subject_id
                    effective_id_var = 'subject_id'
                elif not effective_id_var:
                    # 如果没有配置 id_var，尝试检测常见的ID列
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
                            # 需要在 metadata 中保存原始的 stay_id/icustay_id，供 datasource 在 join 后精确过滤
                            metadata = None
                            db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                            hospital_tables = ['labevents', 'prescriptions', 'microbiologyevents', 'emar', 'pharmacy']
                            
                            # 🔧 FIX 2026-02-08: 同时支持 MIMIC-III 和 MIMIC-IV
                            # MIMIC-IV 使用 stay_id，MIMIC-III 使用 icustay_id
                            if (db_name in ['miiv', 'mimic_demo', 'mimic'] and 
                                source.table in hospital_tables and 
                                effective_id_var == 'subject_id'):
                                # 确定目标 ID 列名
                                target_id_col = 'icustay_id' if db_name == 'mimic' else 'stay_id'
                                if target_id_col in expanded_patient_ids:
                                    original_stay_ids = expanded_patient_ids.get(target_id_col)
                                    if original_stay_ids:
                                        metadata = {'original_stay_ids': original_stay_ids}
                                        if DEBUG_MODE:
                                            print(f"   💾 在 subject_id 过滤器中附加原始 {target_id_col}: {len(original_stay_ids)} 个")
                            
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
                    f.op == FilterOp.IN and
                    _is_patient_id_filter_column(f.column, effective_id_var)
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
                
                # 🔧 FIX: HiRID缓存分支时间转换
                # 缓存的数据保留原始datetime格式，需要在使用时转换为相对小时数
                # 这确保从缓存加载的概念也能正确处理时间
                # 🚀 但是跳过 general 表本身，它不需要时间转换
                if db_name == 'hirid' and source.table != 'general':
                    time_col = table.index_column or 'datetime'
                    if time_col in frame.columns and pd.api.types.is_datetime64_any_dtype(frame[time_col]):
                        try:
                            # 加载general表获取入院时间
                            general = data_source.load_table('general', verbose=False)
                            if hasattr(general, 'data'):
                                general_df = general.data
                            else:
                                general_df = general
                            
                            if 'admissiontime' in general_df.columns and 'patientid' in general_df.columns:
                                # 获取目标患者的入院时间
                                target_patient_ids = frame['patientid'].unique().tolist()
                                adm_df = general_df[general_df['patientid'].isin(target_patient_ids)][['patientid', 'admissiontime']].copy()
                                adm_df['admissiontime'] = pd.to_datetime(adm_df['admissiontime'], errors='coerce')
                                
                                # 确保datetime列没有时区信息（与admissiontime对齐）
                                if frame[time_col].dt.tz is not None:
                                    frame[time_col] = frame[time_col].dt.tz_localize(None)
                                
                                # 合并入院时间
                                frame = frame.merge(adm_df, on='patientid', how='left')
                                
                                # 计算相对小时数：(datetime - admissiontime) / 3600
                                if 'admissiontime' in frame.columns:
                                    frame[time_col] = (frame[time_col] - frame['admissiontime']).dt.total_seconds() / 3600.0
                                    frame = frame.drop(columns=['admissiontime'])
                                    
                                    # 更新 table 对象
                                    table = ICUTable(
                                        data=frame,
                                        id_columns=cached_table.id_columns,
                                        index_column=cached_table.index_column,
                                        value_column=cached_table.value_column,
                                        unit_column=cached_table.unit_column,
                                    )
                                    
                                    if verbose or DEBUG_MODE:
                                        print(f"   🕐 [HiRID缓存] 时间转换: {time_col} 从datetime → 相对小时数")
                        except Exception as e:
                            if DEBUG_MODE:
                                print(f"   ⚠️  [HiRID缓存] 时间转换失败: {e}")
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
                    
                    # 🚀 DuckDB层聚合优化：对有分桶目录的表直接在DuckDB中降采样
                    # 支持所有数据库（AUMC/HiRID/MIIV/MIMIC/SIC/eICU），只要表有分桶目录
                    # 性能提升：MIIV 6 vitals 50 patients 从 545s → ~10s
                    db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                    use_duckdb_aggregation = False
                    _idtbl_done = False
                    _convert_unit_callback_for_duckdb = False
                    _convert_unit_factor = None
                    _convert_unit_op = None
                    _convert_unit_filter = None
                    
                    # 通用检查：表是否有分桶目录或扁平parquet目录 + 源是否有 sub_var + ids + 无复杂 callback
                    _has_bucket_dir = False
                    try:
                        _bucket_dir_check = data_source._resolve_bucket_directory(source.table)
                        if _bucket_dir_check is None:
                            _bucket_dir_check = data_source._resolve_flat_parquet_directory(source.table)
                        _has_bucket_dir = _bucket_dir_check is not None
                    except Exception:
                        pass
                    
                    # 🚀 DuckDB 可内联回调检测
                    _is_percent_as_numeric = False
                    _is_set_val_na = False
                    _is_fahr_to_cels = False
                    _is_transform_binary_op = False  # 🚀 transform_fun(binary_op(...))
                    _transform_binary_op_operator = None  # '*', '/', '+', '-'
                    _transform_binary_op_value = None     # numeric factor
                    _duckdb_value_transform = None  # SQL transform expression
                    
                    if _has_bucket_dir:
                        has_sub_var = getattr(source, 'sub_var', None) is not None
                        has_callback = getattr(source, 'callback', None) is not None
                        is_convert_unit = (
                            has_callback and isinstance(source.callback, str) and
                            source.callback.strip().startswith('convert_unit(')
                        )
                        # 🚀 检测 transform_fun(percent_as_numeric) — 可内联到 DuckDB
                        _is_percent_as_numeric = (
                            has_callback and isinstance(source.callback, str) and
                            source.callback.strip() == 'transform_fun(percent_as_numeric)'
                        )
                        # 🚀 检测 transform_fun(binary_op(`*`, N)) — 简单乘除可内联到 DuckDB
                        # 例: o2sat AUMC item 12311 callback='transform_fun(binary_op(`*`, 100))'
                        if (has_callback and isinstance(source.callback, str) and
                                source.callback.strip().startswith('transform_fun(binary_op(')):
                            import re as _re_tfm
                            _tfm_match = _re_tfm.match(
                                r"transform_fun\(binary_op\(`([*/+-])`,\s*([0-9.]+)\)\)",
                                source.callback.strip()
                            )
                            if _tfm_match:
                                _is_transform_binary_op = True
                                _transform_binary_op_operator = _tfm_match.group(1)
                                _transform_binary_op_value = float(_tfm_match.group(2))
                        _mimic_hospital_tables = {'prescriptions', 'labevents', 'microbiologyevents', 'emar', 'pharmacy', 'services'}
                        _needs_hadm_to_stay_mapping = (
                            db_name in ('miiv', 'miiv_demo', 'mimic', 'mimic_demo')
                            and source.table in _mimic_hospital_tables
                        )
                        # 🔧 FIX: id_tbl 概念（height, weight 等）需要 per-patient 聚合，不适合 DuckDB 的 per-hour GROUP BY
                        _target = getattr(definition, 'target', 'ts_tbl')
                        # 🔧 eICU nursecharting 暂不启用 DuckDB（string itemid）
                        _skip_db_duckdb = (
                            (db_name in ('eicu', 'eicu_demo') and source.table == 'nursecharting')
                        )
                        # 🚀 使用 regex 预匹配的 IDs 作为 source.ids 的补充
                        _effective_ids = source.ids
                        if not _effective_ids and _rgx_pre_matched_ids:
                            _effective_ids = _rgx_pre_matched_ids
                        # 🚀 扩展 DuckDB 门控：接受 percent_as_numeric/set_val_na/fahr_to_cels/binary_op 回调
                        # 但多源概念的 value_transform 会被禁止（防止 median-of-medians）
                        _can_inline_callback = not has_callback or is_convert_unit or _is_percent_as_numeric or _is_transform_binary_op
                        if _block_duckdb_value_transform and (_is_percent_as_numeric or is_convert_unit or _is_transform_binary_op):
                            # 多源 value_transform：禁止内联，回退到 Python 回调路径
                            _can_inline_callback = not has_callback
                        # 🚀 id_tbl DuckDB 快速路径：per-patient 聚合（MEDIAN）代替全表加载
                        # 例如 height/weight 从 chartevents(5.7GB) 只需 ≤500 行
                        if has_sub_var and _can_inline_callback and _effective_ids and _target == 'id_tbl' and not _skip_db_duckdb:
                            _idtbl_ids = list(_effective_ids) if hasattr(_effective_ids, '__iter__') else [_effective_ids]
                            if _idtbl_ids:
                                try:
                                    from .datasource import _get_duckdb_connection
                                    _idtbl_conn = _get_duckdb_connection()
                                    _idtbl_sub_var = source.sub_var
                                    _idtbl_val_var = getattr(source, 'value_var', None) or 'value'
                                    # Resolve bucket/flat directory
                                    _idtbl_bucket_dir = data_source._resolve_bucket_directory(source.table)
                                    _idtbl_flat_dir = None
                                    if _idtbl_bucket_dir is None:
                                        _idtbl_flat_dir = data_source._resolve_flat_parquet_directory(source.table)
                                    if _idtbl_bucket_dir is not None or _idtbl_flat_dir is not None:
                                        # Determine glob pattern
                                        if _idtbl_bucket_dir is not None:
                                            _idtbl_bucket_path = Path(_idtbl_bucket_dir) if not isinstance(_idtbl_bucket_dir, Path) else _idtbl_bucket_dir
                                            _idtbl_glob = _duckdb_path(_idtbl_bucket_path / '**' / '*.parquet')
                                        else:
                                            _idtbl_flat_path = Path(_idtbl_flat_dir) if not isinstance(_idtbl_flat_dir, Path) else _idtbl_flat_dir
                                            _idtbl_glob = _duckdb_path(_idtbl_flat_path / '*.parquet')
                                        # Determine ID column
                                        _idtbl_id_cfg = data_source.config.id_configs.get('icustay') or data_source.config.id_configs.get('patient')
                                        _idtbl_id_col = _idtbl_id_cfg.id if hasattr(_idtbl_id_cfg, 'id') else _idtbl_id_cfg
                                        if not _idtbl_id_col:
                                            _idtbl_id_col = {'aumc': 'admissionid', 'hirid': 'patientid',
                                                            'mimic': 'icustay_id', 'mimic_demo': 'icustay_id',
                                                            'sic': 'CaseID', 'eicu': 'patientunitstayid',
                                                            'miiv': 'stay_id'}.get(db_name, 'stay_id')
                                        # Build WHERE
                                        _is_str = any(isinstance(x, str) for x in _idtbl_ids)
                                        _ids_str = ", ".join(f"'{x}'" if _is_str else str(x) for x in _idtbl_ids)
                                        _idtbl_where = f"{_idtbl_sub_var} IN ({_ids_str})"
                                        # Patient filter
                                        _idtbl_pid_filter = ""
                                        if patient_ids:
                                            _pids = list(next(iter(patient_ids.values()))) if isinstance(patient_ids, dict) else list(patient_ids)
                                            _pid_str = ",".join(str(p) for p in _pids)
                                            _idtbl_pid_filter = f"AND {_idtbl_id_col} IN ({_pid_str})"
                                        # DuckDB convert_unit inline — CASE WHEN unit filter → convert, ELSE → raw
                                        _idtbl_val_expr = f'TRY_CAST("{_idtbl_val_var}" AS DOUBLE)'
                                        if is_convert_unit and source.callback:
                                            import re as _re_cu
                                            _cu_m = _re_cu.match(r"convert_unit\((.+)\)", source.callback.strip(), _re_cu.DOTALL)
                                            if _cu_m:
                                                _cu_args_str = _cu_m.group(1)
                                                _cu_args = []
                                                _cu_lvl, _cu_cur = 0, []
                                                for _cu_ch in _cu_args_str:
                                                    if _cu_ch == '(': _cu_lvl += 1
                                                    elif _cu_ch == ')': _cu_lvl -= 1
                                                    elif _cu_ch == ',' and _cu_lvl == 0:
                                                        _cu_args.append(''.join(_cu_cur).strip())
                                                        _cu_cur = []
                                                        continue
                                                    _cu_cur.append(_cu_ch)
                                                if _cu_cur: _cu_args.append(''.join(_cu_cur).strip())
                                                # Get unit column name
                                                _cu_unit_col = None
                                                try:
                                                    _cu_tcfg = data_source.config.get_table(source.table)
                                                    _cu_unit_col = getattr(_cu_tcfg.defaults, 'unit_var', None)
                                                except Exception:
                                                    pass
                                                if not _cu_unit_col:
                                                    # Fallback: known unit column names per DB
                                                    _cu_unit_col = {'aumc': 'unitname', 'sic': 'Unit'}.get(db_name, 'valueuom')
                                                _cu_filter = _cu_args[2].strip().strip("'\"") if len(_cu_args) > 2 else None
                                                _cu_first = _cu_args[0].strip() if _cu_args else ''
                                                _cu_bm = _re_cu.match(r"binary_op\(`(.+?)`,\s*(.+)\)", _cu_first)
                                                if _cu_bm and _cu_filter and _cu_unit_col:
                                                    _cu_op = _cu_bm.group(1)
                                                    _cu_factor = _cu_bm.group(2)
                                                    _idtbl_val_expr = (
                                                        f'CASE WHEN REGEXP_MATCHES(LOWER("{_cu_unit_col}"), \'{_cu_filter}\') '
                                                        f'THEN TRY_CAST("{_idtbl_val_var}" AS DOUBLE) {_cu_op} {_cu_factor} '
                                                        f'ELSE TRY_CAST("{_idtbl_val_var}" AS DOUBLE) END'
                                                    )
                                                    _convert_unit_callback_for_duckdb = True
                                                elif _cu_first == 'set_val(NA)' and _cu_filter and _cu_unit_col:
                                                    _idtbl_val_expr = (
                                                        f'CASE WHEN REGEXP_MATCHES(LOWER("{_cu_unit_col}"), \'{_cu_filter}\') '
                                                        f'THEN NULL '
                                                        f'ELSE TRY_CAST("{_idtbl_val_var}" AS DOUBLE) END'
                                                    )
                                                    _convert_unit_callback_for_duckdb = True
                                                elif _cu_first == 'fahr_to_cels' and _cu_filter and _cu_unit_col:
                                                    _idtbl_val_expr = (
                                                        f'CASE WHEN REGEXP_MATCHES(LOWER("{_cu_unit_col}"), \'{_cu_filter}\') '
                                                        f'THEN (TRY_CAST("{_idtbl_val_var}" AS DOUBLE) - 32.0) * 5.0 / 9.0 '
                                                        f'ELSE TRY_CAST("{_idtbl_val_var}" AS DOUBLE) END'
                                                    )
                                                    _convert_unit_callback_for_duckdb = True
                                        # Query: GROUP BY patient, MEDIAN(value)
                                        _idtbl_sql = f"""
                                            SELECT {_idtbl_id_col},
                                                   MEDIAN({_idtbl_val_expr}) AS {concept_name}
                                            FROM read_parquet('{_idtbl_glob}', hive_partitioning=true, union_by_name=true)
                                            WHERE {_idtbl_where} {_idtbl_pid_filter}
                                              AND {_idtbl_val_var} IS NOT NULL
                                            GROUP BY {_idtbl_id_col}
                                        """
                                        frame = _idtbl_conn.execute(_idtbl_sql).fetchdf()
                                        if len(frame) > 0:
                                            table = ICUTable(
                                                data=frame,
                                                id_columns=[_idtbl_id_col],
                                                index_column=_idtbl_id_col,
                                                value_column=concept_name,
                                            )
                                            table._pre_aggregated = True
                                            use_duckdb_aggregation = True
                                            _idtbl_done = True
                                            _duckdb_source_count += 1
                                            if verbose:
                                                print(f"   ✅ id_tbl DuckDB聚合: {concept_name} → {len(frame):,} 行")
                                except Exception as _idtbl_err:
                                    if verbose:
                                        print(f"   ⚠️ id_tbl DuckDB失败({_idtbl_err}), 回退到原始路径")
                        if has_sub_var and _can_inline_callback and _effective_ids and _target != 'id_tbl' and not _skip_db_duckdb:
                            itemids = list(_effective_ids) if hasattr(_effective_ids, '__iter__') else [_effective_ids]
                            use_duckdb_aggregation = len(itemids) > 0
                            if is_convert_unit and use_duckdb_aggregation:
                                _convert_unit_callback_for_duckdb = True
                                import re as _re
                                _cb = source.callback.strip()
                                _m = _re.match(r"convert_unit\((.+)\)", _cb, _re.DOTALL)
                                if _m:
                                    _args_str = _m.group(1)
                                    _args = []
                                    _lvl, _cur = 0, []
                                    for _ch in _args_str:
                                        if _ch == '(': _lvl += 1
                                        elif _ch == ')': _lvl -= 1
                                        elif _ch == ',' and _lvl == 0:
                                            _args.append(''.join(_cur).strip())
                                            _cur = []
                                            continue
                                        _cur.append(_ch)
                                    if _cur: _args.append(''.join(_cur).strip())
                                    _bm = _re.match(r"binary_op\(`(.+?)`,\s*(.+)\)", _args[0].strip()) if _args else None
                                    if _bm:
                                        _convert_unit_op = _bm.group(1)
                                        try: _convert_unit_factor = float(_bm.group(2))
                                        except Exception: pass
                                    # 🚀 检测 set_val(NA) 和 fahr_to_cels
                                    elif _args:
                                        _first_arg = _args[0].strip()
                                        if _first_arg == 'set_val(NA)':
                                            _is_set_val_na = True
                                        elif _first_arg == 'fahr_to_cels':
                                            _is_fahr_to_cels = True
                                    _convert_unit_filter = _args[2].strip().strip("'\"") if len(_args) > 2 else None
                                # 🚀 set_val(NA) / fahr_to_cels 可内联到 DuckDB
                                if _convert_unit_factor is None:
                                    if _is_set_val_na or _is_fahr_to_cels:
                                        _convert_unit_callback_for_duckdb = True
                                        # value_transform will be built below after value_col is known
                                    else:
                                        _convert_unit_callback_for_duckdb = False
                                        use_duckdb_aggregation = False
                            if _is_percent_as_numeric and use_duckdb_aggregation:
                                _convert_unit_callback_for_duckdb = True
                                # value_transform will be built below after value_col is known
                            if _is_transform_binary_op and use_duckdb_aggregation:
                                _convert_unit_callback_for_duckdb = True
                                # value_transform will be built below after value_col is known
                    
                    if _idtbl_done:
                        # id_tbl DuckDB 快速路径已完成 — table 已设置，只需提取 frame
                        frame = table.data.copy()
                    elif use_duckdb_aggregation and not _idtbl_done:
                        # 使用DuckDB层聚合
                        from .datasource import load_bucketed_table_aggregated
                        
                        # 获取value列名：优先使用source.value_var，否则使用表默认值
                        value_col = getattr(source, 'value_var', None)
                        if not value_col:
                            # 从表配置中获取默认value_var
                            table_cfg = data_source.config.get_table(source.table)
                            value_col = table_cfg.defaults.val_var or 'value'
                        
                        interval = kwargs.get('interval', pd.Timedelta(hours=1))
                        interval_minutes = interval.total_seconds() / 60.0 if isinstance(interval, pd.Timedelta) else 60.0
                        
                        # 🔧 获取患者ID列表 (关键: 处理dict和list两种格式)
                        # patient_ids 可能是 dict 格式 {'patientid': [1,2,3]} 或 list 格式 [1,2,3]
                        if patient_ids is None:
                            patient_ids_list = None
                        elif isinstance(patient_ids, dict):
                            # dict格式：取第一个values
                            patient_ids_list = list(next(iter(patient_ids.values())))
                        else:
                            patient_ids_list = list(patient_ids)
                        
                        if verbose:
                            print(f"   🚀 使用DuckDB聚合优化: {source.table} itemids={len(itemids)} value_col={value_col} patients={len(patient_ids_list) if patient_ids_list else 'all'}")
                        
                        # ﻿ 构建 DuckDB 值转换表达式（用于内联回调）
                        if _is_percent_as_numeric:
                            # percent_as_numeric: 去除 '%' 并转为数值
                            _duckdb_value_transform = f"TRY_CAST(REPLACE(TRIM(CAST({value_col} AS VARCHAR)), '%', '') AS DOUBLE)"
                        elif _is_transform_binary_op:
                            # transform_fun(binary_op(`*`, N)): 简单算术内联到 DuckDB
                            _duckdb_value_transform = f'(TRY_CAST("{value_col}" AS DOUBLE) {_transform_binary_op_operator} {_transform_binary_op_value})'
                        elif (_is_set_val_na or _is_fahr_to_cels) and _convert_unit_filter:
                            # 需要 unit 列匹配过滤模式 — 从表配置获取实际列名
                            _unit_var_for_duckdb = None
                            try:
                                _tcfg_for_unit = data_source.config.get_table(source.table)
                                _unit_var_for_duckdb = getattr(_tcfg_for_unit.defaults, "unit_var", None)
                            except Exception:
                                pass
                            if not _unit_var_for_duckdb:
                                _unit_var_for_duckdb = "unit"
                            if _is_set_val_na:
                                # set_val(NA): 匹配单位模式时设为 NULL
                                _duckdb_value_transform = (
                                    f'CASE WHEN regexp_matches(COALESCE(CAST("{_unit_var_for_duckdb}" AS VARCHAR), '
                                    f"''), '(?i){_convert_unit_filter}') "
                                    f'THEN NULL ELSE TRY_CAST("{value_col}" AS DOUBLE) END'
                                )
                            elif _is_fahr_to_cels:
                                # fahr_to_cels: 无条件 F→C 转换
                                # 源 itemid 已确保数据为华氏度（如 MIIV 223761/224027）
                                # 不能依赖 unit 列（可能为 NaN），否则未转换的 F 值
                                # 会被 filter_bounds(max=42) 错误丢弃
                                _duckdb_value_transform = (
                                    f'(TRY_CAST("{value_col}" AS DOUBLE) - 32.0) * 5.0 / 9.0'
                                )
                        # �🔧 FIX 2026-03: 使用调用者指定的聚合函数，而非硬编码 median
                        # 关键场景: sofa_cardio 的 MAP 子概念需要 min 聚合
                        # 如果 DuckDB 层使用 median，后续 Python 层的 min 聚合无效
                        # （因为每小时已只有1个 median 值，min(single_value) = same_value）
                        _duckdb_agg = 'median'  # 默认与 R ricu 一致
                        if isinstance(aggregator, str) and aggregator in ('min', 'max', 'mean', 'sum', 'first'):
                            _duckdb_agg = aggregator
                        
                        frame = load_bucketed_table_aggregated(
                            data_source,
                            source.table,
                            value_col,
                            itemids,
                            interval_minutes=interval_minutes,
                            patient_ids=patient_ids_list,  # 🔧 修复: 传入患者ID过滤
                            agg_func=_duckdb_agg,
                            # 2026-03-11: 在 DuckDB WHERE 中过滤 min/max（匹配 R ricu）
                            # R ricu 流程: load_id → callback → filter_bounds → change_interval → aggregate
                            # 对于 value_transform 内联回调，min/max 应用于转换后的值（在聚合表达式内部）
                            # 对于无回调概念，min/max 直接在 WHERE 子句过滤 raw value
                            value_min=definition.minimum,
                            value_max=definition.maximum,
                            include_unit=False,  # unit 不再通过 ANY_VALUE 获取
                            # 🚀 convert_unit 内联参数：在DuckDB中直接做单位转换
                            convert_unit_op=_convert_unit_op if _convert_unit_callback_for_duckdb else None,
                            convert_unit_factor=_convert_unit_factor if _convert_unit_callback_for_duckdb else None,
                            convert_unit_filter=_convert_unit_filter if _convert_unit_callback_for_duckdb else None,
                            # 🚀 通用值转换表达式（percent_as_numeric, set_val_na, fahr_to_cels）
                            value_transform=_duckdb_value_transform,
                        )
                        
                        if _convert_unit_callback_for_duckdb and verbose and len(frame) > 0:
                            print(f"   🔧 convert_unit DuckDB内联完成: {len(frame):,} 行")
                        
                        # 2026-03-11: 移除 AUMC 早期 filter_bounds（不再需要）
                        # 现在 DuckDB 池化所有 itemid 后直接聚合（匹配 R ricu），
                        # 标准 filter_bounds 在 change_interval 之后执行（~L3839）
                        
                        # 创建ICUTable对象
                        # 确定输出列名（匹配 load_bucketed_table_aggregated 的输出）
                        if db_name == 'aumc':
                            _duckdb_id_col = 'admissionid'
                            _duckdb_time_col = 'measuredat_minutes'
                        elif db_name == 'hirid':
                            _duckdb_id_col = 'patientid'
                            _duckdb_time_col = 'charttime'
                        elif db_name in ('mimic', 'mimic_demo'):
                            _duckdb_id_col = 'icustay_id'
                            _duckdb_time_col = 'charttime'
                        elif db_name in ('sic', 'sic_demo'):
                            _duckdb_id_col = 'CaseID'
                            _duckdb_time_col = 'charttime'
                        elif db_name in ('eicu', 'eicu_demo'):
                            _duckdb_id_col = 'patientunitstayid'
                            _duckdb_time_col = 'charttime'
                        else:
                            _duckdb_id_col = 'stay_id'
                            _duckdb_time_col = 'charttime'
                        
                        table = ICUTable(
                            data=frame,
                            id_columns=[_duckdb_id_col],
                            index_column=_duckdb_time_col,
                            value_column=value_col,
                        )
                        # 🚀 标记为 DuckDB 预聚合，change_interval 可跳过冗余 groupby
                        table._pre_aggregated = True
                        _duckdb_source_count += 1
                        
                        if verbose:
                            print(f"   ✅ DuckDB聚合完成: {len(frame):,} 行")
                    elif (db_name == 'hirid' and 
                          getattr(source, 'callback', '') == 'hirid_death' and
                          source.table == 'observations'):
                        # 🚀 hirid_death 快速路径：跳过加载 115M 行原始数据
                        # 直接在 DuckDB 中 GROUP BY patientid → MAX(datetime)
                        # 然后与 general 表的 discharge_status 合并
                        import duckdb as _hd_duckdb
                        bucket_dir = data_source.base_path / 'observations_bucket'
                        _hd_conn = _hd_duckdb.connect()
                        _hd_conn.execute("SET memory_limit = '2GB'")
                        
                        # 先获取死亡患者ID
                        try:
                            general_tbl = data_source.load_table('general', columns=['patientid', 'discharge_status'], verbose=False)
                            general_df = general_tbl.data if hasattr(general_tbl, 'data') else general_tbl
                            if not isinstance(general_df, pd.DataFrame):
                                general_df = pd.DataFrame(general_df)
                            dead_pids = set(general_df.loc[
                                general_df['discharge_status'].astype(str).str.lower() == 'dead',
                                'patientid'
                            ].unique())
                        except Exception:
                            dead_pids = set()
                        
                        # 确定要查询的死亡患者列表
                        _query_pids = dead_pids
                        if patient_ids and dead_pids:
                            if isinstance(patient_ids, dict):
                                pid_list = list(next(iter(patient_ids.values())))
                            else:
                                pid_list = list(patient_ids)
                            _query_pids = set(p for p in pid_list if p in dead_pids)
                        
                        if _query_pids and bucket_dir.exists():
                            glob_pattern = _duckdb_path(bucket_dir / 'bucket_id=*' / '*.parquet')
                            src_ids = list(source.ids) if hasattr(source.ids, '__iter__') else [source.ids]
                            ids_str = ', '.join(str(x) for x in src_ids)
                            pids_str = ', '.join(str(p) for p in _query_pids)
                            
                            query = f"""
                                SELECT patientid, MAX(datetime) as datetime
                                FROM read_parquet('{glob_pattern}', union_by_name=true)
                                WHERE variableid IN ({ids_str})
                                  AND patientid IN ({pids_str})
                                GROUP BY patientid
                            """
                            frame = _hd_conn.execute(query).fetchdf()
                            frame[concept_name] = True
                        else:
                            frame = pd.DataFrame(columns=['patientid', 'datetime', concept_name])
                        
                        _hd_conn.close()
                        
                        table = ICUTable(
                            data=frame,
                            id_columns=['patientid'],
                            index_column='datetime',
                            value_column=concept_name,
                        )
                        
                        if verbose:
                            print(f"   ✅ hirid_death 快速路径: {len(frame):,} 行")
                        
                        # 标记callback已处理，跳过后续callback
                        _convert_unit_callback_for_duckdb = True
                    elif (
                        _has_bucket_dir
                        and not has_sub_var
                        and not getattr(source, 'ids', None)
                        and getattr(source, 'value_var', None)
                        and not has_callback
                        and _target != 'id_tbl'
                    ):
                        logger.debug(f"🚀 宽表单列DuckDB聚合触发: {source.table}.{source.value_var}")
                        # 🚀 Wide table single-column DuckDB aggregation
                        # eICU vitalperiodic/vitalaperiodic: column-based tables without itemid
                        # Raw loading reads ALL rows (146M+ for vitalperiodic), this does
                        # GROUP BY in DuckDB to reduce to ~2M rows before pandas processing
                        import duckdb as _wt_duckdb
                        from .datasource import _get_duckdb_connection

                        _wt_val_var = source.value_var
                        _wt_table_cfg = data_source.config.get_table(source.table)
                        _wt_id_col = _wt_table_cfg.defaults.id_var or 'patientunitstayid'
                        _wt_time_col = _wt_table_cfg.defaults.index_var or 'observationoffset'
                        _wt_interval_minutes = 60.0

                        _wt_agg_func = 'MEDIAN'
                        if isinstance(aggregator, str) and aggregator in ('min', 'max', 'mean', 'sum'):
                            _wt_agg_func = aggregator.upper()

                        # Find parquet files
                        _wt_table_path = data_source._resolve_loader_from_disk(source.table)
                        if _wt_table_path is not None:
                            _wt_dir = Path(_wt_table_path) if not isinstance(_wt_table_path, Path) else _wt_table_path
                            _wt_glob = _duckdb_path(_wt_dir / '*.parquet') if _wt_dir.is_dir() else _duckdb_path(_wt_dir)

                            # Patient ID filter
                            if patient_ids is None:
                                _wt_pid_list = None
                            elif isinstance(patient_ids, dict):
                                _wt_pid_list = list(next(iter(patient_ids.values())))
                            else:
                                _wt_pid_list = list(patient_ids)

                            _wt_pid_filter = ''
                            if _wt_pid_list:
                                _wt_ids_str = ','.join(str(x) for x in _wt_pid_list)
                                _wt_pid_filter = f'AND {_wt_id_col} IN ({_wt_ids_str})'

                            # Output in ORIGINAL time units (minutes for eICU) but floored to hourly
                            # This keeps compatibility with raw loading path — downstream
                            # time alignment and change_interval work unchanged
                            _wt_time_expr = f'FLOOR({_wt_time_col} / {_wt_interval_minutes}) * {_wt_interval_minutes}'
                            _wt_query = f"""
                                SELECT {_wt_id_col},
                                       {_wt_time_expr} AS {_wt_time_col},
                                       {_wt_agg_func}({_wt_val_var}) AS {_wt_val_var}
                                FROM read_parquet('{_wt_glob}', union_by_name=true)
                                WHERE {_wt_val_var} IS NOT NULL {_wt_pid_filter}
                                GROUP BY {_wt_id_col}, {_wt_time_expr}
                                ORDER BY {_wt_id_col}, 2
                            """

                            _wt_conn = _get_duckdb_connection()
                            try:
                                frame = _wt_conn.execute(_wt_query).fetchdf()
                            except Exception:
                                # Fallback to raw loading
                                frame = None

                            if frame is not None and len(frame) > 0:
                                table = ICUTable(
                                    data=frame,
                                    id_columns=[_wt_id_col],
                                    index_column=_wt_time_col,
                                    value_column=_wt_val_var,
                                )
                                table._pre_aggregated = True
                                if verbose:
                                    logger.info(f"   🚀 宽表单列DuckDB聚合: {source.table}.{_wt_val_var} -> {len(frame):,} 行")
                            else:
                                # Fallback
                                table = data_source.load_table(
                                    source.table,
                                    columns=extra_columns if extra_columns else None,
                                    filters=filters, verbose=verbose
                                )
                                frame = table.data.copy()
                        else:
                            table = data_source.load_table(
                                source.table,
                                columns=extra_columns if extra_columns else None,
                                filters=filters, verbose=verbose
                            )
                            frame = table.data.copy()
                    else:
                        # 原始加载路径
                        table = data_source.load_table(
                            source.table, 
                            columns=extra_columns if extra_columns else None,
                            filters=filters, 
                            verbose=verbose
                        )
                        
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
                    
                    # 🔧 HiRID特殊处理：将datetime时间转换为相对入院时间的小时数
                    # HiRID使用绝对datetime时间戳，需要从general表获取admissiontime
                    # 🚀 但是跳过 general 表本身，它不需要时间转换（admissiontime 本身就是入院时间）
                    if db_name == 'hirid' and source.table != 'general':
                        time_col = table.index_column or 'datetime'
                        if time_col in frame.columns and pd.api.types.is_datetime64_any_dtype(frame[time_col]):
                            try:
                                # 加载general表获取入院时间
                                general = data_source.load_table('general', verbose=False)
                                if hasattr(general, 'data'):
                                    general_df = general.data
                                else:
                                    general_df = general
                                
                                if 'admissiontime' in general_df.columns and 'patientid' in general_df.columns:
                                    # 获取目标患者的入院时间
                                    target_patient_ids = frame['patientid'].unique().tolist()
                                    adm_df = general_df[general_df['patientid'].isin(target_patient_ids)][['patientid', 'admissiontime']].copy()
                                    adm_df['admissiontime'] = pd.to_datetime(adm_df['admissiontime'], errors='coerce')
                                    
                                    # 确保datetime列没有时区信息（与admissiontime对齐）
                                    if frame[time_col].dt.tz is not None:
                                        frame[time_col] = frame[time_col].dt.tz_localize(None)
                                    
                                    # 合并入院时间
                                    frame = frame.merge(adm_df, on='patientid', how='left')
                                    
                                    # 计算相对小时数：(datetime - admissiontime) / 3600
                                    if 'admissiontime' in frame.columns:
                                        frame[time_col] = (frame[time_col] - frame['admissiontime']).dt.total_seconds() / 3600.0
                                        frame = frame.drop(columns=['admissiontime'])
                                        
                                        if verbose or DEBUG_MODE:
                                            print(f"   🕐 [HiRID] 时间转换: {time_col} 从datetime → 相对小时数")
                            except Exception as e:
                                if DEBUG_MODE:
                                    print(f"   ⚠️  [HiRID] 时间转换失败: {e}")
                    
                    # 性能优化：对于AUMC/HiRID等高频数据，在表加载后立即降采样
                    # 检测数据库类型和数据频率
                    db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                    is_high_freq_db = db_name in ['aumc', 'hirid']
                    
                    # 🔧 FIX: 不对有callback的源做预降采样！
                    # 原因：callback（如convert_unit, aumc_rate_kg等）需要原始数据
                    # 预降采样会在callback之前聚合值，导致：
                    # - convert_unit: 先median再×7.6 ≠ 先×7.6再median（非线性变换时）
                    # - aumc_rate_kg: 先median再/kg ≠ 先/kg再median
                    # 🔧 FIX 2026-02: 对所有有callback的源都跳过预降采样
                    has_callback = getattr(source, 'callback', None) is not None
                    skip_resample = has_callback
                    
                    if is_high_freq_db and table.index_column and len(frame) > 1000 and not skip_resample:
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
                                    # 对于AUMC，数值时间列单位为分钟（而不是小时）
                                    # 对于HiRID，时间已经转换为小时，不需要乘以60
                                    # 因此需要在原始单位上进行取整，以保留负时间点并避免单位错位。
                                    if db_name == 'aumc':
                                        # AUMC: 原始单位为分钟，将 interval 从小时转换为分钟
                                        native_interval = interval_hours * 60.0
                                    else:
                                        # HiRID和其他数据库：时间已经是小时
                                        native_interval = interval_hours
                                    # 使用向下取整保留入ICU前的负时间点（避免 .round() 将小于0的值四舍五入到0）
                                    frame[time_col + '_rounded'] = np.floor(frame[time_col] / native_interval) * native_interval
                                    
                                    # 聚合：根据数据类型选择聚合函数
                                    # 对于输出类数据（尿量等）使用sum，其他使用median (R ricu默认)
                                    # 🔧 FIX 2026-02: 当callback是aggregate_fun('sum')时，使用sum聚合
                                    agg_func = 'median'  # 默认
                                    if 'urine' in value_col.lower() or 'output' in value_col.lower():
                                        agg_func = 'sum'
                                    elif hasattr(source, 'callback') and source.callback:
                                        import re as re_module
                                        agg_match = re_module.search(r"aggregate_fun\(['\"]([^'\"]+)['\"]", source.callback)
                                        if agg_match:
                                            agg_func = agg_match.group(1)  # e.g., 'sum'
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
                                                print("       frame列类型:")
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
                                        # 🔧 FIX 2026-02: 当callback是aggregate_fun('sum')时，使用sum聚合
                                        agg_func = 'median'  # 默认
                                        if 'urine' in value_col.lower() or 'output' in value_col.lower():
                                            agg_func = 'sum'
                                        elif hasattr(source, 'callback') and source.callback:
                                            import re as re_module
                                            agg_match = re_module.search(r"aggregate_fun\(['\"]([^'\"]+)['\"]", source.callback)
                                            if agg_match:
                                                agg_func = agg_match.group(1)  # e.g., 'sum'
                                        
                                        for group_id, group_df in frame.groupby(id_cols):
                                            group_df = group_df.set_index(time_col)
                                            
                                            # 聚合所有数值列
                                            numeric_cols = group_df.select_dtypes(include=[np.number]).columns.tolist()
                                            if value_col in numeric_cols:
                                                # value_col使用特定的聚合函数
                                                agg_dict = {value_col: agg_func}
                                                # 其他数值列使用median (R ricu默认)
                                                for col in numeric_cols:
                                                    if col != value_col:
                                                        agg_dict[col] = 'median'
                                            else:
                                                agg_dict = {col: 'median' for col in numeric_cols}
                                            
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
                                        agg_func = 'sum' if 'urine' in value_col.lower() or 'output' in value_col.lower() else 'median'
                                        numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
                                        agg_dict = {col: agg_func if col == value_col else 'median' for col in numeric_cols}
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
                    # 🚀 重要：使用DuckDB聚合路径时跳过缓存！因为DuckDB已经高效加载数据，
                    # 再加载一次只为缓存会导致严重性能问题（16秒 → 0.3秒的差距）
                    # 🔧 FIX 2026-02-02: MIMIC-III chartevents 使用 CSV 回退模式时跳过缓存
                    # 因为 CSV 回退需要 itemid filter，重新加载不带 filter 会失败
                    db_name_for_cache = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                    skip_cache_for_mimic3_chartevents = (db_name_for_cache == 'mimic' and source.table == 'chartevents')
                    
                    # 🚀 FIX 2026-02-09: 跳过分桶表/扁平parquet表的缓存重载！
                    # 原问题：对于分桶表(如AUMC numericitems_bucket)或扁平parquet目录，
                    # 缓存逻辑会重新加载整张表（不带itemid过滤），导致内存爆炸
                    # 例如 AUMC numericitems 5000 patients → ~2.7GB 仅为缓存
                    # 分桶/扁平parquet的单概念加载已经很快（DuckDB），每个概念独立读取
                    # 比缓存整张表更高效且内存友好
                    skip_cache_for_bucket_table = False
                    try:
                        _cache_skip_dir = data_source._resolve_bucket_directory(source.table)
                        if _cache_skip_dir is None:
                            _cache_skip_dir = data_source._resolve_flat_parquet_directory(source.table)
                        if _cache_skip_dir is not None:
                            skip_cache_for_bucket_table = True
                            if DEBUG_MODE and verbose:
                                print(f"   ⏭️  跳过分桶/扁平parquet表缓存: {source.table}")
                    except Exception:
                        pass
                    
                    if patient_filter_in_filters and not skip_cache_for_special_tables and not use_duckdb_aggregation and not skip_cache_for_mimic3_chartevents and not skip_cache_for_bucket_table:
                        # 🔧 FIX: 缓存只应用了患者过滤器的表（不包含 sub_var/itemid 过滤）
                        # 这样其他概念可以正确地从缓存中过滤出它们需要的 itemid
                        # 注意：需要确保加载的表包含所有可能需要的列（使用 value_var 参数）
                        # 对于像 vitalperiodic.sao2 这样的列，它们是通过 value_var 指定的
                        # 我们需要在缓存时确保这些列被加载
                        #
                        # 但是！我们不能缓存当前的 table.data，因为它已经应用了 itemid 过滤
                        # （例如 hr 加载时只有 itemid=220045 的数据）
                        # 所以我们需要重新加载不带 itemid 过滤的表
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
                        # 🔧 FIX 2026-01-26: patient_ids 可能是 dict 格式（如 {'icustay_id': [...]}）
                        # 需要支持 MIMIC-III icustay_id 和 MIMIC-IV stay_id
                        if isinstance(patient_ids, dict):
                            stay_ids = patient_ids.get('stay_id', []) or patient_ids.get('icustay_id', [])
                        else:
                            stay_ids = patient_ids
                        if stay_ids:
                            # 确定正确的 ID 列名（支持 MIMIC-III icustay_id 和 MIMIC-IV stay_id）
                            db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                            id_col = 'icustay_id' if db_name == 'mimic' else 'stay_id'
                            icustay_filters.append(
                                FilterSpec(column=id_col, op=FilterOp.IN, value=stay_ids)
                            )
                    
                    icustays = data_source.load_table('icustays', filters=icustay_filters if icustay_filters else None, verbose=False)
                    # 🔧 FIX 2026-01-26: 支持 MIMIC-III icustay_id 列名
                    db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                    stay_col = 'icustay_id' if db_name == 'mimic' else 'stay_id'
                    if hasattr(icustays, 'data'):
                        icu_df = icustays.data[['subject_id', stay_col]].drop_duplicates()
                    else:
                        icu_df = icustays[['subject_id', stay_col]].drop_duplicates()
                    
                    # 🔧 FIX: 将 icustay_id 重命名为 stay_id 以便后续处理统一
                    if stay_col == 'icustay_id' and 'icustay_id' in icu_df.columns:
                        icu_df = icu_df.rename(columns={'icustay_id': 'stay_id'})
                    
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
                if DEBUG_MODE: print(f"   🔍 调试 {source.table}: 'subject_id' in frame={('subject_id' in frame.columns)}, 'stay_id' in frame={('stay_id' in frame.columns)}, 'icustay_id' in frame={('icustay_id' in frame.columns)}, defaults.id_var={defaults.id_var}")
            
            # 🔧 FIX 2026-02-08: 同时支持 MIMIC-III (icustay_id) 和 MIMIC-IV (stay_id)
            has_stay_id = 'stay_id' in frame.columns or 'icustay_id' in frame.columns
            if source.table in ['labevents', 'microbiologyevents', 'inputevents'] and 'subject_id' in frame.columns and not has_stay_id:
                # 确定目标 ID 列名
                db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                target_id_col = 'icustay_id' if db_name == 'mimic' else 'stay_id'
                if DEBUG_MODE: print(f"   ➡️  进入 MIMIC 特殊处理: {source.table} (db={db_name}, target_id={target_id_col})")
                try:
                    # 仅加载相关stay的icustays，并携带intime/outtime用于窗口过滤
                    icustay_filters = []
                    # 保存expanded_patient_ids到当前作用域,避免后续locals()检查失效
                    # 🔧 FIX 2026-02-08: 初始化为空字典而非 None，确保后面可以正确填充
                    current_expanded_patient_ids = {}
                    
                    # 🔥 关键修复: 使用原始 stay_id/icustay_id 而不是 subject_id
                    # 这样避免加载同一患者的所有ICU入住记录
                    if patient_ids:
                        # 🔧 FIX 2026-01-26: patient_ids 可能是 dict 格式（如 {'icustay_id': [...]}）
                        # 需要提取列表形式的值，并确定正确的列名
                        if isinstance(patient_ids, dict):
                            # 确定正确的 ID 列名（支持 MIMIC-III icustay_id 和 MIMIC-IV stay_id）
                            id_col = 'stay_id'  # 默认 MIMIC-IV
                            id_vals = None
                            for key in ['stay_id', 'icustay_id', 'subject_id']:
                                if key in patient_ids and patient_ids[key]:
                                    id_col = key
                                    id_vals = patient_ids[key]
                                    break
                            if id_vals:
                                icustay_filters.append(
                                    FilterSpec(column=id_col, op=FilterOp.IN, value=id_vals)
                                )
                                # 🔧 FIX 2026-02-08: 保存 patient_ids 到 current_expanded_patient_ids 用于后续过滤
                                current_expanded_patient_ids = patient_ids.copy()
                                if DEBUG_MODE: print(f"   🎯 [icustays] 使用 {id_col} 过滤: {len(id_vals)} 个, IDs={id_vals[:5]}...")
                        else:
                            # 原有逻辑: patient_ids 是列表，使用目标 ID 列名
                            icustay_filters.append(
                                FilterSpec(column=target_id_col, op=FilterOp.IN, value=patient_ids)
                            )
                            # 🔧 FIX 2026-02-08: 保存 patient_ids 到 current_expanded_patient_ids 用于后续过滤
                            current_expanded_patient_ids = {target_id_col: patient_ids}
                            if DEBUG_MODE: print(f"   🎯 [icustays] 使用原始 {target_id_col} 过滤: {len(patient_ids)} 个, IDs={patient_ids}")
                    
                    icustays = data_source.load_table('icustays', filters=icustay_filters if icustay_filters else None, verbose=verbose)
                    if hasattr(icustays, 'data'):
                        # 包含hadm_id以便匹配同一住院的数据
                        # 🔧 FIX: 同时支持 stay_id 和 icustay_id
                        cols = ['subject_id', 'stay_id', 'icustay_id', 'hadm_id', 'intime', 'outtime']
                        icu_df = icustays.data[[c for c in cols if c in icustays.data.columns]].drop_duplicates()
                    else:
                        cols = ['subject_id', 'stay_id', 'icustay_id', 'hadm_id', 'intime', 'outtime']
                        icu_df = icustays[[c for c in cols if c in icustays.columns]].drop_duplicates()
                    
                    # 确定实际的 stay ID 列名
                    actual_stay_col = target_id_col if target_id_col in icu_df.columns else ('stay_id' if 'stay_id' in icu_df.columns else 'icustay_id')
                    if DEBUG_MODE: print(f"   ✅ [icustays] 加载后: {len(icu_df)} stays, {actual_stay_col}={sorted(icu_df[actual_stay_col].unique())[:10]}")
                    
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
                            for cand in ['charttime', 'storetime', 'specimen_time', 'startdate']:
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
                        # 当同一个 hadm_id/subject_id 有多个 stay_id/icustay_id 时，数据会被复制到所有匹配的 stay
                        # 需要根据时间将数据只保留在正确的 stay 下
                        # ricu 使用 roll = -Inf (向前滚动)：数据分配给时间之后最近的 stay
                        # 🔧 FIX 2026-02-08: patient_ids 可能是 dict（如 {'icustay_id': [...]}），需要提取实际的 ID 值
                        target_stay_ids = None
                        if patient_ids:
                            if isinstance(patient_ids, dict):
                                # 从 dict 中提取实际的 ID 值
                                for key in ['stay_id', 'icustay_id', 'subject_id']:
                                    if key in patient_ids and patient_ids[key]:
                                        target_stay_ids = set(patient_ids[key])
                                        break
                            else:
                                target_stay_ids = set(patient_ids)
                        
                        # 确定合并后的 stay ID 列名
                        merged_stay_col = actual_stay_col if actual_stay_col in tmp.columns else ('stay_id' if 'stay_id' in tmp.columns else 'icustay_id' if 'icustay_id' in tmp.columns else None)
                        
                        if time_col is not None and merged_stay_col and merged_stay_col in tmp.columns and 'intime' in tmp.columns and len(tmp) > 0:
                            # 获取所有唯一的 stay_id 及其 intime，按 intime 排序
                            stay_info = tmp[[merged_stay_col, 'intime']].drop_duplicates().sort_values('intime')
                            
                            if len(stay_info) > 1:
                                # 有多个 stay_id，需要实现 rolling join
                                stays_list = stay_info[merged_stay_col].tolist()
                                intimes_list = stay_info['intime'].tolist()
                                
                                if DEBUG_MODE:
                                    print(f"      🔄 [Rolling Join] 检测到多个 {merged_stay_col}: {stays_list}")
                                    print(f"      🔄 [Rolling Join] 对应 intime: {intimes_list}")
                                    print(f"      🔄 [Rolling Join] 目标 {merged_stay_col}: {target_stay_ids}")
                                
                                # 为每个 stay 计算其有效时间范围
                                # stay_i 的有效范围是: [prev_stay_outtime, next_stay_intime)
                                # 但使用 roll = -Inf 意味着：data_time < next_stay_intime
                                
                                result_frames = []
                                for i, (stay_id, intime) in enumerate(zip(stays_list, intimes_list)):
                                    # 只处理用户请求的 stay_id
                                    if target_stay_ids and stay_id not in target_stay_ids:
                                        continue
                                    
                                    # 过滤属于当前 stay_id 的行
                                    stay_mask = tmp[merged_stay_col] == stay_id
                                    
                                    if i < len(stays_list) - 1:
                                        # 不是最后一个 stay，数据时间必须小于下一个 stay 的 intime
                                        next_intime = intimes_list[i + 1]
                                        time_mask = tmp[time_col] < next_intime
                                        stay_data = tmp[stay_mask & time_mask].copy()
                                        if DEBUG_MODE:
                                            print(f"      🔄 [Rolling Join] {merged_stay_col}={stay_id}: time < {next_intime}, 保留 {len(stay_data)} 行")
                                    else:
                                        # 最后一个 stay，没有时间上限
                                        stay_data = tmp[stay_mask].copy()
                                        if DEBUG_MODE:
                                            print(f"      🔄 [Rolling Join] {merged_stay_col}={stay_id}: 最后一个stay, 保留 {len(stay_data)} 行")
                                    
                                    result_frames.append(stay_data)
                                
                                if result_frames:
                                    tmp = pd.concat(result_frames, ignore_index=True)
                                    if DEBUG_MODE:
                                        print(f"      🔄 [Rolling Join] 多 {merged_stay_col} 时间过滤完成: {len(tmp)} 行")
                        
                        # 确保只保留用户请求的 stay_id/icustay_id（防止遗漏过滤）
                        if target_stay_ids and merged_stay_col and merged_stay_col in tmp.columns:
                            before_filter = len(tmp)
                            tmp = tmp[tmp[merged_stay_col].isin(target_stay_ids)]
                            if DEBUG_MODE and len(tmp) != before_filter:
                                print(f"      🎯 [最终过滤] 只保留目标 {merged_stay_col}: {before_filter} → {len(tmp)} 行")

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
                                print("      🏥 [ICU窗口] ❌ tmp不包含outtime列!")

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
                            # 🔧 FIX: Convert datetime columns to relative hours BEFORE removing intime
                            # This matches R ricu's load_mihi which converts times relative to origin
                            # before any callbacks (like calc_dur) are called.
                            # 
                            # R ricu flow: dt_round_min <- function(x, y) round_to(difftime(x, y, units = "mins"))
                            # This floors all time columns to integer minutes, then later to hours.
                            # 
                            # For duration calculation, we need: floor(end_h) - floor(start_h)
                            if 'intime' in tmp.columns and tmp['intime'].notna().any():
                                # Convert all datetime time columns to relative hours
                                datetime_cols = []
                                for col in tmp.columns:
                                    if col in ['starttime', 'endtime', 'charttime', 'storetime', 'startdate', 'enddate'] and col != 'intime':
                                        if pd.api.types.is_datetime64_any_dtype(tmp[col]):
                                            datetime_cols.append(col)
                                
                                if datetime_cols:
                                    for col in datetime_cols:
                                        # Convert to relative hours (from intime)
                                        # 🔧 FIX: Match R ricu's round_to(difftime(x, y, units = "mins")) logic:
                                        # 1. First floor to integer minutes (dt_round_min)
                                        # 2. Then floor to integer hours when aggregating
                                        # This ensures times slightly before intime go to hour -1, not 0
                                        relative_td = tmp[col] - tmp['intime']
                                        # Step 1: floor to integer minutes (matching R's floor(difftime(..., units="mins")))
                                        relative_mins = np.floor(relative_td.dt.total_seconds() / 60.0)
                                        # Step 2: convert to hours (will be floored during aggregation)
                                        tmp[col] = relative_mins / 60.0
                                    if DEBUG_MODE:
                                        print(f"      🕐 [时间转换] {datetime_cols} 从 datetime → 相对小时数 (floor to mins first)")
                            
                            # 将过滤后的数据作为新frame，仅保留必要列
                            drop_cols = [c for c in ['intime', 'outtime'] if c in tmp.columns]
                            frame = tmp.drop(columns=drop_cols)
                            if DEBUG_MODE: print(f"   ✅ [{concept_name}] MIMIC {source.table}: 合并+过滤后 {len(frame)} 行")
                        else:
                            # tmp为空的原因可能是：1) 没有匹配的住院数据，2) 时间过滤后为空
                            # 这是正常的数据过滤行为（例如实验室结果在ICU出院后采集，或在miiv中是ICU入院前的数据）
                            if DEBUG_MODE:
                                reason = "ricu.R-style时间过滤" if before_filter > 0 else "ICU住院匹配"
                                print(f"   ⚠️  [{concept_name}] MIMIC {source.table}: {reason}后为空 (原始{len(frame)}行 → 匹配{before_filter}行 → 过滤后0行)")
                            frame = pd.DataFrame(columns=frame.columns)
                            
                        # 🔗 关键修复：如果用户提供了特定的 stay_id/icustay_id，在映射后再次过滤
                        # 确保只返回用户指定的 stay 的数据
                        final_stay_col = merged_stay_col if merged_stay_col and merged_stay_col in frame.columns else None
                        if final_stay_col and patient_ids:
                            # 使用之前保存的current_expanded_patient_ids
                            if current_expanded_patient_ids and isinstance(current_expanded_patient_ids, dict):
                                # 尝试获取 stay_id 或 icustay_id
                                specified_stay_ids = current_expanded_patient_ids.get(final_stay_col) or current_expanded_patient_ids.get('stay_id') or current_expanded_patient_ids.get('icustay_id')
                                if specified_stay_ids:
                                    before_stay_filter = len(frame)
                                    frame = frame[frame[final_stay_col].isin(specified_stay_ids)].copy()
                                    if DEBUG_MODE and before_stay_filter > len(frame):
                                        print(f"      🔍 [{concept_name}] {final_stay_col}过滤: {before_stay_filter}行 → {len(frame)}行 (保留{len(specified_stay_ids)}个{final_stay_col})")
                        
                        if defaults.id_var == 'subject_id' and final_stay_col and final_stay_col in frame.columns:
                                id_columns = [final_stay_col]
                                if DEBUG_MODE: print(f"   🔄 MIMIC特殊处理: {source.table} ID列从 subject_id → {final_stay_col} (行数: {len(frame)})")
                    else:
                        # 没有明确时间列，退化为subject级合并（可能产生冗余），但仍补充stay_id/icustay_id
                        merge_cols = ['subject_id']
                        if actual_stay_col in icu_df.columns:
                            merge_cols.append(actual_stay_col)
                            frame = frame.merge(icu_df[merge_cols], on='subject_id', how='inner')
                            if defaults.id_var == 'subject_id':
                                id_columns = [actual_stay_col]
                                if DEBUG_MODE: print(f"   🔄 MIMIC特殊处理(无时间列): {source.table} ID列从 subject_id → {actual_stay_col} (行数: {len(frame)})")
                except Exception as ex:
                    print(f"⚠️  Warning: Failed to time-map labevents to icu stays: {ex}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    # 失败时不做强制映射，保持原逻辑
            
            # MIMIC-IV特殊处理：admissions表只有subject_id和hadm_id，需要映射到stay_id
            if source.table == 'admissions' and 'subject_id' in frame.columns and 'stay_id' not in frame.columns:
                if DEBUG_MODE: print("   ➡️  进入 MIMIC-IV admissions特殊处理")
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
                    
                    # 🔧 FIX 2026-02: 支持 MIMIC-III (icustay_id) 和 MIMIC-IV (stay_id)
                    db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                    target_id_col = 'icustay_id' if db_name == 'mimic' else 'stay_id'
                    icu_cols = ['subject_id', 'hadm_id', target_id_col, 'intime']
                    icu_cols = [c for c in icu_cols if c in (icustays.data.columns if hasattr(icustays, 'data') else icustays.columns)]
                    
                    if hasattr(icustays, 'data'):
                        icu_df = icustays.data[icu_cols].drop_duplicates()
                    else:
                        icu_df = icustays[icu_cols].drop_duplicates()
                    
                    # 重命名为统一的 stay_id 以便后续代码使用
                    if target_id_col != 'stay_id' and target_id_col in icu_df.columns:
                        icu_df = icu_df.rename(columns={target_id_col: 'stay_id'})
                    
                    # 🔧 FIX: 实现 ricu 的 rolling join 逻辑用于 death 概念
                    # 当同一个 hadm_id 有多个 ICU stay 时，death 应该分配给最近的 stay
                    # ricu 使用 roll = -Inf (向前滚动): 找第一个 intime >= data_time 的 stay
                    # 如果没有找到，分配给最后一个 stay
                    
                    # 通过hadm_id映射到stay_id（admissions是hospital级别，icustays是ICU级别）
                    if 'hadm_id' in frame.columns and 'hadm_id' in icu_df.columns:
                        before_merge = len(frame)
                        
                        # 获取用户请求的 stay_id 列表（用于最终过滤）
                        specified_stay_ids = None
                        if patient_ids and current_expanded_patient_ids and isinstance(current_expanded_patient_ids, dict) and 'stay_id' in current_expanded_patient_ids:
                            specified_stay_ids = set(current_expanded_patient_ids['stay_id'])
                        
                        # 🚀 性能优化：使用向量化操作替代逐行循环
                        # 检查是否需要 rolling join（同一 hadm_id 有多个 stay）
                        stays_per_hadm = icu_df.groupby('hadm_id')['stay_id'].count()
                        multi_stay_hadms = set(stays_per_hadm[stays_per_hadm > 1].index.tolist())
                        single_stay_hadms = set(stays_per_hadm[stays_per_hadm == 1].index.tolist())
                        
                        # 获取时间列: 使用 source.index_var（配置中的 index_var，如 deathtime）
                        time_col_for_rolling = source.index_var
                        
                        # 🚀 优化策略：
                        # 1. 单 stay 的 hadm_id 使用向量化 merge（大多数情况）
                        # 2. 多 stay 的 hadm_id 使用 merge_asof 进行 rolling join
                        
                        result_frames = []
                        
                        # 步骤1: 处理单 stay 的 hadm_id（向量化 merge，非常快）
                        single_stay_mask = frame['hadm_id'].isin(single_stay_hadms)
                        if single_stay_mask.any():
                            single_stay_frame = frame[single_stay_mask].copy()
                            single_stay_icu = icu_df[icu_df['hadm_id'].isin(single_stay_hadms)][['hadm_id', 'stay_id']]
                            merged_single = single_stay_frame.merge(single_stay_icu, on='hadm_id', how='inner')
                            if not merged_single.empty:
                                result_frames.append(merged_single)
                        
                        # 步骤2: 处理多 stay 的 hadm_id（使用 merge_asof 或简化逻辑）
                        if multi_stay_hadms and time_col_for_rolling and time_col_for_rolling in frame.columns:
                            multi_stay_mask = frame['hadm_id'].isin(multi_stay_hadms)
                            if multi_stay_mask.any():
                                multi_stay_frame = frame[multi_stay_mask].copy()
                                multi_stay_icu = icu_df[icu_df['hadm_id'].isin(multi_stay_hadms)].copy()
                                
                                # 确保时间列是 datetime 类型
                                multi_stay_frame[time_col_for_rolling] = pd.to_datetime(multi_stay_frame[time_col_for_rolling], errors='coerce')
                                multi_stay_icu['intime'] = pd.to_datetime(multi_stay_icu['intime'], errors='coerce')
                                
                                # 过滤掉无效时间
                                valid_time_mask = multi_stay_frame[time_col_for_rolling].notna()
                                multi_stay_frame = multi_stay_frame[valid_time_mask].copy()
                                
                                if not multi_stay_frame.empty:
                                    # 使用 merge_asof 进行 rolling join（向量化操作）
                                    # 排序要求
                                    multi_stay_frame = multi_stay_frame.sort_values(time_col_for_rolling)
                                    multi_stay_icu = multi_stay_icu.sort_values('intime')
                                    
                                    # merge_asof: direction='forward' 找第一个 >= 的值
                                    # 这对应 ricu 的 roll = -Inf 行为
                                    merged_multi = pd.merge_asof(
                                        multi_stay_frame,
                                        multi_stay_icu[['hadm_id', 'stay_id', 'intime']],
                                        left_on=time_col_for_rolling,
                                        right_on='intime',
                                        by='hadm_id',
                                        direction='forward'
                                    )
                                    
                                    # 对于没有匹配到的（时间在所有 stay 之后），分配给最后一个 stay
                                    no_match = merged_multi['stay_id'].isna()
                                    if no_match.any():
                                        # 获取每个 hadm_id 的最后一个 stay
                                        last_stays = multi_stay_icu.groupby('hadm_id').last().reset_index()[['hadm_id', 'stay_id']]
                                        last_stays = last_stays.rename(columns={'stay_id': 'last_stay_id'})
                                        merged_multi = merged_multi.merge(last_stays, on='hadm_id', how='left')
                                        merged_multi.loc[no_match, 'stay_id'] = merged_multi.loc[no_match, 'last_stay_id']
                                        merged_multi = merged_multi.drop(columns=['last_stay_id'], errors='ignore')
                                    
                                    # 删除临时列
                                    merged_multi = merged_multi.drop(columns=['intime'], errors='ignore')
                                    
                                    if not merged_multi.empty:
                                        result_frames.append(merged_multi)
                        elif multi_stay_hadms:
                            # 没有时间列，对多 stay 使用第一个 stay
                            multi_stay_mask = frame['hadm_id'].isin(multi_stay_hadms)
                            if multi_stay_mask.any():
                                multi_stay_frame = frame[multi_stay_mask].copy()
                                # 获取每个 hadm_id 的第一个 stay
                                first_stays = icu_df[icu_df['hadm_id'].isin(multi_stay_hadms)].groupby('hadm_id').first().reset_index()[['hadm_id', 'stay_id']]
                                merged_multi = multi_stay_frame.merge(first_stays, on='hadm_id', how='inner')
                                if not merged_multi.empty:
                                    result_frames.append(merged_multi)
                        
                        # 合并所有结果
                        if result_frames:
                            frame = pd.concat(result_frames, ignore_index=True)
                        else:
                            frame = pd.DataFrame(columns=list(frame.columns) + ['stay_id'])
                        
                        if DEBUG_MODE:
                            print(f"      🏥 [{concept_name}] admissions→icustays映射: {before_merge}行 → {len(frame)}行")
                        
                        # 最终stay_id过滤（如果还没有在 rolling join 中过滤）
                        if specified_stay_ids and 'stay_id' in frame.columns:
                            before_stay_filter = len(frame)
                            frame = frame[frame['stay_id'].isin(specified_stay_ids)].copy()
                            if DEBUG_MODE and before_stay_filter > len(frame):
                                print(f"      🔍 [{concept_name}] stay_id过滤: {before_stay_filter}行 → {len(frame)}行")
                        
                        # 🔧 FIX 2026-02: MIMIC-III 需要将 stay_id 重命名回 icustay_id
                        if target_id_col == 'icustay_id' and 'stay_id' in frame.columns:
                            frame = frame.rename(columns={'stay_id': 'icustay_id'})
                            id_columns = ['icustay_id']
                            if DEBUG_MODE: print("   🔄 MIMIC-III特殊处理: admissions ID列从 subject_id → icustay_id")
                        elif defaults.id_var == 'subject_id' and 'stay_id' in frame.columns:
                            id_columns = ['stay_id']
                            if DEBUG_MODE: print("   🔄 MIMIC-IV特殊处理: admissions ID列从 subject_id → stay_id")
                except Exception as ex:
                    print(f"⚠️  Warning: Failed to map admissions to icu stays: {ex}")
                    if verbose:
                        import traceback
                        traceback.print_exc()

            # 如果配置中没有ID列，尝试从数据中自动检测
            # 🔧 FIX 2026-02-05: 对于 MIMIC-III，始终优先使用 icustay_id（如果存在）
            db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
            
            # 检查是否需要覆盖表配置的 id_columns
            should_detect_id = not table.id_columns
            
            # 🔧 MIMIC-III 特殊处理：labevents 配置了 hadm_id，但实际需要用 icustay_id
            if db_name == 'mimic' and 'icustay_id' in frame.columns:
                # 如果数据中有 icustay_id，强制使用它
                should_detect_id = True
            
            if should_detect_id:
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
                elif db_name == 'mimic':
                    # 🔧 FIX 2026-02-05: MIMIC-III 使用 icustay_id（不是 stay_id）
                    common_id_cols = ['icustay_id', 'hadm_id', 'subject_id']
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
            
            # 🔧 FIX 2025-02: 只有当frame有数据时才更新全局的index_column
            # 问题：多源概念中，第一个源（DuckDB聚合）可能返回0行但设置了index_column='measuredat_minutes'
            # 第二个源（callback）返回有效数据，时间列是'measuredat'
            # 如果总是用第一个源的index_column，合并后数据会因为该列全是NaN而被丢弃
            # 解决：只有当源返回非空数据时，才更新全局index_column
            if not index_column and len(frame) > 0:
                index_column = source_index_column
            if not unit_column and len(frame) > 0:
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
            # 🔧 CRITICAL FIX 2025-02-14: R ricu uses 'dur_var' as the column name, not 'concept_name_dur'
            # And the value is in MINUTES for MIIV (not timedelta or hours)
            if source.dur_var and source.dur_var in frame.columns:
                if source_index_column and source_index_column in frame.columns:
                    # R ricu always uses 'dur_var' as the output column name
                    duration_col = 'dur_var'
                    dur_is_end = False  # 是否需要计算 duration = endtime - starttime
                    
                    # Case 1: datetime 类型的 endtime
                    if pd.api.types.is_datetime64_any_dtype(frame[source.dur_var]):
                        dur_is_end = True
                        # 确保 starttime 也是 datetime
                        if not pd.api.types.is_datetime64_any_dtype(frame[source_index_column]):
                            frame[source_index_column] = pd.to_datetime(frame[source_index_column], errors='coerce')
                        
                        # 计算 duration (timedelta) 然后转为分钟（匹配 R ricu）
                        frame[duration_col] = (frame[source.dur_var] - frame[source_index_column]).dt.total_seconds() / 60
                    
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
                                    # 🔧 FIX 2025-02-14: R ricu keeps duration in MINUTES, NOT hours
                                    # AUMC: 分钟（datasource.py 已将 ms 转为分钟）
                                    # eICU: 分钟（offset 列本身就是分钟）
                                    frame[duration_col] = frame[source.dur_var] - frame[source_index_column]
                                    
                                    if DEBUG_MODE:
                                        db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                                        print(f"   🔧 {db_name} dur_is_end=True: {source.dur_var}={dur_vals.head(3).tolist()}, "
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
            # 🚀 如果 convert_unit 已在 DuckDB 后置处理中应用，标记为已完成
            callback_applied = _convert_unit_callback_for_duckdb
            
            # 🚀 id_tbl DuckDB 路径：value_column 已经是 concept_name（DuckDB SELECT AS）
            if _idtbl_done and concept_name in frame.columns:
                value_column = concept_name
                callback_applied = True
            
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
                    interval=interval,
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
                    interval=interval,
                )

            # 🔧 FIX: After callback, if source_index_column was consumed/renamed by
            # the callback (e.g. sic_death renames OffsetOfDeath→death and adds charttime),
            # update source_index_column to the new time column so it appears in ordered_cols.
            if (source_index_column and
                    source_index_column not in frame.columns and
                    len(frame) > 0):
                for fallback_time in ['charttime', 'starttime', 'datetime']:
                    if fallback_time in frame.columns:
                        source_index_column = fallback_time
                        break
            
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
                
                # 🔧 CRITICAL: 与 R ricu 保持一致，只报告单位警告，不过滤数据
                # R ricu 的 report_set_unit() 函数只打印警告消息:
                #   "not all units are in [expected]: actual_units"
                # 它不会删除数据，只是记录警告信息
                # 
                # 之前的实现对非 AUMC 数据库应用严格单位过滤，但这会导致:
                # - MIMIC-III resp 概念丢失 itemid 618/619 数据（使用 BPM 单位而非 insp/min）
                # - 其他单位变体的数据丢失
                # 
                # 为与 R ricu 完全一致，现在对所有数据库都只报告警告，不过滤
                db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                
                # 🔧 只报告警告，不过滤数据（与 R ricu 一致）
                unique_units = frame[source_unit_column].unique()
                unique_units_str = {str(u).strip().lower() for u in unique_units if pd.notna(u)}
                mismatched = unique_units_str - allowed_units
                if mismatched and (DEBUG_MODE or len(frame) > 0):
                    # 只在调试模式或有数据时记录
                    if DEBUG_MODE:
                        print(f"   ⚠️ 单位警告 (允许{definition.units}): 发现不匹配单位 {mismatched}")
                    # 发出Python警告供日志记录
                    import warnings
                    warnings.warn(
                        f"概念 '{concept_name}': 不是所有单位都在允许列表中 {definition.units}, "
                        f"发现: {mismatched}",
                        UserWarning
                    )

            db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
            if (
                db_name in ['mimic', 'mimic_demo']
                and source_index_column
                and source_index_column in frame.columns
                and pd.api.types.is_datetime64_any_dtype(frame[source_index_column])
                and 'icustay_id' in frame.columns
                and len(frame) > 0
            ):
                try:
                    icustays_table = data_source.load_table(
                        'icustays',
                        columns=['icustay_id', 'intime'],
                        verbose=False,
                    )
                    icustays_df = icustays_table.data if hasattr(icustays_table, 'data') else icustays_table
                    if 'intime' in icustays_df.columns:
                        intime_map = icustays_df[['icustay_id', 'intime']].drop_duplicates().copy()
                        intime_map['intime'] = pd.to_datetime(intime_map['intime'], errors='coerce')
                        if intime_map['intime'].dt.tz is not None:
                            intime_map['intime'] = intime_map['intime'].dt.tz_localize(None)

                        frame = frame.merge(intime_map, on='icustay_id', how='left')
                        time_values = pd.to_datetime(frame[source_index_column], errors='coerce')
                        if hasattr(time_values.dt, 'tz') and time_values.dt.tz is not None:
                            time_values = time_values.dt.tz_localize(None)
                        rel_minutes = np.floor((time_values - frame['intime']).dt.total_seconds() / 60.0)
                        frame[source_index_column] = rel_minutes / 60.0
                        frame = frame.drop(columns=['intime'], errors='ignore')
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"   ⚠️ [MIMIC-III] source级时间标准化失败: {e}")
                
                # 🔧 以下是被禁用的严格过滤逻辑（保留作为参考）
                skip_unit_filter = True  # 与 R ricu 一致，不过滤数据
                
                if not skip_unit_filter:
                    # 非 AUMC 数据库：应用严格单位过滤
                    # 🚀 快速路径：先检查唯一值，如果所有唯一值都在允许列表中，跳过昂贵的字符串操作
                    unique_units = frame[source_unit_column].unique()
                    # 转换为小写字符串集合
                    unique_units_lower = {str(u).strip().lower() for u in unique_units if pd.notna(u)}
                    
                    # 检查是否所有唯一单位都在允许列表中
                    if unique_units_lower.issubset(allowed_units):
                        # 快速路径：所有单位都匹配，无需逐行处理
                        if DEBUG_MODE:
                            print(f"   ⚡ 快速单位过滤: 所有 {len(unique_units)} 个唯一值都在允许列表中")
                    else:
                        # 慢速路径：需要逐行处理
                        # 归一化数据中的单位
                        series = frame[source_unit_column].astype(str).str.strip()
                        normalized_series = series.replace(unit_equivalents).str.lower()
                
                        # 🔧 进一步归一化：去除非字母数字字符后比较
                        # 这处理了 mmHg 的各种变体：mm Hg, mm/Hg, mm(hg), mm[Hg] 等
                        # 🚀 性能优化: 使用向量化 .str.replace() 而非 .apply() + re.sub()
                        # 原代码对 875万行数据逐行调用正则表达式，耗时 18 秒
                        # 优化后使用 pandas 内置的向量化字符串操作，快 10 倍以上
                        def normalize_unit_for_comparison(unit_str):
                            """归一化单位字符串，仅保留字母数字字符（用于小集合）"""
                            if not unit_str or pd.isna(unit_str) or unit_str in ['', 'none', 'None', 'nan']:
                                return ''
                            return re.sub(r'[^a-z0-9]', '', str(unit_str).lower())
                        
                        normalized_allowed = {normalize_unit_for_comparison(u) for u in definition.units}
                        # 🚀 向量化归一化：替换非字母数字字符为空
                        normalized_data = normalized_series.str.replace(r'[^a-z0-9]', '', regex=True)

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

            # 🔧 FIX 2026-02-15: filter_bounds (min/max) 移到 change_interval 之后
            # R ricu 的 load_concepts.num_cncpt 流程:
            #   load_concepts(as_item(x)) [含 change_interval] -> filter_bounds -> aggregate
            # 之前错误地在 per-source 循环内执行 filter_bounds，导致先过滤再聚合
            # 现在只在此处删除 NaN，filter_bounds 延迟到 change_interval 之后执行
            
            # 删除无效的NaN
            if concept_name in frame.columns:
                # 检查是否在merge模式（通过kwargs传递）
                keep_na_rows = kwargs.get('_keep_na_rows', False)
                if not keep_na_rows:
                    _before_dropna = len(frame)
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
                time_aliases = {"starttime", "endtime", "charttime", "storetime", "startdate", "enddate"}
                time_cols = []
                for col in frame.columns:
                    if not isinstance(col, str):
                        continue
                    lowered = col.lower()
                    if "time" in lowered or "date" in lowered or lowered in time_aliases:
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
            
            # 🔧 FIX 2025-02-14: R ricu uses 'dur_var' as the duration column name
            # Add 'dur_var' before concept_name + '_dur' for win_tbl concepts
            if 'dur_var' in frame.columns and 'dur_var' not in ordered_cols:
                ordered_cols.append('dur_var')
            
            # Legacy: 添加 duration 列（如果存在）
            duration_col_name = concept_name + '_dur'
            if duration_col_name in frame.columns and duration_col_name not in ordered_cols:
                ordered_cols.append(duration_col_name)
            
            # 🔧 FIX: 保留 endtime/stoptime 列用于窗口概念展开
            # mech_vent 等窗口概念需要 endtime 来进行时间展开
            # prescriptions 表使用 stoptime 作为结束时间
            # 如果有 dur_var="endtime" 的定义，endtime 列必须保留
            for endtime_candidate in ['endtime', 'end_time', 'stop', 'stoptime']:
                if endtime_candidate in frame.columns and endtime_candidate not in ordered_cols:
                    ordered_cols.append(endtime_candidate)
                    # 不要 break，保留所有存在的结束时间列
            
            ordered_cols = [col for col in ordered_cols if col in frame.columns]
            
            # DEBUG: 检查dur_var是否被保留
            if DEBUG_MODE and 'dur_var' in frame.columns:
                print(f"   🔍 DEBUG: frame有dur_var列, ordered_cols中是否有dur_var: {'dur_var' in ordered_cols}")
                print(f"   🔍 DEBUG: ordered_cols = {ordered_cols}")
            
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
            
            # DEBUG: 检查combined是否有dur_var
            if DEBUG_MODE and 'dur_var' in combined.columns:
                print(f"   🔍 DEBUG: combined有dur_var列, 行数={len(combined)}")
        
        # DEBUG
        # Standardize time column name for eICU BEFORE any processing
        # eICU uses different time column names (labresultoffset, observationoffset, etc.)
        # For multi-source concepts (like abx), different sources may use different offset columns
        # Rename all offset columns to 'charttime' to enable unified processing
        db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
        if db_name in ['eicu', 'eicu_demo']:
            # All possible eICU time offset columns
            eicu_time_cols = [
                'labresultoffset', 'observationoffset', 'nursingchartoffset', 
                'respiratorycharting_offset', 'intakeoutput_offset', 'respchartoffset',
                'infusionoffset', 'drugstartoffset', 'drugstopoffset', 'drugorderoffset',
                'culturetakenoffset', 'cultureoffset',
                # 🔥 添加 respiratorycare 表的时间列
                'respcarestatusoffset', 'ventstartoffset', 'ventendoffset',
                'priorventstartoffset', 'priorventendoffset',
                # 🔥 添加 treatment 和 nursecharting 表的时间列
                'treatmentoffset', 'nursingchartentryoffset',
            ]
            
            # 🔧 CRITICAL FIX: 查找有效的时间列（非全NaN）
            # 当多个源合并时，如果第一个源是空的，index_column 可能指向一个全是 NaN 的列
            # 需要找到第一个有数据的时间列
            offset_cols_in_data = [col for col in combined.columns if col in eicu_time_cols]
            
            # 对offset列按有效数据量排序，优先使用数据更多的列
            def count_valid(col):
                if col in combined.columns:
                    return combined[col].notna().sum()
                return 0
            
            offset_cols_in_data = sorted(offset_cols_in_data, key=count_valid, reverse=True)
            
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
        
        # 🔧 CRITICAL FIX 2026-02-05: AUMC 多源时间列统一
        # 当 DuckDB 聚合路径返回 measuredat_minutes，原始加载路径返回 measuredat 时，
        # 需要统一成一个时间列 (charttime)，否则 change_interval 会丢失数据
        elif db_name == 'aumc':
            aumc_time_cols = ['measuredat_minutes', 'measuredat', 'givenat', 'start', 'starttime']
            time_cols_in_data = [col for col in combined.columns if col in aumc_time_cols]
            
            if len(time_cols_in_data) > 1:
                if DEBUG_MODE:
                    print(f"   🔧 [AUMC] 检测到多个时间列: {time_cols_in_data}, 需要统一")
                
                # 按有效数据量排序
                def count_valid_aumc(col):
                    return combined[col].notna().sum() if col in combined.columns else 0
                
                time_cols_in_data = sorted(time_cols_in_data, key=count_valid_aumc, reverse=True)
                
                # 统一时间列为 'charttime'
                # 所有时间列到达此处时都已经是分钟单位:
                # - measuredat_minutes: DuckDB聚合路径直接输出分钟
                # - measuredat: datasource层已将ms转换为分钟
                # 不需要额外的单位转换
                
                # 2. 合并所有时间列到 charttime
                combined['charttime'] = combined[time_cols_in_data[0]]
                for col in time_cols_in_data[1:]:
                    if col in combined.columns:
                        combined['charttime'] = combined['charttime'].fillna(combined[col])
                        combined = combined.drop(columns=[col])
                
                # 删除第一个时间列（如果不同于 charttime）
                if time_cols_in_data[0] != 'charttime' and time_cols_in_data[0] in combined.columns:
                    combined = combined.drop(columns=[time_cols_in_data[0]])
                
                index_column = 'charttime'
                
                if DEBUG_MODE:
                    print(f"   🔧 [AUMC] 时间列统一完成, charttime 有效值: {combined['charttime'].notna().sum()}/{len(combined)}")
        
        # 🔧 CRITICAL FIX 2026-02-09: MIMIC-III 多源时间列统一
        # MIMIC-III 的 inputevents_cv 使用 charttime，inputevents_mv 使用 starttime
        # 当两个源合并时，需要将 starttime 统一到 charttime
        # 
        # 🔧 2026-02-10 扩展修复：处理类型不一致问题
        # 问题：CV 源的 charttime 是 datetime 格式，MV 源经过 distribute_amount callback 后
        #      starttime 变成了 float（相对小时数），合并后 fillna 无法正确处理
        # 解决：检测并统一时间列类型，将 datetime 转换为相对小时数（与 callback 输出一致）
        elif db_name in ['mimic', 'mimic_demo']:
            mimic_time_cols = ['charttime', 'starttime', 'storetime', 'admittime', 'startdate', 'enddate']
            time_cols_in_data = [col for col in combined.columns if col in mimic_time_cols]
            
            if len(time_cols_in_data) > 1:
                if DEBUG_MODE:
                    print(f"   🔧 [MIMIC-III] 检测到多个时间列: {time_cols_in_data}, 需要统一")
                
                # 🔧 检测时间列类型不一致的问题
                # 如果一个列是 float（callback 返回的相对时间），另一个是 datetime（原始时间）
                # 需要将 datetime 也转换为相对时间
                col_types = {}
                for col in time_cols_in_data:
                    if col in combined.columns:
                        col_dtype = combined[col].dtype
                        if pd.api.types.is_numeric_dtype(col_dtype):
                            col_types[col] = 'numeric'
                        elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                            col_types[col] = 'datetime'
                        else:
                            col_types[col] = 'other'
                
                # 如果存在类型不一致，需要统一为 numeric（相对时间）
                has_numeric = 'numeric' in col_types.values()
                has_datetime = 'datetime' in col_types.values()
                
                if has_numeric and has_datetime:
                    if DEBUG_MODE:
                        print(f"   🔧 [MIMIC-III] 时间列类型不一致: {col_types}, 需要转换 datetime → numeric")
                    
                    # 获取 icustays 表的 intime 用于计算相对时间
                    try:
                        icustays_table = data_source.load_table('icustays', columns=['icustay_id', 'intime'], verbose=False)
                        icustays_df = icustays_table.data if hasattr(icustays_table, 'data') else icustays_table
                        
                        if 'intime' in icustays_df.columns:
                            # 确保 intime 是 datetime 类型
                            icustays_df['intime'] = pd.to_datetime(icustays_df['intime'], errors='coerce')
                            if icustays_df['intime'].dt.tz is not None:
                                icustays_df['intime'] = icustays_df['intime'].dt.tz_localize(None)
                            
                            # 合并 intime
                            if 'icustay_id' in combined.columns:
                                combined = combined.merge(
                                    icustays_df[['icustay_id', 'intime']].drop_duplicates(),
                                    on='icustay_id', how='left'
                                )
                                
                                # 转换所有 datetime 时间列为相对小时数
                                for col, ctype in col_types.items():
                                    if ctype == 'datetime' and col in combined.columns:
                                        combined[col] = pd.to_datetime(combined[col], errors='coerce')
                                        if combined[col].dt.tz is not None:
                                            combined[col] = combined[col].dt.tz_localize(None)
                                        # 计算相对小时数
                                        combined[col] = (combined[col] - combined['intime']).dt.total_seconds() / 3600.0
                                        if DEBUG_MODE:
                                            print(f"   🔧 [MIMIC-III] 已将 {col} 从 datetime 转换为相对小时数")
                                
                                # 删除临时的 intime 列
                                if 'intime' in combined.columns:
                                    combined = combined.drop(columns=['intime'])
                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"   ⚠️ [MIMIC-III] datetime→numeric 转换失败: {e}")
                
                # 优先使用 charttime，如果不存在则创建
                if 'charttime' not in combined.columns:
                    # 按有效数据量排序，找到最佳时间列
                    def count_valid_mimic(col):
                        return combined[col].notna().sum() if col in combined.columns else 0
                    
                    time_cols_in_data = sorted(time_cols_in_data, key=count_valid_mimic, reverse=True)
                    # 将第一个（最有数据的）时间列重命名为 charttime
                    combined = combined.rename(columns={time_cols_in_data[0]: 'charttime'})
                    time_cols_in_data = time_cols_in_data[1:]
                
                # 合并其他时间列到 charttime（使用第一个非NaN值）
                for col in time_cols_in_data:
                    if col in combined.columns and col != 'charttime':
                        combined['charttime'] = combined['charttime'].fillna(combined[col])
                        combined = combined.drop(columns=[col])
                
                index_column = 'charttime'
                
                if DEBUG_MODE:
                    print(f"   🔧 [MIMIC-III] 时间列统一完成, charttime 有效值: {combined['charttime'].notna().sum()}/{len(combined)}")

            # Multi-source concat may leave charttime as object with a mix of
            # relative-hour numerics and absolute datetimes. Converting the whole
            # object column with pd.to_datetime() would reinterpret numeric hours
            # as Unix epoch timestamps, producing huge negative relative times.
            if 'charttime' in combined.columns and combined['charttime'].dtype == 'object':
                raw_charttime = combined['charttime'].copy()
                numeric_like_mask = raw_charttime.map(
                    lambda value: (
                        isinstance(value, (int, float, np.integer, np.floating))
                        and not isinstance(value, bool)
                    ) or (
                        isinstance(value, str)
                        and value.strip() != ''
                        and pd.to_numeric(pd.Series([value]), errors='coerce').notna().iloc[0]
                    )
                )
                datetime_like_mask = raw_charttime.map(
                    lambda value: isinstance(value, (pd.Timestamp, datetime, np.datetime64))
                )

                charttime_numeric = pd.Series(np.nan, index=raw_charttime.index, dtype='float64')
                if numeric_like_mask.any():
                    charttime_numeric.loc[numeric_like_mask] = pd.to_numeric(
                        raw_charttime.loc[numeric_like_mask],
                        errors='coerce',
                    )
                combined['charttime'] = charttime_numeric

                remaining_mask = datetime_like_mask | ((~numeric_like_mask) & raw_charttime.notna())
                if remaining_mask.any():
                    raw_time = pd.to_datetime(
                        raw_charttime.loc[remaining_mask],
                        errors='coerce',
                    )
                    if raw_time.notna().any() and 'icustay_id' in combined.columns:
                        try:
                            icustays_table = data_source.load_table(
                                'icustays',
                                columns=['icustay_id', 'intime'],
                                verbose=False,
                            )
                            icustays_df = icustays_table.data if hasattr(icustays_table, 'data') else icustays_table
                            if 'intime' in icustays_df.columns:
                                icu_intime = icustays_df[['icustay_id', 'intime']].drop_duplicates().copy()
                                icu_intime['intime'] = pd.to_datetime(icu_intime['intime'], errors='coerce')
                                if icu_intime['intime'].dt.tz is not None:
                                    icu_intime['intime'] = icu_intime['intime'].dt.tz_localize(None)

                                converted = combined.loc[remaining_mask, ['icustay_id']].copy()
                                converted['charttime_abs'] = raw_time.values
                                if hasattr(converted['charttime_abs'].dt, 'tz') and converted['charttime_abs'].dt.tz is not None:
                                    converted['charttime_abs'] = converted['charttime_abs'].dt.tz_localize(None)
                                converted = converted.merge(icu_intime, on='icustay_id', how='left')
                                rel_minutes = np.floor(
                                    (converted['charttime_abs'] - converted['intime']).dt.total_seconds() / 60.0
                                )
                                combined.loc[remaining_mask, 'charttime'] = rel_minutes / 60.0
                        except Exception as e:
                            if DEBUG_MODE:
                                print(f"   ⚠️ [MIMIC-III] charttime mixed-type 规范化失败: {e}")

                combined['charttime'] = pd.to_numeric(combined['charttime'], errors='coerce') if not pd.api.types.is_numeric_dtype(combined['charttime']) else combined['charttime']
        
        # 🔧 CRITICAL FIX 2026-03-10: Multi-source concat produces object dtype for value column
        # When frames from different sources (e.g., respiratorycharting + lab) are concatenated,
        # the value column may become object dtype because each frame has different extra columns.
        # This causes change_interval to use 'first' aggregation instead of 'median'/'max',
        # silently dropping higher values from one source.
        # Example: eICU fio2 - lab FiO2=28 + resp FiO2=50 at hour 30 → first(28,50)=28 instead of max(28,50)=50
        # Fix: Coerce the concept value column to numeric after concat,
        # but ONLY if most values are actually numeric strings (not legitimate strings like 'invasive').
        if concept_name in combined.columns and not combined.empty:
            if not pd.api.types.is_numeric_dtype(combined[concept_name]):
                sample = combined[concept_name].dropna().head(100)
                if len(sample) > 0:
                    numeric_converted = pd.to_numeric(sample, errors='coerce')
                    pct_convertible = float(numeric_converted.notna().sum()) / len(sample)
                    if pct_convertible > 0.5:
                        combined[concept_name] = pd.to_numeric(combined[concept_name], errors='coerce')
        
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

                # 🚀 跳过中间排序：change_interval 使用 groupby 不需要预排序，
                # merge_concepts_r_style 会做最终排序。
                # 仅修复混合类型问题（DuckDB float64 + datetime64 concat → object）
                for key in sort_keys:
                    if key in combined.columns and combined[key].dtype == object:
                        if 'time' in key.lower() or key == 'charttime':
                            # Bug 32 fix: DuckDB返回float64(相对小时)，非DuckDB返回datetime64
                            # concat后变为object dtype — 统一为float
                            original = combined[key]
                            numeric_vals = pd.to_numeric(original, errors='coerce')
                            dt_mask = numeric_vals.isna() & original.notna()
                            if dt_mask.any():
                                dt_vals = pd.to_datetime(original[dt_mask], errors='coerce')
                                if dt_vals.notna().any():
                                    # 加载 intime 以转换 datetime → 相对小时
                                    intime_col = None
                                    if 'intime' in combined.columns:
                                        intime_col = pd.to_datetime(combined.loc[dt_mask, 'intime'], errors='coerce')
                                    else:
                                        try:
                                            icu_tbl = data_source.load_table('icustays',
                                                columns=[id_columns[0], 'intime'], verbose=False)
                                            icu_df = icu_tbl.data if hasattr(icu_tbl, 'data') else icu_tbl
                                            if pd.api.types.is_datetime64_any_dtype(icu_df['intime']):
                                                if hasattr(icu_df['intime'].dt, 'tz') and icu_df['intime'].dt.tz is not None:
                                                    icu_df['intime'] = icu_df['intime'].dt.tz_localize(None)
                                            combined = combined.merge(
                                                icu_df[[id_columns[0], 'intime']], on=id_columns[0], how='left')
                                            intime_col = pd.to_datetime(combined.loc[dt_mask, 'intime'], errors='coerce')
                                        except Exception:
                                            pass
                                    if intime_col is not None:
                                        if hasattr(intime_col.dt, 'tz') and intime_col.dt.tz is not None:
                                            intime_col = intime_col.dt.tz_localize(None)
                                        if hasattr(dt_vals.dt, 'tz') and dt_vals.dt.tz is not None:
                                            dt_vals = dt_vals.dt.tz_localize(None)
                                        rel_hours = np.floor((dt_vals - intime_col).dt.total_seconds() / 3600.0)
                                        numeric_vals.loc[dt_mask] = rel_hours
                            combined[key] = numeric_vals
        combined = combined.reset_index(drop=True)
        agg_value = self._coerce_final_aggregator(aggregator)
        if agg_value in (None, "auto"):
            fallback_agg = definition.aggregate
            if fallback_agg is not None:
                agg_value = self._coerce_final_aggregator(fallback_agg)

        # CRITICAL FIX: Avoid double aggregation issue
        # Strategy: Only use change_interval's aggregation (on relative time after floor)
        # Do NOT use _apply_aggregation before time alignment
        
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
        
        # 🔧 FIX 2025-02-13: Skip change_interval for win_tbl target concepts
        # R ricu returns win_tbl format directly without time aggregation
        # 🔧 FIX 2026-02-15: Only skip for lgl_cncpt/fct_cncpt with target=win_tbl
        # num_cncpt/unt_cncpt (like dex) still need change_interval even with target=win_tbl
        # R ricu's load_concepts.num_cncpt always runs change_interval
        _cls = getattr(definition, 'class_name', '')
        _cls_list = _cls if isinstance(_cls, list) else [_cls]
        _is_true_win_tbl_class = any(c in ('lgl_cncpt', 'fct_cncpt') for c in _cls_list)
        is_win_tbl_target = getattr(definition, 'target', 'ts_tbl') == 'win_tbl' and _is_true_win_tbl_class
        
        # 🔧 FIX 2026-02: Skip change_interval for id_tbl target concepts (height, weight, etc.)
        # R ricu's load_id doesn't aggregate by time — it only does per-patient aggregation
        # (aggregate.id_tbl groups by meta_vars=id_cols, numeric→median)
        # If we apply change_interval first, we get median-of-means instead of median-of-all-values
        is_id_tbl_target = getattr(definition, 'target', 'ts_tbl') == 'id_tbl'
        
        # num_cncpt with target=win_tbl (like dex): needs expansion, not change_interval
        _is_num_win_tbl = (
            getattr(definition, 'target', 'ts_tbl') == 'win_tbl'
            and not _is_true_win_tbl_class  # Not lgl/fct
        )
        
        # Apply interval alignment and aggregation if interval is specified
        # BUT skip for win_tbl/id_tbl target concepts
        if interval is not None and index_column and index_column in combined.columns and not is_win_tbl_target and not is_id_tbl_target:
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
                    # 🔧 FIX 2025-01-29: 添加 measuredat_minutes 支持 AUMC DuckDB 聚合返回的列名
                    time_cols = [c for c in combined.columns if c in ['start', 'charttime', 'measuredat', 'measuredat_minutes']]
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
                        unit_column=final_unit_column if final_unit_column and final_unit_column in combined.columns else None,
                        time_columns=[col for col in time_columns if col and col in combined.columns],
                    )
                
                # DEBUG
            # Determine aggregation method for change_interval
            # This is the ONLY aggregation we should do (on relative time)
            # 🔧 FIX 2024-12-17: When agg_value is explicitly False, skip ALL aggregation
            # This is critical for vaso60 sub-concepts which need raw data for callback's own max aggregation
            if agg_value is False:
                agg_method = False  # Explicitly no aggregation
            else:
                agg_method = agg_value if agg_value not in (None, "auto") else None
                if agg_method in (None, "auto"):
                    agg_method = None
                
                # 🔧 FIX 2025-02: When callback is aggregate_fun('sum', ...), use 'sum' aggregation
                # in change_interval to preserve the sum semantics
                if agg_method is None and sources:
                    for src in sources:
                        if src.callback:
                            import re as re_module
                            agg_match = re_module.search(r"aggregate_fun\(['\"](\w+)['\"]", src.callback)
                            if agg_match:
                                agg_method = agg_match.group(1)  # e.g., 'sum'
                                break
                
                # 🔧 FIX 2026-03-13: For num_cncpt with target=win_tbl (like dex),
                # R ricu's expand.win_tbl uses raw (unfloored) start times to compute
                # end = floor(raw_start + dur). change_interval would floor start first,
                # causing floor(floored_start + dur) ≠ floor(raw_start + dur).
                # Skip change_interval entirely; expansion in _ensure_concept_loaded handles it.
                if _is_num_win_tbl:
                    agg_method = '__skip_change_interval__'
                
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
            
            # 🚀 CRITICAL FIX: Expand win_tbl data BEFORE change_interval
            # R ricu does: expand(win_tbl) -> change_interval(ts_tbl)
            # Without this, mech_vent with 3 events only returns 2 rows instead of 439
            # 🔧 FIX: Also check for 'stoptime' (used by prescriptions table) and 'stop' (used by AUMC drugitems)
            has_endtime = 'endtime' in combined.columns
            has_stoptime = 'stoptime' in combined.columns  # prescriptions uses stoptime
            has_duration = 'duration' in combined.columns
            # 🔧 FIX 2024-12-26: Also check for 'dur_var' column (from grp_mount_to_rate callback for HiRID dex, etc.)
            has_dur_var = 'dur_var' in combined.columns
            
            # 🔧 FIX 2024-12-17: 'stop' column should ONLY trigger expand if the concept has dur_var='stop' in its source definition
            # AUMC drugitems table always has start/stop columns, but only concepts with dur_var='stop' should be expanded
            # e.g., 'dex' has dur_var='stop' → expand; 'ins' has no dur_var → no expand
            has_stop = False
            if 'stop' in combined.columns:
                # Check if any source for this concept has dur_var='stop'
                db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                if db_name and hasattr(definition, 'sources') and db_name in definition.sources:
                    db_sources = definition.sources[db_name]
                    if isinstance(db_sources, list):
                        for src in db_sources:
                            src_dict = src.__dict__ if hasattr(src, '__dict__') else src
                            if src_dict.get('dur_var') == 'stop':
                                has_stop = True
                                break
            
            # 🔧 FIX: Only expand true window concepts, NOT point event concepts
            # POINT_EVENT_CONCEPTS like 'abx' have endtime/stoptime columns from source tables
            # but should NOT be expanded - they use set_val(TRUE) callback for point events
            from .compat import POINT_EVENT_CONCEPTS, DURATION_CONCEPTS
            is_point_event = concept_name in POINT_EVENT_CONCEPTS
            is_duration_concept = concept_name in DURATION_CONCEPTS or concept_name.endswith('_dur')
            
            # 🔧 FIX 2026-02-07: Skip expand for callbacks that already call expand internally
            # mimic_rate_cv/mimic_rate_mv, expand_intervals, etc. already handle time expansion
            # Re-expanding would cause issues (e.g., NaN endtime from inputevents_cv merge with inputevents_mv)
            callbacks_that_expand = [
                'mimic_rate_cv', 'mimic_rate_mv', 'expand_intervals',
                'hirid_rate', 'hirid_rate_kg', 'hirid_duration',
                'eicu_rate', 'eicu_rate_kg',
                'aumc_rate', 'aumc_rate_kg', 'aumc_rate_units', 'aumc_dur',
                'sic_rate_kg', 'sic_dur',
                # 🔧 FIX 2026-02-05: distribute_amount 内部已经处理时间展开，不需要再次 expand
                # MIMIC-III ins 概念使用 inputevents_mv 的 distribute_amount callback
                'distribute_amount',
            ]
            callback_already_expanded = False
            if sources:
                for src in sources:
                    cb = getattr(src, 'callback', None)
                    if cb:
                        for expand_cb in callbacks_that_expand:
                            if expand_cb in cb:
                                callback_already_expanded = True
                                break
                    if callback_already_expanded:
                        break
            
            # 🔧 FIX 2026-03-12: DO expand concepts with target="win_tbl"
            # R ricu expands win_tbl (starttime + dur_var) to hourly time series:
            # - fct_cncpt (mech_vent): expand + fill_gaps in load_concepts.fct_cncpt
            # - num_cncpt (dex): expand as part of change_interval then aggregate
            # Previously expansion was disabled, causing dex to have ~6 rows vs R's ~142.
            # WinTbl concepts used as dependencies (mech_vent for GCS) will be expanded;
            # the GCS callback handles both expanded (Path B: merge) and WinTbl (Path A: window_match).
            is_win_tbl_target_expand = getattr(definition, 'target', 'ts_tbl') == 'win_tbl'
            
            should_expand = (has_endtime or has_stoptime or has_stop or has_duration or has_dur_var) and not is_point_event and not is_duration_concept and not callback_already_expanded and not is_win_tbl_target_expand
            
            # DEBUG
            if DEBUG_MODE and (has_dur_var or has_endtime or has_stoptime):
                print(f"   🔍 DEBUG: should_expand={should_expand}, has_dur_var={has_dur_var}, has_endtime={has_endtime}, is_point_event={is_point_event}, callback_already_expanded={callback_already_expanded}, is_win_tbl_target={is_win_tbl_target}")
            if should_expand:
                from .ts_utils import expand
                
                # 🔧 FIX 2024-12-26: Handle dur_var column from grp_mount_to_rate callback
                # dur_var contains duration as Timedelta or numeric hours, calculate endtime = starttime + dur_var
                # 🔧 FIX 2026-03-12: Also handle case where endtime exists but has NaN
                # (multi-source concat: one source has endtime, other has dur_var)
                if has_dur_var:
                    dur_col = combined['dur_var']
                    start_col = combined[index_column]
                    
                    # Compute endtime from dur_var for rows that need it
                    needs_endtime = not has_endtime or (has_endtime and combined['endtime'].isna().any())
                    
                    if needs_endtime:
                        # Normalize dur_col to numeric hours
                        if 'timedelta' in str(dur_col.dtype).lower():
                            dur_numeric = dur_col.dt.total_seconds() / 3600.0
                        elif pd.api.types.is_numeric_dtype(dur_col):
                            dur_numeric = dur_col
                        else:
                            # Mixed or object dtype: try to_numeric, fallback to timedelta
                            dur_numeric = pd.to_numeric(dur_col, errors='coerce')
                            # For any remaining NaN that were timedelta strings
                            still_na = dur_numeric.isna() & dur_col.notna()
                            if still_na.any():
                                td = pd.to_timedelta(dur_col[still_na], errors='coerce')
                                dur_numeric.loc[still_na] = td.dt.total_seconds() / 3600.0
                        
                        if pd.api.types.is_datetime64_any_dtype(start_col):
                            endtime_from_dur = start_col + pd.to_timedelta(dur_numeric, unit='h')
                        else:
                            endtime_from_dur = start_col + dur_numeric
                        
                        if has_endtime:
                            # Fill NaN endtimes with computed values from dur_var
                            combined['endtime'] = combined['endtime'].fillna(endtime_from_dur)
                        else:
                            combined['endtime'] = endtime_from_dur
                    
                    has_endtime = True
                    # Drop dur_var to avoid confusion
                    combined = combined.drop(columns=['dur_var'])
                    has_dur_var = False
                
                # 🔧 FIX: When both endtime and stoptime exist (multi-source concepts like abx),
                # merge stoptime into endtime to handle rows from different tables
                if has_endtime and has_stoptime:
                    # Fill endtime nulls with stoptime values
                    combined['endtime'] = combined['endtime'].fillna(combined['stoptime'])
                    # Drop the stoptime column after merging
                    combined = combined.drop(columns=['stoptime'])
                    has_stoptime = False  # Reset flag since we merged it
                
                # 🔧 FIX: Merge stop column into endtime (for AUMC drugitems)
                if has_stop:
                    if has_endtime:
                        combined['endtime'] = combined['endtime'].fillna(combined['stop'])
                    else:
                        combined['endtime'] = combined['stop']
                    combined = combined.drop(columns=['stop'])
                    has_endtime = True
                    has_stop = False
                
                # Determine end column: prefer duration, then endtime, then stoptime
                if has_duration:
                    end_col = 'duration'
                elif has_endtime:
                    end_col = 'endtime'
                else:
                    end_col = 'stoptime'
                
                # Determine columns to keep (value columns + unit)
                keep_vars = [concept_name] if concept_name in combined.columns else []
                if final_unit_column and final_unit_column in combined.columns:
                    keep_vars.append(final_unit_column)
                
                # Additional value columns (not ID, not time, not end, not unit)
                # 🔧 FIX: Also exclude duration columns like {concept_name}_dur
                excluded = set(id_columns + [index_column, end_col])
                if final_unit_column:
                    excluded.add(final_unit_column)
                # Exclude all duration-related columns (they shouldn't be expanded)
                dur_cols = [col for col in combined.columns 
                           if col.endswith('_dur') or col == 'duration' or col.endswith('_duration')]
                excluded.update(dur_cols)
                
                value_cols = [col for col in combined.columns 
                             if col not in excluded and col != concept_name]
                keep_vars.extend(value_cols)
                
                # Expand windows to hourly time series
                try:
                    if DEBUG_MODE:
                        print(f"   🔍 DEBUG: expand前, 行数={len(combined)}, start_var={index_column}, end_var={end_col}")
                        print(f"   🔍 DEBUG: endtime样本: {combined[end_col].head(3).tolist() if end_col in combined.columns else 'N/A'}")
                    combined = expand(
                        combined,
                        start_var=index_column,
                        end_var=end_col,
                        step_size=interval,
                        id_cols=id_columns,
                        keep_vars=keep_vars,
                    )
                    if DEBUG_MODE:
                        print(f"   🔍 DEBUG: expand后, 行数={len(combined)}")
                    if verbose:
                        logger.info(f"   ✅ 展开 win_tbl '{concept_name}' 到 {len(combined)} 行")
                except Exception as e:
                    logger.warning(f"Failed to expand win_tbl data for {concept_name}: {e}")
                    # Continue without expansion
            
            # 🔧 FIX 2026-03-10: filter_bounds (min/max) BEFORE change_interval aggregation
            # R ricu load_concepts.num_cncpt actual flow:
            #   1. load_concepts(as_item(x)):
            #      - do_callback → expand (produces multiple rows per hour)
            #      - change_interval.data.table (only re-discretizes, does NOT aggregate duplicates)
            #   2. filter_bounds(res, min, max) → removes out-of-range individual values
            #   3. stats::aggregate(median) → final aggregation
            # easyicu's change_interval DOES aggregate, so filter_bounds must go BEFORE it.
            # Previously filter_bounds was incorrectly placed AFTER change_interval, causing
            # outlier values to participate in median aggregation (e.g. SIC epi_rate 1.9% error).
            if concept_name in combined.columns:
                if definition.minimum is not None or definition.maximum is not None:
                    combined[concept_name] = pd.to_numeric(combined[concept_name], errors='coerce')
                if definition.minimum is not None:
                    before_len = len(combined)
                    combined = combined[combined[concept_name] >= definition.minimum]
                    if DEBUG_MODE and len(combined) < before_len:
                        print(f"   🔍 DEBUG: filter_bounds(pre-agg) min={definition.minimum}: {before_len} -> {len(combined)}")
                if definition.maximum is not None:
                    before_len = len(combined)
                    combined = combined[combined[concept_name] <= definition.maximum]
                    if DEBUG_MODE and len(combined) < before_len:
                        print(f"   🔍 DEBUG: filter_bounds(pre-agg) max={definition.maximum}: {before_len} -> {len(combined)}")
                if definition.minimum is not None or definition.maximum is not None:
                    combined = combined.dropna(subset=[concept_name])

            # Create ICUTable temporarily to use change_interval
            temp_table = ICUTable(
                data=combined,
                id_columns=id_columns,
                index_column=index_column,
                value_column=concept_name,
                unit_column=final_unit_column if final_unit_column and final_unit_column in combined.columns else None,
                time_columns=[col for col in time_columns if col],
            )
            # 🚀 传播 DuckDB 预聚合标记：单源 DuckDB → 无需 change_interval groupby
            if len(frames) <= 1 and _duckdb_source_count == 1:
                temp_table._pre_aggregated = True

            fill_missing = self._should_fill_gaps(concept_name, definition)
            fill_method = self._get_fill_method(concept_name, definition)
            
            # DEBUG
            if DEBUG_MODE:
                print(f"   🔍 DEBUG: change_interval前, 行数={len(combined)}")
            
            # 🔧 FIX 2026-03-13: Skip change_interval for num_cncpt win_tbl (like dex).
            # R ricu's expand.win_tbl uses raw start times: end = floor(raw_start + dur).
            # change_interval floors start first, causing precision loss:
            # floor(2.9 + 0.2)=3 vs floor(floor(2.9) + 0.2)=floor(2.2)=2
            if agg_method != '__skip_change_interval__':
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
                    if DEBUG_MODE:
                        print(f"   🔍 DEBUG: change_interval后, 行数={len(combined)}")
                    # 更新index_column：change_interval可能改变了时间列名(如变为'start')
                    if hasattr(combined_result, 'index_column') and combined_result.index_column:
                        index_column = combined_result.index_column
                else:
                    combined = combined_result
                    if DEBUG_MODE:
                        print(f"   🔍 DEBUG: change_interval后(raw), 行数={len(combined)}")
        elif align_to_admission:
            # Just alignment, no interval/aggregation
            combined = self._align_time_to_admission(
                combined,
                data_source,
                id_columns,
                index_column
            )
        
        # NOTE: filter_bounds已移至change_interval之前（见上方FIX 2026-03-10注释）
        
        # 🔧 NOTE: 不过滤负时间（入ICU前的数据），ricu 保留这些数据
        # 例如：AUMC esr measuredat=-2 表示入院前2小时的数据，ricu 也保留
        
        # 最终验证：确保index_column存在于combined中
        if index_column and index_column not in combined.columns:
            # 尝试查找可能的时间列
            # 🔧 FIX 2025-01-29: 添加 measuredat_minutes 支持 AUMC DuckDB 聚合返回的列名
            time_cols = [c for c in combined.columns if c in ['start', 'charttime', 'measuredat', 'measuredat_minutes', index_column]]
            if time_cols:
                index_column = time_cols[0]
            else:
                # 没有有效的时间列，设为None
                index_column = None
        
        # NOTE: An earlier WinTbl conversion path keyed on `definition.target == 'win_tbl'`
        # was permanently gated behind `if False`; it had unresolved endtime handling
        # and is fully superseded by the `_any_win_tbl` block below (which uses the
        # `dur_var` column produced by upstream expansion). Removed to keep this
        # function unambiguous. See git history for the original draft if needed.

        if concept_name == "infusionoffset" and index_column and index_column in combined.columns:
            combined[concept_name] = combined[index_column]
            combined = combined.drop(columns=["drugrate"], errors="ignore")
        
        # 🔧 FIX 2025-02-14: For win_tbl target concepts that have dur_var column,
        # return WinTbl instead of ICUTable so that fwd_concept can properly
        # detect the WinTbl and preserve dur_var for concepts like ett_gcs
        # 🔧 FIX 2026-03-13: Use broader condition — ANY concept with target='win_tbl'
        # (not just lgl/fct). num_cncpt win_tbl (dex) also needs WinTbl return path
        # so _ensure_concept_loaded can expand it with proper clamping.
        _any_win_tbl = getattr(definition, 'target', 'ts_tbl') == 'win_tbl'
        if _any_win_tbl and 'dur_var' in combined.columns and index_column:
            from .table import WinTbl
            # 🔧 FIX: Ensure dur_var is numeric (can become object after pd.concat with NaN)
            combined['dur_var'] = pd.to_numeric(combined['dur_var'], errors='coerce')
            # Separate rows: with dur_var (WinTbl sources) and without (TsTbl sources)
            ts_source_rows = combined[combined['dur_var'].isna()].copy()
            combined = combined.dropna(subset=['dur_var'])
            
            # 🔧 FIX 2026-02-15: Apply R ricu's change_interval behavior for win_tbl
            # For lgl/fct win_tbl (mech_vent): floor + dedup as before
            # For num_cncpt win_tbl (dex): skip floor+dedup — expansion needs raw start times
            # R ricu's expand.win_tbl uses raw start: end = floor(raw_start + dur)
            if not _is_num_win_tbl and interval is not None and index_column in combined.columns:
                if pd.api.types.is_numeric_dtype(combined[index_column]):
                    interval_hours = interval.total_seconds() / 3600.0
                    combined = combined.copy()
                    combined[index_column] = (combined[index_column] // interval_hours) * interval_hours
                elif pd.api.types.is_datetime64_any_dtype(combined[index_column]):
                    combined = combined.copy()
                    combined[index_column] = combined[index_column].dt.floor(interval)
                # unique by (id_cols + index_col), keep first
                dedup_cols = [c for c in id_columns if c in combined.columns] + [index_column]
                combined = combined.drop_duplicates(subset=dedup_cols, keep='first')
            
            win_tbl = WinTbl(
                data=combined,
                id_vars=id_columns,
                index_var=index_column,
                dur_var='dur_var',
            )
            # 🔧 FIX 2026-03-11: Save TsTbl source rows for later expansion
            # These rows came from sources without dur_var (e.g., eICU infusiondrug, MIMIC inputevents_cv)
            # They'll be combined with expanded WinTbl rows in _ensure_concept_loaded
            win_tbl._ts_source_rows = ts_source_rows
            return win_tbl
        
        try:
            return ICUTable(
                data=combined,
                id_columns=id_columns,
                index_column=index_column,  # Already updated for eICU if needed
                value_column=concept_name,
                unit_column=final_unit_column if final_unit_column and final_unit_column in combined.columns else None,
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
                    unit_column=final_unit_column if final_unit_column and final_unit_column in combined.columns else None,
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
            
            # 自动检测其他可能的时间列 (start, stop, dur_var)
            for col in data.columns:
                if col in ['start', 'stop', 'dur_var']:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        cols_to_convert.add(col)
            
            # 转换所有时间列（从分钟到小时）
            for col in cols_to_convert:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col] / 60.0
            return data
        
        if db_name == 'aumc':
            # AUMC时间列是绝对时间戳（毫秒，已在datasource.py中转换为分钟）
            # R ricu 的行为：使用绝对时间（小时），不减去 admittedat
            # 
            # 🔧 ricu 兼容模式：
            # ricu 的 change_id 对于 AUMC 不做时间相对化，因为数据默认是 admissionid 级别
            # 当 id_var == target_id 时，change_id 直接返回不处理时间
            # 因此 ricu 导出的 CSV 使用绝对时间（floor(ms/3600000) = 小时）
            # 
            # 为了与 ricu 兼容，easyicu 也使用绝对时间：
            # - 不减去 admittedat
            # - 只将分钟转换为小时
            
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
                if col in ['start', 'stop', 'dur_var']:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        cols_to_convert.add(col)
            
            if not cols_to_convert:
                return data
            
            # 时间转换：只将分钟转换为小时（不减去 admittedat）
            for col in cols_to_convert:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    # 分钟转小时
                    data[col] = data[col] / 60.0
            
            return data
        
        if db_name == 'sic':
            # SIC tables (data_float_h, laboratory, medication) use Offset in SECONDS.
            # R ricu converts via change_interval(hours(1)) → divide by 3600.
            # Some callbacks (sic_dur, sic_rate_kg) already convert medication Offset
            # to hours internally — detect via magnitude check to avoid double-conversion.
            
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
            
            for col in cols_to_convert:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    max_abs = data[col].abs().max()
                    if pd.notna(max_abs) and max_abs > 5000:
                        # Values > 5000 cannot be hours (= 208 days), must be seconds
                        data[col] = data[col] / 3600.0
            
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
        
        # 🔧 FIX: 确定数据库特定的 stay-level ID 列名
        # MIMIC-III 使用 icustay_id，MIMIC-IV 使用 stay_id
        db_stay_id_col = 'icustay_id' if db_name == 'mimic' else 'stay_id'
        
        # 特殊处理：如果primary_id不是stay_id/icustay_id，需要先join icustays获取
        # 这对于labevents（使用subject_id）很重要
        if primary_id != db_stay_id_col and db_stay_id_col not in data.columns:
            try:
                # Use cached icustays table if available
                cache_key = f"{primary_id}_{db_stay_id_col}_intime_{db_name}"
                
                # 确定要加载的列
                cols_to_load = [primary_id, 'intime']
                if primary_id != db_stay_id_col:
                    cols_to_load.append(db_stay_id_col)
                
                if self._icustays_cache is None or cache_key not in str(self._icustays_cache.columns.tolist()):
                    icustays_temp = data_source.load_table('icustays', columns=cols_to_load, verbose=False)
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
                
                # Join获取stay_id/icustay_id和intime
                merge_cols = [c for c in [primary_id, db_stay_id_col, 'intime'] if c in icustays_temp_df.columns]
                merge_cols = list(set(merge_cols))  # 去重
                data = data.merge(icustays_temp_df[merge_cols], on=primary_id, how='left')
                
                # 更新primary_id为数据库特定的stay_id列
                if db_stay_id_col in data.columns:
                    primary_id = db_stay_id_col
                # 已经有intime了，后面不需要再加载
            except Exception:
                return data
        
        # 🔧 FIX: 如果 primary_id 就是 icustay_id（MIMIC-III chartevents），
        # 仍然需要加载 intime 进行时间转换
        if primary_id == db_stay_id_col and 'intime' not in data.columns:
            try:
                # 加载 icustays 获取 intime
                if self._icustays_cache is not None and 'intime' in self._icustays_cache.columns and primary_id in self._icustays_cache.columns:
                    icustays_temp_df = self._icustays_cache
                else:
                    icustays_temp = data_source.load_table('icustays', columns=[primary_id, 'intime', 'outtime', 'los'], verbose=False)
                    if hasattr(icustays_temp, 'data'):
                        icustays_temp_df = icustays_temp.data
                    else:
                        icustays_temp_df = icustays_temp
                    
                    # 确保intime是tz-naive datetime
                    if pd.api.types.is_datetime64_any_dtype(icustays_temp_df['intime']):
                        if hasattr(icustays_temp_df['intime'].dt, 'tz') and icustays_temp_df['intime'].dt.tz is not None:
                            icustays_temp_df['intime'] = icustays_temp_df['intime'].dt.tz_localize(None)
                    
                    self._icustays_cache = icustays_temp_df
                
                # 只合并 intime 列
                data = data.merge(icustays_temp_df[[primary_id, 'intime']], on=primary_id, how='left')
            except Exception:
                return data
        
        # 🔧 FIX Bug 32: Handle dtype=object time column from multi-source concat.
        # When DuckDB returns float64 (relative hours) for one source and another source
        # contributes datetime64 values, pd.concat produces dtype=object. Without this,
        # is_numeric_dtype fails and the datetime path reinterprets float hours as
        # nanosecond Unix timestamps (e.g., 38.0 → 1970-01-01), corrupting ALL rows
        # to the same time value, which change_interval then collapses to 1 row.
        # Example: MIIV o2sat — chartevents (DuckDB float64) + labevents (datetime64)
        if data[index_column].dtype == 'object':
            original_vals = data[index_column].copy()
            numeric_vals = pd.to_numeric(original_vals, errors='coerce')
            datetime_mask = numeric_vals.isna() & original_vals.notna()
            
            if datetime_mask.any() and 'intime' in data.columns:
                # Convert datetime remnants to relative hours using already-available intime
                dt_vals = pd.to_datetime(original_vals[datetime_mask], errors='coerce')
                if dt_vals.notna().any():
                    intime = pd.to_datetime(data.loc[datetime_mask, 'intime'], errors='coerce')
                    if hasattr(intime.dt, 'tz') and intime.dt.tz is not None:
                        intime = intime.dt.tz_localize(None)
                    if hasattr(dt_vals.dt, 'tz') and dt_vals.dt.tz is not None:
                        dt_vals = dt_vals.dt.tz_localize(None)
                    rel_minutes = np.floor((dt_vals - intime).dt.total_seconds() / 60.0)
                    numeric_vals.loc[datetime_mask] = rel_minutes / 60.0
            
            data[index_column] = numeric_vals
        
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
                if 'outtime' in data.columns and data['outtime'].notna().any():
                    icu_len = (pd.to_datetime(data['outtime']) - pd.to_datetime(data['intime']))
                    icu_len.dt.total_seconds() / 3600.0

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
            # Convert dur_var from minutes to hours (same as datetime path at L4388)
            # Callbacks like hirid_vent produce dur_var in minutes; index is already in hours
            if 'dur_var' in data.columns and pd.api.types.is_numeric_dtype(data['dur_var']):
                data['dur_var'] = data['dur_var'] / 60.0
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
            # 🔧 CRITICAL FIX: 使用 R ricu 的时间转换逻辑：
            # R ricu 使用 round_to(difftime(x, y, units = "mins"))
            # round_to() 对于 to=1 使用 floor()
            # 即: floor((charttime - intime).total_seconds() / 60) / 60
            # 这与直接 total_seconds() / 3600 的结果不同，特别是对于负时间
            # 例如: -12秒 → floor(-12/60)/60 = floor(-0.2)/60 = -1/60 = -0.0167 小时
            #       -12秒 → -12/3600 = -0.003 小时 (WRONG)
            time_diff = data[index_column] - data['intime']
            # 🔧 R ricu compatible: floor to minutes first, then convert to hours
            minutes = np.floor(time_diff.dt.total_seconds() / 60.0)
            hours = minutes / 60.0
            
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
                # Convert to hours since admission (R ricu compatible: floor to minutes first)
                time_diff_col = data[time_col] - data['intime']
                minutes_col = np.floor(time_diff_col.dt.total_seconds() / 60.0)
                data[time_col] = minutes_col / 60.0
            
            # 注意：不过滤负时间（入ICU前）或超过outtime的数据，匹配 R ricu 行为
            
            # Convert dur_var from minutes to hours (dur_is_end outputs minutes)
            if 'dur_var' in data.columns and pd.api.types.is_numeric_dtype(data['dur_var']):
                data['dur_var'] = data['dur_var'] / 60.0
            
            # Drop the temporary alignment columns
            drop_cols = ['intime']
            if 'outtime' in data.columns:
                drop_cols.append('outtime')
            data = data.drop(columns=drop_cols)
            
        except Exception:
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
        
        # 🔧 FIX 2024-12-16: vaso60 callback requires UNAGGREGATED rate data!
        # R ricu's vaso60 joins raw rate data to duration windows, then does its own
        # change_interval + aggregate("max"). If we pre-aggregate with median,
        # we lose the ability to get the correct max value for each hour.
        # 
        # Example: Patient has two rate records in hour 13: 1.0 and 2.0
        # - R ricu: vaso60 receives both → max(1.0, 2.0) = 2.0
        # - Pyricu (old): sub-concept aggregated first → median(1.0, 2.0) = 1.5 → vaso60 → 1.5
        #
        # Solution: For vaso60 callback, set aggregate=False for all sub-concepts
        if definition.callback == "vaso60":
            aggregate_mapping = {name: False for name in sub_names}

        # Prepare kwargs for sub-concepts, allowing them to be optional
        sub_kwargs = {**kwargs, '_allow_missing_concept': True}
        
        # 🔧 FIX: Use concept's own interval (like R ricu's coalesce(x[["interval"]], interval))
        # This is critical for vaso60-type concepts where interval="00:01:00" (1 minute)
        # ensures rate data is loaded at fine granularity before being aggregated by callback
        sub_interval = definition.interval if definition.interval is not None else interval
        
        # 🔥 CRITICAL: 内部递归调用必须使用 r_compatible=False
        # 否则会返回 DataFrame 而不是 Dict[str, ICUTable]，导致后续处理失败
        # 🔥 优化：使用缓存以避免重复加载相同的子概念（如 fio2 被 pafi 和 safi 共享）
        # 缓存命中时会返回深拷贝，所以后续修改不会污染缓存
        sub_tables = self.load_concepts(
            sub_names,
            data_source,
            merge=False,
            aggregate=aggregate_mapping,
            patient_ids=patient_ids,
            verbose=verbose,
            interval=sub_interval,  # Use concept's own interval (1min for vaso60 concepts)
            align_to_admission=align_to_admission,  # Pass align flag
            r_compatible=False,  # 🔥 内部调用必须返回 Dict[str, ICUTable]
            concept_workers=1,  # 🔧 子概念顺序加载，避免过度并行导致线程竞争
            _skip_concept_cache=False,  # 🚀 启用缓存以避免重复加载共享子概念
            **sub_kwargs,  # Pass kwargs with allow_missing flag
        )

        # 🔧 释放子概念加载过程中的碎片内存
        from .memory_manager import release_memory
        release_memory()

        # 🔧 FIX 2026-03-10: Also handle WinTbl/TsTbl which don't inherit from ICUTable
        if isinstance(sub_tables, ICUTable) or (hasattr(sub_tables, 'data') and not isinstance(sub_tables, dict)):
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
                # 🔧 FIX 2026-03-10: Handle both ICUTable (.index_column) and WinTbl/TsTbl (.index_var)
                idx_col = getattr(table, 'index_column', None) or getattr(table, 'index_var', None)
                if idx_col:
                    # Check if this table uses an eICU-specific time column
                    if idx_col in eicu_time_cols and idx_col != 'charttime':
                        # Rename the column in the DataFrame
                        if idx_col in table.data.columns:
                            renamed_data = table.data.rename(columns={idx_col: 'charttime'})
                            # Create new ICUTable with updated index_column
                            table = ICUTable(
                                data=renamed_data,
                                id_columns=table.id_columns if hasattr(table, 'id_columns') else list(getattr(table, 'id_vars', [])),
                                index_column='charttime',  # Update metadata
                                value_column=getattr(table, 'value_column', None),
                                unit_column=getattr(table, 'unit_column', None),
                                time_columns=getattr(table, 'time_columns', []),
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

        # 🔧 FIX 2025-01: Use passed interval parameter, falling back to definition.interval
        # This ensures user-specified interval (e.g., '1h') is passed to callbacks
        # instead of always using the concept's default interval (which may be '1min')
        effective_interval = interval if interval is not None else definition.interval
        
        # 🚀 优化：预缓存子概念的原始数据到 _raw_concept_cache
        # 这样回调函数中通过 get_raw_concept 可以直接命中缓存，避免重复加载
        # 特别适用于 vaso_ind, vent_ind 等需要原始时间数据的回调
        if effective_interval is not None:
            # 🔧 使用统一的 hash 函数
            patient_ids_hash = _compute_patient_ids_hash(patient_ids)
            
            # 将已加载的子概念数据缓存为原始数据
            with self._cache_lock:
                for sub_name, sub_table in sub_tables.items():
                    sub_agg = (aggregate_mapping or {}).get(sub_name, "auto")
                    if sub_agg in (None, "auto"):
                        sub_def = self.dictionary.get(sub_name)
                        if sub_def and sub_def.aggregate is not None:
                            sub_agg = sub_def.aggregate
                    self._store_raw_concept_cache(
                        sub_name,
                        patient_ids_hash,
                        sub_table,
                        aggregator=sub_agg,
                        store_legacy=False,
                    )
        
        ctx = ConceptCallbackContext(
            concept_name=concept_name,
            target=definition.target,
            interval=effective_interval,
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

        # 🔧 回调执行后释放子概念的中间内存
        # sub_tables 在回调内已经被消费，可以释放
        del sub_tables
        release_memory()

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
            # 🔧 FIX 2024-12-01: 删除 VASO_RATE_CONCEPTS 使用 max 聚合的特殊处理
            # R ricu 对所有数值概念默认使用 median 聚合，VASO_RATE_CONCEPTS 也不例外
            # 之前的 max 聚合导致 norepi_rate 等概念与 ricu 不一致
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
            # 🔧 FIX 2026-02: 对于 target='id_tbl' 的概念（如 height, weight），
            # 跳过 change_interval。R ricu 中 load_id 不按时间聚合，
            # 只在 aggregate.id_tbl 中按患者 ID 取中位数。
            # 如果先按时间聚合再按患者聚合，会得到 median-of-medians 而非 median-of-all。
            is_id_tbl_target = definition and getattr(definition, 'target', 'ts_tbl') == 'id_tbl'
            has_time_column = getattr(result, 'index_column', None)
            if agg_method and has_time_column and has_time_column in result.data.columns and not result.data.empty and not isinstance(result, WinTbl) and not is_id_tbl_target:
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

    def _merge_partial_wide_result(
        self,
        merged_df: pd.DataFrame,
        tables: Mapping[str, ICUTable],
        concept_names: List[str],
        covered_names: set[str],
        data_source: Optional['ICUDataSource'] = None,
    ) -> Optional[pd.DataFrame]:
        """Merge a pre-built wide batch result with a small number of remaining simple tables."""
        if merged_df is None or merged_df.empty:
            return None

        result = merged_df
        id_col = result.columns[0]
        if 'charttime' not in result.columns:
            return None

        remaining = [name for name in concept_names if name not in covered_names]

        merge_keys = [id_col, 'charttime']

        # 🚀 Collect remaining concept frames first, then merge with batch result once
        # This avoids N sequential merges on the large batch DataFrame (12M+ rows).
        # Instead: merge small remaining frames together, then single merge with batch.
        remaining_frames = []
        for name in remaining:
            table = tables.get(name)
            if table is None:
                continue
            if isinstance(table, WinTbl):
                return None
            if not hasattr(table, 'data') or getattr(table, 'index_column', None) != 'charttime':
                return None
            frame = table.data
            if frame is None or frame.empty:
                continue
            if id_col not in frame.columns or 'charttime' not in frame.columns:
                return None
            if name not in frame.columns:
                return None
            frame = frame[merge_keys + [name]]
            remaining_frames.append(frame)

        if remaining_frames:
            # 🚀 Hash-based vectorized merge: avoid pd.merge entirely for 98%+ of rows
            # Build combined int64 key for O(1) hash lookup. Most remaining keys overlap
            # with batch (13.2M batch → 13.4M final = only ~1.5% extra rows).
            # This replaces N sequential merges (16.4s) + 1 big merge (4.2s) = 20.6s
            # with hash-build (~0.5s) + N get_indexer (~1.5s) = ~2s total.
            id_vals = result[id_col].values.astype(np.int64)
            time_vals = result[merge_keys[1]].values.astype(np.int64)
            time_min = int(time_vals.min()) if len(time_vals) > 0 else 0
            stride = int(time_vals.max()) - time_min + 2
            batch_key = id_vals * stride + (time_vals - time_min)
            # Hash table for O(1) key → row_index lookup
            batch_idx_map = pd.Index(batch_key)

            extra_frames = []
            remaining_names = []
            for frame in remaining_frames:
                name = [c for c in frame.columns if c not in merge_keys][0]
                remaining_names.append(name)
                fid = frame[id_col].values.astype(np.int64)
                ftime = frame[merge_keys[1]].values.astype(np.int64)
                fkey = fid * stride + (ftime - time_min)

                indexer = batch_idx_map.get_indexer(fkey)
                matched = indexer >= 0

                # Assign matched values directly (no merge needed)
                col_data = np.full(len(result), np.nan)
                col_data[indexer[matched]] = pd.to_numeric(
                    frame[name].iloc[np.where(matched)[0]], errors='coerce'
                ).values
                result[name] = col_data

                # Collect unmatched rows for append
                if not matched.all():
                    extra_frames.append(frame.iloc[np.where(~matched)[0]])

            # Handle extra rows (keys not in batch) — typically < 2% of data
            if extra_frames:
                extras = pd.concat(extra_frames, ignore_index=True)
                # Merge extra rows together (small: ~200K rows)
                extra_names = [c for c in extras.columns if c not in merge_keys]
                if len(extra_names) > 1:
                    # Multiple extra columns — need to consolidate duplicates
                    extras = extras.groupby(merge_keys, sort=False).first().reset_index()
                result = pd.concat([result, extras], ignore_index=True)

        final_cols = [id_col, 'charttime']
        final_cols.extend([name for name in concept_names if name in result.columns])
        final_cols.extend([c for c in result.columns if c not in final_cols])
        return result[final_cols].reset_index(drop=True)

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
            # 🚀 性能优化：传递 target 类型，对于 id_tbl 不生成时间网格
            target_type = definition.target if hasattr(definition, 'target') else None
            raw = self._load_fun_item_los(concept_name, source, data_source, patient_ids, target_type=target_type)
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
        target_type: Optional[str] = None,
    ) -> ICUTable:
        """Load LOS (Length of Stay) concept.
        
        RICU behavior: los_hosp uses win_type='hadm' to calculate hospital LOS,
        but returns results keyed by stay_id (ICU level), not hadm_id.
        This requires joining admissions table with icustays table.
        
        Args:
            concept_name: Name of the concept (e.g., 'los_hosp', 'los_icu')
            source: Concept source configuration
            data_source: Data source instance
            patient_ids: Optional patient ID filter
            target_type: Concept target type ('id_tbl' for static values, None for time series)
        
        Returns:
            ICUTable with LOS data keyed by the primary ID (stay_id for MIIV)
        """
        win_type = source.params.get("win_type")
        if not win_type:
            raise ValueError("los_callback requires 'win_type' parameter.")

        id_cfg = data_source.config.id_configs.get(win_type)
        # 🔧 FIX: 允许 end 为空（HiRID 需要从 observations 合成 end 时间）
        if id_cfg is None or not id_cfg.table or not id_cfg.start:
            raise ValueError(f"Identifier configuration for '{win_type}' is incomplete.")
        
        # 🔧 2026-02-04 FIX: 确定目标 ID 类型（RICU 默认使用 icustay 级别）
        # 获取数据库的主 ID 配置（通常是 icustay）
        primary_id_type = None
        primary_id_cfg = None
        for id_type, cfg in data_source.config.id_configs.items():
            if id_type == 'icustay' or (cfg.position and cfg.position == 3):
                primary_id_type = id_type
                primary_id_cfg = cfg
                break
        
        # 如果没有 icustay，使用 win_type 本身
        if primary_id_cfg is None:
            primary_id_type = win_type
            primary_id_cfg = id_cfg
        
        # 决定是否需要 ID 映射
        need_id_mapping = (win_type != primary_id_type and primary_id_cfg is not None)

        # 🔧 FIX: 如果 id_cfg 没有 end，使用占位符名称（将在后续合成）
        end_col_name = id_cfg.end if id_cfg.end else '_synthesized_end'
        
        required_cols = [id_cfg.id, id_cfg.start]
        if id_cfg.end:
            required_cols.append(id_cfg.end)
        
        table = data_source.load_table(id_cfg.table, columns=required_cols)

        base_frame = table.data.copy()
        
        # 🔧 HiRID 特殊处理：如果没有 end 列，从 observations 合成
        if not id_cfg.end or end_col_name not in base_frame.columns:
            fallback = self._synthesise_los_column(end_col_name, data_source, base_frame)
            if fallback is not None:
                base_frame[end_col_name] = fallback
            else:
                raise KeyError(
                    f"Required end column missing for LOS calculation and cannot be synthesized for '{data_source.config.name}'"
                )
        
        missing_required = [col for col in [id_cfg.id, id_cfg.start, end_col_name] if col not in base_frame.columns]
        if missing_required:
            for column in missing_required:
                fallback = self._synthesise_los_column(column, data_source, base_frame)
                if fallback is None:
                    raise KeyError(
                        f"Required column '{column}' missing for LOS calculation in table '{id_cfg.table}'"
                    )
                base_frame[column] = fallback

        frame = base_frame[[id_cfg.id, id_cfg.start, end_col_name]].copy()
        frame = frame.dropna(subset=[id_cfg.start, end_col_name])

        # Detect time format and database type
        start_col = frame[id_cfg.start]
        end_col = frame[end_col_name]  # 🔧 FIX: 使用 end_col_name 而不是 id_cfg.end
        is_numeric_time = pd.api.types.is_numeric_dtype(start_col)
        ds_name = (data_source.config.name or "").lower()
        
        # Determine time unit: eICU uses minutes, AUMC uses milliseconds, SICdb uses seconds
        is_eicu = ds_name.startswith("eicu")
        is_sic = ds_name == "sic"
        
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
            elif is_sic:
                # SICdb: times are relative SECONDS from admission
                los_days = (end_val.loc[valid_mask] - start_val.loc[valid_mask]) / (3600 * 24)
                duration_hours = (end_val.loc[valid_mask] - start_val.loc[valid_mask]) / 3600
                start_hours = start_val.loc[valid_mask] / 3600
            else:
                # AUMC/HiRID: times are relative MILLISECONDS from admission
                los_days = (end_val.loc[valid_mask] - start_val.loc[valid_mask]) / (1000 * 60 * 60 * 24)
                duration_hours = (end_val.loc[valid_mask] - start_val.loc[valid_mask]) / (1000 * 60 * 60)
                start_hours = start_val.loc[valid_mask] / (1000 * 60 * 60)
            
            frame[concept_name] = los_days
        else:
            # MIIV/eICU/HiRID: times are datetime objects
            start_time = pd.to_datetime(start_col, errors="coerce")
            end_time = pd.to_datetime(end_col, errors="coerce")
            
            # 🔧 FIX: 统一时区（如果一个有时区一个没有）
            # HiRID 的合成 end_time 可能是 tz-aware (UTC)，而 start_time 可能是 tz-naive
            if start_time.dt.tz is None and end_time.dt.tz is not None:
                # 移除 end_time 的时区信息
                end_time = end_time.dt.tz_localize(None)
            elif start_time.dt.tz is not None and end_time.dt.tz is None:
                # 移除 start_time 的时区信息
                start_time = start_time.dt.tz_localize(None)
            
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

        # 🚀 性能优化：对于 id_tbl 目标的概念（如 los_hosp），不生成时间网格
        # 直接返回每个患者一行数据，而不是按小时展开
        if target_type == 'id_tbl':
            # 🔧 2026-02-04 FIX: 如果需要 ID 映射，将 hadm_id 映射到 stay_id
            if need_id_mapping and primary_id_cfg is not None:
                # 加载 ICU stays 表来获取 hadm_id -> stay_id 映射
                icu_table = data_source.load_table(primary_id_cfg.table)
                icu_df = icu_table.data.copy()
                
                # 确保两个表有共同的连接键（通常是 hadm_id 或 subject_id）
                # 对于 MIIV: admissions.hadm_id -> icustays.hadm_id -> icustays.stay_id
                join_col = id_cfg.id  # hadm_id
                
                if join_col in icu_df.columns and primary_id_cfg.id in icu_df.columns:
                    # 合并：frame[hadm_id, los_hosp] + icu_df[hadm_id, stay_id]
                    result_df = frame[[id_cfg.id, concept_name]].merge(
                        icu_df[[join_col, primary_id_cfg.id]].drop_duplicates(),
                        on=join_col,
                        how='inner'
                    )
                    # 每个 stay_id 只保留一行（如果一个 hadm 有多个 ICU stays，取第一个）
                    result_df = result_df.drop_duplicates(subset=[primary_id_cfg.id])
                    result_df = result_df[[primary_id_cfg.id, concept_name]]
                    
                    # 应用患者过滤
                    if patient_ids is not None:
                        if isinstance(patient_ids, dict):
                            candidates = patient_ids.get(primary_id_cfg.id) or list(patient_ids.values())[0] if patient_ids else []
                        else:
                            candidates = list(patient_ids)
                        if candidates:
                            result_df = result_df[result_df[primary_id_cfg.id].isin(set(candidates))]
                    
                    return ICUTable(
                        data=result_df,
                        id_columns=[primary_id_cfg.id],
                        index_column=None,
                        value_column=concept_name,
                    )
            
            # 静态值：每个患者只有一行（无需映射的情况）
            result_df = frame[[id_cfg.id, concept_name]].copy()
            return ICUTable(
                data=result_df,
                id_columns=[id_cfg.id],
                index_column=None,
                value_column=concept_name,
            )

        # Generate hourly time grid (only for time-series concepts)
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
                # MIIV/eICU/HiRID: use datetime and convert to relative hours
                start_dt = start_time.loc[idx]
                end_dt = end_time.loc[idx]
                
                # 🔧 FIX: 对于 HiRID 和其他使用 datetime 的数据库，
                # 直接计算相对小时数，而不是存储 datetime
                # 从 start_dt - 1小时 开始，到 end_dt 结束
                start_hour = -1  # 从 admission 前 1 小时开始
                end_hour = int((end_dt - start_dt).total_seconds() / 3600) + 1
                for hour in range(start_hour, end_hour):
                    rows.append({
                        id_cfg.id: stay_id,
                        "index_var": float(hour),
                        concept_name: los_val,
                    })

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
            # eICU 数据库没有 unitadmitoffset 列，这是预期行为
            # 所有时间都相对于 ICU 入院时刻（offset=0），这是正确的
            logger.debug(
                "Column '%s' not in eICU patient table; using zero offset (expected behavior).",
                column_name,
            )
            return pd.Series(0, index=frame.index, dtype="float64")
        
        # 🔧 HiRID: 合成 end 时间（从 observations 表获取每个患者的最后观察时间）
        # R ricu 在 id_win_helper.hirid_env 中使用 max(datetime) from observations
        if ds_name == "hirid" and "patientid" in frame.columns:
            try:
                from pathlib import Path
                import duckdb
                
                # 获取目标患者
                target_patients = frame['patientid'].unique().tolist()
                
                # 🚀 优化(2026-02-09): 使用 DuckDB 直接聚合，替代 PyArrow 逐文件扫描
                # 原方案：PyArrow 读250个parquet分片 → 12GB内存, 39s
                # 新方案：DuckDB SELECT MAX(datetime) GROUP BY patientid → ~100MB内存, <2s
                if hasattr(data_source, 'base_path') and data_source.base_path is not None:
                    base_path = Path(data_source.base_path)
                    bucket_path = base_path / 'observations_bucket'
                    obs_path = base_path / 'observations'

                    read_expr = None
                    index_path = base_path / 'observation_tables' / 'observation_tables_index.csv'
                    if index_path.is_file() and obs_path.is_dir():
                        try:
                            obs_index = pd.read_csv(index_path, usecols=['patientid', 'part'])
                            target_set = set(target_patients)
                            target_parts = (
                                obs_index.loc[obs_index['patientid'].isin(target_set), 'part']
                                .dropna()
                                .astype(int)
                                .unique()
                                .tolist()
                            )
                            target_files = []
                            seen_files = set()
                            for part in target_parts:
                                candidates = [
                                    obs_path / f"{part + 1}.parquet",
                                    obs_path / f"{part}.parquet",
                                ]
                                for candidate in candidates:
                                    if candidate.is_file() and candidate not in seen_files:
                                        target_files.append(candidate)
                                        seen_files.add(candidate)
                                        break
                            if target_files:
                                file_list = ", ".join(f"'{_duckdb_path(file)}'" for file in target_files)
                                read_expr = f"read_parquet([{file_list}], union_by_name=true)"
                        except Exception as index_error:
                            logger.debug("HiRID observation part index lookup failed: %s", index_error)

                    if read_expr is None:
                        if bucket_path.is_dir():
                            glob_pattern = _duckdb_path(bucket_path / '**' / '*.parquet')
                            read_expr = f"read_parquet('{glob_pattern}', hive_partitioning=true)"
                        elif obs_path.is_dir():
                            glob_pattern = _duckdb_path(obs_path / '*.parquet')
                            read_expr = f"read_parquet('{glob_pattern}', union_by_name=true)"

                    if read_expr:
                        con = duckdb.connect()
                        con.execute("SET memory_limit = '2GB'")
                        try:
                            numeric_target_patients = [
                                int(pid)
                                for pid in target_patients
                                if pd.notna(pid)
                            ]
                            if numeric_target_patients and len(numeric_target_patients) <= 1000:
                                patient_filter_sql = ", ".join(str(pid) for pid in sorted(set(numeric_target_patients)))
                                result = con.execute(f"""
                                    SELECT o.patientid, MAX(o.datetime) as end_time
                                    FROM {read_expr} o
                                    WHERE o.patientid IN ({patient_filter_sql})
                                    GROUP BY o.patientid
                                """).fetchdf()
                            else:
                                # 将患者ID列表注册为 DuckDB 表以支持大队列过滤
                                pid_df = pd.DataFrame({'patientid': target_patients})
                                con.register('target_pids', pid_df)
                                result = con.execute(f"""
                                    SELECT o.patientid, MAX(o.datetime) as end_time
                                    FROM {read_expr} o
                                    INNER JOIN target_pids t ON o.patientid = t.patientid
                                    GROUP BY o.patientid
                                """).fetchdf()
                            
                            if len(result) > 0:
                                # 与 frame 合并（保持原始索引）
                                merged = frame[['patientid']].reset_index().merge(
                                    result, on='patientid', how='left'
                                ).set_index('index')
                                
                                if 'end_time' in merged.columns:
                                    return merged['end_time']
                        finally:
                            con.close()
                            
            except Exception as e:
                logger.warning(f"Failed to synthesize HiRID end time: {e}")
        
        # 🔧 AUMC/通用: 转换后的 parquet 可能缺少时间戳列（admittedat/dischargedat），
        # 但原始 parquet 仍保留这些列。从原始文件直接读取缺失列。
        if hasattr(data_source, 'base_path') and data_source.base_path is not None:
            try:
                from pathlib import Path
                import pyarrow.parquet as pq
                
                id_cfg = data_source.config.id_configs.get('icustay') or data_source.config.id_configs.get('patient')
                if id_cfg and id_cfg.table:
                    raw_path = Path(data_source.base_path) / f"{id_cfg.table}.parquet"
                    if raw_path.is_file():
                        schema = pq.read_schema(str(raw_path))
                        if column_name in schema.names:
                            id_col = id_cfg.id
                            raw_tbl = pq.read_table(str(raw_path), columns=[id_col, column_name])
                            raw_df = raw_tbl.to_pandas()
                            merged = frame[[id_col]].reset_index().merge(
                                raw_df, on=id_col, how='left'
                            ).set_index('index')
                            if column_name in merged.columns:
                                logger.info(f"Synthesized '{column_name}' from raw parquet for '{ds_name}'")
                                return merged[column_name]
            except Exception as e:
                logger.warning(f"Failed to synthesize column '{column_name}' from raw parquet: {e}")
        
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
        # 🔧 FIX 2025-02-13: Disable fill_gaps for fwd_concept base concepts
        base_tables = self.load_concepts(
            [base_name],
            data_source,
            merge=False,
            aggregate=None,
            patient_ids=patient_ids,
            r_compatible=False,  # 确保返回原始 ICUTable 格式
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
        # 🔧 FIX 2025-03-07: WinTbl 可能带有错误的 value_column 元数据（例如被设成 index 列）
        # 对 fwd_concept 来说应优先从 WinTbl 的实际列中推断值列，否则像 HiRID ett_gcs
        # 这种从 mech_vent 转发再 comp_na 的逻辑会错误地拿 datetime 去比较，最终全为 False。
        if isinstance(base_table, WinTbl):
            id_cols = set(base_table.id_vars or [])
            idx_col = base_table.index_var
            dur_col = base_table.dur_var
            excluded = id_cols | {idx_col, dur_col}
            potential_value_cols = [c for c in data.columns if c not in excluded]
            if base_name in potential_value_cols:
                value_col = base_name
            elif potential_value_cols:
                value_col = potential_value_cols[-1]
            else:
                value_col = base_name
        elif hasattr(base_table, 'value_column') and base_table.value_column:
            value_col = base_table.value_column
        else:
            value_col = base_name

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
            # 🔧 FIX 2025-02-14: Handle WinTbl which uses index_var instead of index_column
            base_idx_col = (base_table.index_var if isinstance(base_table, WinTbl) 
                           else base_table.index_column)
            # 如果 base_table 为空或没有 index_column，返回空的 WinTbl
            if base_idx_col is None or base_table.data.empty:
                # 使用 base_table 的 ID 列（优先），否则使用数据库特定的默认值
                # WinTbl 已在模块顶部导入，不需要重复导入
                
                # 确定数据库特定的默认 ID 列
                db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else 'unknown'
                default_id_cols = _default_id_columns_for_db(db_name)
                
                if isinstance(base_table, WinTbl):
                    id_cols = list(base_table.id_vars) if base_table.id_vars else default_id_cols
                else:
                    id_cols = list(base_table.id_columns) if base_table.id_columns else default_id_cols
                idx_col = base_idx_col if base_idx_col else 'charttime'  # 默认时间列
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
            # 🔧 FIX 2025-02-13: Keep duration in minutes to match R ricu (not convert to hours)
            # R ricu's ts_to_win_tbl uses mins(360L) which means 360 minutes, not 6 hours
            if isinstance(duration, pd.Timedelta):
                duration_mins = duration.total_seconds() / 60.0  # Convert to minutes, not hours
            else:
                duration_mins = float(duration)
            
            # FIX: 为所有行创建 WinTbl，True 行有窗口持续时间，False 行持续时间为 0
            # 这样在 downsampling 时，True 的窗口会扩展，False 的只保留原始时间点
            # 🔧 FIX 2025-02-14: Handle WinTbl which uses id_vars/index_var instead of id_columns/index_column
            base_id_cols = list(base_table.id_vars) if isinstance(base_table, WinTbl) else list(base_table.id_columns)
            base_idx_col = base_table.index_var if isinstance(base_table, WinTbl) else base_table.index_column
            win_df = data[base_id_cols + [base_idx_col]].copy()
            # True 行使用完整窗口持续时间（分钟），False 行使用 0（只表示该时间点存在）
            # 🔧 FIX 2025-02-13: Use 'dur_var' as column name to match R ricu output format
            win_df["dur_var"] = np.where(mask.values, duration_mins, 0.0)
            win_df[concept_name] = mask.values
            return WinTbl(
                data=win_df,  # No rename needed since we use 'dur_var' directly
                id_vars=base_id_cols,
                index_var=base_idx_col,
                dur_var="dur_var",  # Use 'dur_var' to match R ricu
            )

        # 🔧 FIX 2025-02-13: 当 base_table 是 WinTbl 时，保留 dur_var 列
        # 这修复了 HiRID ett_gcs 返回的结果缺少 dur_var 的问题
        is_win_tbl_source = isinstance(base_table, WinTbl)
        
        # 🔧 FIX 2025-02-14: Use id_vars/index_var for WinTbl
        if is_win_tbl_source:
            cols = list(base_table.id_vars)
            idx_col = base_table.index_var
        else:
            cols = list(base_table.id_columns) if hasattr(base_table, 'id_columns') else []
            idx_col = base_table.index_column if hasattr(base_table, 'index_column') else None
        
        if idx_col:
            cols.append(idx_col)
        
        # 如果是 WinTbl，需要保留 dur_var 列
        dur_col = None
        if is_win_tbl_source and hasattr(base_table, 'dur_var') and base_table.dur_var:
            dur_col = base_table.dur_var
            if dur_col in data.columns and dur_col not in cols:
                cols.append(dur_col)
        
        if comp_match:
            # 有 comp_na 比较
            # 🔧 FIX 2025-02-14: 返回所有行，设置布尔值，而不是过滤
            # R ricu 的 comp_na 函数返回 !is.na(x) & op(x, y)，即返回布尔值不是过滤
            # ett_gcs 应该返回所有 mech_vent 行，每行有 True/False 表示是否 invasive
            result = data[cols].copy()
            result[concept_name] = mask  # 布尔值列：True=满足条件，False=不满足
        else:
            # 没有比较，返回原始值
            cols.append(value_col)
            result = data[cols].rename(columns={value_col: concept_name})

        # 如果源是 WinTbl，返回 WinTbl 格式
        # 🔧 FIX 2025-02-14: Use id_vars for WinTbl
        if is_win_tbl_source and dur_col:
            base_id_cols = list(base_table.id_vars) if isinstance(base_table, WinTbl) else list(base_table.id_columns)
            return WinTbl(
                data=result.reset_index(drop=True),
                id_vars=base_id_cols,
                index_var=idx_col,
                dur_var=dur_col,
            )
        
        # 🔧 FIX 2025-02-14: Use id_vars for WinTbl  
        base_id_cols = list(base_table.id_vars) if isinstance(base_table, WinTbl) else list(base_table.id_columns)
        return ICUTable(
            data=result.reset_index(drop=True),
            id_columns=base_id_cols,
            index_column=idx_col,
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
        all_frames: list = []
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
            frame = frame.set_index(key_cols)

            all_frames.append(frame)

        if not all_frames:
            return pd.DataFrame()

        if len(all_frames) == 1:
            merged = all_frames[0].reset_index()
        else:
            # 🚀 优化: 用 pd.concat(axis=1) 替代 sequential join
            # 单次操作完成所有概念的 outer join，避免 N-1 次 _get_indexer
            try:
                merged = pd.concat(all_frames, axis=1).reset_index()
            except Exception:
                # Fallback: sequential join for edge cases (mismatched index levels)
                merged = all_frames[0]
                for frame in all_frames[1:]:
                    if merged.index.nlevels != frame.index.nlevels:
                        common_keys = [col for col in merged.index.names if col in frame.index.names]
                        merged = merged.reset_index()
                        frame = frame.reset_index()
                        merged = merged.merge(frame, on=common_keys, how='outer')
                        merged = merged.sort_values(common_keys)
                        merged = merged.set_index(common_keys)
                    else:
                        merged = merged.join(frame, how="outer", rsuffix='_dup')
                        merged = merged[[c for c in merged.columns if not c.endswith('_dup')]]
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
            "patient_ids": _normalize_patient_ids_for_cache(patient_ids),
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
                
            # Verify the cached data is an ICUTable or WinTbl/TsTbl/IdTbl
            if isinstance(cached_data, ICUTable) or hasattr(cached_data, 'data'):
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
        # 🔧 使用统一的 hash 函数，确保 list 和 dict 形式得到相同的 hash
        patient_ids_hash = _compute_patient_ids_hash(patient_ids)
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
                    # � 缓存中已存储独立副本（在路径D存入时copy），返回时无需再次copy
                    # 调用方如需修改，应自行copy（如_to_r_format已自己copy）
                    cached = self._concept_data_cache[concept_cache_key]
                    return cached
                
                # �🚀🚀 关键优化：如果原始数据已存在于 _raw_concept_cache，
                # 直接从缓存中获取并应用当前的 interval/aggregate，避免重复读取数据库！
                # 这解决了 dopa_dur 在 vaso_ind、sofa_cardio、dopa60 中被重复加载的问题
                is_simple_aggregator = agg_value in (None, False, "auto") or isinstance(agg_value, str)
                # 🔧 FIX: 特定聚合器 (min/max/sum 等) 不应退回到 None/auto 缓存
                # 否则 pafi 请求 po2(min) 会命中批量加载缓存的 po2(median)
                _is_specific_agg = isinstance(agg_value, str) and agg_value not in (None, "auto")
                raw_cached = None
                if is_simple_aggregator:
                    raw_cached = self._get_raw_concept_from_cache(
                        concept_name,
                        patient_ids_hash,
                        aggregator=agg_value,
                        allow_aggregated=not _is_specific_agg,
                    )
                if raw_cached is not None:
                    if hasattr(raw_cached, 'copy'):
                        raw_cached = raw_cached.copy()
                    if verbose and logger.isEnabledFor(logging.DEBUG):
                        logger.debug("🚀 从原始缓存重建概念 '%s' (interval=%s, agg=%s)", concept_name, interval, agg_value)
                    
                    # 应用聚合（如果需要）
                    if agg_value not in (None, False, "auto"):
                        # 🚀 PERF: 如果缓存命中是精确的聚合方式匹配，数据已经聚合完毕，
                        # 跳过冗余的 groupby().agg()（94K 患者下节省 ~30s）
                        exact_key = self._raw_cache_key(concept_name, patient_ids_hash, agg_value)
                        if exact_key in self._raw_concept_cache:
                            result = raw_cached
                        else:
                            result = self._apply_aggregation_to_icutable(
                                raw_cached, concept_name, interval, agg_value
                            )
                    else:
                        result = raw_cached
                    
                    # 缓存处理后的结果
                    self._concept_data_cache[concept_cache_key] = result
                    return result
                
                # �🔧 FIX: 移除旧的简单缓存和概念缓存回退逻辑
                # 这些旧缓存不区分聚合方式，导致错误的缓存命中
                # 例如：safi 内部用 min 聚合加载 o2sat，缓存后
                # 独立加载 o2sat（应该用默认聚合）会错误地命中这个缓存
                # 只使用 _concept_data_cache（包含完整的聚合信息）
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

        # 🔧 FIX 2026-03-09: For recursive concepts (like sofa_cardio), skip pre-loading
        # dependencies that are also sub_concepts. These will be loaded later by the
        # recursive concept callback with the CORRECT per-sub-concept aggregation
        # (e.g., map→min, dopa60→max). Pre-loading them here with 'auto' (median)
        # pollutes _raw_concept_cache, causing the later min/max reload to re-aggregate
        # already-aggregated data (one value per hour), effectively losing the min/max.
        skip_deps = set()
        if definition.sub_concepts and definition.callback:
            # This is a recursive concept with a callback; sub_concepts will be loaded
            # by _load_recursive_concept with the correct aggregation from definition.aggregate
            skip_deps = set(definition.sub_concepts)

        for dependency in definition.depends_on:
            if dependency in skip_deps:
                continue
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
                    self._concept_data_cache[concept_cache_key] = disk_hit
                    self._get_inflight().discard(concept_name)
                # Return a copy to prevent caller from corrupting cached data
                if hasattr(disk_hit, 'copy'):
                    return disk_hit.copy()
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
            # NOTE: Do NOT expand win_tbl target concepts here — they need to stay WinTbl
            # for callbacks that check isinstance(table, WinTbl). Expansion for top-level
            # r_compatible loading happens in load_concepts_enhanced at the return path.
            from .table import WinTbl
            
            definition = self.dictionary.get(concept_name)
            _cls = getattr(definition, 'class_name', None) if definition else None
            _cls_list = _cls if isinstance(_cls, list) else ([_cls] if _cls else ['num_cncpt'])
            _is_true_win_tbl_class = any(c in ('lgl_cncpt', 'fct_cncpt') for c in _cls_list)
            is_win_tbl_target = (
                definition is not None
                and getattr(definition, 'target', 'ts_tbl') == 'win_tbl'
                and _is_true_win_tbl_class
            )
            
            if isinstance(result, WinTbl) and interval is not None and not result.data.empty and not is_win_tbl_target:
                idx_col = result.index_var
                dur_col = result.dur_var
                id_cols = result.id_vars
                
                if idx_col and dur_col and idx_col in result.data.columns and dur_col in result.data.columns:
                    if verbose:
                        logger.info("   扩展 WinTbl '%s' 到时间序列 (interval=%s)", concept_name, interval)
                    
                    # 扩展窗口到时间序列
                    expanded_rows = []
                    interval_hours = interval.total_seconds() / 3600.0
                    for _, row in result.data.iterrows():
                        start_time = row[idx_col]
                        duration = row[dur_col]
                        
                        # 🔧 FIX 2026-03-13: R ricu expand.win_tbl behavior:
                        # In R ricu, merge_time floors start BEFORE expand.win_tbl runs.
                        # So end = re_time(floored_start + dur, interval).
                        # Using raw start gives incorrect expansion for short durations:
                        #   raw: end = floor(45.77 + 0.5) = floor(46.27) = 46 → [45,46] (wrong)
                        #   floored: end = floor(45 + 0.5) = floor(45.5) = 45 → [45] (correct)
                        start_floored = np.floor(start_time / interval_hours) * interval_hours
                        end_time = start_floored + duration
                        if interval_hours > 0:
                            end_time = np.floor(end_time / interval_hours) * interval_hours
                        # Clamp negative end times to 0 (R ricu: x[end < 0, end := 0])
                        end_time = max(end_time, 0.0)
                        
                        # 生成时间序列（每个 interval）
                        current_time = start_floored
                        
                        while current_time <= end_time:
                            new_row = {idx_col: current_time}
                            # 复制 ID 列
                            for col in id_cols:
                                if col in row.index:
                                    new_row[col] = row[col]
                            # 🔧 FIX 2026-03-13: Only copy the concept value column
                            # Extra columns (stop, doseunit, etc.) from original WinTbl
                            # cause value_column detection to pick wrong column.
                            if concept_name in row.index:
                                new_row[concept_name] = row[concept_name]
                            expanded_rows.append(new_row)
                            current_time += interval_hours
                    
                    # 转换为 DataFrame
                    if expanded_rows:
                        expanded_df = pd.DataFrame(expanded_rows)
                    else:
                        expanded_df = pd.DataFrame()
                    

                    
                    # 🔧 FIX 2026-03-11: Incorporate TsTbl source rows (from sources without dur_var)
                    # These were saved in _load_single_concept when combining WinTbl + TsTbl sources
                    ts_rows = getattr(result, '_ts_source_rows', None)
                    if ts_rows is not None and not ts_rows.empty:
                        # Drop dur_var from TsTbl rows (it's all NaN) and align columns
                        ts_rows = ts_rows.drop(columns=[dur_col], errors='ignore')
                        combined_parts = [expanded_df, ts_rows] if not expanded_df.empty else [ts_rows]
                        expanded_df = pd.concat(combined_parts, ignore_index=True)
                    
                    if not expanded_df.empty:
                        # Aggregate by (id, time) using median — matches R ricu's stats::aggregate
                        group_cols = [c for c in id_cols if c in expanded_df.columns] + [idx_col]
                        val_cols = [c for c in expanded_df.columns if c not in group_cols]
                        # Ensure numeric columns for median aggregation
                        for vc in val_cols:
                            if not pd.api.types.is_numeric_dtype(expanded_df[vc]):
                                expanded_df[vc] = pd.to_numeric(expanded_df[vc], errors='coerce')
                        if val_cols:
                            expanded_df = expanded_df.groupby(group_cols, dropna=False)[val_cols].median().reset_index()
                        
                        # 转换为 ICUTable — use concept_name as value_column
                        from .table import ICUTable
                        result = ICUTable(
                            data=expanded_df,
                            id_columns=id_cols,
                            index_column=idx_col,
                            value_column=concept_name if concept_name in expanded_df.columns else None,
                            unit_column=None,
                            time_columns=[],
                        )
                        if verbose:
                            logger.info("   ✅ 扩展完成: %d 行", len(expanded_df))
            
            # 🔧 FIX 2026-02: 对于 target='id_tbl' 的概念（如 height, weight），
            # 需要聚合到每个患者只有一行。R ricu 使用 median 作为数值类型的默认聚合。
            # 参考：ricu/R/tbl-utils.R aggregate.id_tbl 和 ricu/R/concept-load.R load_concepts.num_cncpt
            definition = self.dictionary.get(concept_name)
            if definition and getattr(definition, 'target', None) == 'id_tbl':
                from .table import ICUTable as ICUTableClass
                if isinstance(result, ICUTableClass) and not result.data.empty:
                    df = result.data
                    id_cols = list(result.id_columns)
                    value_col = result.value_column or concept_name
                    
                    if value_col in df.columns and id_cols:
                        # R ricu aggregate.id_tbl: numeric -> median, logical -> any, character -> first
                        col_dtype = df[value_col].dtype
                        if pd.api.types.is_bool_dtype(col_dtype):
                            agg_func = 'any'
                        elif pd.api.types.is_numeric_dtype(col_dtype):
                            agg_func = 'median'
                        else:
                            agg_func = 'first'
                        
                        # 聚合到每个患者一行
                        agg_df = df.groupby(id_cols, as_index=False).agg({value_col: agg_func})
                        
                        if verbose:
                            logger.info("   📊 聚合 id_tbl 概念 '%s': %d 行 -> %d 行 (agg=%s)", 
                                       concept_name, len(df), len(agg_df), agg_func)
                        
                        result = ICUTableClass(
                            data=agg_df,
                            id_columns=id_cols,
                            index_column=None,  # id_tbl 没有时间列
                            value_column=value_col,
                            unit_column=None,
                            time_columns=[],
                        )
                        
        except Exception:
            with self._cache_lock:
                self._get_inflight().discard(concept_name)
            raise

        # 🔧 如果 _skip_concept_cache=True，跳过缓存写入
        if not _skip_concept_cache:
            self._store_in_disk_cache(concept_name, data_source, cache_key, result)

            with self._cache_lock:
                # 🔧 FIX: 缓存写入使用共享引用（不 copy）
                # 缓存会在顶层调用结束后清除，不需要防护性 copy
                # 读取时再 copy，节省内存
                self._concept_data_cache[concept_cache_key] = result
                
                # 🚀 同时存入 _raw_concept_cache，供回调函数使用
                # 共享同一个引用，避免重复内存开销
                # 🔧 FIX 2026-03-10: Include aggregator in key to prevent cross-aggregation pollution
                self._store_raw_concept_cache(
                    concept_name,
                    patient_ids_hash,
                    result,
                    aggregator=agg_value,
                    store_legacy=interval is None and agg_value in (None, False, "auto"),
                )
                
                self._get_inflight().discard(concept_name)
        else:
            # 仅清除 inflight 标记
            with self._cache_lock:
                self._get_inflight().discard(concept_name)
        return result

    def _apply_aggregation_to_icutable(
        self,
        table: ICUTable,
        concept_name: str,
        interval: Optional[pd.Timedelta],
        aggregator: object,
    ) -> ICUTable:
        """从原始 ICUTable 应用聚合，返回新的 ICUTable。
        
        这个方法用于从 _raw_concept_cache 重建处理后的数据，
        避免重复读取数据库。
        
        Args:
            table: 原始 ICUTable
            concept_name: 概念名称
            interval: 时间间隔
            aggregator: 聚合方式
            
        Returns:
            处理后的 ICUTable
        """
        frame = table.data.copy()
        # 🔧 FIX 2026-02-06: Support both ICUTable (.id_columns) and WinTbl/TsTbl (.id_vars)
        if hasattr(table, 'id_vars'):
            id_columns = list(table.id_vars)
            index_column = getattr(table, 'index_var', None)
            unit_column = None
            value_column = None
        else:
            id_columns = list(table.id_columns)
            index_column = table.index_column
            unit_column = table.unit_column
            value_column = table.value_column
        
        # 如果不需要聚合，直接返回副本
        # "auto" 也表示不聚合（让后续流程决定）
        if aggregator in (None, False, "auto"):
            return ICUTable(
                data=frame,
                id_columns=id_columns,
                index_column=index_column,
                value_column=value_column,
                unit_column=unit_column,
            )
        
        # 应用聚合
        aggregated_frame = self._apply_aggregation(
            frame,
            concept_name if concept_name in frame.columns else (value_column or concept_name),
            id_columns,
            index_column,
            unit_column,
            aggregator,
        )
        
        # 🔧 FIX 2025-01-31: Handle empty tables gracefully
        # When the original table is empty (e.g., dopa_dur for HiRID), 
        # don't try to use concept_name as value_column if it doesn't exist
        final_value_column = value_column
        if value_column and value_column in aggregated_frame.columns:
            final_value_column = value_column
        elif concept_name in aggregated_frame.columns:
            final_value_column = concept_name
        else:
            # For empty tables, keep value_column as None
            final_value_column = None
        
        return ICUTable(
            data=aggregated_frame,
            id_columns=id_columns,
            index_column=index_column,
            value_column=final_value_column,
            unit_column=unit_column if unit_column and unit_column in aggregated_frame.columns else None,
        )

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
        # R ricu does: aggregate(expand(...)), but easyicu was skipping expand
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

    def _to_r_format(self, icu_table: ICUTable, concept_name: str, interval_hours: float = 1.0) -> pd.DataFrame:
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
            if concept_name in compat.WINDOW_CONCEPTS or concept_name.endswith('_rate'):
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
                    expanded = compat.expand_interval_rows(
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

    def _to_r_format_merged(self, merged_df: pd.DataFrame, concept_names: List[str]) -> pd.DataFrame:
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
        [col for col in frame.columns if any(id_key in col.lower() for id_key in ['id', 'stay_id', 'subject_id', 'patient'])]

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

    def _to_r_format_merged_enhanced(
        self, 
        tables: Mapping[str, ICUTable], 
        concept_names: List[str],
        interval: Optional[pd.Timedelta] = None,
        data_source: Optional['ICUDataSource'] = None,  # 🔧 FIX 2025-01-31: 添加数据源参数
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
            data_source: 数据源，用于确定默认ID列
            
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
        
        # 🔧 FIX 2025-01-31: 根据数据源确定默认ID列
        default_id_col = 'stay_id'
        if data_source is not None and hasattr(data_source, 'config'):
            db_name = getattr(data_source.config, 'name', '')
            if db_name in ['eicu', 'eicu_demo']:
                default_id_col = 'patientunitstayid'
            elif db_name == 'aumc':
                default_id_col = 'admissionid'
            elif db_name == 'hirid':
                default_id_col = 'patientid'
            elif db_name == 'sic':
                default_id_col = 'CaseID'  # SICdb uses CaseID (uppercase)
            elif db_name == 'mimic':
                default_id_col = 'icustay_id'  # MIMIC-III uses icustay_id
            # 也可以从 id_configs 获取
            if hasattr(data_source.config, 'id_configs') and 'icustay' in data_source.config.id_configs:
                default_id_col = data_source.config.id_configs['icustay'].id
        
        # 将ICUTable/WinTbl/IdTbl转换为DataFrame字典
        # 🔧 FIX 2026-03-10: WinTbl inherits from IdTbl (not ICUTable), so isinstance(table, ICUTable)
        # fails for WinTbl/TsTbl/IdTbl. Use hasattr(table, 'data') as a generic check.
        # 🚀 优化：不再 .copy()。merge_concepts_r_style 的 rename/drop_duplicates 
        # 都会创建新对象，不会修改原始 DataFrame。缓存在顶层 finally 中清空。
        concept_data: Dict[str, pd.DataFrame] = {}
        for name, table in tables.items():
            if isinstance(table, ICUTable):
                df = table.data
                # 重命名值列为概念名
                if name not in df.columns:
                    # 查找可能的值列 - 优先使用 ICUTable 元数据中的 value_column
                    value_candidates = []
                    if table.value_column and table.value_column in df.columns:
                        value_candidates.append(table.value_column)
                    value_candidates.extend(['value', 'valuenum'])
                    for cand in value_candidates:
                        if cand in df.columns and cand != name:
                            df = df.rename(columns={cand: name})
                            break
                concept_data[name] = df
            elif hasattr(table, 'data') and isinstance(table.data, pd.DataFrame):
                # Handle WinTbl/TsTbl/IdTbl which have .data but don't inherit from ICUTable
                df = table.data
                if name not in df.columns:
                    # For WinTbl, try index_var as value column candidate
                    value_candidates = ['value', 'valuenum']
                    if hasattr(table, 'index_var') and table.index_var:
                        value_candidates.append(table.index_var)
                    for cand in value_candidates:
                        if cand in df.columns and cand != name:
                            df = df.rename(columns={cand: name})
                            break
                concept_data[name] = df
            elif isinstance(table, pd.DataFrame):
                df = table
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
        _time_candidates = ['charttime', 'datetime', 'givenat', 'time', 'starttime', 'start', 'index_var',
                     'measuredat_minutes', 'measuredat',  # AUMC: measuredat_minutes first!
                     'Offset', 'offset',  # SICdb: Offset (uppercase)
                     'nursingchartoffset', 'labresultoffset', 'observationoffset',
                     'respchartoffset', 'intakeoutputoffset', 'infusionoffset']
        for df in concept_data.values():
            if df is None or df.empty:
                continue
            id_candidates = [default_id_col, 'CaseID', 'caseid', 'icustay_id', 'stay_id', 'subject_id', 'patientunitstayid', 'admissionid', 'patientid']
            for cand in id_candidates:
                if cand in df.columns:
                    id_col = cand
                    break
            for cand in _time_candidates:
                if cand in df.columns:
                    time_col = cand
                    break
            if id_col and time_col:
                break
        
        if not id_col:
            id_col = default_id_col
        if not time_col:
            time_col = 'charttime'
        
        # 🔧 FIX: 标准化每个概念 DataFrame 的时间列名
        # DuckDB path 输出 'charttime'，raw path 输出原始列名（如 SIC 的 'Offset'）
        # 如果不统一，merge 时会因时间列名不同而丢失数据
        for name in list(concept_data.keys()):
            df = concept_data[name]
            if df is None or df.empty:
                continue
            if time_col in df.columns:
                continue
            for cand in _time_candidates:
                if cand in df.columns:
                    concept_data[name] = df.rename(columns={cand: time_col})
                    break
        
        # 使用compat模块进行合并
        result = compat.merge_concepts_r_style(
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
        
        # 统一时间列名为 'charttime' 以保持所有数据库输出一致性
        if time_col != 'charttime' and time_col in result.columns:
            result = result.rename(columns={time_col: 'charttime'})
        
        # 🚀 内存优化：将 float64 值列降级为 float32
        # 临床数据精度（HR=80, MAP=65.5, FiO2=0.21）完全在 float32 范围内（~7位有效数字）
        # 对 198K 患者 × 15 概念可节省 ~800MB
        import numpy as np
        for col in result.columns:
            if result[col].dtype == np.float64 and col not in (id_col, 'charttime', time_col):
                result[col] = result[col].astype(np.float32)
        
        return result


@functools.lru_cache(maxsize=1)
def _load_concept_dict_cached():
    """Load concept-dict.json once and cache in memory."""
    import json
    dict_path = Path(__file__).parent / 'data' / 'concept-dict.json'
    with open(dict_path) as f:
        return json.load(f)


def _get_concept_bounds(concept_name: str, bound: str) -> Optional[float]:
    """Get min/max bounds from concept-dict.json for filter_bounds."""
    try:
        d = _load_concept_dict_cached()
        c = d.get(concept_name, {})
        val = c.get(bound)
        return float(val) if val is not None else None
    except Exception:
        return None


def _apply_callback(
    frame: pd.DataFrame,
    source: ConceptSource,
    concept_name: str,
    unit_column: Optional[str] = None,
    resolver: Optional['ConceptResolver'] = None,
    patient_ids: Optional[List] = None,
    data_source: Optional['ICUDataSource'] = None,
    interval: Optional[Union[str, pd.Timedelta]] = None,
) -> pd.DataFrame:
    callback = source.callback
    if not callback:
        return frame

    expr = callback.strip()

    if expr == "identity_callback":
        return frame

    if expr == "aumc_death":
        # R ricu logic: is_true(index_var - val_var < hours(72L))
        # where index_var = dateofdeath, val_var = dischargedat
        # AUMC times are in milliseconds, 72 hours = 72 * 3600 * 1000 = 259200000 ms
        # is_true(x) returns TRUE if x is TRUE (not NA)
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
        # Use raw millisecond values directly (like R ricu does)
        # Don't convert to datetime - AUMC stores times as milliseconds relative to admission
        dateofdeath = pd.to_numeric(df[index_col], errors='coerce')
        dischargedat = pd.to_numeric(df[value_col], errors='coerce')
        
        # 72 hours in milliseconds
        hours_72_ms = 72 * 3600 * 1000
        
        # Calculate diff in ms
        diff_ms = dateofdeath - dischargedat
        
        # is_true: TRUE if dateofdeath is not NA AND (dateofdeath - dischargedat) < 72h
        # For rows where dateofdeath is NA, result should be NA (not FALSE)
        # This matches ricu behavior where survived patients have death=NA
        within_window = (diff_ms < hours_72_ms) & dateofdeath.notna()
        
        # Set death value: TRUE if within 72h, FALSE if beyond 72h, NA if no dateofdeath
        # Use object dtype to support True/False/None
        death_values = pd.Series(index=df.index, dtype=object)
        death_values[dateofdeath.notna()] = within_window[dateofdeath.notna()]
        # Rows with dateofdeath NA remain as None (NA)
        
        df[value_col] = death_values
        return df

    # 🔧 SICdb death callback — OffsetOfDeath in seconds, NaN = survived
    if expr == "sic_death":
        df = frame.copy()
        # OffsetOfDeath is both index_var and val_var, so it gets renamed to
        # concept_name ('death') before this callback runs. The numeric values
        # (seconds from ICU admission) are now in the 'death' column.
        offset_col = None
        # First try the original column name
        for c in ['OffsetOfDeath', 'offsetofdeath']:
            if c in df.columns:
                offset_col = c
                break
        # Fallback: val_var was renamed to concept_name
        if offset_col is None and concept_name in df.columns:
            offset_col = concept_name

        if offset_col is None:
            return df.head(0)

        offset_vals = pd.to_numeric(df[offset_col], errors='coerce')
        # death = TRUE if OffsetOfDeath is not NaN, NA otherwise (matches ricu behavior)
        death_values = pd.Series(index=df.index, dtype=object)
        death_values[offset_vals.notna()] = True
        # NaN = survived (NA, not FALSE)
        df[concept_name] = death_values
        # Add charttime as OffsetOfDeath converted to hours
        df['charttime'] = offset_vals / 3600.0
        return df

    # 🔧 HiRID death callback — matches R ricu hirid_death (callback-itm.R:197)
    # R ricu flow:
    #   1. Load observations for variableid IN [110, 200]
    #   2. dt_gforce(x, "last", by=idc, vars=idx) → last observation time per patient
    #   3. load_id(env[["general"]], cols="discharge_status") → load general table
    #   4. merge with dead patients → keep only patients who died
    #   5. Set val_var = TRUE
    # 🚀 优化: variableids 110, 200 有 115M 行（高频数据），
    #    直接在 DuckDB 中 GROUP BY patientid → MAX(datetime) 避免加载全量。
    if expr == "hirid_death":
        id_col = 'patientid'
        
        # Step 1: Load general table and find dead patients
        dead_pids = set()
        if data_source is not None:
            try:
                general_tbl = data_source.load_table('general', columns=[id_col, 'discharge_status'])
                general_df = general_tbl.data if hasattr(general_tbl, 'data') else general_tbl
                if not isinstance(general_df, pd.DataFrame):
                    general_df = pd.DataFrame(general_df)
                dead_pids = set(general_df.loc[
                    general_df['discharge_status'].astype(str).str.lower() == 'dead',
                    id_col
                ].unique())
            except Exception:
                pass
        
        if not dead_pids:
            return frame.head(0) if hasattr(frame, 'head') else pd.DataFrame()
        
        # Step 2: 🚀 使用 DuckDB 直接聚合获取最后观测时间（避免加载 115M 行）
        last_obs = None
        if data_source is not None and hasattr(data_source, 'base_path'):
            bucket_dir = data_source.base_path / 'observations_bucket'
            if bucket_dir.exists():
                try:
                    import duckdb
                    conn = duckdb.connect()
                    conn.execute("SET memory_limit = '2GB'")
                    glob_pattern = _duckdb_path(bucket_dir / 'bucket_id=*' / '*.parquet')
                    dead_pids_str = ', '.join(str(p) for p in dead_pids)
                    query = f"""
                        SELECT patientid, MAX(datetime) as datetime
                        FROM read_parquet('{glob_pattern}', union_by_name=true)
                        WHERE variableid IN (110, 200)
                          AND patientid IN ({dead_pids_str})
                        GROUP BY patientid
                    """
                    last_obs = conn.execute(query).fetchdf()
                    conn.close()
                except Exception:
                    pass
        
        # Fallback: 使用已加载的 frame（旧行为）
        if last_obs is None or last_obs.empty:
            df = frame.copy() if hasattr(frame, 'copy') else pd.DataFrame()
            if id_col in df.columns:
                time_col = 'datetime' if 'datetime' in df.columns else ('charttime' if 'charttime' in df.columns else None)
                if time_col:
                    last_obs = df.groupby(id_col, as_index=False).agg({time_col: 'max'})
                    last_obs = last_obs[last_obs[id_col].isin(dead_pids)]
        
        if last_obs is None or last_obs.empty:
            return frame.head(0) if hasattr(frame, 'head') else pd.DataFrame()
        
        # Step 3: Set death = TRUE
        result = last_obs.copy()
        result[concept_name] = True
        
        return result

    # Handle eicu_age - process eICU age data (convert '> 89' to 90)
    if re.fullmatch(r"transform_fun\(eicu_age\)", expr):
        from .callback_utils import eicu_age
        return eicu_age(frame, val_col=concept_name)

    # Handle eicu_adx - process eICU admission diagnosis to categorize as med/surg/other
    if expr == "eicu_adx":
        """
        Map eICU admitdxpath to admission type (med/surg/other).
        
        The admitdxpath contains hierarchical diagnosis path like:
        "admission diagnosis|All Diagnosis|Operative|Diagnosis|Cardiovascular|..."
        "admission diagnosis|All Diagnosis|Non-operative|Diagnosis|Genitourinary|..."
        
        Rules from R ricu (callback-itm.R eicu_adx):
        1. Split path by "|"
        2. Keep only rows where parts[1] == "All Diagnosis"
        3. If parts[4] in ["Genitourinary", "Transplant"] -> "other"
        4. Else if parts[2] == "Operative" -> "surg"
        5. Else -> "med"
        """
        frame = frame.copy()
        
        # Get the diagnosis path column
        # 🔧 FIX: 回调在重命名后调用，所以 value_var (admitdxpath) 已变为 concept_name (adm)
        # 优先使用 concept_name，然后再尝试 source.value_var
        val_col = None
        # 1. 优先使用 concept_name（重命名后的列名）
        if concept_name in frame.columns:
            val_col = concept_name
        # 2. 如果 concept_name 不存在，尝试 source.value_var
        elif source.value_var and source.value_var in frame.columns:
            val_col = source.value_var
        # 3. 最后尝试常见列名
        else:
            for col in ['admitdxpath', 'diagnosispath', 'diagnosis']:
                if col in frame.columns:
                    val_col = col
                    break
        
        if val_col is None:
            # No diagnosis column found, return empty
            frame[concept_name] = pd.Series(dtype='object')
            return frame
        
        def classify_adm_type(path):
            if pd.isna(path):
                return None  # Will be filtered out
            
            parts = str(path).split('|')
            
            # Require at least 3 segments (0, 1, 2) and check parts[1] == "All Diagnosis"
            if len(parts) < 3:
                return None
            
            if parts[1].strip() != "All Diagnosis":
                return None
            
            # Check parts[4] for Genitourinary or Transplant (if exists)
            if len(parts) > 4:
                seg4 = parts[4].strip()
                if seg4 in ["Genitourinary", "Transplant"]:
                    return 'other'
            
            # Check parts[2] for Operative
            seg2 = parts[2].strip()
            if seg2 == "Operative":
                return 'surg'
            
            # Default to med (Non-operative)
            return 'med'
        
        frame[concept_name] = frame[val_col].apply(classify_adm_type)
        
        # Filter out None values (rows that didn't match "All Diagnosis" criteria)
        frame = frame[frame[concept_name].notna()].copy()
        
        # Drop the original diagnosis path column if it's different from concept_name
        if val_col != concept_name and val_col in frame.columns:
            frame = frame.drop(columns=[val_col])
        
        return frame

    # Handle percent_as_numeric - remove '%' and convert to numeric
    if re.fullmatch(r"transform_fun\(percent_as_numeric\)", expr):
        series = frame[concept_name]

        # 🚀 Fast path: if already numeric (DuckDB pre-processed), skip all string ops
        if pd.api.types.is_numeric_dtype(series):
            na_mask = series.isna()
            if na_mask.any():
                for fallback_col in ("value", "valuetext"):
                    if fallback_col in frame.columns and fallback_col != concept_name:
                        fallback = pd.to_numeric(frame[fallback_col], errors='coerce')
                        series = series.where(~na_mask, fallback)
                        na_mask = series.isna()
                        if not na_mask.any():
                            break
                frame.loc[:, concept_name] = series
            return frame

        # 🚀 Optimized slow path: try to_numeric first, only strip '%' on failures
        # Most string values are plain numbers ('50', '0.21') that pd.to_numeric handles directly.
        # Only actual percent strings ('50%') need the rstrip. This reduces _str_map calls from
        # 2 (strip+rstrip on ALL 12.9M rows) to at most 1 (rstrip on the small failed subset).
        na_mask = series.isna()
        if na_mask.any():
            for fallback_col in ("value", "valuetext"):
                if fallback_col in frame.columns and fallback_col != concept_name:
                    series = series.where(~na_mask, frame[fallback_col])
                    na_mask = series.isna()
                    if not na_mask.any():
                        break

        result = pd.to_numeric(series, errors='coerce')
        failed_mask = result.isna() & series.notna()
        if failed_mask.any():
            fixed = pd.to_numeric(
                series[failed_mask].astype(str).str.rstrip('%'), errors='coerce'
            )
            result = result.copy()
            result.loc[failed_mask] = fixed
        frame.loc[:, concept_name] = result
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
        # 🔧 FIX: 确定要检查的列 - 优先使用 source.value_var，否则用 concept_name
        # MIMIC-III 的列名是大写的，需要智能匹配
        val_col = source.value_var if source.value_var else concept_name
        if val_col not in frame.columns:
            # 尝试大写/小写匹配
            col_map = {c.lower(): c for c in frame.columns}
            if val_col.lower() in col_map:
                val_col = col_map[val_col.lower()]
            elif concept_name.lower() in col_map:
                val_col = col_map[concept_name.lower()]
            else:
                # 最后尝试用原始列名
                for col in frame.columns:
                    if 'itemid' in col.lower() or 'org' in col.lower():
                        val_col = col
                        break
        series = frame[val_col]
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
        # 🔧 FIX: 确定要比较的列 - 优先使用 concept_name，如果不存在则使用 source.value_var
        # 这修复了 ett_gcs 等概念在 callback 执行前 value_column 尚未重命名的情况
        compare_col = concept_name
        if compare_col not in frame.columns:
            # 尝试使用 source.value_var
            if source.value_var and source.value_var in frame.columns:
                compare_col = source.value_var
            # 或者尝试常见的值列名
            elif 'value' in frame.columns:
                compare_col = 'value'
            elif 'valuenum' in frame.columns:
                compare_col = 'valuenum'
        
        if compare_col not in frame.columns:
            # 如果仍然找不到列，返回原始 frame（不做任何处理）
            return frame
        
        series = frame[compare_col]
        if isinstance(value, (int, float)) and not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
        comparator = op_map[op_token]
        # 🚀 Vectorized comparison instead of .apply(lambda) — avoids N python calls
        na_mask = series.isna()
        comparison = comparator(series, value)
        comparison = comparison.where(~na_mask, False).astype("boolean")
        frame = frame.copy()
        # 如果比较的是 concept_name 列，删除它；否则保留原列
        if compare_col == concept_name:
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

    # Handle transform_fun(floor) - apply floor function to values
    if re.fullmatch(r"transform_fun\(floor\)", expr):
        frame = frame.copy()
        val_col = concept_name if concept_name in frame.columns else (source.value_var or 'value')
        if val_col in frame.columns:
            frame[val_col] = pd.to_numeric(frame[val_col], errors='coerce').apply(np.floor)
        return frame

    # Handle transform_fun(ceiling) or transform_fun(ceil) - apply ceiling function
    if re.fullmatch(r"transform_fun\(ceil(ing)?\)", expr):
        frame = frame.copy()
        val_col = concept_name if concept_name in frame.columns else (source.value_var or 'value')
        if val_col in frame.columns:
            frame[val_col] = pd.to_numeric(frame[val_col], errors='coerce').apply(np.ceil)
        return frame

    # Handle transform_fun(round) - apply round function
    if re.fullmatch(r"transform_fun\(round\)", expr):
        frame = frame.copy()
        val_col = concept_name if concept_name in frame.columns else (source.value_var or 'value')
        if val_col in frame.columns:
            frame[val_col] = pd.to_numeric(frame[val_col], errors='coerce').round()
        return frame

    # Handle aggregate_fun('sum', 'units') - aggregate by sum and set unit
    match = re.fullmatch(r"aggregate_fun\(['\"](\w+)['\"],\s*['\"](.+?)['\"]\)", expr)
    if match:
        agg_func = match.group(1)  # e.g., 'sum'
        new_unit = match.group(2)  # e.g., 'units'
        
        frame = frame.copy()
        val_col = concept_name if concept_name in frame.columns else (source.value_var or 'value')
        unit_col = source.unit_var
        
        # Identify ID and time columns
        id_col = None
        for cand in ['patientid', 'stay_id', 'admissionid', 'patientunitstayid', 'subject_id']:
            if cand in frame.columns:
                id_col = cand
                break
        
        time_col = None
        for cand in ['datetime', 'charttime', 'time', 'givenat']:
            if cand in frame.columns:
                time_col = cand
                break
        
        if id_col and time_col and val_col in frame.columns:
            # Convert to numeric
            frame[val_col] = pd.to_numeric(frame[val_col], errors='coerce')
            
            # Group by id and time, apply aggregation
            group_cols = [id_col, time_col]
            if agg_func == 'sum':
                result = frame.groupby(group_cols, as_index=False)[val_col].sum()
            elif agg_func == 'mean':
                result = frame.groupby(group_cols, as_index=False)[val_col].mean()
            elif agg_func == 'max':
                result = frame.groupby(group_cols, as_index=False)[val_col].max()
            elif agg_func == 'min':
                result = frame.groupby(group_cols, as_index=False)[val_col].min()
            else:
                result = frame  # Unknown aggregation, return as-is
            
            # Set unit
            if unit_col:
                result[unit_col] = new_unit
            
            return result
        
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
            frame_result = _apply_callback(
                frame_result, nested_source, concept_name, unit_column,
                resolver=resolver, patient_ids=patient_ids, data_source=data_source,
                interval=interval,
            )
        return frame_result
    
    # Handle dex_to_10 callback (convert different dextrose concentrations to D10 equivalent)
    # Format: dex_to_10(ids, factors) or dex_to_10(c(...), c(...)) or dex_to_10(list(...), c(...))
    match = re.fullmatch(r"dex_to_10\((.+)\)", expr, flags=re.DOTALL)
    if match:
        args = _split_arguments(match.group(1))
        if len(args) >= 2:
            # Parse itemids and factors using _parse_r_value which handles nested list/c structures
            id_arg = args[0].strip()
            factor_arg = args[1].strip()
            
            try:
                itemids = _parse_r_value(id_arg)    # e.g., list(7255L, 7256L, c(8940L, 9571L)) -> [7255, 7256, [8940, 9571]]
                factors = _parse_r_value(factor_arg)  # e.g., c(2, 3, 4) -> [2, 3, 4]
                
                # 🔧 FIX: Ensure itemids and factors are always lists (handle single-value case)
                # e.g., dex_to_10(30017L, c(0.5)) → itemids=30017 (int), factors=[0.5]
                if not isinstance(itemids, (list, tuple)):
                    itemids = [itemids]
                if not isinstance(factors, (list, tuple)):
                    factors = [factors]
                
                # Flatten itemids if needed for simple structure, or handle nested mapping
                # Nested structure means: itemids[i] can be a list, all items in that list get factors[i]
                
                # Apply conversion factors
                sub_var = source.sub_var if hasattr(source, 'sub_var') else 'itemid'
                # 🔧 FIX: 确定值列
                # 对于 MIIV: mimv_rate 会将计算结果写入 rate 列
                # 对于 AUMC drugitems: 默认值列是 dose（不是 rate）
                # 策略：优先使用 source 配置的 val_var，然后 concept_name，最后回退到 dose/rate
                val_col = None
                # 1. 优先使用 source 配置的 value_var
                if hasattr(source, 'value_var') and source.value_var and source.value_var in frame.columns:
                    val_col = source.value_var
                # 2. 使用 concept_name 列（如果已创建）
                elif concept_name in frame.columns:
                    val_col = concept_name
                # 3. AUMC drugitems 默认用 dose 列
                elif 'dose' in frame.columns and frame['dose'].notna().any():
                    val_col = 'dose'
                # 4. 回退到 rate 列（MIIV inputevents）
                elif 'rate' in frame.columns and frame['rate'].notna().any():
                    val_col = 'rate'
                # 5. 其他常见值列
                elif 'amount' in frame.columns:
                    val_col = 'amount'
                elif 'valuenum' in frame.columns:
                    val_col = 'valuenum'
                
                if sub_var in frame.columns and val_col:
                    frame = frame.copy()
                    for ids_item, factor in zip(itemids, factors):
                        # ids_item can be a single value or a list
                        if isinstance(ids_item, (list, tuple)):
                            ids_to_check = ids_item
                        else:
                            ids_to_check = [ids_item]
                        
                        for itemid in ids_to_check:
                            mask = frame[sub_var] == itemid
                            if mask.any():
                                frame.loc[mask, val_col] = frame.loc[mask, val_col] * factor
            except Exception as e:
                # Log warning but continue
                import logging
                logging.warning(f"dex_to_10 parsing failed: {e}")
        return frame
    
    # Handle grp_mount_to_rate callback (convert grouped amounts to rates)
    # Format: grp_mount_to_rate(mins(1L), hours(1L)) or similar
    match = re.fullmatch(r"grp_mount_to_rate\((.+)\)", expr, flags=re.DOTALL)
    if match:
        args = _split_arguments(match.group(1))
        if len(args) >= 2:
            try:
                # Parse min_dur and extra_dur
                min_dur_expr = args[0].strip()
                extra_dur_expr = args[1].strip()
                
                def _parse_duration_expr(dur_expr: str) -> pd.Timedelta:
                    """Parse R duration expression like mins(1L), hours(1L)."""
                    if 'mins(' in dur_expr:
                        mins_match = re.search(r'mins\((\d+)', dur_expr)
                        if mins_match:
                            return pd.Timedelta(minutes=int(mins_match.group(1)))
                    elif 'hours(' in dur_expr:
                        hours_match = re.search(r'hours\((\d+)', dur_expr)
                        if hours_match:
                            return pd.Timedelta(hours=int(hours_match.group(1)))
                    elif 'secs(' in dur_expr:
                        secs_match = re.search(r'secs\((\d+)', dur_expr)
                        if secs_match:
                            return pd.Timedelta(seconds=int(secs_match.group(1)))
                    # Default to 1 minute
                    return pd.Timedelta(minutes=1)
                
                min_dur = _parse_duration_expr(min_dur_expr)
                extra_dur = _parse_duration_expr(extra_dur_expr)
                
                # Get group variable from source
                # 🔧 FIX: grp_var 可能在 source.grp_var 或 source.params['grp_var'] 中
                grp_var = None
                if hasattr(source, 'grp_var') and source.grp_var:
                    grp_var = source.grp_var
                elif hasattr(source, 'params') and source.params and 'grp_var' in source.params:
                    grp_var = source.params['grp_var']
                
                # Get value and unit columns
                val_col = source.value_var if hasattr(source, 'value_var') and source.value_var else None
                if not val_col:
                    # Try common value columns
                    for candidate in ['val', 'value', 'amount', 'dose', 'givendose', 'pharmavalue']:
                        if candidate in frame.columns:
                            val_col = candidate
                            break
                if not val_col:
                    val_col = concept_name
                
                unit_col = source.unit_var if hasattr(source, 'unit_var') and source.unit_var else None
                if not unit_col:
                    for candidate in ['unit', 'unit_var', 'amountuom', 'doserateunit', 'doseunit']:
                        if candidate in frame.columns:
                            unit_col = candidate
                            break
                
                # Get index_var (time column)
                index_var = source.index_var if hasattr(source, 'index_var') and source.index_var else None
                if not index_var:
                    for candidate in ['datetime', 'givenat', 'starttime', 'charttime', 'time']:
                        if candidate in frame.columns:
                            index_var = candidate
                            break
                
                # Get ID columns - only use standard patient/stay ID columns
                # 🔧 FIX: Don't use "id" substring matching - it incorrectly includes columns like
                # 'fluidamount_calc', 'typeid', 'subtypeid' which contain "id" but are not patient IDs
                standard_id_cols = ['patientid', 'stay_id', 'admissionid', 'patientunitstayid', 'subject_id', 'hadm_id', 'icustay_id']
                id_cols = [col for col in standard_id_cols if col in frame.columns]
                
                from .callback_utils import grp_mount_to_rate as grp_mount_fn
                callback_fn = grp_mount_fn(
                    min_dur=min_dur,
                    extra_dur=extra_dur,
                    grp_var=grp_var
                )
                
                result = callback_fn(
                    frame,
                    val_col=val_col if val_col in frame.columns else (concept_name if concept_name in frame.columns else 'value'),
                    unit_col=unit_col if unit_col and unit_col in frame.columns else 'unit',
                    index_var=index_var,
                    id_cols=id_cols
                )
                
                return result
            except Exception as e:
                import logging
                logging.warning(f"grp_mount_to_rate parsing failed: {e}")
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
        
        # 🔧 FIX: 检测时间列的类型，确保dur_var与其兼容
        # 如果时间列是数值型（小时），则dur_var也应该是数值型（小时）
        # 🔧 FIX 2026-02-15: 添加 measuredat 支持 AUMC
        index_col = None
        for col in ['charttime', 'starttime', 'start', 'time', 'measuredat', 'measuredat_minutes', 'datetime']:
            if col in frame.columns:
                index_col = col
                break
        
        if index_col and index_col in frame.columns and pd.api.types.is_numeric_dtype(frame[index_col]):
            # 时间列是数值型（小时或分钟），dur_var用分钟数值
            # R ricu: ts_to_win_tbl(mins(1L)) → dur_var = difftime(1, units="mins")
            # 写入CSV时序列化为数值 1.0（分钟）
            frame['dur_var'] = duration.total_seconds() / 60.0  # 转换为分钟
        else:
            # 🔧 FIX: 始终使用数值分钟，而非 Timedelta 对象
            # 原因：后续 _align_to_admission_time 会将 datetime → 相对小时数，
            # 但 Timedelta dur_var 会被转为 int64 纳秒（而非分钟），
            # 导致 _expand_public_numeric_win_tbl_output 中 duration 被误解为
            # 60 000 000 000 小时 → 无限循环。
            # R ricu 的 dur_var 也是数值型（分钟），所以统一用分钟。
            frame['dur_var'] = duration.total_seconds() / 60.0  # 转换为分钟
            
        return frame
    
    # Handle mimic_rate_mv callback (for infusion rates)
    if expr.strip() == "mimic_rate_mv":
        from .callback_utils import mimic_rate_mv
        # Call the callback with appropriate parameters
        id_cols = [col for col in frame.columns if 'id' in col.lower()]
        # stop_var is stored in params dict
        stop_var = source.params.get('stop_var', None) if source.params else None
        unit_col = source.unit_var if hasattr(source, 'unit_var') else None
        # 🔧 FIX: mimic_rate_mv 应使用表的 'rate' 列，而不是 concept_name
        # R ricu 中 mimic_rate_mv 使用 inputevents 表的 'rate' 列作为输出值
        # 原始数据中 'rate' 是速率 (mcg/kg/min)，'amount' 是总量 (mg)
        val_col = 'rate' if 'rate' in frame.columns else concept_name
        
        # 🔧 CRITICAL FIX 2024-11-30: Get admission times for R ricu-compatible floor behavior
        # R ricu converts datetime to relative time BEFORE callbacks (in load_mihi).
        # This affects floor() behavior in expand().
        admission_times = None
        if data_source is not None:
            try:
                # Load icustays to get admission times
                icustays_result = data_source.load_table('icustays')
                # Handle ICUTable or DataFrame result
                if hasattr(icustays_result, 'data'):
                    icustays = icustays_result.data
                else:
                    icustays = icustays_result
                    
                if icustays is not None and len(icustays) > 0:
                    # Find ID column
                    id_col = None
                    for col in id_cols:
                        if col in icustays.columns:
                            id_col = col
                            break
                    if id_col is not None:
                        # Filter to patients in the current frame
                        patient_ids_in_frame = frame[id_col].unique() if id_col in frame.columns else None
                        if patient_ids_in_frame is not None:
                            admission_times = icustays[icustays[id_col].isin(patient_ids_in_frame)][[id_col, 'intime']].drop_duplicates()
            except Exception:
                pass  # Fail silently - will use fallback floor behavior
        
        result = mimic_rate_mv(
            frame,
            val_col=val_col,
            unit_col=unit_col,
            stop_var=stop_var,
            id_cols=id_cols,
            admission_times=admission_times,  # 🔧 Pass admission times for proper floor behavior
        )
        # 🔧 FIX: 将 'rate' 列重命名为 concept_name（如果不同）
        if val_col != concept_name and val_col in result.columns:
            result = result.rename(columns={val_col: concept_name})
        return result
    
    # Handle mimic_dur_inmv callback (for infusion durations)
    if expr.strip() == "mimic_dur_inmv":
        from .callback_utils import mimic_dur_inmv
        # 🔧 FIX 2025-02-10: Only use the PRIMARY patient ID column, not all "id" columns
        primary_id_candidates = ['icustay_id', 'stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
        id_cols = None
        for cand in primary_id_candidates:
            if cand in frame.columns:
                id_cols = [cand]
                break
        # stop_var and grp_var are stored in params dict
        stop_var = source.params.get('stop_var', None) if source.params else None
        grp_var = source.params.get('grp_var', None) if source.params else None
        # Use unit_column from parent context or source.unit_var
        unit_col = unit_column or (source.unit_var if hasattr(source, 'unit_var') else None)
        val_col = concept_name
        
        # Load admission times for proper floor(end_h) - floor(start_h) calculation
        # R ricu uses: duration = floor(end_hours) - floor(start_hours)
        # where hours are relative to intime
        admission_times = None
        if data_source is not None:
            try:
                icustays_result = data_source.load_table('icustays')
                # ICUTable has .data attribute for the underlying DataFrame
                if hasattr(icustays_result, 'data'):
                    icustays_table = icustays_result.data
                else:
                    icustays_table = icustays_result
                if icustays_table is not None and not icustays_table.empty:
                    # Find the id column and intime column
                    id_col_name = id_cols[0] if id_cols else 'stay_id'
                    if id_col_name in icustays_table.columns:
                        # Keep the original id column name (e.g., 'stay_id') instead of renaming to 'id'
                        admission_times = icustays_table[[id_col_name, 'intime']].copy()
            except Exception:
                pass  # Fallback to floor(duration) if icustays not available
        
        return mimic_dur_inmv(
            frame,
            val_col=val_col,
            grp_var=grp_var,
            stop_var=stop_var,
            id_cols=id_cols,
            unit_col=unit_col,
            admission_times=admission_times,
        )
    
    # Handle mimic_dur_incv callback (for CareVue durations)
    if expr.strip() == "mimic_dur_incv":
        from .callback_utils import mimic_dur_incv
        # 🔧 FIX 2025-02-10: Only use the PRIMARY patient ID column, not all "id" columns
        # R ricu's calc_dur uses id_vars(x) which returns only the patient ID (e.g., icustay_id)
        primary_id_candidates = ['icustay_id', 'stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
        id_cols = None
        for cand in primary_id_candidates:
            if cand in frame.columns:
                id_cols = [cand]
                break
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
        # 🔧 FIX 2025-02-10: Only use the PRIMARY patient ID column, not all "id" columns
        primary_id_candidates = ['icustay_id', 'stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
        id_cols = None
        for cand in primary_id_candidates:
            if cand in frame.columns:
                id_cols = [cand]
                break
        # grp_var is stored in params dict
        grp_var = source.params.get('grp_var', None) if source.params else None
        unit_col = source.unit_var if hasattr(source, 'unit_var') else None
        val_col = concept_name
        
        # 🔧 FIX: Load admission_times for R ricu-compatible relative time flooring
        # R ricu converts datetime to relative difftime BEFORE callbacks (in load_difftime).
        # CareVue expand_intervals needs this to correctly floor to hour boundaries.
        admission_times = None
        if data_source is not None:
            try:
                icustays_result = data_source.load_table('icustays')
                if hasattr(icustays_result, 'data'):
                    icustays = icustays_result.data
                else:
                    icustays = icustays_result
                if icustays is not None and len(icustays) > 0:
                    id_col = id_cols[0] if id_cols else None
                    if id_col and id_col in icustays.columns:
                        patient_ids_in_frame = frame[id_col].unique() if id_col in frame.columns else None
                        if patient_ids_in_frame is not None:
                            admission_times = icustays[icustays[id_col].isin(patient_ids_in_frame)][[id_col, 'intime']].drop_duplicates()
            except Exception:
                pass
        
        return mimic_rate_cv(
            frame,
            val_col=val_col,
            grp_var=grp_var,
            unit_col=unit_col,
            id_cols=id_cols,
            admission_times=admission_times,
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
        # easyicu 需要在调用回调前加载 weight 概念
        # 🔧 FIX 2: Only try to get weight if frame is not empty
        if not frame.empty and 'weight' not in frame.columns and resolver is not None and data_source is not None:
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

    # Handle hirid_duration callback - calculate infusion durations
    if expr == "hirid_duration":
        from .callback_utils import hirid_duration
        
        index_var = source.index_var or 'givenat'
        val_var = source.value_var
        grp_var = source.params.get("grp_var") if source.params else None
        if not grp_var:
            grp_var = getattr(source, 'grp_var', None)
        if not grp_var and 'infusionid' in frame.columns:
            grp_var = 'infusionid'
        
        return hirid_duration(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            index_col=index_var,
            grp_var=grp_var,
        )

    # Handle hirid_vent callback - convert ventilation records to window table
    if expr == "hirid_vent":
        from .callback_utils import hirid_vent
        
        index_var = source.index_var or 'datetime'
        val_var = source.value_var
        
        return hirid_vent(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            index_col=index_var,
            dur_var='dur_var',
            padding_hours=4.0,
            max_gap_hours=12.0,
            expand_to_hourly=False,  # Return win_tbl format, not expanded ts_tbl
        )

    # Handle hirid_urine callback - convert cumulative urine to incremental
    if expr == "hirid_urine":
        from .callback_utils import hirid_urine
        
        val_var = source.value_var or 'value'
        unit_var = source.unit_var
        
        return hirid_urine(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            unit_col=unit_var,
        )

    # Handle hirid_rate_kg callback - HiRID dose rate per kg
    if expr == "hirid_rate_kg":
        from .callback_utils import hirid_rate_kg

        val_var = source.value_var or 'givendose'
        unit_var = source.unit_var or 'doseunit'
        grp_var = source.params.get("grp_var") if source.params else None
        if not grp_var:
            grp_var = getattr(source, 'grp_var', None)
        if not grp_var and 'infusionid' in frame.columns:
            grp_var = 'infusionid'
        index_var = source.index_var or 'givenat'
        
        # 🔧 FIX: Only try to get weight if frame is not empty
        # Avoids reading huge observations table (70M rows) when there's no data
        if not frame.empty and 'weight' not in frame.columns and resolver is not None and data_source is not None:
            try:
                id_col = None
                for cand in ['patientid', 'stay_id', 'admissionid']:
                    if cand in frame.columns:
                        id_col = cand
                        break
                if id_col:
                    unique_ids = frame[id_col].unique().tolist()
                    weight_per_patient = None
                    
                    # 🔧 FIX 2026-03-12: For HiRID, load raw weight from parquet
                    # and compute direct per-patient median (bypassing DuckDB hourly aggregation).
                    # R ricu: load_concepts("weight", aggregate=NULL) → median(all_raw_values)
                    # Previous easyicu: DuckDB GROUP BY (patient,hour) MEDIAN → groupby(patient).median() = "median of medians" ≠ direct median
                    db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                    if db_name == 'hirid':
                        try:
                            import duckdb
                            bucket_dir = data_source.base_path / 'observations_bucket'
                            if bucket_dir.exists():
                                conn = duckdb.connect()
                                conn.execute("SET memory_limit = '2GB'")
                                # weight variableid = 10000400
                                pid_list = ','.join(str(int(p)) for p in unique_ids)
                                sql = f"""
                                    SELECT patientid, MEDIAN(value) as weight
                                    FROM read_parquet('{_duckdb_path(bucket_dir)}/**/*.parquet', hive_partitioning=true)
                                    WHERE variableid = 10000400 AND patientid IN ({pid_list})
                                      AND value IS NOT NULL AND value >= 1 AND value <= 500
                                    GROUP BY patientid
                                """
                                weight_per_patient = conn.execute(sql).fetchdf()
                                conn.close()
                        except Exception:
                            weight_per_patient = None
                    
                    if weight_per_patient is None or weight_per_patient.empty:
                        # Fallback: use standard loading path
                        weight_table = resolver._load_single_concept(
                            'weight',
                            data_source,
                            aggregator=False,
                            patient_ids={id_col: unique_ids},
                            verbose=False,
                            _bypass_callback=True,
                        )
                        if weight_table is not None and not weight_table.data.empty:
                            weight_df = weight_table.data
                            if 'weight' in weight_df.columns:
                                weight_df['weight'] = pd.to_numeric(weight_df['weight'], errors='coerce')
                                weight_per_patient = weight_df.groupby(id_col)['weight'].median().reset_index()
                    
                    if weight_per_patient is not None and not weight_per_patient.empty:
                        frame = frame.merge(weight_per_patient, on=id_col, how='left')
            except Exception as e:
                if DEBUG_MODE:
                    print(f"   ⚠️  获取体重失败: {e}")
                pass

        # 🔧 FIX: Calculate interval_minutes from concept's interval
        # R ricu uses frac = 1 / interval(x), where interval(x) is the concept's interval.
        # For dobu_rate (no interval): default 60min (1 hour)
        # For dobu60 (interval="00:01:00"): 1min → rate is 60x higher
        interval_minutes = 60.0  # default
        if interval is not None:
            if isinstance(interval, str):
                # Parse string like "00:01:00" (1 minute) or "01:00:00" (1 hour)
                try:
                    td = pd.to_timedelta(interval)
                    interval_minutes = td.total_seconds() / 60.0
                except Exception:
                    pass
            elif isinstance(interval, pd.Timedelta):
                interval_minutes = interval.total_seconds() / 60.0
        
        return hirid_rate_kg(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            unit_col=unit_var,
            grp_var=grp_var,
            index_col=index_var,
            interval_minutes=interval_minutes,
            value_min=_get_concept_bounds(concept_name, 'min'),
            value_max=_get_concept_bounds(concept_name, 'max'),
        )

    # Handle hirid_rate callback - HiRID dose rate (no weight normalization)
    if expr == "hirid_rate":
        from .callback_utils import hirid_rate

        val_var = source.value_var or 'givendose'
        unit_var = source.unit_var or 'doseunit'
        grp_var = source.params.get("grp_var") if source.params else None
        if not grp_var:
            grp_var = getattr(source, 'grp_var', None)
        if not grp_var and 'infusionid' in frame.columns:
            grp_var = 'infusionid'
        index_var = source.index_var or 'givenat'

        return hirid_rate(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            unit_col=unit_var,
            grp_var=grp_var,
            index_col=index_var,
        )

    # Handle aumc_rate callback - combine unit_var and rate_var into unit/rate format
    # R: x <- x[, c(unit_var) := do_call(.SD, paste, sep = "/"), .SDcols = c(unit_var, rate_var)]
    # 🔧 FIX 2025-02-03: Also normalize rate units (min -> hr conversion)
    if expr == "aumc_rate":
        rate_var = getattr(source, 'rate_var', None)
        if not rate_var and source.params:
            rate_var = source.params.get("rate_var")
        unit_var = source.unit_var or unit_column
        val_var = source.value_var or concept_name
        
        if rate_var and rate_var in frame.columns:
            frame = frame.copy()
            # Normalize rate units: 'min' means per-minute, need to multiply by 60 to get per-hour
            # R ricu does this in aumc_rate_kg with hr_to_min, but aumc_rate needs it too for dex
            rate_lower = frame[rate_var].astype(str).str.lower().str.strip()
            
            # If rate_var is 'min' (per minute), multiply value by 60 to get per hour
            mask_min = rate_lower.isin({'min', 'minute', 'minutes', 'm'})
            if mask_min.any() and val_var in frame.columns:
                frame.loc[mask_min, val_var] = frame.loc[mask_min, val_var] * 60.0
                frame.loc[mask_min, rate_var] = 'uur'  # Now it's per hour
            
            # Combine unit and rate into "unit/rate" format
            if unit_var and unit_var in frame.columns:
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

    # Handle MIMIC-III mimic_age callback
    # R ricu logic:
    #   1. change_id mechanism converts dob column to (intime - dob) time difference
    #   2. mimic_age: x <- as.double(x, units = "days") / -365; ifelse(x > 90, 90, x)
    # EasyICU: need to manually join patients with icustays to get intime and calculate age
    if expr == "transform_fun(mimic_age)" or expr == "mimic_age":
        frame = frame.copy()
        val_col = source.value_var if source else 'dob'
        
        # Check if we have dob (birth date) that needs to be converted
        # Note: At this point, dob may have been renamed to concept_name (e.g., 'age')
        # So we check for either 'dob' column or concept_name column that was originally 'dob'
        has_dob = 'dob' in frame.columns
        dob_renamed_to_concept = (val_col == 'dob' and concept_name in frame.columns and 'dob' not in frame.columns)
        
        if has_dob or dob_renamed_to_concept:
            # Determine actual column name containing DOB data
            actual_dob_col = 'dob' if has_dob else concept_name
            # Need to load icustays to get intime for each patient
            if data_source is not None:
                try:
                    # For MIMIC-III: 
                    # frame's 'stay_id' column contains 'icustay_id' values (already joined/replaced)
                    # We need to merge with icustays on icustay_id to get intime
                    
                    db_name = data_source.config.name if hasattr(data_source, 'config') else 'mimic'
                    
                    # Load icustays with intime
                    icustays = data_source.load_table(
                        'icustays',
                        columns=['icustay_id', 'intime'],
                        verbose=False
                    )
                    if hasattr(icustays, 'data'):
                        icustays = icustays.data
                    
                    # Determine the ID column in frame
                    # In MIMIC-III, frame's 'stay_id' contains icustay_id values
                    if 'stay_id' in frame.columns:
                        # Rename for consistent merge
                        frame = frame.rename(columns={'stay_id': 'icustay_id'})
                        merge_col = 'icustay_id'
                    elif 'icustay_id' in frame.columns:
                        merge_col = 'icustay_id'
                    else:
                        merge_col = None
                    
                    if merge_col is not None and merge_col in icustays.columns:
                        # Merge to get intime
                        frame = frame.merge(icustays[['icustay_id', 'intime']], on=merge_col, how='left')
                        
                        if len(frame) == 0:
                            print("⚠️ [mimic_age] MERGE PRODUCED 0 ROWS!")
                            return frame
                        
                        # Calculate age using actual_dob_col
                        dob = pd.to_datetime(frame[actual_dob_col], errors='coerce')
                        intime = pd.to_datetime(frame['intime'], errors='coerce')
                        
                        # R ricu: as.double(x, units = "days") / -365
                        # Use total_seconds() for more precise day calculation (matching R)
                        time_diff = intime - dob
                        age_days = time_diff.dt.total_seconds() / (24 * 60 * 60)
                        age_years = age_days / 365.0
                        # Cap at 90 (R ricu: ifelse(x > 90, 90, x))
                        age_years = np.where(age_years > 90, 90, age_years)
                        frame[concept_name] = age_years
                        
                        # Use icustay_id as the final stay_id
                        if 'icustay_id' in frame.columns:
                            frame = frame.rename(columns={'icustay_id': 'stay_id'})
                        
                        # Clean up temporary columns
                        for col in ['intime', actual_dob_col, 'subject_id']:
                            if col in frame.columns and col != concept_name and col != 'stay_id':
                                frame = frame.drop(columns=[col])
                    
                except Exception as e:
                    # If loading fails, try simpler approach
                    import traceback
                    print(f"⚠️ mimic_age callback failed: {e}")
                    traceback.print_exc()
                    if concept_name in frame.columns:
                        frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                        frame.loc[frame[concept_name] > 90, concept_name] = 90
            else:
                # No data_source - just cap at 90 if age already exists
                if concept_name in frame.columns:
                    frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                    frame.loc[frame[concept_name] > 90, concept_name] = 90
        elif concept_name in frame.columns:
            # Age already calculated - just cap at 90
            frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            frame.loc[frame[concept_name] > 90, concept_name] = 90
        
        return frame

    # Handle MIMIC-III mimic_abx_presc callback
    # R ricu logic: x[, c(idx, val_var) := list(get(idx) + mins(720L), TRUE)]
    if expr == "mimic_abx_presc":
        frame = frame.copy()
        index_col = source.index_var
        if not index_col:
            for candidate in ["charttime", "starttime", "startdate"]:
                if candidate in frame.columns:
                    index_col = candidate
                    break
        # Shift time forward by 720 minutes (12 hours)
        if index_col and index_col in frame.columns:
            frame[index_col] = pd.to_numeric(frame[index_col], errors='coerce') + 720
        # Set value to TRUE
        frame[concept_name] = True
        return frame

    # Handle MIMIC-III mimic_kg_rate callback
    # R ricu logic: add_weight + divide by weight + update unit
    if expr == "mimic_kg_rate":
        val_var = source.value_var or concept_name
        unit_var = source.unit_var or unit_column
        
        # Try to add weight and divide
        if 'weight' not in frame.columns and resolver is not None and data_source is not None:
            try:
                id_cols = [c for c in frame.columns if c.lower().endswith('id') and c != 'itemid']
                if id_cols:
                    unique_ids = frame[id_cols[0]].unique().tolist()
                    weight_table = resolver._load_single_concept(
                        'weight',
                        data_source,
                        aggregator=False,
                        patient_ids={id_cols[0]: unique_ids},
                        verbose=False,
                        _bypass_callback=True,
                    )
                    if weight_table is not None and not weight_table.data.empty:
                        weight_df = weight_table.data
                        if 'weight' in weight_df.columns:
                            weight_df['weight'] = pd.to_numeric(weight_df['weight'], errors='coerce')
                            merge_cols = [c for c in id_cols if c in weight_df.columns]
                            if merge_cols:
                                frame = frame.merge(
                                    weight_df[merge_cols + ['weight']].drop_duplicates(),
                                    on=merge_cols,
                                    how='left'
                                )
            except Exception:
                pass
        
        # Divide rate by weight
        if 'weight' in frame.columns and val_var in frame.columns:
            frame = frame.copy()
            frame[val_var] = pd.to_numeric(frame[val_var], errors='coerce')
            frame['weight'] = pd.to_numeric(frame['weight'], errors='coerce')
            mask = frame['weight'] > 0
            frame.loc[mask, val_var] = frame.loc[mask, val_var] / frame.loc[mask, 'weight']
            # Update unit
            if unit_var and unit_var in frame.columns:
                frame[unit_var] = frame[unit_var].str.replace('mcgmin', 'mcg/kg/min', regex=False)
            frame = frame.drop(columns=['weight'], errors='ignore')
        return frame

    # Handle SICdb sic_dur callback
    # R ricu logic: calc_dur(x, val_var, index_var(x), stop_var, grp_var)
    if expr == "sic_dur":
        val_var = source.value_var or concept_name
        index_var = source.index_var
        stop_var = source.params.get("stop_var") if source.params else None
        grp_var = source.params.get("grp_var") if source.params else None
        
        if not stop_var:
            for candidate in ["OffsetDrugEnd", "stop", "endtime"]:
                if candidate in frame.columns:
                    stop_var = candidate
                    break
        
        if not index_var:
            for candidate in ["Offset", "OffsetDrugStart", "start", "charttime"]:
                if candidate in frame.columns:
                    index_var = candidate
                    break
        
        if stop_var and stop_var in frame.columns and index_var and index_var in frame.columns:
            frame = frame.copy()
            # Use standard patient-level ID columns only (not row-level id, bucket_id, PatientID etc.)
            _PATIENT_ID_COLS = ['CaseID', 'stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']
            id_cols = [c for c in _PATIENT_ID_COLS if c in frame.columns]
            
            # Group by ID (and optionally grp_var)
            group_cols = list(id_cols)
            if grp_var and grp_var in frame.columns:
                group_cols = id_cols + [grp_var]
            
            if group_cols:
                # Calculate duration = max(stop) - min(start) per group
                agg_df = frame.groupby(group_cols).agg({
                    index_var: 'min',
                    stop_var: 'max'
                }).reset_index()
                
                # SICdb medication.Offset is in seconds → convert to hours
                # Duration = floor(max_stop/3600) - floor(min_start/3600)
                # This matches R ricu's change_interval(hours(1)) behavior
                min_start = pd.to_numeric(agg_df[index_var], errors='coerce')
                max_stop = pd.to_numeric(agg_df[stop_var], errors='coerce')
                start_hours = (min_start // 3600).astype(int)
                stop_hours = (max_stop // 3600).astype(int)
                agg_df[val_var] = stop_hours - start_hours
                agg_df[index_var] = start_hours
                
                # If grp_var was used, set index to min per patient and pick max duration
                if grp_var and grp_var in frame.columns and id_cols:
                    min_idx = agg_df.groupby(id_cols)[index_var].transform('min')
                    agg_df[index_var] = min_idx
                    agg_df = agg_df.sort_values(val_var, ascending=False).drop_duplicates(
                        subset=id_cols + [index_var], keep='first')
                
                # Keep only required columns
                result_cols = id_cols + [index_var, val_var]
                frame = agg_df[[c for c in result_cols if c in agg_df.columns]]
        
        return frame

    # Handle SICdb sic_rate_kg callback
    # R ricu logic: add_weight + multiply by 10^6 / weight + expand
    if expr == "sic_rate_kg":
        val_var = source.value_var or concept_name
        # Fix: source.value_var may have been renamed to concept_name during loading
        if val_var not in frame.columns and concept_name in frame.columns:
            val_var = concept_name
        stop_var = source.params.get("stop_var") if source.params else None
        
        if not stop_var:
            for candidate in ["OffsetDrugEnd", "stop", "endtime"]:
                if candidate in frame.columns:
                    stop_var = candidate
                    break
        
        # Try to add weight
        if 'weight' not in frame.columns and resolver is not None and data_source is not None:
            try:
                _PATIENT_ID_COLS_W = ['CaseID', 'stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']
                id_cols = [c for c in _PATIENT_ID_COLS_W if c in frame.columns]
                if id_cols:
                    unique_ids = frame[id_cols[0]].unique().tolist()
                    weight_table = resolver._load_single_concept(
                        'weight',
                        data_source,
                        aggregator=False,
                        patient_ids={id_cols[0]: unique_ids},
                        verbose=False,
                        _bypass_callback=True,
                    )
                    if weight_table is not None and not weight_table.data.empty:
                        weight_df = weight_table.data
                        if 'weight' in weight_df.columns:
                            weight_df['weight'] = pd.to_numeric(weight_df['weight'], errors='coerce')
                            # Get first weight per patient
                            weight_id_col = id_cols[0] if id_cols[0] in weight_df.columns else (
                                'CaseID' if 'CaseID' in weight_df.columns else None
                            )
                            if weight_id_col:
                                weight_agg = weight_df.groupby(weight_id_col)['weight'].first().reset_index()
                                frame = frame.merge(weight_agg, on=weight_id_col, how='left')
            except Exception:
                pass
        
        # Convert rate: multiply by 10^6 / weight (mg -> mcg/kg)
        if 'weight' in frame.columns and val_var in frame.columns:
            frame = frame.copy()
            frame[val_var] = pd.to_numeric(frame[val_var], errors='coerce')
            frame['weight'] = pd.to_numeric(frame['weight'], errors='coerce')
            mask = frame['weight'] > 0
            frame.loc[mask, val_var] = frame.loc[mask, val_var] * 1e6 / frame.loc[mask, 'weight']
            frame = frame.drop(columns=['weight'], errors='ignore')
        
        # Expand time range: convert each (start, stop) interval into hourly rows
        index_var = source.index_var
        if not index_var:
            for candidate in ["Offset", "OffsetDrugStart", "start", "charttime"]:
                if candidate in frame.columns:
                    index_var = candidate
                    break
        
        if stop_var and stop_var in frame.columns and index_var and index_var in frame.columns:
            # R ricu expand(): generate hourly observations between start and stop
            # Use standard patient-level ID columns only
            _PATIENT_ID_COLS_E = ['CaseID', 'stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']
            id_cols = [c for c in _PATIENT_ID_COLS_E if c in frame.columns]
            keep_cols = id_cols + [val_var] if val_var in frame.columns else id_cols
            
            expanded_rows = []
            for _, row in frame.iterrows():
                start_val = pd.to_numeric(row.get(index_var), errors='coerce')
                stop_val = pd.to_numeric(row.get(stop_var), errors='coerce')
                if pd.isna(start_val) or pd.isna(stop_val) or stop_val <= start_val:
                    continue
                # SICdb medication.Offset is in seconds → convert to hourly steps
                # R ricu floor(): floor to nearest hour
                start_hour = int(start_val // 3600)
                stop_hour = int(stop_val // 3600)
                for t in range(start_hour, stop_hour + 1):
                    new_row = {index_var: t}  # Output Offset in hours
                    for c in keep_cols:
                        if c in row.index:
                            new_row[c] = row[c]
                    expanded_rows.append(new_row)
            
            if expanded_rows:
                frame = pd.DataFrame(expanded_rows)
                # 🔧 FIX 2026-03-11: Do NOT hardcode median aggregation here!
                # Previously this did groupby(...).agg({val_var: 'median'}) which pre-aggregated
                # the expanded rates. This prevents vaso60 callback from getting raw per-interval
                # rates needed for MAX aggregation (dobu60/norepi60/epi60/dopa60 etc).
                # change_interval() in _load_single_concept handles aggregation correctly:
                #   - standalone dobu_rate: change_interval(median) → correct median
                #   - vaso60 sub-concept: change_interval(False) → preserves all rows → vaso60 takes max
                if val_var in frame.columns:
                    frame[val_var] = pd.to_numeric(frame[val_var], errors='coerce')
        
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
        
        # 🔧 FIX 2025-01: Get admission times for R ricu-compatible floor behavior
        # R ricu converts datetime to relative time BEFORE callbacks (in load_mihi).
        # This affects floor() behavior in expand().
        admission_times = None
        if data_source is not None:
            try:
                # 🔧 FIX 2026-02-09: 正确检测 ID 列
                # MIMIC-III 使用 icustay_id，需要明确指定
                db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                if db_name in ['mimic', 'mimic_demo']:
                    id_cols_for_icustays = ['icustay_id']
                else:
                    id_cols_for_icustays = ['stay_id', 'icustay_id', 'hadm_id', 'admissionid', 'patientid', 'patientunitstayid']
                
                # Load icustays to get admission times
                icustays_result = data_source.load_table('icustays')
                # Handle ICUTable or DataFrame result
                if hasattr(icustays_result, 'data'):
                    icustays = icustays_result.data
                else:
                    icustays = icustays_result
                    
                if icustays is not None and len(icustays) > 0:
                    # Find ID column
                    id_col = None
                    for col in id_cols_for_icustays:
                        if col in icustays.columns:
                            id_col = col
                            break
                    if id_col is not None:
                        # Filter to patients in the current frame
                        patient_ids_in_frame = frame[id_col].unique() if id_col in frame.columns else None
                        if patient_ids_in_frame is not None:
                            admission_times = icustays[icustays[id_col].isin(patient_ids_in_frame)][[id_col, 'intime']].drop_duplicates()
            except Exception:
                pass  # Fail silently - will use fallback floor behavior
        
        return distribute_amount(
            frame,
            val_col=concept_name,
            unit_col=unit_col,
            end_col=end_col,
            index_col=index_col,
            admission_times=admission_times,  # 🔧 Pass admission times for proper floor behavior
        )

    if expr.strip() == "mimv_rate":
        from .callback_utils import mimv_rate
        duration_col = None
        start_col = source.index_var
        if not start_col and "starttime" in frame.columns:
            start_col = "starttime"

        end_col = None
        if "endtime" in frame.columns:
            end_col = "endtime"
        elif source.dur_var and source.dur_var in frame.columns:
            end_col = source.dur_var

        # 首先检查是否已经有计算好的 duration 列
        possible_dur_cols = [concept_name + '_dur', 'duration', '__duration__', 'dur_var']
        for col in possible_dur_cols:
            if col in frame.columns:
                duration_col = col
                break

        if duration_col is None and end_col and end_col in frame.columns:
            duration_col = end_col
        
        if not duration_col or duration_col not in frame.columns:
            return frame
        # 🔧 FIX: amount_col 应优先使用 'amount' 列（inputevents 表的默认列）
        # R ricu mimv_rate 使用 amount 列来计算 rate = amount / duration
        # concept_name (如 'dex') 在回调执行时还不存在
        amount_col = None
        if source.params:
            alt_amount = source.params.get("amount_var")
            if alt_amount and alt_amount in frame.columns:
                amount_col = alt_amount
        if not amount_col:
            # 优先使用 'amount' 列（inputevents 表的标准列名）
            if 'amount' in frame.columns:
                amount_col = 'amount'
            elif concept_name in frame.columns:
                amount_col = concept_name
        if not amount_col or amount_col not in frame.columns:
            return frame
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
        
        # 🔧 FIX: mimv_rate 应使用表的默认 rate 列，而不是 concept_name
        # R ricu 中 mimv_rate 使用 val_var='rate' (来自 inputevents 表配置)
        # mimv_rate 计算 rate = amount / duration，结果写入 rate 列
        rate_col = 'rate' if 'rate' in frame.columns else concept_name
        
        return mimv_rate(
            frame,
            val_col=rate_col,
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
        
        # 优先使用已计算好的duration列 (dur_is_end逻辑产生的 {concept_name}_dur)
        # 这个列包含真正的duration = stopoffset - startoffset
        dur_var = None
        duration_col = concept_name + '_dur'
        if duration_col in frame.columns:
            dur_var = duration_col
        elif "duration" in frame.columns:
            dur_var = "duration"
        else:
            # 回退到原始配置
            if source.params:
                dur_var = source.params.get("dur_var") or source.params.get("stop_var")
            if (not dur_var or dur_var not in frame.columns) and "drugstopoffset" in frame.columns:
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
        DEBUG_CALLBACK = False  # Toggle for debugging (set to True for trace)
        if DEBUG_CALLBACK:
            print(f"  [CALLBACK DEBUG] {concept_name} blood_cell_ratio 开始")
            print(f"    frame.shape = {frame.shape}, columns = {list(frame.columns)}")
            if concept_name in frame.columns:
                print(f"    输入值: {frame[concept_name].values}")
        
        if resolver is None:
            if DEBUG_CALLBACK:
                print("    [SKIP] resolver is None")
            # Cannot convert without resolver to load WBC, return as-is
            if concept_name in frame.columns:
                frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            return frame
        
        # Determine ID column based on database
        # AUMC uses 'admissionid', MIMIC uses 'stay_id', eICU uses 'patientunitstayid'
        # HiRID uses 'patientid', SICdb uses 'CaseID'
        id_col = None
        for possible_id in ['admissionid', 'stay_id', 'patientunitstayid', 'subject_id', 'icustay_id', 'patientid', 'CaseID']:
            if possible_id in frame.columns:
                id_col = possible_id
                break
        
        if id_col is None:
            if DEBUG_CALLBACK:
                print("    [SKIP] id_col is None")
            if concept_name in frame.columns:
                frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            return frame
        
        if DEBUG_CALLBACK:
            print(f"    id_col = {id_col}")
        
        frame_patient_ids = frame[id_col].unique().tolist()
        if len(frame_patient_ids) == 0:
            if DEBUG_CALLBACK:
                print("    [SKIP] no patients")
            if concept_name in frame.columns:
                frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            return frame
        
        if DEBUG_CALLBACK:
            print(f"    patients = {frame_patient_ids}")
        
        try:
            # Load WBC concept for the same patients
            # IMPORTANT: Use merge=False to get Dict[str, ICUTable] instead of merged DataFrame
            # IMPORTANT: Must pass data_source for resolver.load_concepts to work
            # Cache WBC across blood_cell_ratio concepts for performance
            if data_source is None:
                if DEBUG_CALLBACK:
                    print("    [SKIP] data_source is None")
                if concept_name in frame.columns:
                    frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                return frame
            
            if DEBUG_CALLBACK:
                print("    加载 WBC (使用缓存)...")
            
            # Use full patient_ids for cache efficiency when available
            _wbc_pids = patient_ids if patient_ids else frame_patient_ids
            wbc_result = resolver.load_concepts(
                ['wbc'],
                data_source,
                patient_ids=_wbc_pids,
                r_compatible=False,
                merge=False,
            )
            
            if 'wbc' not in wbc_result or wbc_result['wbc'].data.empty:
                if DEBUG_CALLBACK:
                    print("    [SKIP] WBC 为空或不存在")
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
            
            # 🔧 FIX 2026-03-09: Handle time column name mismatch between raw source
            # data and WBC loaded via load_concepts (DuckDB aggregation).
            # e.g. AUMC raw source has 'measuredat' (minutes) but WBC has
            # 'measuredat_minutes' (hourly-binned minutes) from DuckDB aggregation.
            # Rename WBC's time column to match frame's time column for merge_asof.
            wbc_index_col = wbc_result['wbc'].index_column
            if (index_col and wbc_index_col and
                    index_col != wbc_index_col and
                    index_col not in wbc_df.columns and
                    wbc_index_col in wbc_df.columns):
                if DEBUG_CALLBACK:
                    print(f"    [TIME COL FIX] Renaming WBC time col: "
                          f"{wbc_index_col} -> {index_col}")
                wbc_df = wbc_df.rename(columns={wbc_index_col: index_col})
            
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
                frame_time_max = frame[index_col].abs().max()
                wbc_time_max = wbc_df[index_col].abs().max() if not wbc_df.empty else 0
                
                # Create copies to avoid modifying original
                frame_work = frame.copy()
                wbc_work = wbc_df.copy()
                
                # CRITICAL: Filter WBC to frame patients BEFORE time unit detection.
                # Otherwise, long-stay patients not in the frame can push wbc_time_max
                # past the 1000-hour threshold, breaking the minutes-vs-hours heuristic.
                _unique_pids = frame_work[id_col].unique()
                wbc_work = wbc_work[wbc_work[id_col].isin(set(_unique_pids))].copy()
                
                # Recalculate time maxes after filtering
                frame_time_max = frame_work[index_col].abs().max()
                wbc_time_max = wbc_work[index_col].abs().max() if not wbc_work.empty else 0
                
                # Improved time unit detection:
                # 1. Large absolute threshold (>1000) clearly indicates minutes
                # 2. Relative comparison: if frame_time >> wbc_time (e.g., 5x+), convert
                # 3. For AUMC with measuredat, frame comes from raw table (minutes) while
                #    wbc comes from load_concepts (hours)
                need_frame_to_hours = False
                need_wbc_to_hours = False
                
                if frame_time_max > 1000 and wbc_time_max < 1000 and wbc_time_max > 0:
                    # Clear case: frame is in minutes (>1000), wbc is in hours
                    need_frame_to_hours = True
                elif frame_time_max < 1000 and wbc_time_max > 1000:
                    # Opposite: wbc is in minutes, frame is in hours
                    need_wbc_to_hours = True
                elif frame_time_max > 0 and wbc_time_max > 0:
                    # Both are < 1000, but may still have different units
                    # If ratio is significantly different (5x+), assume different units
                    ratio = frame_time_max / wbc_time_max if wbc_time_max > 0 else 0
                    if ratio > 5:
                        # frame is much larger, likely in minutes vs hours
                        need_frame_to_hours = True
                        if DEBUG_CALLBACK:
                            print("    [TIME FIX] 基于比率检测时间单位不匹配:")
                            print(f"      ratio = {ratio:.2f}")
                    elif ratio < 0.2 and ratio > 0:
                        # wbc is much larger
                        need_wbc_to_hours = True
                
                if need_frame_to_hours:
                    if DEBUG_CALLBACK:
                        print("    [TIME FIX] 检测到时间单位不匹配:")
                        print(f"      frame max time: {frame_time_max} (分钟)")
                        print(f"      wbc max time: {wbc_time_max} (小时)")
                        print("      -> 将 frame 时间从分钟转换为小时")
                    frame_work[index_col] = frame_work[index_col] / 60.0
                elif need_wbc_to_hours:
                    if DEBUG_CALLBACK:
                        print("    [TIME FIX] 检测到时间单位不匹配（反向）:")
                        print(f"      frame max time: {frame_time_max}")
                        print(f"      wbc max time: {wbc_time_max}")
                        print("      -> 将 wbc 时间从分钟转换为小时")
                    wbc_work[index_col] = wbc_work[index_col] / 60.0
                
                # Ensure matching dtypes for index column
                wbc_work[index_col] = wbc_work[index_col].astype(frame_work[index_col].dtype)
                
                # CRITICAL: merge_asof requires the 'on' column to be sorted globally.
                # With multiple patients, their time ranges may overlap.
                # Solution: Add per-patient offset to make times globally monotonic,
                # then do a single merge_asof call instead of per-patient loops.
                _max_time = max(
                    frame_work[index_col].abs().max() if len(frame_work) > 0 else 0,
                    wbc_work[index_col].abs().max() if len(wbc_work) > 0 else 0,
                ) + 1000  # generous padding

                # Build offset map: each patient gets a non-overlapping time range
                _pid_offset = {pid: i * _max_time * 2 for i, pid in enumerate(_unique_pids)}

                # Add offset to make global time monotonic
                frame_work['_gtime'] = frame_work[id_col].map(_pid_offset) + frame_work[index_col]
                wbc_work['_gtime'] = wbc_work[id_col].map(_pid_offset) + wbc_work[index_col]

                # Sort by global time for merge_asof
                frame_work = frame_work.sort_values('_gtime')
                wbc_work = wbc_work.sort_values('_gtime')

                try:
                    frame_merged = pd.merge_asof(
                        frame_work,
                        wbc_work[[id_col, '_gtime', 'wbc']],
                        on='_gtime',
                        by=id_col,
                        direction='nearest',
                    )
                except Exception:
                    # Fallback: per-patient merge_asof
                    merged_parts = []
                    for patient_id in _unique_pids:
                        fp = frame_work[frame_work[id_col] == patient_id].sort_values(index_col)
                        wp = wbc_work[wbc_work[id_col] == patient_id].sort_values(index_col)
                        if wp.empty:
                            merged_parts.append(fp)
                        else:
                            try:
                                mp = pd.merge_asof(fp, wp[[id_col, index_col, 'wbc']],
                                                   on=index_col, by=id_col, direction='nearest')
                                merged_parts.append(mp)
                            except Exception:
                                merged_parts.append(fp)
                    frame_merged = pd.concat(merged_parts, ignore_index=True) if merged_parts else frame_work.copy()

                # Clean up temp column
                frame_merged = frame_merged.drop(columns=['_gtime'], errors='ignore')
                
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
                        print("    [WARNING] 'wbc' not in frame_merged.columns!")
                
                # CRITICAL: Convert time back to original format (minutes) for AUMC
                # The subsequent processing will apply the minutes->hours conversion again
                if need_frame_to_hours:
                    # We converted frame from minutes to hours, now convert back
                    frame_merged[index_col] = frame_merged[index_col] * 60.0
                    if DEBUG_CALLBACK:
                        print("    [TIME RESTORE] 将时间从小时转换回分钟")
                
                if DEBUG_CALLBACK:
                    print(f"    返回 frame_merged, shape={frame_merged.shape}")
                return frame_merged
            else:
                if DEBUG_CALLBACK:
                    print("    [FALLBACK] index_col 不在两个 frame 中, 使用平均 WBC")
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
    in_backtick = False
    current: List[str] = []

    for char in argument_str:
        if char == "`":
            in_backtick = not in_backtick
        elif char == "(" and not in_backtick:
            level += 1
        elif char == ")" and not in_backtick:
            level = max(level - 1, 0)
        elif char == "," and level == 0 and not in_backtick:
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
    # 🔧 FIX: 只对包含 R 风格转义序列（如 \n, \t）的字符串进行 unicode_escape 解码
    # 直接的 UTF-8 字符（如荷兰语 ï）不应该被转换
    # unicode_escape 会错误地将 UTF-8 字节解释为转义序列
    if '\\' in text:
        try:
            return text.encode("utf8").decode("unicode_escape")
        except UnicodeDecodeError:
            return text
    return text

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
