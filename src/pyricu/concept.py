"""Concept dictionary utilities inspired by ricu."""

from __future__ import annotations

import json
import re
import operator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Union

import numpy as np
import pandas as pd

from .config import DataSourceConfig
from .datasource import FilterOp, FilterSpec, ICUDataSource
from .table import ICUTable, WinTbl
from .concept_callbacks import ConceptCallbackContext, execute_concept_callback

# 全局调试开关 - 设置为 False 可以减少输出
DEBUG_MODE = False


@dataclass
class ConceptSource:
    """Describe how to load a concept for a specific data source."""

    table: Optional[str] = None
    sub_var: Optional[str] = None
    ids: Optional[List[object]] = None
    value_var: Optional[str] = None
    unit_var: Optional[str] = None
    index_var: Optional[str] = None
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

        regex = payload.pop("regex", None)
        class_name = payload.pop("class", None)
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
            class_name=payload.get("class"),
            callback=payload.get("callback"),
            sub_concepts=sub_concepts,
            levels=payload.get("levels"),
            keep_components=payload.get("keep_components"),
            omop_id=_maybe_int(payload.get("omopid")),
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

    def __init__(self, dictionary: ConceptDictionary) -> None:
        self.dictionary = dictionary
        # Cache for icustays table to avoid repeated loading
        self._icustays_cache: Optional[pd.DataFrame] = None
        # Cache for ID mappings (stay_id <-> subject_id)
        self._id_mapping_cache: Optional[pd.DataFrame] = None
        # Cache for loaded tables to avoid repeated loading
        # Key: (table_name, frozenset(patient_ids), frozenset(filters))
        self._table_cache: Dict[tuple, pd.DataFrame] = {}

    def available_concepts(self) -> List[str]:
        return sorted(self.dictionary.keys())
    
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
            >>> patient_ids = {'stay_id': [99000004]}
            >>> # 表需要 subject_id
            >>> expanded = _expand_patient_ids(patient_ids, 'subject_id', ds)
            >>> # 结果: {'stay_id': [99000004], 'subject_id': [99000005]}
        """
        if not patient_ids:
            return patient_ids
        
        # 转换为字典格式
        if not isinstance(patient_ids, dict):
            # 如果是列表，假设是 stay_id（最常用）
            patient_ids = {'stay_id': list(patient_ids)}
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
        **kwargs,  # Additional parameters for callbacks (e.g., win_length, worst_val_fun)
    ):
        names = [name for name in concept_names]
        tables: Dict[str, ICUTable] = {}
        aggregators = self._normalise_aggregators(aggregate, names)

        if merge and len(names) > 1 and any(
            aggregators[name] is False for name in names
        ):
            raise ValueError(
                "Aggregation must be enabled for all concepts when merge=True."
            )

        # Set default interval to 1 hour if not specified
        if interval is None:
            interval = pd.Timedelta(hours=1)
        
        for name in names:
            if name not in self.dictionary:
                raise KeyError(f"Concept '{name}' not present in dictionary")
            definition = self.dictionary[name]

            agg_value = aggregators[name]
            if agg_value in (None, "auto") and definition.aggregate is not None:
                agg_value = definition.aggregate

            concept_table = self._load_single_concept(
                name,
                data_source,
                aggregator=agg_value,
                patient_ids=patient_ids,
                verbose=verbose,  # 传递verbose参数
                interval=interval,  # Pass interval to _load_single_concept
                align_to_admission=align_to_admission,  # Pass align flag
                **kwargs,  # Pass kwargs to _load_single_concept
            )
            tables[name] = concept_table

        if not merge:
            return tables

        return self._merge_tables(tables)

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
        definition = self.dictionary[concept_name]
        if definition.sub_concepts:
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
        if definition.callback:
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
            table_cfg = data_source.config.get_table(source.table)
            defaults = table_cfg.defaults
            filters = [
                FilterSpec(
                    column=source.sub_var,
                    op=FilterOp.IN,
                    value=source.ids,
                )
            ] if source.ids is not None else []
            
            # 🔧 修复：添加患者过滤器
            # 即使 defaults.id_var 为 None，仍尝试添加患者过滤器
            # 对于 MIMIC-IV hosp 表（如 microbiologyevents），使用 subject_id
            effective_id_var = defaults.id_var
            if patient_ids:
                if not effective_id_var:
                    # 如果没有配置 id_var，尝试检测常见的ID列
                    # 先检查表是否有 subject_id（MIMIC-IV hosp 表）
                    # 注意：这里我们还不知道表的列，所以先假设是 subject_id
                    # 实际过滤会在 load_table 中根据实际列名进行
                    if source.table in ['microbiologyevents', 'labevents', 'd_labitems', 'prescriptions']:
                        effective_id_var = 'subject_id'
                    elif source.table in ['inputevents', 'chartevents', 'outputevents', 'procedureevents']:
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
                    
                    # patient_ids可能是dict(包含stay_id和subject_id)或列表
                    if isinstance(expanded_patient_ids, dict):
                        # 使用对应列的ID
                        id_values = expanded_patient_ids.get(effective_id_var)
                        if id_values:
                            filters.append(
                                FilterSpec(
                                    column=effective_id_var,
                                    op=FilterOp.IN,
                                    value=id_values,
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
            
            # 尝试从缓存获取
            if cache_key in self._table_cache:
                if verbose:
                    if DEBUG_MODE: print(f"   ♻️  使用缓存的表: {source.table} (跳过 {len(patient_filter_in_filters.value) if patient_filter_in_filters else 0} 个患者的加载)")
                # 从缓存获取ICUTable对象
                cached_table = self._table_cache[cache_key]
                frame = cached_table.data.copy()
                # 应用其他过滤器（如 sub_var/ids）
                for f in other_filters_list:
                    frame = f.apply(frame)
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
                    table = data_source.load_table(source.table, filters=filters, verbose=verbose)
                    frame = table.data.copy()
                    # 仅当有患者过滤器时才缓存
                    if patient_filter_in_filters:
                        # 缓存只应用了患者过滤器的表
                        patient_only_table = data_source.load_table(
                            source.table,
                            filters=[patient_filter_in_filters],
                            verbose=False
                        )
                        self._table_cache[cache_key] = patient_only_table
                        if verbose:
                            if DEBUG_MODE: print(f"   💾 缓存表 {source.table}: {len(patient_filter_in_filters.value)} 个患者")
                except (KeyError, FileNotFoundError, ValueError) as e:
                    # 如果表不存在，跳过这个源
                    if DEBUG_MODE:
                        print(f"   ⚠️  表 '{source.table}' 不存在或无法加载，跳过此源: {e}")
                    continue
            
            # MIMIC-IV特殊处理：若表为labevents/microbiologyevents/inputevents，仅有subject_id，按时间窗口映射到对应ICU stay
            if DEBUG_MODE:
                print(f"   📊 加载后数据: {source.table}, 行数={len(frame)}, itemid过滤={source.ids}")
                if source.ids and source.sub_var and source.sub_var in frame.columns:
                    print(f"       - {source.sub_var} 唯一值: {sorted(frame[source.sub_var].unique())[:10]}")
            if DEBUG_MODE:
                if DEBUG_MODE: print(f"   🔍 调试 {source.table}: 'subject_id' in frame={('subject_id' in frame.columns)}, 'stay_id' in frame={('stay_id' in frame.columns)}, defaults.id_var={defaults.id_var}")
            if source.table in ['labevents', 'microbiologyevents', 'inputevents'] and 'subject_id' in frame.columns and 'stay_id' not in frame.columns:
                if DEBUG_MODE: print(f"   ➡️  进入 MIMIC-IV 特殊处理: {source.table}")
                try:
                    # 仅加载相关subject的icustays，并携带intime/outtime用于窗口过滤
                    icustay_filters = []
                    if patient_ids:
                        # 使用扩展后的 patient_ids（如果之前已经扩展）
                        current_patient_ids = expanded_patient_ids if 'expanded_patient_ids' in locals() else patient_ids
                        subj_vals = current_patient_ids.get('subject_id') if isinstance(current_patient_ids, dict) else current_patient_ids
                        if subj_vals:
                            icustay_filters.append(
                                FilterSpec(column='subject_id', op=FilterOp.IN, value=subj_vals)
                            )
                    icustays = data_source.load_table('icustays', filters=icustay_filters if icustay_filters else None, verbose=verbose)
                    if hasattr(icustays, 'data'):
                        # 包含hadm_id以便匹配同一住院的数据
                        cols = ['subject_id', 'stay_id', 'hadm_id', 'intime', 'outtime']
                        icu_df = icustays.data[[c for c in cols if c in icustays.data.columns]].drop_duplicates()
                    else:
                        cols = ['subject_id', 'stay_id', 'hadm_id', 'intime', 'outtime']
                        icu_df = icustays[[c for c in cols if c in icustays.columns]].drop_duplicates()

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
                        # 🔧 修复：只保留同一住院（hadm_id）的数据，避免混入患者其他住院的历史数据
                        if 'hadm_id' in frame.columns and 'hadm_id' in icu_df.columns:
                            # 同时匹配subject_id和hadm_id，确保只取同一次住院的数据
                            tmp = frame.merge(icu_df, on=['subject_id', 'hadm_id'], how='inner')
                        else:
                            # 如果没有hadm_id，只能按subject_id匹配（可能混入其他住院数据）
                            tmp = frame.merge(icu_df, on='subject_id', how='inner')
                        
                        # CRITICAL FIX: Filter by time window to remove data after ICU discharge
                        # R ricu behavior:
                        # 1. KEEPS data before ICU admission (negative time) - clinically relevant
                        # 2. REMOVES data after ICU discharge (beyond outtime) - not part of ICU stay
                        # The hadm_id matching above ensures same hospitalization
                        # Time window filtering removes post-ICU-discharge lab results
                        before_filter = len(tmp)
                        # Keep: charttime <= outtime (remove post-discharge data)
                        # Note: We do NOT filter charttime >= intime, allowing negative time (pre-ICU)
                        mask_out = tmp['outtime'].isna() | (tmp[time_col] <= tmp['outtime'])
                        tmp = tmp[mask_out].copy()
                        after_filter = len(tmp)
                        if DEBUG_MODE and verbose and before_filter > after_filter:
                            print(f"      ⏱️  时间窗口过滤（仅过滤outtime后）: {source.table} 从 {before_filter} 行过滤到 {after_filter} 行")
                        
                        # CRITICAL FIX: 无论tmp是否为空，都要更新frame
                        # 如果tmp为空（没有匹配的数据），frame也应该为空
                        if not tmp.empty:
                            # 将过滤后的数据作为新frame，仅保留必要列
                            frame = tmp.drop(columns=['intime', 'outtime'])
                            if DEBUG_MODE: print(f"   ✅ MIMIC-IV: {source.table} 合并后非空，行数={len(frame)}, 列={list(frame.columns)[:10]}")
                        else:
                            # tmp为空，说明没有匹配的住院数据，frame也应该为空
                            print(f"   ⚠️  MIMIC-IV: {source.table} 合并后为空！原始列={list(frame.columns)[:10]}")
                            frame = pd.DataFrame(columns=frame.columns)
                            
                        # 🔗 关键修复：如果用户提供了特定的 stay_id，在映射后再次过滤
                        # 确保只返回用户指定的 stay_id 的数据
                        if 'stay_id' in frame.columns and patient_ids:
                            current_patient_ids = expanded_patient_ids if 'expanded_patient_ids' in locals() else patient_ids
                            if isinstance(current_patient_ids, dict) and 'stay_id' in current_patient_ids:
                                specified_stay_ids = current_patient_ids['stay_id']
                                if specified_stay_ids:
                                    frame = frame[frame['stay_id'].isin(specified_stay_ids)].copy()
                        
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

            # 如果配置中没有ID列，尝试从数据中自动检测
            if not table.id_columns:
                # 检查数据中是否有常见的ID列
                common_id_cols = ['stay_id', 'icustay_id', 'hadm_id', 'subject_id', 
                                 'patientunitstayid', 'patientid', 'admissionid']
                found_id_cols = [col for col in common_id_cols if col in frame.columns]
                if found_id_cols:
                    # 优先使用stay_id（MIMIC-IV）或第一个找到的ID列
                    preferred_id = 'stay_id' if 'stay_id' in found_id_cols else found_id_cols[0]
                    id_columns = [preferred_id]
                    if DEBUG_MODE: print(f"   🔍 自动检测到ID列: {preferred_id}")
            else:
                id_columns = id_columns or list(table.id_columns)
            
            index_column = index_column or source.index_var or table.index_column
            unit_column = unit_column or source.unit_var or table.unit_column

            time_columns = list(
                {
                    *time_columns,
                    *(table.time_columns or []),
                    *( [index_column] if index_column else []),
                }
            )

            value_column = source.value_var or table.value_column
            if value_column is None:
                raise ValueError(
                    f"Concept '{concept_name}' has no value column in table "
                    f"'{source.table}'. Provide 'value_var' in the dictionary."
                )

            # 🔧 FIX: 检查是否有 apply_map(var='sub_var') 回调
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
            
            if value_column not in frame.columns:
                # 对于某些概念（如lgl_cncpt），value_column可能通过callback创建
                # 先尝试应用callback，然后再检查
                frame = _apply_callback(
                    frame,
                    source,
                    concept_name,
                    unit_column,
                )
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
                if not source.sub_var:
                    raise ValueError(
                        f"Concept '{concept_name}' specifies a regex but no 'sub_var'."
                    )
                if source.sub_var not in frame.columns:
                    # 如果sub_var列不存在，跳过这个源
                    frame = pd.DataFrame()
                    continue
                # 🔧 FIX: 使用 regex=True 并抑制 UserWarning
                # str.contains 会警告如果正则表达式有捕获组但没有使用 str.extract
                # 这里我们只需要匹配，不需要提取，所以抑制这个警告
                pattern = source.regex
                series = frame[source.sub_var].astype(str)
                # 使用 regex=True, na=False, 并抑制捕获组警告
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 
                                          message='This pattern is interpreted as a regular expression',
                                          category=UserWarning)
                    frame = frame[series.str.contains(pattern, case=False, na=False, regex=True)]

            # ⚠️ CRITICAL FIX: 必须先应用回调函数（如 convert_unit）再应用值范围过滤和单位过滤
            # 原因：temp 等概念可能需要先将华氏度转换为摄氏度，然后再过滤 32-42°C 的范围和 C/°C 单位
            # 如果先过滤，华氏度值（97-100°F）和单位（°F）会因为不符合要求而被误删
            
            # 应用回调（在值范围过滤和单位过滤之前）
            frame = _apply_callback(
                frame,
                source,
                concept_name,
                unit_column,
            )
            
            # 单位过滤（在回调之后）
            if definition.units and unit_column and unit_column in frame.columns:
                allowed_units = {unit.lower() for unit in definition.units}
                series = frame[unit_column].astype(str).str.lower()
                frame = frame[series.isin(allowed_units)]
            
            # 🔧 FIX: 只有在concept_name列存在时才dropna
            # 但不要过早删除，因为某些回调函数可能会处理NaN值
            # 只在明确需要时才删除NaN（例如，在应用min/max过滤之前）
            if concept_name in frame.columns:
                # 先不删除NaN，因为某些概念（如urine24）可能需要保留NaN
                # 只在值范围过滤之前删除明显无效的NaN
                # 但如果值范围已定义，可以在过滤后删除NaN
                pass  # 暂时不删除NaN，让后续处理决定

            # 🔧 FIX: 值范围过滤（在回调之后）
            # 现在值已经经过转换（如华氏度→摄氏度），可以安全过滤
            if definition.minimum is not None:
                frame = frame[frame[concept_name] >= definition.minimum]
            if definition.maximum is not None:
                frame = frame[frame[concept_name] <= definition.maximum]
            
            # 在值范围过滤后，删除无效的NaN（但保留有效范围内的NaN用于后续处理）
            if concept_name in frame.columns:
                # 只删除明显无效的NaN（在值范围过滤之后）
                # 这样可以确保有效值不会被误删
                frame = frame.dropna(subset=[concept_name])

            # 🔧 FIX: 如果使用了 apply_map(var='sub_var')，将映射后的 sub_var 复制到 concept_name
            if uses_sub_var_mapping and source.sub_var in frame.columns:
                # sub_var 列已经被 apply_map 映射为类别值，将其复制到 concept_name 列
                # 但是要先保存原始的数值列（如果需要的话，用于后续持续时间计算）
                # 对于 mech_vent 这种概念，原始 value 列包含持续时间，需要保留
                if concept_name in frame.columns and source.value_var:
                    # 保存原始数值列为 _duration_val
                    frame['_duration_val'] = frame[concept_name]
                # 将映射后的类别值复制到 concept_name
                frame[concept_name] = frame[source.sub_var]

            keep_cols = {
                *(id_columns or []),
                *( [index_column] if index_column else []),
                concept_name,
            }
            # 添加实际存在的time_columns（不强制要求所有time_columns都存在）
            for tc in (time_columns or []):
                if tc in frame.columns:
                    keep_cols.add(tc)
            
            if unit_column and unit_column in frame.columns:
                keep_cols.add(unit_column)
            
            # 保留 _duration_val 列（如果存在），用于后续持续时间计算
            if '_duration_val' in frame.columns:
                keep_cols.add('_duration_val')
            
            # 只检查必需的列：id_columns, index_column, concept_name
            # 注意：对于多源概念，不同源可能使用不同的时间列名（如starttime vs charttime）
            # 所以对于索引列，我们只检查是否在数据中有任何时间列
            required_cols = {
                *(id_columns or []),
                concept_name,
            }
            missing = required_cols - set(frame.columns)
            
            # 对于索引列，检查是否在数据中有任何时间列
            if index_column:
                # 检查是否有index_column，或者有类似的时间列
                time_cols = [col for col in frame.columns if 'time' in col.lower() or col in ['starttime', 'endtime', 'charttime', 'storetime']]
                if index_column not in frame.columns and not time_cols:
                    missing.add(index_column)
            
            if missing:
                # 对于labevents等表，如果缺少stay_id但映射过程已处理，应该已经有stay_id了
                # 如果还是没有，说明映射失败，跳过这个源并继续（不报错）
                if 'stay_id' in missing and source.table in ['labevents', 'microbiologyevents']:
                    frame = pd.DataFrame()
                    continue
                # 对于多源概念，如果某个源缺少index_column但其他源有，这是可以接受的
                if index_column in missing and len(sources) > 1:
                    missing.discard(index_column)
                
                if missing:
                    raise KeyError(
                        f"Missing expected columns {sorted(missing)} in concept "
                        f"data for '{concept_name}' (table '{source.table}')"
                    )
            # 确保ID列在数据中
            available_id_cols = [col for col in id_columns if col in frame.columns]
            if not available_id_cols and id_columns:
                print(f"   ⚠️  警告: 配置的ID列 {id_columns} 不在数据中，可用列: {list(frame.columns)[:10]}")
            
            ordered_cols: List[str] = []
            # 保留所有可用的ID列（不只是第一个）
            ordered_cols.extend(available_id_cols)
            if index_column and index_column not in ordered_cols:
                ordered_cols.append(index_column)
            extra_time = [
                col for col in time_columns if col and col not in ordered_cols
            ]
            ordered_cols.extend(extra_time)
            ordered_cols.append(concept_name)
            if unit_column and unit_column not in ordered_cols:
                ordered_cols.append(unit_column)
            ordered_cols = [col for col in ordered_cols if col in frame.columns]
            frames.append(frame.loc[:, ordered_cols])

        combined = pd.concat(frames, ignore_index=True)
        sort_keys = [col for col in id_columns if col]
        if index_column:
            sort_keys.append(index_column)
        if sort_keys:
            combined = combined.sort_values(by=sort_keys)
        combined = combined.reset_index(drop=True)
        agg_value = self._coerce_final_aggregator(aggregator)

        # CRITICAL FIX: Avoid double aggregation issue
        # Strategy: Only use change_interval's aggregation (on relative time after floor)
        # Do NOT use _apply_aggregation before time alignment
        should_aggregate_in_change_interval = agg_value is not False
        
        # Only set unit_column if it actually exists in the combined data
        final_unit_column = unit_column if unit_column and unit_column in combined.columns else None
        
        # Apply interval alignment and aggregation if interval is specified
        if interval is not None and index_column and index_column in combined.columns:
            from .ts_utils import change_interval
            
            # Align time to ICU admission if requested (BEFORE any aggregation)
            if align_to_admission:
                combined = self._align_time_to_admission(
                    combined,
                    data_source,
                    id_columns,
                    index_column
                )
            
            # Determine aggregation method for change_interval
            # This is the ONLY aggregation we should do (on relative time)
            agg_method = aggregator if aggregator not in (None, False, "auto") else None
            if agg_method is None:
                # Check if definition has aggregate
                definition = self.dictionary[concept_name]
                if definition.aggregate is not None:
                    agg_method = definition.aggregate
            # Default to 'median' for numeric values if no aggregate specified (matches R ricu)
            if agg_method is None:
                # Check if value column is numeric
                if concept_name in combined.columns:
                    if pd.api.types.is_numeric_dtype(combined[concept_name]):
                        agg_method = 'median'  # Matches R ricu default
                    else:
                        agg_method = 'first'  # For non-numeric, keep first
            
            # Create ICUTable temporarily to use change_interval
            temp_table = ICUTable(
                data=combined,
                id_columns=id_columns,
                index_column=index_column,
                value_column=concept_name,
                unit_column=final_unit_column,
                time_columns=[col for col in time_columns if col],
            )
            
            # Apply interval change with aggregation (SINGLE aggregation on relative time)
            combined_result = change_interval(
                temp_table,
                interval=interval,
                aggregation=agg_method,
                fill_gaps=False  # 不填充gaps，因为数据应该已经对齐
            )
            
            # Extract data if ICUTable is returned
            if hasattr(combined_result, 'data'):
                combined = combined_result.data
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
        
        return ICUTable(
            data=combined,
            id_columns=id_columns,
            index_column=index_column,
            value_column=concept_name,
            unit_column=final_unit_column,
            time_columns=[col for col in time_columns if col],
        )
    
    def _align_time_to_admission(
        self,
        data: pd.DataFrame,
        data_source: ICUDataSource,
        id_columns: List[str],
        index_column: str,
    ) -> pd.DataFrame:
        """Align time column to ICU admission time as anchor (R ricu as_dt_min).
        
        Converts absolute time to relative time (hours or minutes since ICU admission).
        This replicates R ricu's behavior where time is relative to admission.
        
        Args:
            data: Input DataFrame with time column
            data_source: Data source instance
            id_columns: ID columns (e.g., ['stay_id'])
            index_column: Time column name (e.g., 'charttime')
            
        Returns:
            DataFrame with time converted to hours since ICU admission
        """
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

                # 🔧 修复：R ricu保留负时间（入ICU前的数据）
                # 不过滤 >= 0，只过滤ICU窗口后的数据
                # 应用范围过滤：仅过滤 <= ICU长度（若可得）
                if icu_len_hours is not None:
                    mask = data[index_column] <= icu_len_hours
                    data = data[mask].copy()
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
            
            # 🔧 关键修复：R ricu 不过滤超出 ICU 时间窗口的数据
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

        sub_names = list(definition.sub_concepts)
        if not sub_names:
            raise ValueError(
                f"Recursive concept '{concept_name}' specifies no sub concepts."
            )

        aggregate_mapping = self._build_sub_aggregate(definition.aggregate, sub_names)

        sub_tables = self.load_concepts(
            sub_names,
            data_source,
            merge=False,
            aggregate=aggregate_mapping,
            patient_ids=patient_ids,
            verbose=True,  # 临时开启verbose查看子概念加载
            interval=interval,  # Pass interval to recursive calls
            align_to_admission=align_to_admission,  # Pass align flag
            **kwargs,  # Pass kwargs to sub concepts
        )

        if isinstance(sub_tables, ICUTable):
            sub_tables = {sub_names[0]: sub_tables}

        # 🔧 CRITICAL FIX: Align WinTbl time columns BEFORE passing to callbacks
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
                    
                    if verbose:
                        print(f"   🔧 对齐 WinTbl '{name}': index_var={idx_col}, dur_var={dur_col}")
                        if idx_col in table.data.columns:
                            print(f"      index_var 类型: {table.data[idx_col].dtype}")
                        if dur_col and dur_col in table.data.columns:
                            print(f"      dur_var 类型: {table.data[dur_col].dtype}")
                    
                    # Align index_var (start time) if it's datetime
                    if idx_col and idx_col in table.data.columns and pd.api.types.is_datetime64_any_dtype(table.data[idx_col]):
                        if verbose:
                            print(f"      ✅ 转换 index_var 从 datetime 到小时")
                        table.data = self._align_time_to_admission(
                            table.data,
                            data_source,
                            id_cols,
                            idx_col
                        )
                    
                    # Convert dur_var (duration) from timedelta to hours
                    if dur_col and dur_col in table.data.columns:
                        if pd.api.types.is_timedelta64_dtype(table.data[dur_col]):
                            if verbose:
                                print(f"      ✅ 转换 dur_var 从 timedelta 到小时")
                            table.data[dur_col] = table.data[dur_col].dt.total_seconds() / 3600.0
                        elif pd.api.types.is_datetime64_any_dtype(table.data[dur_col]):
                            # If dur_var is datetime (shouldn't happen), warn
                            print(f"   ⚠️  警告: WinTbl '{name}' 的 dur_var '{dur_col}' 是 datetime 类型，预期是 timedelta")
                
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
        result = execute_concept_callback(definition.callback, sub_tables, ctx)

        # 🔧 CRITICAL: Align WinTbl result time columns immediately after callback
        # This ensures that when this concept is used as a sub-concept in parent recursion,
        # it already has numeric time columns (not datetime)
        from .table import WinTbl
        if isinstance(result, WinTbl) and align_to_admission and not result.data.empty:
            idx_col = result.index_var
            dur_col = result.dur_var
            id_cols = result.id_vars
            
            # Align index_var if it's still datetime
            if idx_col and idx_col in result.data.columns and pd.api.types.is_datetime64_any_dtype(result.data[idx_col]):
                if verbose:
                    print(f"   🔧 对齐 WinTbl 结果 '{concept_name}': index_var={idx_col} (datetime → 小时)")
                result.data = self._align_time_to_admission(
                    result.data,
                    data_source,
                    id_cols,
                    idx_col
                )
            
            # Convert dur_var from timedelta to hours
            if dur_col and dur_col in result.data.columns and pd.api.types.is_timedelta64_dtype(result.data[dur_col]):
                if verbose:
                    print(f"   🔧 转换 WinTbl 结果 '{concept_name}': dur_var={dur_col} (timedelta → 小时)")
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
            
            # 🔧 关键修复：如果时间列是datetime类型但应该是numeric（align_to_admission=True），
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
                    
                    # 🔧 WinTbl 特殊处理：dur_var（持续时间）也需要转换
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
                    
                    # 🔧 WinTbl: 同时转换 dur_var
                    if dur_col and dur_col in result.data.columns:
                        if pd.api.types.is_timedelta64_dtype(result.data[dur_col]):
                            result.data[dur_col] = result.data[dur_col].dt.total_seconds() / 3600.0
            
            # 🔧 CRITICAL: Expand WinTbl to time series before applying interval aggregation
            # WinTbl represents time windows (start_time, duration) and must be expanded
            # to individual time points when interval is specified
            if isinstance(result, WinTbl) and not result.data.empty:
                idx_col = result.index_var
                dur_col = result.dur_var
                id_cols = result.id_vars
                
                if idx_col and dur_col and idx_col in result.data.columns and dur_col in result.data.columns:
                    if verbose:
                        print(f"   🔧 扩展 WinTbl '{concept_name}' 到时间序列 (interval={interval})")
                    
                    # 扩展窗口到时间序列
                    expanded_rows = []
                    for _, row in result.data.iterrows():
                        start_time = row[idx_col]
                        duration = row[dur_col]
                        
                        # 计算结束时间（小时）
                        end_time = start_time + duration
                        
                        # 生成时间序列（每个 interval）
                        interval_hours = interval.total_seconds() / 3600.0
                        current_time = np.floor(start_time / interval_hours) * interval_hours
                        
                        while current_time < end_time:
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
                        if verbose:
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
            agg_method = aggregator if aggregator not in (None, False, "auto") else None
            if agg_method is None and definition.aggregate is not None:
                agg_method = definition.aggregate
            # 🔧 FIX: GCS total score should use 'min' aggregation (for recursive concepts)
            # But GCS sub-components should use default aggregation (median)
            if concept_name == 'gcs':
                if agg_method is None or (isinstance(agg_method, str) and agg_method != 'min'):
                    agg_method = 'min'
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
            if agg_method and not result.data.empty and not isinstance(result, WinTbl):
                try:
                    combined_result = change_interval(
                        result,
                        interval=interval,
                        aggregation=agg_method
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

        table = data_source.load_table(id_cfg.table)
        frame = table.data[[id_cfg.id, id_cfg.start, id_cfg.end]].copy()
        frame = frame.dropna(subset=[id_cfg.start, id_cfg.end])

        frame[concept_name] = (
            pd.to_datetime(frame[id_cfg.end]) - pd.to_datetime(frame[id_cfg.start])
        ).dt.total_seconds() / 86400.0

        frame = frame[frame[concept_name] >= 0]
        if patient_ids is not None:
            frame = frame[frame[id_cfg.id].isin(set(patient_ids))]

        frame = frame[[id_cfg.id, concept_name]].reset_index(drop=True)
        return ICUTable(
            data=frame,
            id_columns=[id_cfg.id],
            index_column=None,
            value_column=concept_name,
        )

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
        base_tables = self.load_concepts(
            [base_name],
            data_source,
            merge=False,
            aggregate=None,
            patient_ids=patient_ids,
        )
        if isinstance(base_tables, ICUTable):
            base_table = base_tables
        else:
            base_table = base_tables[base_name]

        data = base_table.data.copy()
        value_col = base_table.value_column or base_name

        comp_match = re.search(r"comp_na\(`(.+?)`,\s*(.+?)\)", callback, flags=re.DOTALL)
        if comp_match:
            op_symbol = comp_match.group(1)
            literal = _parse_literal(comp_match.group(2))
            series = data[value_col]
            if op_symbol == "<=":
                mask = pd.to_numeric(series, errors="coerce") <= literal
            elif op_symbol == "==":
                mask = series.astype(str) == str(literal)
            elif op_symbol == "!=":
                mask = series.astype(str) != str(literal)
            else:
                raise NotImplementedError(f"Unsupported comparison operator '{op_symbol}'")
        else:
            mask = pd.Series(True, index=data.index)

        data = data[mask].copy()

        if "ts_to_win_tbl" in callback:
            if base_table.index_column is None:
                raise ValueError("ts_to_win_tbl requires a time indexed table.")
            dur_match = re.search(r"ts_to_win_tbl\((.+)\)", callback, flags=re.DOTALL)
            duration = self._parse_interval_expression(dur_match.group(1) if dur_match else "mins(60)")
            win_df = data[list(base_table.id_columns) + [base_table.index_column]].copy()
            win_df["duration"] = duration
            win_df[concept_name] = True
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
        value = _parse_literal(match.group(2))
        if isinstance(value, pd.Timedelta):
            return value
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
        merged: Optional[pd.DataFrame] = None
        index_column: Optional[str] = None
        id_columns: Optional[List[str]] = None

        for name, table in tables.items():
            frame = table.data.copy()

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
            # 🔧 FIX: 保留所有值列，不仅仅是概念名称列
            # 这对于 keep_components=True 的情况很重要（如 SOFA 组件）
            # 但是要排除单位列（valueuom），因为它会导致合并时的列冲突
            # 单位列通常不需要保留，因为值已经标准化了
            excluded_cols = ['valueuom', 'unit']  # 排除这些列以避免合并冲突
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
                # 🔧 确保索引层级一致
                if merged.index.nlevels != frame.index.nlevels:
                    print(f"⚠️  警告: 索引层级不一致 ({merged.index.nlevels} vs {frame.index.nlevels})，重置索引后重新合并")
                    # 重置为共同的索引列
                    common_keys = [col for col in merged.index.names if col in frame.index.names]
                    merged = merged.reset_index()
                    frame = frame.reset_index()
                    merged = merged.merge(frame, on=common_keys, how='outer')
                    merged = merged.sort_values(common_keys)
                    merged = merged.set_index(common_keys)
                else:
                    merged = merged.join(frame, how="outer")

        if merged is None:
            return pd.DataFrame()

        merged = merged.reset_index()
        return merged

    def _apply_aggregation(
        self,
        frame: pd.DataFrame,
        concept_name: str,
        id_columns: List[str],
        index_column: Optional[str],
        unit_column: Optional[str],
        aggregator: object,
    ) -> pd.DataFrame:
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


def _apply_callback(
    frame: pd.DataFrame,
    source: ConceptSource,
    concept_name: str,
    unit_column: Optional[str] = None,
) -> pd.DataFrame:
    callback = source.callback
    if not callback:
        return frame

    expr = callback.strip()
    
    if DEBUG_MODE:
        print(f"   🔧 应用回调: {expr} (输入行数={len(frame)})")

    if expr == "identity_callback":
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
        frame.loc[:, concept_name] = series.apply(
            lambda item: False if pd.isna(item) else comparator(item, value)
        )
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
        
        # 🔧 FIX: 解析 var_param，如果是 'sub_var'，使用 source.sub_var 的实际值
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
            
            # 🔧 FIX: 显式转换为 object 类型以避免 FutureWarning
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
            
            # 🔧 FIX: 同样处理类型不兼容问题
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
        
        if DEBUG_MODE:
            print(f"       convert_unit: symbol={symbol}, value={value}, new_unit={new_unit}, old_unit={old_unit}")

        frame = frame.copy()
        
        # 🔧 FIX: 如果 source.unit_var 未指定，尝试自动检测单位列
        actual_unit_var = source.unit_var or unit_column
        
        # 如果仍然没有，尝试常见的单位列名
        if not actual_unit_var and 'valueuom' in frame.columns:
            actual_unit_var = 'valueuom'
        elif not actual_unit_var and 'unit' in frame.columns:
            actual_unit_var = 'unit'
        
        if actual_unit_var and actual_unit_var in frame.columns:
            unit_series = frame[actual_unit_var].astype(str)
            if old_unit:
                # 🔧 FIX: 更宽松的单位匹配
                # - 'f' 应该匹配 '°F', 'deg F', 'F', 'f', '°f', 'degF' 等
                # - 'c' 应该匹配 '°C', 'deg C', 'C', 'c', '°c', 'degC' 等
                # 方法：移除度数符号和空格，然后比较
                old_unit_clean = old_unit.replace('°', '').replace('deg', '').replace(' ', '').lower()
                unit_series_clean = unit_series.str.replace('°', '', regex=False).str.replace('deg', '', regex=False).str.replace(' ', '', regex=False).str.lower()
                mask = unit_series_clean == old_unit_clean
            else:
                mask = pd.Series(True, index=frame.index)
            
            if DEBUG_MODE:
                print(f"       unit_var={actual_unit_var}, 匹配行数={mask.sum()}/{len(frame)}")
                if mask.sum() > 0:
                    print(f"       匹配的单位: {unit_series[mask].unique()[:5]}")
        else:
            mask = pd.Series(True, index=frame.index)
            if DEBUG_MODE:
                print(f"       无unit_var，处理所有行")

        numeric = pd.to_numeric(frame.loc[mask, concept_name], errors="coerce")
        transformed = _apply_binary_op(symbol, numeric, value)
        frame.loc[mask, concept_name] = transformed
        
        if DEBUG_MODE:
            print(f"       转换后非NaN行数: {transformed.notna().sum()}/{len(transformed)}")

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
        return vent_flag(
            frame,
            val_col=concept_name,
            index_var=index_var,
            id_cols=id_cols,
        )

    match = re.fullmatch(r"eicu_duration\(\s*gap_length\s*=\s*(.+)\)", expr, flags=re.DOTALL)
    if match:
        from .callback_utils import eicu_duration_callback

        gap_arg = match.group(1)
        gap = _parse_interval_expression(gap_arg)
        callback_fn = eicu_duration_callback(gap)
        id_cols = [col for col in frame.columns if 'id' in col.lower()]
        index_var = source.index_var
        return callback_fn(
            frame,
            val_col=concept_name,
            index_var=index_var,
            id_cols=id_cols,
        )

    raise NotImplementedError(
        f"Callback '{callback}' is not yet supported."
    )


def _apply_binary_op(symbol: str, series: pd.Series, value: object) -> pd.Series:
    """Apply binary operation or conversion function."""
    # Import conversion functions
    from .callback_utils import fahr_to_cels
    from .unit_conversion import celsius_to_fahrenheit, fahrenheit_to_celsius
    
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

    return op_map[symbol](series, value)


def _parse_binary_op(expr: str) -> tuple[str, object]:
    """Parse binary_op expression.
    
    Handles both:
    - binary_op(`+`, 10)
    - fahr_to_cels (function name only)
    """
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
    # 🔧 FIX: 支持反引号（R语言中用于标识符）
    if raw.startswith("`") and raw.endswith("`"):
        # 去掉反引号，然后尝试解析为数字或返回字符串
        raw = raw[1:-1]
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return _strip_quotes(raw)
    if raw.endswith("L"):
        raw = raw[:-1]
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


# 别名 - 为了兼容性
Concept = ConceptDefinition  # Concept 类别名，指向 ConceptDefinition


def load_dictionary(src_name: Optional[str] = None) -> ConceptDictionary:
    """
    加载概念字典 - 兼容函数
    
    Args:
        src_name: 数据源名称（可选）
        
    Returns:
        ConceptDictionary 实例
    """
    from .resources import load_dictionary as _load_dictionary

    # 当前实现不根据数据源过滤概念，但保留参数以兼容既有调用
    return _load_dictionary()
