"""
完整的概念加载系统
实现 R ricu 的 load_concepts 功能
"""
from typing import List, Optional, Union, Dict, Any, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta

from .concept import Concept, load_dictionary
from .datasource import ICUDataSource
from .table import load_table
from .ts_utils import change_interval, aggregate_data
from .callback_utils import combine_callbacks

# DataSource 别名用于向后兼容
DataSource = ICUDataSource


class ConceptLoader:
    """概念加载器 - 复刻 R ricu 的 load_concepts"""
    
    def __init__(self, src: Union[str, DataSource]):
        """
        初始化概念加载器
        
        Args:
            src: 数据源名称或 DataSource 对象
        """
        if isinstance(src, str):
            from .config import load_src_cfg
            self.src = load_src_cfg(src)
        else:
            self.src = src
            
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
            concept_dict = load_dictionary(self.src.name)
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
        
        try:
            df = load_table(self.src.name, table_name)
        except Exception as e:
            print(f"警告: 无法加载表 {table_name}: {e}")
            return pd.DataFrame()
        
        # 2. 过滤患者
        if patient_ids is not None:
            id_col = self._get_id_column(df, id_type)
            if id_col:
                if isinstance(patient_ids, pd.DataFrame):
                    df = df.merge(patient_ids, on=id_col, how='inner')
                else:
                    df = df[df[id_col].isin(patient_ids)]
        
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
        
        try:
            df = load_table(self.src.name, table_name)
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
                if isinstance(patient_ids, pd.DataFrame):
                    df = df.merge(patient_ids, on=id_col, how='inner')
                else:
                    df = df[df[id_col].isin(patient_ids)]
        
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
                        concept_dict = load_dictionary(self.src.name)
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
        
        src_name = self.src.name if hasattr(self.src, 'name') else str(self.src)
        
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
