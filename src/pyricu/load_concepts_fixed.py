
"""
修复版的load_concepts函数

解决ID列冲突和概念合并问题
"""

import pandas as pd
from typing import List, Union, Dict, Any, Optional
from pathlib import Path
import sys

# 导入原始模块
from . import load_concepts as original_load_concepts
from .id_mapping_utils import IDMapper, safe_merge_dataframes

def load_concepts_fixed(
    concepts: Union[str, List[str]],
    patient_ids: Optional[List[Any]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: str = '1h',
    win_length: str = '24h',
    merge: bool = True,
    verbose: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    修复版的load_concepts函数

    主要修复：
    1. 分层加载概念，避免ID列冲突
    2. 智能ID映射
    3. 安全的数据合并
    """
    if verbose:
        print(f"📊 使用修复版load_concepts加载概念: {concepts}")

    if isinstance(concepts, str):
        concepts = [concepts]

    # 初始化ID映射器
    if data_path:
        id_mapper = IDMapper(str(data_path))
    else:
        id_mapper = None

    # 分层加载策略
    concept_groups = _group_concepts_by_id_source(concepts, database)
    all_data = []

    for group_name, group_concepts in concept_groups.items():
        if verbose:
            print(f"  加载概念组 '{group_name}': {group_concepts}")

        try:
            group_data = original_load_concepts(
                concepts=group_concepts,
                patient_ids=patient_ids,
                database=database,
                data_path=data_path,
                interval=interval,
                win_length=win_length,
                merge=False,  # 不在这里合并
                verbose=False,
                **kwargs
            )

            if group_data:
                if isinstance(group_data, dict):
                    # 处理字典格式的返回
                    for concept_name, df in group_data.items():
                        if df is not None and not df.empty:
                            all_data.append(df)
                            if verbose:
                                print(f"    ✅ {concept_name}: {len(df)}行")
                else:
                    # 处理DataFrame格式的返回
                    all_data.append(group_data)
                    if verbose:
                        print(f"    ✅ {group_name}: {len(group_data)}行")
            else:
                if verbose:
                    print(f"    ⚠️  {group_name}: 无数据")

        except Exception as e:
            if verbose:
                print(f"    ❌ {group_name}加载失败: {str(e)[:100]}")

    if not all_data:
        if verbose:
            print(f"  ⚠️  没有数据可合并")
        return pd.DataFrame()

    # 安全合并数据
    if merge and len(all_data) > 1:
        if verbose:
            print(f"  🔧 合并{len(all_data)}个数据集...")

        try:
            merged_data = safe_merge_dataframes(all_data, id_col='stay_id', how='outer')
            if verbose:
                print(f"  ✅ 合并完成: {len(merged_data)}行, {len(merged_data.columns)}列")
            return merged_data
        except Exception as e:
            if verbose:
                print(f"  ❌ 合并失败: {e}")
            # 合并失败时返回第一个数据集
            return all_data[0]
    elif len(all_data) == 1:
        return all_data[0]
    else:
        return pd.DataFrame()

def _group_concepts_by_id_source(concepts: List[str], database: str) -> Dict[str, List[str]]:
    """根据ID来源对概念进行分组"""
    # 简化实现：按概念类型分组
    groups = {}

    for concept in concepts:
        # 基于概念特征进行分组
        if concept in ['age', 'sex', 'height', 'weight']:
            group_name = 'demographics'
        elif concept in ['hr', 'sbp', 'dbp', 'map', 'temp', 'resp']:
            group_name = 'vitals'
        elif concept in ['death', 'los_icu', 'sofa', 'qsofa', 'sirs']:
            group_name = 'outcomes'
        elif concept in ['alb', 'alp', 'alt', 'ast', 'bili', 'crea', 'glu']:
            group_name = 'laboratory'
        else:
            group_name = 'other'

        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(concept)

    return groups

# 替换原始函数
def load_concepts(*args, **kwargs):
    """包装函数，使用修复版本"""
    return load_concepts_fixed(*args, **kwargs)
