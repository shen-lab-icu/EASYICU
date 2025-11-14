"""
ricu.R兼容性API - 简化版本

使用现有pyricu API但添加ricu.R兼容的时间窗口参数
"""

from typing import List, Union, Optional, Dict
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from .api import load_concepts


def load_concepts_ricu_compatible(
    concepts: Union[str, List[str]],
    patient_ids: Optional[List[int]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: str = '1h',
    window_hours: int = 2000,
    merge: bool = False,
    verbose: bool = False,
    **kwargs
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    ricu.R兼容的概念加载函数

    Args:
        concepts: 概念名称或列表
        patient_ids: 患者ID列表
        database: 数据库类型
        data_path: 数据路径
        interval: 时间间隔（默认1小时，匹配ricu.R的hours(1L)）
        window_hours: 扩展时间窗口（默认2000小时，匹配ricu.R的宽窗口）
        merge: 是否合并结果
        verbose: 是否显示详细信息

    Returns:
        DataFrame或概念字典
    """
    if isinstance(concepts, str):
        concepts = [concepts]

    if verbose:
        print(f"🔬 ricu.R兼容加载概念: {', '.join(concepts)}")
        print(f"   时间间隔: {interval}")
        print(f"   扩展窗口: {window_hours}小时")

    # 使用扩展窗口长度匹配ricu.R的数据范围
    extended_win_length = f"{window_hours}h"

    try:
        # 使用现有的load_concepts API，但设置大的时间窗口
        result = load_concepts(
            concepts=concepts,
            patient_ids=patient_ids,
            database=database,
            data_path=data_path,
            interval=interval,
            win_length=extended_win_length,
            merge=merge,
            verbose=verbose,
            **kwargs
        )

        if verbose:
            if isinstance(result, dict):
                for concept, df in result.items():
                    if df is not None and not df.empty:
                        print(f"  ✅ {concept}: {len(df)}行")
                    else:
                        print(f"  ⚠️  {concept}: 无数据")
            elif result is not None and not result.empty:
                print(f"  ✅ 合并结果: {len(result)}行")
            else:
                print(f"  ❌ 无数据")

        return result

    except Exception as e:
        if verbose:
            print(f"❌ ricu.R兼容加载失败: {e}")
        # 如果扩展窗口失败，尝试使用默认设置
        try:
            if verbose:
                print("🔄 尝试使用默认设置...")
            return load_concepts(
                concepts=concepts,
                patient_ids=patient_ids,
                database=database,
                data_path=data_path,
                interval=interval,
                merge=merge,
                verbose=verbose,
                **kwargs
            )
        except Exception as e2:
            if verbose:
                print(f"❌ 默认设置也失败: {e2}")
            return pd.DataFrame() if merge else {}


def load_lab_ricu_compatible(
    patient_ids: Optional[List[int]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    window_hours: int = 2000,
    verbose: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    ricu.R兼容的LAB模块加载

    Args:
        patient_ids: 患者ID列表
        database: 数据库类型
        data_path: 数据路径
        window_hours: 扩展时间窗口
        verbose: 是否显示详细信息

    Returns:
        LAB模块DataFrame
    """
    # LAB模块概念列表（基于ricu.R）
    lab_concepts = [
        'crea', 'glu', 'alb', 'alp', 'alt', 'ast', 'bicar', 'bili', 'bun',
        'ca', 'cl', 'k', 'mg', 'na', 'phos'
    ]

    if verbose:
        print(f"🔬 ricu.R兼容LAB模块加载")
        print(f"   概念: {', '.join(lab_concepts)}")

    try:
        result = load_concepts_ricu_compatible(
            concepts=lab_concepts,
            patient_ids=patient_ids,
            database=database,
            data_path=data_path,
            window_hours=window_hours,
            merge=True,
            verbose=verbose,
            **kwargs
        )

        return result

    except Exception as e:
        if verbose:
            print(f"❌ LAB模块加载失败: {e}")
        return pd.DataFrame()


def load_vitals_ricu_compatible(
    patient_ids: Optional[List[int]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    window_hours: int = 2000,
    verbose: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    ricu.R兼容的VITALS模块加载

    Args:
        patient_ids: 患者ID列表
        database: 数据库类型
        data_path: 数据路径
        window_hours: 扩展时间窗口
        verbose: 是否显示详细信息

    Returns:
        VITALS模块DataFrame
    """
    # VITALS模块概念列表（基于ricu.R）
    vitals_concepts = ['hr', 'sbp', 'dbp', 'map', 'temp']

    if verbose:
        print(f"💗 ricu.R兼容VITALS模块加载")
        print(f"   概念: {', '.join(vitals_concepts)}")

    try:
        result = load_concepts_ricu_compatible(
            concepts=vitals_concepts,
            patient_ids=patient_ids,
            database=database,
            data_path=data_path,
            window_hours=window_hours,
            merge=True,
            verbose=verbose,
            **kwargs
        )

        return result

    except Exception as e:
        if verbose:
            print(f"❌ VITALS模块加载失败: {e}")
        return pd.DataFrame()