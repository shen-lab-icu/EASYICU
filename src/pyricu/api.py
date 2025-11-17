"""
pyricu 高层API - 提供简单易用的接口，同时支持高级自定义

重构后的统一API，整合了多个模块的功能:
- api.py: 原始高层API
- api_enhanced.py: 缓存功能
- api_unified.py: 统一加载器
- load_concepts.py: 加载逻辑

两层设计:
1. Easy API - 预定义的便捷函数 (load_vitals, load_sofa等)
2. Concept API - 灵活的主API (load_concepts) 带智能默认值

使用示例:
    >>> from pyricu import load_concepts, load_sofa, load_vitals
    >>>
    >>> # 简单用法 - 自动检测数据库
    >>> hr = load_concepts('hr', patient_ids=[123, 456])
    >>>
    >>> # 完全自定义
    >>> sofa = load_concepts('sofa', patient_ids=[123, 456],
    ...                      database='miiv', data_path='/path/to/data',
    ...                      interval='6h', win_length='24h', aggregate='max')
    >>>
    >>> # Easy API - 开箱即用
    >>> vitals = load_vitals(patient_ids=[123, 456])
"""

from typing import List, Union, Optional, Dict
from pathlib import Path
import pandas as pd
import os
import logging

from .base import BaseICULoader, get_default_data_path, detect_database_type
from .resources import load_dictionary
from .config import load_data_sources

logger = logging.getLogger(__name__)

# 全局加载器实例，用于复用初始化开销
_global_loader = None
_loader_config = None


def _get_global_loader(
    database: Optional[str] = None,
    data_path: Optional[Path] = None,
    dict_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    **kwargs,
) -> BaseICULoader:
    """获取或创建全局加载器实例（减少重复初始化）"""
    global _global_loader, _loader_config

    if dict_path is None:
        dict_key = None
    elif isinstance(dict_path, (list, tuple)):
        dict_key = tuple(map(str, dict_path))
    else:
        dict_key = str(dict_path)

    current_config = (database, data_path, dict_key, frozenset(kwargs.items()))

    if _global_loader is None or _loader_config != current_config:
        _global_loader = BaseICULoader(
            database=database,
            data_path=data_path,
            dict_path=dict_path,
            **kwargs,
        )
        _loader_config = current_config

    return _global_loader


def load_concepts(
    concepts: Union[str, List[str]],
    patient_ids: Optional[Union[List, Dict]] = None,
    # 数据源参数 - 智能默认值
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    # 时间参数 - 可选
    interval: Optional[Union[str, pd.Timedelta]] = None,
    win_length: Optional[Union[str, pd.Timedelta]] = None,
    # 聚合参数
    aggregate: Optional[Union[str, Dict]] = None,
    # SOFA相关
    keep_components: bool = False,
    # 其他
    verbose: bool = False,
    use_sofa2: bool = False,  # 新增：是否使用SOFA2字典
    merge: bool = True,       # 新增：是否合并结果
    ricu_compatible: bool = False,  # 新增：返回ricu.R兼容格式
    dict_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    chunk_size: Optional[int] = None,
    progress: bool = False,
    parallel_workers: Optional[int] = None,
    concept_workers: int = 1,
    parallel_backend: str = 'auto',
    **kwargs,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    加载ICU概念数据 - pyricu的主要API (重构版本)

    这个函数使用统一的BaseICULoader，整合了多个模块的功能：
    - 原api.py的所有功能
    - api_enhanced.py的缓存支持
    - api_unified.py的统一逻辑
    - load_concepts.py的加载实现

    Args:
        concepts: 概念名称或概念名称列表
            例如: 'hr', ['hr', 'sbp', 'temp'], 'sofa', 'sofa2'
        patient_ids: 可选的患者ID列表或字典
            - List: [123, 456] (自动转换为正确的ID列)
            - Dict: {'stay_id': [123, 456]} (显式指定ID列)
            - None: 加载所有患者

        # === 数据源参数 (可选，有智能默认值) ===
        database: 数据库类型
            - None: 自动检测（从环境变量）
            - 'miiv', 'mimic', 'eicu', 'hirid', 'aumc'
        data_path: 数据路径
            - None: 从环境变量或常见路径自动查找
            - str/Path: 显式指定路径

        # === 时间参数 (可选) ===
        interval: 时间对齐间隔
            - None: 使用原始时间点（不对齐）
            - '1h', '6h': 字符串格式
            - pd.Timedelta(hours=1): Timedelta对象
        win_length: 滑动窗口长度（用于SOFA等评分）
            - None: 点数据（不使用窗口）
            - '24h': 字符串格式
            - pd.Timedelta(hours=24): Timedelta对象

        # === 聚合参数 (可选) ===
        aggregate: 聚合方式
            - None: 使用默认聚合（通常是'mean'）
            - 'mean', 'max', 'min', 'median': 单一聚合函数
            - {'hr': 'mean', 'sbp': 'max'}: 每个概念指定聚合

        # === SOFA相关 ===
        keep_components: 是否保留SOFA组件列
            - False: 只返回总分
            - True: 返回 sofa + sofa_resp + sofa_coag + ...
        use_sofa2: 是否加载SOFA2字典（自动检测SOFA2概念时启用）

        # === 其他 ===
        merge: 是否合并多个概念到一个DataFrame
        verbose: 是否显示详细信息
        **kwargs: 其他参数传递给底层API

    Returns:
        DataFrame 或 dict of DataFrames

    Examples:
        >>> # 最简单的用法 - 自动检测所有参数
        >>> hr = load_concepts('hr')
        >>>
        >>> # 指定患者ID
        >>> hr = load_concepts('hr', patient_ids=[123, 456, 789])
        >>>
        >>> # 加载多个概念并对齐到1小时间隔
        >>> vitals = load_concepts(['hr', 'sbp', 'temp'],
        ...                        patient_ids=[123, 456],
        ...                        interval='1h')
        >>>
        >>> # SOFA评分 - 24小时窗口，保留组件
        >>> sofa = load_concepts('sofa',
        ...                      patient_ids=[123, 456],
        ...                      interval='6h',
        ...                      win_length='24h',
        ...                      keep_components=True)
        >>>
        >>> # SOFA2评分 (2025标准)
        >>> sofa2 = load_concepts('sofa2',
        ...                       patient_ids=[123, 456],
        ...                       use_sofa2=True)
        >>>
        >>> # 完全自定义
        >>> data = load_concepts('sofa2',
        ...                      patient_ids={'stay_id': [123, 456]},
        ...                      database='miiv',
        ...                      data_path='/custom/path',
        ...                      interval=pd.Timedelta(hours=6),
        ...                      win_length=pd.Timedelta(hours=24),
        ...                      aggregate='max',
        ...                      verbose=True)
    """
    # 自动检测SOFA2需求
    if isinstance(concepts, str):
        concepts_list = [concepts]
    else:
        concepts_list = list(concepts)

    if any('sofa2' in c.lower() for c in concepts_list):
        use_sofa2 = True

    if verbose:
        print(f"📊 使用统一API加载 {len(concepts_list)} 个概念...")
        print(f"   概念: {', '.join(concepts_list)}")

    # 创建或获取全局加载器
    loader = _get_global_loader(
        database=database,
        data_path=data_path,
        dict_path=dict_path,
        use_sofa2=use_sofa2,
        verbose=verbose
    )

    # 规范化患者ID
    if patient_ids is not None and not isinstance(patient_ids, dict):
        database_name = loader.database
        if database_name in ['eicu', 'eicu_demo']:
            patient_ids = {'patientunitstayid': patient_ids}
        elif database_name in ['aumc']:
            patient_ids = {'admissionid': patient_ids}
        elif database_name in ['hirid']:
            patient_ids = {'patientid': patient_ids}
        else:
            patient_ids = {'stay_id': patient_ids}

    # 使用统一加载器加载概念
    return loader.load_concepts(
        concepts=concepts_list,
        patient_ids=patient_ids,
        interval=interval,
        win_length=win_length,
        aggregate=aggregate,
        keep_components=keep_components,
        merge=merge,
        ricu_compatible=ricu_compatible,
        chunk_size=chunk_size,
        progress=progress,
        parallel_workers=parallel_workers,
        concept_workers=concept_workers,
        parallel_backend=parallel_backend,
        **kwargs
    )


# 为了兼容旧代码，保留旧的函数名
def load_concept(*args, **kwargs):
    """load_concepts的别名（向后兼容）"""
    return load_concepts(*args, **kwargs)


def load_sofa(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    keep_components: bool = True,
    verbose: bool = False,
    **kwargs  # 允许传递额外参数如align_to_admission
) -> pd.DataFrame:
    """
    加载SOFA评分（便捷函数）- 重构版本

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        keep_components: 是否保留组件（默认True）
        verbose: 是否显示详细信息
        **kwargs: 额外参数传递给load_concepts（如align_to_admission）

    Returns:
        SOFA评分DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> sofa = load_sofa(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> sofa = load_sofa(patient_ids=[123, 456],
        ...                  database='miiv', data_path='/data/miiv',
        ...                  win_length='12h', interval='6h')
        >>>
        >>> # 使用时间对齐
        >>> sofa = load_sofa(patient_ids=[123, 456],
        ...                  align_to_admission=True)
    """
    if verbose:
        print("🏥 加载SOFA评分...")

    return load_concepts(
        'sofa',
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        keep_components=keep_components,
        verbose=verbose,
        **kwargs  # 传递额外参数
    )


def load_sofa2(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    keep_components: bool = True,
    verbose: bool = False,
    **kwargs  # 允许传递额外参数如align_to_admission
) -> pd.DataFrame:
    """
    加载SOFA-2评分（2025年新标准）- 重构版本

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        keep_components: 是否保留组件（默认True）
        verbose: 是否显示详细信息
        **kwargs: 额外参数传递给load_concepts（如align_to_admission）

    Returns:
        SOFA-2评分DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> sofa2 = load_sofa2(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> sofa2 = load_sofa2(patient_ids=[123, 456],
        ...                   database='miiv', data_path='/data/miiv')
    """
    if verbose:
        print("🏥 加载SOFA-2评分（2025标准）...")

    return load_concepts(
        'sofa2',
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        keep_components=keep_components,
        verbose=verbose,
        use_sofa2=True,  # 强制使用SOFA2字典
        **kwargs  # 传递额外参数
    )


def load_sepsis3(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载Sepsis-3诊断相关数据 - 重构版本

    包含: SOFA, abx, samp, susp_inf, sep3

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        verbose: 是否显示详细信息

    Returns:
        Sepsis-3数据DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> sep3 = load_sepsis3(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> sep3 = load_sepsis3(patient_ids=[123, 456],
        ...                     database='miiv', data_path='/data/miiv')
    """
    if verbose:
        print("🦠 加载Sepsis-3相关数据...")

    # 只加载sep3概念，它已经包含了所有必需的诊断信息
    # 如果需要详细的组件（SOFA, abx等），用户可以分别加载
    return load_concepts(
        'sep3',
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        verbose=verbose
    )


def load_vitals(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载生命体征数据（便捷函数）- 重构版本

    包含: hr, sbp, dbp, temp, resp, spo2

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        verbose: 是否显示详细信息

    Returns:
        生命体征DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> vitals = load_vitals(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> vitals = load_vitals(patient_ids=[123, 456],
        ...                      database='miiv', data_path='/data/miiv',
        ...                      interval='30m')
    """
    vital_concepts = ['hr', 'sbp', 'dbp', 'temp', 'resp', 'spo2']

    if verbose:
        print("❤️  加载生命体征...")

    return load_concepts(
        vital_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        verbose=verbose
    )


def load_labs(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '6h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载实验室检查数据（便捷函数）- 重构版本

    包含: wbc, plt, crea, bili, lact, ph

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认6小时，实验室检查频率较低）
        verbose: 是否显示详细信息

    Returns:
        实验室检查DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> labs = load_labs(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> labs = load_labs(patient_ids=[123, 456],
        ...                   database='miiv', data_path='/data/miiv',
        ...                   interval='12h')
    """
    lab_concepts = ['wbc', 'plt', 'crea', 'bili', 'lac', 'ph']

    if verbose:
        print("🔬 加载实验室检查...")

    return load_concepts(
        lab_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        verbose=verbose
    )


def list_available_concepts(source: Optional[str] = None) -> List[str]:
    """
    列出可用的概念
    
    Args:
        source: 如果指定，只列出该数据源支持的概念
        
    Returns:
        概念名称列表
        
    Examples:
        >>> # 列出所有概念
        >>> all_concepts = list_available_concepts()
        >>> 
        >>> # 列出MIMIC支持的概念
        >>> mimic_concepts = list_available_concepts('mimic')
    """
    dict_obj = load_dictionary()
    
    if source is None:
        # 返回所有概念
        return list(dict_obj.concepts.keys())
    
    # 返回特定数据源支持的概念
    supported = []
    for name, concept in dict_obj.concepts.items():
        if hasattr(concept, 'sources') and source in concept.sources:
            supported.append(name)
    
    return sorted(supported)


def list_available_sources() -> List[str]:
    """
    列出可用的数据源
    
    Returns:
        数据源名称列表
        
    Examples:
        >>> sources = list_available_sources()
        >>> print(sources)
        ['mimic', 'hirid', 'eicu', 'aumc']
    """
    registry = load_data_sources()
    return [cfg.name for cfg in registry]


def get_concept_info(concept_name: str) -> Dict:
    """
    获取概念的详细信息
    
    Args:
        concept_name: 概念名称
        
    Returns:
        包含概念信息的字典
        
    Examples:
        >>> info = get_concept_info('hr')
        >>> print(info['description'])
        'heart rate'
    """
    dict_obj = load_dictionary()
    
    if concept_name not in dict_obj.concepts:
        raise ValueError(f"未知概念: {concept_name}")
    
    concept = dict_obj.concepts[concept_name]
    
    info = {
        'name': concept_name,
        'description': getattr(concept, 'description', ''),
        'category': getattr(concept, 'category', ''),
        'unit': getattr(concept, 'unit', ''),
        'sources': list(getattr(concept, 'sources', {}).keys()),
    }
    
    return info


# === 新增模块函数（参考ricu.R） ===

def _validate_concepts(concepts: List[str], verbose: bool = False) -> List[str]:
    """
    验证概念是否存在于字典中，返回可用的概念列表

    Args:
        concepts: 要验证的概念列表
        verbose: 是否显示详细信息

    Returns:
        可用的概念列表
    """
    try:
        dict_obj = load_dictionary()
        available_concepts = [c for c in concepts if c in dict_obj.concepts]
        missing_concepts = [c for c in concepts if c not in dict_obj.concepts]

        if verbose and missing_concepts:
            print(f"  ⚠️  以下概念在字典中不存在，将被跳过: {missing_concepts}")

        return available_concepts
    except Exception:
        return concepts  # 如果验证失败，返回原列表


def load_demographics(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载基础人口统计学数据（参考ricu.R的data_demo）

    包含: age, bmi, height, sex, weight

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        人口统计学DataFrame

    Examples:
        >>> demo = load_demographics(patient_ids=[123, 456])
    """
    if verbose:
        print("👥 加载基础人口统计学数据...")

    # 🔧 修复：分别加载概念以避免ID列冲突
    try:
        all_data = []

        # 加载age和sex（来自patients表，使用subject_id）
        try:
            age_sex_data = load_concepts(
                concepts=['age', 'sex'],
                patient_ids=patient_ids,
                database=database,
                data_path=data_path,
                merge=True,
                verbose=False
            )
            if age_sex_data is not None and not age_sex_data.empty:
                all_data.append(age_sex_data)
                if verbose:
                    print(f"  ✅ age/sex: {len(age_sex_data)}行")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  age/sex加载失败: {str(e)[:50]}")

        # 加载height和weight（来自chartevents表，使用stay_id）
        try:
            height_weight_data = load_concepts(
                concepts=['height', 'weight'],
                patient_ids=patient_ids,
                database=database,
                data_path=data_path,
                merge=True,
                verbose=False
            )
            if height_weight_data is not None and not height_weight_data.empty:
                all_data.append(height_weight_data)
                if verbose:
                    print(f"  ✅ height/weight: {len(height_weight_data)}行")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  height/weight加载失败: {str(e)[:50]}")

        # 如果没有数据，返回空DataFrame
        if not all_data:
            if verbose:
                print("  ❌ 没有可用的人口统计学数据")
            return pd.DataFrame()

        # 🔧 手动合并数据，处理ID列差异
        merged_data = all_data[0]
        for i, df in enumerate(all_data[1:], 1):
            if df.empty:
                continue

            # 确定共同的ID列
            common_cols = set(merged_data.columns) & set(df.columns)
            id_cols = [col for col in common_cols if 'id' in col.lower() or col in ['stay_id', 'subject_id', 'patientunitstayid']]

            if id_cols:
                id_col = id_cols[0]
                try:
                    merged_data = pd.merge(merged_data, df, on=id_col, how='outer', suffixes=('', f'_{i}'))
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  合并失败: {str(e)[:50]}")
                    # 如果合并失败，使用concat
                    merged_data = pd.concat([merged_data, df], ignore_index=True)
            else:
                # 如果没有共同ID列，使用concat
                merged_data = pd.concat([merged_data, df], ignore_index=True)

        if verbose:
            print(f"  ✅ 最终合并结果: {len(merged_data)}行, {len(merged_data.columns)}列")

        return merged_data

    except Exception as e:
        if verbose:
            print(f"  ❌ 人口统计学数据加载失败: {e}")
        return pd.DataFrame()


def load_outcomes(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    keep_components: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载结局指标数据（参考ricu.R的data_outcome）

    包含: death, los_icu, qsofa, sirs

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        keep_components: 是否保留组件（默认True）
        verbose: 是否显示详细信息

    Returns:
        结局指标DataFrame

    Examples:
        >>> outcomes = load_outcomes(patient_ids=[123, 456])
    """
    if verbose:
        print("📊 加载结局指标数据...")

    concepts = ['death', 'los_icu', 'qsofa', 'sirs']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        keep_components=keep_components,
        merge=True,
        verbose=verbose
    )


def load_vitals_detailed(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载详细生命体征数据（参考ricu.R的data_vital）

    包含: dbp, etco2, hr, map, sbp, temp

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        详细生命体征DataFrame

    Examples:
        >>> vitals = load_vitals_detailed(patient_ids=[123, 456])
    """
    if verbose:
        print("❤️ 加载详细生命体征数据...")

    concepts = ['dbp', 'etco2', 'hr', 'map', 'sbp', 'temp']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )


def load_neurological(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载神经系统评估数据（参考ricu.R的data_neu）

    包含: avpu, egcs, gcs, mgcs, rass, vgcs

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        神经系统评估DataFrame

    Examples:
        >>> neuro = load_neurological(patient_ids=[123, 456])
    """
    if verbose:
        print("🧠 加载神经系统评估数据...")

    concepts = ['avpu', 'egcs', 'gcs', 'mgcs', 'rass', 'vgcs']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )


def load_output(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载输出量数据（参考ricu.R的data_output）

    包含: urine, urine24

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        输出量DataFrame

    Examples:
        >>> output = load_output(patient_ids=[123, 456])
    """
    if verbose:
        print("💧 加载输出量数据...")

    concepts = ['urine', 'urine24']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )


def load_respiratory(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载呼吸系统数据（参考ricu.R的data_resp）

    包含: ett_gcs, mech_vent, o2sat, sao2, pafi, resp, safi, supp_o2, vent_ind

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        呼吸系统DataFrame

    Examples:
        >>> resp = load_respiratory(patient_ids=[123, 456])
    """
    if verbose:
        print("🫁 加载呼吸系统数据...")

    concepts = ['ett_gcs', 'mech_vent', 'o2sat', 'sao2', 'pafi', 'resp', 'safi', 'supp_o2', 'vent_ind']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )


def load_lab_comprehensive(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载全面的实验室检查数据（参考ricu.R的data_lab）

    包含: alb, alp, alt, ast, bicar, bili, bili_dir, bun, ca, ck, ckmb,
          cl, crea, crp, glu, k, mg, na, phos, tnt

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        实验室检查DataFrame

    Examples:
        >>> labs = load_lab_comprehensive(patient_ids=[123, 456])
    """
    if verbose:
        print("🧪 加载全面的实验室检查数据...")

    concepts = ['alb', 'alp', 'alt', 'ast', 'bicar', 'bili', 'bili_dir', 'bun',
               'ca', 'ck', 'ckmb', 'cl', 'crea', 'crp', 'glu', 'k', 'mg', 'na', 'phos', 'tnt']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )


def load_blood_gas(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载血气分析数据（参考ricu.R的data_blood）

    包含: be, cai, fio2, hbco, lact, methb, pco2, ph, po2, tco2

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        血气分析DataFrame

    Examples:
        >>> blood_gas = load_blood_gas(patient_ids=[123, 456])
    """
    if verbose:
        print("🩸 加载血气分析数据...")

    concepts = ['be', 'cai', 'fio2', 'hbco', 'lact', 'methb', 'pco2', 'ph', 'po2', 'tco2']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )


def load_hematology(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载血液学检查数据（参考ricu.R的data_hematology）

    包含: bnd, esr, fgn, hgb, inr_pt, lymph, mch, mchc, mcv, neut, plt, ptt, wbc

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        血液学DataFrame

    Examples:
        >>> hematology = load_hematology(patient_ids=[123, 456])
    """
    if verbose:
        print("🩸 加载血液学检查数据...")

    concepts = ['bnd', 'esr', 'fgn', 'hgb', 'inr_pt', 'lymph', 'mch', 'mchc',
               'mcv', 'neut', 'plt', 'ptt', 'wbc']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )


def load_medications(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载药物治疗数据（参考ricu.R的data_med）

    包含: abx, adh_rate, cort, dex, dobu_dur, dobu_rate, dobu60,
          epi_dur, epi_rate, ins, norepi_dur, norepi_equiv, norepi_rate, vaso_ind

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        药物治疗DataFrame

    Examples:
        >>> meds = load_medications(patient_ids=[123, 456])
    """
    if verbose:
        print("💊 加载药物治疗数据...")

    concepts = ['abx', 'adh_rate', 'cort', 'dex', 'dobu_dur', 'dobu_rate', 'dobu60',
               'epi_dur', 'epi_rate', 'ins', 'norepi_dur', 'norepi_equiv', 'norepi_rate', 'vaso_ind']
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose
    )


# 为了兼容性，也导出原始的类和函数
__all__ = [
    # 主要API
    'load_concepts',      # 主API（智能默认值）
    'load_concept',       # 别名（向后兼容）

    # Easy API（便捷函数）
    'load_sofa',
    'load_sofa2',
    'load_sepsis3',
    'load_vitals',
    'load_labs',

    # 新增模块函数（参考ricu.R）
    'load_demographics',     # 基础人口统计学
    'load_outcomes',         # 结局指标
    'load_vitals_detailed',   # 详细生命体征
    'load_neurological',     # 神经系统评估
    'load_output',           # 输出量
    'load_respiratory',      # 呼吸系统
    'load_lab_comprehensive', # 全面实验室检查
    'load_blood_gas',        # 血气分析
    'load_hematology',       # 血液学检查
    'load_medications',      # 药物治疗

    # 工具函数
    'list_available_concepts',
    'list_available_sources',
    'get_concept_info',
]
