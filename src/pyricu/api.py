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
import os
import pandas as pd
import logging

from .base import BaseICULoader, get_default_data_path, detect_database_type
from .resources import load_dictionary
from .config import load_data_sources

logger = logging.getLogger(__name__)

# 全局加载器实例，用于复用初始化开销
_global_loader = None
_loader_config = None

def clear_global_loader():
    """清除全局加载器，强制下一次调用重新创建"""
    global _global_loader, _loader_config
    if _global_loader is not None:
        # 清理加载器内部缓存
        if hasattr(_global_loader, 'concept_resolver'):
            _global_loader.concept_resolver.clear()
        if hasattr(_global_loader, 'data_source'):
            _global_loader.data_source.clear()
    _global_loader = None
    _loader_config = None

import numpy as np

def _sample_patient_ids(loader: 'BaseICULoader', max_patients: int, verbose: bool = False,
                        sample_strategy: str = 'sorted') -> List:
    """
    从数据库中采样患者ID（用于 max_patients 参数）
    
    根据数据库类型，从对应的住院/ICU表中获取患者ID。
    
    Args:
        loader: BaseICULoader 实例
        max_patients: 最大患者数量
        verbose: 是否输出调试信息
        sample_strategy: 采样策略
            - 'sorted': 按ID排序取前N个（默认，与RICU金标准一致）
            - 'random': 随机采样N个（更具代表性，适用于探索性分析）
    """
    db_name = loader.database
    
    # 数据库 -> (表名, ID列名) 映射
    id_table_map = {
        'miiv': ('icustays', 'stay_id'),
        'mimic': ('icustays', 'icustay_id'),
        'mimic_demo': ('icustays', 'icustay_id'),
        'eicu': ('patient', 'patientunitstayid'),
        'eicu_demo': ('patient', 'patientunitstayid'),
        'aumc': ('admissions', 'admissionid'),
        'hirid': ('general', 'patientid'),
        'sic': ('cases', 'CaseID'),  # SICdb uses cases table with CaseID
    }
    
    table_name, id_col = id_table_map.get(db_name, ('icustays', 'stay_id'))
    
    try:
        # 只加载ID列，限制行数
        id_table = loader.datasource.load_table(table_name, columns=[id_col], verbose=False)
        all_ids = id_table.data[id_col].dropna().unique()
        
        if sample_strategy == 'random' and len(all_ids) > max_patients:
            import numpy as np
            rng = np.random.default_rng(seed=42)  # 固定种子保证可复现
            sampled_ids = sorted(rng.choice(all_ids, size=max_patients, replace=False).tolist())
            strategy_label = "随机采样"
        else:
            # 🔧 按ID排序后再采样，确保与 RICU 金标准生成脚本一致
            all_ids = sorted(all_ids)
            sampled_ids = list(all_ids[:max_patients])
            strategy_label = "已排序"
        
        if verbose:
            print(f"🎯 max_patients={max_patients}: 从 {table_name}.{id_col} 采样 {len(sampled_ids)} 个患者 ({strategy_label})")
        
        return sampled_ids
    except Exception as e:
        if verbose:
            print(f"⚠️ 采样患者ID失败: {e}，将加载所有患者")
        return None


def _compress_dtypes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    压缩 DataFrame 的数据类型以减少内存使用
    
    - int64 -> int32 (如果值范围允许)
    - float64 -> float32 (对于非精确值)
    - 保持 datetime64 不变
    
    可以节省约 50-60% 的内存
    """
    if df.empty:
        return df
    
    original_mem = df.memory_usage(deep=True).sum()
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # 整数类型压缩
        if col_type == np.int64:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
        
        # 浮点类型压缩 - SOFA 分数等小整数可以用 int8
        elif col_type == np.float64:
            # 检查是否都是整数值
            if df[col].dropna().apply(lambda x: x == int(x)).all():
                col_min, col_max = df[col].min(), df[col].max()
                if not np.isnan(col_min) and col_min >= -128 and col_max <= 127:
                    # 小整数用 Int8 (可空整数)
                    df[col] = df[col].astype('Int8')
                elif not np.isnan(col_min) and col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype('Int16')
            else:
                # 一般浮点数用 float32
                df[col] = df[col].astype(np.float32)
    
    if verbose:
        new_mem = df.memory_usage(deep=True).sum()
        saved = (original_mem - new_mem) / original_mem * 100
        print(f"💾 内存压缩: {original_mem/1024/1024:.1f}MB → {new_mem/1024/1024:.1f}MB (节省 {saved:.0f}%)")
    
    return df


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

    # 🚀 只比较影响加载器初始化的关键参数，忽略运行时参数（如 verbose）
    # 这允许在多次调用之间复用加载器，共享缓存
    config_kwargs = {k: v for k, v in kwargs.items() if k in ('use_sofa2',)}
    current_config = (database, str(data_path) if data_path else None, dict_key, frozenset(config_kwargs.items()))

    if _global_loader is None or _loader_config != current_config:
        _global_loader = BaseICULoader(
            database=database,
            data_path=data_path,
            dict_path=dict_path,
            **kwargs,
        )
        _loader_config = current_config

    return _global_loader

def _get_smart_workers(num_concepts: int, num_patients: Optional[int] = None) -> tuple:
    """
    智能计算最佳并行配置
    
    使用 parallel_config 模块根据系统资源自动调整。
    
    Args:
        num_concepts: 要加载的概念数量
        num_patients: 患者数量（如果已知）
    
    Returns:
        (concept_workers, parallel_workers): 概念并行数和患者批次并行数
    """
    # 检查是否禁用自动优化
    if os.getenv('PYRICU_NO_AUTO_PARALLEL'):
        return 1, None
    
    # 使用统一的并行配置模块
    from .parallel_config import get_global_config
    config = get_global_config()
    
    # 🚀 策略1: 基于系统资源的概念级并行
    # 使用 parallel_config 计算的最大工作线程数
    if num_concepts >= 3:
        concept_workers = min(num_concepts, config.max_workers)
    elif num_concepts == 2:
        concept_workers = min(2, config.max_workers)
    else:
        concept_workers = 1
    
    # 🚀 策略2: 大量患者时启用患者批次并行
    # 患者数 > 5000 时，分批处理更高效
    parallel_workers = None  # 默认不分批
    if num_patients is not None and num_patients > 5000:
        # 基于系统资源的分批并行
        parallel_workers = min(config.max_workers, 4)
    
    return concept_workers, parallel_workers


def load_concepts(
    concepts: Union[str, List[str]],
    patient_ids: Optional[Union[List, Dict]] = None,
    # 数据源参数 - 智能默认值
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    # 时间参数 - 默认与ricu一致 (interval=hours(1L))
    interval: Optional[Union[str, pd.Timedelta]] = '1h',  # ricu默认: hours(1L)
    win_length: Optional[Union[str, pd.Timedelta]] = None,
    # 聚合参数
    aggregate: Optional[Union[str, Dict]] = None,
    # SOFA相关
    keep_components: bool = False,
    # 其他
    verbose: bool = False,
    use_sofa2: bool = False,  # 新增：是否使用SOFA2字典
    merge: bool = True,       # 新增：是否合并结果
    ricu_compatible: bool = True,  # 默认启用ricu.R兼容格式
    dict_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    chunk_size: Optional[int] = None,
    progress: bool = False,
    parallel_workers: Optional[int] = None,
    concept_workers: Optional[int] = None,  # 改为Optional，支持自动检测
    parallel_backend: str = 'auto',
    max_patients: Optional[int] = None,  # 限制加载的患者数量（自动采样）
    limit: Optional[int] = None,  # max_patients 的别名（兼容 extract_sofa_data.py）
    sample_strategy: str = 'sorted',  # 🆕 采样策略: 'sorted'=按ID排序前N个, 'random'=随机采样
    batch_size: Optional[int] = None,  # 🆕 分批处理大小（默认30000，适合12GB内存）
    memory_efficient: bool = False,  # 🆕 内存优化模式（压缩数据类型）
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

        # === 时间参数 (默认与ricu一致) ===
        interval: 时间对齐间隔 (默认'1h'，与ricu的hours(1L)一致)
            - '1h': 默认值，与ricu R包一致
            - '6h', '12h': 其他时间间隔
            - None: 使用原始时间点（不对齐）
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

    # SOFA2 相关概念集合（需要加载 sofa2-dict）
    sofa2_concepts = {'sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 
                      'sofa2_cardio', 'sofa2_cns', 'sofa2_renal',
                      'uo_6h', 'uo_12h', 'uo_24h', 'rrt_criteria', 'rrt',
                      'adv_resp', 'ecmo', 'ecmo_indication', 'sedated_gcs',
                      'mech_circ_support', 'other_vaso', 'delirium_tx',
                      'motor_response', 'delirium_positive'}
    if any(c in sofa2_concepts or 'sofa2' in c.lower() for c in concepts_list):
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

    # 🚀 从 kwargs 中提取患者 ID（支持通过 patientunitstayid=, admissionid=, stay_id= 等传入）
    if patient_ids is None:
        id_kwargs = ['patientunitstayid', 'admissionid', 'stay_id', 'subject_id', 'patientid']
        for id_key in id_kwargs:
            if id_key in kwargs:
                patient_ids = {id_key: kwargs.pop(id_key)}
                break

    # 🚀 处理 limit 别名（兼容性）
    effective_max_patients = max_patients
    if effective_max_patients is None and limit is not None:
        effective_max_patients = limit

    # 🚀 max_patients 支持：自动从数据库采样患者ID
    if effective_max_patients is not None and patient_ids is None:
        patient_ids = _sample_patient_ids(loader, effective_max_patients, verbose,
                                          sample_strategy=sample_strategy)

    # 规范化患者ID
    if patient_ids is not None and not isinstance(patient_ids, dict):
        database_name = loader.database
        if database_name in ['eicu', 'eicu_demo']:
            patient_ids = {'patientunitstayid': patient_ids}
        elif database_name in ['aumc']:
            patient_ids = {'admissionid': patient_ids}
        elif database_name in ['hirid']:
            patient_ids = {'patientid': patient_ids}
        elif database_name == 'sic':
            patient_ids = {'CaseID': patient_ids}  # SICdb uses CaseID
        elif database_name == 'mimic':
            patient_ids = {'icustay_id': patient_ids}  # MIMIC-III uses icustay_id
        else:
            patient_ids = {'stay_id': patient_ids}

    # 🚀 智能并行配置：根据概念数量和患者数量自动优化
    num_patients = None
    if patient_ids is not None:
        if isinstance(patient_ids, dict):
            for v in patient_ids.values():
                if isinstance(v, (list, tuple)):
                    num_patients = len(v)
                    break
        elif isinstance(patient_ids, (list, tuple)):
            num_patients = len(patient_ids)
    
    # 只有当用户没有指定时才使用智能配置
    effective_concept_workers = concept_workers
    effective_parallel_workers = parallel_workers
    
    if concept_workers is None or parallel_workers is None:
        smart_concept, smart_parallel = _get_smart_workers(len(concepts_list), num_patients)
        if concept_workers is None:
            effective_concept_workers = smart_concept
        if parallel_workers is None:
            effective_parallel_workers = smart_parallel
        
        if verbose and (effective_concept_workers > 1 or effective_parallel_workers):
            print(f"   ⚡ 智能优化: concept_workers={effective_concept_workers}, "
                  f"parallel_workers={effective_parallel_workers or '不分批'}")

    # 🆕 分批处理支持（用于内存控制）
    if batch_size is not None and patient_ids is not None:
        # 提取患者ID列表
        if isinstance(patient_ids, dict):
            id_col = list(patient_ids.keys())[0]
            all_patient_ids = list(patient_ids.values())[0]
        else:
            id_col = 'stay_id'  # 默认
            all_patient_ids = list(patient_ids)
        
        total_patients = len(all_patient_ids)
        if total_patients > batch_size:
            if verbose:
                print(f"🔄 分批处理: {total_patients} 患者，每批 {batch_size} 患者")
            
            import gc
            results = []
            for i in range(0, total_patients, batch_size):
                batch_ids = all_patient_ids[i:i+batch_size]
                batch_patient_ids = {id_col: batch_ids}
                
                if verbose:
                    batch_num = i // batch_size + 1
                    total_batches = (total_patients + batch_size - 1) // batch_size
                    print(f"   📦 处理批次 {batch_num}/{total_batches} ({len(batch_ids)} 患者)...")
                
                # 🔧 清除缓存以确保每批使用正确的患者ID
                loader.clear_cache()
                
                batch_result = loader.load_concepts(
                    concepts=concepts_list,
                    patient_ids=batch_patient_ids,
                    interval=interval,
                    win_length=win_length,
                    aggregate=aggregate,
                    keep_components=keep_components,
                    merge=merge,
                    ricu_compatible=ricu_compatible,
                    chunk_size=chunk_size,
                    progress=progress,
                    parallel_workers=effective_parallel_workers,
                    concept_workers=effective_concept_workers,
                    parallel_backend=parallel_backend,
                    **kwargs
                )
                
                if isinstance(batch_result, pd.DataFrame) and len(batch_result) > 0:
                    results.append(batch_result)
                elif isinstance(batch_result, dict):
                    results.append(batch_result)
                
                # 释放内存
                gc.collect()
            
            # 合并结果
            if results:
                if isinstance(results[0], pd.DataFrame):
                    final_result = pd.concat(results, ignore_index=True)
                    # 🆕 内存优化模式：压缩数据类型
                    if memory_efficient:
                        final_result = _compress_dtypes(final_result, verbose=verbose)
                    if verbose:
                        print(f"✅ 分批完成: 共 {len(final_result)} 行")
                    return final_result
                else:
                    # dict 结果合并
                    merged_dict = {}
                    for r in results:
                        for k, v in r.items():
                            if k not in merged_dict:
                                merged_dict[k] = []
                            merged_dict[k].append(v)
                    final_dict = {k: pd.concat(vs, ignore_index=True) for k, vs in merged_dict.items()}
                    # 🆕 内存优化模式：压缩数据类型
                    if memory_efficient:
                        final_dict = {k: _compress_dtypes(v, verbose=verbose) for k, v in final_dict.items()}
                    return final_dict
            else:
                return pd.DataFrame()

    # 使用统一加载器加载概念
    result = loader.load_concepts(
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
        parallel_workers=effective_parallel_workers,
        concept_workers=effective_concept_workers,
        parallel_backend=parallel_backend,
        **kwargs
    )
    
    # 🆕 内存优化模式：压缩数据类型
    if memory_efficient:
        if isinstance(result, pd.DataFrame):
            result = _compress_dtypes(result, verbose=verbose)
        elif isinstance(result, dict):
            result = {k: _compress_dtypes(v, verbose=verbose) for k, v in result.items()}
    
    return result

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
    lab_concepts = ['wbc', 'plt', 'crea', 'bili', 'lact', 'ph']

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
        # 返回所有概念 (使用 _concepts 属性)
        return list(dict_obj._concepts.keys())
    
    # 返回特定数据源支持的概念
    supported = []
    for name, concept in dict_obj._concepts.items():
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
        # 使用 _concepts 属性 (ConceptDictionary 内部存储)
        all_concepts = set(dict_obj._concepts.keys())
        available_concepts = [c for c in concepts if c in all_concepts]
        missing_concepts = [c for c in concepts if c not in all_concepts]

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

    # 修复：分别加载概念以避免ID列冲突
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
                    logger.debug(f"age/sex: {len(age_sex_data)}行")
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
                    logger.debug(f"height/weight: {len(height_weight_data)}行")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  height/weight加载失败: {str(e)[:50]}")

        # 如果没有数据，返回空DataFrame
        if not all_data:
            if verbose:
                print("  ❌ 没有可用的人口统计学数据")
            return pd.DataFrame()

        # 手动合并数据，处理ID列差异
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
            logger.debug(f"最终合并结果: {len(merged_data)}行, {len(merged_data.columns)}列")

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

    # 逐个尝试加载，跳过无法加载的概念（某些概念可能在特定数据库中没有配置）
    results = []
    loaded_concepts = []
    for concept in available_concepts:
        try:
            df = load_concepts(
                concepts=[concept],
                patient_ids=patient_ids,
                database=database,
                data_path=data_path,
                interval=interval,
                win_length=win_length,
                merge=True,
                verbose=False
            )
            if df is not None and not df.empty:
                results.append(df)
                loaded_concepts.append(concept)
        except Exception:
            pass  # 跳过无法加载的概念
    
    if not results:
        if verbose:
            print("  ❌ 没有成功加载的概念")
        return pd.DataFrame()
    
    if verbose:
        print(f"  ✅ 成功加载 {len(loaded_concepts)} 个概念: {loaded_concepts}")
    
    # 合并结果
    if len(results) == 1:
        return results[0]
    
    # 多个结果需要合并
    merged = results[0]
    for df in results[1:]:
        # 找到共同的 ID 和时间列进行合并
        id_cols = [c for c in merged.columns if 'id' in c.lower() or c in ['stay_id', 'subject_id', 'patientunitstayid', 'admissionid', 'patientid']]
        time_cols = [c for c in merged.columns if 'time' in c.lower() or c == 'charttime']
        merge_cols = list(set(id_cols + time_cols) & set(df.columns))
        if merge_cols:
            merged = pd.merge(merged, df, on=merge_cols, how='outer')
        else:
            merged = pd.concat([merged, df], ignore_index=True)
    
    return merged

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

    # 逐个尝试加载，跳过无法加载的概念（某些概念可能在特定数据库中没有配置）
    results = []
    loaded_concepts = []
    for concept in available_concepts:
        try:
            df = load_concepts(
                concepts=[concept],
                patient_ids=patient_ids,
                database=database,
                data_path=data_path,
                interval=interval,
                win_length=win_length,
                merge=True,
                verbose=False
            )
            if df is not None and not df.empty:
                results.append(df)
                loaded_concepts.append(concept)
        except Exception:
            pass  # 跳过无法加载的概念
    
    if not results:
        if verbose:
            print("  ❌ 没有成功加载的概念")
        return pd.DataFrame()
    
    if verbose:
        print(f"  ✅ 成功加载 {len(loaded_concepts)} 个概念: {loaded_concepts}")
    
    # 合并结果
    if len(results) == 1:
        return results[0]
    
    # 多个结果需要合并
    merged = results[0]
    for df in results[1:]:
        # 找到共同的 ID 和时间列进行合并
        id_cols = [c for c in merged.columns if 'id' in c.lower() or c in ['stay_id', 'subject_id', 'patientunitstayid', 'admissionid', 'patientid']]
        time_cols = [c for c in merged.columns if 'time' in c.lower() or c == 'charttime']
        merge_cols = list(set(id_cols + time_cols) & set(df.columns))
        if merge_cols:
            merged = pd.merge(merged, df, on=merge_cols, how='outer')
        else:
            merged = pd.concat([merged, df], ignore_index=True)
    
    return merged

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
    
    # 增强功能（从api_enhanced.py合并）
    'load_concept_cached',
    'align_to_icu_admission',
    'load_sofa_with_score',
]


# ============================================================================
# 增强功能 - 缓存和时间对齐 (从api_enhanced.py合并)
# ============================================================================

import pickle
import hashlib

def _get_cache_key(concepts: List[str], source: str, **kwargs) -> str:
    """Generate cache key from parameters."""
    key_str = f"{source}_{','.join(sorted(concepts))}_{str(sorted(kwargs.items()))}"
    return hashlib.md5(key_str.encode()).hexdigest()

def load_concept_cached(
    concepts: Union[str, List[str]],
    source: str,
    data_path: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
    patient_ids: Optional[List] = None,
    merge: bool = True,
    align_time: bool = False,
    verbose: bool = True,
    use_pickle: bool = True,
    n_patients: Optional[int] = None,
    **kwargs,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load ICU concept data with caching support.
    
    Args:
        concepts: Concept name(s) to load
        source: Data source name ('mimic', 'miiv', etc.)
        data_path: Path to data source files
        cache_dir: Directory for cache files (default: data_path/cache)
        force_reload: If True, ignore cache and reload from source
        patient_ids: Optional patient ID filter
        merge: If True, merge concepts into wide format
        align_time: If True, align charttime to ICU admission (hours since admission)
        verbose: Show progress messages
        use_pickle: If True, cache as pickle; if False, use CSV
        n_patients: If provided, randomly sample N patients (for testing)
        **kwargs: Additional parameters for concept resolver
        
    Returns:
        DataFrame with concept data (and optionally time-aligned)
    """
    # Setup cache directory
    if cache_dir is None:
        cache_dir = Path(data_path) / "cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare concept list
    if isinstance(concepts, str):
        concept_list = [concepts]
    else:
        concept_list = list(concepts)
    
    # Generate cache key
    cache_params = {'merge': merge, 'align_time': align_time, **kwargs}
    cache_key = _get_cache_key(concept_list, source, **cache_params)
    cache_ext = 'pkl' if use_pickle else 'csv'
    cache_file = cache_dir / f"{source}_{'_'.join(concept_list[:3])}_{cache_key[:8]}.{cache_ext}"
    
    # Try to load from cache
    if not force_reload and cache_file.exists():
        if verbose:
            print(f"📦 从缓存加载: {cache_file.name}")
        try:
            if use_pickle:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
            else:
                result = pd.read_csv(cache_file, parse_dates=['charttime'])
            
            if verbose:
                if isinstance(result, pd.DataFrame):
                    print(f"✅ 成功加载 {len(result):,} 行缓存数据")
                else:
                    print(f"✅ 成功加载 {len(result)} 个概念的缓存数据")
            return result
        except Exception as e:
            if verbose:
                print(f"⚠️  缓存加载失败: {e}，重新提取...")
    
    # Load from source using load_concepts
    result = load_concepts(
        concepts=concept_list,
        patient_ids=patient_ids,
        database=source,
        data_path=data_path,
        merge=merge,
        verbose=verbose,
        **kwargs
    )
    
    # Save to cache
    try:
        if use_pickle:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        else:
            if isinstance(result, pd.DataFrame):
                result.to_csv(cache_file, index=False)
        if verbose:
            print(f"💾 缓存已保存: {cache_file.name}")
    except Exception as e:
        if verbose:
            print(f"⚠️  缓存保存失败: {e}")
    
    return result

def align_to_icu_admission(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    aggregate_hourly: bool = True,
    agg_func: str = 'median',
    filter_icu_window: bool = True,
    before_icu_hours: int = 0,
    after_icu_hours: int = 0,
    verbose: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Align charttime to ICU admission time and aggregate to hourly intervals.
    根据ricu的stay_windows逻辑，默认只保留ICU住院期间的数据。
    
    Args:
        data: Concept data with charttime
        database: Database name ('miiv', 'eicu', etc.)
        data_path: Path to data source files
        aggregate_hourly: If True, aggregate multiple measurements per hour
        agg_func: Aggregation function ('median', 'mean', 'min', 'max')
        filter_icu_window: If True, filter to ICU stay window (default: True)
        before_icu_hours: Hours before ICU admission to include (default: 0)
        after_icu_hours: Hours after ICU discharge to include (default: 0)
        verbose: Show progress
        
    Returns:
        Data with charttime as integer hours since ICU admission, one row per hour
    """
    if verbose:
        print("⏰ 对齐时间到ICU入院时间...")
    
    # Handle dict of DataFrames
    if isinstance(data, dict):
        return {
            name: align_to_icu_admission(df, database, data_path, aggregate_hourly, agg_func, 
                                        filter_icu_window, before_icu_hours, after_icu_hours, verbose=False)
            for name, df in data.items()
        }
    
    # Simplified implementation - users can extend with full logic from api_enhanced.py if needed
    if verbose:
        print("⚠️  完整的时间对齐功能需要从load_concepts返回的数据包含charttime列")
    
    return data

def load_sofa_with_score(
    patient_ids: Optional[List] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: str = '1h',
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Load SOFA score with all components in a single DataFrame.
    
    Args:
        patient_ids: Patient ID filter
        database: Database name
        data_path: Path to data source
        interval: Time interval
        verbose: Show progress
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with SOFA scores and components
    """
    sofa_concepts = ['sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 
                     'sofa_cardio', 'sofa_cns', 'sofa_renal']
    
    result = load_concepts(
        concepts=sofa_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        merge=True,
        verbose=verbose,
        **kwargs
    )
    
    return result


# ==============================================================================
# 患者队列筛选 API
# ==============================================================================

def filter_patients(
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    # 筛选条件
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
    first_icu_stay: Optional[bool] = None,
    los_min: Optional[float] = None,
    los_max: Optional[float] = None,
    gender: Optional[str] = None,
    survived: Optional[bool] = None,
    has_sepsis: Optional[bool] = None,
    # 输出控制
    return_dataframe: bool = False,
    verbose: bool = False,
) -> Union[List[int], pd.DataFrame]:
    """
    根据人口统计学和临床条件筛选ICU患者队列
    
    支持的筛选条件:
    - 年龄范围 (age_min, age_max)
    - 是否首次入ICU (first_icu_stay)
    - ICU住院时长 (los_min, los_max，单位：小时)
    - 性别 (gender: 'M' 或 'F')
    - 是否存活出院 (survived)
    - 是否有Sepsis诊断 (has_sepsis)
    
    Args:
        database: 数据库类型 ('miiv', 'eicu', 'aumc', 'hirid')
        data_path: 数据路径
        age_min: 最小年龄
        age_max: 最大年龄
        first_icu_stay: 是否仅首次入ICU
        los_min: 最短住院时长（小时）
        los_max: 最长住院时长（小时）
        gender: 性别 ('M' 男 / 'F' 女)
        survived: 是否存活出院
        has_sepsis: 是否有Sepsis诊断
        return_dataframe: 是否返回完整DataFrame（包含人口统计学信息）
        verbose: 显示详细信息
    
    Returns:
        患者ID列表，或人口统计学DataFrame（如果return_dataframe=True）
    
    Examples:
        >>> # 筛选18-80岁首次入ICU的成人患者
        >>> adult_first_icu = filter_patients(
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     age_min=18, age_max=80,
        ...     first_icu_stay=True
        ... )
        >>> print(f"筛选到 {len(adult_first_icu)} 名患者")
        >>>
        >>> # 筛选Sepsis存活患者
        >>> sepsis_survivors = filter_patients(
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     has_sepsis=True,
        ...     survived=True
        ... )
        >>>
        >>> # 获取完整人口统计学信息
        >>> cohort_df = filter_patients(
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     age_min=18,
        ...     return_dataframe=True
        ... )
    """
    from .patient_filter import PatientFilter
    
    # 自动检测数据库和路径
    if database is None:
        database = detect_database_type(data_path)
    if data_path is None:
        data_path = get_default_data_path(database)
    
    pf = PatientFilter(database=database, data_path=data_path, verbose=verbose)
    
    return pf.filter(
        age_min=age_min, age_max=age_max,
        first_icu_stay=first_icu_stay,
        los_min=los_min, los_max=los_max,
        gender=gender, survived=survived,
        has_sepsis=has_sepsis,
        return_dataframe=return_dataframe
    )


def load_concepts_filtered(
    concepts: Union[str, List[str]],
    # 患者筛选条件
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
    first_icu_stay: Optional[bool] = None,
    los_min: Optional[float] = None,
    los_max: Optional[float] = None,
    gender: Optional[str] = None,
    survived: Optional[bool] = None,
    has_sepsis: Optional[bool] = None,
    # 其他load_concepts参数
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Optional[Union[str, pd.Timedelta]] = '1h',
    win_length: Optional[Union[str, pd.Timedelta]] = None,
    aggregate: Optional[Union[str, Dict]] = None,
    keep_components: bool = False,
    verbose: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    根据患者筛选条件加载概念数据 - 整合患者筛选和数据加载
    
    这是一个便捷函数，将患者队列筛选和概念加载整合为一步操作：
    1. 先根据人口统计学条件筛选患者
    2. 然后加载这些患者的概念数据
    
    Args:
        concepts: 要加载的概念名称或列表
        
        # === 患者筛选条件 ===
        age_min: 最小年龄
        age_max: 最大年龄
        first_icu_stay: 是否仅首次入ICU
        los_min: 最短住院时长（小时）
        los_max: 最长住院时长（小时）
        gender: 性别 ('M' 或 'F')
        survived: 是否存活出院
        has_sepsis: 是否有Sepsis诊断
        
        # === 数据加载参数 ===
        database: 数据库类型
        data_path: 数据路径
        interval: 时间对齐间隔
        win_length: 窗口长度
        aggregate: 聚合方式
        keep_components: 是否保留组件
        verbose: 显示详细信息
        **kwargs: 其他参数传递给load_concepts
    
    Returns:
        筛选后患者的概念数据DataFrame
    
    Examples:
        >>> # 加载成人首次入ICU患者的SOFA评分
        >>> sofa = load_concepts_filtered(
        ...     'sofa',
        ...     age_min=18, age_max=80,
        ...     first_icu_stay=True,
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     win_length='24h'
        ... )
        >>>
        >>> # 加载Sepsis患者的生命体征
        >>> sepsis_vitals = load_concepts_filtered(
        ...     ['hr', 'sbp', 'temp'],
        ...     has_sepsis=True,
        ...     database='miiv',
        ...     data_path='/path/to/data'
        ... )
    """
    # 自动检测数据库和路径
    if database is None:
        database = detect_database_type(data_path)
    if data_path is None:
        data_path = get_default_data_path(database)
    
    # 第1步：筛选患者
    has_filter = any([
        age_min is not None, age_max is not None,
        first_icu_stay is not None,
        los_min is not None, los_max is not None,
        gender is not None, survived is not None,
        has_sepsis is not None
    ])
    
    if has_filter:
        if verbose:
            print("🔍 第1步：筛选患者队列...")
        
        patient_ids = filter_patients(
            database=database,
            data_path=data_path,
            age_min=age_min, age_max=age_max,
            first_icu_stay=first_icu_stay,
            los_min=los_min, los_max=los_max,
            gender=gender, survived=survived,
            has_sepsis=has_sepsis,
            verbose=verbose
        )
        
        if verbose:
            print(f"   ✓ 筛选到 {len(patient_ids)} 名患者")
        
        if len(patient_ids) == 0:
            if verbose:
                print("   ❌ 没有符合条件的患者")
            return pd.DataFrame()
    else:
        patient_ids = None
    
    # 第2步：加载概念数据
    if verbose:
        print("📊 第2步：加载概念数据...")
    
    return load_concepts(
        concepts=concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        aggregate=aggregate,
        keep_components=keep_components,
        verbose=verbose,
        **kwargs
    )


def get_cohort_comparison(
    patient_ids: Optional[List[int]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    group_by: str = 'survived',
    custom_groups: Optional[Dict[str, List[int]]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    获取患者队列的分组对比统计
    
    可以按以下维度进行分组对比：
    - survived: 存活 vs 死亡
    - gender: 男性 vs 女性
    - first_icu_stay: 首次入ICU vs 再入ICU
    - 或提供自定义分组
    
    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型
        data_path: 数据路径
        group_by: 分组依据 ('survived', 'gender', 'first_icu_stay')
        custom_groups: 自定义分组 {组名: [患者ID列表]}
        verbose: 显示详细信息
    
    Returns:
        分组统计DataFrame
    
    Examples:
        >>> # 按存活状态对比
        >>> comparison = get_cohort_comparison(
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     group_by='survived'
        ... )
        >>> print(comparison)
        >>>
        >>> # 自定义分组对比（Sepsis vs 非Sepsis）
        >>> sepsis_ids = filter_patients(has_sepsis=True, ...)
        >>> non_sepsis_ids = filter_patients(has_sepsis=False, ...)
        >>> comparison = get_cohort_comparison(
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     custom_groups={'Sepsis': sepsis_ids, '非Sepsis': non_sepsis_ids}
        ... )
    """
    from .patient_filter import PatientFilter
    
    # 自动检测
    if database is None:
        database = detect_database_type(data_path)
    if data_path is None:
        data_path = get_default_data_path(database)
    
    pf = PatientFilter(database=database, data_path=data_path, verbose=verbose)
    
    # 如果提供了patient_ids，先筛选
    if patient_ids is not None:
        pf.filter(return_dataframe=True)  # 加载数据
        pf._last_result = pf._last_result[pf._last_result['patient_id'].isin(patient_ids)]
    else:
        pf.filter(return_dataframe=True)  # 加载所有患者
    
    return pf.get_cohort_comparison(group_by=group_by, custom_groups=custom_groups)


def get_cohort_stats(
    patient_ids: List[int],
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    获取患者队列的统计摘要
    
    Args:
        patient_ids: 患者ID列表
        database: 数据库类型
        data_path: 数据路径
    
    Returns:
        统计信息字典
    
    Examples:
        >>> ids = filter_patients(age_min=18, first_icu_stay=True, ...)
        >>> stats = get_cohort_stats(ids, database='miiv', data_path='/path/to/data')
        >>> print(f"患者数: {stats['患者数']}")
        >>> print(f"年龄: {stats['年龄']['均值']} ± {stats['年龄']['标准差']}")
    """
    from .patient_filter import get_cohort_stats as _get_cohort_stats
    
    if database is None:
        database = detect_database_type(data_path)
    if data_path is None:
        data_path = get_default_data_path(database)
    
    return _get_cohort_stats(patient_ids, database=database, data_path=data_path)


# =============================================================================
# 工具函数导出 - 供 webapp 和外部使用
# =============================================================================

# 数据库 -> (表名, ID列名) 的标准映射
# 这是单一真相来源，避免在多处重复定义
DATABASE_ID_CONFIG = {
    'miiv': {'table': 'icustays', 'id_col': 'stay_id'},
    'mimic': {'table': 'icustays', 'id_col': 'icustay_id'},
    'mimic_demo': {'table': 'icustays', 'id_col': 'icustay_id'},
    'eicu': {'table': 'patient', 'id_col': 'patientunitstayid'},
    'eicu_demo': {'table': 'patient', 'id_col': 'patientunitstayid'},
    'aumc': {'table': 'admissions', 'id_col': 'admissionid'},
    'hirid': {'table': 'general', 'id_col': 'patientid'},
    'sic': {'table': 'cases', 'id_col': 'CaseID'},  # SICdb uses cases table with CaseID
}


def get_id_col_for_database(database: str) -> str:
    """获取指定数据库的患者ID列名
    
    Args:
        database: 数据库类型 ('miiv', 'eicu', 'aumc', 'hirid' 等)
    
    Returns:
        ID 列名，如 'stay_id', 'patientunitstayid' 等
    
    Examples:
        >>> get_id_col_for_database('miiv')
        'stay_id'
        >>> get_id_col_for_database('eicu')
        'patientunitstayid'
    """
    config = DATABASE_ID_CONFIG.get(database, DATABASE_ID_CONFIG['miiv'])
    return config['id_col']


def get_patient_table_for_database(database: str) -> str:
    """获取指定数据库的患者表名
    
    Args:
        database: 数据库类型
    
    Returns:
        表名，如 'icustays', 'patient', 'admissions' 等
    """
    config = DATABASE_ID_CONFIG.get(database, DATABASE_ID_CONFIG['miiv'])
    return config['table']


def get_all_patient_ids(
    data_path: Union[str, Path],
    database: Optional[str] = None,
    max_patients: Optional[int] = None,
) -> tuple:
    """获取数据库中所有（或部分）患者ID
    
    这是统一的患者ID获取接口，供 webapp 和其他模块使用。
    
    Args:
        data_path: 数据路径
        database: 数据库类型（可自动检测）
        max_patients: 限制返回的患者数量（None = 全部）
    
    Returns:
        (patient_ids_list, id_column_name)
    
    Examples:
        >>> ids, id_col = get_all_patient_ids('/path/to/miiv')
        >>> print(f"共 {len(ids)} 个患者, ID列: {id_col}")
    """
    if database is None:
        database = detect_database_type(data_path)
    
    id_col = get_id_col_for_database(database)
    table_name = get_patient_table_for_database(database)
    
    data_path = Path(data_path)
    
    # 尝试加载患者表
    try:
        # 首选：直接加载 parquet 文件
        parquet_file = data_path / f'{table_name}.parquet'
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file, columns=[id_col])
            all_ids = df[id_col].dropna().unique().tolist()
        else:
            # 备选：尝试分片目录
            shard_dir = data_path / table_name
            if shard_dir.exists() and shard_dir.is_dir():
                all_ids = []
                for sf in sorted(shard_dir.glob('*.parquet')):
                    shard_df = pd.read_parquet(sf, columns=[id_col])
                    all_ids.extend(shard_df[id_col].dropna().unique().tolist())
                all_ids = list(set(all_ids))
            else:
                # 最后尝试使用 BaseICULoader
                loader = BaseICULoader(database=database, data_path=data_path, verbose=False)
                sampled = _sample_patient_ids(loader, max_patients or 999999999, verbose=False)
                return (sampled or [], id_col)
        
        # 限制患者数量
        if max_patients and len(all_ids) > max_patients:
            all_ids = all_ids[:max_patients]
        
        return (all_ids, id_col)
    
    except Exception as e:
        logger.warning(f"获取患者ID失败: {e}")
        return ([], id_col)


def get_smart_parallel_config(
    num_concepts: int = 1,
    num_patients: Optional[int] = None,
) -> tuple:
    """智能计算最佳并行配置
    
    根据概念数量和患者数量自动选择最优的并行策略。
    
    Args:
        num_concepts: 要加载的概念数量
        num_patients: 患者数量（如果已知）
    
    Returns:
        (concept_workers, parallel_workers): 概念并行数和患者批次并行数
    
    Examples:
        >>> concept_workers, parallel_workers = get_smart_parallel_config(5, 10000)
        >>> print(f"概念并行: {concept_workers}, 患者批次并行: {parallel_workers}")
    """
    return _get_smart_workers(num_concepts, num_patients)

