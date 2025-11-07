"""
pyricu 极简 API - 一行代码完成数据提取

这个模块提供了最简单的接口，无需了解内部细节。

Examples:
    >>> from pyricu.easy import load_vitals, load_labs, load_sofa_score, load_sepsis
    >>> 
    >>> # 加载生命体征（自动处理所有细节）
    >>> vitals = load_vitals(
    ...     data_path='/path/to/mimic',
    ...     patient_ids=[10001, 10002, 10003]
    ... )
    >>> print(vitals.columns)
    >>> # ['stay_id', 'charttime', 'hr', 'sbp', 'dbp', 'mbp', 'resp', 'temp', 'spo2']
    >>> 
    >>> # 加载 SOFA 评分
    >>> sofa = load_sofa_score('/path/to/mimic', patient_ids=[10001, 10002])
    >>> 
    >>> # 加载 Sepsis-3 诊断
    >>> sepsis = load_sepsis('/path/to/mimic', patient_ids=[10001, 10002])
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd

from .quickstart import ICUQuickLoader


def load_vitals(
    data_path: Union[str, Path],
    patient_ids: Optional[List[int]] = None,
    database: str = 'miiv',
    interval_hours: float = 1.0,
    concepts: Optional[List[str]] = None
) -> pd.DataFrame:
    """加载生命体征数据（极简接口）
    
    自动处理所有细节，返回宽格式表格，每行一个时间点。
    
    Args:
        data_path: ICU 数据路径
        patient_ids: 患者 ID 列表（可选，默认加载所有患者）
        database: 数据库类型 ('miiv', 'mimic', 'eicu', 'hirid', 'aumc')
        interval_hours: 时间间隔（小时，默认 1 小时）
        concepts: 要加载的概念列表（可选，默认加载所有常见生命体征）
    
    Returns:
        DataFrame，包含列：
            - stay_id: 患者 ICU 住院 ID
            - charttime: 时间点（相对入 ICU 时间的小时数）
            - hr: 心率
            - sbp/dbp: 血压（收缩压/舒张压）
            - resp: 呼吸频率
            - temp: 体温（摄氏度）
            - spo2: 血氧饱和度
    
    Examples:
        >>> # 加载所有生命体征
        >>> vitals = load_vitals('/data/mimic', patient_ids=[10001, 10002, 10003])
        >>> 
        >>> # 只加载心率和血压
        >>> hr_bp = load_vitals('/data/mimic', patient_ids=[10001], 
        ...                     concepts=['hr', 'sbp', 'dbp'])
        >>> 
        >>> # 每 6 小时采样
        >>> vitals_6h = load_vitals('/data/mimic', interval_hours=6)
    """
    # 默认生命体征概念
    if concepts is None:
        concepts = ['hr', 'sbp', 'dbp', 'resp', 'temp', 'spo2']
    
    loader = ICUQuickLoader(data_path, database=database)
    
    df = loader.load_concepts(
        concepts,
        patient_ids=patient_ids,
        interval=pd.Timedelta(hours=interval_hours),
        verbose=False
    )
    
    return df


def load_labs(
    data_path: Union[str, Path],
    patient_ids: Optional[List[int]] = None,
    database: str = 'miiv',
    interval_hours: float = 6.0,
    concepts: Optional[List[str]] = None
) -> pd.DataFrame:
    """加载实验室检查数据（极简接口）
    
    Args:
        data_path: ICU 数据路径
        patient_ids: 患者 ID 列表
        database: 数据库类型
        interval_hours: 时间间隔（小时，默认 6 小时，因为实验室检查频率低）
        concepts: 要加载的实验室检查（可选）
    
    Returns:
        DataFrame，包含常见实验室检查结果
    
    Examples:
        >>> labs = load_labs('/data/mimic', patient_ids=[10001, 10002])
        >>> print(labs.columns)
        >>> # ['stay_id', 'charttime', 'wbc', 'hgb', 'plt', 'creat', 'bili', ...]
    """
    if concepts is None:
        # 常见实验室检查
        concepts = [
            'wbc',      # 白细胞计数
            'hgb',      # 血红蛋白
            'plt',      # 血小板
            'crea',     # 肌酐
            'bili',     # 胆红素
            'lact',     # 乳酸
        ]
    
    loader = ICUQuickLoader(data_path, database=database)
    
    df = loader.load_concepts(
        concepts,
        patient_ids=patient_ids,
        interval=pd.Timedelta(hours=interval_hours),
        verbose=False
    )
    
    return df


def load_sofa_score(
    data_path: Union[str, Path],
    patient_ids: Optional[List[int]] = None,
    database: str = 'miiv',
    interval_hours: float = 1.0,
    keep_components: bool = True
) -> pd.DataFrame:
    """加载 SOFA 评分（极简接口）
    
    Args:
        data_path: ICU 数据路径
        patient_ids: 患者 ID 列表
        database: 数据库类型
        interval_hours: 时间间隔（小时）
        keep_components: 是否保留 SOFA 组件分数
    
    Returns:
        DataFrame，包含 SOFA 总分和各组件分数
    
    Examples:
        >>> sofa = load_sofa_score('/data/mimic', patient_ids=[10001, 10002])
        >>> print(sofa.columns)
        >>> # ['stay_id', 'charttime', 'sofa', 'sofa_resp', 'sofa_coag', ...]
    """
    loader = ICUQuickLoader(data_path, database=database)
    
    df = loader.load_concepts(
        'sofa',
        patient_ids=patient_ids,
        interval=pd.Timedelta(hours=interval_hours),
        keep_components=keep_components,
        verbose=False
    )
    
    return df


def load_sepsis(
    data_path: Union[str, Path],
    patient_ids: Optional[List[int]] = None,
    database: str = 'miiv',
    interval_hours: float = 1.0,
    definition: str = 'sepsis3'
) -> pd.DataFrame:
    """加载脓毒症诊断数据（极简接口）
    
    Args:
        data_path: ICU 数据路径
        patient_ids: 患者 ID 列表
        database: 数据库类型
        interval_hours: 时间间隔（小时）
        definition: 脓毒症定义 ('sepsis3' 或 'sepsis2')
    
    Returns:
        DataFrame，包含脓毒症诊断相关特征
    
    Examples:
        >>> sepsis = load_sepsis('/data/mimic', patient_ids=[10001, 10002])
        >>> print(sepsis.columns)
        >>> # ['stay_id', 'charttime', 'sofa', 'abx', 'samp', 'susp_inf', 'sep3']
        >>> 
        >>> # 查看 Sepsis-3 阳性患者
        >>> positive = sepsis[sepsis['sep3'] == True]
    """
    loader = ICUQuickLoader(data_path, database=database)
    
    if definition == 'sepsis3':
        concept = 'sep3'
    elif definition == 'sepsis2':
        concept = 'sep2'  # 如果已定义
    else:
        raise ValueError(f"Unknown sepsis definition: {definition}")
    
    df = loader.load_concepts(
        concept,
        patient_ids=patient_ids,
        interval=pd.Timedelta(hours=interval_hours),
        verbose=False
    )
    
    return df


def load_custom(
    data_path: Union[str, Path],
    concepts: Union[str, List[str]],
    patient_ids: Optional[List[int]] = None,
    database: str = 'miiv',
    interval_hours: float = 1.0
) -> pd.DataFrame:
    """加载自定义概念组合（通用接口）
    
    最灵活的接口，可以加载任意概念组合。
    
    Args:
        data_path: ICU 数据路径
        concepts: 概念名称（字符串或列表）
        patient_ids: 患者 ID 列表
        database: 数据库类型
        interval_hours: 时间间隔（小时）
    
    Returns:
        DataFrame，包含请求的概念数据
    
    Examples:
        >>> # 加载单个概念
        >>> hr = load_custom('/data/mimic', 'hr', patient_ids=[10001])
        >>> 
        >>> # 加载多个概念
        >>> features = load_custom('/data/mimic', 
        ...                        ['hr', 'sbp', 'temp', 'sofa'],
        ...                        patient_ids=[10001, 10002])
    """
    loader = ICUQuickLoader(data_path, database=database)
    
    df = loader.load_concepts(
        concepts,
        patient_ids=patient_ids,
        interval=pd.Timedelta(hours=interval_hours),
        verbose=False
    )
    
    return df


def quick_summary(
    data_path: Union[str, Path],
    patient_ids: Optional[List[int]] = None,
    database: str = 'miiv'
) -> dict:
    """快速生成患者数据摘要
    
    Args:
        data_path: ICU 数据路径
        patient_ids: 患者 ID 列表
        database: 数据库类型
    
    Returns:
        包含各类数据统计的字典
    
    Examples:
        >>> summary = quick_summary('/data/mimic', patient_ids=[10001, 10002, 10003])
        >>> print(summary)
        >>> # {
        >>> #     'patients': 3,
        >>> #     'vitals_records': 230,
        >>> #     'lab_records': 45,
        >>> #     'sofa_mean': 2.5,
        >>> #     'sepsis_positive': 1
        >>> # }
    """
    try:
        vitals = load_vitals(data_path, patient_ids, database, concepts=['hr'])
        vitals_count = len(vitals)
    except Exception:
        vitals_count = 0
    
    try:
        labs = load_labs(data_path, patient_ids, database, concepts=['wbc'])
        labs_count = len(labs)
    except Exception:
        labs_count = 0
    
    try:
        sofa = load_sofa_score(data_path, patient_ids, database)
        sofa_mean = sofa['sofa'].mean() if 'sofa' in sofa.columns else None
    except Exception:
        sofa_mean = None
    
    try:
        sepsis = load_sepsis(data_path, patient_ids, database)
        sepsis_positive = sepsis['sep3'].sum() if 'sep3' in sepsis.columns else 0
    except Exception:
        sepsis_positive = 0
    
    return {
        'patients': len(patient_ids) if patient_ids else 'all',
        'vitals_records': vitals_count,
        'lab_records': labs_count,
        'sofa_mean': sofa_mean,
        'sepsis_positive': sepsis_positive
    }
