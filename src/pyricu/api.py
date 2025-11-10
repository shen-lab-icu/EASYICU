"""
pyricu 高层API - 提供简单易用的接口，同时支持高级自定义

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

from .concept import ConceptDictionary, ConceptResolver
from .datasource import ICUDataSource
from .config import DataSourceConfig
from .resources import load_data_sources, load_dictionary


def _detect_database() -> str:
    """自动检测数据库类型（从环境变量或路径）"""
    # 检查环境变量
    for db_name in ['miiv', 'mimic', 'eicu', 'hirid', 'aumc']:
        env_var = f'{db_name.upper()}_PATH'
        if os.getenv(env_var):
            return db_name
    
    # 默认返回 miiv
    return 'miiv'


def _get_default_data_path(database: str) -> Optional[Path]:
    """获取数据库的默认路径"""
    # 检查环境变量
    env_var = f'{database.upper()}_PATH'
    path = os.getenv(env_var)
    if path:
        return Path(path)
    
    # 检查常见路径
    common_paths = [
        Path.home() / 'data' / database,
        Path('/data') / database,
        Path('.') / 'data' / database,
    ]
    
    for path in common_paths:
        if path.exists():
            return path
    
    return None


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
    **kwargs,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    加载ICU概念数据 - pyricu的主要API
    
    这个函数提供了灵活的接口，既支持简单用法（自动检测），
    也支持完全自定义（显式指定所有参数）
    
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
        
        # === 其他 ===
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
    # 1. 准备概念列表
    if isinstance(concepts, str):
        concept_list = [concepts]
    else:
        concept_list = list(concepts)
    
    # 2. 自动检测数据库（如果未指定）
    if database is None:
        database = _detect_database()
        if verbose:
            print(f"� 自动检测数据库: {database}")
    
    # 3. 自动查找数据路径（如果未指定）
    if data_path is None:
        data_path = _get_default_data_path(database)
        if data_path is None:
            raise ValueError(
                f"无法找到 {database} 数据路径。请:\n"
                f"1. 设置环境变量 {database.upper()}_PATH=/path/to/data\n"
                f"2. 或显式传递 data_path 参数"
            )
        if verbose:
            print(f"📁 使用数据路径: {data_path}")
    else:
        data_path = Path(data_path)
    
    if verbose:
        print(f"📊 从 {database.upper()} 加载 {len(concept_list)} 个概念...")
        print(f"   概念: {', '.join(concept_list)}")
    
    # 4. 加载数据源配置
    registry = load_data_sources()
    if database not in registry:
        available = list(registry.keys())
        raise ValueError(f"未知数据源 '{database}'。可用: {available}")
    
    source_config = registry.get(database)
    
    # 5. 创建数据源实例
    datasource = ICUDataSource(
        config=source_config,
        base_path=data_path
    )
    
    # 6. 加载概念字典（检查是否需要SOFA2）
    need_sofa2 = any('sofa2' in c.lower() for c in concept_list)
    dict_obj = load_dictionary(include_sofa2=need_sofa2)
    
    # 7. 创建概念解析器
    resolver = ConceptResolver(dict_obj)
    
    # 8. 规范化患者ID
    if patient_ids is not None and not isinstance(patient_ids, dict):
        # 根据数据库类型选择正确的ID列
        if database in ['eicu', 'eicu_demo']:
            patient_ids = {'patientunitstayid': patient_ids}
        elif database in ['aumc']:
            patient_ids = {'admissionid': patient_ids}
        elif database in ['hirid']:
            patient_ids = {'patientid': patient_ids}
        else:
            # MIMIC-IV 等使用 stay_id
            patient_ids = {'stay_id': patient_ids}
    
    # 9. 处理时间参数
    if isinstance(interval, str):
        # 将字符串转换为Timedelta
        interval = pd.Timedelta(interval)
    
    if isinstance(win_length, str):
        win_length = pd.Timedelta(win_length)
    
    # 10. 准备kwargs
    load_kwargs = {
        'patient_ids': patient_ids,
        'verbose': verbose,
    }
    
    if interval is not None:
        load_kwargs['interval'] = interval
        load_kwargs['align_to_admission'] = True
    
    if win_length is not None:
        load_kwargs['win_length'] = win_length
    
    if aggregate is not None:
        load_kwargs['aggregate'] = aggregate
    
    if keep_components:
        load_kwargs['keep_components'] = keep_components
    
    # 合并额外的kwargs
    load_kwargs.update(kwargs)
    
    # 11. 加载概念数据
    try:
        result = resolver.load_concepts(
            concept_list,
            datasource,
            **load_kwargs,
        )
        
        if verbose:
            if hasattr(result, 'data'):
                df_result = result.data
                print(f"✅ 成功加载 {len(df_result):,} 行数据")
                print(f"   列: {list(df_result.columns)}")
            elif isinstance(result, pd.DataFrame):
                print(f"✅ 成功加载 {len(result):,} 行数据")
                print(f"   列: {list(result.columns)}")
            elif isinstance(result, dict):
                total_rows = sum(
                    len(df.data) if hasattr(df, 'data') else len(df) 
                    for df in result.values()
                )
                print(f"✅ 成功加载 {total_rows:,} 行数据，{len(result)} 个概念")
        
        # 如果返回的是ICUTable，转换为DataFrame
        if hasattr(result, 'data'):
            return result.data
        return result
        
    except Exception as e:
        if verbose:
            print(f"❌ 加载失败: {e}")
        raise


# 为了兼容旧代码，保留旧的函数名
def load_concept(*args, **kwargs):
    """load_concepts的别名（向后兼容）"""
    return load_concepts(*args, **kwargs)


def load_sofa(
    database: str,
    data_path: Union[str, Path],
    patient_ids: Optional[Union[List, Dict]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    keep_components: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载SOFA评分（便捷函数）
    
    Args:
        database: 数据库类型 ('miiv', 'eicu', 'hirid', 'aumc')
        data_path: 数据路径
        patient_ids: 患者ID列表（None=所有患者）
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        keep_components: 是否保留组件（默认True）
        verbose: 是否显示详细信息
        
    Returns:
        SOFA评分DataFrame
        
    Examples:
        >>> # 基本用法
        >>> sofa = load_sofa('miiv', '/data/miiv', patient_ids=[123, 456])
        >>> 
        >>> # 自定义窗口
        >>> sofa = load_sofa('miiv', '/data/miiv', patient_ids=[123, 456],
        ...                  win_length='12h', interval='6h')
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
        verbose=verbose
    )


def load_sofa2(
    database: str,
    data_path: Union[str, Path],
    patient_ids: Optional[Union[List, Dict]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    win_length: Union[str, pd.Timedelta] = '24h',
    keep_components: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载SOFA-2评分（2025年新标准）
    
    Args:
        database: 数据库类型 ('miiv', 'eicu', 'hirid', 'aumc')
        data_path: 数据路径
        patient_ids: 患者ID列表（None=所有患者）
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        keep_components: 是否保留组件（默认True）
        verbose: 是否显示详细信息
        
    Returns:
        SOFA-2评分DataFrame
        
    Examples:
        >>> sofa2 = load_sofa2('miiv', '/data/miiv', patient_ids=[123, 456])
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
        verbose=verbose
    )


def load_sepsis3(
    database: str,
    data_path: Union[str, Path],
    patient_ids: Optional[Union[List, Dict]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载Sepsis-3诊断相关数据
    
    包含: SOFA, abx, samp, susp_inf, sep3
    
    Args:
        database: 数据库类型 ('miiv', 'eicu', 'hirid', 'aumc')
        data_path: 数据路径
        patient_ids: 患者ID列表（None=所有患者）
        interval: 时间间隔（默认1小时）
        verbose: 是否显示详细信息
        
    Returns:
        Sepsis-3数据DataFrame
        
    Examples:
        >>> sep3 = load_sepsis3('miiv', '/data/miiv', patient_ids=[123, 456])
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
    database: str,
    data_path: Union[str, Path],
    patient_ids: Optional[Union[List, Dict]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载生命体征数据（便捷函数）
    
    包含: hr, sbp, dbp, map, temp, resp, spo2
    
    Args:
        database: 数据库类型 ('miiv', 'eicu', 'hirid', 'aumc')
        data_path: 数据路径
        patient_ids: 患者ID列表（None=所有患者）
        interval: 时间间隔（默认1小时）
        verbose: 是否显示详细信息
        
    Returns:
        生命体征DataFrame
        
    Examples:
        >>> vitals = load_vitals('miiv', '/data/miiv', patient_ids=[123, 456])
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
    database: str,
    data_path: Union[str, Path],
    patient_ids: Optional[Union[List, Dict]] = None,
    interval: Union[str, pd.Timedelta] = '1h',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载实验室检查数据（便捷函数）
    
    包含: wbc, hgb, plt, na, k, crea, bili, lactate
    
    Args:
        database: 数据库类型 ('miiv', 'eicu', 'hirid', 'aumc')
        data_path: 数据路径
        patient_ids: 患者ID列表（None=所有患者）
        interval: 时间间隔（默认1小时）
        verbose: 是否显示详细信息
        
    Returns:
        实验室检查DataFrame
        
    Examples:
        >>> labs = load_labs('miiv', '/data/miiv', patient_ids=[123, 456])
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
    
    # 工具函数
    'list_available_concepts',
    'list_available_sources',
    'get_concept_info',
]
