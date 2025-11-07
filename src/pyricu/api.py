"""
pyricu 高层API - 提供像 R ricu 一样简单的接口
让用户可以直接 load_concept() 提取特征

使用示例:
    >>> from pyricu import load_concept
    >>> 
    >>> # 从MIMIC加载心率数据
    >>> hr = load_concept('hr', 'mimic', '/path/to/mimic')
    >>> 
    >>> # 加载多个概念
    >>> vitals = load_concept(['hr', 'sbp', 'dbp', 'temp'], 'mimic', '/path/to/mimic')
    >>> 
    >>> # 加载SOFA相关指标
    >>> sofa_data = load_concept(['pafi', 'plt', 'bili', 'map', 'gcs', 'crea'], 
    ...                          'mimic', '/path/to/mimic')
"""

from typing import List, Union, Optional, Dict
from pathlib import Path
import pandas as pd

from .concept import ConceptDictionary, ConceptResolver
from .datasource import ICUDataSource
from .config import DataSourceConfig
from .resources import load_data_sources, load_dictionary


def load_concept(
    concepts: Union[str, List[str]],
    source: str,
    data_path: Union[str, Path],
    patient_ids: Optional[List] = None,
    merge: bool = True,
    verbose: bool = True,
    **kwargs,  # Additional parameters for callbacks (e.g., win_length, worst_val_fun)
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    加载ICU概念数据 - 主要的便捷函数
    
    这个函数提供了最简单的接口来加载概念数据，类似于 R ricu 的 load_concepts()
    
    Args:
        concepts: 概念名称或概念名称列表
            例如: 'hr', ['hr', 'sbp', 'temp'], ['pafi', 'plt', 'bili']
        source: 数据源名称
            支持: 'mimic', 'hirid', 'eicu', 'aumc'
        data_path: 数据源的文件路径
            例如: '/home/data/mimiciv/3.1'
        patient_ids: 可选的患者ID列表，用于过滤
        merge: 如果为True，将多个概念合并为一个宽格式DataFrame
        verbose: 是否显示加载进度
        **kwargs: 额外的回调参数
            win_length: 滑动窗口长度 (例如: pd.Timedelta(hours=24))
            worst_val_fun: 聚合函数 ('max', 'min', 'mean', 或自定义函数)
            keep_components: 对于SOFA，是否保留各组件
            full_window: 是否要求完整窗口
        
    Returns:
        如果 merge=True: 返回合并后的DataFrame
        如果 merge=False: 返回 dict {概念名: DataFrame}
        
    Examples:
        >>> # 加载单个概念
        >>> hr_data = load_concept('hr', 'mimic', '/data/mimic')
        >>> 
        >>> # 加载多个生命体征
        >>> vitals = load_concept(['hr', 'sbp', 'dbp', 'temp', 'spo2'], 
        ...                       'mimic', '/data/mimic')
        >>> 
        >>> # 加载SOFA相关指标（带24小时滑动窗口）
        >>> sofa = load_concept('sofa', 'mimic', '/data/mimic',
        ...                     win_length=pd.Timedelta(hours=24),
        ...                     worst_val_fun='max')
    """
    # 1. 准备概念列表
    if isinstance(concepts, str):
        concept_list = [concepts]
    else:
        concept_list = list(concepts)
    
    if verbose:
        print(f"📊 从 {source.upper()} 加载 {len(concept_list)} 个概念...")
        print(f"   概念: {', '.join(concept_list)}")
        if kwargs:
            print(f"   参数: {kwargs}")
    
    # 2. 加载数据源配置
    registry = load_data_sources()
    if source not in registry:
        available = [cfg.name for cfg in registry]
        raise ValueError(f"未知数据源 '{source}'。可用: {available}")
    
    source_config = registry.get(source)
    
    # 3. 创建数据源实例
    datasource = ICUDataSource(
        config=source_config,
        base_path=Path(data_path)
    )
    
    # 4. 加载概念字典
    dict_obj = load_dictionary()
    
    # 5. 创建概念解析器
    resolver = ConceptResolver(dict_obj)
    
    # 6. 加载概念数据
    try:
        result = resolver.load_concepts(
            concept_list,
            datasource,
            patient_ids=patient_ids,
            merge=merge,
            **kwargs,  # Pass kwargs to resolver
        )
        
        if verbose:
            if merge:
                # result可能是DataFrame或ICUTable
                if hasattr(result, 'data'):
                    df_result = result.data
                    print(f"✅ 成功加载 {len(df_result)} 行数据")
                    print(f"   列: {list(df_result.columns)}")
                elif isinstance(result, pd.DataFrame):
                    print(f"✅ 成功加载 {len(result)} 行数据")
                    print(f"   列: {list(result.columns)}")
                else:
                    print(f"✅ 成功加载数据 (类型: {type(result)})")
            else:
                total_rows = sum(len(df.data) if hasattr(df, 'data') else len(df) 
                               for df in result.values())
                print(f"✅ 成功加载 {total_rows} 行数据，{len(result)} 个概念")
        
        # 如果merge=True且返回的是ICUTable，转换为DataFrame
        if merge and hasattr(result, 'data'):
            return result.data
        return result
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        raise


def load_sofa_components(
    source: str,
    data_path: Union[str, Path],
    patient_ids: Optional[List] = None,
    merge: bool = True,
    verbose: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    加载SOFA评分所需的所有组件
    
    这是一个便捷函数，一次性加载SOFA评分需要的所有临床指标
    
    Args:
        source: 数据源名称 ('mimic', 'hirid', 'eicu', 'aumc')
        data_path: 数据源路径
        patient_ids: 可选的患者ID过滤列表
        merge: 是否合并为宽格式
        verbose: 是否显示进度
        
    Returns:
        包含SOFA组件数据的DataFrame或字典
        
    Examples:
        >>> # 加载MIMIC的SOFA组件
        >>> sofa = load_sofa_components('mimic', '/data/mimic')
        >>> 
        >>> # 获取的数据包含:
        >>> # - pafi: PaO2/FiO2 比值 (呼吸)
        >>> # - plt: 血小板计数 (凝血)
        >>> # - bili: 胆红素 (肝脏)
        >>> # - map: 平均动脉压 (心血管)
        >>> # - gcs: Glasgow昏迷评分 (神经)
        >>> # - crea: 肌酐 (肾脏)
    """
    sofa_concepts = [
        'pafi',      # 呼吸: PaO2/FiO2
        'plt',       # 凝血: 血小板
        'bili',      # 肝脏: 胆红素
        'map',       # 心血管: 平均动脉压
        'gcs',       # 神经: Glasgow昏迷评分
        'crea',      # 肾脏: 肌酐
    ]
    
    if verbose:
        print("🏥 加载SOFA评分组件...")
    
    return load_concept(
        sofa_concepts,
        source,
        data_path,
        patient_ids=patient_ids,
        merge=merge,
        verbose=verbose
    )


def load_vitals(
    source: str,
    data_path: Union[str, Path],
    patient_ids: Optional[List] = None,
    merge: bool = True,
    verbose: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    加载常用生命体征
    
    Args:
        source: 数据源名称
        data_path: 数据源路径
        patient_ids: 可选的患者ID过滤
        merge: 是否合并
        verbose: 显示进度
        
    Returns:
        生命体征数据
        
    Examples:
        >>> vitals = load_vitals('mimic', '/data/mimic')
    """
    vital_concepts = [
        'hr',        # 心率
        'sbp',       # 收缩压
        'dbp',       # 舒张压
        'temp',      # 体温
        'resp',      # 呼吸频率
        'spo2',      # 血氧饱和度
    ]
    
    if verbose:
        print("❤️  加载生命体征...")
    
    return load_concept(
        vital_concepts,
        source,
        data_path,
        patient_ids=patient_ids,
        merge=merge,
        verbose=verbose
    )


def load_labs(
    source: str,
    data_path: Union[str, Path],
    patient_ids: Optional[List] = None,
    merge: bool = True,
    verbose: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    加载常用实验室检查
    
    Args:
        source: 数据源名称
        data_path: 数据源路径
        patient_ids: 可选的患者ID过滤
        merge: 是否合并
        verbose: 显示进度
        
    Returns:
        实验室检查数据
        
    Examples:
        >>> labs = load_labs('mimic', '/data/mimic')
    """
    lab_concepts = [
        'wbc',       # 白细胞
        'plt',       # 血小板
        'crea',      # 肌酐
        'bili',      # 胆红素
        'lac',       # 乳酸
        'ph',        # pH值
    ]
    
    if verbose:
        print("🔬 加载实验室检查...")
    
    return load_concept(
        lab_concepts,
        source,
        data_path,
        patient_ids=patient_ids,
        merge=merge,
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
    'load_concept',
    'load_sofa_components',
    'load_vitals',
    'load_labs',
    'list_available_concepts',
    'list_available_sources',
    'get_concept_info',
]
