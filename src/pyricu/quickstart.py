"""
pyricu 快速启动 API - DEPRECATED

⚠️  此模块已被弃用，将在未来版本中移除。

请使用新的简化API:
    from pyricu import load_concepts, load_sofa, load_vitals
    
    # 旧代码:
    loader = ICUQuickLoader('miiv', '/data')
    sofa = loader.load_sofa(patient_ids=[123])
    
    # 新代码（更简单）:
    sofa = load_sofa(patient_ids=[123], database='miiv', data_path='/data')
    
    # 或者更简单（自动检测）:
    sofa = load_sofa(patient_ids=[123])

新API的优势:
- 无需创建loader对象
- 支持智能默认值（自动检测数据库和路径）
- 更少的代码，更清晰的语义
- 保留所有高级功能（完全向后兼容）

Examples:
    >>> from pyricu import load_sofa, load_concepts
    >>> 
    >>> # 简单用法
    >>> sofa = load_sofa(patient_ids=[10001, 10002])
    >>> 
    >>> # 完全自定义
    >>> data = load_concepts('sofa2',
    ...                      patient_ids=[10001, 10002],
    ...                      database='miiv',
    ...                      interval='6h',
    ...                      win_length='24h')
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Union, Dict
import pandas as pd

from .datasource import ICUDataSource
from .concept import ConceptResolver, ConceptDictionary
from .resources import load_data_sources


# 显示弃用警告
warnings.warn(
    "pyricu.quickstart模块已被弃用，将在未来版本中移除。"
    "请使用 pyricu.api 模块（from pyricu import load_concepts, load_sofa等）。"
    "详见文档: https://github.com/shen-lab-icu/pyricu",
    DeprecationWarning,
    stacklevel=2
)


class ICUQuickLoader:
    """ICU 数据快速加载器
    
    封装了所有初始化逻辑，提供简洁的 API
    支持多个ICU数据库：MIMIC-IV, MIMIC-III, eICU, HiRID, AUMC 等
    """
    
    def __init__(
        self, 
        data_path: Union[str, Path],
        database: str = 'miiv',
        dict_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        use_sofa2: bool = False
    ):
        """初始化加载器
        
        Args:
            data_path: ICU 数据路径
            database: 数据库类型 ('miiv', 'mimic', 'eicu', 'hirid', 'aumc' 等)
            dict_path: 概念字典路径（可选，可以是单个文件或文件列表）
            use_sofa2: 是否加载 SOFA2 字典（默认 False）
        """
        self.data_path = Path(data_path)
        self.database = database
        
        # 加载配置
        registry = load_data_sources()
        self.datasource = ICUDataSource(
            config=registry.get(database),
            base_path=data_path
        )
        
        # 加载概念字典
        if dict_path is None:
            # 使用内置字典
            from importlib.resources import files
            base_dict = files('pyricu').joinpath('data/concept-dict.json')
            
            if use_sofa2:
                # 同时加载 SOFA2 字典
                sofa2_dict = files('pyricu').joinpath('data/sofa2-dict.json')
                self.dictionary = ConceptDictionary.from_multiple_json([str(base_dict), str(sofa2_dict)])
            else:
                self.dictionary = ConceptDictionary.from_json(base_dict)
        elif isinstance(dict_path, list):
            # 加载多个字典文件
            self.dictionary = ConceptDictionary.from_multiple_json(dict_path)
        else:
            # 加载单个字典文件
            self.dictionary = ConceptDictionary.from_json(dict_path)
        
        self.resolver = ConceptResolver(self.dictionary)
    
    def load_concepts(
        self,
        concept_names: Union[str, List[str]],
        patient_ids: Optional[Union[List, Dict]] = None,
        interval: pd.Timedelta = pd.Timedelta(hours=1),
        win_length: Optional[pd.Timedelta] = None,
        keep_components: bool = False,
        verbose: bool = False
    ) -> pd.DataFrame:
        """通用概念加载方法
        
        Args:
            concept_names: 概念名称（字符串或列表）
            patient_ids: 患者ID列表或字典 {'stay_id': [...]}
            interval: 时间间隔
            win_length: 窗口长度（用于 SOFA 等评分）
            keep_components: 是否保留组件列（如 sofa_resp, sofa_coag 等）
            verbose: 是否显示详细信息
        
        Returns:
            包含概念数据的 DataFrame
        """
        # 规范化概念名称
        if isinstance(concept_names, str):
            concept_names = [concept_names]
        
        # 规范化患者ID - 根据数据库类型选择正确的ID列名
        if patient_ids is not None and not isinstance(patient_ids, dict):
            # 根据数据库类型选择合适的ID列名
            if self.database in ['eicu', 'eicu_demo']:
                patient_ids = {'patientunitstayid': patient_ids}
            else:
                # MIMIC-IV等使用stay_id
                patient_ids = {'stay_id': patient_ids}
        
        # 加载概念
        kwargs = {
            'interval': interval,
            'align_to_admission': True,
            'verbose': verbose,
            'keep_components': keep_components
        }
        
        if win_length is not None:
            kwargs['win_length'] = win_length
        
        result = self.resolver.load_concepts(
            concept_names,
            self.datasource,
            patient_ids=patient_ids,
            **kwargs
        )
        
        return result


def load_sofa(
    data_path: Union[str, Path],
    patient_ids: Optional[Union[List, Dict]] = None,
    interval: pd.Timedelta = pd.Timedelta(hours=1),
    win_length: pd.Timedelta = pd.Timedelta(hours=24),
    keep_components: bool = True,
    verbose: bool = False,
    database: str = 'miiv'
) -> pd.DataFrame:
    """一行代码加载 SOFA 评分及其组件（适用于所有ICU数据库）
    
    Args:
        data_path: ICU 数据路径
        patient_ids: 患者ID列表，例如 [10001, 10002] 或 {'stay_id': [10001, 10002]}
        interval: 时间间隔（默认 1 小时）
        win_length: 窗口长度（默认 24 小时）
        keep_components: 是否保留 SOFA 组件（默认 True）
        verbose: 是否显示详细信息
        database: 数据库类型（'miiv', 'mimic', 'eicu', 'hirid', 'aumc' 等）
    
    Returns:
        DataFrame，包含列：stay_id, charttime, sofa, sofa_resp, sofa_coag, 
                          sofa_liver, sofa_cardio, sofa_cns, sofa_renal
    
    Examples:
        >>> # 加载所有患者的 SOFA
        >>> df = load_sofa('/path/to/icu_data', database='miiv')
        >>> 
        >>> # 加载特定患者
        >>> df = load_sofa('/path/to/icu_data', patient_ids=[10001, 10002, 99003], database='eicu')
        >>> 
        >>> # 只要总分，不要组件
        >>> df = load_sofa('/path/to/icu_data', keep_components=False)
    """
    loader = ICUQuickLoader(data_path, database=database)
    return loader.load_concepts(
        'sofa',
        patient_ids=patient_ids,
        interval=interval,
        win_length=win_length,
        keep_components=keep_components,
        verbose=verbose
    )


def load_sepsis3(
    data_path: Union[str, Path],
    patient_ids: Optional[Union[List, Dict]] = None,
    interval: pd.Timedelta = pd.Timedelta(hours=1),
    verbose: bool = False,
    database: str = 'miiv',
    include_components: bool = True
) -> pd.DataFrame:
    """一行代码加载 Sepsis-3 相关特征（适用于所有ICU数据库）
    
    自动加载并合并：
    - SOFA 评分（及其组件）
    - 抗生素使用 (abx)
    - 体液采样 (samp)
    - 疑似感染 (susp_inf)
    - Sepsis-3 诊断 (sep3)
    
    Args:
        data_path: ICU 数据路径
        patient_ids: 患者ID列表
        interval: 时间间隔
        verbose: 是否显示详细信息
        database: 数据库类型（'miiv', 'mimic', 'eicu', 'hirid', 'aumc' 等）
        include_components: 是否包含 SOFA 组件
    
    Returns:
        DataFrame，包含所有 Sepsis-3 相关特征
    
    Examples:
        >>> df = load_sepsis3('/path/to/icu_data', patient_ids=[10001, 10002], database='miiv')
        >>> 
        >>> # 查看 Sepsis-3 阳性的记录
        >>> sepsis_positive = df[df['sep3'] > 0]
    """
    loader = ICUQuickLoader(data_path, database=database)
    
    # 加载所有 Sepsis-3 相关概念
    concepts = ['sofa', 'abx', 'samp', 'susp_inf', 'sep3']
    
    # 分别加载并合并
    all_data = {}
    
    # SOFA（带组件）
    if verbose:
        print("📊 加载 SOFA 评分...")
    sofa_df = loader.load_concepts(
        'sofa',
        patient_ids=patient_ids,
        interval=interval,
        win_length=pd.Timedelta(hours=24),
        keep_components=include_components,
        verbose=verbose
    )
    all_data['sofa'] = sofa_df
    
    # 其他概念
    for concept in ['abx', 'samp', 'susp_inf', 'sep3']:
        try:
            if verbose:
                print(f"📊 加载 {concept}...")
            df = loader.load_concepts(
                concept,
                patient_ids=patient_ids,
                interval=interval,
                verbose=verbose
            )
            all_data[concept] = df
        except Exception as e:
            if verbose:
                print(f"⚠️  跳过 {concept}: {e}")
    
    # 合并所有数据
    if verbose:
        print("🔗 合并数据...")
    
    result = all_data['sofa'].copy()
    
    # 确定主ID列（来自sofa）
    # 支持多种数据库的ID列名
    primary_id_col = None
    for col in ['stay_id', 'patientunitstayid', 'hadm_id', 'subject_id', 'icustay_id']:
        if col in result.columns:
            primary_id_col = col
            break
    
    if primary_id_col is None:
        raise ValueError("无法确定主ID列")
    
    # 加载 ID 映射表（用于 hadm_id <-> stay_id 转换）
    id_mapping = None
    if primary_id_col == 'stay_id':
        # 尝试加载 icustays 表以获取 hadm_id <-> stay_id 映射
        try:
            from .fst_reader import read_fst
            icustays_file = Path(data_path) / 'icustays.fst'
            if icustays_file.exists():
                icustays = read_fst(icustays_file)
                if 'stay_id' in icustays.columns and 'hadm_id' in icustays.columns:
                    id_mapping = icustays[['stay_id', 'hadm_id']].drop_duplicates()
                    if verbose:
                        print(f"   📋 加载 ID 映射表: stay_id ↔ hadm_id ({len(id_mapping)} 条)")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  无法加载 ID 映射表: {e}")
    
    for concept, df in all_data.items():
        if concept == 'sofa':
            continue
        
        if df.empty:
            if verbose:
                print(f"   ⚠️  {concept} 数据为空，跳过合并")
            continue
        
        df_to_merge = df.copy()
        
        # 步骤 1: ID 列转换
        # 检测 df 的 ID 列，支持多种数据库
        df_id_col = None
        for col in ['stay_id', 'patientunitstayid', 'hadm_id', 'subject_id', 'icustay_id']:
            if col in df_to_merge.columns:
                df_id_col = col
                break
        
        if df_id_col is None:
            if verbose:
                print(f"   ⚠️  {concept} 没有ID列，跳过合并")
            continue
        
        # 如果 ID 列不匹配，尝试转换
        if df_id_col != primary_id_col:
            if id_mapping is not None:
                # 尝试转换 ID
                if df_id_col == 'hadm_id' and primary_id_col == 'stay_id':
                    # hadm_id -> stay_id
                    df_to_merge = df_to_merge.merge(id_mapping, on='hadm_id', how='left')
                    df_to_merge = df_to_merge.drop(columns=['hadm_id'])
                    df_id_col = 'stay_id'
                    if verbose:
                        print(f"   🔄 {concept}: 转换 hadm_id → stay_id")
                elif df_id_col == 'stay_id' and primary_id_col == 'hadm_id':
                    # stay_id -> hadm_id
                    df_to_merge = df_to_merge.merge(id_mapping, on='stay_id', how='left')
                    df_to_merge = df_to_merge.drop(columns=['stay_id'])
                    df_id_col = 'hadm_id'
                    if verbose:
                        print(f"   🔄 {concept}: 转换 stay_id → hadm_id")
            
            # 如果仍然不匹配，跳过
            if df_id_col != primary_id_col:
                if verbose:
                    print(f"   ⚠️  {concept} ID列不匹配（{df_id_col} vs {primary_id_col}），跳过合并")
                continue
        
        # 步骤 2: 时间列标准化
        time_col_in_df = None
        for col in ['charttime', 'starttime', 'endtime', 'chartdate']:
            if col in df_to_merge.columns:
                time_col_in_df = col
                break
        
        # 标准化时间列名为 charttime
        if time_col_in_df and time_col_in_df != 'charttime':
            df_to_merge = df_to_merge.rename(columns={time_col_in_df: 'charttime'})
            if verbose:
                print(f"   🔄 {concept}: 重命名 {time_col_in_df} → charttime")
        
        # 步骤 3: 移除冗余的 chartdate（如果 charttime 已存在）
        if 'chartdate' in df_to_merge.columns and 'charttime' in df_to_merge.columns:
            df_to_merge = df_to_merge.drop(columns=['chartdate'])
        
        # 步骤 4: 时间列类型对齐
        # 确保两边的 charttime 类型一致
        if 'charttime' in df_to_merge.columns and 'charttime' in result.columns:
            result_time_dtype = result['charttime'].dtype
            df_time_dtype = df_to_merge['charttime'].dtype
            
            # 如果类型不同，需要转换
            if result_time_dtype != df_time_dtype:
                # 优先保持 result 的类型（通常是 float64，相对时间）
                if pd.api.types.is_numeric_dtype(result_time_dtype):
                    # result 是数值型（相对时间），需要转换 df_to_merge
                    if pd.api.types.is_datetime64_any_dtype(df_time_dtype):
                        # df_to_merge 是 datetime，需要转换为相对时间
                        # 尝试通过 icustays 表获取入院时间进行转换
                        try:
                            from .fst_reader import read_fst
                            icustays_file = Path(data_path) / 'icustays.fst'
                            if icustays_file.exists():
                                icustays = read_fst(icustays_file)
                                if primary_id_col in icustays.columns and 'intime' in icustays.columns:
                                    # 合并入院时间
                                    df_with_intime = df_to_merge.merge(
                                        icustays[[primary_id_col, 'intime']].drop_duplicates(),
                                        on=primary_id_col,
                                        how='left'
                                    )
                                    # 转换为相对小时数（处理时区问题）
                                    df_with_intime['intime'] = pd.to_datetime(df_with_intime['intime'], errors='coerce', utc=True).dt.tz_localize(None)
                                    df_with_intime['charttime'] = pd.to_datetime(df_with_intime['charttime'], errors='coerce', utc=True).dt.tz_localize(None)
                                    time_diff = (df_with_intime['charttime'] - df_with_intime['intime']).dt.total_seconds() / 3600.0
                                    df_to_merge['charttime'] = time_diff
                                    if verbose:
                                        print(f"   ✅ {concept}: charttime 已转换为相对时间（小时）")
                                else:
                                    raise ValueError("icustays 缺少必要的列")
                            else:
                                raise FileNotFoundError("找不到 icustays.fst")
                        except Exception as e:
                            # 转换失败，跳过时间列合并
                            if verbose:
                                print(f"   ⚠️  {concept}: 无法转换 charttime 为相对时间（{e}），仅按 ID 合并")
                            # 不使用 charttime 作为合并键，但保留原始 charttime
                            df_to_merge = df_to_merge.drop(columns=['charttime'])
                    else:
                        # 尝试转换为数值型
                        try:
                            df_to_merge['charttime'] = pd.to_numeric(df_to_merge['charttime'], errors='coerce')
                        except:
                            if verbose:
                                print(f"   ⚠️  {concept}: 无法转换 charttime 为数值型，仅按 ID 合并")
                            df_to_merge = df_to_merge.drop(columns=['charttime'])
                elif pd.api.types.is_datetime64_any_dtype(result_time_dtype):
                    # result 是 datetime，需要转换 df_to_merge
                    if not pd.api.types.is_datetime64_any_dtype(df_time_dtype):
                        # 尝试转换为 datetime
                        try:
                            df_to_merge['charttime'] = pd.to_datetime(df_to_merge['charttime'], errors='coerce')
                        except:
                            if verbose:
                                print(f"   ⚠️  {concept}: 无法转换 charttime 为 datetime，仅按 ID 合并")
                            df_to_merge = df_to_merge.drop(columns=['charttime'])
        
        # 步骤 5: 确定合并键
        merge_keys = [primary_id_col]
        if 'charttime' in df_to_merge.columns and 'charttime' in result.columns:
            merge_keys.append('charttime')
        
        # 步骤 6: 合并
        try:
            result = result.merge(
                df_to_merge,
                on=merge_keys,
                how='left',
                suffixes=('', f'_{concept}')
            )
            if verbose:
                print(f"   ✅ {concept}: 合并成功（键: {merge_keys}）")
        except Exception as e:
            if verbose:
                print(f"   ❌ {concept}: 合并失败 - {e}")
    
    # 最后清理
    if 'chartdate' in result.columns:
        result = result.drop(columns=['chartdate'])
    
    if verbose:
        print(f"✅ 完成！总共 {len(result):,} 行，{len(result.columns)} 列")
    
    return result


def load_vitals(
    data_path: Union[str, Path],
    patient_ids: Optional[Union[List, Dict]] = None,
    interval: pd.Timedelta = pd.Timedelta(hours=1),
    verbose: bool = False,
    database: str = 'miiv'
) -> pd.DataFrame:
    """一行代码加载生命体征（适用于所有ICU数据库）
    
    包含：心率、血压、呼吸频率、体温、血氧饱和度等
    
    Args:
        data_path: ICU 数据路径
        patient_ids: 患者ID列表
        interval: 时间间隔
        verbose: 是否显示详细信息
        database: 数据库类型
    
    Returns:
        DataFrame，包含生命体征数据
    """
    loader = ICUQuickLoader(data_path, database=database)
    
    vital_concepts = ['hr', 'sbp', 'dbp', 'mbp', 'resp', 'temp', 'spo2']
    
    return loader.load_concepts(
        vital_concepts,
        patient_ids=patient_ids,
        interval=interval,
        verbose=verbose
    )


def load_labs(
    data_path: Union[str, Path],
    patient_ids: Optional[Union[List, Dict]] = None,
    interval: pd.Timedelta = pd.Timedelta(hours=1),
    verbose: bool = False,
    database: str = 'miiv',
    lab_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """一行代码加载实验室检验数据（适用于所有ICU数据库）
    
    Args:
        data_path: ICU 数据路径
        patient_ids: 患者ID列表
        interval: 时间间隔
        verbose: 是否显示详细信息
        database: 数据库类型
        lab_names: 要加载的实验室检验名称列表（可选）
    
    Returns:
        DataFrame，包含实验室检验数据
    """
    loader = ICUQuickLoader(data_path, database=database)
    
    if lab_names is None:
        # 默认加载常用实验室检验
        lab_names = ['wbc', 'hgb', 'plt', 'na', 'k', 'crea', 'bili']
    
    return loader.load_concepts(
        lab_names,
        patient_ids=patient_ids,
        interval=interval,
        verbose=verbose
    )


def load_mimic_labs(
    data_path: Union[str, Path],
    patient_ids: Optional[Union[List, Dict]] = None,
    interval: pd.Timedelta = pd.Timedelta(hours=1),
    verbose: bool = False,
    database: str = 'miiv',
    lab_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """一行代码加载 MIMIC 实验室检查结果
    
    Args:
        data_path: ICU 数据路径
        patient_ids: 患者ID列表
        interval: 时间间隔
        verbose: 是否显示详细信息
        database: 数据库类型
        lab_names: 指定实验室项目（可选），例如 ['lact', 'crea', 'bili', 'plt', 'wbc']
    
    Returns:
        DataFrame，包含实验室检查数据
    """
    loader = ICUQuickLoader(data_path, database=database)
    
    if lab_names is None:
        # 默认加载常用实验室指标
        lab_names = ['lact', 'crea', 'bili', 'plt', 'wbc', 'hb', 'po2', 'pco2', 'ph']
    
    return loader.load_concepts(
        lab_names,
        patient_ids=patient_ids,
        interval=interval,
        verbose=verbose
    )


def get_patient_ids(
    data_path: Union[str, Path],
    database: str = 'miiv',
    max_patients: Optional[int] = None
) -> List:
    """获取数据集中的患者ID列表（适用于所有ICU数据库）
    
    Args:
        data_path: ICU 数据路径
        database: 数据库类型
        max_patients: 最大患者数（可选，用于快速测试）
    
    Returns:
        患者ID列表
    
    Examples:
        >>> # 获取所有患者
        >>> all_patients = get_patient_ids('/path/to/icu_data', database='miiv')
        >>> 
        >>> # 获取前100个患者用于测试
        >>> test_patients = get_patient_ids('/path/to/icu_data', max_patients=100, database='eicu')
    """
    from .fst_reader import read_fst
    
    data_path = Path(data_path)
    
    # 尝试读取 icustays 表
    for fmt in ['fst', 'parquet', 'csv']:
        icustays_file = data_path / f'icustays.{fmt}'
        if icustays_file.exists():
            if fmt == 'fst':
                icustays = read_fst(icustays_file)
            elif fmt == 'parquet':
                icustays = pd.read_parquet(icustays_file)
            else:
                icustays = pd.read_csv(icustays_file)
            
            patient_ids = icustays['stay_id'].tolist()
            
            if max_patients:
                patient_ids = patient_ids[:max_patients]
            
            return patient_ids
    
    raise FileNotFoundError(f"Cannot find icustays table in {data_path}")


# 🔧 向后兼容的别名（保留旧名称）
load_mimic_sofa = load_sofa
load_mimic_sepsis3 = load_sepsis3
load_mimic_vitals = load_vitals
load_mimic_labs = load_labs
MIMICQuickLoader = ICUQuickLoader


__all__ = [
    # 主要API（新名称）
    'ICUQuickLoader',
    'load_sofa',
    'load_sepsis3',
    'load_vitals',
    'load_labs',
    'get_patient_ids',
    # 向后兼容的别名
    'MIMICQuickLoader',
    'load_mimic_sofa',
    'load_mimic_sepsis3',
    'load_mimic_vitals',
    'load_mimic_labs',
]





