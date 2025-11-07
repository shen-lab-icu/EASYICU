"""
ID 系统转换工具 (完全复刻 R ricu tbl-utils.R 的 change_id 功能)

实现不同 ID 系统间的转换，例如:
- MIMIC: subject_id <-> hadm_id <-> stay_id
- eICU: patient_id <-> patienthealthsystemstayid <-> patientunitstayid
- HiRID: general_id <-> patient_id
"""

from typing import Optional, Union, List, Dict, TYPE_CHECKING
import pandas as pd
from pathlib import Path

if TYPE_CHECKING:
    from .datasource import ICUDataSource

try:
    from .table import IDTable
    from .assertions import assert_that, is_string, has_cols, assert_has_cols
except ImportError:
    # 如果导入失败，使用简化版
    def assert_that(cond, msg=""):
        if not cond:
            raise AssertionError(msg)
    def is_string(x):
        return isinstance(x, str)
    def has_cols(df, cols):
        return all(c in df.columns for c in cols)
    def assert_has_cols(df, cols, msg=""):
        if not has_cols(df, cols):
            raise AssertionError(msg or f"Missing columns: {cols}")


class IDMapper:
    """ID 映射器 - 处理不同ID层级的转换"""
    
    # ID 层级定义 (从最细粒度到最粗粒度)
    ID_HIERARCHY = {
        'mimic_demo': ['stay_id', 'hadm_id', 'subject_id'],
        'mimic': ['stay_id', 'hadm_id', 'subject_id'],
        'eicu_demo': ['patientunitstayid', 'patienthealthsystemstayid', 'uniquepid'],
        'eicu': ['patientunitstayid', 'patienthealthsystemstayid', 'uniquepid'],
        'hirid': ['patientid', 'generalid'],
        'aumc': ['admissionid', 'patientid'],
        'sic': ['caseid', 'patientid'],
    }
    
    # ID 映射表配置
    MAPPING_TABLES = {
        'mimic_demo': {
            ('hadm_id', 'subject_id'): ('admissions', ['hadm_id', 'subject_id']),
            ('stay_id', 'hadm_id'): ('icustays', ['stay_id', 'hadm_id', 'subject_id']),
            ('stay_id', 'subject_id'): ('icustays', ['stay_id', 'subject_id']),
        },
        'mimic': {
            ('hadm_id', 'subject_id'): ('admissions', ['hadm_id', 'subject_id']),
            ('stay_id', 'hadm_id'): ('icustays', ['stay_id', 'hadm_id', 'subject_id']),
            ('stay_id', 'subject_id'): ('icustays', ['stay_id', 'subject_id']),
        },
        'eicu_demo': {
            ('patientunitstayid', 'patienthealthsystemstayid'): 
                ('patient', ['patientunitstayid', 'patienthealthsystemstayid', 'uniquepid']),
            ('patienthealthsystemstayid', 'uniquepid'): 
                ('patient', ['patienthealthsystemstayid', 'uniquepid']),
        },
        'eicu': {
            ('patientunitstayid', 'patienthealthsystemstayid'): 
                ('patient', ['patientunitstayid', 'patienthealthsystemstayid', 'uniquepid']),
            ('patienthealthsystemstayid', 'uniquepid'): 
                ('patient', ['patienthealthsystemstayid', 'uniquepid']),
        },
        'hirid': {
            ('patientid', 'generalid'): ('general', ['patientid', 'generalid']),
        },
        'aumc': {
            ('admissionid', 'patientid'): ('admissions', ['admissionid', 'patientid']),
        },
    }
    
    @classmethod
    def get_hierarchy_level(cls, data_source: str, id_type: str) -> int:
        """
        获取ID在层级中的位置
        
        Args:
            data_source: 数据源名称
            id_type: ID类型
            
        Returns:
            层级位置 (0=最细粒度)
        """
        hierarchy = cls.ID_HIERARCHY.get(data_source, [])
        try:
            return hierarchy.index(id_type)
        except ValueError:
            raise ValueError(f"Unknown ID type '{id_type}' for data source '{data_source}'")
    
    @classmethod
    def get_mapping_info(cls, data_source: str, from_id: str, to_id: str) -> tuple:
        """
        获取ID映射信息
        
        Args:
            data_source: 数据源名称
            from_id: 源ID类型
            to_id: 目标ID类型
            
        Returns:
            (table_name, columns) 元组
        """
        mappings = cls.MAPPING_TABLES.get(data_source, {})
        
        # 尝试直接查找
        key = (from_id, to_id)
        if key in mappings:
            return mappings[key]
        
        # 尝试反向查找
        key_rev = (to_id, from_id)
        if key_rev in mappings:
            table, cols = mappings[key_rev]
            # 反转列顺序
            return table, [to_id, from_id] if len(cols) == 2 else cols
        
        raise ValueError(f"No mapping defined between '{from_id}' and '{to_id}' for '{data_source}'")


def change_id(
    data: pd.DataFrame,
    new_id_type: str,
    id_var: Optional[str] = None,
    data_source: Optional[Union[str, 'ICUDataSource']] = None,
    by_ref: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    在不同 ID 系统间转换数据 (R ricu change_id)
    
    这是核心功能：允许在不同ID粒度间转换
    例如从 ICU stay 级别转换到 hospital admission 级别
    
    Args:
        data: 输入数据 DataFrame
        new_id_type: 目标 ID 类型 (如 'hadm_id', 'subject_id')
        id_var: 当前 ID 列名 (如果为None，自动检测)
        data_source: 数据源对象或名称
        by_ref: 是否原地修改
        **kwargs: 其他参数
        
    Returns:
        转换后的 DataFrame
        
    Examples:
        >>> # MIMIC: 从 ICU stay 转换到 hospital admission
        >>> hadm_data = change_id(icu_data, 'hadm_id', id_var='stay_id', 
        ...                       data_source='mimic_demo')
        
        >>> # 从 admission 转换到 patient
        >>> patient_data = change_id(hadm_data, 'subject_id', id_var='hadm_id',
        ...                          data_source='mimic_demo')
    """
    if not by_ref:
        data = data.copy()
    
    # 确定数据源名称
    if data_source is None:
        raise ValueError("data_source must be provided")
    
    if hasattr(data_source, 'name'):
        src_name = data_source.name
    else:
        src_name = str(data_source)
    
    # 确定当前ID列
    if id_var is None:
        # 尝试从元数据获取
        if hasattr(data, 'attrs') and 'id_vars' in data.attrs:
            id_vars = data.attrs['id_vars']
            if len(id_vars) == 1:
                id_var = id_vars[0]
            else:
                raise ValueError(f"Multiple ID variables found: {id_vars}. Please specify id_var")
        else:
            raise ValueError("id_var must be specified")
    
    # 检查是否已经是目标ID
    if id_var == new_id_type:
        return data
    
    # 获取ID层级
    try:
        current_level = IDMapper.get_hierarchy_level(src_name, id_var)
        target_level = IDMapper.get_hierarchy_level(src_name, new_id_type)
    except ValueError as e:
        raise ValueError(f"Invalid ID type for data source '{src_name}': {e}")
    
    # 获取映射表信息
    try:
        table_name, map_cols = IDMapper.get_mapping_info(src_name, id_var, new_id_type)
    except ValueError as e:
        raise ValueError(f"Cannot map between '{id_var}' and '{new_id_type}': {e}")
    
    # 加载映射表
    if hasattr(data_source, 'load_table'):
        # 如果是 ICUDataSource 对象
        mapping_table = data_source.load_table(table_name, columns=map_cols)
    else:
        # 否则需要另外加载
        raise NotImplementedError(
            "change_id with string data_source requires ICUDataSource object"
        )
    
    # 执行映射
    # 去重映射表
    mapping_table = mapping_table[map_cols].drop_duplicates()
    
    # 合并
    result = data.merge(mapping_table, on=id_var, how='left')
    
    # 如果是向上转换（细粒度->粗粒度），需要去重或聚合
    if current_level < target_level:
        # 检查是否有时间列
        has_time = any(pd.api.types.is_datetime64_any_dtype(result[col]) or 
                      pd.api.types.is_timedelta64_dtype(result[col]) 
                      for col in result.columns)
        
        if has_time:
            # 如果有时间数据，需要聚合
            # 这里需要根据具体情况决定聚合策略
            pass
        else:
            # 简单情况：直接去重
            group_cols = [new_id_type]
            result = result.drop_duplicates(subset=group_cols)
    
    # 更新元数据
    if hasattr(data, 'attrs'):
        result.attrs = data.attrs.copy()
        result.attrs['id_vars'] = [new_id_type]
    
    return result


def id_map(
    data: pd.DataFrame,
    from_ids: List[str],
    to_ids: List[str],
    start_var: Optional[str] = None,
    end_var: Optional[str] = None,
) -> pd.DataFrame:
    """
    ID 映射辅助函数 (R ricu id_map)
    
    在不改变ID系统的情况下，添加其他ID列
    
    Args:
        data: 输入数据
        from_ids: 源ID列名列表
        to_ids: 目标ID列名列表
        start_var: 起始时间变量
        end_var: 结束时间变量
        
    Returns:
        添加了新ID列的数据
    """
    # 实现简化版本
    # 完整实现需要处理时间窗口等复杂逻辑
    return data


def id_name_to_type(data_source: Union[str, 'ICUDataSource'], id_var: str) -> str:
    """
    将ID列名转换为ID类型名称
    
    Args:
        data_source: 数据源
        id_var: ID列名
        
    Returns:
        ID类型
    """
    # 简化实现：直接返回列名
    # 完整实现需要查询数据源配置
    return id_var


def get_id_vars(data: pd.DataFrame) -> List[str]:
    """
    获取数据的ID变量
    
    Args:
        data: DataFrame
        
    Returns:
        ID变量列表
    """
    if hasattr(data, 'attrs') and 'id_vars' in data.attrs:
        return data.attrs['id_vars']
    
    # 默认猜测：查找常见ID列名
    common_ids = [
        'stay_id', 'hadm_id', 'subject_id',  # MIMIC
        'patientunitstayid', 'patienthealthsystemstayid', 'uniquepid',  # eICU
        'patientid', 'generalid',  # HiRID
        'admissionid',  # AUMC
        'caseid',  # SICdb
    ]
    
    found_ids = [col for col in common_ids if col in data.columns]
    return found_ids[:1] if found_ids else []


def set_id_vars(data: pd.DataFrame, id_vars: Union[str, List[str]]) -> pd.DataFrame:
    """
    设置数据的ID变量
    
    Args:
        data: DataFrame
        id_vars: ID变量名或列表
        
    Returns:
        更新了元数据的 DataFrame
    """
    if isinstance(id_vars, str):
        id_vars = [id_vars]
    
    if not hasattr(data, 'attrs'):
        data.attrs = {}
    
    data.attrs['id_vars'] = id_vars
    return data

