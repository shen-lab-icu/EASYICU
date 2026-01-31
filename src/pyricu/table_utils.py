"""
表工具函数 - 对应 R ricu 的 tbl-utils.R
提供表操作、聚合、排序等功能
"""
from typing import List, Optional, Union, Callable, Dict, Any
import pandas as pd
from datetime import timedelta

def rename_cols(
    df: pd.DataFrame,
    new: Union[Dict[str, str], List[str], Callable],
    old: Optional[List[str]] = None,
    skip_absent: bool = False,
    by_ref: bool = False
) -> pd.DataFrame:
    """
    重命名列 - 对应 R ricu rename_cols
    
    Args:
        df: DataFrame
        new: 新列名（字典、列表或函数）
        old: 旧列名列表
        skip_absent: 是否忽略不存在的列
        by_ref: 是否原地修改
        
    Returns:
        重命名后的 DataFrame
    """
    if not by_ref:
        df = df.copy()
    
    if callable(new):
        # 如果是函数，应用到列名
        new_names = {col: new(col) for col in (old if old else df.columns)}
    elif isinstance(new, dict):
        new_names = new
    elif isinstance(new, list):
        if old is None:
            old = list(df.columns[:len(new)])
        new_names = dict(zip(old, new))
    else:
        raise ValueError(f"Unsupported type for new: {type(new)}")
    
    if skip_absent:
        new_names = {k: v for k, v in new_names.items() if k in df.columns}
    
    return df.rename(columns=new_names)

def rm_cols(
    df: pd.DataFrame,
    cols: Union[str, List[str]],
    skip_absent: bool = False,
    by_ref: bool = False
) -> pd.DataFrame:
    """
    删除列 - 对应 R ricu rm_cols
    
    Args:
        df: DataFrame
        cols: 要删除的列名
        skip_absent: 是否忽略不存在的列
        by_ref: 是否原地修改
        
    Returns:
        删除列后的 DataFrame
    """
    if not by_ref:
        df = df.copy()
    
    if isinstance(cols, str):
        cols = [cols]
    
    if skip_absent:
        cols = [col for col in cols if col in df.columns]
    
    return df.drop(columns=cols, errors='ignore' if skip_absent else 'raise')

def rm_na(
    df: pd.DataFrame,
    cols: Optional[Union[str, List[str]]] = None,
    mode: str = 'all'
) -> pd.DataFrame:
    """
    删除包含 NA 的行 - 对应 R ricu rm_na
    
    Args:
        df: DataFrame
        cols: 要检查的列（None 表示所有列）
        mode: 'all' 或 'any'
        
    Returns:
        删除 NA 后的 DataFrame
    """
    if cols is None:
        cols = df.columns.tolist()
    elif isinstance(cols, str):
        cols = [cols]
    
    if mode == 'all':
        # 只有当所有指定列都是 NA 时才删除
        return df[~df[cols].isna().all(axis=1)]
    else:  # mode == 'any'
        # 任何指定列是 NA 就删除
        return df.dropna(subset=cols)

def is_sorted(
    df: pd.DataFrame,
    by: Optional[Union[str, List[str]]] = None,
    ascending: bool = True
) -> bool:
    """
    检查 DataFrame 是否已排序 - 对应 R ricu is_sorted
    
    Args:
        df: DataFrame
        by: 排序列（None 表示检查所有列）
        ascending: 是否升序
        
    Returns:
        是否已排序
    """
    if by is None:
        by = df.columns.tolist()
    elif isinstance(by, str):
        by = [by]
    
    if len(df) <= 1:
        return True
    
    # 检查是否已排序
    sorted_df = df[by].sort_values(by=by, ascending=ascending)
    return df[by].equals(sorted_df)

def sort_by(
    df: pd.DataFrame,
    by: Union[str, List[str]],
    ascending: bool = True,
    by_ref: bool = False
) -> pd.DataFrame:
    """
    排序 DataFrame - 对应 R ricu sort
    
    Args:
        df: DataFrame
        by: 排序列
        ascending: 是否升序
        by_ref: 是否原地修改
        
    Returns:
        排序后的 DataFrame
    """
    if isinstance(by, str):
        by = [by]
    
    if by_ref:
        df.sort_values(by=by, ascending=ascending, inplace=True)
        return df
    else:
        return df.sort_values(by=by, ascending=ascending)

def unique_rows(
    df: pd.DataFrame,
    by: Optional[Union[str, List[str]]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    保留唯一行 - 对应 R ricu unique
    
    Args:
        df: DataFrame
        by: 用于判断唯一性的列（None 表示所有列）
        keep: 'first', 'last' 或 False
        
    Returns:
        去重后的 DataFrame
    """
    if by is None:
        return df.drop_duplicates(keep=keep)
    else:
        if isinstance(by, str):
            by = [by]
        return df.drop_duplicates(subset=by, keep=keep)

def is_unique(
    df: pd.DataFrame,
    by: Optional[Union[str, List[str]]] = None
) -> bool:
    """
    检查行是否唯一 - 对应 R ricu is_unique
    
    Args:
        df: DataFrame
        by: 用于判断唯一性的列
        
    Returns:
        是否唯一
    """
    if by is None:
        return not df.duplicated().any()
    else:
        if isinstance(by, str):
            by = [by]
        return not df.duplicated(subset=by).any()

def aggregate_by(
    df: pd.DataFrame,
    by: Union[str, List[str]],
    agg_dict: Optional[Dict[str, Union[str, Callable]]] = None,
    agg_func: Union[str, Callable] = 'mean'
) -> pd.DataFrame:
    """
    聚合数据 - 对应 R ricu aggregate
    
    Args:
        df: DataFrame
        by: 分组列
        agg_dict: 列到聚合函数的映射
        agg_func: 默认聚合函数
        
    Returns:
        聚合后的 DataFrame
    """
    if isinstance(by, str):
        by = [by]
    
    if agg_dict is None:
        # 对所有非分组列应用默认函数
        agg_dict = {col: agg_func for col in df.columns if col not in by}
    
    return df.groupby(by, as_index=False).agg(agg_dict)

def dt_gforce(
    df: pd.DataFrame,
    fun: str,
    by: Union[str, List[str]],
    vars: Optional[Union[str, List[str]]] = None,
    na_rm: bool = True
) -> pd.DataFrame:
    """
    使用 GForce 优化的聚合 - 对应 R ricu dt_gforce
    
    Args:
        df: DataFrame
        fun: 聚合函数名 ('mean', 'median', 'min', 'max', 'sum', 'first', 'last')
        by: 分组列
        vars: 要聚合的列
        na_rm: 是否忽略 NA
        
    Returns:
        聚合后的 DataFrame
    """
    if isinstance(by, str):
        by = [by]
    
    if vars is None:
        vars = [col for col in df.columns if col not in by]
    elif isinstance(vars, str):
        vars = [vars]
    
    # 映射函数名到 pandas 方法
    func_map = {
        'mean': 'mean',
        'median': 'median',
        'min': 'min',
        'max': 'max',
        'sum': 'sum',
        'prod': 'prod',
        'var': 'var',
        'std': 'std',
        'first': 'first',
        'last': 'last',
        'any': 'any',
        'all': 'all',
    }
    
    if fun not in func_map:
        raise ValueError(f"Unsupported function: {fun}")
    
    # 构建聚合字典
    agg_dict = {var: func_map[fun] for var in vars}
    
    grouped = df.groupby(by, as_index=False)
    
    if na_rm and fun not in ['first', 'last', 'any', 'all']:
        # 大多数函数有 skipna 参数
        result = grouped.agg(agg_dict)
    else:
        result = grouped.agg(agg_dict)
    
    return result

def replace_na(
    df: pd.DataFrame,
    value: Any = None,
    method: Optional[str] = None,
    cols: Optional[Union[str, List[str]]] = None,
    by_ref: bool = False
) -> pd.DataFrame:
    """
    替换 NA 值 - 对应 R ricu replace_na
    
    Args:
        df: DataFrame
        value: 替换值
        method: 填充方法 ('ffill', 'bfill', 'const')
        cols: 要处理的列
        by_ref: 是否原地修改
        
    Returns:
        替换 NA 后的 DataFrame
    """
    if not by_ref:
        df = df.copy()
    
    if cols is None:
        cols = df.columns.tolist()
    elif isinstance(cols, str):
        cols = [cols]
    
    if method == 'ffill':
        df[cols] = df[cols].fillna(method='ffill')
    elif method == 'bfill':
        df[cols] = df[cols].fillna(method='bfill')
    else:  # const or None
        if value is not None:
            df[cols] = df[cols].fillna(value)
    
    return df

def change_interval(
    df: pd.DataFrame,
    new_interval: timedelta,
    time_col: str = 'time',
    by_ref: bool = False
) -> pd.DataFrame:
    """
    更改时间间隔 - 对应 R ricu change_interval
    
    Args:
        df: DataFrame
        new_interval: 新的时间间隔
        time_col: 时间列名
        by_ref: 是否原地修改
        
    Returns:
        更改间隔后的 DataFrame
    """
    if not by_ref:
        df = df.copy()
    
    if time_col not in df.columns:
        return df
    
    # 将时间舍入到新间隔
    from .ts_utils import round_to_interval
    df[time_col] = round_to_interval(df[time_col], new_interval)
    
    return df

def expand_window(
    df: pd.DataFrame,
    start_var: str,
    end_var: str,
    interval: timedelta,
    keep_vars: Optional[List[str]] = None,
    by: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    展开时间窗口 - 对应 R ricu expand
    
    将窗口数据展开为时间序列数据。
    
    Args:
        df: DataFrame
        start_var: 起始时间列
        end_var: 结束时间列
        interval: 时间间隔
        keep_vars: 要保留的额外列
        by: 分组列
        
    Returns:
        展开后的 DataFrame
    """
    if by is None:
        by = []
    elif isinstance(by, str):
        by = [by]
    
    if keep_vars is None:
        keep_vars = []
    
    results = []
    
    for idx, row in df.iterrows():
        start = row[start_var]
        end = row[end_var]
        
        # 生成时间序列
        times = pd.timedelta_range(start=start, end=end, freq=interval)
        
        # 为每个时间点创建一行
        for t in times:
            new_row = {start_var: t}
            
            # 复制分组列
            for col in by:
                new_row[col] = row[col]
            
            # 复制保留列
            for col in keep_vars:
                if col in row:
                    new_row[col] = row[col]
            
            results.append(new_row)
    
    return pd.DataFrame(results)

# ============================================================================
# ID System Conversion Functions (R ricu change_id, upgrade_id, downgrade_id)
# ============================================================================

def change_id(
    df: pd.DataFrame,
    target_id: str,
    id_map: pd.DataFrame,
    src_id: Optional[str] = None,
    time_cols: Optional[List[str]] = None,
    keep_old_id: bool = True,
    by_ref: bool = False
) -> pd.DataFrame:
    """
    更改 ID 系统 - 对应 R ricu change_id
    
    在不同的 ID 层级之间转换（如 icustay_id <-> hadm_id <-> subject_id）。
    
    Args:
        df: 数据 DataFrame
        target_id: 目标 ID 列名
        id_map: ID 映射表，包含源 ID、目标 ID 和可选的时间偏移
        src_id: 源 ID 列名（如果为 None，自动检测）
        time_cols: 需要调整的时间列
        keep_old_id: 是否保留原 ID 列
        by_ref: 是否原地修改
        
    Returns:
        转换后的 DataFrame
        
    Examples:
        >>> # 从 icustay_id 升级到 hadm_id
        >>> id_map = pd.DataFrame({
        ...     'icustay_id': [1, 2, 3],
        ...     'hadm_id': [100, 100, 101],
        ...     'intime': [pd.Timedelta(0), pd.Timedelta(hours=5), pd.Timedelta(0)]
        ... })
        >>> data = pd.DataFrame({
        ...     'icustay_id': [1, 1, 2],
        ...     'time': [pd.Timedelta(hours=1), pd.Timedelta(hours=2), pd.Timedelta(hours=1)],
        ...     'value': [10, 20, 30]
        ... })
        >>> change_id(data, 'hadm_id', id_map, time_cols=['time'])
    """
    if not by_ref:
        df = df.copy()
    
    # 自动检测源 ID
    if src_id is None:
        # 查找 id_map 中存在的 ID 列
        for col in df.columns:
            if col in id_map.columns and col != target_id:
                src_id = col
                break
        
        if src_id is None:
            raise ValueError("Cannot detect source ID column in DataFrame")
    
    # 检查是升级还是降级
    # 简单逻辑：如果 src_id 到 target_id 是多对一，则为降级；一对多则为升级
    id_cardinality = id_map.groupby(src_id)[target_id].nunique()
    
    if (id_cardinality > 1).any():
        # 升级：一个 src_id 对应多个 target_id
        return upgrade_id(df, target_id, id_map, src_id, time_cols, keep_old_id, by_ref=True)
    else:
        # 降级：多个 src_id 对应一个 target_id
        return downgrade_id(df, target_id, id_map, src_id, time_cols, keep_old_id, by_ref=True)

def upgrade_id(
    df: pd.DataFrame,
    target_id: str,
    id_map: pd.DataFrame,
    src_id: str,
    time_cols: Optional[List[str]] = None,
    keep_old_id: bool = True,
    by_ref: bool = False
) -> pd.DataFrame:
    """
    升级 ID - 对应 R ricu upgrade_id
    
    从更高粒度 ID 转换到更低粒度 ID（如 icustay_id -> hadm_id）。
    这通常涉及到合并多个细粒度记录。
    
    Args:
        df: 数据 DataFrame
        target_id: 目标 ID 列名（更低粒度）
        id_map: ID 映射表
        src_id: 源 ID 列名（更高粒度）
        time_cols: 需要调整的时间列
        keep_old_id: 是否保留原 ID 列
        by_ref: 是否原地修改
        
    Returns:
        升级后的 DataFrame
    """
    if not by_ref:
        df = df.copy()
    
    if time_cols is None:
        time_cols = []
    
    # 准备映射表
    map_cols = [src_id, target_id]
    
    # 检查是否有时间偏移列
    time_offset_col = None
    for col in id_map.columns:
        if col not in [src_id, target_id] and pd.api.types.is_timedelta64_dtype(id_map[col]):
            time_offset_col = col
            map_cols.append(col)
            break
    
    # 合并映射表
    merge_map = id_map[map_cols].drop_duplicates()
    df = df.merge(merge_map, on=src_id, how='left')
    
    # 调整时间列（从 src_id 时间转换为 target_id 时间）
    if time_offset_col and time_cols:
        for time_col in time_cols:
            if time_col in df.columns:
                # 减去偏移量（因为是升级，时间参考点改变）
                df[time_col] = df[time_col] - df[time_offset_col]
        
        # 删除临时偏移列
        df = df.drop(columns=[time_offset_col])
    
    # 删除旧 ID 列
    if not keep_old_id:
        df = df.drop(columns=[src_id])
    
    return df

def downgrade_id(
    df: pd.DataFrame,
    target_id: str,
    id_map: pd.DataFrame,
    src_id: str,
    time_cols: Optional[List[str]] = None,
    keep_old_id: bool = True,
    by_ref: bool = False
) -> pd.DataFrame:
    """
    降级 ID - 对应 R ricu downgrade_id
    
    从更低粒度 ID 转换到更高粒度 ID（如 hadm_id -> icustay_id）。
    这通常会扩展数据，因为一个低粒度 ID 可能对应多个高粒度 ID。
    
    Args:
        df: 数据 DataFrame
        target_id: 目标 ID 列名（更高粒度）
        id_map: ID 映射表
        src_id: 源 ID 列名（更低粒度）
        time_cols: 需要调整的时间列
        keep_old_id: 是否保留原 ID 列
        by_ref: 是否原地修改
        
    Returns:
        降级后的 DataFrame
    """
    if not by_ref:
        df = df.copy()
    
    if time_cols is None:
        time_cols = []
    
    # 准备映射表
    map_cols = [src_id, target_id]
    
    # 检查是否有时间偏移列
    time_offset_col = None
    for col in id_map.columns:
        if col not in [src_id, target_id] and pd.api.types.is_timedelta64_dtype(id_map[col]):
            time_offset_col = col
            map_cols.append(col)
            break
    
    # 合并映射表（可能会扩展行数）
    merge_map = id_map[map_cols].drop_duplicates()
    df = df.merge(merge_map, on=src_id, how='left')
    
    # 调整时间列（从 src_id 时间转换为 target_id 时间）
    if time_offset_col and time_cols:
        for time_col in time_cols:
            if time_col in df.columns:
                # 加上偏移量（因为是降级，时间参考点改变）
                df[time_col] = df[time_col] + df[time_offset_col]
        
        # 删除临时偏移列
        df = df.drop(columns=[time_offset_col])
    
    # 删除旧 ID 列
    if not keep_old_id:
        df = df.drop(columns=[src_id])
    
    return df

def id_map_helper(
    src_config,
    from_id: str,
    to_id: str,
    time_offset_col: Optional[str] = None,
    index_col: Optional[str] = None
) -> pd.DataFrame:
    """
    创建 ID 映射表辅助函数 - 对应 R ricu id_map
    
    Args:
        src_config: 数据源配置
        from_id: 源 ID 列名
        to_id: 目标 ID 列名
        time_offset_col: 时间偏移列名
        index_col: 时间索引列名
        
    Returns:
        ID 映射 DataFrame
    """
    # 这里需要从数据源加载相关表
    # 简化实现，实际应该从配置中获取正确的表
    from .data_env import load_table
    
    # 通常 ID 映射存储在特定的表中（如 icustays, admissions 等）
    # 这里简化处理
    try:
        # 尝试加载包含两个 ID 的表
        id_table = load_table(src_config, from_id, to_id)
        
        cols = [from_id, to_id]
        if time_offset_col and time_offset_col in id_table.columns:
            cols.append(time_offset_col)
        if index_col and index_col in id_table.columns:
            cols.append(index_col)
        
        return id_table[cols].drop_duplicates()
    except Exception as e:
        raise ValueError(f"Cannot create ID map from {from_id} to {to_id}: {e}")

# ============================================================================
# Time adjustment functions for ID changes
# ============================================================================

def adjust_time_for_id_change(
    df: pd.DataFrame,
    time_col: str,
    offset_col: str,
    direction: str = 'upgrade',  # 'upgrade' or 'downgrade'
    by_ref: bool = False
) -> pd.DataFrame:
    """
    调整时间列以适应 ID 变更 - pyricu 辅助函数
    
    当改变 ID 系统时，时间参考点会改变，需要相应调整时间值。
    
    Args:
        df: DataFrame
        time_col: 时间列名
        offset_col: 偏移量列名
        direction: 'upgrade' (减去偏移) 或 'downgrade' (加上偏移)
        by_ref: 是否原地修改
        
    Returns:
        调整后的 DataFrame
    """
    if not by_ref:
        df = df.copy()
    
    if time_col not in df.columns or offset_col not in df.columns:
        return df
    
    if direction == 'upgrade':
        df[time_col] = df[time_col] - df[offset_col]
    else:  # downgrade
        df[time_col] = df[time_col] + df[offset_col]
    
    return df

def set_id_var(
    df: pd.DataFrame,
    id_var: Union[str, List[str]],
    by_ref: bool = False
) -> pd.DataFrame:
    """
    设置 ID 变量属性 - 对应 R ricu set_id_vars
    
    在 pyricu 中，我们使用 DataFrame 属性来存储元数据。
    
    Args:
        df: DataFrame
        id_var: ID 列名（单个或列表）
        by_ref: 是否原地修改
        
    Returns:
        设置属性后的 DataFrame
    """
    if not by_ref:
        df = df.copy()
    
    if isinstance(id_var, str):
        id_var = [id_var]
    
    # 存储为 DataFrame 属性
    df.attrs['id_vars'] = id_var
    
    return df

def get_id_var(df: pd.DataFrame) -> Optional[Union[str, List[str]]]:
    """
    获取 ID 变量 - 对应 R ricu id_vars
    
    Args:
        df: DataFrame
        
    Returns:
        ID 列名（单个或列表）
    """
    return df.attrs.get('id_vars', None)

def set_index_var(
    df: pd.DataFrame,
    index_var: str,
    by_ref: bool = False
) -> pd.DataFrame:
    """
    设置索引变量属性 - 对应 R ricu set_index_var
    
    Args:
        df: DataFrame
        index_var: 索引列名
        by_ref: 是否原地修改
        
    Returns:
        设置属性后的 DataFrame
    """
    if not by_ref:
        df = df.copy()
    
    df.attrs['index_var'] = index_var
    
    return df

def get_index_var(df: pd.DataFrame) -> Optional[str]:
    """
    获取索引变量 - 对应 R ricu index_var
    
    Args:
        df: DataFrame
        
    Returns:
        索引列名
    """
    return df.attrs.get('index_var', None)

def meta_vars(df: pd.DataFrame) -> List[str]:
    """
    获取元数据变量（ID + 索引）- 对应 R ricu meta_vars
    
    Args:
        df: DataFrame
        
    Returns:
        元数据列名列表
    """
    vars_list = []
    
    id_vars = get_id_var(df)
    if id_vars:
        if isinstance(id_vars, str):
            vars_list.append(id_vars)
        else:
            vars_list.extend(id_vars)
    
    index_var = get_index_var(df)
    if index_var:
        vars_list.append(index_var)
    
    return vars_list

def data_vars(df: pd.DataFrame) -> List[str]:
    """
    获取数据变量（非元数据列）- 对应 R ricu data_vars
    
    Args:
        df: DataFrame
        
    Returns:
        数据列名列表
    """
    meta = meta_vars(df)
    return [col for col in df.columns if col not in meta]

def time_vars(df: pd.DataFrame) -> List[str]:
    """
    获取时间变量 - 对应 R ricu time_vars
    
    Args:
        df: DataFrame
        
    Returns:
        时间列名列表
    """
    time_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or \
           pd.api.types.is_timedelta64_dtype(df[col]):
            time_cols.append(col)
    return time_cols
