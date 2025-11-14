"""
ricu.R兼容性API

实现与R版本ricu完全相同的数据提取逻辑，包括：
1. 扩展时间窗口（包含ICU外数据）
2. 1小时间隔聚合
3. 宽格式输出
4. 相对时间系统
"""

from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .base import BaseICULoader, get_default_data_path, detect_database_type
from .api import _get_global_loader


class RicuCompatibleLoader:
    """ricu.R兼容的数据加载器"""

    def __init__(self, database: Optional[str] = None, data_path: Optional[Path] = None, **kwargs):
        """初始化ricu.R兼容加载器"""
        self.loader = _get_global_loader(database=database, data_path=data_path, **kwargs)
        self.database = self.loader.database
        self.data_path = self.loader.data_path

        # 加载基础表格
        self._load_basic_tables()

    def _load_basic_tables(self):
        """加载基础表格用于时间窗口计算"""
        try:
            # 使用pandas直接读取parquet文件
            if self.database in ['miiv', 'mimic_demo']:
                icustays_path = self.data_path / "icustays.parquet"
                if icustays_path.exists():
                    self.icustays_df = pd.read_parquet(icustays_path)
                    self.stay_col = 'stay_id'
                    self.subject_col = 'subject_id'
                else:
                    print(f"⚠️  icustays.parquet文件不存在: {icustays_path}")
                    self.icustays_df = pd.DataFrame()
                    self.stay_col = 'stay_id'
                    self.subject_col = 'subject_id'
            else:
                # 其他数据库的处理
                self.icustays_df = pd.DataFrame()
                self.stay_col = 'stay_id'
                self.subject_col = 'subject_id'

            print(f"✅ 基础表格加载完成 ({self.database}数据库)")
        except Exception as e:
            print(f"❌ 基础表格加载失败: {e}")
            self.icustays_df = pd.DataFrame()
            self.stay_col = 'stay_id'
            self.subject_col = 'subject_id'

    def _get_extended_time_window(self, patient_ids: List[int], window_hours: int = 2000) -> Dict[int, tuple]:
        """获取扩展时间窗口

        Args:
            patient_ids: 患者ID列表
            window_hours: 扩展窗口大小（小时）

        Returns:
            Dict[patient_id: (start_time, end_time, intime)]
        """
        time_windows = {}

        if self.icustays_df.empty:
            return time_windows

        for patient_id in patient_ids:
            # 查找患者对应的ICU停留信息
            if self.database in ['miiv', 'mimic_demo']:
                # MIMIC-IV需要转换stay_id到subject_id
                patient_stays = self.icustays_df[self.icustays_df[self.stay_col] == patient_id]
            else:
                patient_stays = self.icustays_df[self.icustays_df[self.stay_col] == patient_id]

            if not patient_stays.empty:
                stay_info = patient_stays.iloc[0]
                intime = pd.to_datetime(stay_info['intime'])

                # 扩展时间窗口：ICU入院前window_hours小时到入院后window_hours小时
                start_time = intime - timedelta(hours=window_hours)
                end_time = intime + timedelta(hours=window_hours)

                time_windows[patient_id] = (start_time, end_time, intime)

        return time_windows

    def load_concepts_ricu_style(
        self,
        concepts: Union[str, List[str]],
        patient_ids: Optional[List[int]] = None,
        interval: str = '1h',
        window_hours: int = 2000,
        merge: bool = False,
        verbose: bool = False
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        以ricu.R风格加载概念数据

        Args:
            concepts: 概念名称或列表
            patient_ids: 患者ID列表
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
            print(f"🔬 ricu.R风格加载概念: {', '.join(concepts)}")
            print(f"   时间间隔: {interval}")
            print(f"   扩展窗口: {window_hours}小时")

        if patient_ids is None:
            # 如果没有指定患者ID，获取所有患者
            if not self.icustays_df.empty:
                patient_ids = self.icustays_df[self.stay_col].unique()[:100].tolist()  # 限制数量避免内存问题
            else:
                patient_ids = []

        # 获取扩展时间窗口
        time_windows = self._get_extended_time_window(patient_ids, window_hours)

        if not time_windows:
            print("⚠️  无法获取时间窗口信息")
            return {}

        # 逐个概念加载
        concept_results = {}

        for concept in concepts:
            if verbose:
                print(f"  📊 加载概念: {concept}")

            try:
                # 使用扩展时间窗口加载数据
                concept_df = self._load_single_concept_extended(
                    concept, patient_ids, time_windows, interval, verbose
                )

                if not concept_df.empty:
                    concept_results[concept] = concept_df
                    if verbose:
                        print(f"    ✅ {concept}: {len(concept_df)}行")
                else:
                    if verbose:
                        print(f"    ⚠️  {concept}: 无数据")

            except Exception as e:
                if verbose:
                    print(f"    ❌ {concept}: {str(e)[:50]}")
                concept_results[concept] = pd.DataFrame()

        # 决定返回格式
        if merge:
            # 合并所有概念到一个DataFrame
            if concept_results:
                merged_result = self._merge_concepts_ricu_style(concept_results)
                return merged_result
            else:
                return pd.DataFrame()
        else:
            # 返回概念字典
            return concept_results

    def _load_single_concept_extended(
        self,
        concept: str,
        patient_ids: List[int],
        time_windows: Dict[int, tuple],
        interval: str,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        加载单个概念的扩展数据
        """
        try:
            # 使用原始加载器加载概念，但设置扩展时间窗口
            # 这里需要访问底层的数据加载逻辑
            result_dfs = []

            for patient_id in time_windows:
                start_time, end_time, intime = time_windows[patient_id]

                # 计算相对时间偏移
                time_offset = (start_time - intime).total_seconds() / 3600  # 小时

                try:
                    # 尝试使用load_concepts加载单个患者数据
                    patient_data = self.loader.load_concepts(
                        concepts=[concept],
                        patient_ids={self.stay_col: [patient_id]},
                        interval=interval,
                        win_length=f"{(end_time - start_time).total_seconds() / 3600:.0f}h",
                        merge=False,
                        verbose=False
                    )

                    if patient_data and concept in patient_data:
                        df = patient_data[concept]
                        if df is not None and not df.empty:
                            # 转换为相对时间
                            if hasattr(df, 'index') and hasattr(df.index, 'names'):
                                # MultiIndex情况
                                if 'index_time' in df.index.names:
                                    df_copy = df.reset_index()
                                    # 转换为相对于ICU入院的时间
                                    df_copy['relative_time'] = (
                                        pd.to_datetime(df_copy['index_time']) - intime
                                    ).dt.total_seconds() / 3600
                                    df_copy[self.stay_col] = patient_id
                                    result_dfs.append(df_copy)
                                else:
                                    # 处理其他索引结构
                                    df_copy = df.copy()
                                    df_copy[self.stay_col] = patient_id
                                    result_dfs.append(df_copy)
                            else:
                                # 普通DataFrame
                                df_copy = df.copy()
                                df_copy[self.stay_col] = patient_id
                                result_dfs.append(df_copy)

                except Exception as e:
                    if verbose:
                        print(f"      患者ID {patient_id} 加载失败: {str(e)[:30]}")
                    continue

            if result_dfs:
                # 合并所有患者数据
                combined_df = pd.concat(result_dfs, ignore_index=True)

                # 按时间间隔重新聚合到1小时间格
                if 'relative_time' in combined_df.columns:
                    combined_df['hour_bin'] = np.floor(combined_df['relative_time']).astype(int)
                    aggregated = combined_df.groupby([self.stay_col, 'hour_bin'])[concept].mean().reset_index()

                    # 创建宽格式输出
                    pivot_result = aggregated.pivot(
                        index=self.stay_col,
                        columns='hour_bin',
                        values=concept
                    )

                    return pivot_result

                return combined_df
            else:
                return pd.DataFrame()

        except Exception as e:
            if verbose:
                print(f"    ❌ 概念加载失败: {e}")
            return pd.DataFrame()

    def _merge_concepts_ricu_style(self, concept_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        以ricu.R风格合并多个概念
        """
        if not concept_results:
            return pd.DataFrame()

        # 获取所有唯一的时间点
        all_timepoints = set()
        for concept, df in concept_results.items():
            if not df.empty:
                # 获取时间点列（非索引列）
                time_cols = [col for col in df.columns if isinstance(col, (int, np.integer, float))]
                all_timepoints.update(time_cols)

        if not all_timepoints:
            return pd.DataFrame()

        # 排序时间点
        sorted_timepoints = sorted(all_timepoints)

        # 创建合并结果
        merged_df = None

        for concept, df in concept_results.items():
            if df.empty:
                continue

            # 确保所有时间点都存在
            df_expanded = df.copy()
            for tp in sorted_timepoints:
                if tp not in df_expanded.columns:
                    df_expanded[tp] = np.nan

            # 按时间点排序
            time_cols = [tp for tp in sorted_timepoints if tp in df_expanded.columns]
            df_sorted = df_expanded[[col for col in df_expanded.columns if col not in time_cols] + time_cols]

            if merged_df is None:
                merged_df = df_sorted
            else:
                # 使用外连接合并
                merged_df = pd.merge(
                    merged_df, df_sorted,
                    left_index=True,
                    right_index=True,
                    how='outer',
                    suffixes=('', f'_{concept}')
                )

        return merged_df


# 全局兼容加载器实例
_ricu_loader = None

def get_ricu_loader(database: Optional[str] = None, data_path: Optional[Path] = None, **kwargs) -> RicuCompatibleLoader:
    """获取全局ricu.R兼容加载器"""
    global _ricu_loader
    if _ricu_loader is None:
        _ricu_loader = RicuCompatibleLoader(database=database, data_path=data_path, **kwargs)
    return _ricu_loader


def load_concepts_ricu(
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
    以ricu.R风格加载概念数据

    Args:
        concepts: 概念名称或列表
        patient_ids: 患者ID列表
        database: 数据库类型
        data_path: 数据路径
        interval: 时间间隔（默认1小时，匹配ricu.R）
        window_hours: 扩展时间窗口（默认2000小时）
        merge: 是否合并结果
        verbose: 是否显示详细信息

    Returns:
        DataFrame或概念字典
    """
    loader = get_ricu_loader(database=database, data_path=data_path, **kwargs)
    return loader.load_concepts_ricu_style(
        concepts=concepts,
        patient_ids=patient_ids,
        interval=interval,
        window_hours=window_hours,
        merge=merge,
        verbose=verbose
    )


def load_lab_ricu(
    patient_ids: Optional[List[int]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    以ricu.R风格加载LAB模块

    Args:
        patient_ids: 患者ID列表
        database: 数据库类型
        data_path: 数据路径
        verbose: 是否显示详细信息

    Returns:
        LAB模块DataFrame（宽格式）
    """
    # LAB模块概念列表（基于ricu.R）
    lab_concepts = [
        'alb', 'alp', 'alt', 'ast', 'bicar', 'bili', 'bili_dir',
        'bun', 'ca', 'ck', 'ckmb', 'cl', 'crea', 'crp',
        'glu', 'k', 'mg', 'na', 'phos', 'tnt'
    ]

    return load_concepts_ricu(
        concepts=lab_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        window_hours=2000,  # 使用扩展窗口
        merge=True,
        verbose=verbose,
        **kwargs
    )


def load_vitals_ricu(
    patient_ids: Optional[List[int]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    以ricu.R风格加载VITALS模块

    Args:
        patient_ids: 患者ID列表
        database: 数据库类型
        data_path: 数据路径
        verbose: 是否显示详细信息

    Returns:
        VITALS模块DataFrame（宽格式）
    """
    # VITALS模块概念列表（基于ricu.R）
    vitals_concepts = ['dbp', 'etco2', 'hr', 'map', 'sbp', 'temp']

    return load_concepts_ricu(
        concepts=vitals_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        merge=True,
        verbose=verbose,
        **kwargs
    )