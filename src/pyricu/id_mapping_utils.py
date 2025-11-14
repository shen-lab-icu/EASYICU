
"""
ID映射工具函数

解决MIMIC-IV中不同表之间的ID列映射问题
"""

import pandas as pd
from typing import Dict, List, Any, Optional

class IDMapper:
    """ID映射器"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.stay_to_subject = {}
        self.subject_to_stay = {}
        self.initialize_mappings()

    def initialize_mappings(self):
        """初始化ID映射"""
        try:
            icustays_df = pd.read_parquet(f"{self.data_path}/icustays.parquet")
            self.stay_to_subject = dict(zip(icustays_df['stay_id'], icustays_df['subject_id']))
            self.subject_to_stay = dict(zip(icustays_df['subject_id'], icustays_df['stay_id']))
            print(f"ID映射初始化完成: {len(self.stay_to_subject)}个映射")
        except Exception as e:
            print(f"ID映射初始化失败: {e}")

    def convert_to_stay_id(self, df: pd.DataFrame, id_col: str = 'subject_id') -> pd.DataFrame:
        """将subject_id转换为stay_id"""
        if id_col not in df.columns:
            return df

        df_copy = df.copy()
        df_copy['stay_id'] = df_copy[id_col].map(self.subject_to_stay)

        # 移除无法映射的行
        df_copy = df_copy.dropna(subset=['stay_id'])
        df_copy['stay_id'] = df_copy['stay_id'].astype(int)

        return df_copy

    def convert_to_subject_id(self, df: pd.DataFrame, id_col: str = 'stay_id') -> pd.DataFrame:
        """将stay_id转换为subject_id"""
        if id_col not in df.columns:
            return df

        df_copy = df.copy()
        df_copy['subject_id'] = df_copy[id_col].map(self.stay_to_subject)

        # 移除无法映射的行
        df_copy = df_copy.dropna(subset=['subject_id'])
        df_copy['subject_id'] = df_copy['subject_id'].astype(int)

        return df_copy

def safe_merge_dataframes(dfs: List[pd.DataFrame],
                         id_col: str = 'stay_id',
                         how: str = 'outer') -> pd.DataFrame:
    """安全地合并多个DataFrame，处理ID列冲突"""
    if not dfs:
        return pd.DataFrame()

    if len(dfs) == 1:
        return dfs[0].copy()

    # 确保所有DataFrame都有ID列
    merged = dfs[0].copy()

    for df in dfs[1:]:
        if df.empty:
            continue

        # 检查ID列
        if id_col not in df.columns:
            print(f"⚠️  DataFrame缺少ID列{id_col}，跳过合并")
            continue

        # 处理重复列名
        common_cols = set(merged.columns) & set(df.columns)
        rename_cols = {col: f"{col}_dup_{len(merged.columns)}" for col in common_cols if col != id_col}

        if rename_cols:
            df_renamed = df.rename(columns=rename_cols)
        else:
            df_renamed = df

        # 合并
        try:
            merged = pd.merge(merged, df_renamed, on=id_col, how=how, suffixes=('', '_dup'))
        except Exception as e:
            print(f"⚠️  合并失败: {e}，使用concat")
            merged = pd.concat([merged, df_renamed], ignore_index=True)

    return merged
