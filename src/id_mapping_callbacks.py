
"""
ID映射回调函数

处理MIMIC-IV中不同表之间的ID列映射问题
"""

import pandas as pd
from typing import Dict, List, Any

# 全局ID映射表
_STAY_TO_SUBJECT = {}
_SUBJECT_TO_STAY = {}

def initialize_id_mappings(data_path: str):
    """初始化ID映射表"""
    global _STAY_TO_SUBJECT, _SUBJECT_TO_STAY

    try:
        icustays_df = pd.read_parquet(f"{data_path}/icustays.parquet")

        for _, row in icustays_df.iterrows():
            stay_id = row['stay_id']
            subject_id = row['subject_id']

            _STAY_TO_SUBJECT[stay_id] = subject_id
            _SUBJECT_TO_STAY[subject_id] = stay_id

        print(f"ID映射初始化完成: {len(_STAY_TO_SUBJECT)}个映射")

    except Exception as e:
        print(f"ID映射初始化失败: {e}")

def map_patient_ids(data, from_col: str, to_col: str, target_type: str = "stay_id") -> pd.DataFrame:
    """
    映射患者ID列

    Args:
        data: 输入数据
        from_col: 当前ID列名
        to_col: 目标ID列名
        target_type: 目标ID类型 ("stay_id" 或 "subject_id")

    Returns:
        映射后的DataFrame
    """
    if from_col not in data.columns:
        return data

    result = data.copy()

    def map_single_id(patient_id):
        """映射单个ID"""
        if target_type == "stay_id":
            return _SUBJECT_TO_STAY.get(patient_id, patient_id)
        else:  # subject_id
            return _STAY_TO_SUBJECT.get(patient_id, patient_id)

    # 应用映射
    result[to_col] = result[from_col].apply(map_single_id)

    return result

def bmi_callback(tables, ctx, **kwargs):
    """BMI计算回调"""
    try:
        # 获取height和weight数据
        height_data = tables.get('height')
        weight_data = tables.get('weight')

        if height_data is None or weight_data is None or height_data.empty or weight_data.empty:
            return ctx.create_empty_result()

        # 确保两个DataFrame使用相同的ID列
        if 'stay_id' in height_data.columns and 'subject_id' in weight_data.columns:
            weight_data = map_patient_ids(weight_data, 'subject_id', 'stay_id', 'stay_id')
        elif 'subject_id' in height_data.columns and 'stay_id' in weight_data.columns:
            height_data = map_patient_ids(height_data, 'subject_id', 'stay_id', 'stay_id')

        # 合并数据
        merged = pd.merge(height_data, weight_data, on='stay_id', how='inner', suffixes=('_height', '_weight'))

        if merged.empty:
            return ctx.create_empty_result()

        # 计算BMI
        merged['bmi'] = merged['weight'] / ((merged['height'] / 100) ** 2)

        # 创建结果
        result_data = merged[['stay_id', 'bmi']].copy()
        if 'charttime' in merged.columns:
            result_data['charttime'] = merged['charttime']

        return ctx.create_result(result_data)

    except Exception as e:
        print(f"BMI计算失败: {e}")
        return ctx.create_empty_result()

def qsofa_callback(tables, ctx, **kwargs):
    """qSOFA评分回调"""
    # 简化实现：基于HR、RR、SBP计算
    try:
        hr_data = tables.get('hr')
        resp_data = tables.get('resp')
        sbp_data = tables.get('sbp')

        # 这里应该实现完整的qSOFA计算逻辑
        # 为了简化，暂时返回空结果
        return ctx.create_empty_result()

    except Exception as e:
        print(f"qSOFA计算失败: {e}")
        return ctx.create_empty_result()

def sirs_callback(tables, ctx, **kwargs):
    """SIRS评分回调"""
    # 简化实现
    try:
        return ctx.create_empty_result()
    except Exception as e:
        print(f"SIRS计算失败: {e}")
        return ctx.create_empty_result()
