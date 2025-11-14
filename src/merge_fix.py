
"""
概念合并修复补丁

修复不同ID列概念无法合并的问题
"""

import pandas as pd
from typing import List, Dict, Any

def fix_concept_merge(concepts_data: Dict[str, pd.DataFrame],
                     patient_mappings: Dict[int, int]) -> Dict[str, pd.DataFrame]:
    """
    修复概念数据合并

    Args:
        concepts_data: 概念数据字典
        patient_mappings: 患者ID映射 {subject_id: stay_id}

    Returns:
        修复后的概念数据字典
    """
    fixed_data = {}

    for concept_name, df in concepts_data.items():
        if df.empty:
            fixed_data[concept_name] = df
            continue

        # 检查ID列类型
        if 'subject_id' in df.columns:
            # 需要转换为stay_id
            df_fixed = df.copy()
            df_fixed['stay_id'] = df_fixed['subject_id'].map(patient_mappings)

            # 移除无法映射的行
            df_fixed = df_fixed.dropna(subset=['stay_id'])
            df_fixed['stay_id'] = df_fixed['stay_id'].astype(int)

            # 移除subject_id列
            df_fixed = df_fixed.drop(columns=['subject_id'])

            fixed_data[concept_name] = df_fixed
        else:
            fixed_data[concept_name] = df

    return fixed_data
