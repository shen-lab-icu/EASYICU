"""Demo and mock cohort data helpers for the EasyICU webapp."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from easyicu.webapp.mock_data import generate_mock_data


# ============ 辅助函数：获取完整的 mock_params（包含最新的 cohort_filter） ============
def get_mock_params_with_cohort():
    """
    获取完整的 mock_params，包含最新的 cohort_filter。

    由于 Streamlit 的渲染顺序，Step 1 (数据源) 在 Step 2 (队列筛选) 之前执行，
    所以 mock_params 中的 cohort_filter 可能不是最新的。

    此函数确保在调用 generate_mock_data 时使用最新的 cohort_filter。
    """
    params = st.session_state.get('mock_params', {'n_patients': 100, 'hours': 72}).copy()

    # 如果启用了队列筛选，添加最新的 cohort_filter
    if st.session_state.get('cohort_enabled', False):
        cohort_filter = st.session_state.get('cohort_filter', None)
        if cohort_filter:
            params['cohort_filter'] = cohort_filter

    return params


def _generate_mock_demographics(n_patients: int, lang: str = 'en') -> pd.DataFrame:
    """生成模拟的人口统计学数据用于Cohort Comparison演示。

    🔧 改进：复用 generate_mock_data 的逻辑，确保数据一致性。

    Args:
        n_patients: 患者数量
        lang: 语言

    Returns:
        包含人口统计学数据的DataFrame
    """
    # 🔧 使用统一的 generate_mock_data 函数生成基础数据
    # 注意：generate_mock_data 返回 (data_dict, patient_ids) 元组
    mock_data_tuple = generate_mock_data(n_patients=n_patients, hours=72)
    mock_data = mock_data_tuple[0] if isinstance(mock_data_tuple, tuple) else mock_data_tuple

    # 提取需要的人口统计学字段
    age_df = mock_data.get('age', pd.DataFrame(columns=['stay_id', 'age']))
    sex_df = mock_data.get('sex', pd.DataFrame(columns=['stay_id', 'sex']))
    death_df = mock_data.get('death', pd.DataFrame(columns=['stay_id', 'death']))
    los_icu_df = mock_data.get('los_icu', pd.DataFrame(columns=['stay_id', 'los_icu']))
    sofa_df = mock_data.get('sofa', pd.DataFrame(columns=['stay_id', 'time', 'sofa']))

    # 创建基础 DataFrame
    patient_ids = age_df['stay_id'].tolist() if 'stay_id' in age_df.columns else list(range(1, n_patients + 1))

    df = pd.DataFrame({'stay_id': patient_ids})

    # 合并年龄
    if not age_df.empty and 'age' in age_df.columns:
        df = df.merge(age_df[['stay_id', 'age']], on='stay_id', how='left')
    else:
        df['age'] = np.clip(np.random.normal(65, 15, len(df)), 18, 95).astype(int)

    # 合并性别
    if not sex_df.empty and 'sex' in sex_df.columns:
        df = df.merge(sex_df[['stay_id', 'sex']], on='stay_id', how='left')
        df['gender'] = df['sex']
    else:
        df['gender'] = np.random.choice(['M', 'F'], len(df), p=[0.55, 0.45])

    # 合并死亡状态
    if not death_df.empty and 'death' in death_df.columns:
        df = df.merge(death_df[['stay_id', 'death']], on='stay_id', how='left')
        df['survived'] = (1 - df['death']).astype(int)
    else:
        df['survived'] = np.random.choice([0, 1], len(df), p=[0.15, 0.85])

    # 合并LOS
    if not los_icu_df.empty and 'los_icu' in los_icu_df.columns:
        df = df.merge(los_icu_df[['stay_id', 'los_icu']], on='stay_id', how='left')
        df['los_days'] = df['los_icu']
        df['los_hours'] = (df['los_icu'] * 24).astype(int)
    else:
        df['los_hours'] = np.clip(np.random.lognormal(4.5, 0.8, len(df)), 24, 1000).astype(int)
        df['los_days'] = df['los_hours'] / 24

    # 计算 SOFA max
    if not sofa_df.empty and 'sofa' in sofa_df.columns:
        sofa_max = sofa_df.groupby('stay_id')['sofa'].max().reset_index()
        sofa_max.columns = ['stay_id', 'sofa_max']
        df = df.merge(sofa_max, on='stay_id', how='left')
        df['sofa_max'] = df['sofa_max'].fillna(0).astype(int)
    else:
        df['sofa_max'] = np.random.choice(range(0, 20), len(df))

    # 首次ICU入住
    df['first_icu_stay'] = np.random.choice([True, False], len(df), p=[0.65, 0.35])

    # 选择需要的列
    result_cols = ['stay_id', 'age', 'gender', 'los_hours', 'los_days', 'first_icu_stay', 'survived', 'sofa_max']
    available_cols = [c for c in result_cols if c in df.columns]

    return df[available_cols]


def _build_mock_group_feature_data(patient_ids: list, concepts: list, id_col: str = 'stay_id') -> Dict[str, pd.DataFrame]:
    """Build realistic demo feature data for cohort comparison.

    Prefer aggregating from generate_mock_data() so demo comparisons use the same
    clinical ranges as the rest of the web demo, especially for SOFA-related concepts.
    """
    patient_ids = [int(pid) for pid in patient_ids]
    if not patient_ids or not concepts:
        return {}

    mock_data_tuple = generate_mock_data(n_patients=max(len(patient_ids), 10), hours=72)
    mock_data = mock_data_tuple[0] if isinstance(mock_data_tuple, tuple) else mock_data_tuple

    age_df = mock_data.get('age', pd.DataFrame(columns=['stay_id']))
    source_ids = sorted(age_df['stay_id'].dropna().astype(int).unique().tolist()) if 'stay_id' in age_df.columns else []
    if not source_ids:
        source_ids = list(range(1, len(patient_ids) + 1))
    id_map = {src_id: patient_ids[idx] for idx, src_id in enumerate(source_ids[:len(patient_ids)])}

    fallback_specs = {
        'hr': (80, 15, 35, 180, False),
        'sbp': (120, 20, 70, 220, False),
        'dbp': (70, 12, 30, 140, False),
        'map': (85, 15, 45, 160, False),
        'resp': (18, 4, 8, 45, False),
        'temp': (37.0, 0.6, 34.0, 41.5, False),
        'spo2': (96, 3, 70, 100, False),
        'o2sat': (96, 3, 70, 100, False),
        'glu': (120, 40, 40, 450, False),
        'na': (140, 4, 118, 165, False),
        'k': (4.2, 0.5, 2.2, 7.0, False),
        'crea': (1.2, 0.8, 0.2, 8.0, False),
        'bili': (1.5, 2.0, 0.1, 20.0, False),
        'lact': (1.5, 1.0, 0.2, 12.0, False),
        'hgb': (11, 2, 5, 19, False),
        'plt': (200, 80, 10, 600, False),
        'wbc': (10, 4, 0.5, 45, False),
        'alb': (3.5, 0.6, 1.0, 5.5, False),
        'pco2': (40, 8, 20, 90, False),
        'po2': (90, 20, 35, 220, False),
        'ph': (7.38, 0.08, 7.0, 7.65, False),
        'fio2': (40, 20, 21, 100, False),
        'pafi': (260, 90, 40, 500, False),
        'safi': (260, 70, 80, 500, False),
        'sofa': (5.0, 3.0, 0, 24, True),
        'sofa_resp': (1.2, 1.0, 0, 4, True),
        'sofa_coag': (0.8, 0.9, 0, 4, True),
        'sofa_liver': (0.6, 0.8, 0, 4, True),
        'sofa_cardio': (1.0, 1.1, 0, 4, True),
        'sofa_cns': (0.8, 1.0, 0, 4, True),
        'sofa_renal': (0.9, 1.0, 0, 4, True),
        'sofa2': (4.8, 3.2, 0, 24, True),
        'sofa2_resp': (1.1, 1.0, 0, 4, True),
        'sofa2_coag': (0.7, 0.8, 0, 4, True),
        'sofa2_liver': (0.5, 0.7, 0, 4, True),
        'sofa2_cardio': (0.9, 1.0, 0, 4, True),
        'sofa2_cns': (0.7, 0.9, 0, 4, True),
        'sofa2_renal': (0.8, 0.9, 0, 4, True),
    }

    feature_data: Dict[str, pd.DataFrame] = {}
    for concept in concepts:
        source_df = mock_data.get(concept)
        if isinstance(source_df, pd.DataFrame) and not source_df.empty and concept in source_df.columns and 'stay_id' in source_df.columns:
            agg_df = source_df[['stay_id', concept]].copy()
            agg_df['stay_id'] = agg_df['stay_id'].astype(int)
            agg_df = agg_df.groupby('stay_id', as_index=False)[concept].mean()
            agg_df['stay_id'] = agg_df['stay_id'].map(id_map)
            agg_df = agg_df.dropna(subset=['stay_id'])
            if not agg_df.empty:
                if concept.startswith('sofa'):
                    agg_df[concept] = np.clip(np.round(agg_df[concept]), 0, 24 if concept in {'sofa', 'sofa2'} else 4)
                feature_data[concept] = agg_df.rename(columns={'stay_id': id_col})
                continue

        mean, std, min_val, max_val, integer_like = fallback_specs.get(concept, (50, 15, 0, 100, False))
        values = np.random.normal(mean, std, len(patient_ids))
        values = np.clip(values, min_val, max_val)
        if integer_like:
            values = np.round(values).astype(int)
        feature_data[concept] = pd.DataFrame({id_col: patient_ids, concept: values})

    return feature_data


def _build_group_feature_data_from_loaded_concepts(
    patient_ids: list[Any],
    concepts: list[str],
    loaded_concepts: dict[str, Any],
    *,
    id_col: str = 'stay_id',
) -> Dict[str, pd.DataFrame]:
    """Reuse already loaded concept tables to build cohort-comparison feature summaries."""
    patient_id_set = {int(pid) for pid in patient_ids}
    feature_data: Dict[str, pd.DataFrame] = {}
    for concept in concepts:
        frame = loaded_concepts.get(concept)
        if not isinstance(frame, pd.DataFrame) or frame.empty or concept not in frame.columns:
            continue

        feat_id_col = None
        for col in [id_col, 'stay_id', 'patient_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']:
            if col in frame.columns:
                feat_id_col = col
                break
        if feat_id_col is None:
            continue

        compact = frame[[feat_id_col, concept]].copy()
        compact[feat_id_col] = pd.to_numeric(compact[feat_id_col], errors='coerce')
        compact = compact.dropna(subset=[feat_id_col])
        if compact.empty:
            continue
        compact[feat_id_col] = compact[feat_id_col].astype(int)
        compact = compact[compact[feat_id_col].isin(patient_id_set)]
        if compact.empty:
            continue

        agg_func = 'max' if concept.startswith('sep3_') else 'mean'
        aggregated = compact.groupby(feat_id_col, as_index=False)[concept].agg(agg_func)
        aggregated = aggregated.rename(columns={feat_id_col: id_col})
        feature_data[concept] = aggregated
    return feature_data


def _generate_mock_multidb_data(lang: str = 'en') -> Dict[str, pd.DataFrame]:
    """生成模拟的多数据库特征分布数据用于演示。

    Args:
        lang: 语言

    Returns:
        字典，键为数据库名，值为特征数据DataFrame（长格式，含concept和value列）
    """
    np.random.seed(42)

    # 🔧 扩展特征列表，涵盖更多临床指标
    # 🔧 FIX: 模拟6个数据库（添加 MIMIC-III 和 SICdb）
    databases = {
        'miiv': {
            # Vital Signs
            'hr': (80, 15), 'sbp': (120, 20), 'dbp': (70, 12), 'map': (85, 15),
            'temp': (37.2, 0.5), 'resp': (18, 4), 'spo2': (96, 3),
            # Laboratory
            'glu': (140, 50), 'na': (140, 5), 'k': (4.2, 0.6), 'crea': (1.2, 0.8),
            'bili': (1.5, 1.2), 'lact': (2.2, 1.5),
            # Hematology
            'hgb': (11, 2), 'plt': (200, 80), 'wbc': (12, 5),
            # Blood Gas
            'ph': (7.38, 0.08), 'po2': (90, 20), 'pco2': (40, 8), 'fio2': (45, 20),
            # SOFA-2
            'sofa2': (5.2, 3.8), 'sofa2_resp': (1.2, 1.1), 'sofa2_coag': (0.8, 0.9),
            'sofa2_liver': (0.6, 0.8), 'sofa2_cardio': (1.0, 1.2), 'sofa2_cns': (0.8, 1.0), 'sofa2_renal': (0.8, 1.0),
        },
        'eicu': {
            'hr': (85, 18), 'sbp': (125, 25), 'dbp': (72, 14), 'map': (88, 18),
            'temp': (37.0, 0.6), 'resp': (20, 5), 'spo2': (95, 4),
            'glu': (150, 60), 'na': (139, 6), 'k': (4.0, 0.7), 'crea': (1.4, 1.0),
            'bili': (1.8, 1.5), 'lact': (2.5, 1.8),
            'hgb': (10.5, 2.2), 'plt': (180, 90), 'wbc': (13, 6),
            'ph': (7.36, 0.09), 'po2': (85, 22), 'pco2': (42, 10), 'fio2': (50, 25),
            # SOFA-2
            'sofa2': (6.0, 4.2), 'sofa2_resp': (1.4, 1.2), 'sofa2_coag': (0.9, 1.0),
            'sofa2_liver': (0.7, 0.9), 'sofa2_cardio': (1.2, 1.3), 'sofa2_cns': (0.9, 1.1), 'sofa2_renal': (0.9, 1.1),
        },
        'aumc': {
            'hr': (75, 12), 'sbp': (115, 18), 'dbp': (65, 10), 'map': (80, 12),
            'temp': (37.4, 0.4), 'resp': (16, 3), 'spo2': (97, 2),
            'glu': (130, 45), 'na': (141, 4), 'k': (4.3, 0.5), 'crea': (1.0, 0.6),
            'bili': (1.2, 1.0), 'lact': (1.8, 1.2),
            'hgb': (11.5, 1.8), 'plt': (220, 70), 'wbc': (11, 4),
            'ph': (7.40, 0.06), 'po2': (95, 18), 'pco2': (38, 6), 'fio2': (40, 18),
            # SOFA-2
            'sofa2': (4.5, 3.5), 'sofa2_resp': (1.0, 1.0), 'sofa2_coag': (0.7, 0.8),
            'sofa2_liver': (0.5, 0.7), 'sofa2_cardio': (0.9, 1.1), 'sofa2_cns': (0.7, 0.9), 'sofa2_renal': (0.7, 0.9),
        },
        'hirid': {
            'hr': (78, 14), 'sbp': (118, 22), 'dbp': (68, 11), 'map': (83, 14),
            'temp': (37.3, 0.5), 'resp': (17, 4), 'spo2': (96, 3),
            'glu': (135, 48), 'na': (140, 5), 'k': (4.1, 0.6), 'crea': (1.1, 0.7),
            'bili': (1.4, 1.1), 'lact': (2.0, 1.4),
            'hgb': (11.2, 2.0), 'plt': (210, 75), 'wbc': (11.5, 4.5),
            'ph': (7.39, 0.07), 'po2': (92, 19), 'pco2': (39, 7), 'fio2': (42, 19),
            # SOFA-2
            'sofa2': (4.8, 3.6), 'sofa2_resp': (1.1, 1.0), 'sofa2_coag': (0.7, 0.9),
            'sofa2_liver': (0.5, 0.7), 'sofa2_cardio': (1.0, 1.1), 'sofa2_cns': (0.7, 0.9), 'sofa2_renal': (0.8, 1.0),
        },
        # 🆕 MIMIC-III
        'mimic': {
            'hr': (82, 16), 'sbp': (122, 21), 'dbp': (71, 13), 'map': (86, 16),
            'temp': (37.1, 0.5), 'resp': (19, 4), 'spo2': (95, 3),
            'glu': (145, 55), 'na': (139, 5), 'k': (4.1, 0.6), 'crea': (1.3, 0.9),
            'bili': (1.6, 1.3), 'lact': (2.3, 1.6),
            'hgb': (10.8, 2.1), 'plt': (190, 85), 'wbc': (12.5, 5.5),
            'ph': (7.37, 0.08), 'po2': (88, 21), 'pco2': (41, 9), 'fio2': (48, 22),
            # SOFA-2
            'sofa2': (5.5, 4.0), 'sofa2_resp': (1.3, 1.1), 'sofa2_coag': (0.8, 0.9),
            'sofa2_liver': (0.6, 0.8), 'sofa2_cardio': (1.1, 1.2), 'sofa2_cns': (0.8, 1.0), 'sofa2_renal': (0.9, 1.0),
        },
        # 🆕 SICdb
        'sic': {
            'hr': (77, 13), 'sbp': (116, 19), 'dbp': (67, 11), 'map': (82, 13),
            'temp': (37.3, 0.4), 'resp': (17, 3), 'spo2': (97, 2),
            'glu': (132, 46), 'na': (141, 4), 'k': (4.2, 0.5), 'crea': (1.05, 0.65),
            'bili': (1.3, 1.0), 'lact': (1.9, 1.3),
            'hgb': (11.3, 1.9), 'plt': (215, 72), 'wbc': (11.2, 4.2),
            'ph': (7.40, 0.06), 'po2': (93, 18), 'pco2': (38, 6), 'fio2': (41, 18),
            # SOFA-2
            'sofa2': (4.2, 3.3), 'sofa2_resp': (1.0, 1.0), 'sofa2_coag': (0.6, 0.8),
            'sofa2_liver': (0.5, 0.7), 'sofa2_cardio': (0.8, 1.0), 'sofa2_cns': (0.6, 0.8), 'sofa2_renal': (0.7, 0.9),
        },
    }

    result = {}
    for db_name, features in databases.items():
        n_records_per_feat = np.random.randint(300, 600)

        # 生成长格式数据（concept + value）
        rows = []
        for feat, (mean, std) in features.items():
            values = np.random.normal(mean, std, n_records_per_feat)
            # Clip SOFA scores to valid ranges
            if feat == 'sofa2':
                values = np.clip(np.round(values), 0, 24).astype(int)
            elif feat.startswith('sofa2_'):
                values = np.clip(np.round(values), 0, 4).astype(int)
            patient_ids = np.random.randint(1000, 9999, n_records_per_feat)
            for pid, val in zip(patient_ids, values):
                rows.append({
                    'stay_id': pid,
                    'concept': feat,
                    'value': val,
                })

        result[db_name] = pd.DataFrame(rows)

    return result


def _generate_mock_cohort_dashboard_data(lang: str = 'en', n_patients: int = 500) -> pd.DataFrame:
    """生成模拟的队列仪表盘数据用于演示。

    Args:
        lang: 语言
        n_patients: 患者人数，由共享队列工作区的滑块控制

    Returns:
        包含患者人口统计学和结局数据的DataFrame
    """
    np.random.seed(42)
    try:
        n_patients = max(1, int(n_patients))
    except (TypeError, ValueError):
        n_patients = 500

    # 基本人口统计学
    patient_ids = list(range(30000000, 30000000 + n_patients))
    ages = np.clip(np.random.normal(62, 16, n_patients), 18, 95).astype(int)
    genders = np.random.choice(['M', 'F'], n_patients, p=[0.56, 0.44])  # 使用M/F格式

    # 入住类型
    admission_types = np.random.choice(
        ['EMERGENCY', 'ELECTIVE', 'URGENT', 'OBSERVATION'],
        n_patients,
        p=[0.55, 0.25, 0.15, 0.05]
    )

    # 住院时长
    los_days = np.clip(np.random.lognormal(1.2, 0.9, n_patients), 0.5, 60)

    # 机械通气状态 - 约35%需要
    mech_vent = np.random.choice([True, False], n_patients, p=[0.35, 0.65])

    # 血管活性药物 - 约25%使用
    vasopressors = np.random.choice([True, False], n_patients, p=[0.25, 0.75])

    # SOFA-1 / SOFA-2 organ scores - enables cohort-level reclassification demos.
    sofa1_resp = np.clip(np.random.poisson(1.0, n_patients) + mech_vent.astype(int), 0, 4)
    sofa1_coag = np.clip(np.random.poisson(0.7, n_patients), 0, 4)
    sofa1_liver = np.clip(np.random.poisson(0.45, n_patients), 0, 4)
    sofa1_cardio = np.clip(np.random.poisson(0.75, n_patients) + vasopressors.astype(int), 0, 4)
    sofa1_cns = np.clip(np.random.poisson(0.65, n_patients), 0, 4)
    sofa1_renal = np.clip(np.random.poisson(0.7, n_patients), 0, 4)

    def _shift_sofa_component(base, p_down=0.18, p_same=0.66, p_up=0.16, extra_up=None):
        prob_total = p_down + p_same + p_up
        delta = np.random.choice(
            [-1, 0, 1],
            n_patients,
            p=[p_down / prob_total, p_same / prob_total, p_up / prob_total],
        )
        if extra_up is not None:
            delta = delta + extra_up.astype(int)
        return np.clip(base + delta, 0, 4)

    sofa2_resp = _shift_sofa_component(sofa1_resp, p_down=0.16, p_same=0.64, p_up=0.20, extra_up=mech_vent & (np.random.random(n_patients) < 0.10))
    sofa2_coag = _shift_sofa_component(sofa1_coag, p_down=0.20, p_same=0.66, p_up=0.14)
    sofa2_liver = _shift_sofa_component(sofa1_liver, p_down=0.20, p_same=0.70, p_up=0.10)
    sofa2_cardio = _shift_sofa_component(sofa1_cardio, p_down=0.16, p_same=0.62, p_up=0.22, extra_up=vasopressors & (np.random.random(n_patients) < 0.12))
    sofa2_cns = _shift_sofa_component(sofa1_cns, p_down=0.18, p_same=0.68, p_up=0.14)
    sofa2_renal = _shift_sofa_component(sofa1_renal, p_down=0.16, p_same=0.66, p_up=0.18)
    sofa1_scores = np.clip(sofa1_resp + sofa1_coag + sofa1_liver + sofa1_cardio + sofa1_cns + sofa1_renal, 0, 20)
    sofa2_scores = np.clip(sofa2_resp + sofa2_coag + sofa2_liver + sofa2_cardio + sofa2_cns + sofa2_renal, 0, 20)
    sofa_scores = sofa2_scores
    sofa_delta = sofa2_scores - sofa1_scores
    los_days = np.clip(
        los_days + sofa_scores * 0.18 + mech_vent.astype(float) * 0.9 + vasopressors.astype(float) * 0.8,
        0.5,
        60,
    )

    # 关键队列表型 - 让演示仪表板更接近真实队列审阅场景
    sepsis = np.random.random(n_patients) < np.clip(0.16 + sofa_scores / 50, 0.05, 0.62)
    aki = np.random.random(n_patients) < np.clip(0.12 + sofa_scores / 60 + ages / 700, 0.04, 0.55)
    rrt = np.random.random(n_patients) < np.clip(0.02 + sofa_scores / 140 + aki.astype(int) * 0.08, 0.01, 0.28)
    abx = sepsis | (np.random.random(n_patients) < 0.18)

    # 死亡结局 - 用SOFA-2和SOFA reclassification驱动，确保demo呈现清晰的临床梯度。
    mortality_logit = (
        -3.6
        + sofa_scores * 0.30
        + np.maximum(sofa_delta, 0) * 0.25
        - np.maximum(-sofa_delta, 0) * 0.10
        + (ages - 60) * 0.015
        + sepsis.astype(float) * 0.40
        + vasopressors.astype(float) * 0.30
        + mech_vent.astype(float) * 0.20
    )
    mortality_prob = 1 / (1 + np.exp(-mortality_logit))
    mortality_prob = np.clip(mortality_prob, 0.02, 0.72)
    mortality = np.random.random(n_patients) < mortality_prob

    # 诊断类别
    diagnoses = np.random.choice(
        ['Sepsis', 'Respiratory Failure', 'Cardiac', 'Neurological', 'Post-surgical', 'Trauma', 'Other'],
        n_patients,
        p=[0.25, 0.20, 0.15, 0.12, 0.15, 0.08, 0.05]
    )

    df = pd.DataFrame({
        'stay_id': patient_ids,
        'age': ages,
        'gender': genders,
        'admission_type': admission_types,
        'los_days': los_days,
        'los_hours': los_days * 24,  # 添加los_hours列
        'mech_vent': mech_vent,
        'vasopressors': vasopressors,
        'sepsis': sepsis,
        'aki': aki,
        'rrt': rrt,
        'abx': abx,
        'sofa_max': sofa_scores,
        'sofa1_max': sofa1_scores,
        'sofa2_max': sofa2_scores,
        'sofa1_resp': sofa1_resp,
        'sofa2_resp': sofa2_resp,
        'sofa1_coag': sofa1_coag,
        'sofa2_coag': sofa2_coag,
        'sofa1_liver': sofa1_liver,
        'sofa2_liver': sofa2_liver,
        'sofa1_cardio': sofa1_cardio,
        'sofa2_cardio': sofa2_cardio,
        'sofa1_cns': sofa1_cns,
        'sofa2_cns': sofa2_cns,
        'sofa1_renal': sofa1_renal,
        'sofa2_renal': sofa2_renal,
        'mortality': mortality,
        'survived': [1 if not m else 0 for m in mortality],  # 添加survived列（1=存活，0=死亡）
        'first_icu_stay': np.random.choice([True, False], n_patients, p=[0.65, 0.35]),  # 添加first_icu_stay列
        'diagnosis_group': diagnoses,
    })

    return df

