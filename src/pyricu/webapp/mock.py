"""PyRICU Webapp Mock 数据生成模块。

包含生成模拟 ICU 数据的函数，用于演示模式。

注意：主要的 generate_mock_data 函数仍在 app.py 中，
因为它涉及大量的数据生成逻辑，迁移风险较高。
这里放置较小的辅助 mock 数据生成函数。
"""

import numpy as np
import pandas as pd
from typing import Dict


def generate_mock_demographics(n_patients: int, lang: str = 'en') -> pd.DataFrame:
    """生成模拟的人口统计学数据。
    
    Args:
        n_patients: 患者数量
        lang: 语言（'en' 或 'zh'）
        
    Returns:
        包含人口统计学数据的 DataFrame
    """
    np.random.seed(42)
    
    # 生成基础数据
    ages = np.random.normal(65, 15, n_patients).clip(18, 95)
    sexes = np.random.choice(['Male', 'Female'] if lang == 'en' else ['男', '女'], n_patients)
    weights = np.random.normal(70, 15, n_patients).clip(40, 150)
    heights = np.random.normal(170, 10, n_patients).clip(140, 200)
    bmis = weights / (heights / 100) ** 2
    
    # 生成 ICU 住院时长（对数正态分布，中位数约3天）
    los_icu = np.random.lognormal(1.1, 0.8, n_patients).clip(0.5, 30)
    
    # 生成死亡率（约15%）
    death_rates = 0.15
    deaths = np.random.random(n_patients) < death_rates
    
    # 生成 SOFA 评分
    sofa_scores = np.random.poisson(5, n_patients).clip(0, 24)
    
    df = pd.DataFrame({
        'patient_id': range(1, n_patients + 1),
        'age': ages.round(1),
        'sex': sexes,
        'weight': weights.round(1),
        'height': heights.round(1),
        'bmi': bmis.round(1),
        'los_icu': los_icu.round(1),
        'death': deaths.astype(int),
        'sofa_max': sofa_scores,
    })
    
    return df


def generate_mock_multidb_data(lang: str = 'en') -> Dict[str, pd.DataFrame]:
    """生成多数据库对比的模拟数据。
    
    Args:
        lang: 语言（'en' 或 'zh'）
        
    Returns:
        字典，键为数据库名，值为 DataFrame
    """
    databases = ['MIMIC-IV', 'eICU-CRD', 'AmsterdamUMCdb', 'HiRID']
    n_patients_per_db = [200, 150, 100, 80]
    
    result = {}
    for db, n in zip(databases, n_patients_per_db):
        np.random.seed(hash(db) % 2**32)
        
        df = pd.DataFrame({
            'patient_id': range(1, n + 1),
            'database': db,
            'age': np.random.normal(65, 12, n).clip(18, 95),
            'los_icu': np.random.lognormal(1.0, 0.7, n).clip(0.5, 30),
            'sofa_max': np.random.poisson(5, n).clip(0, 20),
            'mortality': (np.random.random(n) < 0.12).astype(int),
        })
        result[db] = df
    
    return result


def generate_mock_cohort_dashboard_data(lang: str = 'en') -> pd.DataFrame:
    """生成队列仪表盘的模拟数据。
    
    Args:
        lang: 语言（'en' 或 'zh'）
        
    Returns:
        包含队列统计数据的 DataFrame
    """
    np.random.seed(42)
    n_patients = 500
    
    # 生成基础人口统计学
    ages = np.random.normal(62, 16, n_patients).clip(18, 95)
    sexes = np.random.choice(['M', 'F'], n_patients, p=[0.55, 0.45])
    
    # 生成诊断类别
    diagnoses = np.random.choice(
        ['Sepsis', 'Respiratory Failure', 'Cardiac', 'Trauma', 'Neurological', 'Other'],
        n_patients,
        p=[0.25, 0.20, 0.18, 0.12, 0.10, 0.15]
    )
    
    # 生成严重程度和结局
    sofa_scores = np.random.poisson(6, n_patients).clip(0, 24)
    los_icu = np.random.lognormal(1.2, 0.9, n_patients).clip(0.5, 45)
    mortality = (np.random.random(n_patients) < 0.18).astype(int)
    
    # Sepsis 患者的特殊处理
    sepsis_mask = diagnoses == 'Sepsis'
    sofa_scores[sepsis_mask] = (sofa_scores[sepsis_mask] * 1.3).clip(0, 24)
    mortality[sepsis_mask] = (np.random.random(sepsis_mask.sum()) < 0.25).astype(int)
    
    df = pd.DataFrame({
        'patient_id': range(1, n_patients + 1),
        'age': ages.round(1),
        'sex': sexes,
        'diagnosis': diagnoses,
        'sofa_max': sofa_scores.astype(int),
        'los_icu': los_icu.round(1),
        'mortality': mortality,
        'ventilator': (np.random.random(n_patients) < 0.45).astype(int),
        'vasopressor': (np.random.random(n_patients) < 0.35).astype(int),
        'rrt': (np.random.random(n_patients) < 0.12).astype(int),
    })
    
    return df


def generate_mock_timeseries(
    patient_ids: list,
    concepts: list,
    hours: int = 72,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """生成模拟时间序列数据。
    
    Args:
        patient_ids: 患者ID列表
        concepts: 概念列表
        hours: 时间长度（小时）
        seed: 随机种子
        
    Returns:
        字典，键为概念名，值为 DataFrame
    """
    np.random.seed(seed)
    data = {}
    time_points = np.arange(0, hours, 1)
    
    for concept in concepts:
        records = []
        for pid in patient_ids:
            # 根据概念类型生成不同的数据
            if concept == 'hr':
                base = np.random.uniform(70, 90)
                for t in time_points:
                    if np.random.random() < 0.9:  # 10% 缺失
                        val = base + np.sin(t / 6) * 10 + np.random.normal(0, 5)
                        records.append({'stay_id': pid, 'time': t, concept: max(40, min(150, val))})
            elif concept == 'sbp':
                base = np.random.uniform(110, 140)
                for t in time_points:
                    if np.random.random() < 0.9:
                        val = base + np.sin(t / 5) * 15 + np.random.normal(0, 8)
                        records.append({'stay_id': pid, 'time': t, concept: max(70, min(200, val))})
            elif concept == 'temp':
                base = np.random.uniform(36.5, 37.5)
                for t in time_points[::4]:  # 每4小时
                    val = base + np.random.normal(0, 0.3)
                    records.append({'stay_id': pid, 'time': t, concept: max(35, min(41, val))})
            else:
                # 默认：生成随机正态分布数据
                base = np.random.uniform(50, 100)
                for t in time_points[::2]:
                    val = base + np.random.normal(0, base * 0.1)
                    records.append({'stay_id': pid, 'time': t, concept: max(0, val)})
        
        data[concept] = pd.DataFrame(records) if records else pd.DataFrame(columns=['stay_id', 'time', concept])
    
    return data
