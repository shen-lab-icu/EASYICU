"""
PyRICU Demo Data Generator

生成合成的ICU演示数据，供审稿人和用户在没有真实数据库的情况下体验工具功能。

数据完全合成，不包含任何真实患者信息。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import json


# 演示数据配置
DEMO_CONFIG = {
    'num_patients': 100,  # 患者数量
    'hours_per_patient': 72,  # 每个患者的ICU住院小时数
    'sampling_interval_hours': 1,  # 采样间隔（小时）
}

# 概念的正常范围和分布参数
CONCEPT_DISTRIBUTIONS = {
    # 生命体征
    'hr': {'mean': 80, 'std': 15, 'min': 40, 'max': 180, 'unit': 'bpm'},
    'sbp': {'mean': 120, 'std': 20, 'min': 70, 'max': 200, 'unit': 'mmHg'},
    'dbp': {'mean': 70, 'std': 12, 'min': 40, 'max': 120, 'unit': 'mmHg'},
    'map': {'mean': 85, 'std': 15, 'min': 50, 'max': 150, 'unit': 'mmHg'},
    'resp': {'mean': 16, 'std': 4, 'min': 8, 'max': 40, 'unit': '/min'},
    'temp': {'mean': 37.0, 'std': 0.7, 'min': 35.0, 'max': 40.0, 'unit': '°C'},
    'spo2': {'mean': 96, 'std': 3, 'min': 80, 'max': 100, 'unit': '%'},
    
    # 实验室检查
    'bili': {'mean': 1.2, 'std': 2.5, 'min': 0.1, 'max': 30, 'unit': 'mg/dL'},
    'crea': {'mean': 1.2, 'std': 1.5, 'min': 0.3, 'max': 15, 'unit': 'mg/dL'},
    'glu': {'mean': 120, 'std': 40, 'min': 40, 'max': 400, 'unit': 'mg/dL'},
    'k': {'mean': 4.0, 'std': 0.5, 'min': 2.5, 'max': 7.0, 'unit': 'mEq/L'},
    'na': {'mean': 140, 'std': 4, 'min': 120, 'max': 160, 'unit': 'mEq/L'},
    'phos': {'mean': 3.5, 'std': 1.0, 'min': 1.0, 'max': 8.0, 'unit': 'mg/dL'},
    'alb': {'mean': 3.5, 'std': 0.6, 'min': 1.5, 'max': 5.0, 'unit': 'g/dL'},
    
    # 血气分析
    'po2': {'mean': 90, 'std': 20, 'min': 40, 'max': 500, 'unit': 'mmHg'},
    'pco2': {'mean': 40, 'std': 8, 'min': 20, 'max': 80, 'unit': 'mmHg'},
    'ph': {'mean': 7.40, 'std': 0.08, 'min': 7.0, 'max': 7.6, 'unit': ''},
    'fio2': {'mean': 40, 'std': 20, 'min': 21, 'max': 100, 'unit': '%'},
    'o2sat': {'mean': 95, 'std': 4, 'min': 70, 'max': 100, 'unit': '%'},
    
    # 血液学
    'hgb': {'mean': 11, 'std': 2, 'min': 5, 'max': 18, 'unit': 'g/dL'},
    'plt': {'mean': 200, 'std': 80, 'min': 20, 'max': 600, 'unit': '10^9/L'},
    'wbc': {'mean': 10, 'std': 5, 'min': 1, 'max': 40, 'unit': '10^9/L'},
    
    # GCS
    'gcs': {'mean': 13, 'std': 3, 'min': 3, 'max': 15, 'unit': ''},
    'egcs': {'mean': 4, 'std': 0.5, 'min': 1, 'max': 4, 'unit': ''},
    'mgcs': {'mean': 5, 'std': 1, 'min': 1, 'max': 6, 'unit': ''},
    'vgcs': {'mean': 4, 'std': 1, 'min': 1, 'max': 5, 'unit': ''},
    
    # 人口学
    'age': {'mean': 65, 'std': 15, 'min': 18, 'max': 95, 'unit': 'years'},
    'weight': {'mean': 75, 'std': 18, 'min': 40, 'max': 200, 'unit': 'kg'},
    'height': {'mean': 170, 'std': 12, 'min': 140, 'max': 210, 'unit': 'cm'},
    
    # 尿量
    'urine': {'mean': 80, 'std': 40, 'min': 0, 'max': 300, 'unit': 'mL/h'},
    
    # 乳酸
    'lact': {'mean': 1.5, 'std': 1.5, 'min': 0.5, 'max': 15, 'unit': 'mmol/L'},
}


def generate_patient_ids(n: int, database: str = 'demo') -> List[int]:
    """生成患者ID列表"""
    if database == 'demo':
        return list(range(1000001, 1000001 + n))
    return list(range(1, n + 1))


def generate_timestamps(start_time: datetime, hours: int, interval_hours: float = 1.0) -> List[datetime]:
    """生成时间戳序列"""
    timestamps = []
    current = start_time
    for _ in range(int(hours / interval_hours)):
        timestamps.append(current)
        current += timedelta(hours=interval_hours)
    return timestamps


def generate_concept_timeseries(
    patient_id: int,
    concept: str,
    timestamps: List[datetime],
    config: Dict[str, Any],
    missing_rate: float = 0.1,
    severity: float = 0.0,  # 0-1, 0=正常, 1=危重
) -> pd.DataFrame:
    """
    为单个患者生成概念时间序列数据
    
    Args:
        patient_id: 患者ID
        concept: 概念名称
        timestamps: 时间戳列表
        config: 概念配置（均值、标准差等）
        missing_rate: 缺失率
        severity: 疾病严重程度（影响数值偏离程度）
    """
    n = len(timestamps)
    
    # 根据严重程度调整均值
    base_mean = config['mean']
    if concept in ['hr', 'resp', 'temp', 'lact', 'crea', 'bili', 'wbc']:
        # 这些指标在病情加重时升高
        adjusted_mean = base_mean * (1 + severity * 0.3)
    elif concept in ['sbp', 'map', 'spo2', 'o2sat', 'plt', 'gcs']:
        # 这些指标在病情加重时降低
        adjusted_mean = base_mean * (1 - severity * 0.2)
    else:
        adjusted_mean = base_mean
    
    # 生成基础值
    values = np.random.normal(adjusted_mean, config['std'], n)
    
    # 添加一些时间相关性（AR过程）
    for i in range(1, n):
        values[i] = 0.7 * values[i-1] + 0.3 * values[i]
    
    # 裁剪到有效范围
    values = np.clip(values, config['min'], config['max'])
    
    # 随机设置缺失值
    mask = np.random.random(n) < missing_rate
    values[mask] = np.nan
    
    # 构建DataFrame
    df = pd.DataFrame({
        'stay_id': patient_id,
        'charttime': timestamps,
        concept: values,
    })
    
    # 移除缺失行
    df = df.dropna(subset=[concept])
    
    return df


def generate_demo_data(
    concepts: List[str],
    num_patients: int = 100,
    hours_per_patient: int = 72,
    sampling_interval: float = 1.0,
    database: str = 'demo',
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    生成演示数据
    
    Args:
        concepts: 要生成的概念列表
        num_patients: 患者数量
        hours_per_patient: 每个患者的ICU住院小时数
        sampling_interval: 采样间隔（小时）
        database: 数据库名称（用于ID格式）
        seed: 随机种子
        
    Returns:
        Dict[concept_name, DataFrame] 格式的数据
    """
    np.random.seed(seed)
    
    # 生成患者ID
    patient_ids = generate_patient_ids(num_patients, database)
    
    # 基准时间
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # 为每个患者分配随机的严重程度
    severities = np.random.beta(2, 5, num_patients)  # 大多数患者病情较轻
    
    results = {}
    
    for concept in concepts:
        if concept not in CONCEPT_DISTRIBUTIONS:
            print(f"⚠️ Unknown concept: {concept}, skipping")
            continue
        
        config = CONCEPT_DISTRIBUTIONS[concept]
        
        # 确定采样间隔和缺失率
        if concept in ['hr', 'sbp', 'dbp', 'map', 'resp', 'spo2']:
            # 生命体征：频繁采样，低缺失
            interval = sampling_interval
            missing_rate = 0.05
        elif concept in ['bili', 'crea', 'glu', 'k', 'na', 'plt', 'wbc', 'hgb']:
            # 实验室检查：每6-8小时，中等缺失
            interval = 6.0
            missing_rate = 0.2
        elif concept in ['po2', 'pco2', 'ph', 'fio2', 'lact']:
            # 血气：每4小时，中等缺失
            interval = 4.0
            missing_rate = 0.25
        elif concept in ['age', 'weight', 'height']:
            # 人口学：只有入院时一个值
            interval = hours_per_patient  # 整个住院期间只有一个值
            missing_rate = 0.02
        elif concept in ['gcs', 'egcs', 'mgcs', 'vgcs']:
            # GCS：每4小时
            interval = 4.0
            missing_rate = 0.15
        else:
            interval = sampling_interval
            missing_rate = 0.1
        
        all_data = []
        
        for i, patient_id in enumerate(patient_ids):
            # 每个患者的入院时间随机偏移
            patient_base_time = base_time + timedelta(hours=np.random.randint(0, 24))
            timestamps = generate_timestamps(patient_base_time, hours_per_patient, interval)
            
            df = generate_concept_timeseries(
                patient_id=patient_id,
                concept=concept,
                timestamps=timestamps,
                config=config,
                missing_rate=missing_rate,
                severity=severities[i],
            )
            
            all_data.append(df)
        
        # 合并所有患者数据
        results[concept] = pd.concat(all_data, ignore_index=True)
    
    return results


def generate_sofa_demo_data(
    num_patients: int = 100,
    hours_per_patient: int = 72,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成包含SOFA评分的演示数据
    
    Returns:
        包含sofa及其子分数的DataFrame
    """
    np.random.seed(seed)
    
    patient_ids = generate_patient_ids(num_patients, 'demo')
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # SOFA子分数范围
    sofa_components = {
        'sofa_resp': (0, 4),
        'sofa_coag': (0, 4),
        'sofa_liver': (0, 4),
        'sofa_cardio': (0, 4),
        'sofa_cns': (0, 4),
        'sofa_renal': (0, 4),
    }
    
    all_data = []
    
    for patient_id in patient_ids:
        patient_base_time = base_time + timedelta(hours=np.random.randint(0, 24))
        timestamps = generate_timestamps(patient_base_time, hours_per_patient, 1.0)
        
        # 生成每个时间点的SOFA子分数
        for t in timestamps:
            row = {'stay_id': patient_id, 'charttime': t}
            total_sofa = 0
            
            for component, (min_val, max_val) in sofa_components.items():
                # 大多数时间点分数较低
                score = np.random.choice(
                    [0, 1, 2, 3, 4],
                    p=[0.4, 0.25, 0.2, 0.1, 0.05]
                )
                row[component] = score
                total_sofa += score
            
            row['sofa'] = total_sofa
            all_data.append(row)
    
    return pd.DataFrame(all_data)


def generate_aki_demo_data(
    num_patients: int = 100,
    hours_per_patient: int = 72,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成包含KDIGO AKI分期的演示数据
    """
    np.random.seed(seed)
    
    patient_ids = generate_patient_ids(num_patients, 'demo')
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    all_data = []
    
    for patient_id in patient_ids:
        patient_base_time = base_time + timedelta(hours=np.random.randint(0, 24))
        timestamps = generate_timestamps(patient_base_time, hours_per_patient, 4.0)  # 每4小时
        
        # 基线肌酐
        baseline_crea = np.random.uniform(0.6, 1.2)
        
        # 是否发生AKI
        has_aki = np.random.random() < 0.3  # 30%患者发生AKI
        
        for i, t in enumerate(timestamps):
            if has_aki and i > len(timestamps) // 3:
                # AKI发生后肌酐升高
                crea = baseline_crea * np.random.uniform(1.5, 3.0)
                aki_stage = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            else:
                crea = baseline_crea * np.random.uniform(0.9, 1.1)
                aki_stage = 0
            
            all_data.append({
                'stay_id': patient_id,
                'charttime': t,
                'crea': round(crea, 2),
                'creat_low_past_7day': round(baseline_crea, 2),
                'aki_stage': aki_stage,
                'aki': aki_stage > 0,
            })
    
    return pd.DataFrame(all_data)


def generate_circ_failure_demo_data(
    num_patients: int = 100,
    hours_per_patient: int = 72,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成包含循环衰竭状态的演示数据
    """
    np.random.seed(seed)
    
    patient_ids = generate_patient_ids(num_patients, 'demo')
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    all_data = []
    
    for patient_id in patient_ids:
        patient_base_time = base_time + timedelta(hours=np.random.randint(0, 24))
        timestamps = generate_timestamps(patient_base_time, hours_per_patient, 1.0)
        
        # 是否发生循环衰竭
        has_circ_failure = np.random.random() < 0.25  # 25%患者发生循环衰竭
        
        for i, t in enumerate(timestamps):
            if has_circ_failure and i > len(timestamps) // 4:
                lactate = np.random.uniform(2.5, 8.0)
                map_val = np.random.uniform(50, 70)
                circ_event = np.random.choice([1, 2, 3], p=[0.4, 0.35, 0.25])
            else:
                lactate = np.random.uniform(0.8, 1.8)
                map_val = np.random.uniform(70, 100)
                circ_event = 0
            
            all_data.append({
                'stay_id': patient_id,
                'charttime': t,
                'lact': round(lactate, 2),
                'map': round(map_val, 1),
                'circ_event': circ_event,
                'circ_failure': circ_event > 0,
            })
    
    return pd.DataFrame(all_data)


def save_demo_data(output_dir: str, num_patients: int = 100) -> Dict[str, str]:
    """
    生成并保存完整的演示数据集
    
    Args:
        output_dir: 输出目录
        num_patients: 患者数量
        
    Returns:
        生成的文件路径字典
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Generating demo data for {num_patients} patients...")
    
    # 生成基础概念数据
    basic_concepts = ['hr', 'sbp', 'dbp', 'map', 'resp', 'temp', 'spo2']
    lab_concepts = ['bili', 'crea', 'glu', 'k', 'na', 'plt', 'wbc', 'hgb']
    blood_gas_concepts = ['po2', 'pco2', 'ph', 'fio2', 'o2sat', 'lact']
    other_concepts = ['gcs', 'age', 'weight', 'height', 'urine']
    
    all_concepts = basic_concepts + lab_concepts + blood_gas_concepts + other_concepts
    
    # 生成概念数据
    print("  📊 Generating concept data...")
    concept_data = generate_demo_data(
        concepts=all_concepts,
        num_patients=num_patients,
        hours_per_patient=72,
    )
    
    # 保存概念数据
    saved_files = {}
    
    for concept, df in concept_data.items():
        file_path = output_path / f"demo_{concept}.parquet"
        df.to_parquet(file_path, index=False)
        saved_files[concept] = str(file_path)
        print(f"    ✅ {concept}: {len(df):,} rows")
    
    # 生成SOFA数据
    print("  📊 Generating SOFA data...")
    sofa_df = generate_sofa_demo_data(num_patients=num_patients)
    sofa_path = output_path / "demo_sofa.parquet"
    sofa_df.to_parquet(sofa_path, index=False)
    saved_files['sofa'] = str(sofa_path)
    print(f"    ✅ sofa: {len(sofa_df):,} rows")
    
    # 生成AKI数据
    print("  📊 Generating AKI data...")
    aki_df = generate_aki_demo_data(num_patients=num_patients)
    aki_path = output_path / "demo_aki.parquet"
    aki_df.to_parquet(aki_path, index=False)
    saved_files['aki'] = str(aki_path)
    print(f"    ✅ aki: {len(aki_df):,} rows")
    
    # 生成循环衰竭数据
    print("  📊 Generating circulatory failure data...")
    circ_df = generate_circ_failure_demo_data(num_patients=num_patients)
    circ_path = output_path / "demo_circ_failure.parquet"
    circ_df.to_parquet(circ_path, index=False)
    saved_files['circ_failure'] = str(circ_path)
    print(f"    ✅ circ_failure: {len(circ_df):,} rows")
    
    # 生成元数据文件
    metadata = {
        'version': '1.0',
        'generated_at': datetime.now().isoformat(),
        'num_patients': num_patients,
        'hours_per_patient': 72,
        'concepts': list(saved_files.keys()),
        'files': saved_files,
        'description': 'PyRICU Demo Dataset - Synthetic ICU data for demonstration purposes',
    }
    
    metadata_path = output_path / "demo_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Demo data generated successfully!")
    print(f"   Location: {output_path}")
    print(f"   Patients: {num_patients}")
    print(f"   Concepts: {len(saved_files)}")
    
    return saved_files


def load_demo_data(
    concepts: List[str],
    demo_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    加载演示数据
    
    Args:
        concepts: 要加载的概念列表
        demo_dir: 演示数据目录（默认使用包内数据）
        
    Returns:
        合并的DataFrame
    """
    if demo_dir is None:
        # 使用默认位置
        demo_dir = Path(__file__).parent / "demo_data"
    else:
        demo_dir = Path(demo_dir)
    
    if not demo_dir.exists():
        raise FileNotFoundError(
            f"Demo data not found at {demo_dir}. "
            "Please run `save_demo_data()` first to generate demo data."
        )
    
    dfs = []
    
    for concept in concepts:
        file_path = demo_dir / f"demo_{concept}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dfs.append(df)
        else:
            print(f"⚠️ Demo data for '{concept}' not found")
    
    if not dfs:
        return pd.DataFrame()
    
    # 合并数据
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=['stay_id', 'charttime'], how='outer')
    
    return result.sort_values(['stay_id', 'charttime']).reset_index(drop=True)


def is_demo_mode() -> bool:
    """检查是否处于演示模式"""
    import os
    return os.environ.get('PYRICU_DEMO_MODE', '').lower() in ('1', 'true', 'yes')


def get_demo_patient_ids(n: int = 100) -> List[int]:
    """获取演示模式的患者ID列表"""
    return generate_patient_ids(n, 'demo')


if __name__ == '__main__':
    import sys
    
    # 命令行运行时生成演示数据
    output_dir = sys.argv[1] if len(sys.argv) > 1 else './demo_data'
    num_patients = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    save_demo_data(output_dir, num_patients)
