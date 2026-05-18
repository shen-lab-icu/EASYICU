"""Mock ICU data generation for the EasyICU Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _add_catalog_mock_fallbacks(data, patient_ids, time_points, get_random_sample_times):
    """Fill demo data gaps for concepts present in the web catalog.

    The hand-written mock generator carries clinically shaped fixtures for the
    common concepts.  The web catalog grows faster than those fixtures, so this
    final pass keeps demo exports aligned with the current selectable concept
    list instead of silently dropping newly added concepts.
    """
    try:
        from easyicu.webapp.concept_catalog import CONCEPT_DICTIONARY
    except Exception:
        return

    patient_ids = list(patient_ids)
    if not patient_ids:
        return

    def _empty(concept):
        return pd.DataFrame(columns=['stay_id', 'time', concept])

    def _event_frame(concept, probability=0.12, duration_hours=8):
        records = []
        for pid in patient_ids:
            if np.random.random() >= probability:
                continue
            start = int(np.random.uniform(0, max(1, len(time_points) - 1)))
            for t in range(start, min(int(time_points[-1]) + 1, start + duration_hours), 4):
                records.append({'stay_id': pid, 'time': float(t), concept: 1})
        return pd.DataFrame(records) if records else _empty(concept)

    def _static_event_frame(concept, probability=0.2):
        records = [
            {'stay_id': pid, concept: int(np.random.random() < probability)}
            for pid in patient_ids
        ]
        return pd.DataFrame(records)

    def _timeseries_frame(concept, mean, std, min_val=None, max_val=None, interval=12):
        records = []
        for pid in patient_ids:
            for t in get_random_sample_times(pid, base_interval=interval, jitter=0.4):
                value = float(np.random.normal(mean, std))
                if min_val is not None:
                    value = max(min_val, value)
                if max_val is not None:
                    value = min(max_val, value)
                records.append({'stay_id': pid, 'time': t, concept: value})
        return pd.DataFrame(records) if records else _empty(concept)

    def _numeric_specs(concept):
        specs = {
            'bicar': (24, 4, 5, 45, 12),
            'cl': (104, 5, 80, 130, 12),
            'anion_gap': (13, 4, 0, 40, 12),
            'alp': (90, 45, 20, 500, 24),
            'alt': (35, 30, 5, 500, 24),
            'ast': (42, 35, 5, 600, 24),
            'bili_dir': (0.4, 0.5, 0.0, 15, 24),
            'bun': (24, 14, 4, 150, 12),
            'ca': (8.8, 0.8, 5.5, 13, 12),
            'ck': (180, 150, 10, 4000, 24),
            'ckmb': (4, 4, 0, 120, 24),
            'crp': (55, 45, 0, 300, 24),
            'mg': (2.0, 0.35, 1.0, 4.0, 12),
            'tnt': (0.04, 0.08, 0, 3, 24),
            'tri': (0.05, 0.1, 0, 5, 24),
            'bnd': (4, 5, 0, 40, 24),
            'basos': (0.5, 0.4, 0, 5, 24),
            'eos': (2, 2, 0, 15, 24),
            'esr': (35, 25, 0, 140, 24),
            'fgn': (350, 120, 80, 900, 24),
            'hba1c': (6.2, 1.3, 4.5, 14, 72),
            'hct': (34, 6, 15, 55, 12),
            'inr_pt': (1.2, 0.35, 0.8, 6, 12),
            'lymph': (18, 10, 1, 60, 24),
            'mch': (30, 2.5, 20, 40, 24),
            'mchc': (33, 1.5, 25, 38, 24),
            'mcv': (90, 8, 65, 120, 24),
            'neut': (72, 12, 20, 98, 24),
            'pt': (13, 3, 8, 60, 12),
            'ptt': (32, 10, 18, 150, 12),
            'rbc': (3.8, 0.8, 1.5, 7, 24),
            'rdw': (14.5, 2.0, 10, 25, 24),
            'total_input_ml': (120, 70, 0, 600, 1),
            'fluid_balance': (20, 90, -300, 500, 1),
            'fluid_balance_cumulative': (500, 900, -3000, 6000, 6),
            'pulse_pressure': (50, 14, 10, 110, 1),
        }
        if concept in specs:
            return specs[concept]
        if concept.endswith('_rate') or concept in {'dex', 'ins'}:
            return (0.08, 0.08, 0, 1.5, 4)
        if concept.endswith('_dur'):
            return (8, 6, 0, 72, 24)
        return (50, 15, 0, 100, 12)

    def _add_derived_frames():
        if 'pulse_pressure' not in data and {'sbp', 'dbp'} <= set(data):
            merged = data['sbp'].merge(data['dbp'], on=['stay_id', 'time'], how='inner')
            if not merged.empty:
                merged['pulse_pressure'] = (merged['sbp'] - merged['dbp']).clip(5, 120)
                data['pulse_pressure'] = merged[['stay_id', 'time', 'pulse_pressure']]
        if 'anion_gap' not in data and {'na', 'cl', 'bicar'} <= set(data):
            merged = data['na'].merge(data['cl'], on=['stay_id', 'time'], how='inner')
            merged = merged.merge(data['bicar'], on=['stay_id', 'time'], how='inner')
            if not merged.empty:
                merged['anion_gap'] = (merged['na'] - merged['cl'] - merged['bicar']).clip(0, 45)
                data['anion_gap'] = merged[['stay_id', 'time', 'anion_gap']]

    _add_derived_frames()

    medication_like = {
        'albumin_iv', 'amiodarone', 'apixaban', 'aspirin', 'bicarbonate',
        'calcium_iv', 'cisatracurium', 'dexamethasone', 'dexmedetomidine',
        'dextrose50', 'diltiazem', 'enoxaparin', 'esmolol', 'fentanyl',
        'ffp', 'furosemide', 'heparin', 'insulin', 'ketamine',
        'labetalol', 'levetiracetam', 'lorazepam', 'magnesium_iv',
        'mannitol', 'meropenem', 'midazolam', 'milrinone', 'morphine',
        'neostigmine', 'nicardipine', 'nitroglycerin', 'octreotide',
        'packed_rbc', 'pantoprazole', 'phenytoin', 'platelets',
        'potassium_iv', 'propofol', 'rocuronium', 'vancomycin',
        'vecuronium', 'warfarin',
    }

    for concept, meta in CONCEPT_DICTIONARY.items():
        if concept in data:
            continue
        unit = ''
        if isinstance(meta, (tuple, list)) and len(meta) >= 3:
            unit = str(meta[2]).lower()
        if concept in medication_like or unit == 'boolean' or concept.endswith(('60', '_ind')):
            data[concept] = _event_frame(concept, probability=0.10)
        elif concept.endswith('_dur'):
            mean, std, min_val, max_val, interval = _numeric_specs(concept)
            data[concept] = _timeseries_frame(concept, mean, std, min_val, max_val, interval)
        elif concept in {'fentanyl_rate', 'midazolam_rate', 'propofol_rate'}:
            mean, std, min_val, max_val, interval = _numeric_specs(concept)
            data[concept] = _timeseries_frame(concept, mean, std, min_val, max_val, interval)
        elif concept in {'fluid_balance', 'fluid_balance_cumulative', 'total_input_ml'}:
            mean, std, min_val, max_val, interval = _numeric_specs(concept)
            data[concept] = _timeseries_frame(concept, mean, std, min_val, max_val, interval)
        elif concept == 'adm':
            data[concept] = _static_event_frame(concept, probability=1.0)
        else:
            mean, std, min_val, max_val, interval = _numeric_specs(concept)
            data[concept] = _timeseries_frame(concept, mean, std, min_val, max_val, interval)


def generate_mock_data(n_patients=10, hours=72, cohort_filter=None):
    """生成模拟 ICU 数据用于演示。

    Args:
        n_patients: 要生成的患者数量
        hours: 数据时长（小时）
        cohort_filter: 队列过滤器字典，支持以下字段：
            - age_min/age_max: 年龄范围
            - gender: 'M' 或 'F'
            - survived: True/False
            - has_sepsis: True/False
            - disease_cohort: 'sepsis' | 'aki' | 'circ_failure' | 'mech_vent' | 'rrt'
            - los_min: 最短住院时长（小时）
    """
    data = {}

    # 🔧 如果有过滤器，根据过滤条件计算需要的初始患者数
    # 性别过滤约50%通过，存活过滤约85%通过，sepsis过滤约30%/70%通过
    initial_multiplier = 1
    if cohort_filter:
        # 估算每个过滤器的通过率
        if cohort_filter.get('gender') is not None:
            initial_multiplier *= 2.5  # 性别过滤约50%通过
        if cohort_filter.get('survived') is not None:
            if cohort_filter['survived']:
                initial_multiplier *= 1.3  # 存活约85%
            else:
                initial_multiplier *= 8  # 死亡约15%
        if cohort_filter.get('has_sepsis') is not None:
            if cohort_filter['has_sepsis']:
                initial_multiplier *= 4  # sepsis约30%
            else:
                initial_multiplier *= 1.5  # 非sepsis约70%
        if cohort_filter.get('disease_cohort') not in (None, '', 'none'):
            initial_multiplier *= 2.5
        if cohort_filter.get('age_min') is not None or cohort_filter.get('age_max') is not None:
            initial_multiplier *= 1.5  # 年龄范围过滤
        initial_multiplier = max(3, int(initial_multiplier))  # 最少3倍

    initial_n = n_patients * initial_multiplier
    all_patient_ids = list(range(10001, 10001 + initial_n))

    np.random.seed(42)
    time_points = np.arange(0, hours, 1)

    # 🔧 FIX (2026-02-03): 添加患者级随机采样时间生成函数
    def get_random_sample_times(pid, base_interval, jitter=0.3, min_samples=3):
        """为每个患者生成随机采样时间点

        Args:
            pid: 患者ID，用作随机种子的一部分
            base_interval: 基础采样间隔（小时）
            jitter: 间隔的随机抖动比例 (0-1)
            min_samples: 最少采样次数

        Returns:
            该患者的随机采样时间点列表
        """
        rng = np.random.RandomState(pid * 17 + 31)  # 每个患者有独立的随机状态
        sample_times = [0]  # 从0开始
        current_time = 0

        while current_time < hours - base_interval:
            # 在基础间隔上添加随机抖动
            interval = base_interval * (1 + rng.uniform(-jitter, jitter))
            interval = max(1, interval)  # 至少1小时间隔
            current_time += interval
            if current_time < hours:
                sample_times.append(int(current_time))

        # 确保至少有最少采样次数
        if len(sample_times) < min_samples:
            sample_times = list(np.linspace(0, hours-1, min_samples, dtype=int))

        return sample_times

    # 1. 预先生成患者元数据（用于后续过滤）
    patient_meta = {}
    for pid in all_patient_ids:
        # 年龄 (40-85岁)
        age = np.random.uniform(40, 85)
        # 性别
        sex = np.random.choice(['M', 'F'])
        # 死亡率 15%
        death = 1 if np.random.random() < 0.15 else 0
        # ICU住院时长 (1-14天转换为小时)
        los_icu = np.random.uniform(24, 14*24)  # 改为小时
        # 30% 概率患 sepsis
        is_septic = np.random.random() < 0.3
        # 发病时间随机分布在 10h ~ hours-10h 之间
        onset = np.random.choice(range(10, max(11, hours-10))) if is_septic else -999

        # 确定感染窗口 (samp time)
        samp_time = -1
        if is_septic:
            # 采样时间通常在发病前后
            samp_time = onset + np.random.randint(-4, 4)
            samp_time = max(0, min(hours-1, samp_time))
        has_aki = is_septic or (np.random.random() < 0.18)
        has_circ_failure = is_septic or (np.random.random() < 0.12)
        has_mech_vent = has_circ_failure or (np.random.random() < 0.22)
        has_rrt = has_aki and (np.random.random() < 0.28)
        has_ards = has_mech_vent and (np.random.random() < 0.3)
        has_pneumonia = is_septic or has_ards or (np.random.random() < 0.2)
        has_heart_failure = has_circ_failure or (np.random.random() < 0.14)
        has_ami = has_heart_failure and (np.random.random() < 0.35)
        has_stroke = np.random.random() < 0.1

        patient_meta[pid] = {
            'age': age,
            'sex': sex,
            'death': death,
            'los_icu': los_icu,
            'is_septic': is_septic,
            'has_aki': has_aki,
            'has_circ_failure': has_circ_failure,
            'has_mech_vent': has_mech_vent,
            'has_rrt': has_rrt,
            'has_ards': has_ards,
            'has_pneumonia': has_pneumonia,
            'has_heart_failure': has_heart_failure,
            'has_ami': has_ami,
            'has_stroke': has_stroke,
            'onset': onset,
            'samp_time': samp_time
        }

    # 2. 应用队列过滤器
    filtered_patient_ids = all_patient_ids
    if cohort_filter:
        filtered_patient_ids = []
        for pid in all_patient_ids:
            meta = patient_meta[pid]
            include = True

            # 年龄过滤
            if cohort_filter.get('age_min') is not None:
                if meta['age'] < cohort_filter['age_min']:
                    include = False
            if cohort_filter.get('age_max') is not None:
                if meta['age'] > cohort_filter['age_max']:
                    include = False

            # 性别过滤
            if cohort_filter.get('gender') is not None:
                if meta['sex'] != cohort_filter['gender']:
                    include = False

            # 存活状态过滤
            if cohort_filter.get('survived') is not None:
                if cohort_filter['survived'] and meta['death'] == 1:
                    include = False
                elif not cohort_filter['survived'] and meta['death'] == 0:
                    include = False

            # Sepsis过滤
            if cohort_filter.get('has_sepsis') is not None:
                if cohort_filter['has_sepsis'] and not meta['is_septic']:
                    include = False
                elif not cohort_filter['has_sepsis'] and meta['is_septic']:
                    include = False

            disease_cohort = cohort_filter.get('disease_cohort')
            if disease_cohort == 'sepsis' and not meta['is_septic']:
                include = False
            elif disease_cohort == 'aki' and not meta['has_aki']:
                include = False
            elif disease_cohort == 'circ_failure' and not meta['has_circ_failure']:
                include = False
            elif disease_cohort == 'mech_vent' and not meta['has_mech_vent']:
                include = False
            elif disease_cohort == 'rrt' and not meta['has_rrt']:
                include = False
            elif disease_cohort == 'ards' and not meta['has_ards']:
                include = False
            elif disease_cohort == 'pneumonia' and not meta['has_pneumonia']:
                include = False
            elif disease_cohort == 'heart_failure' and not meta['has_heart_failure']:
                include = False
            elif disease_cohort == 'ami' and not meta['has_ami']:
                include = False
            elif disease_cohort == 'stroke' and not meta['has_stroke']:
                include = False

            # 住院时长过滤
            if cohort_filter.get('los_min') is not None:
                if meta['los_icu'] < cohort_filter['los_min']:
                    include = False

            if include:
                filtered_patient_ids.append(pid)

        # 如果过滤后患者不够，发出警告但仍继续
        if len(filtered_patient_ids) < n_patients:
            print(f"Warning: Only {len(filtered_patient_ids)} patients match cohort criteria (requested {n_patients})")

    # 3. 限制到请求的患者数量
    patient_ids = filtered_patient_ids[:n_patients]

    # 为了兼容后续代码，创建 patient_sepsis_meta
    patient_sepsis_meta = {pid: patient_meta[pid] for pid in patient_ids}

    # 心率（使用患者级随机采样，模拟10%缺失率）
    hr_records = []
    for pid in patient_ids:
        base_hr = np.random.uniform(70, 90)
        # 如果 septic, 心率在发病后升高
        meta = patient_sepsis_meta[pid]

        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间，而非固定间隔
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            # 10%概率缺失（在随机采样基础上再添加随机缺失）
            if np.random.random() < 0.9:
                hr = base_hr + np.sin(t / 6) * 10 + np.random.normal(0, 5)
                if meta['is_septic'] and t >= meta['onset']:
                    hr += 20 # 发病后心率增加

                hr_records.append({'stay_id': pid, 'time': t, 'hr': max(40, min(150, hr))})
    data['hr'] = pd.DataFrame(hr_records)

# MAP（使用患者级随机采样，模拟10%缺失率）
    map_records = []
    for pid in patient_ids:
        base_map = np.random.uniform(65, 85)
        meta = patient_sepsis_meta[pid]

        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            if np.random.random() < 0.9:
                map_val = base_map + np.cos(t / 8) * 8 + np.random.normal(0, 4)
                if meta['is_septic'] and t >= meta['onset']:
                    map_val -= 15 # 发病后血压下降

                map_records.append({'stay_id': pid, 'time': t, 'map': max(40, min(120, map_val))})
    data['map'] = pd.DataFrame(map_records)

    # SBP（使用患者级随机采样，模拟10%缺失率）
    sbp_records = []
    for pid in patient_ids:
        base_sbp = np.random.uniform(110, 140)
        meta = patient_sepsis_meta[pid]

        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            if np.random.random() < 0.9:
                sbp_val = base_sbp + np.sin(t / 5) * 15 + np.random.normal(0, 8)
                if meta['is_septic'] and t >= meta['onset']:
                    sbp_val -= 20

                sbp_records.append({'stay_id': pid, 'time': t, 'sbp': max(70, min(200, sbp_val))})
    data['sbp'] = pd.DataFrame(sbp_records)

    # 体温（使用患者级随机采样，约每4小时）
    temp_records = []
    for pid in patient_ids:
        base_temp = np.random.uniform(36.5, 37.5)
        meta = patient_sepsis_meta[pid]

        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=4, jitter=0.4)
        for t in sample_times:
            temp_val = base_temp + np.random.normal(0, 0.3)
            # 随机发热
            if np.random.random() < 0.1:
                temp_val += 1.5
            # Sepsis 发热
            if meta['is_septic'] and t >= meta['onset']:
                 temp_val += 1.2

            temp_records.append({'stay_id': pid, 'time': t, 'temp': max(35, min(41, temp_val))})
    data['temp'] = pd.DataFrame(temp_records)

    # 呼吸（使用患者级随机采样，模拟15%缺失率）
    resp_records = []
    for pid in patient_ids:
        base_resp = np.random.uniform(14, 18)
        meta = patient_sepsis_meta[pid]

        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            # 15%概率缺失
            if np.random.random() < 0.85:
                resp_val = base_resp + np.random.normal(0, 2)
                if meta['is_septic'] and t >= meta['onset']:
                    resp_val += 8

                resp_records.append({'stay_id': pid, 'time': t, 'resp': max(8, min(40, resp_val))})
    data['resp'] = pd.DataFrame(resp_records)

    # SpO2（使用患者级随机采样，模拟10%缺失率）
    spo2_records = []
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            if np.random.random() < 0.9:
                spo2_val = 97 + np.random.normal(0, 2)
                if np.random.random() < 0.05:
                    spo2_val -= 10
                spo2_records.append({'stay_id': pid, 'time': t, 'spo2': max(80, min(100, spo2_val))})
    data['spo2'] = pd.DataFrame(spo2_records)

    # EtCO2 (End-Tidal CO2，模拟30%缺失率 - 需要特殊监测）
    etco2_records = []
    for pid in patient_ids:
        base_etco2 = np.random.uniform(35, 42)
        # 仅40%患者有EtCO2监测
        if np.random.random() < 0.4:
            for t in time_points:
                if np.random.random() < 0.7:
                    etco2_val = base_etco2 + np.random.normal(0, 3)
                    etco2_records.append({'stay_id': pid, 'time': t, 'etco2': max(20, min(60, etco2_val))})
    data['etco2'] = pd.DataFrame(etco2_records) if etco2_records else pd.DataFrame(columns=['stay_id', 'time', 'etco2'])

    # O2Sat (Oxygen Saturation - alias for spo2)
    data['o2sat'] = data['spo2'].rename(columns={'spo2': 'o2sat'}).copy() if not data['spo2'].empty else pd.DataFrame(columns=['stay_id', 'time', 'o2sat'])
    data['sao2'] = data['spo2'].rename(columns={'spo2': 'sao2'}).copy() if not data['spo2'].empty else pd.DataFrame(columns=['stay_id', 'time', 'sao2'])

    # SOFA（使用患者级随机采样，约每6小时）
    sofa_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]

        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=6, jitter=0.3)
        for t in sample_times:
            # 基础分布
            probs = [0.6, 0.3, 0.1, 0.0, 0.0]

            # 如果是 sepsis 患者且处于发病期，概率向高分偏移
            if meta['is_septic'] and t >= meta['onset']:
                probs = [0.1, 0.2, 0.3, 0.25, 0.15]

            sofa_resp = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_coag = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_liver = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_cardio = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_cns = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_renal = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_total = sofa_resp + sofa_coag + sofa_liver + sofa_cardio + sofa_cns + sofa_renal

            sofa_records.append({
                'stay_id': pid, 'time': t, 'sofa': sofa_total,
                'sofa_resp': sofa_resp, 'sofa_coag': sofa_coag, 'sofa_liver': sofa_liver,
                'sofa_cardio': sofa_cardio, 'sofa_cns': sofa_cns, 'sofa_renal': sofa_renal,
            })
    data['sofa'] = pd.DataFrame(sofa_records)
    # 🔧 FIX: 从父 DF 中提取子组件为独立 DF，然后剥离父 DF 中的子组件列
    # 避免合并时产生 _x/_y 后缀冲突
    _sofa_sub_cols = ['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal']
    for _sc in _sofa_sub_cols:
        data[_sc] = data['sofa'][['stay_id', 'time', _sc]].copy()
    data['sofa'] = data['sofa'][['stay_id', 'time', 'sofa']].copy()

    # 肌酐（使用患者级随机采样，约每8小时）
    crea_records = []
    for pid in patient_ids:
        base_crea = np.random.uniform(0.8, 1.2)
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=8, jitter=0.3)
        for t in sample_times:
            crea_val = base_crea + np.random.normal(0, 0.2)
            crea_records.append({'stay_id': pid, 'time': t, 'crea': max(0.3, crea_val)})
    data['crea'] = pd.DataFrame(crea_records)

    # 胆红素（使用患者级随机采样，约每12小时）
    bili_records = []
    for pid in patient_ids:
        base_bili = np.random.uniform(0.5, 1.5)
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=12, jitter=0.3)
        for t in sample_times:
            bili_val = base_bili + np.random.normal(0, 0.3)
            bili_records.append({'stay_id': pid, 'time': t, 'bili': max(0.1, bili_val)})
    data['bili'] = pd.DataFrame(bili_records)

    # 血糖 (Glucose，使用患者级随机采样，约每4小时)
    glu_records = []
    for pid in patient_ids:
        base_glu = np.random.uniform(80, 120)
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=4, jitter=0.3)
        for t in sample_times:
            glu_val = base_glu + np.random.normal(0, 15)
            glu_records.append({'stay_id': pid, 'time': t, 'glu': max(40, min(400, glu_val))})
    data['glu'] = pd.DataFrame(glu_records)

    # 乳酸（使用患者级随机采样，约每6小时）
    lac_records = []
    for pid in patient_ids:
        base_lac = np.random.uniform(1.0, 2.0)
        meta = patient_sepsis_meta[pid]

        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=6, jitter=0.3)
        for t in sample_times:
            lac_val = base_lac + np.random.normal(0, 0.5)
            if meta['is_septic'] and t >= meta['onset']:
                lac_val += 3.0 # 乳酸升高

            lac_records.append({'stay_id': pid, 'time': t, 'lact': max(0.5, lac_val)})  # 🔧 改为 lact（标准名称）
    data['lact'] = pd.DataFrame(lac_records)  # 🔧 改为 lact（与 CONCEPT_GROUPS_INTERNAL 一致）

    # 血小板（使用患者级随机采样，约每12小时）
    plt_records = []
    for pid in patient_ids:
        base_plt = np.random.uniform(150, 300)
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=12, jitter=0.3)
        for t in sample_times:
            plt_val = base_plt + np.random.normal(0, 30)
            plt_records.append({'stay_id': pid, 'time': t, 'plt': max(10, plt_val)})
    data['plt'] = pd.DataFrame(plt_records)

    # 去甲肾上腺素（使用患者级随机采样）
    norepi_records = []
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            if 12 <= t <= 48 and np.random.random() < 0.6:
                rate = np.random.uniform(0.05, 0.3)
                norepi_records.append({'stay_id': pid, 'time': t, 'norepi_rate': rate})
    data['norepi_rate'] = pd.DataFrame(norepi_records) if norepi_records else pd.DataFrame(
        columns=['stay_id', 'time', 'norepi_rate'])

    # SOFA-2 评分 (2025新标准，使用患者级随机采样，约每6小时)
    sofa2_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]

        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=6, jitter=0.3)
        for t in sample_times:
            # 基础分布
            probs = [0.55, 0.3, 0.1, 0.05, 0.0]
            if meta['is_septic'] and t >= meta['onset']:
                probs = [0.1, 0.2, 0.3, 0.25, 0.15]

            sofa2_resp = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_coag = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_liver = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_cardio = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_cns = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_renal = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_total = sofa2_resp + sofa2_coag + sofa2_liver + sofa2_cardio + sofa2_cns + sofa2_renal

            sofa2_records.append({
                'stay_id': pid, 'time': t, 'sofa2': sofa2_total,
                'sofa2_resp': sofa2_resp, 'sofa2_coag': sofa2_coag, 'sofa2_liver': sofa2_liver,
                'sofa2_cardio': sofa2_cardio, 'sofa2_cns': sofa2_cns, 'sofa2_renal': sofa2_renal,
            })
    data['sofa2'] = pd.DataFrame(sofa2_records)
    # 🔧 FIX: 提取子组件并剥离父 DF，避免合并 _x/_y 冲突
    _sofa2_sub_cols = ['sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal']
    for _sc in _sofa2_sub_cols:
        data[_sc] = data['sofa2'][['stay_id', 'time', _sc]].copy()
    data['sofa2'] = data['sofa2'][['stay_id', 'time', 'sofa2']].copy()

    # Sepsis-3 诊断数据 (严格基于 SOFA 变化)
    sep3_sofa2_records = []

    # 先把 sofa2 转换为 (stay_id, time) 索引以便查询
    sofa2_lookup = data['sofa2'].set_index(['stay_id', 'time'])['sofa2']

    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]

        # 1. 疑似感染 susp_inf
        # 定义: 在发病前24h到发病后72h为“疑似感染窗口”
        # 这里简化：只要是septic患者，我们在 onset 之后标记疑似
        # samp 仅在具体的采样点为 1

        for t in time_points:
            # 默认值
            susp_inf_val = 0
            samp_val = 0
            sep3_val = 0
            infection_icd_val = 0

            if meta['is_septic']:
                # infection_icd 通常是静态诊断，整程为1
                infection_icd_val = 1

                # samp: 仅在采样点有值 (稀疏)
                if t == meta['samp_time']:
                    samp_val = 1

                # susp_inf: 模拟有一个 suspicions window
                # 此处设为：samp_time 前24h 到 后72h
                samp_t = meta['samp_time']
                if samp_t >= 0 and (samp_t - 24 <= t <= samp_t + 72):
                    susp_inf_val = 1

                # Sepsis-3: susp_inf=1 AND (current_sofa >= 2)
                # 真实标准是 delta_sofa >= 2，假设 baseline=0，则 absolute>=2
                try:
                    # 注意：sofa数据是每6小时采样的，中间时间点可能在dataframe里没有
                    # 这里通过 lookup 查找最近的有效值，或者因为我们的 time_points 包含所有点
                    # 但 sofa2_df 是 dense 的吗？ generate logic 是 time_points[::6]
                    # 我们需要插值或对齐。为简化 mock，我们在生成 sofa2 时只生成了部分点
                    # 但 export 要求对齐。这里我们简单处理：若无法查到准确sofa，则沿用上一个
                    pass
                except:
                    pass

            sep3_sofa2_records.append({
                'stay_id': pid,
                'time': t,
                'sep3_sofa2': 0, # 稍后计算
                'susp_inf': susp_inf_val,
                'infection_icd': infection_icd_val,
                'samp': samp_val,
            })

    # 计算 Sep3 结果 (需要合并 sofa 数据)
    sep3_df = pd.DataFrame(sep3_sofa2_records)

    # 将 Sep3 表和 SOFA2 表合并计算最终 Sep3 状态
    # SOFA2 是每6小时一点，我们先 forward fill 到每小时
    sofa2_full = pd.DataFrame({'stay_id': patient_ids, 'key': 1}).merge(pd.DataFrame({'time': time_points, 'key': 1}), on='key').drop(columns=['key'])
    sofa2_source = data['sofa2'][['stay_id', 'time', 'sofa2']]
    sofa2_interpolated = sofa2_full.merge(sofa2_source, on=['stay_id', 'time'], how='left')
    sofa2_interpolated['sofa2'] = sofa2_interpolated.groupby('stay_id')['sofa2'].ffill().fillna(0)

    # 合并
    sep3_final = sep3_df.merge(sofa2_interpolated, on=['stay_id', 'time'], how='left')

    # 应用 Sepsis3 规则: susp_inf == 1 AND sofa2 >= 2
    sep3_final['sep3_sofa2'] = ((sep3_final['susp_inf'] == 1) & (sep3_final['sofa2'] >= 2)).astype(int)

    # 更新到 data
    # 🔧 保留所有时间点（包括0和1），模拟真实的稀疏性和缺失率
    # sep3_sofa2: 保留所有疑似感染窗口内的时间点（含0和1）
    sep3_with_context = sep3_final[
        (sep3_final['susp_inf'] == 1) |  # 疑似感染窗口
        (sep3_final['infection_icd'] == 1)  # 有感染诊断
    ].copy()

    # 对于 sep3_sofa2，保留疑似感染窗口内的所有记录（包含0值，模拟缺失率）
    data['sep3_sofa2'] = sep3_with_context[['stay_id', 'time', 'sep3_sofa2']] if len(sep3_with_context) > 0 else pd.DataFrame(columns=['stay_id', 'time', 'sep3_sofa2'])
    data['susp_inf'] = sep3_with_context[['stay_id', 'time', 'susp_inf']] if len(sep3_with_context) > 0 else pd.DataFrame(columns=['stay_id', 'time', 'susp_inf'])
    data['infection_icd'] = sep3_with_context[['stay_id', 'time', 'infection_icd']] if len(sep3_with_context) > 0 else pd.DataFrame(columns=['stay_id', 'time', 'infection_icd'])
    data['samp'] = sep3_final[sep3_final['samp'] == 1][['stay_id', 'time', 'samp']] if (sep3_final['samp'] == 1).any() else pd.DataFrame(columns=['stay_id', 'time', 'samp'])

    # 🔧 删除组合概念别名（与 CONCEPT_GROUPS_INTERNAL 保持一致）
    # 删除: sep3_sofa2_susp_inf, sep3_sofa2_samp, sep3_sofa2_infection_icd

    # Sepsis-3 (SOFA-1) 同理
    sofa1_source = data['sofa'][['stay_id', 'time', 'sofa']]
    sofa1_interpolated = sofa2_full.merge(sofa1_source, on=['stay_id', 'time'], how='left')
    sofa1_interpolated['sofa'] = sofa1_interpolated.groupby('stay_id')['sofa'].ffill().fillna(0)

    sep3_sofa1_final = sep3_final[['stay_id', 'time', 'susp_inf', 'infection_icd']].merge(sofa1_interpolated, on=['stay_id', 'time'], how='left')
    sep3_sofa1_final['sep3_sofa1'] = ((sep3_sofa1_final['susp_inf'] == 1) & (sep3_sofa1_final['sofa'] >= 2)).astype(int)

    # sep3_sofa1: 保留所有在感染窗口内的记录（包括 0 和 1），模拟真实缺失率
    sep3_sofa1_in_window = sep3_sofa1_final[(sep3_sofa1_final['susp_inf'] == 1) | (sep3_sofa1_final['infection_icd'] == 1)]
    data['sep3_sofa1'] = sep3_sofa1_in_window[['stay_id', 'time', 'sep3_sofa1']] if len(sep3_sofa1_in_window) > 0 else pd.DataFrame(columns=['stay_id', 'time', 'sep3_sofa1'])

    # 🔧 SOFA-1 子组件已在上方提取并剥离，此处无需重复

    # ============ 补充更多常用概念 ============

    # DBP (舒张压，模拟10%缺失率）
    dbp_records = []
    for pid in patient_ids:
        base_dbp = np.random.uniform(60, 80)
        for t in time_points:
            if np.random.random() < 0.9:
                dbp_val = base_dbp + np.sin(t / 5) * 8 + np.random.normal(0, 5)
                dbp_records.append({'stay_id': pid, 'time': t, 'dbp': max(40, min(110, dbp_val))})
    data['dbp'] = pd.DataFrame(dbp_records)

    # GCS (格拉斯哥昏迷评分，使用患者级随机采样，约每4小时)
    gcs_records = []
    for pid in patient_ids:
        base_gcs = np.random.choice([15, 14, 13, 12, 10, 8], p=[0.5, 0.2, 0.1, 0.08, 0.07, 0.05])
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=4, jitter=0.4)
        for t in sample_times:
            gcs_val = base_gcs + np.random.choice([-1, 0, 0, 0, 1], p=[0.1, 0.3, 0.3, 0.2, 0.1])
            gcs_records.append({'stay_id': pid, 'time': t, 'gcs': max(3, min(15, gcs_val))})
    data['gcs'] = pd.DataFrame(gcs_records)

    # 血气分析：pH, pco2, po2, lact（使用患者级随机采样，约每6小时）
    ph_records = []
    pco2_records = []
    po2_records = []
    for pid in patient_ids:
        base_ph = np.random.uniform(7.35, 7.45)
        base_pco2 = np.random.uniform(35, 45)
        base_po2 = np.random.uniform(80, 100)
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=6, jitter=0.4)
        for t in sample_times:
            ph_records.append({'stay_id': pid, 'time': t, 'ph': base_ph + np.random.normal(0, 0.03)})
            pco2_records.append({'stay_id': pid, 'time': t, 'pco2': base_pco2 + np.random.normal(0, 3)})
            po2_records.append({'stay_id': pid, 'time': t, 'po2': max(60, base_po2 + np.random.normal(0, 10))})
    data['ph'] = pd.DataFrame(ph_records)
    data['pco2'] = pd.DataFrame(pco2_records)
    data['po2'] = pd.DataFrame(po2_records)
    # 🔧 lact 已在上方直接生成（不再需要从 lac 创建别名）

    # 呼吸系统：pafi, fio2, vent_ind（使用患者级随机采样，约每4小时）
    pafi_records = []
    fio2_records = []
    vent_ind_records = []
    for pid in patient_ids:
        base_fio2 = np.random.choice([0.21, 0.3, 0.4, 0.5], p=[0.4, 0.3, 0.2, 0.1])
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=4, jitter=0.4)
        for t in sample_times:
            fio2_val = base_fio2 + np.random.uniform(-0.05, 0.05)
            fio2_val = max(0.21, min(1.0, fio2_val))
            po2_val = 80 + np.random.normal(0, 15)
            pafi_val = po2_val / fio2_val
            vent = 1 if fio2_val > 0.3 else 0
            pafi_records.append({'stay_id': pid, 'time': t, 'pafi': pafi_val})
            fio2_records.append({'stay_id': pid, 'time': t, 'fio2': fio2_val * 100})  # 转为百分比
            vent_ind_records.append({'stay_id': pid, 'time': t, 'vent_ind': vent})
    data['pafi'] = pd.DataFrame(pafi_records)
    data['fio2'] = pd.DataFrame(fio2_records)
    data['vent_ind'] = pd.DataFrame(vent_ind_records)

    # 尿量（使用患者级随机采样，模拟30%缺失率）
    urine_records = []
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=1, jitter=0.3)
        for t in sample_times:
            # 30%概率无记录（缺失）
            if np.random.random() < 0.7:
                urine_val = np.random.uniform(30, 100)
                urine_records.append({'stay_id': pid, 'time': t, 'urine': urine_val})
    data['urine'] = pd.DataFrame(urine_records) if urine_records else pd.DataFrame(columns=['stay_id', 'time', 'urine'])

    # WBC (白细胞，使用患者级随机采样，约每12小时)
    wbc_records = []
    for pid in patient_ids:
        base_wbc = np.random.uniform(6, 12)
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=12, jitter=0.4)
        for t in sample_times:
            wbc_val = base_wbc + np.random.normal(0, 2)
            wbc_records.append({'stay_id': pid, 'time': t, 'wbc': max(1, wbc_val)})
    data['wbc'] = pd.DataFrame(wbc_records)

    # 结局数据 (outcome) - 使用预先生成的元数据
    death_records = []
    los_icu_records = []
    los_hosp_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        death_records.append({'stay_id': pid, 'death': meta['death']})
        los_icu_records.append({'stay_id': pid, 'los_icu': meta['los_icu'] / 24})  # 转为天
        los_hosp = meta['los_icu'] / 24 + np.random.uniform(0, 10)  # 住院时间 >= ICU时间
        los_hosp_records.append({'stay_id': pid, 'los_hosp': los_hosp})
    data['death'] = pd.DataFrame(death_records)
    data['los_icu'] = pd.DataFrame(los_icu_records)
    data['los_hosp'] = pd.DataFrame(los_hosp_records)

    # 人口统计 (demographics) - 使用预先生成的元数据
    age_records = []
    weight_records = []
    height_records = []
    sex_records = []
    bmi_records = []
    adm_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        weight_val = np.random.uniform(50, 100)
        height_val = np.random.uniform(150, 190)
        bmi_val = weight_val / (height_val / 100) ** 2

        age_records.append({'stay_id': pid, 'age': meta['age']})
        weight_records.append({'stay_id': pid, 'weight': weight_val})
        height_records.append({'stay_id': pid, 'height': height_val})
        sex_records.append({'stay_id': pid, 'sex': meta['sex']})
        bmi_records.append({'stay_id': pid, 'bmi': bmi_val})
        adm_records.append({'stay_id': pid, 'adm': 1})  # 所有患者均有入院记录

    data['age'] = pd.DataFrame(age_records)
    data['weight'] = pd.DataFrame(weight_records)
    data['height'] = pd.DataFrame(height_records)
    data['sex'] = pd.DataFrame(sex_records)
    data['bmi'] = pd.DataFrame(bmi_records)
    data['adm'] = pd.DataFrame(adm_records)

    # 其他评分（使用患者级随机采样，约每6小时）
    qsofa_records = []
    sirs_records = []
    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=6, jitter=0.4)
        for t in sample_times:
            qsofa_records.append({'stay_id': pid, 'time': t, 'qsofa': np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])})
            sirs_records.append({'stay_id': pid, 'time': t, 'sirs': np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.25, 0.25, 0.2, 0.1])})
    data['qsofa'] = pd.DataFrame(qsofa_records)
    data['sirs'] = pd.DataFrame(sirs_records)

    # 药物：抗生素使用
    abx_records = []
    for pid in patient_ids:
        abx_records.append({'stay_id': pid, 'abx': 1 if np.random.random() < 0.7 else 0})
    data['abx'] = pd.DataFrame(abx_records)

    # 🔧 FIX (2026-02-03): 药物：皮质类固醇 (corticosteroids) - 只记录发生的事件（NaN/1格式）
    # 只有发生时才记录1，没有发生时不生成记录（而不是生成0）
    cort_records = []
    for pid in patient_ids:
        if np.random.random() < 0.25:  # 25%患者使用皮质类固醇
            start_time = np.random.uniform(0, 24)
            cort_records.append({'stay_id': pid, 'time': start_time, 'cort': 1})
    data['cort'] = pd.DataFrame(cort_records) if cort_records else pd.DataFrame(columns=['stay_id', 'time', 'cort'])

    # ============ KDIGO AKI 急性肾损伤数据（使用患者级随机采样，约每4小时） ============
    aki_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        baseline_crea = np.random.uniform(0.6, 1.2)

        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=4, jitter=0.4)
        for t in sample_times:
            # 基线肌酐附近波动
            crea = baseline_crea * (1 + np.random.normal(0, 0.1))

            # Sepsis 患者在发病后可能发生 AKI
            if meta['is_septic'] and t >= meta['onset']:
                # 30% 概率发生 AKI
                if np.random.random() < 0.3:
                    crea = baseline_crea * np.random.uniform(1.5, 3.0)

            # 计算 AKI 分期
            ratio = crea / baseline_crea
            if ratio >= 3.0 or crea >= 4.0:
                aki_stage = 3
            elif ratio >= 2.0:
                aki_stage = 2
            elif ratio >= 1.5 or crea >= baseline_crea + 0.3:
                aki_stage = 1
            else:
                aki_stage = 0

            aki_records.append({
                'stay_id': pid, 'time': t,
                'crea': round(crea, 2),
                'creat_low_past_7day': round(baseline_crea, 2),
                'aki_stage': aki_stage,
                'aki': 1 if aki_stage > 0 else 0
            })
    aki_full = pd.DataFrame(aki_records)
    data['aki'] = aki_full[['stay_id', 'time', 'aki']].copy()
    # 🔧 FIX: 先从完整 AKI 表派生子组件，再保留精简后的父 DF
    data['aki_stage'] = aki_full[['stay_id', 'time', 'aki_stage']].copy()
    data['creat_low_past_7day'] = aki_full[['stay_id', 'time', 'creat_low_past_7day']].copy()
    # 🔧 添加完整的AKI子特征（基于肌酐、尿量、RRT定义的）
    data['aki_stage_creat'] = aki_full[['stay_id', 'time', 'aki_stage']].copy()
    data['aki_stage_creat'].columns = ['stay_id', 'time', 'aki_stage_creat']
    # 尿量定义的AKI（随机生成，因为demo数据简化）
    aki_uo_records = []
    for _, row in aki_full.iterrows():
        # 尿量AKI通常与肌酐AKI相关但不完全一致
        uo_stage = max(0, row['aki_stage'] - np.random.randint(0, 2)) if row['aki_stage'] > 0 else 0
        aki_uo_records.append({'stay_id': row['stay_id'], 'time': row['time'], 'aki_stage_uo': uo_stage})
    data['aki_stage_uo'] = pd.DataFrame(aki_uo_records)
    # RRT定义的AKI（仅接受RRT的患者为Stage 3）
    aki_rrt_records = []
    for _, row in data['aki_stage'].iterrows():
        rrt_stage = 3 if row['aki_stage'] == 3 and np.random.random() < 0.3 else 0
        aki_rrt_records.append({'stay_id': row['stay_id'], 'time': row['time'], 'aki_stage_rrt': rrt_stage})
    data['aki_stage_rrt'] = pd.DataFrame(aki_rrt_records)

    # ============ 新增 KDIGO 相关特征 (2026-02-04) ============
    # creat_low_past_48hr: 过去48小时内最低肌酐（通常与 creat_low_past_7day 相似或稍高）
    creat_48hr_records = []
    for _, row in data['creat_low_past_7day'].iterrows():
        # 48hr内的最低肌酐通常略高于7天内的最低值
        baseline = row['creat_low_past_7day']
        creat_48hr = round(baseline * np.random.uniform(1.0, 1.15), 2)
        creat_48hr_records.append({'stay_id': row['stay_id'], 'time': row['time'], 'creat_low_past_48hr': creat_48hr})
    data['creat_low_past_48hr'] = pd.DataFrame(creat_48hr_records)

    # 尿量率（mL/kg/h）：基于患者体重的尿量产出率
    # 正常值: 0.5-1.5 mL/kg/h，AKI时 <0.5 mL/kg/h（Stage 1）, <0.3（Stage 2/3）
    uo_rate_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        patient_weight = data['weight'][data['weight']['stay_id'] == pid]['weight'].iloc[0] if len(data['weight'][data['weight']['stay_id'] == pid]) > 0 else 70

        # 使用与AKI相同的时间点
        patient_aki = data['aki_stage'][data['aki_stage']['stay_id'] == pid]
        for _, row in patient_aki.iterrows():
            t = row['time']
            aki_stage = row['aki_stage']

            # 根据AKI分期生成尿量率
            if aki_stage == 0:
                base_uo_rate = np.random.uniform(0.6, 1.5)  # 正常
            elif aki_stage == 1:
                base_uo_rate = np.random.uniform(0.3, 0.5)  # Stage 1: <0.5
            elif aki_stage == 2:
                base_uo_rate = np.random.uniform(0.15, 0.35)  # Stage 2: <0.3
            else:
                base_uo_rate = np.random.uniform(0.0, 0.2)  # Stage 3: <0.3或无尿

            # 6hr, 12hr, 24hr 窗口的尿量率（略有变化）
            uo_6hr = round(base_uo_rate * np.random.uniform(0.9, 1.1), 3)
            uo_12hr = round(base_uo_rate * np.random.uniform(0.85, 1.05), 3)
            uo_24hr = round(base_uo_rate * np.random.uniform(0.8, 1.0), 3)  # 24hr窗口通常更平滑

            uo_rate_records.append({
                'stay_id': pid, 'time': t,
                'uo_rt_6hr': uo_6hr,
                'uo_rt_12hr': uo_12hr,
                'uo_rt_24hr': uo_24hr
            })
    uo_rate_df = pd.DataFrame(uo_rate_records)
    data['uo_rt_6hr'] = uo_rate_df[['stay_id', 'time', 'uo_rt_6hr']].copy()
    data['uo_rt_12hr'] = uo_rate_df[['stay_id', 'time', 'uo_rt_12hr']].copy()
    data['uo_rt_24hr'] = uo_rate_df[['stay_id', 'time', 'uo_rt_24hr']].copy()

    # ============ 循环衰竭 (circEWS) 数据 ============
    circ_failure_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]

        for t in time_points:
            # 基线乳酸和MAP
            base_lact = np.random.uniform(0.8, 1.5)
            base_map = np.random.uniform(75, 95)

            lact = base_lact + np.random.normal(0, 0.3)
            map_val = base_map + np.random.normal(0, 5)

            # Sepsis 患者发病后可能发生循环衰竭
            if meta['is_septic'] and t >= meta['onset']:
                if np.random.random() < 0.4:
                    lact = np.random.uniform(2.5, 8.0)
                    map_val = np.random.uniform(50, 70)

            # 计算循环衰竭事件等级
            lactate_elevated = lact >= 2.0
            map_low = map_val <= 65

            if lactate_elevated and map_low:
                circ_event = np.random.choice([1, 2, 3], p=[0.4, 0.35, 0.25])
            elif lactate_elevated:
                circ_event = 1 if np.random.random() < 0.3 else 0
            else:
                circ_event = 0

            circ_failure_records.append({
                'stay_id': pid, 'time': t,
                'lact': round(lact, 2),
                'map': round(map_val, 1),
                'circ_event': circ_event,
                'circ_failure': 1 if circ_event > 0 else 0
            })
    data['circ_failure'] = pd.DataFrame(circ_failure_records)
    # 🔧 FIX: 提取 circ_event 并剥离父 DF
    data['circ_event'] = data['circ_failure'][['stay_id', 'time', 'circ_event']].copy()
    data['circ_failure'] = data['circ_failure'][['stay_id', 'time', 'circ_failure']].copy()

    # ============ 呼吸机参数（使用患者级随机采样） ============
    peep_records = []
    tidal_vol_records = []
    tidal_vol_set_records = []
    pip_records = []
    plateau_pres_records = []
    mean_airway_pres_records = []
    minute_vol_records = []
    vent_rate_records = []
    compliance_records = []
    driving_pres_records = []
    ps_records = []

    for pid in patient_ids:
        # 仅60%患者有呼吸机参数（模拟非所有患者都需要机械通气）
        has_vent = np.random.random() < 0.6
        if has_vent:
            # 呼吸机开始时间随机在6-24小时之间
            vent_start = np.random.choice(range(6, min(24, hours)))
            # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
            sample_times = get_random_sample_times(pid, base_interval=2, jitter=0.4)
            for t in sample_times:
                if t >= vent_start:
                    # 再有20%概率该时间点缺失记录（设备故障/记录缺失）
                    if np.random.random() < 0.8:
                        peep = np.random.uniform(5, 15)
                        tidal_vol = np.random.uniform(350, 550)
                        tidal_vol_set = np.random.uniform(400, 600)
                        pip = np.random.uniform(15, 35)
                        plateau = np.random.uniform(18, 30)
                        mean_airway = np.random.uniform(10, 20)
                        minute_vol = np.random.uniform(6, 12)
                        rate = np.random.uniform(12, 20)
                        compliance = tidal_vol / max(1, plateau - peep)
                        driving = plateau - peep
                        ps = np.random.uniform(5, 15)

                        peep_records.append({'stay_id': pid, 'time': t, 'peep': peep})
                        tidal_vol_records.append({'stay_id': pid, 'time': t, 'tidal_vol': tidal_vol})
                        # tidal_vol_set较少记录（50%概率）
                        if np.random.random() < 0.5:
                            tidal_vol_set_records.append({'stay_id': pid, 'time': t, 'tidal_vol_set': tidal_vol_set})
                        pip_records.append({'stay_id': pid, 'time': t, 'pip': pip})
                        # plateau_pres和compliance较少记录（40%概率）
                        if np.random.random() < 0.4:
                            plateau_pres_records.append({'stay_id': pid, 'time': t, 'plateau_pres': plateau})
                            compliance_records.append({'stay_id': pid, 'time': t, 'compliance': compliance})
                            driving_pres_records.append({'stay_id': pid, 'time': t, 'driving_pres': driving})
                        # mean_airway和minute_vol中等记录率（60%概率）
                        if np.random.random() < 0.6:
                            mean_airway_pres_records.append({'stay_id': pid, 'time': t, 'mean_airway_pres': mean_airway})
                            minute_vol_records.append({'stay_id': pid, 'time': t, 'minute_vol': minute_vol})
                        vent_rate_records.append({'stay_id': pid, 'time': t, 'vent_rate': rate})
                        ps_records.append({'stay_id': pid, 'time': t, 'ps': ps})

    data['peep'] = pd.DataFrame(peep_records) if peep_records else pd.DataFrame(columns=['stay_id', 'time', 'peep'])
    data['tidal_vol'] = pd.DataFrame(tidal_vol_records) if tidal_vol_records else pd.DataFrame(columns=['stay_id', 'time', 'tidal_vol'])
    data['tidal_vol_set'] = pd.DataFrame(tidal_vol_set_records) if tidal_vol_set_records else pd.DataFrame(columns=['stay_id', 'time', 'tidal_vol_set'])
    data['pip'] = pd.DataFrame(pip_records) if pip_records else pd.DataFrame(columns=['stay_id', 'time', 'pip'])
    data['plateau_pres'] = pd.DataFrame(plateau_pres_records) if plateau_pres_records else pd.DataFrame(columns=['stay_id', 'time', 'plateau_pres'])
    data['mean_airway_pres'] = pd.DataFrame(mean_airway_pres_records) if mean_airway_pres_records else pd.DataFrame(columns=['stay_id', 'time', 'mean_airway_pres'])
    data['minute_vol'] = pd.DataFrame(minute_vol_records) if minute_vol_records else pd.DataFrame(columns=['stay_id', 'time', 'minute_vol'])
    data['vent_rate'] = pd.DataFrame(vent_rate_records) if vent_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'vent_rate'])
    data['compliance'] = pd.DataFrame(compliance_records) if compliance_records else pd.DataFrame(columns=['stay_id', 'time', 'compliance'])
    data['driving_pres'] = pd.DataFrame(driving_pres_records) if driving_pres_records else pd.DataFrame(columns=['stay_id', 'time', 'driving_pres'])
    data['ps'] = pd.DataFrame(ps_records) if ps_records else pd.DataFrame(columns=['stay_id', 'time', 'ps'])

    # ============ 补充更多实验室检查（使用患者级随机采样，约每12小时） ============
    alp_records = []
    bun_records = []
    alt_records = []
    ast_records = []
    ca_records = []
    mg_records = []
    cl_records = []
    ck_records = []
    ckmb_records = []
    tri_records = []
    tnt_records = []
    crp_records = []
    bicar_records = []
    bili_dir_records = []
    alb_records = []
    be_records = []
    cai_records = []
    tco2_records = []

    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=12, jitter=0.4)
        for t in sample_times:
            alp_records.append({'stay_id': pid, 'time': t, 'alp': np.random.uniform(40, 120)})
            bun_records.append({'stay_id': pid, 'time': t, 'bun': np.random.uniform(10, 40)})
            alt_records.append({'stay_id': pid, 'time': t, 'alt': np.random.uniform(10, 60)})
            ast_records.append({'stay_id': pid, 'time': t, 'ast': np.random.uniform(10, 60)})
            ca_records.append({'stay_id': pid, 'time': t, 'ca': np.random.uniform(8.5, 10.5)})
            mg_records.append({'stay_id': pid, 'time': t, 'mg': np.random.uniform(1.5, 2.5)})
            cl_records.append({'stay_id': pid, 'time': t, 'cl': np.random.uniform(95, 110)})
            ck_records.append({'stay_id': pid, 'time': t, 'ck': np.random.uniform(50, 300)})
            ckmb_records.append({'stay_id': pid, 'time': t, 'ckmb': np.random.uniform(0, 10)})
            tri_records.append({'stay_id': pid, 'time': t, 'tri': np.random.uniform(0, 0.5)})
            tnt_records.append({'stay_id': pid, 'time': t, 'tnt': np.random.uniform(0, 0.5)})
            crp_records.append({'stay_id': pid, 'time': t, 'crp': np.random.uniform(5, 100)})
            bicar_records.append({'stay_id': pid, 'time': t, 'bicar': np.random.uniform(22, 28)})
            bili_dir_records.append({'stay_id': pid, 'time': t, 'bili_dir': np.random.uniform(0.1, 0.5)})
            alb_records.append({'stay_id': pid, 'time': t, 'alb': np.random.uniform(3.0, 4.5)})
            be_records.append({'stay_id': pid, 'time': t, 'be': np.random.uniform(-3, 3)})
            cai_records.append({'stay_id': pid, 'time': t, 'cai': np.random.uniform(1.1, 1.3)})
            tco2_records.append({'stay_id': pid, 'time': t, 'tco2': np.random.uniform(23, 29)})

    data['alp'] = pd.DataFrame(alp_records)
    data['bun'] = pd.DataFrame(bun_records)
    data['alt'] = pd.DataFrame(alt_records)
    data['ast'] = pd.DataFrame(ast_records)
    data['ca'] = pd.DataFrame(ca_records)
    data['mg'] = pd.DataFrame(mg_records)
    data['cl'] = pd.DataFrame(cl_records)
    data['ck'] = pd.DataFrame(ck_records)
    data['ckmb'] = pd.DataFrame(ckmb_records)
    data['tri'] = pd.DataFrame(tri_records)
    data['tnt'] = pd.DataFrame(tnt_records)
    data['crp'] = pd.DataFrame(crp_records)
    data['bicar'] = pd.DataFrame(bicar_records)
    data['bili_dir'] = pd.DataFrame(bili_dir_records)
    data['alb'] = pd.DataFrame(alb_records)
    data['be'] = pd.DataFrame(be_records)
    data['cai'] = pd.DataFrame(cai_records)
    data['tco2'] = pd.DataFrame(tco2_records)
    # 🔧 删除别名概念（与 CONCEPT_GROUPS_INTERNAL 保持一致）
    # 删除: bicarb (bicar的别名), potassium (k的别名)

    # ============ 血液学扩展（使用患者级随机采样，约每12小时） ============
    hct_records = []
    rbc_records = []
    rdw_records = []
    mcv_records = []
    mch_records = []
    mchc_records = []
    neut_records = []
    lymph_records = []
    eos_records = []
    basos_records = []
    bnd_records = []
    inr_pt_records = []
    ptt_records = []
    pt_records = []
    fgn_records = []
    esr_records = []
    hba1c_records = []

    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=12, jitter=0.4)
        for t in sample_times:
            hct_records.append({'stay_id': pid, 'time': t, 'hct': np.random.uniform(30, 45)})
            rbc_records.append({'stay_id': pid, 'time': t, 'rbc': np.random.uniform(3.5, 5.5)})
            rdw_records.append({'stay_id': pid, 'time': t, 'rdw': np.random.uniform(11, 15)})
            mcv_records.append({'stay_id': pid, 'time': t, 'mcv': np.random.uniform(80, 100)})
            mch_records.append({'stay_id': pid, 'time': t, 'mch': np.random.uniform(27, 32)})
            mchc_records.append({'stay_id': pid, 'time': t, 'mchc': np.random.uniform(32, 36)})
            neut_records.append({'stay_id': pid, 'time': t, 'neut': np.random.uniform(40, 75)})
            lymph_records.append({'stay_id': pid, 'time': t, 'lymph': np.random.uniform(20, 40)})
            eos_records.append({'stay_id': pid, 'time': t, 'eos': np.random.uniform(1, 5)})
            basos_records.append({'stay_id': pid, 'time': t, 'basos': np.random.uniform(0, 2)})
            bnd_records.append({'stay_id': pid, 'time': t, 'bnd': np.random.uniform(0, 10)})
            inr_pt_records.append({'stay_id': pid, 'time': t, 'inr_pt': np.random.uniform(0.9, 1.3)})
            ptt_records.append({'stay_id': pid, 'time': t, 'ptt': np.random.uniform(25, 35)})
            pt_records.append({'stay_id': pid, 'time': t, 'pt': np.random.uniform(11, 14)})
            fgn_records.append({'stay_id': pid, 'time': t, 'fgn': np.random.uniform(200, 400)})
            esr_records.append({'stay_id': pid, 'time': t, 'esr': np.random.uniform(5, 25)})
            hba1c_records.append({'stay_id': pid, 'time': t, 'hba1c': np.random.uniform(5.0, 7.0)})

    data['hct'] = pd.DataFrame(hct_records)
    data['rbc'] = pd.DataFrame(rbc_records)
    data['rdw'] = pd.DataFrame(rdw_records)
    data['mcv'] = pd.DataFrame(mcv_records)
    data['mch'] = pd.DataFrame(mch_records)
    data['mchc'] = pd.DataFrame(mchc_records)
    data['neut'] = pd.DataFrame(neut_records)
    data['lymph'] = pd.DataFrame(lymph_records)
    data['eos'] = pd.DataFrame(eos_records)
    data['basos'] = pd.DataFrame(basos_records)
    data['bnd'] = pd.DataFrame(bnd_records)
    data['inr_pt'] = pd.DataFrame(inr_pt_records)
    data['ptt'] = pd.DataFrame(ptt_records)
    data['pt'] = pd.DataFrame(pt_records)
    data['fgn'] = pd.DataFrame(fgn_records)
    data['esr'] = pd.DataFrame(esr_records)
    data['hba1c'] = pd.DataFrame(hba1c_records)

    # ============ 更多药物 ============
    dopa_rate_records = []
    dopa_dur_records = []
    dopa60_records = []
    epi_dur_records = []
    epi_rate_records = []
    epi60_records = []
    norepi_rate_records = []
    norepi_dur_records = []
    norepi60_records = []
    adh_rate_records = []
    phn_rate_records = []
    dobu_rate_records = []
    dobu_dur_records = []
    dobu60_records = []
    ins_records = []
    dex_records = []

    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=3, jitter=0.4)
        for t in sample_times:
            if np.random.random() < 0.3:
                dopa_rate_records.append({'stay_id': pid, 'time': t, 'dopa_rate': np.random.uniform(2, 10)})
                epi_rate_records.append({'stay_id': pid, 'time': t, 'epi_rate': np.random.uniform(0.01, 0.1)})
                norepi_rate_records.append({'stay_id': pid, 'time': t, 'norepi_rate': np.random.uniform(0.01, 1.0)})
                dobu_rate_records.append({'stay_id': pid, 'time': t, 'dobu_rate': np.random.uniform(2, 10)})
                adh_rate_records.append({'stay_id': pid, 'time': t, 'adh_rate': np.random.uniform(0.01, 0.04)})
                phn_rate_records.append({'stay_id': pid, 'time': t, 'phn_rate': np.random.uniform(0.1, 0.5)})

        dopa_dur_records.append({'stay_id': pid, 'dopa_dur': np.random.uniform(0, 48)})
        epi_dur_records.append({'stay_id': pid, 'epi_dur': np.random.uniform(0, 24)})
        norepi_dur_records.append({'stay_id': pid, 'norepi_dur': np.random.uniform(0, 72)})
        dobu_dur_records.append({'stay_id': pid, 'dobu_dur': np.random.uniform(0, 36)})
        dopa60_records.append({'stay_id': pid, 'dopa60': 1 if np.random.random() < 0.4 else 0})
        epi60_records.append({'stay_id': pid, 'epi60': 1 if np.random.random() < 0.3 else 0})
        norepi60_records.append({'stay_id': pid, 'norepi60': 1 if np.random.random() < 0.5 else 0})
        dobu60_records.append({'stay_id': pid, 'dobu60': 1 if np.random.random() < 0.3 else 0})
        ins_records.append({'stay_id': pid, 'ins': np.random.uniform(0, 10)})
        dex_records.append({'stay_id': pid, 'dex': np.random.uniform(0, 1.5)})

    data['dopa_rate'] = pd.DataFrame(dopa_rate_records) if dopa_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'dopa_rate'])
    data['dopa_dur'] = pd.DataFrame(dopa_dur_records)
    data['dopa60'] = pd.DataFrame(dopa60_records)
    data['epi_rate'] = pd.DataFrame(epi_rate_records) if epi_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'epi_rate'])
    data['epi_dur'] = pd.DataFrame(epi_dur_records)
    data['epi60'] = pd.DataFrame(epi60_records)
    data['norepi_rate'] = pd.DataFrame(norepi_rate_records) if norepi_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'norepi_rate'])
    data['norepi_dur'] = pd.DataFrame(norepi_dur_records)
    data['norepi60'] = pd.DataFrame(norepi60_records)
    data['adh_rate'] = pd.DataFrame(adh_rate_records) if adh_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'adh_rate'])
    data['phn_rate'] = pd.DataFrame(phn_rate_records) if phn_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'phn_rate'])
    data['dobu_rate'] = pd.DataFrame(dobu_rate_records) if dobu_rate_records else pd.DataFrame(columns=['stay_id', 'time', 'dobu_rate'])
    data['dobu_dur'] = pd.DataFrame(dobu_dur_records)
    data['dobu60'] = pd.DataFrame(dobu60_records)
    data['ins'] = pd.DataFrame(ins_records)
    data['dex'] = pd.DataFrame(dex_records)
    if 'norepi_rate' in data and not data['norepi_rate'].empty:
        data['norepi_equiv'] = data['norepi_rate'].rename(columns={'norepi_rate': 'norepi_equiv'}).copy()
    else:
        data['norepi_equiv'] = pd.DataFrame(columns=['stay_id', 'time', 'norepi_equiv'])

    # vaso_ind (血管活性药物指示)
    vaso_ind_records = []
    for pid in patient_ids:
        vaso_ind_records.append({'stay_id': pid, 'vaso_ind': 1 if np.random.random() < 0.6 else 0})
    data['vaso_ind'] = pd.DataFrame(vaso_ind_records)

    # ============ 神经和其他支持（使用患者级随机采样） ============
    rass_records = []
    avpu_records = []
    egcs_records = []
    mgcs_records = []
    vgcs_records = []
    tgcs_records = []
    sedated_gcs_records = []

    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times = get_random_sample_times(pid, base_interval=6, jitter=0.4)
        for t in sample_times:
            rass_records.append({'stay_id': pid, 'time': t, 'rass': np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], p=[0.05, 0.1, 0.15, 0.2, 0.15, 0.2, 0.05, 0.05, 0.03, 0.02])})
            egcs_records.append({'stay_id': pid, 'time': t, 'egcs': np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.4])})
            mgcs_records.append({'stay_id': pid, 'time': t, 'mgcs': np.random.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.1, 0.15, 0.2, 0.25, 0.25])})
            vgcs_records.append({'stay_id': pid, 'time': t, 'vgcs': np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.2, 0.25, 0.3])})
            avpu_records.append({'stay_id': pid, 'time': t, 'avpu': np.random.choice(['A', 'V', 'P', 'U'], p=[0.6, 0.2, 0.1, 0.1])})
        tgcs_records.append({'stay_id': pid, 'tgcs': np.random.choice([15, 14, 13, 12, 10, 8, 6], p=[0.5, 0.2, 0.1, 0.08, 0.07, 0.03, 0.02])})
        sedated_gcs_records.append({'stay_id': pid, 'sedated_gcs': np.random.choice([15, 14, 13], p=[0.7, 0.2, 0.1])})

    data['rass'] = pd.DataFrame(rass_records)
    data['avpu'] = pd.DataFrame(avpu_records)
    data['egcs'] = pd.DataFrame(egcs_records)
    data['mgcs'] = pd.DataFrame(mgcs_records)
    data['vgcs'] = pd.DataFrame(vgcs_records)
    data['tgcs'] = pd.DataFrame(tgcs_records)
    data['sedated_gcs'] = pd.DataFrame(sedated_gcs_records)

    # ============ 其他指标（使用高效循环而非列表推导式）============
    # 静态指标（每患者一个值）
    static_records = {
        'rrt': [], 'ecmo': [], 'height': [], 'bmi': [], 'sex': [], 'adm': [],
        'los_hosp': [], 'vent_start': [], 'vent_end': [], 'cort': []
    }

    # RRT改为时间序列数据（仅10%患者使用）
    rrt_records = []
    rrt_patient_ids = set()  # 记录有RRT的患者ID
    for pid in patient_ids:
        # 10%患者接受RRT
        if np.random.random() < 0.1:
            rrt_patient_ids.add(pid)
            # RRT开始时间随机在12-48小时之间
            rrt_start = np.random.choice(range(12, min(48, hours)))
            for t in time_points:
                if t >= rrt_start:
                    rrt_records.append({'stay_id': pid, 'time': t, 'rrt': 1})
    data['rrt'] = pd.DataFrame(rrt_records) if rrt_records else pd.DataFrame(columns=['stay_id', 'time', 'rrt'])

    # rrt_criteria 也改为时间序列（与rrt相同，但列名不同）
    if rrt_records:
        data['rrt_criteria'] = data['rrt'].rename(columns={'rrt': 'rrt_criteria'}).copy()
    else:
        data['rrt_criteria'] = pd.DataFrame(columns=['stay_id', 'time', 'rrt_criteria'])

    for pid in patient_ids:
        # 保留静态版本用于其他用途（但不覆盖时间序列版本）
        static_records['rrt'].append({'stay_id': pid, 'rrt_static': 1 if pid in rrt_patient_ids else 0})
        # 🔧 FIX (2026-02-03): ecmo只记录发生的事件（NaN/1格式）
        # 只有5%患者使用ECMO，只在发生时记录1
        if np.random.random() < 0.05:
            static_records['ecmo'].append({'stay_id': pid, 'ecmo': 1})
        # 🔧 注意：height, bmi, sex, adm, los_hosp 已在前面使用 patient_sepsis_meta 正确生成
        # 这里只生成那些前面没有生成的静态字段
        static_records['vent_start'].append({'stay_id': pid, 'vent_start': np.random.choice(time_points[:min(24, len(time_points))])})
        static_records['vent_end'].append({'stay_id': pid, 'vent_end': np.random.choice(time_points[-min(24, len(time_points)):])})
        # 🔧 FIX (2026-02-03): cort只记录发生的事件（NaN/1格式）
        if np.random.random() < 0.3:
            static_records['cort'].append({'stay_id': pid, 'cort': 1})

    # 只为非RRT且未在前面生成的静态指标创建DataFrame
    # 🔧 跳过已正确生成的: rrt(时间序列), sex, age, death, los_icu, los_hosp, weight, height, bmi, adm
    already_generated = {'rrt', 'sex', 'age', 'death', 'los_icu', 'los_hosp', 'weight', 'height', 'bmi', 'adm'}
    for key, records in static_records.items():
        if key not in already_generated:
            # 🔧 FIX: 如果记录为空，创建带正确列名的空DataFrame
            if records:
                data[key] = pd.DataFrame(records)
            else:
                # 根据key确定列名
                if key == 'ecmo':
                    data[key] = pd.DataFrame(columns=['stay_id', 'ecmo'])
                elif key == 'cort':
                    data[key] = pd.DataFrame(columns=['stay_id', 'cort'])
                else:
                    data[key] = pd.DataFrame(records)

    # 🔧 注意: ecmo, ecmo_indication, mech_circ_support 在后面的代码中单独生成
    # （约第3298-3320行），此处不再复制，避免生成顺序问题

    # 时间序列指标（使用患者级随机采样）
    mews_records = []
    news_records = []
    hbco_records = []
    methb_records = []
    k_records = []
    na_records = []
    phos_records = []
    hgb_records = []
    safi_records = []

    for pid in patient_ids:
        # 🔧 FIX (2026-02-03): 使用患者级随机采样时间
        sample_times_6h = get_random_sample_times(pid, base_interval=6, jitter=0.4)
        sample_times_12h = get_random_sample_times(pid, base_interval=12, jitter=0.4)
        sample_times_4h = get_random_sample_times(pid, base_interval=4, jitter=0.4)

        for t in sample_times_6h:
            mews_records.append({'stay_id': pid, 'time': t, 'mews': np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03])})
            news_records.append({'stay_id': pid, 'time': t, 'news': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], p=[0.25, 0.2, 0.18, 0.15, 0.1, 0.07, 0.03, 0.02])})

        for t in sample_times_12h:
            if 'k' not in data:
                k_records.append({'stay_id': pid, 'time': t, 'k': np.random.uniform(3.5, 5.0)})
            if 'na' not in data:
                na_records.append({'stay_id': pid, 'time': t, 'na': np.random.uniform(135, 145)})
            if 'phos' not in data:
                phos_records.append({'stay_id': pid, 'time': t, 'phos': np.random.uniform(2.5, 4.5)})
            if 'hgb' not in data:
                hgb_records.append({'stay_id': pid, 'time': t, 'hgb': np.random.uniform(10, 15)})
            hbco_records.append({'stay_id': pid, 'time': t, 'hbco': np.random.uniform(0, 5)})
            methb_records.append({'stay_id': pid, 'time': t, 'methb': np.random.uniform(0, 2)})

        for t in sample_times_4h:
            safi_records.append({'stay_id': pid, 'time': t, 'safi': np.random.uniform(200, 450)})

    data['mews'] = pd.DataFrame(mews_records)
    data['news'] = pd.DataFrame(news_records)
    data['hbco'] = pd.DataFrame(hbco_records)
    data['methb'] = pd.DataFrame(methb_records)
    data['safi'] = pd.DataFrame(safi_records)

    if k_records:
        data['k'] = pd.DataFrame(k_records)
    if na_records:
        data['na'] = pd.DataFrame(na_records)
    if phos_records:
        data['phos'] = pd.DataFrame(phos_records)
    if hgb_records:
        data['hgb'] = pd.DataFrame(hgb_records)

    # 数据复制和别名
    if 'vent_ind' in data and not data['vent_ind'].empty:
        data['mech_vent'] = data['vent_ind'].rename(columns={'vent_ind': 'mech_vent'}).copy()
    else:
        data['mech_vent'] = pd.DataFrame(columns=['stay_id', 'time', 'mech_vent'])

    # vent_start 和 vent_end (机械通气起止时间)
    vent_start_records = []
    vent_end_records = []
    if 'vent_ind' in data and not data['vent_ind'].empty:
        for pid in patient_ids:
            pid_vent = data['vent_ind'][data['vent_ind']['stay_id'] == pid].copy()
            if len(pid_vent) > 0 and (pid_vent['vent_ind'] == 1).any():
                vent_times = pid_vent[pid_vent['vent_ind'] == 1]['time']
                if len(vent_times) > 0:
                    start_t = vent_times.min()
                    end_t = vent_times.max() + 4  # 假设每次测量持续4小时
                    vent_start_records.append({'stay_id': pid, 'time': start_t, 'vent_start': 1})
                    vent_end_records.append({'stay_id': pid, 'time': end_t, 'vent_end': 1})
    data['vent_start'] = pd.DataFrame(vent_start_records) if vent_start_records else pd.DataFrame(columns=['stay_id', 'time', 'vent_start'])
    data['vent_end'] = pd.DataFrame(vent_end_records) if vent_end_records else pd.DataFrame(columns=['stay_id', 'time', 'vent_end'])

    # ECMO (体外膜肺氧合) - 罕见事件，约3%患者
    ecmo_records = []
    ecmo_indication_records = []
    for pid in patient_ids:
        if np.random.random() < 0.03:  # 3%概率使用ECMO
            ecmo_start = np.random.uniform(12, 48)
            ecmo_indication = np.random.choice(['ARDS', 'Cardiogenic_shock', 'Bridge_to_transplant'])
            ecmo_records.append({'stay_id': pid, 'time': ecmo_start, 'ecmo': 1})
            ecmo_indication_records.append({'stay_id': pid, 'time': ecmo_start, 'ecmo_indication': ecmo_indication})
    data['ecmo'] = pd.DataFrame(ecmo_records) if ecmo_records else pd.DataFrame(columns=['stay_id', 'time', 'ecmo'])
    data['ecmo_indication'] = pd.DataFrame(ecmo_indication_records) if ecmo_indication_records else pd.DataFrame(columns=['stay_id', 'time', 'ecmo_indication'])

    # 🔧 FIX (2026-02-04): mech_circ_support - 机械循环支持（IABP/LVAD/Impella/VA-ECMO）
    # 真实数据中非常罕见，约2-3%的ICU患者使用（比ECMO稍多，因为包括IABP等）
    # 这里在ecmo生成之后更新mech_circ_support，确保反映正确的缺失率
    mech_circ_records = []
    for pid in patient_ids:
        # 2.5%概率使用机械循环支持（包括ECMO + IABP + LVAD等）
        if np.random.random() < 0.025:
            mcs_start = np.random.uniform(12, 48)
            mech_circ_records.append({'stay_id': pid, 'time': mcs_start, 'mech_circ_support': 1})
    data['mech_circ_support'] = pd.DataFrame(mech_circ_records) if mech_circ_records else pd.DataFrame(columns=['stay_id', 'time', 'mech_circ_support'])

    if 'fio2' in data and not data['fio2'].empty:
        data['supp_o2'] = data['fio2'].copy()
        data['supp_o2']['supp_o2'] = (data['supp_o2']['fio2'] > 21).astype(int)
        data['supp_o2'] = data['supp_o2'][['stay_id', 'time', 'supp_o2']]
    else:
        data['supp_o2'] = pd.DataFrame(columns=['stay_id', 'time', 'supp_o2'])

    # spo2/sao2 别名处理（避免循环引用）
    if 'spo2' in data and not data['spo2'].empty:
        # spo2已经存在，创建o2sat和sao2别名
        if 'o2sat' not in data or data['o2sat'].empty:
            data['o2sat'] = data['spo2'].rename(columns={'spo2': 'o2sat'}).copy()
        if 'sao2' not in data or data['sao2'].empty:
            data['sao2'] = data['spo2'].rename(columns={'spo2': 'sao2'}).copy()

    if 'gcs' in data and not data['gcs'].empty:
        data['ett_gcs'] = data['gcs'][['stay_id', 'time']].copy()
        data['ett_gcs']['ett_gcs'] = (data['gcs']['gcs'] <= 8).astype(int).to_numpy()
    else:
        data['ett_gcs'] = pd.DataFrame(columns=['stay_id', 'time', 'ett_gcs'])
    # urine24: 24小时累计尿量，每6小时一个记录点（模拟真实采样频率）
    urine24_records = []
    if 'urine' in data and not data['urine'].empty:
        for pid in patient_ids:
            pid_urine = data['urine'][data['urine']['stay_id'] == pid]
            for t in time_points[::6]:  # 每6小时记录一次
                # 计算过去24小时的尿量
                recent_urine = pid_urine[(pid_urine['time'] >= max(0, t-24)) & (pid_urine['time'] <= t)]
                if len(recent_urine) > 0:  # 只有当有数据时才记录
                    urine24_records.append({
                        'stay_id': pid,
                        'time': t,
                        'urine24': recent_urine['urine'].sum()
                    })
    data['urine24'] = pd.DataFrame(urine24_records) if urine24_records else pd.DataFrame(columns=['stay_id', 'time', 'urine24'])

    # === 🆕 新增 12 个缺失的概念（2026-02-03）===

    # 1. uo_6h, uo_12h, uo_24h: 6/12/24小时尿量率 (mL/kg/h)
    uo_6h_records = []
    uo_12h_records = []
    uo_24h_records = []
    if 'urine' in data and not data['urine'].empty and 'weight' in data and not data['weight'].empty:
        weight_dict = data['weight'].set_index('stay_id')['weight'].to_dict()
        for pid in patient_ids:
            if pid not in weight_dict:
                continue
            weight = weight_dict[pid]
            pid_urine = data['urine'][data['urine']['stay_id'] == pid]

            for t in time_points[::3]:  # 每3小时采样一次
                # 6小时尿量率
                recent_6h = pid_urine[(pid_urine['time'] >= max(0, t-6)) & (pid_urine['time'] <= t)]
                if len(recent_6h) > 0:
                    uo_6h = recent_6h['urine'].sum() / weight / 6.0
                    uo_6h_records.append({'stay_id': pid, 'time': t, 'uo_6h': uo_6h})

                # 12小时尿量率
                recent_12h = pid_urine[(pid_urine['time'] >= max(0, t-12)) & (pid_urine['time'] <= t)]
                if len(recent_12h) > 0:
                    uo_12h = recent_12h['urine'].sum() / weight / 12.0
                    uo_12h_records.append({'stay_id': pid, 'time': t, 'uo_12h': uo_12h})

                # 24小时尿量率
                recent_24h = pid_urine[(pid_urine['time'] >= max(0, t-24)) & (pid_urine['time'] <= t)]
                if len(recent_24h) > 0:
                    uo_24h = recent_24h['urine'].sum() / weight / 24.0
                    uo_24h_records.append({'stay_id': pid, 'time': t, 'uo_24h': uo_24h})

    data['uo_6h'] = pd.DataFrame(uo_6h_records) if uo_6h_records else pd.DataFrame(columns=['stay_id', 'time', 'uo_6h'])
    data['uo_12h'] = pd.DataFrame(uo_12h_records) if uo_12h_records else pd.DataFrame(columns=['stay_id', 'time', 'uo_12h'])
    data['uo_24h'] = pd.DataFrame(uo_24h_records) if uo_24h_records else pd.DataFrame(columns=['stay_id', 'time', 'uo_24h'])

    # 🔧 2026-02-04: 移除了重复的 kdigo_creat/kdigo_uo/kdigo_aki 创建代码
    # 这些概念与 aki_stage_creat/aki_stage_uo/aki_stage 完全重复，只保留后者

    # 3. motor_response: GCS运动反应分项（从gcs中提取）
    motor_response_records = []
    if 'gcs' in data and not data['gcs'].empty:
        for pid in patient_ids:
            pid_gcs = data['gcs'][data['gcs']['stay_id'] == pid]
            for _, row in pid_gcs.iterrows():
                # motor response 通常是 GCS 中的一部分 (1-6分)
                # 这里简化为 GCS/3 取整（模拟）
                motor_score = max(1, min(6, int(row['gcs'] / 3)))
                motor_response_records.append({
                    'stay_id': pid,
                    'time': row['time'],
                    'motor_response': motor_score
                })
    data['motor_response'] = pd.DataFrame(motor_response_records) if motor_response_records else pd.DataFrame(columns=['stay_id', 'time', 'motor_response'])

    # 4. delirium_positive: 谵妄阳性（基于RASS和GCS评估）
    delirium_positive_records = []
    if 'rass' in data and not data['rass'].empty:
        for pid in patient_ids:
            pid_rass = data['rass'][data['rass']['stay_id'] == pid]
            for _, row in pid_rass.iterrows():
                # 谵妄通常出现在 RASS > 0 且 < 4，或波动性意识状态
                # 这里简化为 RASS 在 1-3 时约30%几率阳性
                is_delirium = 1 if (1 <= row['rass'] <= 3 and np.random.random() < 0.3) else 0
                delirium_positive_records.append({
                    'stay_id': pid,
                    'time': row['time'],
                    'delirium_positive': is_delirium
                })
    data['delirium_positive'] = pd.DataFrame(delirium_positive_records) if delirium_positive_records else pd.DataFrame(columns=['stay_id', 'time', 'delirium_positive'])

    # 5. delirium_tx: 谵妄治疗（通常使用抗精神病药物）
    delirium_tx_records = []
    if 'delirium_positive' in data and not data['delirium_positive'].empty:
        # 假设约50%的谵妄阳性患者会接受治疗
        delirium_pts = data['delirium_positive'][data['delirium_positive']['delirium_positive'] == 1]['stay_id'].unique()
        for pid in delirium_pts:
            if np.random.random() < 0.5:  # 50%接受治疗
                treatment_start = np.random.uniform(12, 60)
                delirium_tx_records.append({
                    'stay_id': pid,
                    'time': treatment_start,
                    'delirium_tx': 1
                })
    data['delirium_tx'] = pd.DataFrame(delirium_tx_records) if delirium_tx_records else pd.DataFrame(columns=['stay_id', 'time', 'delirium_tx'])

    # 6. adv_resp: 高级呼吸支持（机械通气 + PEEP > 5）
    adv_resp_records = []
    if 'vent_ind' in data and not data['vent_ind'].empty and 'peep' in data and not data['peep'].empty:
        # 合并 vent_ind 和 peep
        vent_peep = pd.merge(
            data['vent_ind'],
            data['peep'],
            on=['stay_id', 'time'],
            how='inner'
        )
        for _, row in vent_peep.iterrows():
            # 高级呼吸支持 = 机械通气 + PEEP > 5
            is_adv = 1 if (row['vent_ind'] == 1 and row['peep'] > 5) else 0
            adv_resp_records.append({
                'stay_id': row['stay_id'],
                'time': row['time'],
                'adv_resp': is_adv
            })
    data['adv_resp'] = pd.DataFrame(adv_resp_records) if adv_resp_records else pd.DataFrame(columns=['stay_id', 'time', 'adv_resp'])

    # 7. other_vaso: 其他血管活性药物（不包括常见的norepi/epi/dopa/dobu）
    # 示例：血管加压素(vasopressin)、去甲肾上腺素(phenylephrine)等
    other_vaso_records = []
    if 'phn_rate' in data and not data['phn_rate'].empty:
        data['other_vaso'] = data['phn_rate'].copy()
        data['other_vaso'] = data['other_vaso'].rename(columns={'phn_rate': 'other_vaso'})
        data['other_vaso']['other_vaso'] = (data['other_vaso']['other_vaso'] > 0).astype(int)
    else:
        # 生成少量记录（约10%患者）
        for pid in patient_ids:
            if np.random.random() < 0.1:
                start_time = np.random.uniform(6, 48)
                for t in range(int(start_time), min(72, int(start_time + 24)), 4):
                    other_vaso_records.append({
                        'stay_id': pid,
                        'time': float(t),
                        'other_vaso': 1
                    })
        data['other_vaso'] = pd.DataFrame(other_vaso_records) if other_vaso_records else pd.DataFrame(columns=['stay_id', 'time', 'other_vaso'])

    # 🔧 已删除 sep3 别名概念（2026-02-13）：不再提取 sep3 特征
    # 保留 sep3_sofa1 和 sep3_sofa2 作为独立的 Sepsis-3 诊断概念

    _add_catalog_mock_fallbacks(data, patient_ids, time_points, get_random_sample_times)

    return data, patient_ids
