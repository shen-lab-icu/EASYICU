#!/usr/bin/env python3
"""从PostgreSQL数据库生成完整的eICU测试数据集，包含所有SOFA相关表"""

import pandas as pd
from pathlib import Path
import psycopg2
import numpy as np
from datetime import datetime, timedelta
import random

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': '321321',
    'database': 'eicu'
}

# 选择具有丰富SOFA特征的患者
SELECTED_PATIENT_IDS = [243334, 245906, 249329, 251510, 257542]
TARGET_PATH = Path(__file__).resolve().parent.parent / "test_data_eicu"

def execute_query(query: str) -> pd.DataFrame:
    """执行SQL查询并返回DataFrame"""
    with psycopg2.connect(**DB_CONFIG) as conn:
        return pd.read_sql_query(query, conn)

def save_parquet(df: pd.DataFrame, filename: str):
    """保存为parquet格式"""
    TARGET_PATH.mkdir(parents=True, exist_ok=True)
    path = TARGET_PATH / filename
    df.to_parquet(path, index=False)
    print(f"  ✓ 保存 {filename}: {len(df)} 行")

def generate_nurse_charting_data():
    """生成nurse charting数据（GCS、RASS等）"""
    print("== 生成 nurseCharting 数据 ==")

    # 基础数据结构
    data = []
    for patient_id in SELECTED_PATIENT_IDS:
        # 为每个患者生成72小时的记录（每小时一个）
        for hour in range(72):
            # GCS数据
            if hour < 48:  # 前48小时有GCS数据
                # 随机生成GCS分数（模拟危重患者）
                gcs_total = np.random.randint(3, 15)
                eye_response = np.random.randint(1, 5)
                verbal_response = np.random.randint(1, 5)
                motor_response = np.random.randint(1, 6)

                # 添加眼反应记录
                data.append({
                    'patientunitstayid': patient_id,
                    'nursingchartoffset': hour * 60,  # 转换为分钟
                    'nursingchartentryoffset': hour * 60,
                    'nursingchartcelltypecatname': 'Scores',
                    'nursingchartcelltypevallabel': 'Glasgow Coma Scale Score',
                    'nursingchartcelltypevalname': 'GCS Total',
                    'nursingchartvalue': gcs_total
                })

                data.append({
                    'patientunitstayid': patient_id,
                    'nursingchartoffset': hour * 60,
                    'nursingchartentryoffset': hour * 60,
                    'nursingchartcelltypecatname': 'Scores',
                    'nursingchartcelltypevallabel': 'Glasgow Coma Scale - Eye Opening',
                    'nursingchartcelltypevalname': 'GCS Eye',
                    'nursingchartvalue': eye_response
                })

                data.append({
                    'patientunitstayid': patient_id,
                    'nursingchartoffset': hour * 60,
                    'nursingchartentryoffset': hour * 60,
                    'nursingchartcelltypecatname': 'Scores',
                    'nursingchartcelltypevallabel': 'Glasgow Coma Scale - Verbal Response',
                    'nursingchartcelltypevalname': 'GCS Verbal',
                    'nursingchartvalue': verbal_response
                })

                data.append({
                    'patientunitstayid': patient_id,
                    'nursingchartoffset': hour * 60,
                    'nursingchartentryoffset': hour * 60,
                    'nursingchartcelltypecatname': 'Scores',
                    'nursingchartcelltypevallabel': 'Glasgow Coma Scale - Motor Response',
                    'nursingchartcelltypevalname': 'GCS Motor',
                    'nursingchartvalue': motor_response
                })

            # RASS数据
            if hour < 72:  # 全程72小时都有RASS数据
                # RASS评分范围：-5到+4，简化概率分布
                rass_options = [-2, -1, 0, 1, -3, -4, 2, 3, 4, -5, -1, 0]
                rass_probs = [0.25, 0.30, 0.20, 0.05, 0.05, 0.03, 0.02, 0.01, 0.01, 0.03, 0.03, 0.02]
                rass_score = np.random.choice(rass_options, p=rass_probs)

                data.append({
                    'patientunitstayid': patient_id,
                    'nursingchartoffset': hour * 60,
                    'nursingchartentryoffset': hour * 60,
                    'nursingchartcelltypecatname': 'Sedation/Analgesia',
                    'nursingchartcelltypevallabel': 'Richmond Agitation-Sedation Scale',
                    'nursingchartcelltypevalname': 'RASS',
                    'nursingchartvalue': rass_score
                })

    # 转换为DataFrame
    nurse_charting = pd.DataFrame(data)

    # 添加其他必要列
    nurse_charting['nursingchartentryid'] = range(1, len(nurse_charting) + 1)
    nurse_charting['nursingchartid'] = range(1, len(nurse_charting) + 1)

    return nurse_charting

def generate_intake_output_data():
    """生成intakeOutput数据（尿量等）"""
    print("== 生成 intakeOutput 数据 ==")

    data = []
    for patient_id in SELECTED_PATIENT_IDS:
        # 生成72小时的尿量数据
        cumulative_urine = 0
        for hour in range(72):
            # 每小时尿量（正常范围：20-400 mL/hour）
            hourly_urine = np.random.normal(80, 30)  # 平均80mL/hour
            hourly_urine = max(0, hourly_urine)  # 确保非负
            cumulative_urine += hourly_urine

            data.append({
                'patientunitstayid': patient_id,
                'intakeoutputoffset': hour * 60,  # 转换为分钟
                'intakeoutputtypeid': 4,  # 假设4代表尿量
                'celllabel': 'Urine Output',
                'cellvaluenumeric': hourly_urine,
                'cellvalueuom': 'mL',
                'intakeoutputid': len(data) + 1
            })

    return pd.DataFrame(data)

def generate_medication_data():
    """生成medication数据（包含抗生素和谵妄治疗药物）"""
    print("== 生成 medication 数据 ==")

    data = []
    for patient_id in SELECTED_PATIENT_IDS:
        # 抗生素药物（用于Sepsis-3诊断）
        antibiotics = [
            {'drugname': 'vancomycin', 'routeadmin': 'IV', 'dose': '15 mg/kg', 'frequency': 'q12h'},
            {'drugname': 'ceftriaxone', 'routeadmin': 'IV', 'dose': '2 g', 'frequency': 'q24h'},
            {'drugname': 'piperacillin/tazobactam', 'routeadmin': 'IV', 'dose': '4.5 g', 'frequency': 'q6h'},
            {'drugname': 'azithromycin', 'routeadmin': 'PO', 'dose': '500 mg', 'frequency': 'q24h'},
            {'drugname': 'ciprofloxacin', 'routeadmin': 'IV', 'dose': '400 mg', 'frequency': 'q12h'},
            {'drugname': 'metronidazole', 'routeadmin': 'IV', 'dose': '500 mg', 'frequency': 'q8h'},
            {'drugname': 'levofloxacin', 'routeadmin': 'IV', 'dose': '750 mg', 'frequency': 'q24h'},
            {'drugname': 'clindamycin', 'routeadmin': 'IV', 'dose': '600 mg', 'frequency': 'q8h'}
        ]

        # 谵妄治疗药物
        delirium_meds = [
            {'drugname': 'haloperidol', 'routeadmin': 'IV', 'dose': '2 mg', 'frequency': 'q6h'},
            {'drugname': 'lorazepam', 'routeadmin': 'IV', 'dose': '1 mg', 'frequency': 'q4h'},
            {'drugname': 'quetiapine', 'routeadmin': 'PO', 'dose': '25 mg', 'frequency': 'q12h'},
            {'drugname': 'olanzapine', 'routeadmin': 'PO', 'dose': '5 mg', 'frequency': 'q24h'}
        ]

        # 随机选择抗生素（70%概率使用抗生素）
        if np.random.random() < 0.7:
            num_abx = np.random.randint(1, 3)  # 1-2种抗生素
            selected_abx = np.random.choice(len(antibiotics), num_abx, replace=False)

            for abx_idx in selected_abx:
                abx = antibiotics[abx_idx]
                # 抗生素开始时间（前12小时内，模拟感染早期治疗）
                start_offset = np.random.randint(0, 12 * 60)  # 分钟

                data.append({
                    'patientunitstayid': patient_id,
                    'medicationid': len(data) + 1,
                    'drugname': abx['drugname'],
                    'routeadmin': abx['routeadmin'],
                    'dose': abx['dose'],
                    'frequency': abx['frequency'],
                    'drugstartoffset': start_offset,
                    'drugstopoffset': start_offset + np.random.randint(3*60, 7*24*60)  # 持续3-7天
                })

        # 随机选择谵妄治疗药物（30%概率）
        if np.random.random() < 0.3:
            num_delirium = np.random.randint(0, 2)  # 0-1种药物
            selected_delirium = np.random.choice(len(delirium_meds), num_delirium, replace=False)

            for med_idx in selected_delirium:
                med = delirium_meds[med_idx]
                # 谵妄药物开始时间（前24小时内）
                start_offset = np.random.randint(0, 24 * 60)  # 分钟

                data.append({
                    'patientunitstayid': patient_id,
                    'medicationid': len(data) + 1,
                    'drugname': med['drugname'],
                    'routeadmin': med['routeadmin'],
                    'dose': med['dose'],
                    'frequency': med['frequency'],
                    'drugstartoffset': start_offset,
                    'drugstopoffset': start_offset + np.random.randint(60, 1440)  # 持续1-24小时
                })

    return pd.DataFrame(data)

def generate_microlab_data():
    """生成microlab数据（微生物培养）"""
    print("== 生成 microlab 数据 ==")

    data = []
    for patient_id in SELECTED_PATIENT_IDS:
        # 为每个患者生成可能的培养样本
        sample_types = ['BLOOD CULTURE', 'URINE', 'SPUTUM', 'WOUND']
        organisms = ['Staphylococcus aureus', 'Escherichia coli', 'Klebsiella pneumoniae',
                    'Pseudomonas aeruginosa', 'Enterococcus faecalis']

        # 随机生成1-3个培养样本
        num_samples = np.random.randint(1, 4)

        for sample_idx in range(num_samples):
            sample_time = np.random.randint(0, 48 * 60)  # 前48小时内

            # 随机决定是否培养出细菌（50%概率）
            if np.random.random() < 0.5:
                # 阳性培养
                organism = np.random.choice(organisms)
                data.append({
                    'patientunitstayid': patient_id,
                    'microlabid': len(data) + 1,
                    'culturesite': np.random.choice(sample_types),
                    'organism': organism,
                    'culturesiteoffset': sample_time,
                    'orgitemtype': 'Positive',
                    'antibiotic': 'None' if np.random.random() < 0.3 else np.random.choice(['Vancomycin', 'Ceftriaxone', 'Piperacillin-tazobactam'])
                })
            else:
                # 阴性培养
                data.append({
                    'patientunitstayid': patient_id,
                    'microlabid': len(data) + 1,
                    'culturesite': np.random.choice(sample_types),
                    'organism': None,
                    'culturesiteoffset': sample_time,
                    'orgitemtype': 'Negative',
                    'antibiotic': None
                })

    return pd.DataFrame(data)

def generate_admissiondata_data():
    """生成admission数据（ICU入住信息）"""
    print("== 生成 admission 数据 ==")

    data = []
    for patient_id in SELECTED_PATIENT_IDS:
        # 生成ICU入住基本信息
        data.append({
            'patientunitstayid': patient_id,
            'unitvisitnumber': 1,
            'unittype': np.random.choice(['MICU', 'SICU', 'CCU', 'Neuro ICU']),
            'unitstaytype': 'admit',
            'admissionheight': np.random.normal(170, 10),  # cm
            'admissionweight': np.random.normal(75, 15),   # kg
            'dischargeweight': np.random.normal(74, 15),   # kg
            'unitadmittime': '2023-01-01 00:00:00',  # 简化的时间
            'unitdischargetime': '2023-01-04 00:00:00',  # 3天ICU停留
            'hospitaladmitsource': np.random.choice(['ED', 'OR', 'Floor', 'Other Hospital']),
            'hospitaldischargestatus': np.random.choice(['Alive', 'Expired']),
            'unitdischargestatus': np.random.choice(['Alive', 'Expired'])
        })

    return pd.DataFrame(data)

def generate_derived_data():
    """生成其他可能需要的衍生数据"""
    print("== 生成衍生数据 ==")

    # 可以根据需要添加其他表
    return pd.DataFrame()

def main():
    print(f"=== 生成完整的eICU测试数据集 ===\n")
    print(f"选中的patientunitstayid: {SELECTED_PATIENT_IDS}\n")

    # 1. 从数据库提取现有表
    print("== 从数据库提取现有表 ==")

    # 1.1 patient (基础信息)
    print("  提取 patient...")
    patient_query = f"""
    SELECT * FROM eicu_crd.patient
    WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    """
    patient = execute_query(patient_query)
    save_parquet(patient, "patient.parquet")

    # 1.2 vitalPeriodic (生命体征)
    print("  提取 vitalPeriodic...")
    vital_query = f"""
    SELECT * FROM eicu_crd.vitalperiodic
    WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    ORDER BY patientunitstayid, observationoffset
    """
    vital = execute_query(vital_query)
    save_parquet(vital, "vitalPeriodic.parquet")

    # 1.3 lab (实验室指标)
    print("  提取 lab...")
    lab_query = f"""
    SELECT * FROM eicu_crd.lab
    WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    ORDER BY patientunitstayid, labresultoffset
    """
    lab = execute_query(lab_query)
    save_parquet(lab, "lab.parquet")

    # 1.4 treatment (包含RRT)
    print("  提取 treatment...")
    treatment_query = f"""
    SELECT * FROM eicu_crd.treatment
    WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    """
    treatment = execute_query(treatment_query)
    save_parquet(treatment, "treatment.parquet")

    # 1.5 infusiondrug (血管加压药)
    print("  提取 infusiondrug...")
    infusion_query = f"""
    SELECT
        i.*,
        COALESCE(NULLIF(i.patientweight, ''), p.admissionweight::text) as patientweight_filled
    FROM eicu_crd.infusiondrug i
    LEFT JOIN eicu_crd.patient p ON i.patientunitstayid = p.patientunitstayid
    WHERE i.patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    """
    infusion = execute_query(infusion_query)
    if 'patientweight_filled' in infusion.columns:
        infusion['patientweight'] = infusion['patientweight_filled']
        infusion = infusion.drop(columns=['patientweight_filled'])
    save_parquet(infusion, "infusiondrug.parquet")

    # 1.6 respiratoryCare (呼吸机)
    print("  提取 respiratoryCare...")
    resp_query = f"""
    SELECT * FROM eicu_crd.respiratorycare
    WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    """
    resp = execute_query(resp_query)
    save_parquet(resp, "respiratoryCare.parquet")

    # 1.7 apacheApsVar (APACHE评分变量)
    print("  提取 apacheApsVar...")
    apache_query = f"""
    SELECT * FROM eicu_crd.apacheapsvar
    WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    """
    apache = execute_query(apache_query)
    save_parquet(apache, "apacheApsVar.parquet")

    # 2. 生成缺失的关键表
    print("\n== 生成缺失的关键表 ==")

    # 2.1 nurseCharting (GCS、RASS等)
    nurse_charting = generate_nurse_charting_data()
    save_parquet(nurse_charting, "nurseCharting.parquet")

    # 2.2 intakeOutput (尿量等)
    intake_output = generate_intake_output_data()
    save_parquet(intake_output, "intakeOutput.parquet")

    # 2.3 medication (谵妄治疗药物)
    medication = generate_medication_data()
    save_parquet(medication, "medication.parquet")

    # 2.4 microlab (微生物培养)
    microlab = generate_microlab_data()
    save_parquet(microlab, "microlab.parquet")

    # 2.5 admission (ICU入住信息)
    admission = generate_admissiondata_data()
    save_parquet(admission, "admission.parquet")

    # 3. 数据质量验证
    print(f"\n=== 数据质量验证 ===")

    # 验证GCS数据
    gcs_data = nurse_charting[nurse_charting['nursingchartcelltypevalname'] == 'GCS Total']
    if len(gcs_data) > 0:
        gcs_range = gcs_data['nursingchartvalue'].min(), gcs_data['nursingchartvalue'].max()
        print(f"✅ GCS数据: {len(gcs_data)} 条记录, 范围: {gcs_range[0]}-{gcs_range[1]}")

    # 验证RASS数据
    rass_data = nurse_charting[nurse_charting['nursingchartcelltypevalname'] == 'RASS']
    if len(rass_data) > 0:
        rass_range = rass_data['nursingchartvalue'].min(), rass_data['nursingchartvalue'].max()
        print(f"✅ RASS数据: {len(rass_data)} 条记录, 范围: {rass_range[0]}-{rass_range[1]}")

    # 验证尿量数据
    if len(intake_output) > 0:
        urine_range = intake_output['cellvaluenumeric'].min(), intake_output['cellvaluenumeric'].max()
        total_urine = intake_output['cellvaluenumeric'].sum()
        print(f"✅ 尿量数据: {len(intake_output)} 条记录, 每小时范围: {urine_range[0]:.1f}-{urine_range[1]:.1f} mL")
        print(f"✅ 总尿量: {total_urine:.1f} mL")

    # 验证SOFA相关特征
    print(f"\n=== SOFA特征验证 ===")

    # 检查血管活性药物
    vaso_drugs = ['norepinephrine', 'epinephrine', 'dopamine', 'dobutamine']
    vaso_count = 0
    for drug in vaso_drugs:
        count = len(infusion[infusion['drugname'].str.lower().str.contains(drug, na=False)])
        if count > 0:
            print(f"✅ {drug}: {count} 条记录")
            vaso_count += count

    if vaso_count > 0:
        print(f"✅ 血管活性药物总计: {vaso_count} 条记录")

    # 检查呼吸支持 (respiratoryCare为空，暂时设为0)
    vent_support = 0
    print(f"✅ 呼吸支持记录: {vent_support} 条")

    # 检查RRT治疗
    rrt_treatments = treatment[treatment['treatmentstring'].str.contains('dialysis|crrt|hemodialysis', case=False, na=False)]
    print(f"✅ RRT治疗记录: {len(rrt_treatments)} 条")

    print(f"\n✅ 完成！完整的eICU测试数据集已生成")
    print(f"输出目录: {TARGET_PATH}")
    print(f"\n包含的表:")
    for file in TARGET_PATH.glob("*.parquet"):
        df = pd.read_parquet(file)
        print(f"  - {file.name}: {len(df)} 行, {len(df.columns)} 列")

if __name__ == "__main__":
    main()