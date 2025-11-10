#!/usr/bin/env python3
"""从PostgreSQL数据库直接生成包含SOFA2特征的eICU测试数据"""

import pandas as pd
from pathlib import Path
import psycopg2

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': '321321',
    'database': 'eicu'
}

# 使用数据库查询得到的包含RRT+血管加压药的eICU患者
SELECTED_PATIENT_IDS = [999007, 999008, 999009, 999010, 999011]
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

def main():
    print(f"=== 从数据库提取SOFA2特征丰富的eICU患者数据 ===\n")
    print(f"选中的patientunitstayid: {SELECTED_PATIENT_IDS}\n")
    
    # 1. patient (基础信息)
    print("== 提取 patient ==")
    patient_query = f"""
    SELECT * FROM eicu_crd.patient
    WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    """
    patient = execute_query(patient_query)
    save_parquet(patient, "patient.parquet")
    
    # 2. vitalPeriodic (生命体征)
    print("== 提取 vitalPeriodic ==")
    vital_query = f"""
    SELECT * FROM eicu_crd.vitalperiodic
    WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    """
    vital = execute_query(vital_query)
    save_parquet(vital, "vitalPeriodic.parquet")
    
    # 3. lab (实验室指标)
    print("== 提取 lab ==")
    lab_query = f"""
    SELECT * FROM eicu_crd.lab
    WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    """
    lab = execute_query(lab_query)
    save_parquet(lab, "lab.parquet")
    
    # 4. treatment (包含RRT)
    print("== 提取 treatment ==")
    treatment_query = f"""
    SELECT * FROM eicu_crd.treatment
    WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    """
    treatment = execute_query(treatment_query)
    save_parquet(treatment, "treatment.parquet")
    
    # 5. infusiondrug (血管加压药)
    print("== 提取 infusiondrug ==")
    # 🔧 FIX: infusiondrug.patientweight通常为空，从patient表获取admissionweight
    infusion_query = f"""
    SELECT 
        i.*,
        COALESCE(NULLIF(i.patientweight, ''), p.admissionweight::text) as patientweight_filled
    FROM eicu_crd.infusiondrug i
    LEFT JOIN eicu_crd.patient p ON i.patientunitstayid = p.patientunitstayid
    WHERE i.patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    """
    infusion = execute_query(infusion_query)
    # 用filled weight替换原patientweight列
    if 'patientweight_filled' in infusion.columns:
        infusion['patientweight'] = infusion['patientweight_filled']
        infusion = infusion.drop(columns=['patientweight_filled'])
    save_parquet(infusion, "infusiondrug.parquet")
    
    # 6. respiratoryCare (呼吸机)
    print("== 提取 respiratoryCare ==")
    resp_query = f"""
    SELECT * FROM eicu_crd.respiratorycare
    WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    """
    resp = execute_query(resp_query)
    save_parquet(resp, "respiratoryCare.parquet")
    
    # 7. apacheApsVar (APACHE评分变量)
    print("== 提取 apacheApsVar ==")
    apache_query = f"""
    SELECT * FROM eicu_crd.apacheapsvar
    WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
    """
    apache = execute_query(apache_query)
    save_parquet(apache, "apacheApsVar.parquet")
    
    print(f"\n✅ 完成！输出目录: {TARGET_PATH}")
    print(f"\n患者特征验证:")
    
    # 验证RRT
    rrt_count = execute_query(f"""
        SELECT patientunitstayid, COUNT(*) as count
        FROM eicu_crd.treatment
        WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
          AND (LOWER(treatmentstring) LIKE '%dialysis%' 
               OR LOWER(treatmentstring) LIKE '%crrt%'
               OR LOWER(treatmentstring) LIKE '%hemodialysis%')
        GROUP BY patientunitstayid
    """)
    print(f"  - RRT治疗记录: {len(rrt_count)} 个患者")
    if len(rrt_count) > 0:
        print(rrt_count.to_string(index=False))
    
    # 验证血管加压药
    vaso_count = execute_query(f"""
        SELECT patientunitstayid, COUNT(*) as count
        FROM eicu_crd.infusiondrug
        WHERE patientunitstayid IN ({','.join(map(str, SELECTED_PATIENT_IDS))})
          AND (LOWER(drugname) LIKE '%norepinephrine%'
               OR LOWER(drugname) LIKE '%epinephrine%'
               OR LOWER(drugname) LIKE '%dopamine%'
               OR LOWER(drugname) LIKE '%vasopressin%')
        GROUP BY patientunitstayid
    """)
    print(f"\n  - 血管加压药记录: {len(vaso_count)} 个患者")
    if len(vaso_count) > 0:
        print(vaso_count.to_string(index=False))

if __name__ == "__main__":
    main()
