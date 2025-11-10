#!/usr/bin/env python3
"""从PostgreSQL数据库直接生成包含SOFA2特征的MIMIC-IV测试数据"""

import pandas as pd
from pathlib import Path
import psycopg2

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': '321321',
    'database': 'mimiciv'
}

# 使用数据库查询得到的包含RRT+血管加压药+谵妄评估的患者
SELECTED_STAY_IDS = [30005000, 30009597, 30017005, 30041848, 30045407]
TARGET_PATH = Path(__file__).resolve().parent.parent / "test_data_miiv"

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
    print(f"=== 从数据库提取SOFA2特征丰富的患者数据 ===\n")
    print(f"选中的stay_ids: {SELECTED_STAY_IDS}\n")
    
    # 1. icustays
    print("== 提取 icustays ==")
    icustays_query = f"""
    SELECT * FROM mimiciv_icu.icustays
    WHERE stay_id IN ({','.join(map(str, SELECTED_STAY_IDS))})
    """
    icustays = execute_query(icustays_query)
    subject_ids = icustays['subject_id'].tolist()
    save_parquet(icustays, "icustays.parquet")
    
    # 2. patients
    print("== 提取 patients ==")
    patients_query = f"""
    SELECT * FROM mimiciv_hosp.patients
    WHERE subject_id IN ({','.join(map(str, subject_ids))})
    """
    patients = execute_query(patients_query)
    save_parquet(patients, "patients.parquet")
    
    # 3. chartevents (重要: 包含谵妄评估、RASS、GCS、生命体征等)
    print("== 提取 chartevents ==")
    chartevents_query = f"""
    SELECT * FROM mimiciv_icu.chartevents
    WHERE stay_id IN ({','.join(map(str, SELECTED_STAY_IDS))})
    """
    chartevents = execute_query(chartevents_query)
    save_parquet(chartevents, "chartevents.parquet")
    
    # 4. labevents
    print("== 提取 labevents ==")
    labevents_query = f"""
    SELECT * FROM mimiciv_hosp.labevents
    WHERE subject_id IN ({','.join(map(str, subject_ids))})
    """
    labevents = execute_query(labevents_query)
    save_parquet(labevents, "labevents.parquet")
    
    # 5. inputevents (包含血管加压药)
    print("== 提取 inputevents ==")
    inputevents_query = f"""
    SELECT * FROM mimiciv_icu.inputevents
    WHERE stay_id IN ({','.join(map(str, SELECTED_STAY_IDS))})
    """
    inputevents = execute_query(inputevents_query)
    save_parquet(inputevents, "inputevents.parquet")
    
    # 6. outputevents (包含尿量)
    print("== 提取 outputevents ==")
    outputevents_query = f"""
    SELECT * FROM mimiciv_icu.outputevents
    WHERE stay_id IN ({','.join(map(str, SELECTED_STAY_IDS))})
    """
    outputevents = execute_query(outputevents_query)
    save_parquet(outputevents, "outputevents.parquet")
    
    # 7. procedureevents (包含RRT)
    print("== 提取 procedureevents ==")
    procedureevents_query = f"""
    SELECT * FROM mimiciv_icu.procedureevents
    WHERE stay_id IN ({','.join(map(str, SELECTED_STAY_IDS))})
    """
    procedureevents = execute_query(procedureevents_query)
    save_parquet(procedureevents, "procedureevents.parquet")
    
    # 8. 字典表
    print("== 提取字典表 ==")
    d_items = execute_query("SELECT * FROM mimiciv_icu.d_items")
    save_parquet(d_items, "d_items.parquet")
    
    d_labitems = execute_query("SELECT * FROM mimiciv_hosp.d_labitems")
    save_parquet(d_labitems, "d_labitems.parquet")
    
    print(f"\n✅ 完成！输出目录: {TARGET_PATH}")
    print(f"\n患者特征验证:")
    
    # 验证关键SOFA2特征
    rrt_count = execute_query(f"""
        SELECT stay_id, COUNT(*) as count
        FROM mimiciv_icu.procedureevents
        WHERE stay_id IN ({','.join(map(str, SELECTED_STAY_IDS))})
          AND itemid IN (225441, 225802, 225803, 225809, 225955)
        GROUP BY stay_id
    """)
    print(f"  - RRT记录: {len(rrt_count)} 个患者")
    print(rrt_count.to_string(index=False))
    
    vaso_count = execute_query(f"""
        SELECT stay_id, COUNT(*) as count
        FROM mimiciv_icu.inputevents
        WHERE stay_id IN ({','.join(map(str, SELECTED_STAY_IDS))})
          AND itemid IN (221906, 221749, 221662, 221289)
        GROUP BY stay_id
    """)
    print(f"\n  - 血管加压药记录: {len(vaso_count)} 个患者")
    print(vaso_count.to_string(index=False))
    
    delirium_count = execute_query(f"""
        SELECT stay_id, COUNT(*) as count
        FROM mimiciv_icu.chartevents
        WHERE stay_id IN ({','.join(map(str, SELECTED_STAY_IDS))})
          AND itemid = 228334
        GROUP BY stay_id
    """)
    print(f"\n  - 谵妄评估记录: {len(delirium_count)} 个患者")
    print(delirium_count.to_string(index=False))

if __name__ == "__main__":
    main()
