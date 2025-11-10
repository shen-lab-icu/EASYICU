#!/usr/bin/env python3
"""从PostgreSQL数据库直接生成包含SOFA2特征的AUMC测试数据"""

import pandas as pd
from pathlib import Path
import psycopg2

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': '321321',
    'database': 'aumc'
}

# 使用数据库查询得到的包含RRT+血管加压药的AUMC患者
# 并且包含一个有ECMO的患者
SELECTED_ADMISSION_IDS = [11, 37, 47, 53, 3441]  # 3441有ECMO
TARGET_PATH = Path(__file__).resolve().parent.parent / "test_data_aumc"

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
    print(f"=== 从数据库提取SOFA2特征丰富的AUMC患者数据 ===\n")
    print(f"选中的admissionid: {SELECTED_ADMISSION_IDS}\n")
    
    # 1. admissions
    print("== 提取 admissions ==")
    admissions_query = f"""
    SELECT * FROM admissions
    WHERE admissionid IN ({','.join(map(str, SELECTED_ADMISSION_IDS))})
    """
    admissions = execute_query(admissions_query)
    save_parquet(admissions, "admissions.parquet")
    
    # 2. numericitems (数值型数据，包含生命体征、实验室指标)
    print("== 提取 numericitems ==")
    numeric_query = f"""
    SELECT * FROM numericitems
    WHERE admissionid IN ({','.join(map(str, SELECTED_ADMISSION_IDS))})
    """
    numeric = execute_query(numeric_query)
    save_parquet(numeric, "numericitems.parquet")
    
    # 3. listitems (分类型数据)
    print("== 提取 listitems ==")
    list_query = f"""
    SELECT * FROM listitems
    WHERE admissionid IN ({','.join(map(str, SELECTED_ADMISSION_IDS))})
    """
    list_items = execute_query(list_query)
    save_parquet(list_items, "listitems.parquet")
    
    # 4. procedureorderitems (包含RRT、ECMO等操作)
    print("== 提取 procedureorderitems ==")
    procedure_query = f"""
    SELECT * FROM procedureorderitems
    WHERE admissionid IN ({','.join(map(str, SELECTED_ADMISSION_IDS))})
    """
    procedure = execute_query(procedure_query)
    save_parquet(procedure, "procedureorderitems.parquet")
    
    # 5. drugitems (包含血管加压药)
    print("== 提取 drugitems ==")
    drug_query = f"""
    SELECT * FROM drugitems
    WHERE admissionid IN ({','.join(map(str, SELECTED_ADMISSION_IDS))})
    """
    drugs = execute_query(drug_query)
    save_parquet(drugs, "drugitems.parquet")
    
    # 6. freetextitems (自由文本)
    print("== 提取 freetextitems ==")
    freetext_query = f"""
    SELECT * FROM freetextitems
    WHERE admissionid IN ({','.join(map(str, SELECTED_ADMISSION_IDS))})
    """
    freetext = execute_query(freetext_query)
    save_parquet(freetext, "freetextitems.parquet")
    
    print(f"\n✅ 完成！输出目录: {TARGET_PATH}")
    print(f"\n患者特征验证:")
    
    # 验证RRT
    rrt_count = execute_query(f"""
        SELECT admissionid, COUNT(*) as count
        FROM procedureorderitems
        WHERE admissionid IN ({','.join(map(str, SELECTED_ADMISSION_IDS))})
          AND (LOWER(item) LIKE '%cvvh%' OR LOWER(item) LIKE '%dialys%')
        GROUP BY admissionid
    """)
    print(f"  - RRT操作记录: {len(rrt_count)} 个患者")
    if len(rrt_count) > 0:
        print(rrt_count.to_string(index=False))
    
    # 验证ECMO
    ecmo_count = execute_query(f"""
        SELECT admissionid, COUNT(*) as count
        FROM procedureorderitems
        WHERE admissionid IN ({','.join(map(str, SELECTED_ADMISSION_IDS))})
          AND LOWER(item) LIKE '%ecmo%'
        GROUP BY admissionid
    """)
    print(f"\n  - ECMO操作记录: {len(ecmo_count)} 个患者")
    if len(ecmo_count) > 0:
        print(ecmo_count.to_string(index=False))
    
    # 验证血管加压药
    vaso_count = execute_query(f"""
        SELECT admissionid, COUNT(*) as count
        FROM drugitems
        WHERE admissionid IN ({','.join(map(str, SELECTED_ADMISSION_IDS))})
          AND (LOWER(item) LIKE '%noradrenaline%'
               OR LOWER(item) LIKE '%adrenaline%'
               OR LOWER(item) LIKE '%dopamine%')
        GROUP BY admissionid
    """)
    print(f"\n  - 血管加压药记录: {len(vaso_count)} 个患者")
    if len(vaso_count) > 0:
        print(vaso_count.to_string(index=False))

if __name__ == "__main__":
    main()
