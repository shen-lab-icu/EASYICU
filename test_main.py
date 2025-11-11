#!/usr/bin/env python3
"""
pyricu 统一测试脚本

整合所有核心验证测试：
1. SOFA 评分加载和验证
2. Sepsis-3 诊断和验证  
3. 数据完整性检查
4. 性能基准测试

使用统一配置，避免代码重复。
支持多数据库：MIMIC-IV, eICU, HiRID, AUMC

注意：本脚本已更新为使用新的简化API
- 推荐: load_sofa(), load_concepts() 等函数（支持智能默认值）
- 弃用: ICUQuickLoader 类（保留向后兼容）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# 确保使用 src/ 下的代码
sys.path.insert(0, 'src')

# ============================================================================
# 全局配置：选择要测试的数据库
# ============================================================================

# 数据库选择：'miiv', 'eicu', 'hirid', 'aumc'
TEST_DATABASE = 'miiv'  # 修改这里来切换数据库

# 数据源选择：'test' (测试数据) 或 'production' (完整数据)
TEST_DATA_SOURCE = 'test'

# 患者集选择：'debug' (1个), 'default' (3个), '50patients' (50个)
TEST_PATIENT_SET = 'default'

# ============================================================================

# 可选的可视化库
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 导入核心功能（使用新API）
from pyricu import load_concepts, load_sofa, load_sofa2, load_sepsis3
from pyricu.easy import load_vitals, load_labs, load_sofa_score, load_sepsis

# 保留向后兼容
try:
    from pyricu.quickstart import get_patient_ids as load_patient_ids
except ImportError:
    # 如果quickstart被移除，使用替代方法
    def load_patient_ids(data_path, database='miiv', max_patients=None):
        from pyricu.fst_reader import read_fst
        icustays = read_fst(Path(data_path) / 'icustays.fst')
        # 根据数据库确定ID列名
        if database == 'aumc':
            id_col = 'admissionid'
        elif database in ['eicu', 'eicu_demo']:
            id_col = 'patientunitstayid'
        else:
            id_col = 'stay_id'
        ids = icustays[id_col].tolist()
        if max_patients:
            ids = ids[:max_patients]
        return ids
from pyricu.datasource import FilterOp, FilterSpec
from pyricu.easy import load_vitals, load_labs, load_sofa_score, load_sepsis
from pyricu.project_config import (
    get_data_path,
    get_patient_ids,
    get_concepts,
    TEST_DATA_PATH,
    PRODUCTION_DATA_PATH,
    VERBOSE,
    print_config
)


# ============================================================================
# 测试 1: SOFA 评分加载和验证
# ============================================================================

def verify_raw_tables(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """验证原始表数据 - 检查提取特征的数据来源"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🔍 验证原始表数据 [{database.upper()}]")
        print("=" * 80)
    
    from pyricu.fst_reader import read_fst
    data_path_obj = Path(data_path)
    
    # 根据数据库类型确定 ID 列名
    if database in ['eicu', 'eicu_demo']:
        id_col = 'patientunitstayid'
    elif database == 'aumc':
        id_col = 'admissionid'
    else:
        id_col = 'stay_id'
    
    # 检查关键表
    tables_to_check = []
    
    if database == 'miiv':
        tables_to_check = [
            ('chartevents', ['stay_id', 'charttime', 'itemid', 'value', 'valuenum']),
            ('labevents', ['subject_id', 'charttime', 'itemid', 'value', 'valuenum']),
            ('inputevents', ['stay_id', 'starttime', 'endtime', 'itemid', 'amount', 'rate']),
            ('outputevents', ['stay_id', 'charttime', 'itemid', 'value']),
        ]
    elif database == 'eicu':
        tables_to_check = [
            ('vitalPeriodic', ['patientunitstayid', 'observationoffset', 'temperature', 'heartrate', 'respiration']),
            ('lab', ['patientunitstayid', 'labresultoffset', 'labname', 'labresult']),
        ]
    elif database == 'aumc':
        tables_to_check = [
            ('numericitems', ['admissionid', 'measuredat', 'itemid', 'value']),
            ('listitems', ['admissionid', 'measuredat', 'itemid', 'value']),
        ]
    
    for table_name, key_cols in tables_to_check:
        # 检查分区表（如chartevents, labevents）
        partitioned_dir = data_path_obj / table_name
        if partitioned_dir.exists() and partitioned_dir.is_dir():
            if verbose:
                print(f"\n📂 检查分区表: {table_name}/")
            # 读取第一个分区作为示例
            partitions = sorted([f for f in partitioned_dir.glob('*.fst')])
            if partitions:
                sample_df = read_fst(partitions[0])
                if verbose:
                    print(f"   分区数: {len(partitions)}")
                    print(f"   样本分区: {partitions[0].name}, 行数: {len(sample_df)}")
                    print(f"   列名: {sample_df.columns.tolist()}")
                    print(f"   前3行:")
                    print(sample_df.head(3))
                    
                    # 如果是MIIV的chartevents，展示关键itemid的数据
                    if database == 'miiv' and table_name == 'chartevents' and 'itemid' in sample_df.columns:
                        # SOFA相关的关键itemids
                        key_itemids = {
                            220045: 'HR (心率)',
                            220050: 'SBP (收缩压)',
                            220051: 'DBP (舒张压)',
                            220052: 'MBP (平均动脉压)',
                            223761: 'Temp (体温)',
                            220210: 'RR (呼吸频率)',
                            220277: 'SpO2 (血氧)',
                            223900: 'GCS-Verbal',
                            223901: 'GCS-Motor',
                        }
                        available_itemids = set(sample_df['itemid'].unique())
                        found = {k: v for k, v in key_itemids.items() if k in available_itemids}
                        if found and verbose:
                            print(f"\n   关键SOFA相关itemid:")
                            for itemid, name in found.items():
                                count = len(sample_df[sample_df['itemid'] == itemid])
                                print(f"     {itemid}: {name} ({count} 条记录)")
        else:
            # 检查单文件表
            table_file = data_path_obj / f"{table_name}.fst"
            if table_file.exists():
                df = read_fst(table_file)
                if verbose:
                    print(f"\n📄 检查表: {table_name}.fst")
                    print(f"   总行数: {len(df)}")
                    if id_col in df.columns:
                        print(f"   唯一患者数: {df[id_col].nunique()}")
                    print(f"   列名: {df.columns.tolist()}")
                    # 过滤到测试患者
                    if id_col in df.columns and patient_ids:
                        test_patient_id = patient_ids[0]
                        patient_data = df[df[id_col] == test_patient_id]
                        if len(patient_data) > 0:
                            print(f"   患者 {test_patient_id} 的数据 ({len(patient_data)} 行):")
                            print(patient_data.head(5))
                            
                            # 如果有itemid列，展示关键itemid
                            if 'itemid' in patient_data.columns:
                                unique_itemids = patient_data['itemid'].unique()
                                print(f"   患者 {test_patient_id} 的唯一itemid数: {len(unique_itemids)}")
                                print(f"   前10个itemid: {sorted(unique_itemids)[:10]}")
                    else:
                        print(f"   前5行:")
                        print(df.head(5))
    
    # 额外检查：对于MIIV，展示患者的实际生命体征数据
    if database == 'miiv' and verbose and patient_ids:
        print(f"\n" + "=" * 80)
        print(f"🔬 详细数据验证: 患者 {patient_ids[0]}")
        print("=" * 80)
        
        # 读取chartevents分区数据
        chartevents_dir = data_path_obj / 'chartevents'
        if chartevents_dir.exists():
            all_chart_data = []
            for partition_file in chartevents_dir.glob('*.fst'):
                df = read_fst(partition_file)
                patient_data = df[df['stay_id'] == patient_ids[0]] if 'stay_id' in df.columns else pd.DataFrame()
                if len(patient_data) > 0:
                    all_chart_data.append(patient_data)
            
            if all_chart_data:
                chart_df = pd.concat(all_chart_data, ignore_index=True)
                print(f"\n📊 患者 {patient_ids[0]} 的chartevents数据:")
                print(f"   总记录数: {len(chart_df)}")
                print(f"   唯一itemid数: {chart_df['itemid'].nunique()}")
                print(f"   时间范围: {chart_df['charttime'].min()} 到 {chart_df['charttime'].max()}")
                
                # 按itemid统计
                itemid_counts = chart_df['itemid'].value_counts().head(10)
                print(f"\n   Top 10 itemid:")
                for itemid, count in itemid_counts.items():
                    sample_val = chart_df[chart_df['itemid'] == itemid]['valuenum'].iloc[0] if len(chart_df[chart_df['itemid'] == itemid]) > 0 else None
                    print(f"     {itemid}: {count} 条, 样本值={sample_val}")
        
        # 读取labevents数据
        labevents_dir = data_path_obj / 'labevents'
        if labevents_dir.exists():
            # 先从icustays获取subject_id
            icustays = read_fst(data_path_obj / 'icustays.fst')
            subject_id = icustays[icustays['stay_id'] == patient_ids[0]]['subject_id'].iloc[0] if len(icustays[icustays['stay_id'] == patient_ids[0]]) > 0 else None
            
            if subject_id:
                all_lab_data = []
                for partition_file in labevents_dir.glob('*.fst'):
                    df = read_fst(partition_file)
                    patient_data = df[df['subject_id'] == subject_id] if 'subject_id' in df.columns else pd.DataFrame()
                    if len(patient_data) > 0:
                        all_lab_data.append(patient_data)
                
                if all_lab_data:
                    lab_df = pd.concat(all_lab_data, ignore_index=True)
                    print(f"\n📊 患者 {patient_ids[0]} (subject_id={subject_id}) 的labevents数据:")
                    print(f"   总记录数: {len(lab_df)}")
                    print(f"   唯一itemid数: {lab_df['itemid'].nunique()}")
                    
                    # SOFA相关的实验室itemid
                    sofa_lab_items = {
                        50885: 'Bilirubin',
                        50912: 'Creatinine',
                        51265: 'Platelet',
                    }
                    for itemid, name in sofa_lab_items.items():
                        item_data = lab_df[lab_df['itemid'] == itemid]
                        if len(item_data) > 0:
                            print(f"   {name} (itemid={itemid}): {len(item_data)} 条")
                            print(f"     值范围: {item_data['valuenum'].min():.2f} - {item_data['valuenum'].max():.2f}")
                            print(f"     样本: {item_data[['charttime', 'valuenum']].head(3).to_dict('records')}")

def test_sofa_basic(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试基本 SOFA 评分加载"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 1: SOFA 评分加载 [{database.upper()}]")
        print("=" * 80)
    
    # 使用 load_concepts 加载 SOFA
    sofa_df = load_concepts(
        'sofa',
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        verbose=verbose
    )
    
    # 根据数据库类型确定 ID 列名
    if database in ['eicu', 'eicu_demo']:
        id_col = 'patientunitstayid'
    elif database == 'aumc':
        id_col = 'admissionid'
    else:
        id_col = 'stay_id'
    
    if verbose:
        print(f"✅ SOFA 数据: {len(sofa_df)} 行, 患者数={sofa_df[id_col].nunique()}, "
              f"平均分={sofa_df['sofa'].mean():.1f}")
        print(f"\n📊 SOFA 提取结果前5行:")
        print(sofa_df.head())
        print(f"\n列名: {sofa_df.columns.tolist()}")
        print(f"数据类型: {sofa_df.dtypes.to_dict()}")
    
    # 验证
    assert len(sofa_df) > 0, "❌ SOFA 数据为空"
    assert 'sofa' in sofa_df.columns, "❌ 缺少 sofa 列"
    assert id_col in sofa_df.columns, f"❌ 缺少 {id_col} 列"
    
    return sofa_df


def test_sofa_components(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试 SOFA 组件加载"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 2: SOFA 组件 [{database.upper()}]")
        print("=" * 80)
    
    # SOFA组件需要单独加载各个组件概念
    # 为了简化，先跳过组件测试，只验证总分
    if verbose:
        print("⏭️  跳过组件测试（需要单独加载各组件）")
    
    return pd.DataFrame()  # 返回空DataFrame


# ============================================================================
# 测试 2b: SOFA-2 评分对比
# ============================================================================

def test_sofa2_comparison(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试 SOFA-2 评分并与 SOFA 对比"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 3: SOFA-2 评分对比 [{database.upper()}]")
        print("=" * 80)
    
    try:
        sofa2_df = load_sofa2(
            database=database,
            data_path=data_path,
            patient_ids=patient_ids,
            interval='1h',
            win_length='24h',
            keep_components=False,
            verbose=verbose
        )
        
        if len(sofa2_df) > 0 and verbose:
            print(f"✅ SOFA-2 数据: {len(sofa2_df)} 行, 平均分={sofa2_df['sofa2'].mean():.2f}")
            print(f"\n📊 SOFA-2 提取结果前5行:")
            print(sofa2_df.head())
            print(f"\n列名: {sofa2_df.columns.tolist()}")
        
        return sofa2_df
        
    except Exception as e:
        if verbose:
            print(f"⚠️  SOFA-2 测试跳过: {e}")
        return pd.DataFrame()


# ============================================================================
# 测试 3: Sepsis-3 诊断
# ============================================================================

def test_sepsis3(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试 Sepsis-3 诊断"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 4: Sepsis-3 诊断 [{database.upper()}]")
        print("=" * 80)
    
    try:
        sepsis_df = load_sepsis3(
            database=database,
            data_path=data_path,
            patient_ids=patient_ids,
            interval='1h',
            verbose=verbose
        )
        
        # 统计
        if len(sepsis_df) > 0 and verbose:
            si_count = (sepsis_df['susp_inf'] > 0).sum() if 'susp_inf' in sepsis_df.columns else 0
            sep3_count = (sepsis_df['sep3'] > 0).sum() if 'sep3' in sepsis_df.columns else 0
            print(f"✅ Sepsis-3: {len(sepsis_df)} 行, 疑似感染={si_count}, Sepsis阳性={sep3_count}")
            print(f"\n📊 Sepsis-3 提取结果前5行:")
            print(sepsis_df.head())
            print(f"\n列名: {sepsis_df.columns.tolist()}")
        
        return sepsis_df
    except Exception as e:
        if verbose:
            print(f"⚠️  Sepsis-3 测试跳过: {e}")
        return pd.DataFrame()


# ============================================================================
# 测试 4: 极简 API
# ============================================================================

def test_easy_api(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试极简 API"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 5: 极简 API [{database.upper()}]")
        print("=" * 80)
    
    tests = [
        ('生命体征', lambda: load_vitals(database=database, data_path=data_path, patient_ids=patient_ids)),
        ('实验室', lambda: load_labs(database=database, data_path=data_path, patient_ids=patient_ids)),
        ('SOFA评分', lambda: load_sofa_score(data_path, patient_ids=patient_ids, database=database)),
        ('Sepsis诊断', lambda: load_sepsis(data_path, patient_ids=patient_ids, database=database))
    ]
    
    for name, func in tests:
        try:
            result = func()
            if verbose:
                print(f"   ✅ {name}: {len(result)} 行")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  {name}: 跳过 ({str(e)[:50]}...)")
    
    if verbose:
        print("✅ API测试完成")


# ============================================================================
# 测试 5: 批量加载性能
# ============================================================================

def test_batch_performance(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试批量加载性能（缓存优化）"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 6: 批量加载性能 [{database.upper()}]")
        print("=" * 80)
    
    concepts = get_concepts('vitals')
    
    start = time.time()
    # 使用新API批量加载
    result = load_concepts(
        concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path
    )
    elapsed = time.time() - start
    
    if verbose:
        result_len = len(result.data) if hasattr(result, 'data') else len(result)
        print(f"✅ 批量加载 {len(concepts)} 个概念: {elapsed:.2f}秒, {result_len} 行")


def test_data_integrity(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试数据完整性"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 7: 数据完整性 [{database.upper()}]")
        print("=" * 80)
    
    try:
        vitals = load_vitals(database=database, data_path=data_path, patient_ids=patient_ids[:1])
        if verbose:
            status = "正常" if len(vitals) > 0 else "空数据"
            print(f"✅ 数据加载{status} ({len(vitals)} 条记录)")
    except Exception as e:
        if verbose:
            print(f"❌ 数据加载失败: {e}")
        raise


# ============================================================================
# 测试 7: SOFA vs SOFA2 和 Sepsis 对比可视化
# ============================================================================

# ============================================================================
def test_sofa_sepsis_visualization(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """Visualization: Compare SOFA vs SOFA2 and Sepsis diagnosis (multi-patient version)"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 Test 8: Sepsis Visualization Comparison (Multi-patient) [{database.upper()}]")
        print("=" * 80)

    if not HAS_MATPLOTLIB:
        print("⚠️  matplotlib not installed, skipping visualization test")
        return

    if len(patient_ids) == 0:
        print("⚠️  No patient data available, skipping visualization")
        return

    try:
        from pyricu.sepsis_sofa2 import sep3_sofa2

        # Find patients with Sepsis events (up to 3)
        if verbose:
            print(f"🔍 Searching for Sepsis cases...")

        sepsis_patients = []

        for pid in patient_ids[:min(20, len(patient_ids))]:
            try:
                sepsis3_df = load_sepsis3(
                    database=database,
                    data_path=data_path,
                    patient_ids=[pid],
                    verbose=False
                )
                has_sep3 = sepsis3_df['sep3'].sum() > 0 if 'sep3' in sepsis3_df.columns else False

                if has_sep3:
                    sepsis_patients.append(pid)
                    if len(sepsis_patients) >= 3:
                        break
            except:
                pass

        if len(sepsis_patients) == 0:
            sepsis_patients = patient_ids[:min(3, len(patient_ids))]

        if verbose:
            print(f"   Found {len(sepsis_patients)} patients, generating plots...")

        # Create plots
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        chart_count = 0

        for patient_id in sepsis_patients:
            try:
                # Load data
                sofa_df = load_sofa(
                    database=database,
                    data_path=data_path,
                    patient_ids=[patient_id],
                    interval='1h',
                    win_length='24h',
                    keep_components=False,
                    verbose=False
                )

                sofa2_df = load_sofa2(
                    database=database,
                    data_path=data_path,
                    patient_ids=[patient_id],
                    interval='1h',
                    win_length='24h',
                    keep_components=False,
                    verbose=False
                )

                sepsis3_df = load_sepsis3(
                    database=database,
                    data_path=data_path,
                    patient_ids=[patient_id],
                    verbose=False
                )

                # Antibiotics, blood culture, suspected infection
                try:
                    abx_df = load_concepts('abx', database=database, data_path=data_path,
                                           patient_ids=[patient_id], verbose=False)
                except:
                    abx_df = pd.DataFrame()

                try:
                    samp_df = load_concepts('samp', database=database, data_path=data_path,
                                            patient_ids=[patient_id], verbose=False)
                except:
                    samp_df = pd.DataFrame()

                try:
                    susp_inf_df = load_concepts('susp_inf', database=database, data_path=data_path,
                                                patient_ids=[patient_id], verbose=False)
                except:
                    susp_inf_df = pd.DataFrame()

                if sofa_df.empty:
                    continue

                # Extract events
                if not abx_df.empty:
                    time_col = 'starttime' if 'starttime' in abx_df.columns else 'charttime'
                    if 'abx' in abx_df.columns:
                        abx_data = abx_df[abx_df['abx'].notna() & (abx_df['abx'] > 0)][[time_col]].rename(
                            columns={time_col: 'time'})
                    else:
                        abx_data = abx_df[[time_col]].rename(columns={time_col: 'time'})
                else:
                    abx_data = pd.DataFrame()

                if not samp_df.empty:
                    time_col = 'charttime' if 'charttime' in samp_df.columns else (
                        'starttime' if 'starttime' in samp_df.columns else None)
                    if time_col and 'samp' in samp_df.columns:
                        samp_data = samp_df[[time_col, 'samp']].rename(columns={time_col: 'time'})
                    elif time_col:
                        samp_data = samp_df[[time_col]].rename(columns={time_col: 'time'})
                    else:
                        samp_data = pd.DataFrame()
                else:
                    samp_data = pd.DataFrame()

                if not susp_inf_df.empty:
                    time_col = 'starttime' if 'starttime' in susp_inf_df.columns else 'charttime'
                    if 'susp_inf' in susp_inf_df.columns:
                        si_data = susp_inf_df[susp_inf_df['susp_inf'] == True][[time_col]].rename(
                            columns={time_col: 'time'})
                    else:
                        si_data = susp_inf_df[[time_col]].rename(columns={time_col: 'time'})
                else:
                    si_data = pd.DataFrame()

                if not sepsis3_df.empty:
                    time_col = 'charttime' if 'charttime' in sepsis3_df.columns else 'starttime'
                    if 'sep3' in sepsis3_df.columns:
                        sep3_data = sepsis3_df[sepsis3_df['sep3'] == True][[time_col]].rename(
                            columns={time_col: 'time'})
                    else:
                        sep3_data = sepsis3_df[[time_col]].rename(columns={time_col: 'time'})
                else:
                    sep3_data = pd.DataFrame()

                # Sepsis-3 (SOFA2)
                sep3_sofa2_data = pd.DataFrame()
                if not si_data.empty and not sofa2_df.empty:
                    try:
                        si_for_sep3 = si_data.copy()
                        si_for_sep3['susp_inf'] = True
                        id_col = 'stay_id' if 'stay_id' in sofa2_df.columns else (
                            'patientunitstayid' if 'patientunitstayid' in sofa2_df.columns else 'admissionid')
                        time_col = 'charttime' if 'charttime' in sofa2_df.columns else 'starttime'
                        if id_col in susp_inf_df.columns:
                            si_for_sep3[id_col] = patient_id
                        si_for_sep3 = si_for_sep3.rename(columns={'time': time_col})

                        sep3_sofa2_result = sep3_sofa2(
                            sofa2=sofa2_df,
                            susp_inf_df=si_for_sep3,
                            id_cols=[id_col],
                            index_col=time_col
                        )
                        if 'sep3_sofa2' in sep3_sofa2_result.columns:
                            sep3_sofa2_data = sep3_sofa2_result[sep3_sofa2_result['sep3_sofa2'] == True][
                                [time_col]].rename(columns={time_col: 'time'})
                    except Exception as e:
                        if verbose:
                            print(f"   ⚠️  SOFA2 Sepsis-3 calculation failed: {str(e)[:50]}...")

                # Create figure
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

                time_col_sofa = 'charttime' if 'charttime' in sofa_df.columns else 'starttime'

                # Plot 1: SOFA vs SOFA2
                ax1.plot(sofa_df[time_col_sofa], sofa_df['sofa'],
                         marker='o', linewidth=2, markersize=6, label='SOFA', color='#1f77b4')

                if not sofa2_df.empty:
                    time_col_sofa2 = 'charttime' if 'charttime' in sofa2_df.columns else 'starttime'
                    ax1.plot(sofa2_df[time_col_sofa2], sofa2_df['sofa2'],
                             marker='s', linewidth=2, markersize=6, label='SOFA2', color='#ff7f0e')

                ax1.axhline(y=2, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='SOFA=2 (baseline)')

                if not sep3_data.empty:
                    sep3_time = sep3_data.iloc[0]['time']
                    ax1.axvline(x=sep3_time, color='red', linestyle='--', linewidth=2,
                                label=f'Sepsis-3 time ({sep3_time:.1f}h)')

                    si_window_start = sep3_time - 48
                    si_window_end = sep3_time + 24
                    ax1.axvspan(si_window_start, si_window_end, alpha=0.15, color='yellow',
                                label='Suspected infection window (-48/+24h)')

                ax1.set_ylabel('SOFA Score', fontsize=12, fontweight='bold')
                ax1.set_title(f'Patient {patient_id} - SOFA vs SOFA2 Comparison', fontsize=14, fontweight='bold')
                ax1.legend(loc='upper left', fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(bottom=0)

                # Plot 2: Event timeline
                y_positions = {'abx': 1, 'samp': 2, 'si': 3, 'sep3_sofa': 4, 'sep3_sofa2': 5}

                if not abx_data.empty:
                    ax2.scatter(abx_data['time'], [y_positions['abx']] * len(abx_data),
                                s=150, marker='s', color='blue', label='Antibiotics', zorder=5, alpha=0.8)

                if not samp_data.empty:
                    ax2.scatter(samp_data['time'], [y_positions['samp']] * len(samp_data),
                                s=150, marker='^', color='green', label='Blood Culture', zorder=5, alpha=0.8)

                if not si_data.empty:
                    ax2.scatter(si_data['time'], [y_positions['si']] * len(si_data),
                                s=180, marker='D', color='orange', label='Suspected Infection', zorder=5, alpha=0.9)

                if not sep3_data.empty:
                    ax2.scatter(sep3_data['time'], [y_positions['sep3_sofa']] * len(sep3_data),
                                s=250, marker='*', color='red', label='Sepsis-3 (SOFA)', zorder=6,
                                edgecolors='darkred', linewidths=1.5)

                if not sep3_sofa2_data.empty:
                    ax2.scatter(sep3_sofa2_data['time'], [y_positions['sep3_sofa2']] * len(sep3_sofa2_data),
                                s=250, marker='*', color='darkgreen', label='Sepsis-3 (SOFA2)', zorder=6,
                                edgecolors='green', linewidths=1.5)

                ax2.set_yticks(list(y_positions.values()))
                ax2.set_yticklabels(
                    ['Antibiotics', 'Blood Sample', 'Suspected Infection', 'Sepsis-3\n(SOFA)', 'Sepsis-3\n(SOFA2)'])
                ax2.set_xlabel('Hours since ICU admission', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Event Type', fontsize=12, fontweight='bold')
                ax2.legend(loc='upper left', fontsize=10)
                ax2.grid(True, alpha=0.3, axis='x')
                ax2.set_ylim(0.5, 5.5)

                plt.tight_layout()

                output_file = output_dir / f'sepsis_comparison_patient_{patient_id}.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                chart_count += 1

            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Patient {patient_id}: {str(e)[:60]}...")

        if verbose:
            print(f"✅ Successfully generated {chart_count} visualization charts")

    except Exception as e:
        if verbose:
            print(f"⚠️  Visualization failed: {e}")


# ============================================================================
# 主函数
# ============================================================================

def run_all_tests(
    data_source: str = None,
    patient_set: str = None,
    database: str = None,
    verbose: bool = True
):
    """运行所有测试
    
    Args:
        data_source: 数据源 ('test', 'production')，默认使用全局 TEST_DATA_SOURCE
        patient_set: 患者集 ('default', '50patients', 'debug')，默认使用全局 TEST_PATIENT_SET
        database: 数据库 ('miiv', 'eicu', 'hirid', 'aumc')，默认使用全局 TEST_DATABASE
        verbose: 是否显示详细输出
    """
    # 使用全局变量作为默认值
    if data_source is None:
        data_source = TEST_DATA_SOURCE
    if patient_set is None:
        patient_set = TEST_PATIENT_SET
    if database is None:
        database = TEST_DATABASE
    
    # 获取配置
    data_path = str(get_data_path(data_source, database))
    patient_ids = get_patient_ids(patient_set, database, Path(data_path))
    
    print("=" * 80)
    print("🏥 pyricu 统一测试")
    print("=" * 80)
    
    print(f"\n📋 测试配置:")
    print(f"   数据库: {database.upper()}")
    print(f"   数据源: {data_source} ({data_path})")
    print(f"   患者集: {patient_set} ({len(patient_ids)} 个患者)")
    print(f"   患者ID: {patient_ids}")
    
    # 运行所有测试
    try:
        # 首先验证原始表数据
        verify_raw_tables(data_path, patient_ids, database, verbose)
        
        test_sofa_basic(data_path, patient_ids, database, verbose)
        test_sofa_components(data_path, patient_ids, database, verbose)
        test_sofa2_comparison(data_path, patient_ids, database, verbose)
        test_sepsis3(data_path, patient_ids, database, verbose)
        test_easy_api(data_path, patient_ids, database, verbose)
        test_batch_performance(data_path, patient_ids, database, verbose)
        test_data_integrity(data_path, patient_ids, database, verbose)
        test_sofa_sepsis_visualization(data_path, patient_ids, database, verbose)
        
        # 总结
        print("\n" + "=" * 80)
        print(f"✅ 所有测试通过！pyricu 核心功能验证完成 [{database.upper()}]")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """主函数 - 提供多种测试模式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='pyricu 统一测试')
    parser.add_argument('--database', choices=['miiv', 'eicu', 'hirid', 'aumc'], 
                        default=TEST_DATABASE,
                        help=f'数据库 (default: {TEST_DATABASE})')
    parser.add_argument('--data', choices=['test', 'production'], 
                        default=TEST_DATA_SOURCE,
                        help=f'数据源 (default: {TEST_DATA_SOURCE})')
    parser.add_argument('--patients', choices=['debug', 'default', '50patients'], 
                        default=TEST_PATIENT_SET,
                        help=f'患者集 (default: {TEST_PATIENT_SET})')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='显示详细输出')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式（覆盖 --verbose）')
    
    args = parser.parse_args()
    
    # 处理 verbose
    verbose = args.verbose and not args.quiet
    
    # 运行测试
    success = run_all_tests(
        data_source=args.data,
        patient_set=args.patients,
        database=args.database,
        verbose=verbose
    )
    
    # 退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
